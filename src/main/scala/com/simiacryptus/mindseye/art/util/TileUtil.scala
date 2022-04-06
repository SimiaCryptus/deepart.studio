package com.simiacryptus.mindseye.art.util

import java.util.concurrent.TimeUnit

import com.simiacryptus.mindseye.eval.ArrayTrainable
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.{AvgReducerLayer, BinarySumLayer, SquareActivationLayer, SumInputsLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.line.QuadraticSearch
import com.simiacryptus.mindseye.opt.orient.GradientDescent
import com.simiacryptus.mindseye.opt.{IterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.notebook.NotebookOutput

import scala.util.Random

object TileUtil {
  /**
   * Defines a loss function that takes in a color mapping layer and a summarizer layer.
   *
   * @docgenVersion 9
   */
  def lossFunction(colorMappingLayer: Layer, summarizer: () => Layer) = {
    val network = new PipelineNetwork(2)
    network.add(
      new AvgReducerLayer(),
      network.add(
        new SquareActivationLayer(),
        network.add(
          new BinarySumLayer(1.0, -1.0),
          network.add(
            summarizer(),
            network.getInput(0)),
          network.add(
            summarizer(),
            network.add(
              colorMappingLayer,
              network.getInput(1)
            ))
        ))
    ).freeRef()
    network
  }

  /**
   * Reassemble the content tensor using the given selectors and tiles.
   *
   * @param contentTensor the tensor to reassemble
   * @param selectors     an array of layers to use as selectors
   * @param tiles         an array of tensors to use as tiles
   * @param output        the notebook output to use
   * @docgenVersion 9
   */
  def reassemble(contentTensor: Tensor, selectors: Array[Layer], tiles: Array[Tensor], output: NotebookOutput) = {
    /**
     * This function optimizes a pipeline network for a given set of tiles.
     *
     * @param network The pipeline network to optimize.
     * @param tiles   The tiles to use for optimization.
     * @return The optimized network.
     * @docgenVersion 9
     */
    def optimize(network: PipelineNetwork, tiles: Array[Tensor]): Double = {
      val trainable = new ArrayTrainable(network, 1)
      trainable.setTrainingData(Array(Array(contentTensor.addRef()) ++ tiles))
      trainable.setMask(Array(true) ++ tiles.map(_ => false): _*)
      val trainer = new IterativeTrainer(trainable)
      trainer.setMaxIterations(1000)
      trainer.setTerminateThreshold(0.0)
      trainer.setTimeout(10, TimeUnit.MINUTES)
      trainer.setLineSearchFactory(_ => new QuadraticSearch)
      trainer.setOrientation(new GradientDescent)
      trainer.setMonitor(new TrainingMonitor {
        /**
         * Overrides the log method to just call the superclass's log method.
         *
         * @docgenVersion 9
         */
        override def log(msg: String): Unit = super.log(msg)

        /**
         * This method is called when a step is completed.
         *
         * @param currentPoint the current point in the iteration
         * @docgenVersion 9
         */
        override def onStepComplete(currentPoint: Step): Unit = {
          output.out(s"Step ${currentPoint.iteration} completed, fitness=${currentPoint.point.sum}")
          currentPoint.freeRef()
        }

        /**
         * Overrides the onStepFail method to return the superclass' onStepFail method.
         *
         * @docgenVersion 9
         */
        override def onStepFail(currentPoint: Step): Boolean = super.onStepFail(currentPoint)
      })
      val result = trainer.run()
      output.out(s"Optimization Finished: $result")
      result.finalValue
    }

    val lowMem = true
    if (lowMem) {
      val results = (for (
        group <- Random.shuffle(selectors.zip(tiles).toStream).grouped(4);
        (selector, tile) <- group.par
      ) yield {
        val network = new PipelineNetwork(2)
        network.add(
          lossFunction(selector.addRef().asInstanceOf[Layer], () => new PipelineNetwork(1)),
          network.getInput(1),
          network.getInput(0)
        ).freeRef()
        optimize(network, Array(tile.addRef()))
      }).toArray
      require(results.forall(!_.isInfinite))
    } else {
      val network = new PipelineNetwork(1 + tiles.size)
      network.add(new SumInputsLayer(),
        (for ((selector, index) <- selectors.zipWithIndex) yield {
          network.add(
            lossFunction(selector.addRef().asInstanceOf[Layer], () => new PipelineNetwork(1)),
            network.getInput(index + 1),
            network.getInput(0))
        }): _*).freeRef()
      optimize(network, tiles.map(_.addRef()))
    }

    output.out(output.jpg(contentTensor.toRgbImage, "Combined Image"))
    contentTensor
  }

}
