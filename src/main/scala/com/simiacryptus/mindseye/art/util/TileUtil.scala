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

  def reassemble(contentTensor: Tensor, selectors: Array[Layer], tiles: Array[Tensor], output: NotebookOutput) = {
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
        override def log(msg: String): Unit = super.log(msg)

        override def onStepComplete(currentPoint: Step): Unit = {
          output.out(s"Step ${currentPoint.iteration} completed, fitness=${currentPoint.point.sum}")
          currentPoint.freeRef()
        }

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
