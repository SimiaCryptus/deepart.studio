/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.util

import java.awt.image.BufferedImage
import java.lang
import java.util.UUID
import java.util.concurrent.TimeUnit

import com.simiacryptus.aws.S3Util
import com.simiacryptus.mindseye.art.util.ArtUtil.{setPrecision, withTrainingMonitor}
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.lang.{Layer, Result, Tensor, TensorArray, TensorList}
import com.simiacryptus.mindseye.layers.WrapperLayer
import com.simiacryptus.mindseye.network.{CountingResult, DAGNetwork, PipelineNetwork}
import com.simiacryptus.mindseye.opt.line.{ArmijoWolfeSearch, LineSearchStrategy}
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{RangeConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.{LoggingIterativeTrainer, Step, TrainingMonitor, TrainingResult}
import com.simiacryptus.mindseye.test.{GraphVizNetworkInspector, MermaidGrapher}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.Logging
import com.simiacryptus.util.FastRandom

import scala.collection.mutable.ArrayBuffer
import scala.util.Try

trait ImageOptimizer extends Logging {

  val maxRetries = 3

  /**
   * This function optimizes the given canvas image using the given trainable object.
   * The notebook output is used for logging purposes.
   *
   * @docgenVersion 9
   */
  def optimize(canvasImage: Tensor, trainable: Trainable)(implicit log: NotebookOutput): TrainingResult = {
    try {
      def currentImage = {
        val tensor = render(canvasImage.addRef())
        val image = tensor.toRgbImage
        tensor.freeRef()
        image
      }

      var result = withMonitoredJpg[TrainingResult](() => currentImage) {
        log.subreport("Optimization", (sub: NotebookOutput) => {
          NetworkUtil.graph(trainable.addRef())(sub)
          optimize(() => currentImage, trainable.addRef())(sub)
        })
      }

      lazy val data = canvasImage.getData
      var noiseMagnitude = 1e-1
      var retries = 0;
      while (result.terminationCause == TrainingResult.TerminationCause.Failed && retries < maxRetries) {
        retries = retries + 1
        (0 until data.length).foreach(i => data(i) = data(i) + FastRandom.INSTANCE.random() * noiseMagnitude)
        noiseMagnitude = noiseMagnitude * 2
        result = withMonitoredJpg[TrainingResult](() => currentImage) {
          log.subreport("Optimization (Retry)", (sub: NotebookOutput) => {
            optimize(() => currentImage, trainable.addRef())(sub)
          })
        }
      }

      result
    } finally {
      trainable.freeRef()
      canvasImage.freeRef()
      try {
        onComplete()
      } catch {
        case e: Throwable => logger.warn("Error running onComplete", e)
      }
    }
  }

  /**
   * Renders the given canvas image.
   *
   * @param canvasImage the image to render
   * @return the rendered image
   * @docgenVersion 9
   */
  def render(canvasImage: Tensor) = {
    val network = renderingNetwork(canvasImage.getDimensions)
    val result = network.eval(canvasImage)
    network.freeRef()
    val data = result.getData
    result.freeRef()
    val tensor = data.get(0)
    data.freeRef()
    tensor
  }

  /**
   * This function creates a new PipelineNetwork with one input layer and the specified dimensions.
   *
   * @docgenVersion 9
   */
  def renderingNetwork(dims: Seq[Int]): Layer = new PipelineNetwork(1)

  /**
   * This function optimizes the current image using the trainable function.
   *
   * @param currentImage The image to be optimized.
   * @param trainable    The function used to optimize the image.
   * @param out          The notebook output.
   * @return The result of the optimization.
   * @docgenVersion 9
   */
  def optimize(currentImage: () => BufferedImage, trainable: Trainable)(implicit out: NotebookOutput): TrainingResult = {
    //    trainable.getLayer match {
    //      case dag: DAGNetwork => new MermaidGrapher(out, false).mermaid(dag);
    //      case _ =>
    //    }
    val timelineAnimation = new ArrayBuffer[BufferedImage]()
    timelineAnimation += currentImage()
    NotebookRunner.withMonitoredGif(() => timelineAnimation.toList ++ List(currentImage()), delay = 1000) {
      withTrainingMonitor(trainingMonitor => {
        val lineSearchInstance: LineSearchStrategy = lineSearchFactory
        val trainer = new LoggingIterativeTrainer(trainable) {
          /**
           * Overrides the protected logState method to do nothing.
           *
           * @param sublog    The notebook output to log to
           * @param iteration The current iteration
           * @docgenVersion 9
           */
          override protected def logState(sublog: NotebookOutput, iteration: Int): Unit = {
            if (true || 0 < logEvery && (0 == iteration % logEvery || iteration < logEvery)) {
              val image = currentImage()
              timelineAnimation += image
              val caption = "Iteration " + iteration
              sublog.p(sublog.jpg(image, caption))
            }
          }
        }
        trainer.setOrientation(orientation())
        trainer.setMonitor(new TrainingMonitor() {
          /**
           * Clears the training monitor.
           *
           * @docgenVersion 9
           */
          override def clear(): Unit = trainingMonitor.clear()

          /**
           * Logs the given message to the training monitor.
           *
           * @docgenVersion 9
           */
          override def log(msg: String): Unit = {
            trainingMonitor.log(msg)
          }

          /**
           * Overrides the onStepFail method to take in a Trainable and a Step, and
           * returns a boolean.
           *
           * @docgenVersion 9
           */
          override def onStepFail(currentPoint: Step): Boolean = {
            ImageOptimizer.this.onStepFail(trainable.addRef().asInstanceOf[Trainable], currentPoint)
          }

          /**
           * This function is called when a training step is completed.
           * It updates the ImageOptimizer with the new Trainable and Step data,
           * and also updates the trainingMonitor.
           *
           * @docgenVersion 9
           */
          override def onStepComplete(currentPoint: Step): Unit = {
            ImageOptimizer.this.onStepComplete(trainable.addRef().asInstanceOf[Trainable], currentPoint.addRef())
            trainingMonitor.onStepComplete(currentPoint)
          }
        })
        trainer.setTimeout(trainingMinutes, TimeUnit.MINUTES)
        trainer.setMaxIterations(trainingIterations)
        trainer.setLineSearchFactory((_: CharSequence) => lineSearchInstance)
        trainer.setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
        try {
          trainer.run(out)
        } finally {
          trainer.freeRef()
        }
      })(out)
    }(out)
  }

  /**
   * This function is called when a training step is completed.
   *
   * @param trainable    The object that is being trained.
   * @param currentPoint The current training step.
   * @docgenVersion 9
   */
  def onStepComplete(trainable: Trainable, currentPoint: Step) = {
    currentPoint.freeRef()
    setPrecision(trainable, Precision.Float)
  }

  /**
   * onStepFail is a function that takes in a Trainable and a Step, and returns a Boolean.
   *
   * @docgenVersion 9
   */
  def onStepFail(trainable: Trainable, currentPoint: Step): Boolean = {
    currentPoint.freeRef()
    setPrecision(trainable, Precision.Double)
  }

  def logEvery = 5

  def trainingMinutes: Int = 60

  def trainingIterations: Int = 20

  def lineSearchFactory: LineSearchStrategy = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setMinAlpha(1e-10).setAlpha(1).setRelativeTolerance(1e-5)

  def maxRate = 1e9

  /**
   * This function defines a new TrustRegionStrategy with a LBFGS solver, and sets the region policy to trustRegion for the given layer.
   *
   * @docgenVersion 9
   */
  def orientation() = {
    new TrustRegionStrategy(new LBFGS) {
      /**
       * Overrides the getRegionPolicy method to use a trustRegion for the given layer.
       *
       * @docgenVersion 9
       */
      override def getRegionPolicy(layer: Layer) = trustRegion(layer)
    }
  }

  /**
   * Returns a trust region for the given layer.
   *
   * @docgenVersion 9
   */
  def trustRegion(layer: Layer): TrustRegion = {
    //    new CompoundRegion(
    //      //                new RangeConstraint().setMin(0).setMax(256),
    //      //                new FixedMagnitudeConstraint(canvasImage.coordStream(true)
    //      //                  .collect(Collectors.toList()).asScala
    //      //                  .groupBy(_.getCoords()(2)).values
    //      //                  .toArray.map(_.map(_.getIndex).toArray): _*),
    //    )
    if (null != layer) layer.freeRef()
    new RangeConstraint().setMin(0).setMax(256)
  }

  /**
   * This function is called when the notebook is completed.
   * It tries to upload the notebook to S3.
   *
   * @docgenVersion 9
   */
  def onComplete()(implicit log: NotebookOutput): Unit = {
    Try {
      S3Util.upload(log)
    }
  }
}
