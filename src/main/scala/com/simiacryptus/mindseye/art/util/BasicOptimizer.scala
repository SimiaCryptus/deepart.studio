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
import java.util.concurrent.TimeUnit

import com.simiacryptus.aws.S3Util
import com.simiacryptus.mindseye.art.util.ArtUtil.{setPrecision, withTrainingMonitor}
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.network.{DAGNetwork, PipelineNetwork}
import com.simiacryptus.mindseye.opt.line.{ArmijoWolfeSearch, LineSearchStrategy}
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{RangeConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.{LoggingIterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.mindseye.test.{GraphVizNetworkInspector, MermaidGrapher}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.Logging

import scala.collection.mutable.ArrayBuffer
import scala.util.Try

trait BasicOptimizer extends Logging {

  def optimize(canvasImage: Tensor, trainable: Trainable)(implicit log: NotebookOutput): Double = {
    try {
      def currentImage = {
        val tensor = render(canvasImage.addRef())
        val image = tensor.toRgbImage
        tensor.freeRef()
        image
      }

      withMonitoredJpg[Double](() => currentImage) {
        log.subreport("Optimization", (sub: NotebookOutput) => {
          graph(trainable.addRef())(sub)
          optimize(() => currentImage, trainable)(sub).asInstanceOf[lang.Double]
        })
      }
    } finally {
      canvasImage.freeRef()
      try {
        onComplete()
      } catch {
        case e: Throwable => logger.warn("Error running onComplete", e)
      }
    }
  }

  def graph(trainable: Trainable)(implicit log: NotebookOutput) = {
    val layer = trainable.getLayer
    try {
      if (layer != null && layer.isInstanceOf[DAGNetwork]) {
        log.subreport("Network Diagram", (sub: NotebookOutput) => {
          GraphVizNetworkInspector.graph(sub, layer.asInstanceOf[DAGNetwork])
          null
        })
      }
    } finally {
      trainable.freeRef()
      layer.freeRef()
    }
  }

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

  def renderingNetwork(dims: Seq[Int]): PipelineNetwork = new PipelineNetwork(1)

  def optimize(currentImage: () => BufferedImage, trainable: Trainable)(implicit out: NotebookOutput): Double = {
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
          override def clear(): Unit = trainingMonitor.clear()

          override def log(msg: String): Unit = {
            trainingMonitor.log(msg)
          }

          override def onStepFail(currentPoint: Step): Boolean = {
            BasicOptimizer.this.onStepFail(trainable.addRef().asInstanceOf[Trainable], currentPoint)
          }

          override def onStepComplete(currentPoint: Step): Unit = {
            BasicOptimizer.this.onStepComplete(trainable.addRef().asInstanceOf[Trainable], currentPoint.addRef())
            trainingMonitor.onStepComplete(currentPoint)
          }
        })
        trainer.setTimeout(trainingMinutes, TimeUnit.MINUTES)
        trainer.setMaxIterations(trainingIterations)
        trainer.setLineSearchFactory((_: CharSequence) => lineSearchInstance)
        trainer.setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
        val result = trainer.run(out).asInstanceOf[lang.Double]
        trainer.freeRef()
        result
      })(out)
    }(out)
  }

  def onStepComplete(trainable: Trainable, currentPoint: Step) = {
    currentPoint.freeRef()
    setPrecision(trainable, Precision.Float)
  }

  def onStepFail(trainable: Trainable, currentPoint: Step): Boolean = {
    currentPoint.freeRef()
    setPrecision(trainable, Precision.Double)
  }

  def logEvery = 5

  def trainingMinutes: Int = 60

  def trainingIterations: Int = 20

  def lineSearchFactory: LineSearchStrategy = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setMinAlpha(1e-10).setAlpha(1).setRelativeTolerance(1e-5)

  def maxRate = 1e9

  def orientation() = {
    new TrustRegionStrategy(new LBFGS) {
      override def getRegionPolicy(layer: Layer) = trustRegion(layer)
    }
  }

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

  def onComplete()(implicit log: NotebookOutput): Unit = {
    Try {
      S3Util.upload(log)
    }
  }
}
