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
import java.io.File
import java.net.URI
import java.util
import java.util.UUID
import java.util.concurrent.TimeUnit

import com.simiacryptus.mindseye.art._
import com.simiacryptus.mindseye.art.ops.GramMatrixMatcher
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
import com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer
import com.simiacryptus.mindseye.network.{DAGNetwork, PipelineNetwork}
import com.simiacryptus.mindseye.opt.line.QuadraticSearch
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy
import com.simiacryptus.mindseye.opt.region.{OrthonormalConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.{IterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.mindseye.test.{GraphVizNetworkInspector, StepRecord, TestUtil}
import com.simiacryptus.mindseye.util.ImageUtil
import com.simiacryptus.notebook._
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.util.Util
import org.apache.hadoop.fs.{FileSystem, Path}

import scala.collection.JavaConverters._
//import scala.jdk.CollectionConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Try

object ArtUtil {

  private val pngCache = new mutable.HashMap[String, Try[File]]()

  /**
   * @param paintings An array of strings representing paintings
   * @param log       An implicit notebook output
   * @return An HTML query of a map from strings to booleans, representing the user's selection of paintings
   * @docgenVersion 9
   */
  def userSelect(paintings: Array[String])(implicit log: NotebookOutput): HtmlQuery[Map[String, Boolean]] = {
    val ids = paintings.map(_ -> UUID.randomUUID().toString).toMap
    new FormQuery[Map[String, Boolean]](log.asInstanceOf[MarkdownNotebookOutput]) {

      /**
       * Returns the height of the object.
       *
       * @return the height of the object
       * @docgenVersion 9
       */
      override def height(): Int = 800

      override protected def getFormInnerHtml: String = {
        (for ((url, v) <- getValue.toList.filter(_._2)) yield {
          val filename = url.split('/').last.toLowerCase.stripSuffix(".png") + ".png"
          if (pngCache.getOrElseUpdate(url, Try {
            log.pngFile(ImageArtUtil.loadImage(log, url, 256))
          }).isSuccess) {
            s"""<input type="checkbox" name="${ids(url)}" value="true"><img src="etc/$filename" alt="$url"><br/>"""
          } else {
            ""
          }
        }).mkString("\n")
      }

      /**
       * @param parms a map of parameter names to parameter values
       * @param files a map of file names to file contents
       * @return a map of parameter names to boolean values
       * @docgenVersion 9
       */
      override def valueFromParams(parms: java.util.Map[String, String], files: java.util.Map[String, String]): Map[String, Boolean] = {
        (for ((k, v) <- getValue) yield {
          k -> parms.getOrDefault(ids(k), "false").toBoolean
        })
      }
    }.setValue(paintings.map(_ -> true).toMap).print()
  }

  /**
   * Sets the precision for a trainable object.
   *
   * @param trainable The trainable object.
   * @param precision The desired precision.
   * @return Whether or not the precision was successfully set.
   * @docgenVersion 9
   */
  def setPrecision(trainable: Trainable, precision: Precision): Boolean = {
    if (CudaSettings.INSTANCE().getDefaultPrecision == precision) {
      trainable.freeRef()
      false
    } else {
      CudaSettings.INSTANCE().setDefaultPrecision(precision)
      resetPrecision(trainable, precision)
      true
    }
  }

  /**
   * Resets the precision of a trainable object.
   *
   * @param trainable The trainable object whose precision will be reset.
   * @param precision The new precision value.
   * @docgenVersion 9
   */
  def resetPrecision(trainable: Trainable, precision: Precision) = {
    trainable match {
      case trainable: SumTrainable =>
        for (layer <- trainable.getInner) setPrecision(layer, precision)
        trainable.freeRef()
      case trainable: TiledTrainable =>
        trainable.setPrecision(precision)
        trainable.freeRef()
      case trainable: Trainable =>
        val layer = trainable.getLayer
        if (null != layer) {
          MultiPrecision.setPrecision(layer.asInstanceOf[DAGNetwork], precision)
          layer.freeRef()
        }
        trainable.freeRef()
    }
  }

  /**
   * This function creates pipeline graphs for the given pipeline.
   *
   * @param pipeline the pipeline to create graphs for
   * @param log      the notebook output to use
   * @docgenVersion 9
   */
  def pipelineGraphs(pipeline: VisionPipeline)(implicit log: NotebookOutput) = {
    log.subreport(pipeline.name + "_Layers", (sublog: NotebookOutput) => {
      pipeline.getLayerList().asScala.foreach(layer => {
        sublog.h1(layer.name())
        GraphVizNetworkInspector.graph(sublog, layer.getLayer.asInstanceOf[PipelineNetwork])
      })
      null
    })
  }

  /**
   * This function creates a cyclical animation from a list of canvases.
   *
   * @docgenVersion 9
   */
  def cyclicalAnimation(canvases: => List[Tensor]): List[BufferedImage] = {
    val list = canvases.filter(_ != null).map(tensor => {
      val image = tensor.toRgbImage
      tensor.freeRef()
      image
    })
    if (list.isEmpty) {
      List.empty
    } else {
      val maxWidth = list.map(_.getWidth).max
      cyclical(list.map(ImageUtil.resize(_, maxWidth, true)))
    }
  }

  /**
   * This function takes a list and returns a new list that is the original list concatenated with the reverse of the original list,
   * minus the last element of the original list.
   *
   * @docgenVersion 9
   */
  def cyclical[T](list: List[T]) = {
    (list ++ list.reverse.tail.dropRight(1))
  }

  /**
   * Loads the content from the specified URL.
   *
   * @param content The content to load.
   * @param url     The URL to load from.
   * @param log     The notebook output.
   * @docgenVersion 9
   */
  def load(content: Tensor, url: String)(implicit log: NotebookOutput): Tensor = {
    val dimensions = content.getDimensions
    content.freeRef()
    load(dimensions, url)
  }

  /**
   * Loads the content from the given URL into a Tensor.
   *
   * @param contentDims The dimensions of the content to load.
   * @param url         The URL to load the content from.
   * @param log         The notebook output to use for logging. Defaults to a NullNotebookOutput.
   * @return The loaded Tensor.
   * @docgenVersion 9
   */
  def load(contentDims: Array[Int], url: String)(implicit log: NotebookOutput = new NullNotebookOutput()): Tensor = {
    ImageArtUtil.getImageTensors(url, log, contentDims(0), contentDims(1)).head
  }

  /**
   * Transfers the style of the style images to the content image.
   *
   * @param contentImage the content image
   * @param styleImages  the style images
   * @param orthogonal   whether to use an orthogonal transformation
   * @param tileSize     the size of each tile
   * @param tilePadding  the padding between each tile
   * @param precision    the precision of the computation
   * @docgenVersion 9
   */
  def colorTransfer
  (
    contentImage: Tensor,
    styleImages: Seq[Tensor],
    orthogonal: Boolean = true,
    tileSize: Int = 400,
    tilePadding: Int = 0,
    precision: Precision = Precision.Float
  )(implicit log: NotebookOutput): Layer = {
    val biasLayer = new ImgBandBiasLayer(3)
    val convolutionLayer = new SimpleConvolutionLayer(1, 1, 3, 3)
    convolutionLayer.set((c: Coordinate) => {
      val coords = c.getCoords()(2)
      if ((coords % 3) == (coords / 3)) 1.0 else 0.0
    })
    biasLayer.setWeights((i: Int) => 0.0)
    val layer = colorTransfer(contentImage, styleImages, tileSize, tilePadding, precision, new PipelineNetwork(1,
      convolutionLayer, biasLayer
    ))
    layer.freeze()
    layer
  }

  /**
   * Transfers the style of the style images to the content image.
   *
   * @param contentImage         The content image.
   * @param styleImages          The style images.
   * @param tileSize             The size of the tiles.
   * @param tilePadding          The padding of the tiles.
   * @param precision            The precision.
   * @param colorAdjustmentLayer The color adjustment layer.
   * @param log                  The notebook output.
   * @return The layer.
   * @docgenVersion 9
   */
  def colorTransfer(contentImage: Tensor, styleImages: Seq[Tensor], tileSize: Int, tilePadding: Int, precision: Precision, colorAdjustmentLayer: PipelineNetwork)(implicit log: NotebookOutput): Layer = {
    def styleMatcher = new GramMatrixMatcher() //.combine(new ChannelMeanMatcher().scale(1e0))

    val styleNetwork = MultiPrecision.setPrecision(styleMatcher.build(VisionPipelineLayer.NOOP, null, null, styleImages: _*), precision).asInstanceOf[PipelineNetwork]
    val trainable_color = new TiledTrainable(contentImage, colorAdjustmentLayer, tileSize, tilePadding, precision) {
      /**
       * Returns the network for the given region selector.
       *
       * @param regionSelector The region selector layer.
       * @return The network for the given region selector.
       * @docgenVersion 9
       */
      override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
        regionSelector.freeRef()
        styleNetwork.addRef()
      }
    }.setMutableCanvas(false)

    val trainingMonitor = getTrainingMonitor()(log)
    val search = new QuadraticSearch().setCurrentRate(1e0).setMaxRate(1e3).setRelativeTolerance(1e-2)
    val trainer = new IterativeTrainer(trainable_color)
    trainer.setOrientation(new TrustRegionStrategy() {
      /**
       * Returns the region policy for the given layer.
       *
       * @param layer the layer to get the region policy for
       * @return the region policy for the given layer, or null if the layer is null or frozen
       * @docgenVersion 9
       */
      override def getRegionPolicy(layer: Layer): TrustRegion = layer match {
        case null => null
        case layer if layer.isFrozen =>
          layer.freeRef()
          null
        case layer: SimpleConvolutionLayer => new OrthonormalConstraint(ImageArtUtil.getIndexMap(layer): _*)
        case _ => null
      }
    });
    trainer.setMonitor(trainingMonitor)
    trainer.setTimeout(5, TimeUnit.MINUTES)
    trainer.setMaxIterations(3)
    trainer.setLineSearchFactory((_: CharSequence) => search)
    trainer.setTerminateThreshold(0)
    trainer.run
    colorAdjustmentLayer
  }

  /**
   * Takes an image and divides it into a grid of smaller images.
   *
   * @param currentImage the image to be divided
   * @param columns      the number of columns in the grid
   * @param rows         the number of rows in the grid
   * @return an Option containing the grid of smaller images, or null if currentImage is null
   * @docgenVersion 9
   */
  def imageGrid(currentImage: BufferedImage, columns: Int = 2, rows: Int = 2) = Option(currentImage).map(tensor => {
    val assemblyLayer = new ImgTileAssemblyLayer(columns, rows)
    val result = assemblyLayer.eval(Stream.continually(Tensor.fromRGB(tensor)).take(columns * rows): _*)
    val data = result.getData
    val tensor1 = data.get(0)
    val grid = tensor1.toRgbImage
    tensor1.freeRef()
    data.freeRef()
    result.freeRef()
    assemblyLayer.freeRef()
    grid
  }).orNull

  /**
   * This function will take in a function that has a TrainingMonitor as a parameter.
   * The function will also have an implicit log parameter.
   * The return type will be of type T.
   *
   * @docgenVersion 9
   */
  def withTrainingMonitor[T](fn: TrainingMonitor => T)(implicit log: NotebookOutput): T = {
    val history = new util.ArrayList[StepRecord]
    NotebookRunner.withMonitoredJpg(() => Util.toImage(TestUtil.plot(history))) {
      fn(getTrainingMonitor(history))
    }
  }

  /**
   * Returns a new TrainingMonitor with the given history.
   *
   * @param history an ArrayList of StepRecords
   * @param out     a NotebookOutput
   * @tparam T the type of the TrainingMonitor
   * @return a new TrainingMonitor
   * @docgenVersion 9
   */
  def getTrainingMonitor[T](history: util.ArrayList[StepRecord] = new util.ArrayList[StepRecord])(implicit out: NotebookOutput): TrainingMonitor = {
    val trainingMonitor = new TrainingMonitor() {
      /**
       * Clears the contents of this buffer.
       *
       * @docgenVersion 9
       */
      override def clear(): Unit = {
        super.clear()
      }

      /**
       * Logs the given message to the output stream and then passes it to the superclass's log method.
       *
       * @docgenVersion 9
       */
      override def log(msg: String): Unit = {
        out.p(msg)
        super.log(msg)
      }

      /**
       * This method overrides the onStepComplete method in the superclass.
       * It adds a new StepRecord to the history, using the currentPoint's mean, time, and iteration.
       * Then, it calls the superclass's onStepComplete method.
       *
       * @docgenVersion 9
       */
      override def onStepComplete(currentPoint: Step): Unit = {
        history.add(new StepRecord(currentPoint.point.getMean, currentPoint.time, currentPoint.iteration))
        super.onStepComplete(currentPoint)
      }
    }
    trainingMonitor
  }

  /**
   * Finds all files with the given key in the given base directory.
   *
   * @docgenVersion 9
   */
  def findFiles(key: String, base: String): Array[String] = findFiles(Set(key), base)

  /**
   * Finds all files with the given key.
   *
   * @docgenVersion 9
   */
  def findFiles(key: String): Array[String] = findFiles(Set(key))

  /**
   * Find files that match the given key and are at least the given size.
   *
   * @param key     The key to match against
   * @param base    The base path to search in
   * @param minSize The minimum size of the files to return
   * @return An array of matching file paths
   * @docgenVersion 9
   */
  def findFiles(key: Set[String], base: String = "s3a://data-cb03c/crawl/wikiart/", minSize: Int = 32 * 1024): Array[String] = {
    val itr = FileSystem.get(new URI(base), ImageArtUtil.getHadoopConfig()).listFiles(new Path(base), true)
    val buffer = new ArrayBuffer[String]()
    while (itr.hasNext) {
      val status = itr.next()
      lazy val string = status.getPath.toString
      if (status.getLen > minSize && key.find(string.contains(_)).isDefined) buffer += string
    }
    buffer.toArray
  }

  /**
   * Transfers the color of a given image to the color reference.
   *
   * @param colorReference The color reference.
   * @param image          The image.
   * @docgenVersion 9
   */
  def colorTransferLAB(colorReference: Tensor, image: Tensor) = {
    ColorTransforms.lab2rgb(ColorTransforms.colorTransfer_primaries(ColorTransforms.rgb2lab(colorReference), ColorTransforms.rgb2lab(image)))
  }

  /**
   * Converts an image from one color space to another.
   *
   * @param colorReference The reference color space.
   * @param image          The image to be converted.
   * @docgenVersion 9
   */
  def colorTransferXYZ(colorReference: Tensor, image: Tensor) = {
    ColorTransforms.xyz2rgb(ColorTransforms.colorTransfer_primaries(ColorTransforms.rgb2xyz(colorReference), ColorTransforms.rgb2xyz(image)))
  }

  /**
   * Converts the given image to the HSL color space using the given color reference.
   *
   * @docgenVersion 9
   */
  def colorTransferHSL(colorReference: Tensor, image: Tensor) = {
    ColorTransforms.hsl2rgb(ColorTransforms.colorTransfer_primaries(ColorTransforms.rgb2hsl(colorReference), ColorTransforms.rgb2hsl(image)))
  }

  /**
   * Converts an image from one color space to another.
   *
   * @param colorReference The reference color.
   * @param image          The image to be converted.
   * @docgenVersion 9
   */
  def colorTransferHSV(colorReference: Tensor, image: Tensor) = {
    ColorTransforms.hsv2rgb(ColorTransforms.colorTransfer_primaries(ColorTransforms.rgb2hsv(colorReference), ColorTransforms.rgb2hsv(image)))
  }

  var transfer_random: (Tensor, Tensor, Int) => Tensor = (colorReference, image, iterations) => {
    //ColorTransforms.colorTransfer_random_geometricDamping(colorReference, image, iterations, 1.0 / 2)
    //ColorTransforms.colorTransfer_random_arithmeticDamping(colorReference, image, iterations, 1.0 / 2)
    ColorTransforms.colorTransfer_random(colorReference, image, iterations)
  }

  /**
   * Transfers the color of a given image to the color reference.
   *
   * @param colorReference The color reference image.
   * @param image          The image to be color transferred.
   * @param iterations     The number of iterations to run.
   * @docgenVersion 9
   */
  def colorTransferLAB(colorReference: Tensor, image: Tensor, iterations: Int) = {
    ColorTransforms.lab2rgb(transfer_random(ColorTransforms.rgb2lab(colorReference), ColorTransforms.rgb2lab(image), iterations))
  }

  /**
   * Transfers the color of a reference image to an input image using the XYZ color space.
   *
   * @param colorReference The reference image whose color to transfer.
   * @param image          The input image to which the reference image's color will be transferred.
   * @param iterations     The number of iterations of the color transfer algorithm to run.
   * @docgenVersion 9
   */
  def colorTransferXYZ(colorReference: Tensor, image: Tensor, iterations: Int) = {
    ColorTransforms.xyz2rgb(transfer_random(ColorTransforms.rgb2xyz(colorReference), ColorTransforms.rgb2xyz(image), iterations))
  }

  /**
   * Transfers the color of a reference image to an input image using the HSL color space.
   *
   * @param colorReference The reference image whose color to transfer.
   * @param image          The input image to which the reference image's color will be transferred.
   * @param iterations     The number of iterations of the color transfer algorithm to run.
   * @docgenVersion 9
   */
  def colorTransferHSL(colorReference: Tensor, image: Tensor, iterations: Int) = {
    ColorTransforms.hsl2rgb(transfer_random(ColorTransforms.rgb2hsl(colorReference), ColorTransforms.rgb2hsl(image), iterations))
  }

  /**
   * Transfers the color of a reference image to an input image using HSV colorspace.
   *
   * @param colorReference The reference image whose color to transfer.
   * @param image          The input image to which to transfer the color.
   * @param iterations     The number of iterations of color transfer to perform.
   * @docgenVersion 9
   */
  def colorTransferHSV(colorReference: Tensor, image: Tensor, iterations: Int) = {
    ColorTransforms.hsv2rgb(transfer_random(ColorTransforms.rgb2hsv(colorReference), ColorTransforms.rgb2hsv(image), iterations))
  }
}
