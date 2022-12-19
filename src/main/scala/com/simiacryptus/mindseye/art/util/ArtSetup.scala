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

import java.io.File
import java.net.{URI, URLEncoder}
import java.text.Normalizer
import java.util

import com.amazonaws.services.ec2.{AmazonEC2, AmazonEC2ClientBuilder}
import com.amazonaws.services.s3.{AmazonS3, AmazonS3ClientBuilder}
import com.fasterxml.jackson.annotation.JsonIgnore
import com.google.gson.GsonBuilder
import com.simiacryptus.aws.EC2Util
import com.simiacryptus.mindseye.art.registry.TaskRegistry
import com.simiacryptus.mindseye.art.util.ArtUtil.{cyclicalAnimation, load}
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.{Result, Tensor}
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, Precision}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.mindseye.util.ImageUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput, NullNotebookOutput}
import com.simiacryptus.ref.wrappers.RefAtomicReference
import com.simiacryptus.sparkbook.{InteractiveSetup, NotebookRunner}
import com.simiacryptus.util.FastRandom
import org.apache.commons.io.{FileUtils, IOUtils}

import scala.collection.JavaConverters._
//import scala.jdk.CollectionConverters._
import scala.collection.mutable

object ArtSetup {
  @JsonIgnore
  @transient implicit val ec2client: AmazonEC2 = AmazonEC2ClientBuilder.standard().withRegion(EC2Util.REGION).build()
}

trait ArtSetup[R <: AnyRef, V <: InteractiveSetup[R, V]] extends InteractiveSetup[R, V] with TaskRegistry {
  var s3bucket: String = ""

  override def description: String = super.description

  override def className: String = getClass.getSimpleName.stripSuffix("$")

  /**
   * Returns an array of paintings that match the given search word and have a width greater than or equal to the given minimum width.
   *
   * @docgenVersion 9
   */
  def getPaintingsBySearch(searchWord: String, minWidth: Int): Array[String] = {
    getPaintings(new URI("https://www.wikiart.org/en/search/" + URLEncoder.encode(searchWord, "UTF-8").replaceAllLiterally("+", "%20") + "/1?json=2"), minWidth, 100)
  }

  /**
   * Returns an array of paintings by the given artist that are at least the given width.
   *
   * @docgenVersion 9
   */
  def getPaintingsByArtist(artist: String, minWidth: Int): Array[String] = {
    getPaintings(new URI("https://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl=" + artist), minWidth, 100)
  }

  /**
   * @param uri        the URI to fetch paintings from
   * @param minWidth   the minimum width of the paintings to fetch
   * @param maxResults the maximum number of results to fetch
   * @return an array of Strings containing the fetched paintings
   * @docgenVersion 9
   */
  def getPaintings(uri: URI, minWidth: Int, maxResults: Int): Array[String] = {
    new GsonBuilder().create().fromJson(IOUtils.toString(
      uri,
      "UTF-8"
    ), classOf[util.ArrayList[util.Map[String, AnyRef]]])
      .asScala.map(_.asScala).filter(_ ("width").asInstanceOf[Number].doubleValue() > minWidth)
      .map(_ ("image").toString.stripSuffix("!Large.jpg"))
      .take(maxResults)
      .map(file => {
        val fileName = Normalizer.normalize(
          file.split("/").takeRight(2).mkString("/"),
          Normalizer.Form.NFD
        ).replaceAll("[^\\p{ASCII}]", "")
        val localFile = new File(new File("wikiart"), fileName)
        try {
          if (!localFile.exists()) {
            FileUtils.writeByteArrayToFile(localFile, IOUtils.toByteArray(new URI(file)))
          }
          "file:///" + localFile.getAbsolutePath.replaceAllLiterally("\\", "/").stripPrefix("/")
        } catch {
          case e: Throwable =>
            e.printStackTrace()
            ""
        }
      }).filterNot(_.isEmpty).toArray
  }

  /**
   * This function takes a list of integers and returns a new list with the same
   * elements, but with a binary fill pattern.
   *
   * For example, given the list [1, 2, 3, 4, 5], the function would return
   * [1, 3, 2, 4, 5].
   *
   * @docgenVersion 9
   */
  def binaryFill(seq: List[Int]): List[Int] = {
    if (seq.size < 3) seq
    else {
      val a = seq.take(1)
      val b = seq.drop(1).take(1)
      val c = seq.drop(2)
      a ++ binaryFill(c) ++ b
    }
  }

  /**
   * Animate the training of a visual network.
   *
   * @param contentUrl  The content image URL.
   * @param initUrl     The initialization image URL.
   * @param canvases    The buffer of canvases.
   * @param networks    The buffer of networks.
   * @param optimizer   The image optimizer.
   * @param resolutions The sequence of resolutions.
   * @param renderingFn The rendering function.
   * @param getParams   The function to get parameters.
   * @param delay       The delay in milliseconds.
   * @param log         The notebook output.
   * @docgenVersion 9
   */
  def animate
  (
    contentUrl: String,
    initUrl: String,
    canvases: mutable.Buffer[RefAtomicReference[Tensor]],
    networks: mutable.Buffer[(Double, VisualNetwork)],
    optimizer: ImageOptimizer,
    resolutions: Seq[Double],
    renderingFn: Seq[Int] => PipelineNetwork = x => new PipelineNetwork(1),
    getParams: (mutable.Buffer[(Double, VisualNetwork)], Double) => VisualNetwork = (networks: mutable.Buffer[(Double, VisualNetwork)], x: Double) => {
      networks.head._2
    },
    delay: Int = 100
  )(implicit log: NotebookOutput) = {
    for (res <- resolutions) {
      log.h1("Resolution " + res)
      NotebookRunner.withMonitoredGif(() => {
        cyclicalAnimation(canvases.map(_.get()).filter(_ != null).map(tensor => {
          if (null == renderingFn) {
            tensor
          } else {
            val renderer = renderingFn(tensor.getDimensions)
            try {
              Result.getData0(renderer.eval(tensor))
            } finally {
              renderer.freeRef()
            }
          }
        }).toList)
      }, delay = delay) {
        for (i <- binaryFill((0 until networks.size).toList)) {
          val (name, network) = networks(i)
          log.h2(name.toString)
          val canvas = canvases(i)
          CudaSettings.INSTANCE().setDefaultPrecision(network.precision)
          val content = ImageArtUtil.loadImage(log, contentUrl, res.toInt)
          if (null == canvas.get) {
            implicit val nullNotebookOutput = new NullNotebookOutput()
            val l = canvases.zipWithIndex.take(i).filter(_._1.get() != null).lastOption
            val r = canvases.zipWithIndex.drop(i + 1).reverse.filter(_._1.get() != null).lastOption
            if (l.isDefined && r.isDefined && l.get._2 != r.get._2) {
              val tensor = l.get._1.get().add(r.get._1.get())
              tensor.scaleInPlace(0.5);
              canvas.set(tensor)
            } else {
              canvas.set(load(Tensor.fromRGB(content), initUrl))
            }
          }
          else {
            canvas.set(Tensor.fromRGB(ImageUtil.resize(canvas.get.toRgbImage, content.getWidth, content.getHeight)))
          }
          val trainable = network(canvas.get, Tensor.fromRGB(content))
          ArtUtil.resetPrecision(trainable.addRef().asInstanceOf[Trainable], network.precision)
          optimizer.optimize(canvas.get, trainable)
        }
      }
      val canvasSize = canvases.size
      (1 until canvasSize).reverse.foreach(canvases.insert(_, new RefAtomicReference[Tensor]()))
      val networkSize = networks.size
      (1 until networkSize).reverse.foreach(i => {
        val after = networks(i)
        val before = networks(i - 1)
        val avg = (after._1.toDouble + before._1.toDouble) / 2
        networks.insert(i, (avg, getParams(networks, avg)))
      })
    }
  }

  /**
   * Paints an image on a given canvas using a given visual network and optimizer.
   *
   * @param contentUrl  the URL of the content image
   * @param initUrl     the URL of the initialization image
   * @param canvas      the canvas to paint on
   * @param network     the visual network to use
   * @param optimizer   the optimizer to use
   * @param aspect      the aspect ratio of the content image
   * @param resolutions the resolutions to use
   * @param sub         the notebook output
   * @return the loss value after painting
   * @docgenVersion 9
   */
  def paint
  (
    contentUrl: String,
    initUrl: String,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: ImageOptimizer,
    aspect: Option[Double],
    resolutions: Seq[Double]
  )(implicit sub: NotebookOutput): Double = paint(
    contentUrl = contentUrl,
    initFn = load(_, initUrl),
    canvas = canvas,
    network = network,
    optimizer = optimizer,
    aspect = aspect,
    resolutions = resolutions)

  /**
   * This function paints the given content URL on a canvas using the given init function, visual network, and image optimizer.
   * The resolutions and rendering function are optional.
   *
   * @docgenVersion 9
   */
  def paint
  (
    contentUrl: String,
    initFn: Tensor => Tensor,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: ImageOptimizer,
    resolutions: Seq[Double],
    renderingFn: Seq[Int] => PipelineNetwork = x => new PipelineNetwork(1),
    aspect: Option[Double] = None
  )(implicit log: NotebookOutput): Double = {
    paint_aspectFn(
      contentUrl,
      initFn,
      canvas,
      network,
      optimizer,
      resolutions,
      heightFn = aspect.map(a => (w: Int) => (w * a).toInt)
    )
  }

  /**
   * This function prepares the style image for training.
   *
   * @param contentTensor The content image tensor.
   * @param network       The visual network.
   * @param canvas        The canvas tensor.
   * @param log           The notebook output.
   * @return The trainable.
   * @docgenVersion 9
   */
  def stylePrepFn
  (
    contentTensor: Tensor,
    network: VisualNetwork,
    canvas: Tensor
  )(implicit log: NotebookOutput): Trainable = {
    CudaSettings.INSTANCE().setDefaultPrecision(Precision.Float)
    val trainable: Trainable = network.apply(canvas, contentTensor)
    ArtUtil.resetPrecision(trainable.addRef(), network.precision)
    trainable
  }

  /**
   * Prepares the style image for training.
   *
   * @param contentUrl The URL of the content image.
   * @param network    The visual network to use.
   * @param canvas     The canvas tensor.
   * @param width      The width of the content image.
   * @param height     The height of the content image.
   * @param log        The notebook output.
   * @docgenVersion 9
   */
  def stylePrepFn
  (
    contentUrl: String,
    network: VisualNetwork,
    canvas: Tensor,
    width: Double,
    height: Option[Int] = None
  )(implicit log: NotebookOutput): Trainable = {
    stylePrepFn(
      contentTensor = Tensor.fromRGB(if (height.isDefined) {
        ImageArtUtil.loadImage(log, contentUrl, width.toInt, height.get)
      } else {
        ImageArtUtil.loadImage(log, contentUrl, width.toInt)
      }),
      network = network,
      canvas = canvas
    )
  }


  /**
   * contentPrepFn prepares contentUrl for use with initFn, canvas, and network.
   * heightFn is an optional function that takes an Int and returns an Int.
   *
   * @docgenVersion 9
   */
  def contentPrepFn
  (
    contentUrl: String,
    initFn: Tensor => Tensor,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    heightFn: Option[Int => Int] = None
  )(implicit log: NotebookOutput): Double => (Tensor, Trainable) = {
    (res: Double) => {
      CudaSettings.INSTANCE().setDefaultPrecision(Precision.Float)
      val content = if (heightFn.isDefined) {
        ImageArtUtil.loadImage(log, contentUrl, res.toInt, heightFn.get.apply(res.toInt))
      } else {
        ImageArtUtil.loadImage(log, contentUrl, res.toInt)
      }
      val currentContent = Tensor.fromRGB(content)
      val currentCanvas: Tensor = updateCanvas(canvas, initFn, currentContent.addRef())(content.getWidth, content.getHeight)
      val trainable: Trainable = network.apply(currentCanvas.addRef(), currentContent)
      ArtUtil.resetPrecision(trainable.addRef(), network.precision)
      (currentCanvas, trainable)
    }
  }

  /**
   * Updates the canvas with the given content tensor.
   *
   * @param canvas        The canvas to update.
   * @param initFn        A function to initialize the canvas.
   * @param contentTensor The tensor to use for the content.
   * @param width         The width of the canvas.
   * @param height        The height of the canvas.
   * @docgenVersion 9
   */
  private def updateCanvas(canvas: RefAtomicReference[Tensor], initFn: Tensor => Tensor, contentTensor: Tensor)(width: Int, height: Int) = {
    require(null != canvas)
    var currentCanvas: Tensor = canvas.get()
    currentCanvas = if (null == currentCanvas) {
      initFn(contentTensor)
    } else {
      contentTensor.freeRef()
      val dims = currentCanvas.getDimensions()
      if (width == dims(0) && height == dims(1)) {
        currentCanvas
      } else {
        val image = currentCanvas.toRgbImage
        currentCanvas.freeRef()
        Tensor.fromRGB(ImageUtil.resize(image, width, height))
      }
    }
    canvas.set(currentCanvas.addRef())
    currentCanvas
  }

  /**
   * This function paints an image using the given optimizer, resolutions, and prep.
   *
   * @param optimizer   The ImageOptimizer to use.
   * @param resolutions The resolutions to use.
   * @param prep        The prep function to use.
   * @param log         The NotebookOutput to use.
   * @return The painted image.
   * @docgenVersion 9
   */
  def paint
  (
    optimizer: ImageOptimizer,
    resolutions: Seq[Double],
    prep: Double => (Tensor, Trainable)
  )(implicit log: NotebookOutput): Double = {
    (for (res <- resolutions.toArray) yield {
      if (resolutions.size > 1) log.h1("Resolution " + res)
      val (currentCanvas: Tensor, trainable: Trainable) = prep(res)
      optimizer.optimize(currentCanvas, trainable)
    }).last.finalValue
  }

  /**
   * This function paints an aspect of the given content URL.
   * The initFn is used to initialize the canvas.
   * The network is used to resolve the VisualNetwork.
   * The optimizer is used to optimize the image.
   * The resolutions are used to determine the height of the canvas.
   * The heightFn is used to determine the height of the canvas.
   *
   * @docgenVersion 9
   */
  def paint_aspectFn
  (
    contentUrl: String,
    initFn: Tensor => Tensor,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: ImageOptimizer,
    resolutions: Seq[Double],
    heightFn: Option[Int => Int]
  )(implicit log: NotebookOutput): Double = {
    paint(
      optimizer = optimizer,
      resolutions = resolutions,
      prep = contentPrepFn(
        contentUrl = contentUrl,
        initFn = initFn,
        canvas = canvas,
        network = network,
        heightFn = heightFn))
  }

  /**
   * Paints a single image.
   *
   * @param contentUrl  The URL of the content image.
   * @param initFn      A function that initializes the canvas.
   * @param canvas      The canvas to paint on.
   * @param network     The visual network.
   * @param optimizer   The image optimizer.
   * @param resolutions The resolutions to use.
   * @param log         The notebook output.
   * @return The loss.
   * @docgenVersion 9
   */
  def paint_single
  (
    contentUrl: String,
    initFn: Tensor => Tensor,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: ImageOptimizer,
    resolutions: Double*
  )(implicit log: NotebookOutput): Double = paint(
    contentUrl = contentUrl,
    initFn = initFn,
    canvas = canvas,
    network = network,
    optimizer = optimizer,
    renderingFn = x => new PipelineNetwork(1),
    resolutions = resolutions)

  /**
   * Paints a single view.
   *
   * @param contentUrl  The content URL.
   * @param initFn      The initialization function.
   * @param canvas      The canvas.
   * @param network     The visual network.
   * @param optimizer   The image optimizer.
   * @param renderingFn The rendering function.
   * @param resolutions The resolutions.
   * @param log         The notebook output.
   * @return The double.
   * @docgenVersion 9
   */
  def paint_single_view
  (
    contentUrl: String,
    initFn: Tensor => Tensor,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: ImageOptimizer,
    renderingFn: Seq[Int] => PipelineNetwork,
    resolutions: Double*
  )(implicit log: NotebookOutput): Double = paint(
    contentUrl = contentUrl,
    initFn = initFn,
    canvas = canvas,
    network = network,
    optimizer = optimizer,
    renderingFn = renderingFn,
    resolutions = resolutions)

  /**
   * This function texture() takes in an aspect ratio, an initial URL, a canvas, a network, and an optimizer, as well as a sequence of resolutions, and returns a double.*
   *
   * @docgenVersion 9
   */
  def texture(aspectRatio: Double, initUrl: String, canvas: RefAtomicReference[Tensor], network: VisualNetwork, optimizer: ImageOptimizer, resolutions: Seq[Double])(implicit log: NotebookOutput): Double = {
    /**
     * Prepares the given width.
     *
     * @docgenVersion 9
     */
    def prep(width: Double) = {
      CudaSettings.INSTANCE().setDefaultPrecision(network.precision)
      val height = width * aspectRatio
      var content = ImageArtUtil.loadImage(log, initUrl, width.toInt, height.toInt)
      val contentTensor = if (null == content) {
        new Tensor(width.toInt, height.toInt, 3).map((x: Double) => FastRandom.INSTANCE.random())
      } else {
        Tensor.fromRGB(content)
      }
      if (null == content) content = contentTensor.toImage
      require(null != canvas)

      /**
       * Updates the canvas with the current content. If there is no current content,
       * the canvas is loaded with the initial URL. Otherwise, the content is resized
       * to fit the width and height of the canvas.
       *
       * @docgenVersion 9
       */
      def updateCanvas(currentCanvas: Tensor) = {
        if (null == currentCanvas) {
          load(contentTensor, initUrl)
        } else {
          val w = if (null == content) width.toInt else content.getWidth
          val h = if (null == content) height.toInt else content.getHeight
          Tensor.fromRGB(ImageUtil.resize(currentCanvas.toRgbImage, w, h))
        }
      }

      val currentCanvas: Tensor = updateCanvas(canvas.get())
      canvas.set(currentCanvas)
      val trainable = network.apply(currentCanvas, contentTensor)
      ArtUtil.resetPrecision(trainable.addRef().asInstanceOf[Trainable], network.precision)
      (currentCanvas, trainable)
    }

    /**
     * Runs the optimizer on the given canvas and trainable object.
     * Renders the result and returns it as an RGB image.
     *
     * @docgenVersion 9
     */
    def run(currentCanvas: Tensor, trainable: Trainable) = {
      try {
        optimizer.optimize(() => {
          val render = optimizer.render(currentCanvas.addRef())
          val image = render.toRgbImage
          render.freeRef()
          image
        }, trainable)
      } finally {
        currentCanvas.freeRef()
      }
    }

    if (resolutions.size == 1) {
      val (currentCanvas: Tensor, trainable: Trainable) = prep(resolutions.head)
      run(currentCanvas, trainable).finalValue
    } else {
      (for (res <- resolutions) yield {
        log.h1("Resolution " + res)
        val (currentCanvas: Tensor, trainable: Trainable) = prep(res)
        run(currentCanvas, trainable)
      }).last.finalValue
    }
  }

  /**
   * Overrides the default apply method to also set up global handlers for the HTTP server and
   * to generate CUDA reports.
   *
   * @docgenVersion 9
   */
  override def apply(log: NotebookOutput): R = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    ImageArtUtil.cudaReports(log, cudaLog)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(maxImageSize)
    super.apply(log)
  }

  def cudaLog = false

  def maxImageSize = 10000
}
