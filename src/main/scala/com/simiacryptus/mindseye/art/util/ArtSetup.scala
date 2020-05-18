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
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.mindseye.util.ImageUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput, NullNotebookOutput}
import com.simiacryptus.ref.wrappers.RefAtomicReference
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.{InteractiveSetup, NotebookRunner, RepeatedInteractiveSetup}
import com.simiacryptus.util.FastRandom
import org.apache.commons.io.{FileUtils, IOUtils}

import scala.collection.JavaConversions._
import scala.collection.mutable

object ArtSetup {
  @JsonIgnore
  @transient implicit val s3client: AmazonS3 = AmazonS3ClientBuilder.standard().withRegion(EC2Util.REGION).build()
  @JsonIgnore
  @transient implicit val ec2client: AmazonEC2 = AmazonEC2ClientBuilder.standard().withRegion(EC2Util.REGION).build()

}

trait ArtSetup[T <: AnyRef] extends InteractiveSetup[T] with TaskRegistry {

  override def description: String = super.description

  override def className: String = getClass.getSimpleName.stripSuffix("$")

  def getPaintingsBySearch(searchWord: String, minWidth: Int): Array[String] = {
    getPaintings(new URI("https://www.wikiart.org/en/search/" + URLEncoder.encode(searchWord, "UTF-8").replaceAllLiterally("+", "%20") + "/1?json=2"), minWidth, 100)
  }

  def getPaintingsByArtist(artist: String, minWidth: Int): Array[String] = {
    getPaintings(new URI("https://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl=" + artist), minWidth, 100)
  }

  def getPaintings(uri: URI, minWidth: Int, maxResults: Int): Array[String] = {
    new GsonBuilder().create().fromJson(IOUtils.toString(
      uri,
      "UTF-8"
    ), classOf[util.ArrayList[util.Map[String, AnyRef]]])
      .filter(_ ("width").asInstanceOf[Number].doubleValue() > minWidth)
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

  def binaryFill(seq: List[Int]): List[Int] = {
    if (seq.size < 3) seq
    else {
      val a = seq.take(1)
      val b = seq.drop(1).take(1)
      val c = seq.drop(2)
      a ++ binaryFill(c) ++ b
    }
  }

  def animate
  (
    contentUrl: String,
    initUrl: String,
    canvases: mutable.Buffer[RefAtomicReference[Tensor]],
    networks: mutable.Buffer[(Double, VisualNetwork)],
    optimizer: BasicOptimizer,
    resolutions: Seq[Double],
    renderingFn: Seq[Int] => PipelineNetwork = null,
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
            val result = renderer.eval(tensor)
            renderer.freeRef()
            val tensorList = result.getData
            result.freeRef()
            val data = tensorList.get(0)
            tensorList.freeRef()
            data
          }
        }))
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

  def paint
  (
    contentUrl: String,
    initUrl: String,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: BasicOptimizer,
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

  def paint
  (
    contentUrl: String,
    initFn: Tensor => Tensor,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: BasicOptimizer,
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
      renderingFn,
      heightFn = aspect.map(a => (w: Int) => (w * a).toInt)
    )
  }

  def paint_aspectFn
  (
    contentUrl: String,
    initFn: Tensor => Tensor,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: BasicOptimizer,
    resolutions: Seq[Double],
    renderingFn: Seq[Int] => PipelineNetwork = x => new PipelineNetwork(1),
    heightFn: Option[Int => Int]
  )(implicit log: NotebookOutput): Double = {

    def prep(res: Double) = {
      CudaSettings.INSTANCE().setDefaultPrecision(network.precision)
      var content = if (heightFn.isDefined) {
        ImageArtUtil.loadImage(log, contentUrl, res.toInt, heightFn.get.apply(res.toInt))
      } else {
        ImageArtUtil.loadImage(log, contentUrl, res.toInt)
      }
      val contentTensor = if (null == content) {
        val tensor = new Tensor(res.toInt, heightFn.map(_.apply(res.toInt)).getOrElse(res.toInt), 3)
        val map = tensor.map((x: Double) => FastRandom.INSTANCE.random())
        tensor.freeRef()
        map
      } else {
        Tensor.fromRGB(content)
      }
      if (null == content) content = contentTensor.toImage

      def updateCanvas(currentCanvas: Tensor) = {
        if (null == currentCanvas) {
          initFn(contentTensor.addRef())
        } else {
          val width = if (null == content) res.toInt else content.getWidth
          val height = if (null == content) res.toInt else content.getHeight
          val image = currentCanvas.toRgbImage
          currentCanvas.freeRef()
          Tensor.fromRGB(ImageUtil.resize(image, width, height))
        }
      }

      require(null != canvas)
      val currentCanvas: Tensor = updateCanvas(canvas.get())
      canvas.set(currentCanvas.addRef())
      val trainable: Trainable = network.apply(currentCanvas.addRef(), contentTensor)
      ArtUtil.resetPrecision(trainable.addRef().asInstanceOf[Trainable], network.precision)
      (currentCanvas, trainable)
    }

    if (resolutions.size == 1) {
      val (currentCanvas: Tensor, trainable: Trainable) = prep(resolutions.head)
      optimizer.optimize(currentCanvas, trainable)
    } else {
      (for (res <- resolutions.toArray) yield {
        log.h1("Resolution " + res)
        val (currentCanvas: Tensor, trainable: Trainable) = prep(res)
        optimizer.optimize(currentCanvas, trainable)
      }).last
    }
    canvas.freeRef()
  }

  def paint_single
  (
    contentUrl: String,
    initFn: Tensor => Tensor,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: BasicOptimizer,
    resolutions: Double*
  )(implicit log: NotebookOutput): Double = paint(
    contentUrl = contentUrl,
    initFn = initFn,
    canvas = canvas,
    network = network,
    optimizer = optimizer,
    renderingFn = x => new PipelineNetwork(1),
    resolutions = resolutions)

  def paint_single_view
  (
    contentUrl: String,
    initFn: Tensor => Tensor,
    canvas: RefAtomicReference[Tensor],
    network: VisualNetwork,
    optimizer: BasicOptimizer,
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

  def texture(aspectRatio: Double, initUrl: String, canvas: RefAtomicReference[Tensor], network: VisualNetwork, optimizer: BasicOptimizer, resolutions: Seq[Double])(implicit log: NotebookOutput): Double = {
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
      run(currentCanvas, trainable)
    } else {
      (for (res <- resolutions) yield {
        log.h1("Resolution " + res)
        val (currentCanvas: Tensor, trainable: Trainable) = prep(res)
        run(currentCanvas, trainable)
      }).last
    }
  }

  override def apply(log: NotebookOutput): T = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    ImageArtUtil.cudaReports(log, cudaLog)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(maxImageSize)
    super.apply(log)
  }

  def cudaLog = false

  def maxImageSize = 10000
}

trait RepeatedArtSetup[T <: AnyRef] extends RepeatedInteractiveSetup[T] {

  override def apply(log: NotebookOutput): T = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    ImageArtUtil.cudaReports(log, cudaLog)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(maxImageSize)
    super.apply(log)
  }

  def cudaLog = false

  def maxImageSize = 10000
}
