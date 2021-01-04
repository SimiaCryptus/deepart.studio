package com.simiacryptus.mindseye.art.util.view

import java.awt.Desktop
import java.awt.image.BufferedImage
import java.io.{File, FileOutputStream, PrintWriter}
import java.net.URL

import com.simiacryptus.math.{Point, Raster}
import com.simiacryptus.mindseye.art.util.view.RotationalGroupView.ICOSOHEDRON
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer
import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer
import com.simiacryptus.mindseye.util.ImageUtil
import javax.imageio.ImageIO
import org.scalatest.flatspec.AnyFlatSpec

class SphericalViewTest extends AnyFlatSpec {
  "File postprocess" should "Generate STL" in {
    val file = new URL("http://symmetry.deepartist.org/IcosohedronTexture/1498e5dc-fb1e-45d5-a41f-7b3fdc5ceeba/etc/image_f5f2a97eba84fddf.jpg")
    var image = ImageIO.read(file)
    image = ImageUtil.resize(image, 128, true)
    var tensor = Tensor.fromRGB(image)
    val poolingLayer = new PoolingLayer()
    poolingLayer.setMode(PoolingLayer.PoolingMode.Avg)
    poolingLayer.setStrideXY(2,2)
    tensor = poolingLayer.asTensorFunction().apply(tensor)
    val Array(width,height,_) = tensor.getDimensions
    val output = File.createTempFile("render", ".stl")

    def toSphereCoords(x: Double, y: Double): Array[Double] = {
//      val v = Math.sin(((x / width)*2-1) * (Math.PI / 2)) * (Math.PI / 2)
      val v = ((x / width) - 0.5) * (Math.PI)
      val u = (y / height) * (2 * Math.PI)
      val z = Math.sin(v)
      Array(
        Math.abs(1 - z * z) * Math.sin(u),
        Math.abs(1 - z * z) * Math.cos(u),
        z
      )
    }

    def toBumpCoords(x: Double, y: Double): Array[Double] = {
      val pixel = tensor.getPixel(x.toInt, y.toInt).map(v => Math.pow(v / 255, 2)).sum
      toSphereCoords(x,y).map(_ * (pixel / 50 + 1))
    }

    val stl = new PrintWriter(new FileOutputStream(output))
    try {
      stl.println(s"solid ball")
      stl.println((for (x <- (0 until width).map(x=>((Math.asin((x.toDouble / width)*2-1) / (Math.PI / 2)) + 1) * (width/2.0)).par; y <- 0 until height) yield {
        s"""facet normal ${toSphereCoords((x + 0.25) % width, (y + 0.25) % height).map(_.toString).reduce(_ + " " + _)}
           |  outer loop
           |    vertex ${toBumpCoords(x, y).map(_.toString).reduce(_ + " " + _)}
           |    vertex ${toBumpCoords(x, (y + 1) % height).map(_.toString).reduce(_ + " " + _)}
           |    vertex ${toBumpCoords((x + 1) % width, y).map(_.toString).reduce(_ + " " + _)}
           |  endloop
           |endfacet
           |facet normal ${toSphereCoords(x + 0.75, y + 0.75).map(_.toString).reduce(_ + " " + _)}
           |  outer loop
           |    vertex ${toBumpCoords((x + 1) % width, y).map(_.toString).reduce(_ + " " + _)}
           |    vertex ${toBumpCoords(x, (y + 1) % height).map(_.toString).reduce(_ + " " + _)}
           |    vertex ${toBumpCoords((x + 1) % width, (y + 1) % height).map(_.toString).reduce(_ + " " + _)}
           |  endloop
           |endfacet""".stripMargin
      }).reduce(_ + "\n" + _))
    } finally {
      stl.close()
    }

    Desktop.getDesktop.browse(output.toURI)
  }

}
