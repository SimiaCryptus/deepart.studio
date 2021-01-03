package com.simiacryptus.mindseye.art.util.view

import java.awt.Desktop
import java.awt.image.BufferedImage
import java.io.File
import java.net.URL

import com.simiacryptus.math.{Point, Raster}
import com.simiacryptus.mindseye.art.util.view.RotationalGroupView.ICOSOHEDRON
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer
import com.simiacryptus.mindseye.util.ImageUtil
import javax.imageio.ImageIO
import org.scalatest.flatspec.AnyFlatSpec

class SphericalViewTest extends AnyFlatSpec {
  "File postprocess" should "Fix image" in {
    val file = new URL("http://symmetry.deepartist.org/IcosohedronTexture/3d5de29a-a5a7-4c1c-891f-a68a53bb9d62/etc/image_36640a6dfaf34e89.jpg")
    //System.out.println(file.getAbsolutePath)
    val image = ImageIO.read(file)
    val raster = new Raster(image.getWidth, image.getHeight).setFilterCircle(true)
    val reproject = new ImgIndexMapViewLayer(raster, raster.buildPixelMap(new RotationalGroupView(0, 0, ICOSOHEDRON).textureView()(_))).eval(Tensor.fromRGB(image)).getData.get(0).toRgbImage
    val output = File.createTempFile("reproject", ".png")
    ImageIO.write(reproject, "png", output)
    Desktop.getDesktop.browse(output.toURI)
  }

}
