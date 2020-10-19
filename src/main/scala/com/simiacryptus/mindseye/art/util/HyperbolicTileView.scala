package com.simiacryptus.mindseye.art.util

import com.simiacryptus.math.{PoincareDisk, Polygon, Raster}
import com.simiacryptus.mindseye.lang.Layer
import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer

import scala.collection.mutable

object HyperbolicTileView {
  private val cache = new mutable.HashMap[(List[Int], (Int,Int,Int)), Layer]()
}

case class HyperbolicTileView(p:Int, q:Int, i:Int) extends SymmetryTransform {
  def getSymmetricView(canvasDims: Array[Int]): Layer = {
    HyperbolicTileView.cache.get((canvasDims.toList, (p,q,i))).getOrElse({
      HyperbolicTileView.cache.synchronized {
        HyperbolicTileView.cache.getOrElseUpdate((canvasDims.toList, (p,q,i)), {
          val raster = new Raster(canvasDims(0), canvasDims(1))
          new ImgIndexMapViewLayer(raster, raster.pixelMap(Polygon.regularPolygon(p, q), i).getPixelMap)
        })
      }
    }).addRef().asInstanceOf[Layer]
  }
}
