package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.{Point, Raster}
import com.simiacryptus.mindseye.lang.Layer
import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer

import scala.collection.mutable

object IndexedView {
  private val cache = new mutable.HashMap[(List[Int], IndexedView), Layer]()
}
abstract class IndexedView extends ImageView {
  def filterCircle = true

  def getSymmetricView(canvasDims: Array[Int]): Layer = {
    IndexedView.cache.get((canvasDims.toList, IndexedView.this)).getOrElse({
      IndexedView.cache.synchronized {
        IndexedView.cache.getOrElseUpdate((canvasDims.toList, IndexedView.this), {
          val raster: Raster = new Raster(canvasDims(0), canvasDims(1)).setFilterCircle(filterCircle)
          new ImgIndexMapViewLayer(raster, raster.buildPixelMap(mappingFunction()(_)))
        })
      }
    }).addRef().asInstanceOf[Layer]
  }

  def mappingFunction(): Point => Point
}