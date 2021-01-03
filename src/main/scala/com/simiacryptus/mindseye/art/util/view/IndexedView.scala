package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.{Point, Raster}
import com.simiacryptus.mindseye.lang.Layer
import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer

import scala.collection.mutable

object IndexedView {
  val cache = new mutable.HashMap[(List[Int], IndexedView), Layer]()
}
abstract class IndexedView extends ImageView {
  def filterCircle = true
  def globalCache = false
  val cache = new mutable.HashMap[List[Int], Layer]()

  def getView(canvasDims: Array[Int]): Layer = {
    def layer = {
      val raster: Raster = new Raster(canvasDims(0), canvasDims(1)).setFilterCircle(filterCircle)
      new ImgIndexMapViewLayer(raster, raster.buildPixelMap(mappingFunction()(_)))
    }
    if(globalCache) IndexedView.cache.get((canvasDims.toList, IndexedView.this)).getOrElse({
      IndexedView.cache.synchronized {
        IndexedView.cache.getOrElseUpdate((canvasDims.toList, IndexedView.this), layer)
      }
    }).addRef().asInstanceOf[Layer] else this.cache.get(canvasDims.toList).getOrElse({
      this.cache.synchronized {
        this.cache.getOrElseUpdate(canvasDims.toList, layer)
      }
    }).addRef().asInstanceOf[Layer]
  }

  def mappingFunction(): Point => Point
}