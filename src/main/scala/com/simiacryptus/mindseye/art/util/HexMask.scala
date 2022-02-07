package com.simiacryptus.mindseye.art.util

import com.simiacryptus.math.Raster
import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer

import scala.collection.mutable

object HexMask {
  private val maskCache = new mutable.HashMap[List[Int], ImgIndexMapViewLayer]()
  private val wrapCache = new mutable.HashMap[List[Int], ImgIndexMapViewLayer]()
  private val a = Math.cos(30 * (Math.PI / 180)) / 2
  private val b = Math.cos(60 * (Math.PI / 180)) / 2

  def maskLayer(dims: Int*) = {
    maskCache.getOrElseUpdate(dims.toList, {
      val raster = new Raster(dims(0), dims(1))
      new ImgIndexMapViewLayer(raster, Array.tabulate(raster.sizeX * raster.sizeY)(idx => {
        val Array(x, y) = raster.fromIndex(idx)
        if (testHexBounds(x, y, raster.sizeX, raster.sizeY)) idx else -1
      }))
    })
  }

  def wrapLayer(dims: Int*) = {
    wrapCache.getOrElseUpdate(dims.toList, {
      val raster = new Raster(dims(0), dims(1))
      new ImgIndexMapViewLayer(raster, Array.tabulate(raster.sizeX * raster.sizeY)(idx => {
        val Array(x, y) = raster.fromIndex(idx)
        val sizeX = raster.sizeX
        val sizeY = raster.sizeY
        if (testHexBounds(x, y, sizeX, sizeY)) {
          idx
        } else {
          val wrappedY = (y + sizeY / 2) % sizeY
          val wrappedX = (x + sizeX / 2) % sizeX
          raster.toIndex(wrappedX, wrappedY)
        }
      }))
    })
  }

  private def testHexBounds(x: Int, y: Int, sizeX: Int, sizeY: Int): Boolean = {
    if (sizeX > sizeY) {
      testHexBounds(y, x, sizeY, sizeX)
    } else {
      val cx = ((x.toDouble - (sizeX / 2)) * (2 * a / sizeX)).abs
      val cy = ((y.toDouble - (sizeY / 2)) * (2 * a / sizeX)).abs
      if (cy < b) {
        true
      } else if (cy > 0.5) {
        false
      } else {
        if (((0.5 - cy) * (a / (0.5 - b))) > cx) {
          true
        } else {
          false
        }
      }
    }
  }
}
