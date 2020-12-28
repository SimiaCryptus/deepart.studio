package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.mindseye.art.util.GeometricArt
import com.simiacryptus.mindseye.lang.Layer

trait ImageView extends GeometricArt {
  def getView(canvasDims: Array[Int]): Layer

  protected def cross[X, Y](xs: Traversable[X], ys: Traversable[Y]) = for {x <- xs; y <- ys} yield (x, y)
}
