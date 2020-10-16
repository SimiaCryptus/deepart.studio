package com.simiacryptus.mindseye.art.util

import com.simiacryptus.mindseye.lang.Layer

trait SymmetryTransform extends GeometricArt {
  def getSymmetricView(canvasDims: Array[Int]): Layer

  protected def cross[X, Y](xs: Traversable[X], ys: Traversable[Y]) = for {x <- xs; y <- ys} yield (x, y)
}
