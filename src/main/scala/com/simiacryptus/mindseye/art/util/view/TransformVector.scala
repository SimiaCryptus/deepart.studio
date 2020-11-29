package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.mindseye.art.util.{Permutation, ViewMask}
import com.simiacryptus.mindseye.lang.Layer
import com.simiacryptus.mindseye.layers.java.AffineImgViewLayer

case class TransformVector
(
  offset: Map[Array[Double], Permutation] = Map(Array(0.0, 0.0) -> Permutation.unity(3)),
  rotation: Double = 0.0,
  symmetry: Boolean = true,
  mask: ViewMask = ViewMask()
) extends ImageView {

  def *(right: TransformVector) = TransformVector(
    offset = cross(offset, right.offset).map(x => x._1._1.zip(x._2._1).map(t => t._1 + t._2) -> x._1._2 * x._2._2).toArray.toMap,
    rotation = rotation + right.rotation
  )

  def ^(n: Int): TransformVector = if (n <= 1) this else this * (this ^ (n - 1))


  def getSymmetricView(canvasDims: Array[Int]): Layer = {
    def singleView(offset: Array[Double], permutation: Permutation) = {
      val layer = new AffineImgViewLayer(canvasDims(0), canvasDims(1), true)
      layer.setOffsetX((offset(0) * canvasDims(0)).toInt)
      layer.setOffsetY((offset(1) * canvasDims(1)).toInt)
      layer.setChannelSelector(permutation.indices)
      val r = Math.pow(canvasDims.reduce(_ * _), 1.0 / canvasDims.size)
      layer.setrMin((mask.radius_min * r).toInt)
      layer.setrMax((mask.radius_max * r).toInt)
      layer.setxMin((mask.x_min * canvasDims(0)).toInt)
      layer.setxMax((mask.x_max * canvasDims(0)).toInt)
      layer.setyMin((mask.y_min * canvasDims(1)).toInt)
      layer.setyMax((mask.y_max * canvasDims(1)).toInt)
      if (rotation != 0) {
        layer.setRotationCenterX(canvasDims(0) / 2)
        layer.setRotationCenterY(canvasDims(1) / 2)
        layer.setRotationRadians(rotation % (2 * Math.PI))
      }
      layer
    }

    if (symmetry) {
      val nonzeroOffsets = offset.toList.filterNot(_._1.toList == List(0.0, 0.0, 0.0))
      if (nonzeroOffsets.isEmpty) {
        avg(Array(None, Option(singleView(Array(0.0, 0.0, 0.0), Permutation.unity(3)))))
      } else if (nonzeroOffsets.size == 1) {
        avg(Array(None, Option(singleView(nonzeroOffsets.head._1, nonzeroOffsets.head._2))))
      } else {
        avg((for ((offset, permutation) <- offset) yield {
          Option(singleView(offset, permutation))
        }).toArray)
      }
    } else {
      if (offset.isEmpty) {
        singleView(Array(0.0, 0.0, 0.0), Permutation.unity(3))
      } else if (offset.size == 1) {
        singleView(offset.head._1, offset.head._2)
      } else {
        avg((for ((offset, permutation) <- offset) yield {
          Option(singleView(offset, permutation))
        }).toArray)
      }
    }

  }

}
