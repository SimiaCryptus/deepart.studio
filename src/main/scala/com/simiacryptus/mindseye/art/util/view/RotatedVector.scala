package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.mindseye.art.util.{Permutation, ViewMask}
import com.simiacryptus.mindseye.lang.Layer
import com.simiacryptus.mindseye.layers.java.AffineImgViewLayer

/**
 * This class represents a rotated vector.
 *
 * @param rotation the rotation of the vector
 * @param symmetry whether the vector is symmetric
 * @param mask     the view mask of the vector
 * @docgenVersion 9
 */
case class RotatedVector
(
  rotation: Map[Double, Permutation] = Map.empty,
  symmetry: Boolean = true,
  mask: ViewMask = ViewMask()
) extends ImageView {

  /**
   * Returns a view of the specified dimensions.
   *
   * @param canvasDims the dimensions of the canvas
   * @return a layer containing the view
   * @docgenVersion 9
   */
  def getView(canvasDims: Array[Int]): Layer = {
    def maybeLayers = for ((rotation, permutation) <- rotation.toArray) yield {
      val layer = new AffineImgViewLayer(canvasDims(0), canvasDims(1), true)
      layer.setChannelSelector(permutation.indices)
      val canvasXY = canvasDims.take(2)
      val r = Math.pow(canvasXY.reduce(_ * _), 1.0 / canvasXY.size)
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
      Option(layer)
    }

    if (symmetry) {
      avg(Array(None) ++ maybeLayers)
    } else {
      avg(maybeLayers)
    }
  }

}
