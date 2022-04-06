package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.Point

/**
 * This class defines a view of a tunnel, with a default step size of 0.2.
 *
 * @docgenVersion 9
 */
case class TunnelView(step: Double = 0.2) extends IndexedView {
  override def filterCircle = false

  /**
   * Returns a mapping function that maps a point to another point.
   *
   * @docgenVersion 9
   */
  def mappingFunction(): Point => Point = {
    (point: Point) => {
      try {
        /**
         * Returns true if the given point is within the bounds of the unit square,
         * false otherwise.
         *
         * @docgenVersion 9
         */
        def inBounds(pt: Point) = {
          pt.x.abs < 1 && pt.y.abs < 1
        }

        var r = point.rms()
        val unit = point.scale(1 / r)
        while (inBounds(unit.scale(r + step))) {
          r += step
        }
        unit.scale(r)
      } catch {
        case e: Throwable => null
      }
    }
  }

}
