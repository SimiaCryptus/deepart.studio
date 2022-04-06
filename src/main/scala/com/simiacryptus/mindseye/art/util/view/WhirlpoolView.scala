package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.Point

/**
 * This is the WhirlpoolView class, which inherits from the IndexedView class.
 * It has four parameters: step (default value 1.5d), borderLocation (default value 0.8d),
 * theta_step (default value Math.PI / 16), and trim (default value 0.05d).
 *
 * @docgenVersion 9
 */
case class WhirlpoolView(step: Double = 1.5, borderLocation: Double = 0.8, theta_step: Double = Math.PI / 16, trim: Double = 0.05) extends IndexedView {
  override def filterCircle = false

  /**
   * Returns a mapping function that maps a point to another point.
   *
   * @docgenVersion 9
   */
  def mappingFunction(): Point => Point = {
    (originalPoint: Point) => {
      try {
        if (1 - originalPoint.x.abs > (1 - trim)) {
          new Point(0, 0)
        } else if (1 - originalPoint.y.abs > (1 - trim)) {
          new Point(0, 0)
        } else {
          var point: Point = originalPoint

          /**
           * Returns true if the given point is within the center of the coordinate system,
           * and false otherwise.
           *
           * @docgenVersion 9
           */
          def withinCenter(pt: Point) = {
            (pt.x.abs < borderLocation && pt.y.abs < borderLocation)
          }

          /**
           * Returns true if the given point is out of bounds, false otherwise.
           *
           * @docgenVersion 9
           */
          def outOfBounds(pt: Point) = {
            (pt.x.abs >= 1 || pt.y.abs >= 1)
          }

          var r = point.rms() * step
          var theta = theta_step
          var continue = true
          while (continue) {
            val newPoint = point.scale(r / point.rms()).rotate(theta)
            if (withinCenter(newPoint)) {
              point = newPoint
              r *= step
              //theta += theta_step
            } else {
              if (!outOfBounds(newPoint)) {
                point = newPoint
              }
              continue = false;
            }
          }
          point
        }
      } catch {
        case e: Throwable => null
      }
    }
  }

}
