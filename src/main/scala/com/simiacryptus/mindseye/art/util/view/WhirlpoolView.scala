package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.Point

case class WhirlpoolView(step: Double = 0.8, seedSize: Double = 0.2, theta_step: Double = Math.PI / 16) extends IndexedView {
  override def filterCircle = false

  def mappingFunction(): Point => Point = {
    (originalPoint: Point) => {
      try {
        var point: Point = originalPoint
        def inBounds(pt: Point) = {
          pt.x.abs > seedSize || pt.y.abs > seedSize
        }
        var r = point.rms() * step
        var theta = theta_step
        var continue = true
        while(continue) {
          val newPoint = point.scale(r / point.rms()).rotate(theta)
          if(inBounds(newPoint)) {
            point = newPoint
            r *= step
            theta += theta_step
          } else {
            point = newPoint
            continue = false;
          }
        }
        point
      } catch {
        case e : Throwable => null
      }
    }
  }

}
