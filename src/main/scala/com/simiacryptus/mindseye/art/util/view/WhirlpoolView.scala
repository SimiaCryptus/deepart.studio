package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.Point

case class WhirlpoolView(step: Double = 1.5, borderLocation: Double = 0.8, theta_step: Double = Math.PI / 16, trim : Double = 0.05) extends IndexedView {
  override def filterCircle = false

  def mappingFunction(): Point => Point = {
    (originalPoint: Point) => {
      try {
        if(1 - originalPoint.x.abs > ( 1 - trim)) {
          new Point(0,0)
        } else if(1 - originalPoint.y.abs > ( 1 - trim)) {
          new Point(0,0)
        } else {
          var point: Point = originalPoint
          def withinCenter(pt: Point) = {
            (pt.x.abs < borderLocation && pt.y.abs < borderLocation)
          }
          def outOfBounds(pt: Point) = {
            (pt.x.abs >= 1 || pt.y.abs >= 1)
          }
          var r = point.rms() * step
          var theta = theta_step
          var continue = true
          while(continue) {
            val newPoint = point.scale(r / point.rms()).rotate(theta)
            if(withinCenter(newPoint)) {
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
        case e : Throwable => null
      }
    }
  }

}
