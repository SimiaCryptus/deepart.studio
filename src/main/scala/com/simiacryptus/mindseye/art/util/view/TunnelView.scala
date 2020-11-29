package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.Point

case class TunnelView(step: Double = 0.2) extends IndexedView {
  override def filterCircle = false

  def mappingFunction(): Point => Point = {
    (point: Point) => {
      try {
        var r = point.rms()
        if(r > 1) {
          r += 0;
        }
        val unit = point.scale(1/r)
        def inBounds(pt: Point) = {
          pt.x.abs < 1 && pt.y.abs < 1
        }

        while(inBounds(unit.scale(r))) {
          r += step
        }
        unit.scale(r - step)
      } catch {
        case e : Throwable => null
      }
    }
  }

}
