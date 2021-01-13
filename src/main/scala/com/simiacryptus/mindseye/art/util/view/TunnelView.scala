package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.Point

case class TunnelView(step: Double = 0.2) extends IndexedView {
  override def filterCircle = false

  def mappingFunction(): Point => Point = {
    (point: Point) => {
      try {
        def inBounds(pt: Point) = {
          pt.x.abs < 1 && pt.y.abs < 1
        }
        var r = point.rms()
        val unit = point.scale(1/r)
        while(inBounds(unit.scale(r + step))) {
          r += step
        }
        unit.scale(r)
      } catch {
        case e : Throwable => null
      }
    }
  }

}
