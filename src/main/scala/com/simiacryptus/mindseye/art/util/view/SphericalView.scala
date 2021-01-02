package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.{Point, Raster}

case class SphericalView(angle1: Double, angle2: Double) extends IndexedView {
  def mappingFunction(): Point => Point = {
    (point: Point) => {
      try {
        var x = point.x
        var y = point.y
        var z = Math.sqrt(1 - x * x - y * y)
        val x1 = x * Math.cos(angle1) + y * Math.sin(angle1)
        val y1 = y * Math.cos(angle1) - x * Math.sin(angle1)
        x = x1;
        y = y1;
        val z2 = z * Math.cos(angle2) + y * Math.sin(angle2)
        val y2 = y * Math.cos(angle2) - z * Math.sin(angle2)
        y = y2;
        z = z2;
        var u = Math.atan2(z, Math.sqrt(1 - z * z)) / Math.PI
        var v = Math.atan2(x, y) / (2 * Math.PI)
        while (u < 0) u = u + 1
        while (v < 0) v = v + 1
        u = u % 1
        v = v % 1
        u = 2 * u - 1
        v = 2 * v - 1
        u = Math.asin(u) / (Math.PI / 2)
        new Point(u, v)
      } catch {
        case e : Throwable => null
      }
    }
  }
}
