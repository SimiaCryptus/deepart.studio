package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.{Point, Raster}

case class SphericalView(angle1: Double, angle2: Double) extends IndexedView {

  def mappingFunction(): Point => Point = {
    (point: Point) => {
      try {
        val (x, y, z) = sceneToCartesianCoords(point)
        val (u: Double, v: Double) = cartisianToAngularCoords(x, y, z)
        angularToCanvasCoords(u,v)
      } catch {
        case e : Throwable => null
      }
    }
  }

  def sceneToCartesianCoords(point: Point): (Double, Double, Double) = {
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
    (x,y,z)
  }

  def cartisianToAngularCoords(x: Double, y: Double, z: Double): (Double, Double) = (
    Math.atan2(z, Math.sqrt(1 - z * z)) / Math.PI,
    Math.atan2(x, y) / (2 * Math.PI)
  )

  def angularToCanvasCoords(u:Double, v:Double) = {
    var x = u
    var y = v
    while (x < 0) x = x + 1
    while (y < 0) y = y + 1
    x = x % 1
    y = y % 1
    x = 2 * x - 1
    y = 2 * y - 1
    x = Math.asin(x) / (Math.PI / 2)
    new Point(x, y)
  }
}
