package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.Point

case class SphericalView(angle1: Double, angle2: Double) extends IndexedView {

  def mappingFunction() = (point: Point) =>
    try {
      angularToCanvasCoords(cartisianToAngularCoords(sceneToCartesianCoords(point)))
    } catch {
      case e: Throwable => null
    }

  def sceneToCartesianCoords(point: Point): Array[Double] = {
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
    Array(x, y, z)
  }

  def cartesianToSceneCoords(pt:Array[Double]): Point = {
    var Array(x,y,z) = pt
    val z2 = z * Math.cos(-angle2) + y * Math.sin(-angle2)
    val y2 = y * Math.cos(-angle2) - z * Math.sin(-angle2)
    y = y2;
    z = z2;
    val x1 = x * Math.cos(angle1) + y * Math.sin(-angle1)
    val y1 = y * Math.cos(-angle1) - x * Math.sin(-angle1)
    x = x1;
    y = y1;
    new Point(x, y)
  }

  def cartisianToAngularCoords(pt: Array[Double]): Array[Double] = {
    _cartisianToAngularCoords(pt)
  }

  private def _cartisianToAngularCoords(pt: Array[Double]) = {
    val Array(x, y, z) = pt
    val v = (Math.atan2(z, Math.sqrt(1 - z * z)) / (Math.PI)) + 0.5
    val u = Math.atan2(x, y) / (2 * Math.PI)
    Array(v, u)
  }

  def angularToCartesianCoords(pt: Array[Double]) = {
    val v = (pt(0) - 0.5) * (Math.PI)
    val u = pt(1) * (2 * Math.PI)
    val z = Math.sin(v)
    val x = Math.abs(1 - z * z) * Math.sin(u)
    val y = Math.abs(1 - z * z) * Math.cos(u)
    Array(x, y, z)
  }

  def canvasToAngularCoords(pt: Point) = {
    var x = Math.sin(pt.x * (Math.PI / 2))
    var y = pt.y
    x = (x + 1) / 2
    y = (y + 1) / 2
    Array(x, y)
  }

  def angularToCanvasCoords(pt: Array[Double]) = {
    var x = pt(0)
    var y = pt(1)
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
