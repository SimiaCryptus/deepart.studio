package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.Point

/**
 * A SphericalView is an IndexedView with two angles.
 *
 * @docgenVersion 9
 */
case class SphericalView(angle1: Double, angle2: Double) extends IndexedView {

  /**
   * This is a mapping function that returns a point.
   * If there is an error, it will return null.
   *
   * @docgenVersion 9
   */
  def mappingFunction() = (point: Point) =>
    try {
      angularToCanvasCoords(cartisianToAngularCoords(sceneToCartesianCoords(point)))
    } catch {
      case e: Throwable => null
    }

  /**
   * Converts a point in scene coordinates to cartesian coordinates.
   *
   * @param point the point in scene coordinates
   * @return the point in cartesian coordinates
   * @docgenVersion 9
   */
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

  /**
   * Converts a point in cartesian coordinates to scene coordinates.
   *
   * @param pt the point in cartesian coordinates
   * @return the point in scene coordinates
   * @docgenVersion 9
   */
  def cartesianToSceneCoords(pt: Array[Double]): Point = {
    var Array(x, y, z) = pt
    val z2 = z * Math.cos(-angle2) + y * Math.sin(-angle2)
    val y2 = y * Math.cos(-angle2) - z * Math.sin(-angle2)
    y = y2;
    z = z2;
    val x1 = x * Math.cos(-angle1) + y * Math.sin(-angle1)
    val y1 = y * Math.cos(-angle1) - x * Math.sin(-angle1)
    x = x1;
    y = y1;
    new Point(x, y)
  }

  /**
   * Converts a point in cartesian coordinates to angular coordinates.
   *
   * @param pt the point in cartesian coordinates
   * @return the point in angular coordinates
   * @docgenVersion 9
   */
  def cartisianToAngularCoords(pt: Array[Double]): Array[Double] = {
    _cartisianToAngularCoords(pt)
  }

  /**
   * Converts a cartesian point to angular coordinates.
   *
   * @param pt the cartesian point to convert
   * @return the angular coordinates
   * @docgenVersion 9
   */
  private def _cartisianToAngularCoords(pt: Array[Double]) = {
    val Array(x, y, z) = pt
    val v = (Math.atan2(z, Math.sqrt(1 - z * z)) / (Math.PI)) + 0.5
    val u = Math.atan2(x, y) / (2 * Math.PI)
    Array(v, u)
  }

  /**
   * Converts an array of angular coordinates to cartesian coordinates.
   *
   * @param pt The array of angular coordinates.
   * @return The array of cartesian coordinates.
   * @docgenVersion 9
   */
  def angularToCartesianCoords(pt: Array[Double]) = {
    val v = (pt(0) - 0.5) * (Math.PI)
    val u = pt(1) * (2 * Math.PI)
    val z = Math.sin(v)
    val x = Math.abs(1 - z * z) * Math.sin(u)
    val y = Math.abs(1 - z * z) * Math.cos(u)
    Array(x, y, z)
  }

  /**
   * Converts a point from canvas coordinates to angular coordinates.
   *
   * @param pt the point to convert, in canvas coordinates
   * @return the point in angular coordinates
   * @docgenVersion 9
   */
  def canvasToAngularCoords(pt: Point) = {
    var x = Math.sin(pt.x * (Math.PI / 2))
    var y = pt.y
    x = (x + 1) / 2
    y = (y + 1) / 2
    Array(x, y)
  }

  /**
   * Converts an angular point to canvas coordinates.
   *
   * @param pt the point to convert, in the form [x, y]
   * @return the converted point
   * @docgenVersion 9
   */
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

  /**
   * Returns true if the 'other' object is an instance of RotationalGroupView.
   *
   * @docgenVersion 9
   */
  override def canEqual(other: Any): Boolean = other.isInstanceOf[RotationalGroupView]

  /**
   * Returns true if the specified object is equal to this one. Two RotationalGroupViews are equal if they have the same angles.
   *
   * @docgenVersion 9
   */
  override def equals(other: Any): Boolean = other match {
    case that: RotationalGroupView =>
      (that canEqual this) &&
        angle1 == angle1 &&
        angle2 == angle2
    case _ => false
  }

  /**
   * Calculates the hash code for this object.
   *
   * @return the hash code for this object
   * @docgenVersion 9
   */
  override def hashCode(): Int = {
    val state = Seq(angle1, angle2)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}
