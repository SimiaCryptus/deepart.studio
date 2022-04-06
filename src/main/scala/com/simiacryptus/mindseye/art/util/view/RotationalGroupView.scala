package com.simiacryptus.mindseye.art.util.view

import java.io.{ByteArrayInputStream, InputStreamReader}

import com.google.gson.GsonBuilder
import com.simiacryptus.math.{Point, Raster}
import com.simiacryptus.mindseye.lang.Layer
import com.simiacryptus.mindseye.layers.java.ImgIndexMapViewLayer

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object RotationalGroupView {

  /**
   * Calculates the distance between two arrays of doubles.
   *
   * @param a the first array
   * @param b the second array
   * @return the distance between the arrays
   * @docgenVersion 9
   */
  def dist(a: Array[Double], b: Array[Double]): Double = {
    rms(a.zip(b).map(t => t._1 - t._2))
  }

  /**
   * This function sums the corresponding elements in two arrays of doubles,
   * returning a new array of doubles.
   *
   * @docgenVersion 9
   */
  def sum(a: Array[Double], b: Array[Double]): Array[Double] = {
    a.zip(b).map(t => t._1 + t._2)
  }

  /**
   * Calculates the dot product of two arrays.
   *
   * @param a The first array.
   * @param b The second array.
   * @return The dot product of the two arrays.
   * @docgenVersion 9
   */
  def dot(a: Array[Double], b: Array[Double]): Double = {
    a.zip(b).map(t => t._1 * t._2).sum
  }

  /**
   * Calculates the unit vector of an array of doubles.
   *
   * @param doubles the array of doubles
   * @return the unit vector of the array
   * @docgenVersion 9
   */
  def unitV(doubles: Array[Double]): Array[Double] = {
    val r = rms(doubles)
    doubles.map(_ / r)
  }

  /**
   * Calculates the root mean square of an array of doubles.
   *
   * @param doubles the array of doubles
   * @return the root mean square
   * @docgenVersion 9
   */
  def rms(doubles: Array[Double]): Double = {
    Math.sqrt(doubles.map(Math.pow(_, 2)).sum)
  }

  /**
   * Multiplies a point by a transformation matrix.
   *
   * @param point     the point to multiply
   * @param transform the transformation matrix
   * @return the transformed point
   * @docgenVersion 9
   */
  def multiply(point: Array[Double], transform: Array[Array[Double]]) = {
    unitV(Array(
      point(0) * transform(0)(0) + point(1) * transform(0)(1) + point(2) * transform(0)(2),
      point(0) * transform(1)(0) + point(1) * transform(1)(1) + point(2) * transform(1)(2),
      point(0) * transform(2)(0) + point(1) * transform(2)(1) + point(2) * transform(2)(2)
    ))
  }

  val TETRAHEDRON = 10
  val OCTOHEDRON = 22
  val ICOSOHEDRON = 60

}
import com.simiacryptus.mindseye.art.util.view.RotationalGroupView._

/**
 * This class represents a rotational group view, which is a type of spherical view.
 *
 * @param x    the x-coordinate
 * @param y    the y-coordinate
 * @param mode the mode (0 = azimuthal, 1 = polar)
 * @docgenVersion 9
 */
class RotationalGroupView(x: Double, y: Double, mode: Int) extends SphericalView(x, y) {

  protected val tileProbe = Array[Double](0, 0.1, 1)
  protected val vertexSeed = Array[Double](1, 0, 0)

  lazy val primaryTile: Array[Double] = {
    val vertices = icosohedronMatrices.map(multiply(vertexSeed, _)).toArray
    val topUnique = new ArrayBuffer[Array[Double]]()
    topUnique += vertices.minBy(dot(_, tileProbe))
    while (topUnique.size < 3) topUnique += vertices.filter(v => topUnique.map(dist(v, _)).min > 1e-2).minBy(p => topUnique.map(dot(p, _)).sum)
    unitV(topUnique.reduce((a, b) => sum(a, b)).map(_ / 3))
  }

  /**
   * Converts a point from the texture view to a point in the scene.
   *
   * @param point the point in the texture view
   * @return the point in the scene, or null if the conversion fails
   * @docgenVersion 9
   */
  def textureView(): Point => Point = {
    (point: Point) => {
      try {
        cartesianToSceneCoords(tileTransform(angularToCartesianCoords(canvasToAngularCoords(point))))
      } catch {
        case e: Throwable => null
      }
    }
  }


  def matrixData: String = {
    // See also https://mathandcode.com/2016/06/06/polyhedramathematica.html
    mode match {
      case TETRAHEDRON =>
        """
          |[[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
          | [[-0.5, -0.5, -0.70711], [0.5, 0.5, -0.70711], [0.70711, -0.70711, 0.]],
          | [[-0.5, -0.5, 0.70711], [0.5, 0.5, 0.70711], [-0.70711, 0.70711, 0.]],
          | [[-0.5, 0.5, -0.70711], [-0.5, 0.5, 0.70711], [0.70711, 0.70711, 0.]],
          | [[-0.5, 0.5, 0.70711], [-0.5, 0.5, -0.70711], [-0.70711, -0.70711, 0.]],
          | [[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]],
          | [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]],
          | [[0.5, -0.5, -0.70711], [0.5, -0.5, 0.70711], [-0.70711, -0.70711, 0.]],
          | [[0.5, -0.5, 0.70711], [0.5, -0.5, -0.70711], [0.70711, 0.70711, 0.]],
          | [[0.5, 0.5, -0.70711], [-0.5, -0.5, -0.70711], [-0.70711, 0.70711, 0.]],
          | [[0.5, 0.5, 0.70711], [-0.5, -0.5, 0.70711], [0.70711, -0.70711, 0.]],
          | [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]
          |""".stripMargin
      case OCTOHEDRON =>
        """
          |[[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
          | [[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]],
          | [[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]],
          | [[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]],
          | [[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]],
          | [[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]],
          | [[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]],
          | [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
          | [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]],
          | [[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]],
          | [[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]],
          | [[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]],
          | [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]],
          | [[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]],
          | [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]],
          | [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
          | [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]],
          | [[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]],
          | [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]],
          | [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]],
          | [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
          | [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
          | [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]],
          | [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]
          |""".stripMargin
      case ICOSOHEDRON =>
        """[[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
          | [[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]],
          | [[-0.5, -0.80902, -0.30902], [-0.80902, 0.30902, 0.5], [-0.30902, 0.5, -0.80902]],
          | [[-0.5, -0.80902, -0.30902], [0.80902, -0.30902, -0.5], [0.30902, -0.5, 0.80902]],
          | [[-0.5, -0.80902, 0.30902], [-0.80902, 0.30902, -0.5], [0.30902, -0.5, -0.80902]],
          | [[-0.5, -0.80902, 0.30902], [0.80902, -0.30902, 0.5], [-0.30902, 0.5, 0.80902]],
          | [[-0.5, 0.80902, -0.30902], [-0.80902, -0.30902, 0.5], [0.30902, 0.5, 0.80902]],
          | [[-0.5, 0.80902, -0.30902], [0.80902, 0.30902, -0.5], [-0.30902, -0.5, -0.80902]],
          | [[-0.5, 0.80902, 0.30902], [-0.80902, -0.30902, -0.5], [-0.30902, -0.5, 0.80902]],
          | [[-0.5, 0.80902, 0.30902], [0.80902, 0.30902, 0.5], [0.30902, 0.5, -0.80902]],
          | [[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]],
          | [[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]],
          | [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]],
          | [[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]],
          | [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]],
          | [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
          | [[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]],
          | [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]],
          | [[0.5, -0.80902, -0.30902], [-0.80902, -0.30902, -0.5], [0.30902, 0.5, -0.80902]],
          | [[0.5, -0.80902, -0.30902], [0.80902, 0.30902, 0.5], [-0.30902, -0.5, 0.80902]],
          | [[0.5, -0.80902, 0.30902], [-0.80902, -0.30902, 0.5], [-0.30902, -0.5, -0.80902]],
          | [[0.5, -0.80902, 0.30902], [0.80902, 0.30902, -0.5], [0.30902, 0.5, 0.80902]],
          | [[0.5, 0.80902, -0.30902], [-0.80902, 0.30902, -0.5], [-0.30902, 0.5, 0.80902]],
          | [[0.5, 0.80902, -0.30902], [0.80902, -0.30902, 0.5], [0.30902, -0.5, -0.80902]],
          | [[0.5, 0.80902, 0.30902], [-0.80902, 0.30902, 0.5], [0.30902, -0.5, 0.80902]],
          | [[0.5, 0.80902, 0.30902], [0.80902, -0.30902, -0.5], [-0.30902, 0.5, -0.80902]],
          | [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
          | [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
          | [[-0.80902, -0.30902, -0.5], [-0.30902, -0.5, 0.80902], [-0.5, 0.80902, 0.30902]],
          | [[-0.80902, -0.30902, -0.5], [0.30902, 0.5, -0.80902], [0.5, -0.80902, -0.30902]],
          | [[-0.80902, -0.30902, 0.5], [-0.30902, -0.5, -0.80902], [0.5, -0.80902, 0.30902]],
          | [[-0.80902, -0.30902, 0.5], [0.30902, 0.5, 0.80902], [-0.5, 0.80902, -0.30902]],
          | [[-0.80902, 0.30902, -0.5], [-0.30902, 0.5, 0.80902], [0.5, 0.80902, -0.30902]],
          | [[-0.80902, 0.30902, -0.5], [0.30902, -0.5, -0.80902], [-0.5, -0.80902, 0.30902]],
          | [[-0.80902, 0.30902, 0.5], [-0.30902, 0.5, -0.80902], [-0.5, -0.80902, -0.30902]],
          | [[-0.80902, 0.30902, 0.5], [0.30902, -0.5, 0.80902], [0.5, 0.80902, 0.30902]],
          | [[-0.30902, -0.5, -0.80902], [-0.5, 0.80902, -0.30902], [0.80902, 0.30902, -0.5]],
          | [[-0.30902, -0.5, -0.80902], [0.5, -0.80902, 0.30902], [-0.80902, -0.30902, 0.5]],
          | [[-0.30902, -0.5, 0.80902], [-0.5, 0.80902, 0.30902], [-0.80902, -0.30902, -0.5]],
          | [[-0.30902, -0.5, 0.80902], [0.5, -0.80902, -0.30902], [0.80902, 0.30902, 0.5]],
          | [[-0.30902, 0.5, -0.80902], [-0.5, -0.80902, -0.30902], [-0.80902, 0.30902, 0.5]],
          | [[-0.30902, 0.5, -0.80902], [0.5, 0.80902, 0.30902], [0.80902, -0.30902, -0.5]],
          | [[-0.30902, 0.5, 0.80902], [-0.5, -0.80902, 0.30902], [0.80902, -0.30902, 0.5]],
          | [[-0.30902, 0.5, 0.80902], [0.5, 0.80902, -0.30902], [-0.80902, 0.30902, -0.5]],
          | [[0.30902, -0.5, -0.80902], [-0.5, -0.80902, 0.30902], [-0.80902, 0.30902, -0.5]],
          | [[0.30902, -0.5, -0.80902], [0.5, 0.80902, -0.30902], [0.80902, -0.30902, 0.5]],
          | [[0.30902, -0.5, 0.80902], [-0.5, -0.80902, -0.30902], [0.80902, -0.30902, -0.5]],
          | [[0.30902, -0.5, 0.80902], [0.5, 0.80902, 0.30902], [-0.80902, 0.30902, 0.5]],
          | [[0.30902, 0.5, -0.80902], [-0.5, 0.80902, 0.30902], [0.80902, 0.30902, 0.5]],
          | [[0.30902, 0.5, -0.80902], [0.5, -0.80902, -0.30902], [-0.80902, -0.30902, -0.5]],
          | [[0.30902, 0.5, 0.80902], [-0.5, 0.80902, -0.30902], [-0.80902, -0.30902, 0.5]],
          | [[0.30902, 0.5, 0.80902], [0.5, -0.80902, 0.30902], [0.80902, 0.30902, -0.5]],
          | [[0.80902, -0.30902, -0.5], [-0.30902, 0.5, -0.80902], [0.5, 0.80902, 0.30902]],
          | [[0.80902, -0.30902, -0.5], [0.30902, -0.5, 0.80902], [-0.5, -0.80902, -0.30902]],
          | [[0.80902, -0.30902, 0.5], [-0.30902, 0.5, 0.80902], [-0.5, -0.80902, 0.30902]],
          | [[0.80902, -0.30902, 0.5], [0.30902, -0.5, -0.80902], [0.5, 0.80902, -0.30902]],
          | [[0.80902, 0.30902, -0.5], [-0.30902, -0.5, -0.80902], [-0.5, 0.80902, -0.30902]],
          | [[0.80902, 0.30902, -0.5], [0.30902, 0.5, 0.80902], [0.5, -0.80902, 0.30902]],
          | [[0.80902, 0.30902, 0.5], [-0.30902, -0.5, 0.80902], [0.5, -0.80902, -0.30902]],
          | [[0.80902, 0.30902, 0.5], [0.30902, 0.5, -0.80902], [-0.5, 0.80902, 0.30902]]]
          | """
    }
  }

  lazy val icosohedronMatrices: Seq[Array[Array[Double]]] = {
    new GsonBuilder().create.fromJson(new InputStreamReader(new ByteArrayInputStream(
      matrixData.stripMargin.getBytes("UTF-8")
    )), classOf[Array[Array[Array[Double]]]]
    ).toList
  }

  /**
   * Converts the given point in Cartesian coordinates to
   * angular coordinates using the tile transform.
   *
   * @docgenVersion 9
   */
  override def cartisianToAngularCoords(pt: Array[Double]) = {
    super.cartisianToAngularCoords(tileTransform(pt))
  }

  /**
   * Transforms the given point according to the icosohedron matrices, and returns the point that is closest to the primary tile.
   *
   * @docgenVersion 9
   */
  def tileTransform(pt: Array[Double]) = {
    icosohedronMatrices.map(multiply(pt, _)).minBy(dot(primaryTile, _))
  }

  def tileExpansion: ImageView = (canvasDims: Array[Int]) => {
    def layer = {
      val raster: Raster = new Raster(canvasDims(0), canvasDims(1)).setFilterCircle(false)
      new ImgIndexMapViewLayer(raster, raster.buildPixelMap(tileExpansionFunction()(_)))
    }

    if (globalCache) IndexedView.cache.get((canvasDims.toList, RotationalGroupView.this)).getOrElse({
      IndexedView.cache.synchronized {
        IndexedView.cache.getOrElseUpdate((canvasDims.toList, RotationalGroupView.this), layer)
      }
    }).addRef().asInstanceOf[Layer] else layer
  }

  /**
   * This is the tile expansion function.
   * It takes a point as input and returns null if there is an error.
   *
   * @docgenVersion 9
   */
  def tileExpansionFunction() = (point: Point) =>
    try {
      val angular1 = canvasToAngularCoords(point)
      val cartesian = angularToCartesianCoords(angular1)
      val tiled = tileTransform(cartesian)
      val angular2 = cartisianToAngularCoords(tiled)
      val result = angularToCanvasCoords(angular2)
      result
    } catch {
      case e: Throwable => null
    }

  /**
   * Returns true if the 'other' object is an instance of RotationalGroupView.
   *
   * @docgenVersion 9
   */
  override def canEqual(other: Any): Boolean = other.isInstanceOf[RotationalGroupView]

  /**
   * Returns true if the given object is a RotationalGroupView with the same x, y, and mode values as this object.
   *
   * @docgenVersion 9
   */
  override def equals(other: Any): Boolean = other match {
    case that: RotationalGroupView =>
      (that canEqual this) &&
        x == x &&
        y == y &&
        mode == mode
    case _ => false
  }

  /**
   * Returns the hash code for this object.
   *
   * The hash code is calculated as the sum of the hash codes of the object's
   * fields, modulo 2^32.
   *
   *
   * @docgenVersion 9
   */
  override def hashCode(): Int = {
    val state = Seq(x, y, mode)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}
