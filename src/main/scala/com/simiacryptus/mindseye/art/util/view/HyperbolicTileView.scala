package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.{HyperbolicPolygon, HyperbolicTiling, Point, Raster}

/**
 * A hyperbolic tile view with parameters p, q, i, maxRadius, and mode.
 *
 * @docgenVersion 9
 */
case class HyperbolicTileView(p: Int, q: Int, i: Int = 3, maxRadius: Double = 0.9, mode: String = "poincare") extends IndexedView {
  override def filterCircle = mode match {
    case "square" => false
    case _ => true
  }

  val polygon = HyperbolicPolygon.regularPolygon(p, q)
  val tiling = new HyperbolicTiling(polygon).expand(i)

  /**
   * Returns a function that maps points to other points, based on the mode.
   * If the mode is "square", the function will return null for points that are too large.
   * Otherwise, the function will return null for points with an RMS value that is too large.
   *
   * @docgenVersion 9
   */
  def mappingFunction(): Point => Point = {
    val transform = mode match {
      case "klien" => tiling.klien()(_)
      case "square" => tiling.square()(_)
      case "poincare" => tiling.transform(_)
    }
    mode match {
      case "square" => (point: Point) => if (Math.max(Math.abs(point.x), Math.abs(point.y)) > maxRadius) null else tiling.invsquare(transform(point))
      case _ => (point: Point) => if (point.rms() > maxRadius) null else transform(point)
    }
  }
}
