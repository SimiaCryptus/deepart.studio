package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.{HyperbolicPolygon, HyperbolicTiling, Point, Raster}

case class HyperbolicTileView(p: Int, q: Int, i: Int = 3, maxRadius: Double = 0.9, mode: String = "poincare") extends IndexedView {
  override def filterCircle = mode match {
    case "square" => false
    case _ => true
  }

  def mappingFunction(): Point => Point = {
    val polygon = HyperbolicPolygon.regularPolygon(p, q)
    val tiling = new HyperbolicTiling(polygon).expand(i)
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
