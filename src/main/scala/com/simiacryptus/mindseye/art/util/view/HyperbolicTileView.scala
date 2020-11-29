package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.{HyperbolicPolygon, HyperbolicTiling, Point, Raster}

case class HyperbolicTileView(p: Int, q: Int, i: Int = 3, maxRadius: Double = 0.9, klien: Boolean = false) extends IndexedView {
  def mappingFunction(): Point => Point = {
    val polygon = HyperbolicPolygon.regularPolygon(p, q)
    val tiling = new HyperbolicTiling(polygon).expand(i)
    val transform = if (klien) {
      tiling.klien()(_)
    } else tiling.transform _
    (point: Point) => {
      if (point.rms() > maxRadius) null else transform(point)
    }
  }
}
