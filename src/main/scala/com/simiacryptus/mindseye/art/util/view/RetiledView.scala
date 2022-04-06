package com.simiacryptus.mindseye.art.util.view

import com.simiacryptus.math.Point

/**
 * A RetiledView is an IndexedView with a default theta of Math.PI / 2.
 *
 * @docgenVersion 9
 */
case class RetiledView(theta: Double = Math.PI / 2) extends IndexedView {
  override def filterCircle = false

  /**
   * Defines a mapping function that transforms a Point into another Point.
   *
   * @docgenVersion 9
   */
  def mappingFunction(): Point => Point = {
    (point: Point) => {
      try {
        val sin = Math.sin(theta)
        val cos = Math.cos(theta)
        if (point.x < 0) {
          if (point.y < 0) {
            val y = point.y + 0.5
            val x = point.x + 0.5
            new Point(
              (cos * x - sin * y) - 0.5,
              (sin * x + cos * y) - 0.5
            )
          } else {
            val y = point.y - 0.5
            val x = point.x + 0.5
            new Point(
              (cos * x - sin * y) - 0.5,
              (sin * x + cos * y) + 0.5
            )
          }
        } else {
          if (point.y < 0) {
            val y = point.y + 0.5
            val x = point.x - 0.5
            new Point(
              (cos * x - sin * y) + 0.5,
              (sin * x + cos * y) - 0.5
            )
          } else {
            val y = point.y - 0.5
            val x = point.x - 0.5
            new Point(
              (cos * x - sin * y) + 0.5,
              (sin * x + cos * y) + 0.5
            )
          }
        }
      } catch {
        case e: Throwable => null
      }
    }
  }

}
