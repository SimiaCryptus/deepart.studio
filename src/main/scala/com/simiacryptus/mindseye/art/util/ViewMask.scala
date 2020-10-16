package com.simiacryptus.mindseye.art.util

case class ViewMask
(
  radius_min: Double = 0,
  radius_max: Double = Double.PositiveInfinity,
  x_min: Double = 0,
  x_max: Double = Double.PositiveInfinity,
  y_min: Double = 0,
  y_max: Double = Double.PositiveInfinity
)
