package com.simiacryptus.mindseye.art.util

/**
 * A view mask is used to specify a particular region of interest in an image.
 * The mask is defined by a minimum and maximum radius, and minimum and maximum x- and y-coordinates.
 * By default, the mask covers the entire image.
 *
 * @docgenVersion 9
 */
case class ViewMask
(
  radius_min: Double = 0,
  radius_max: Double = Double.PositiveInfinity,
  x_min: Double = 0,
  x_max: Double = Double.PositiveInfinity,
  y_min: Double = 0,
  y_max: Double = Double.PositiveInfinity
)
