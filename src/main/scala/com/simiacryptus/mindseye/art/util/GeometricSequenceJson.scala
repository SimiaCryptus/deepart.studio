package com.simiacryptus.mindseye.art.util

/**
 * This is a GeometricSequenceJson case class that takes in a minimum value, maximum value, and number of steps.
 * It extends the GeometricSequence class.
 *
 * @docgenVersion 9
 */
case class GeometricSequenceJson
(
  min: Double,
  max: Double,
  steps: Int
) extends GeometricSequence
