package com.simiacryptus.mindseye.art.util

case class ImageOptimizerJson
(
  override val trainingMinutes: Int,
  override val trainingIterations: Int,
  override val maxRate: Double
) extends ImageOptimizer
