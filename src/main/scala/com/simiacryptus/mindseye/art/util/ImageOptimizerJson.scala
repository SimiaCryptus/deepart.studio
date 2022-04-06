package com.simiacryptus.mindseye.art.util

/**
 * This is the ImageOptimizerJson class, which inherits from the ImageOptimizer class.
 * It contains the following fields:
 * - trainingMinutes: the number of minutes of training
 * - trainingIterations: the number of training iterations
 * - maxRate: the maximum learning rate*
 *
 * @docgenVersion 9
 */
case class ImageOptimizerJson
(
  override val trainingMinutes: Int,
  override val trainingIterations: Int,
  override val maxRate: Double
) extends ImageOptimizer
