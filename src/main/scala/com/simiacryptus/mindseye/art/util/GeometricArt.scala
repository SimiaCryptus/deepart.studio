package com.simiacryptus.mindseye.art.util

import com.simiacryptus.mindseye.lang.Layer
import com.simiacryptus.mindseye.layers.java.{BoundedActivationLayer, AffineImgViewLayer, LinearActivationLayer, SumInputsLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork

trait GeometricArt {

  /**
   * Calculates the average of the given layers.
   *
   * @param views The layers to average.
   * @tparam T The type of layer.
   * @return The average layer.
   * @docgenVersion 9
   */
  def avg[T <: Layer](views: Array[Option[T]]): PipelineNetwork = {
    val network = new PipelineNetwork(1)

    def in = network.getInput(0)

    network.add(new SumInputsLayer(), views.map(_.map(network.add(_, in)).getOrElse(in)): _*).freeRef()
    val layer = new LinearActivationLayer()
    layer.setScale(1.0 / views.length)
    layer.freeze()
    network.add(layer).freeRef()
    val boundedActivationLayer = new BoundedActivationLayer()
    boundedActivationLayer.setMinValue(0)
    boundedActivationLayer.setMaxValue(255)
    boundedActivationLayer.freeze()
    network.add(boundedActivationLayer).freeRef()
    network
  }

  /**
   * Folds the symmetries in the given views.
   *
   * @param views The views to fold.
   * @tparam T The type of layer.
   * @return The folded pipeline network.
   * @docgenVersion 9
   */
  def foldSymmetries[T <: Layer](views: Array[T]): PipelineNetwork = {
    PipelineNetwork.build(1, views.map(l => avg(Array(None, Option(l)))): _*)
  }

  /**
   * Returns a rotor layer given a radian value and canvas dimensions.
   *
   * @param radians    the radian value to use for the rotor layer
   * @param canvasDims the dimensions of the canvas to use for the layer
   * @return a new rotor layer
   * @docgenVersion 9
   */
  def getRotor(radians: Double, canvasDims: Array[Int]) = {
    val layer = new AffineImgViewLayer(canvasDims(0), canvasDims(1), true)
    layer.setRotationCenterX(canvasDims(0) / 2)
    layer.setRotationCenterY(canvasDims(1) / 2)
    layer.setRotationRadians(radians)
    layer
  }

}
