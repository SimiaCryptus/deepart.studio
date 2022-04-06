/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.util

import com.simiacryptus.mindseye.art._
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.{MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.ref.lang.RefUtil

object VisualStyleNetwork {

  /**
   * Returns the number of pixels in the given canvas.
   *
   * @param canvas the canvas to check
   * @return the number of pixels in the canvas
   * @docgenVersion 9
   */
  def pixels(canvas: Tensor) = {
    if (null == canvas) 0 else {
      val dimensions = canvas.getDimensions
      canvas.freeRef()
      val pixels = dimensions(0) * dimensions(1)
      pixels
    }
  }

}

/**
 * This is a case class for a VisualStyleNetwork. It takes in parameters for styleLayers, styleModifiers, styleUrls, precision, viewLayer, filterStyleInput, tileSize, tilePadding, minWidth, maxWidth, maxPixels, magnification, and log.
 *
 * @docgenVersion 9
 */
case class VisualStyleNetwork
(
  styleLayers: Seq[VisionPipelineLayer] = Seq.empty,
  styleModifiers: Seq[VisualModifier] = Seq.empty,
  styleUrls: Seq[String] = Seq.empty,
  precision: Precision = Precision.Float,
  viewLayer: Seq[Int] => List[Layer] = _ => List(new PipelineNetwork(1)),
  filterStyleInput: Boolean = true,
  override val tileSize: Int = 1400,
  override val tilePadding: Int = 64,
  override val minWidth: Int = 1,
  override val maxWidth: Int = 10000,
  override val maxPixels: Double = 5e7,
  override val magnification: Seq[Double] = Array(1.0)
)(implicit override val log: NotebookOutput) extends ImageSource(styleUrls) with VisualNetwork {

  /**
   * This function applies the given content to the given canvas.
   *
   * @param canvas  The canvas to apply the content to.
   * @param content The content to apply to the canvas.
   * @docgenVersion 9
   */
  def apply(canvas: Tensor, content: Tensor): Trainable = {
    val dimensions = content.getDimensions
    content.freeRef()
    apply(canvas, dimensions)
  }

  /**
   * Applies the given canvas to the given dimensions.
   *
   * @docgenVersion 9
   */
  def apply(canvas: Tensor, dimensions: Array[Int]) = {
    val loadedImages = loadImages(dimensions.reduce(_ * _))
    try {
      val contentDimensions = dimensions
      new SumTrainable((for (
        pipelineLayers <- styleLayers.groupBy(_.getPipelineName).values
      ) yield {
        var styleNetwork: PipelineNetwork = null
        if (!filterStyleInput) styleNetwork = SumInputsLayer.combine(pipelineLayers.map(pipelineLayer => {
          val network = styleModifiers.reduce(_ combine _).build(pipelineLayer, contentDimensions, (x: Tensor) => x, RefUtil.addRef(loadedImages): _*)
          network.freeze()
          network
        }): _*)
        for (layer <- viewLayer(contentDimensions)) yield {
          if (filterStyleInput) styleNetwork = SumInputsLayer.combine(pipelineLayers.map(pipelineLayer => {
            val network = styleModifiers.reduce(_ combine _).build(pipelineLayer, contentDimensions, layer.asTensorFunction(), RefUtil.addRef(loadedImages): _*)
            network.freeze()
            network
          }): _*)
          new TiledTrainable(canvas.addRef(), layer, tileSize, tilePadding, precision) {

            /**
             * Returns the layer of the style network.
             *
             * @docgenVersion 9
             */
            override def getLayer(): Layer = {
              styleNetwork.addRef()
            }

            /**
             * Returns the network for the given region selector.
             *
             * @param regionSelector The region selector layer.
             * @return The network for the given region selector.
             * @docgenVersion 9
             */
            override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
              regionSelector.freeRef()
              val network = styleNetwork.addRef()
              MultiPrecision.setPrecision(network.addRef(), precision)
              network
            }

            /**
             * Frees the resources used by this object.
             *
             * @docgenVersion 9
             */
            override def _free(): Unit = {
              styleNetwork.freeRef()
              super._free()
            }
          }
        }
      }).flatten.toArray: _*)
    } finally {
      RefUtil.freeRef(loadedImages)
      canvas.freeRef()
    }
  }

  /**
   * This function defines a content network for visual style transfer.
   *
   * @param contentLayers    A sequence of vision pipeline layers that represent the content of the image.
   * @param contentModifiers A sequence of visual modifiers that are applied to the content layers.
   * @return A visual style content network.
   * @docgenVersion 9
   */
  def withContent(
                   contentLayers: Seq[VisionPipelineLayer],
                   contentModifiers: Seq[VisualModifier] = List(new ContentMatcher)
                 ) = VisualStyleContentNetwork(
    styleLayers = styleLayers,
    styleModifiers = styleModifiers,
    styleUrls = styleUrls,
    precision = precision,
    viewLayer = viewLayer,
    contentLayers = contentLayers,
    contentModifiers = contentModifiers,
    tilePadding = tilePadding,
    tileSize = tileSize,
    minWidth = minWidth,
    maxWidth = maxWidth,
    maxPixels = maxPixels,
    magnification = magnification
  )
}
