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

  def pixels(canvas: Tensor) = {
    if (null == canvas) 0 else {
      val dimensions = canvas.getDimensions
      canvas.freeRef()
      val pixels = dimensions(0) * dimensions(1)
      pixels
    }
  }

}

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

  def apply(canvas: Tensor, content: Tensor): Trainable = {
    val dimensions = content.getDimensions
    content.freeRef()
    apply(canvas, dimensions)
  }

  def apply(canvas: Tensor, dimensions: Array[Int]) = {
    val loadedImages = loadImages(dimensions.reduce(_*_))
    try {
      val contentDimensions = dimensions
      new SumTrainable((for (
        pipelineLayers <- styleLayers.groupBy(_.getPipelineName).values
      ) yield {
        var styleNetwork: PipelineNetwork = null
        if(!filterStyleInput) styleNetwork = SumInputsLayer.combine(pipelineLayers.map(pipelineLayer => {
          val network = styleModifiers.reduce(_ combine _).build(pipelineLayer, contentDimensions, (x:Tensor)=>x, RefUtil.addRef(loadedImages): _*)
          network.freeze()
          network
        }): _*)
        for(layer <- viewLayer(contentDimensions)) yield {
          if(filterStyleInput) styleNetwork = SumInputsLayer.combine(pipelineLayers.map(pipelineLayer => {
            val network = styleModifiers.reduce(_ combine _).build(pipelineLayer, contentDimensions, layer.asTensorFunction(), RefUtil.addRef(loadedImages): _*)
            network.freeze()
            network
          }): _*)
          new TiledTrainable(canvas.addRef(), layer, tileSize, tilePadding, precision) {

            override def getLayer(): Layer = {
              styleNetwork.addRef()
            }

            override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
              regionSelector.freeRef()
              val network = styleNetwork.addRef()
              MultiPrecision.setPrecision(network.addRef(), precision)
              network
            }

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
