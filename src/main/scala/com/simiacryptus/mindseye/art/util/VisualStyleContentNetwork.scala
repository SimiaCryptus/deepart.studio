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
import com.simiacryptus.mindseye.art.ops.ContentMatcher
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.{MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.{AssertDimensionsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.ref.lang.RefUtil

object VisualStyleContentNetwork {

}


case class VisualStyleContentNetwork
(
  styleLayers: Seq[VisionPipelineLayer] = Seq.empty,
  styleModifiers: Seq[VisualModifier] = Seq.empty,
  contentLayers: Seq[VisionPipelineLayer] = Seq.empty,
  contentModifiers: Seq[VisualModifier] = List(new ContentMatcher),
  styleUrl: Seq[String] = Seq.empty,
  styleUrls: Seq[String] = Seq.empty,
  precision: Precision = Precision.Float,
  viewLayer: Seq[Int] => Layer = _ => new PipelineNetwork(1),
  override val tileSize: Int = 1200,
  override val tilePadding: Int = 64,
  override val minWidth: Int = 1,
  override val maxWidth: Int = 2048,
  override val maxPixels: Double = 5e7,
  override val magnification: Double = 1.0
)(implicit override val log: NotebookOutput) extends ImageSource(styleUrl, styleUrls) with VisualNetwork {

  def apply(canvas: Tensor, content: Tensor): Trainable = {
    val loadedImages = loadImages(VisualStyleNetwork.pixels(canvas.addRef()))
    val styleModifier = styleModifiers.reduceOption(_ combine _).getOrElse(new VisualModifier {
      override def build(visualModifierParameters: VisualModifierParameters): PipelineNetwork = {
        visualModifierParameters.freeRef()
        new PipelineNetwork(1)
      }
    })
    val contentModifier = contentModifiers.reduce(_ combine _)
    if (styleModifier.isLocalized()) {
      trainable_tiledStyle(canvas, content, loadedImages, styleModifier, contentModifier)
    } else {
      trainable_sharedStyle(canvas, content, loadedImages, styleModifier, contentModifier)
    }
  }

  def trainable_tiledStyle(canvas: Tensor, content: Tensor, loadedImages: Array[Tensor], styleModifier: VisualModifier, contentModifier: VisualModifier) = {
    require(!loadedImages.isEmpty)
    val grouped: Array[String] = ((contentLayers.map(_.getPipelineName -> null) ++ styleLayers.groupBy(_.getPipelineName).toList).map(_._1).distinct).toArray
    val contentDims = content.getDimensions()
    val canvasDims = canvas.getDimensions
    if (contentDims.toList != canvasDims.toList) {
      val msg = s"""${contentDims.toList} != ${canvasDims.toList}"""
      throw new IllegalArgumentException(msg)
    }
    val resView = viewLayer(contentDims)
    val contentView: Tensor = if (prefilterContent) {
      val result = resView.eval(content)
      val data = result.getData
      val tensor = data.get(0)
      data.freeRef()
      result.freeRef()
      tensor
    } else content
    val trainable = new SumTrainable(grouped.map(name => {
      new TileTrainer(canvas.addRef(), loadedImages.map(_.addRef()), contentDims, resView, contentView.addRef(), name, styleModifier, contentModifier)
    }).toArray: _*)
    contentView.freeRef()
    canvas.freeRef()
    loadedImages.foreach(_.freeRef())
    resView.freeRef()
    trainable
  }

  def trainable_sharedStyle(canvas: Tensor, content: Tensor, loadedImages: Array[Tensor], styleModifier: VisualModifier, contentModifier: VisualModifier) = {
    val contentDims = content.getDimensions()
    val canvasDims = canvas.getDimensions
    val grouped: Map[String, PipelineNetwork] = ((contentLayers.map(_.getPipelineName -> null) ++ styleLayers.groupBy(_.getPipelineName).toList).groupBy(_._1).mapValues(pipelineLayers => {
      val layers: Seq[VisionPipelineLayer] = pipelineLayers.flatMap(x => Option(x._2).toList.flatten)
      if (layers.isEmpty) null
      else SumInputsLayer.combine(layers.map(styleLayer => {
        val network = styleModifier.build(styleLayer.addRef(), contentDims, null, RefUtil.addRef(loadedImages): _*)
        val layer = new AssertDimensionsLayer(1)
        layer.setName(s"$styleModifier - $styleLayer")
        network.add(layer).freeRef()
        network
      }): _*)
    })).toArray.map(identity).toMap
    loadedImages.foreach(_.freeRef())
    if (contentDims.toList != canvasDims.toList) {
      val msg = s"""${contentDims.toList} != ${canvasDims.toList}"""
      throw new IllegalArgumentException(msg)
    }
    val resView = viewLayer(contentDims)
    val contentView = if (prefilterContent) {
      val result = resView.eval(content)
      val data = result.getData
      val tensor = data.get(0)
      data.freeRef()
      result.freeRef()
      tensor
    } else content
    val sumTrainable = new SumTrainable(grouped.map(t => {
      val (name, styleNetwork) = t
      new TiledTrainable(canvas.addRef(), resView, tileSize, tilePadding, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          val result = regionSelector.eval(contentView.addRef())
          val data = result.getData
          result.freeRef()
          val selection = data.get(0)
          data.freeRef()
          val network = SumInputsLayer.combine({
            Option(styleNetwork).map(_.addRef()).toList ++ contentLayers.filter(x => x != null && x.getPipelineName == name)
              .map(contentLayer => {
                val name = s"$contentModifier - $contentLayer"
                val network = contentModifier.build(contentLayer.addRef(), contentDims, regionSelector.asTensorFunction(), selection.addRef())
                val layer = new AssertDimensionsLayer(1)
                layer.setName(name)
                network.add(layer).freeRef()
                network
              })
          }: _*)
          selection.freeRef()
          MultiPrecision.setPrecision(network.addRef(), precision)
          regionSelector.freeRef()
          network
        }

        override protected def _free(): Unit = {
          if (null != styleNetwork) styleNetwork.freeRef()
          super._free()
        }
      }
    }).toArray: _*)
    contentView.freeRef()
    canvas.freeRef()
    sumTrainable
  }

  def prefilterContent = false

  class TileTrainer
  (
    canvas: Tensor,
    loadedImages: Array[Tensor],
    contentDims: Array[Int],
    resView: Layer,
    contentView: Tensor,
    name: String,
    styleModifier: VisualModifier,
    contentModifier: VisualModifier
  ) extends TiledTrainable(canvas, resView, tileSize, tilePadding, precision) {
    override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
      try {
        val regionFn = regionSelector.asTensorFunction()
        val result = regionSelector.eval(contentView)
        val data = result.getData
        val contentRegion = data.get(0)
        data.freeRef()
        result.freeRef()
        val network = SumInputsLayer.combine({
          contentLayers.filter(x => x.getPipelineName == name).map(contentLayer => {
            contentModifier.build(contentLayer.addRef(), contentDims, regionFn, contentRegion)
          }) ++ styleLayers.filter(x => x.getPipelineName == name).map(styleLayer => {
            styleModifier.build(styleLayer.addRef(), contentDims, regionFn, loadedImages: _*)
          })
        }: _*)
        MultiPrecision.setPrecision(network, precision)
        network
      } finally {
        regionSelector.freeRef()
      }
    }

    override protected def _free(): Unit = {
      loadedImages.foreach(_.freeRef())
      super._free()
    }
  }

}
