package com.simiacryptus.mindseye.art.util

import com.fasterxml.jackson.annotation.JsonIgnore
import com.simiacryptus.mindseye.art.VisionPipelineLayer
import com.simiacryptus.mindseye.art.models.{VGG16, VGG19}

class VisionLayerListJson(
                          val layers: Array[String]
                        ) {
  def this() = {
    this(Array.empty[String])
  }

  def this(layerA: VisionPipelineLayer, layerList: VisionPipelineLayer*) = {
    this((List(layerA)++layerList).map(_.name()).toArray)
  }

  @JsonIgnore
  def getLayerList(): List[VisionPipelineLayer] = {
    layers.map(getLayer).toList
  }

  def getLayer(layer:String): VisionPipelineLayer = {
    List(
      VGG16.VGG16_1a,
      VGG16.VGG16_1b1,
      VGG16.VGG16_1b2,
      VGG16.VGG16_1c1,
      VGG16.VGG16_1c2,
      VGG16.VGG16_1c3,
      VGG16.VGG16_1d1,
      VGG16.VGG16_1d2,
      VGG16.VGG16_1d3,
      VGG16.VGG16_1e1,
      VGG16.VGG16_1e2,
      VGG16.VGG16_1e3,
      VGG16.VGG16_2,
      VGG19.VGG19_1a,
      VGG19.VGG19_1b1,
      VGG19.VGG19_1b2,
      VGG19.VGG19_1c1,
      VGG19.VGG19_1c2,
      VGG19.VGG19_1c3,
      VGG19.VGG19_1c4,
      VGG19.VGG19_1d1,
      VGG19.VGG19_1d2,
      VGG19.VGG19_1d3,
      VGG19.VGG19_1d4,
      VGG19.VGG19_1e1,
      VGG19.VGG19_1e2,
      VGG19.VGG19_1e3,
      VGG19.VGG19_1e4,
      VGG19.VGG19_2
    ).find(l => layer == l.name()).getOrElse(throw new IllegalArgumentException(layer))
  }
}
