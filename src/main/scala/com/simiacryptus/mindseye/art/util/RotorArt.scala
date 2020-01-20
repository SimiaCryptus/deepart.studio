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

import com.simiacryptus.mindseye.layers.java.{BoundedActivationLayer, ImgViewLayer, LinearActivationLayer, SumInputsLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork

abstract class RotorArt
(
  val rotationalSegments: Int = 3
) extends ArtSetup[Object] {

  val rotationalChannelPermutation: Array[Int] = Permutation.random(3, rotationalSegments)

  def getKaleidoscope(canvasDims: Array[Int]) = {
    val permutation = Permutation(this.rotationalChannelPermutation: _*)
    require(permutation.unity == (permutation ^ rotationalSegments), s"$permutation ^ $rotationalSegments => ${(permutation ^ rotationalSegments)} != ${permutation.unity}")
    val network = new PipelineNetwork(1)
    network.add(new SumInputsLayer(), (0 until rotationalSegments)
      .map(segment => {
        if (0 == segment) network.getInput(0) else {
          val layer = getRotor(segment * 2 * Math.PI / rotationalSegments, canvasDims)
          layer.setChannelSelector((permutation ^ segment).indices)
          network.add(
            layer,
            network.getInput(0)
          )
        }
      }): _*).freeRef()
    val layer = new LinearActivationLayer()
    layer.setScale(1.0 / rotationalSegments)
    layer.freeze()
    network.add(layer).freeRef()
    val boundedActivationLayer = new BoundedActivationLayer()
    boundedActivationLayer.setMinValue(0)
    boundedActivationLayer.setMaxValue(255)
    boundedActivationLayer.freeze()
    network.add(boundedActivationLayer).freeRef()
    network
  }

  def getRotor(radians: Double, canvasDims: Array[Int]) = {
    val layer = new ImgViewLayer(canvasDims(0), canvasDims(1), true)
    layer.setRotationCenterX(canvasDims(0) / 2)
    layer.setRotationCenterY(canvasDims(1) / 2)
    layer.setRotationRadians(radians)
    layer
  }

}
