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

import com.simiacryptus.mindseye.art.SumTrainable
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.lang.{Layer, PointSample, Tensor}
import com.simiacryptus.mindseye.opt.TrainingMonitor
import com.simiacryptus.ref.lang.ReferenceCountingBase

trait VisualNetwork {
  def precision: Precision

  def apply(canvas: Tensor, content: Tensor): Trainable

  /**
   * Adds the specified value to this VisualNetwork.
   *
   * @param value the value to add
   * @return the VisualNetwork with the added value
   * @docgenVersion 9
   */
  def +(value: VisualNetwork): VisualNetwork = {
    val inner = this
    new VisualNetwork {
      require(inner.precision == value.precision)

      override def precision: Precision = inner.precision

      /**
       * This function overrides the apply function in order to create a new SumTrainable.
       * The new SumTrainable consists of the inner apply function of the canvas and content,
       * as well as the value function of the canvas and content.
       *
       * @docgenVersion 9
       */
      override def apply(canvas: Tensor, content: Tensor): Trainable = new SumTrainable(
        inner.apply(canvas.addRef(), content.addRef()),
        value.apply(canvas, content)
      )
    }
  }


  /**
   * Multiplies this VisualNetwork by a scalar value.
   *
   * @param value the value to multiply by
   * @return the new VisualNetwork
   * @docgenVersion 9
   */
  def *(value: Double): VisualNetwork = {
    val inner = this
    new VisualNetwork {
      override def precision: Precision = inner.precision

      /**
       * This is a reference counting base with a trainable. The trainable is inner and is applied to the canvas and content.
       * This returns a PointSample.
       *
       * @docgenVersion 9
       */
      override def apply(canvas: Tensor, content: Tensor): Trainable = new ReferenceCountingBase with Trainable {
        lazy val innerTrainable = inner.apply(canvas, content)


        /**
         * Adds a reference to this Trainable and returns it.
         *
         * @docgenVersion 9
         */
        override def addRef(): Trainable = super[ReferenceCountingBase].addRef().asInstanceOf[Trainable]

        /**
         * Overrides the measure method to return a PointSample.
         *
         * @param monitor The TrainingMonitor to use.
         * @return A PointSample.
         * @docgenVersion 9
         */
        override def measure(monitor: TrainingMonitor): PointSample = {
          val pointSample = innerTrainable.measure(monitor)
          val scaled = new PointSample(
            pointSample.delta.scale(value),
            pointSample.weights.addRef(),
            pointSample.sum * value,
            pointSample.rate,
            pointSample.count
          )
          pointSample.freeRef()
          scaled
        }

        override def getLayer: Layer = {
          innerTrainable.getLayer().addRef().asInstanceOf[Layer]
        }
      }
    }
  }
}
