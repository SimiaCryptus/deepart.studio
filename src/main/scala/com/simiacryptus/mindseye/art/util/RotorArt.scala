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

import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.InteractiveSetup

/**
 * An abstract class for a RotorArt object. The class takes in an integer for the number of rotational segments and extends the ArtSetup and GeometricArt traits.
 *
 * @docgenVersion 9
 */
abstract class RotorArt[U <: AnyRef with InteractiveSetup[Object, U]](
                                           val rotationalSegments: Int = 3
                                         ) extends ArtSetup[Object, U] with GeometricArt {

  val rotationalChannelPermutation: Array[Int] = Permutation.random(3, rotationalSegments)

  /**
   * Overrides the reference method to do nothing.
   *
   * @param log The notebook output.
   * @docgenVersion 9
   */
  override def reference(log: NotebookOutput): Unit = {
    val l2 = Permutation.rings(3).toList.sortBy(_._1).map(t => {
      val permutations = t._2.map(x => x.head).toList.sortBy(_.indices.sum).map(x => x.indices.map(_.toString).reduce(_ + "," + _))
      <li>
        Order:
        {t._1}<br/>
        <ul>
          {permutations.map(p => {
          <li>
            {p}
          </li>
        }).toList}
        </ul>
      </li>
    })
    log.out(<div>
      <h2>Color Permutations:</h2> <ol>
        {l2}
      </ol> <h2>Tiling Aspect Ratios:</h2> <ol>
        <li>Triangular or Hexagonal: 1.732 or 0.5774</li>
        <li>Square: 1.0</li>
      </ol>
    </div>.toString())
  }

  /**
   * Returns a kaleidoscope image given the dimensions of the canvas.
   *
   * @docgenVersion 9
   */
  def getKaleidoscope(canvasDims: Array[Int]) = {
    val permutation = Permutation(this.rotationalChannelPermutation: _*)
    require(permutation.unity == (permutation ^ rotationalSegments), s"$permutation ^ $rotationalSegments => ${(permutation ^ rotationalSegments)} != ${permutation.unity}")
    List(avg((0 until rotationalSegments).toArray.map(segment => {
      if (0 == segment) None else {
        val layer = getRotor(segment * 2 * Math.PI / rotationalSegments, canvasDims)
        layer.setChannelSelector((permutation ^ segment).indices)
        Option(layer)
      }
    }))
    )
  }

}


