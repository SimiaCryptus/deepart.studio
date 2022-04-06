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

import org.apache.commons.math3.linear._

import scala.util.Random

object Permutation {
  /**
   * This function prints out the rings of a graph with 3 nodes.
   * The rings are sorted by the number of edges in each ring.
   *
   * @docgenVersion 9
   */
  def main(args: Array[String]): Unit = {
    rings(3).values.flatten.toList.sortBy(_.size).foreach(ring => {
      println(ring.map(_.toString).mkString(" -> "))
    })
  }

  /**
   * Returns a map of ring sizes to unique rings,
   * where each ring is represented as a set of
   * indices into the original permutation.
   *
   * @docgenVersion 9
   */
  def rings(rank: Int) = permutations(rank).map(_.ring).groupBy(_.size).mapValues(
    _.sortBy(_.map(_.indices.mkString(",")).mkString(";"))
      .groupBy(_.map(_.indices.mkString(",")).sorted.mkString(";"))
      .values.map(_.head).toSet)

  /**
   * Returns a stream of all permutations of the integers from 1 to rank,
   * where each permutation is represented as a list.
   *
   * @docgenVersion 9
   */
  def permutations(rank: Int) = {
    (1 to rank).toStream.map(i => Stream(List(-i), List(i))).reduce((xs, ys) => {
      for {x <- xs; y <- ys} yield (x ++ y)
    }).flatMap(_.permutations).map(Permutation(_: _*))
  }

  /**
   * Returns a list of roots for a given rank and power.
   * The roots are shuffled so that the order is random.
   *
   * @docgenVersion 9
   */
  def roots_whole(rank: Int, power: Int) = Random.shuffle(rings(rank)(power).flatten.toList)

  /**
   * Returns a list of roots that are a factor of the given power,
   * for the given rank. The roots are shuffled before being returned.
   *
   * @docgenVersion 9
   */
  def roots_factor(rank: Int, power: Int) = Random.shuffle(rings(rank).filter(t => power % t._1 == 0).map(_._2).flatten.flatten.toList)

  /**
   * Returns a random number between 0 and the given rank, to the given power.
   *
   * @docgenVersion 9
   */
  def random(rank: Int, power: Int) = roots_whole(rank, power).headOption.getOrElse(roots_factor(rank, power).head).indices

  /**
   * Returns the permutation of unity for a given number of elements.
   *
   * @param n the number of elements
   * @return the permutation of unity
   * @docgenVersion 9
   */
  def unity(n: Int) = Permutation((1 to n).toArray: _*)

  /**
   * Creates a new permutation with the given indices.
   *
   * @docgenVersion 9
   */
  def apply(indices: Int*) = new Permutation(indices.toArray)
}

/**
 * Class Permutation
 *
 * @param indices the array of indices
 * @docgenVersion 9
 */
class Permutation(val indices: Array[Int]) {
  require(indices.distinct.size == indices.size)
  require(indices.map(_.abs).distinct.size == indices.size)
  require(indices.map(_.abs).min == 1)
  require(indices.map(_.abs).max == indices.size)

  /**
   * Returns the nth power of this permutation.
   *
   * @docgenVersion 9
   */
  def ^(n: Int): Permutation = Stream.iterate(unity)(this * _)(n)

  def unity = Permutation.unity(rank)

  def rank: Int = indices.length

  /**
   * Multiplies this permutation with another permutation on the right.
   *
   * @param right the permutation to multiply with
   * @return the product permutation
   * @docgenVersion 9
   */
  def *(right: Permutation): Permutation = Permutation(this * right.indices: _*)

  /**
   * Multiplies this array by the given array, element-wise.
   * If an index in this array is negative, the corresponding element in the given array is negated.
   *
   * @docgenVersion 9
   */
  def *(right: Array[Int]) = {
    indices.map(idx => {
      if (idx < 0) {
        -right(-idx - 1)
      } else {
        right(idx - 1)
      }
    })
  }

  def matrix: RealMatrix = {
    val rank = this.rank
    val matrix = new Array2DRowRealMatrix(3, 3)
    val tuples = indices.zipWithIndex.map(t => (t._1.abs - 1, t._2, t._1.signum))
    for ((x, y, v) <- tuples) matrix.setEntry(x, y, v)
    matrix
  }

  def ring = {
    List(this) ++ Stream.iterate(this)(_ * this).drop(1).takeWhile(_ != this)
  }

  override def toString: String = "[" + indices.mkString(",") + "]"

  /**
   * This method overrides the equals method in the Any class.
   * It returns true if the hashCode of the object being compared is the same as the hashCode of this object,
   * and if the indices of the object being compared are the same as the indices of this object.
   * Otherwise, it returns false.
   *
   * @docgenVersion 9
   */
  override def equals(obj: scala.Any): Boolean = obj match {
    case obj: Permutation if (obj.hashCode() == this.hashCode()) => indices.toList.equals(obj.indices.toList)
    case _ => false
  }

  /**
   * Overrides the default hashCode method to use the hashCode of the list of indices
   *
   * @docgenVersion 9
   */
  override def hashCode(): Int = indices.toList.hashCode()
}
