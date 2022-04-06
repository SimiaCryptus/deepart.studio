package com.simiacryptus.mindseye.art.util

import com.simiacryptus.mindseye.art.photo.affinity.{AffinityWrapper, ContextAffinity, RasterAffinity, RelativeAffinity}
import com.simiacryptus.mindseye.art.photo.cuda.SmoothSolver_Cuda
import com.simiacryptus.mindseye.art.photo.topology.{ContentTopology, SearchRadiusTopology}
import com.simiacryptus.mindseye.art.photo.{SmoothSolver, SmoothSolver_EJML}
import com.simiacryptus.mindseye.lang.Tensor

/**
 * Posterizor is a class that takes in parameters for lambda, contrast, mixing, selfRef, spatialPriority, neighborhoodSize, initialRadius, and useCuda.
 * The default values for these parameters are 0.00003d, 80, 0.001d, true, true, 3, 2, and true respectively.
 *
 * @docgenVersion 9
 */
case class Posterizor
(
  lambda: Double = 3e-5,
  contrast: Int = 80,
  mixing: Double = 0.001,
  selfRef: Boolean = true,
  spatialPriority: Boolean = true,
  neighborhoodSize: Int = 3,
  initialRadius: Int = 2,
  useCuda: Boolean = true
) {

  def solver: SmoothSolver = if (useCuda) new SmoothSolver_Cuda() else new SmoothSolver_EJML()

  /**
   * Applies the given source tensor to the target tensor using the given topology and affinity.
   *
   * @param source   the source tensor
   * @param target   the target tensor
   * @param topology the topology to use
   * @param affinity the affinity to use
   * @return the recolored tensor
   * @docgenVersion 9
   */
  def apply(source: Tensor, target: Tensor) = {
    val topology = getTopology(source.addRef())
    val affinity = getAffinity(source, topology.addRef())
    val operator = solver.solve(topology, affinity, lambda)
    val recoloredTensor = operator.apply(target)
    operator.freeRef()
    recoloredTensor
  }

  /**
   * Returns the affinity of the source tensor with the given topology.
   *
   * @docgenVersion 9
   */
  def getAffinity(source: Tensor, topology: ContentTopology): RasterAffinity = {
    val affinity = new RelativeAffinity(source, topology)
    affinity.setContrast(contrast)
    affinity.setGraphPower1(2)
    affinity.setMixing(mixing)
    affinity.wrap((graphEdges, innerResult) => RasterAffinity.adjust(graphEdges, innerResult, RasterAffinity.degree(innerResult), 0.5))
  }

  /**
   * Returns a ContentTopology object, which is used to find the
   * topology of the content in a given source Tensor.
   *
   * @param source the source Tensor to find the topology of
   * @return a ContentTopology object
   * @docgenVersion 9
   */
  def getTopology(source: Tensor): ContentTopology = {
    val topology = new SearchRadiusTopology(source)
    topology.setSelfRef(selfRef)
    topology.setVerbose(true)
    topology.setSpatialPriority(spatialPriority)
    topology.setNeighborhoodSize(neighborhoodSize)
    topology.setInitialRadius(initialRadius)
    topology
  }

}
