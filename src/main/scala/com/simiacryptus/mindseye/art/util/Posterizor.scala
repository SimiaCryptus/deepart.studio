package com.simiacryptus.mindseye.art.util

import com.simiacryptus.mindseye.art.photo.affinity.{AffinityWrapper, ContextAffinity, RasterAffinity, RelativeAffinity}
import com.simiacryptus.mindseye.art.photo.cuda.SmoothSolver_Cuda
import com.simiacryptus.mindseye.art.photo.topology.{ContentTopology, SearchRadiusTopology}
import com.simiacryptus.mindseye.art.photo.{SmoothSolver, SmoothSolver_EJML}
import com.simiacryptus.mindseye.lang.Tensor

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

  def apply(source: Tensor, target: Tensor) = {
    val topology = getTopology(source.addRef())
    val affinity = getAffinity(source, topology.addRef())
    val operator = solver.solve(topology, affinity, lambda)
    val recoloredTensor = operator.apply(target)
    operator.freeRef()
    recoloredTensor
  }

  def getAffinity(source: Tensor, topology: ContentTopology): RasterAffinity = {
    val affinity = new RelativeAffinity(source, topology)
    affinity.setContrast(contrast)
    affinity.setGraphPower1(2)
    affinity.setMixing(mixing)
    affinity.wrap((graphEdges, innerResult) => RasterAffinity.adjust(graphEdges, innerResult, RasterAffinity.degree(innerResult), 0.5))
  }

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
