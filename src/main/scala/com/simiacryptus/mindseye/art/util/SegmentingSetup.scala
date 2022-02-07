/*
 * Copyright (c) 2020 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.util

import java.awt.Color
import java.awt.image.BufferedImage
import java.net.URI
import java.util.stream.{Collectors, IntStream}
import java.util.zip.ZipFile
import java.{lang, util}

import com.fasterxml.jackson.annotation.JsonIgnore
import com.simiacryptus.mindseye.art.photo.SegmentUtil.{paintWithRandomColors, removeTinyInclusions}
import com.simiacryptus.mindseye.art.photo._
import com.simiacryptus.mindseye.art.photo.affinity.RasterAffinity.{adjust, degree}
import com.simiacryptus.mindseye.art.photo.affinity.RelativeAffinity
import com.simiacryptus.mindseye.art.photo.cuda.SmoothSolver_Cuda
import com.simiacryptus.mindseye.art.photo.topology.{RadiusRasterTopology, SearchRadiusTopology}
import com.simiacryptus.mindseye.lang.{Coordinate, Tensor}
import com.simiacryptus.mindseye.util.ImageUtil
import com.simiacryptus.notebook.{EditImageQuery, NotebookOutput, UploadImageQuery}
import com.simiacryptus.util.Util
import javax.imageio.ImageIO

import scala.collection.JavaConverters._
import scala.collection.mutable

abstract class SegmentingSetup extends ArtSetup[Object, SegmentingSetup] {

  @JsonIgnore lazy val fastPhotoStyleTransfer = FastPhotoStyleTransfer.fromZip(new ZipFile(Util.cacheFile(new URI(
    "https://simiacryptus.s3-us-west-2.amazonaws.com/photo_wct.zip"))))

  def smoothStyle(content: Tensor, style: Tensor, mask: Tensor)(implicit log: NotebookOutput) = {
    maskedDelta(mask, content, smoother(content)(wct(content, style, mask)))
  }

  def wct(content: Tensor, style: Tensor, mask: Tensor) = {
    val wctRestyled = fastPhotoStyleTransfer.photoWCT(style, content, mask.doubleStream().average().getAsDouble, 1.0)
    maskedDelta(mask, content, wctRestyled)
  }

  def maskedDelta(mask: Tensor, base: Tensor, changed: Tensor) = {
    val tensor = changed.mapCoords((c: Coordinate) => {
      val bg = mask.get(c)
      if (bg == 1) changed.get(c)
      else base.get(c)
    })
    changed.freeRef()
    base.freeRef()
    mask.freeRef()
    tensor
  }

  def smoother(content: Tensor) = {
    val topology = new SearchRadiusTopology(content)
    topology.setSelfRef(true)
    //.setVerbose(true)
    var affinity = new RelativeAffinity(content, topology)
    affinity.setContrast(10)
    affinity.setGraphPower1(2)
    affinity.setMixing(0.2)
    affinity.wrap((graphEdges: util.List[Array[Int]], innerResult: util.List[Array[Double]]) => adjust(graphEdges, innerResult, degree(innerResult), 0.5))
    solver.solve(topology, affinity, 1e-4)
  }

  def solver: SmoothSolver = new SmoothSolver_Cuda()

  def drawMask(content: BufferedImage, colors: Color*)(implicit log: NotebookOutput) = {
    val image_tensor: Tensor = Tensor.fromRGB(content)
    val dimensions: Array[Int] = image_tensor.getDimensions
    val pixels: Int = dimensions(0) * dimensions(1)

    log.h1("Image Region Decomposition")
    log.h2("Flattened Image")
    val topology = new RadiusRasterTopology(image_tensor.getDimensions, RadiusRasterTopology.getRadius(1, 1), 0).cached()
    //new SimpleRasterTopology(image_tensor.getDimensions).cached()
    val topology_analysis = new SearchRadiusTopology(image_tensor)
    topology_analysis.setInitialRadius(1)
    topology_analysis.setMaxSpatialDist(8)
    topology_analysis.setMaxChromaDist(0.1)
    topology_analysis.setNeighborhoodSize(5)
    topology_analysis.setSelfRef(true)
    topology_analysis.setVerbose(true)
    val rasterTopology = topology_analysis.cached()
    val affinity_regions = new RelativeAffinity(image_tensor, rasterTopology)
    affinity_regions.setContrast(25)
    affinity_regions.setGraphPower1(2)
    affinity_regions.setMixing(0.2)
    val flattenedColors = log.eval(() => {
      SegmentUtil.flattenColors(image_tensor,
        rasterTopology,
        affinity_regions.wrap((graphEdges: util.List[Array[Int]], innerResult: util.List[Array[Double]]) => adjust(graphEdges, innerResult, degree(innerResult), 0.5)), 3, solver)
    })
    val flattenedTensor = Tensor.fromRGB(flattenedColors)
    var pixelMap: Array[Int] = SegmentUtil.markIslands(
      topology,
      (x: Array[Int]) => flattenedTensor.getPixel(x: _*),
      (a: Array[Double], b: Array[Double]) => IntStream.range(0, a.length)
        .mapToDouble((i: Int) => a(i) - b(i))
        .map((x: Double) => x * x).average.getAsDouble < 0.2,
      64,
      pixels
    )
    val affinity_graph = new RelativeAffinity(image_tensor, topology_analysis)
    affinity_graph.setContrast(50)
    affinity_graph.setGraphPower1(2)
    affinity_graph.setMixing(0.5)
    var graph = SmoothSolver_Cuda.laplacian(affinity_graph, topology).matrix.project(pixelMap)

    log.h2("Basic Regions")
    log.eval(() => {
      paintWithRandomColors(topology, pixelMap, graph)
    })

    log.h2("Remove tiny islands")
    log.eval(() => {
      var projection1 = removeTinyInclusions(pixelMap, graph, 1, 4)
      graph = graph.project(projection1).assertSymmetric()
      pixelMap = pixelMap.map(projection1(_))
      val activeRows = graph.activeRows()
      graph = graph.select(activeRows).assertSymmetric()
      var projection2 = activeRows.zipWithIndex.toMap
      pixelMap = pixelMap.map(projection2.get(_).getOrElse(0))
      paintWithRandomColors(topology, pixelMap, graph)
    })

    log.h2("Reduced Regions")
    log.eval(() => {
      val regionAssembler = RegionAssembler.volumeEntropy(graph, pixelMap, image_tensor, topology)
        .reduceTo(5000)
      graph = graph.project(regionAssembler.getProjection)
      pixelMap = regionAssembler.getPixelMap

      val denseProjection = graph.getDenseProjection
      graph = graph.project(denseProjection)
      pixelMap = pixelMap.map(denseProjection.apply(_))
      paintWithRandomColors(topology, pixelMap, graph)
    })

    log.eval(() => {
      graph = graph.filterDiagonal().sortAndPrune().recalculateConnectionWeights(topology, image_tensor, pixelMap, 1e-3, 0.5, 1e-9).sortAndPrune()
      val denseProjection = graph.getDenseProjection
      graph = graph.project(denseProjection)
      pixelMap = pixelMap.map(denseProjection.apply(_))
      paintWithRandomColors(topology, pixelMap, graph)
      null
    })

    log.eval(() => {
      val regionAssembler = RegionAssembler.volumeEntropy(graph, pixelMap, image_tensor, topology)
        .reduceTo(1000)
      graph = graph.project(regionAssembler.getProjection)
      pixelMap = regionAssembler.getPixelMap

      val denseProjection = graph.getDenseProjection
      graph = graph.project(denseProjection)
      pixelMap = pixelMap.map(denseProjection.apply(_))
      paintWithRandomColors(topology, pixelMap, graph)
    })

    log.eval(() => {
      graph = graph.filterDiagonal().sortAndPrune().recalculateConnectionWeights(topology, image_tensor, pixelMap, 1e-3, 0.5, 1e-9).sortAndPrune()
      val denseProjection = graph.getDenseProjection
      graph = graph.project(denseProjection)
      pixelMap = pixelMap.map(denseProjection.apply(_))
      paintWithRandomColors(topology, pixelMap, graph)
      null
    })

    //val sortedValues = SparseMatrixFloat.toDouble(graph.values).toStream.sorted.toArray
    val selection = select(log, content, colors: _*)
    val eigenSystem: util.Map[Array[Float], lang.Float] = log.eval(() => {
      graph.dense_graph_eigensys()
    })
    val seedVectors: List[Array[Double]] = log.eval(() => {
      val seedMarks = (for (
        x <- 0 until dimensions(0);
        y <- 0 until dimensions(1);
        color <- selection(x, y)
      ) yield pixelMap(topology.getIndexFromCoords(x, y)) -> color).groupBy(_._1).mapValues(_.head._2)
      for (markColor <- seedMarks.values.toList.distinct.sorted) yield {
        val array = (0 until graph.rows).map(elementIndex => if (seedMarks.get(elementIndex).filter(_ == markColor).isDefined) 1.0 else 0.0).toArray
        val mag = Math.sqrt(array.map(x => x * x).sum)
        array.map(_ / mag)
      }
    })

    var finalPartitioning: Map[Int, Int] = Map.empty
    for (eigenLimit <- List(10, 20, 30, 40, 50, 60, 80, 100, 200, 500, 750, 1000, 50)) {
      log.h2("Eigenvalue selection: " + eigenLimit)
      finalPartitioning = log.eval(() => {
        val projectedSelection = seedVectors.map(seedVector => {
          eigenSystem.asScala.toList.sortBy(_._2).take(eigenLimit).map(eigenEntry => {
            val eigenVector = eigenEntry._1
            val mag = Math.sqrt(eigenVector.map(x => x * x).sum)
            val dot = eigenVector.map(_ / mag).zip(seedVector).map(x => x._1 * x._2).sum
            eigenVector.map(_ * dot)
          }).reduce(_.zip(_).map(x => x._1 + x._2))
        })
        (0 until graph.rows).map(elementIndex => elementIndex -> projectedSelection.zipWithIndex.maxBy(_._1(elementIndex))._2).toMap
      })
      log.eval(() => {
        val colorMap = colors.zipWithIndex.map(tuple => {
          val (color, index) = tuple
          index.asInstanceOf[Integer] -> Array(color.getRed.toDouble, color.getGreen.toDouble, color.getBlue.toDouble)
        }).toMap
        SegmentUtil.paint(topology, pixelMap.map(finalPartitioning(_)), colorMap.asJava)
      })
      for (clr <- 0 until colors.size) yield {
        log.eval(() => {
          image_tensor.mapCoords((coordinate: Coordinate) => {
            val coords = coordinate.getCoords()
            if (finalPartitioning.get(pixelMap(topology.getIndexFromCoords(coords(0), coords(1)))).map(_ == clr).getOrElse(false)) {
              image_tensor.get(coordinate)
            } else {
              0.0
            }
          }).toImage
        })
      }
    }

    //      val finalPartitioning = RegionAssembler.epidemic(graph, pixelMap, image_tensor, topology,
    //        seedMarks.map(t => t._1.asInstanceOf[Integer] -> t._2.asInstanceOf[Integer]).asJava
    //      ).reduceTo(1).regions.asScala.flatMap((region) =>
    //        region.original_regions.asScala.map(_.toInt -> region.marks)
    //      ).toMap.filter(_._2.size() == 1).mapValues(_.asScala.head.toInt)

    for (clr <- 0 until colors.size) yield {
      image_tensor.mapCoords((coordinate: Coordinate) => {
        val coords = coordinate.getCoords()
        if (finalPartitioning.get(pixelMap(topology.getIndexFromCoords(coords(0), coords(1)))).map(_ == clr).getOrElse(false)) {
          image_tensor.get(coordinate)
        } else {
          0.0
        }
      })
    }
  }

  def select(log: NotebookOutput, image: BufferedImage, colors: Color*) = {
    val editResult = new EditImageQuery(log, image).print().get()
    val diff_tensor = diff(Tensor.fromRGB(image), Tensor.fromRGB(editResult))

    def apxColor(a: Array[Double]): List[Int] = a.map(x => x.toInt).toList

    val colorList = diff_tensor.getPixelStream.collect(Collectors.toList())
      .asScala.map(apxColor).filter(_.sum != 0).groupBy(x => x).mapValues(_.size)
      .toList.sortBy(-_._2).take(colors.size).map(_._1).toArray

    val selectionIndexToColorIndex = colors.zipWithIndex.map(tuple => {
      val (color, index) = tuple
      colorList.zipWithIndex.sortBy(x => dist(color, x._1.map(_.doubleValue()))).head._2 -> index
    }).toArray.toMap
    val colorsMap = colorList.zipWithIndex.toMap
    (x: Int, y: Int) => colorsMap.get(apxColor(diff_tensor.getPixel(x, y))).flatMap(selectionIndexToColorIndex.get(_))
  }

  def uploadMask(content: BufferedImage, colors: Color*)(implicit log: NotebookOutput) = {
    val maskFile = new UploadImageQuery("Upload Mask", log).print().get()
    val maskTensor = Tensor.fromRGB(ImageUtil.resize(ImageIO.read(maskFile), content.getWidth, content.getHeight))
    val tensor = Tensor.fromRGB(content)
    val tensors = for (clr <- 0 until colors.size) yield {
      tensor.mapCoords((coordinate: Coordinate) => {
        val Array(x, y, c) = coordinate.getCoords()
        val pixelColor = maskTensor.getPixel(x, y)
        val closestColor = colors.zipWithIndex.sortBy(x => dist(x._1, pixelColor)).head
        if (closestColor._2 == clr) {
          tensor.get(coordinate)
        } else {
          0.0
        }
      })
    }
    tensor.freeRef()
    tensors
  }

  def dist(color: Color, x: Seq[Double]) = {
    List(
      color.getRed - x(2).doubleValue(),
      color.getGreen - x(1).doubleValue(),
      color.getBlue - x(0).doubleValue()
    ).map(x => x * x).sum
  }

  def select(log: NotebookOutput, image: BufferedImage, partitions: Int) = {
    val editResult = new EditImageQuery(log, image).print().get()
    var diff_tensor = diff(Tensor.fromRGB(image), Tensor.fromRGB(editResult))

    def apxColor(a: Array[Double]): List[Int] = a.map(x => x.toInt).toList

    val dimensions = diff_tensor.getDimensions
    val colors: Map[List[Int], Int] = diff_tensor.getPixelStream.collect(Collectors.toList())
      .asScala.map(apxColor).filter(_.sum != 0).groupBy(x => x).mapValues(_.size)
      .toList.sortBy(-_._2).take(partitions).map(_._1).toArray.zipWithIndex.toMap
    (x: Int, y: Int) => colors.get(apxColor(diff_tensor.getPixel(x, y)))
  }

  def diff(image_tensor: Tensor, edit_tensor: Tensor): Tensor = {
    edit_tensor.mapCoords((c: Coordinate) => {
      val val_tensor = image_tensor.get(c.getIndex)
      val val_edit = edit_tensor.get(c.getIndex)
      if (Math.abs(val_edit - val_tensor) > 1) {
        val_edit
      } else {
        0
      }
    })
  }

  def expand(tree: RegionAssembler.RegionTree, markup: Map[Int, Set[Int]], recursion: Int = 0): Map[Int, Int] = {
    if (recursion > 1000) throw new RuntimeException()
    val paints = tree.regions.flatMap(r => markup.get(r)).flatten.distinct
    if (paints.length > 1) {
      val children = tree.children
      val buffer: mutable.Buffer[(Int, Int)] = List.empty.toBuffer // children.flatMap(expand(_, markup).toList).toMap.toBuffer
      for (child <- children) buffer ++= expand(child, markup, recursion + 1)
      buffer.toMap
    } else if (paints.length == 0) {
      Map.empty
    } else {
      tree.regions.map(_ -> paints.head).toMap
    }
  }
}
