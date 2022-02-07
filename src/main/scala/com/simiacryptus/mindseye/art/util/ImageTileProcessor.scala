package com.simiacryptus.mindseye.art.util

import java.util.concurrent.atomic.AtomicBoolean

import com.simiacryptus.mindseye.art.TiledTrainable
import com.simiacryptus.mindseye.lang.{Layer, Result, Tensor}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.ref.wrappers.RefAtomicReference
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg

trait ImageTileProcessor {
  def tile_padding: Int

  def paintPerTile(canvasRef: RefAtomicReference[Tensor], tile_size: Int, tilePainter: (Tensor, RefAtomicReference[Tensor]) => Unit)(implicit log: NotebookOutput) = {
    val canvas = canvasRef.get()
    val canvasSize = canvas.getDimensions()
    val priorTiles = Array.empty[Tensor].toBuffer

    def fade(tile: Tensor, tileView_fade: Layer) = {
      val maskTile = Result.getData0(tileView_fade.eval(canvas.map(x => 1)))
      require(maskTile.getDimensions.toList == tile.getDimensions.toList, s"${maskTile.getDimensions.toList} != ${tile.getDimensions.toList}")
      tile.mapCoords(c => tile.get(c) * maskTile.get(c))
    }

    val selectors_fade = TiledTrainable.selectors(tile_padding, canvasSize(0), canvasSize(1), tile_size, true)
    val selectors_sharp = TiledTrainable.selectors(tile_padding, canvasSize(0), canvasSize(1), tile_size, false)
    val tileRefs = for (tileView_sharp <- selectors_sharp) yield {
      new RefAtomicReference(Result.getData0(tileView_sharp.eval(canvas.addRef())))
    }
    canvas.freeRef()
    val tileProcessedFlags = for (_ <- tileRefs) yield {
      new AtomicBoolean(false)
    }

    def tiles = for (((tile, tileProcessed), tileView_fade) <- tileRefs.zip(tileProcessedFlags).zip(selectors_fade)) yield {
      if (tileProcessed.get()) {
        tile.get()
      } else {
        fade(tile.get(), tileView_fade)
      }
    }

    def reassembled = TiledTrainable.reassemble(canvasSize(0), canvasSize(1), tiles, tile_padding, tile_size, true, false)

    log.eval("Tile Sizes", () => {
      tileRefs.map(_.get().getDimensions.toList).groupBy(x => x).mapValues(_.size)
    })

    withMonitoredJpg(() => {
      val tensor = reassembled
      val image = tensor.toRgbImage
      tensor.freeRef()
      image
    }) {
      for (((tileView_fade, tileRef), tileProcessed) <- selectors_fade.zip(tileRefs).zip(tileProcessedFlags)) yield {
        if (priorTiles.isEmpty) {
          tilePainter(tileRef.get(), tileRef)
          val result = fade(tileRef.get(), tileView_fade)
          tileRef.set(result)
          tileProcessed.set(true)
        } else {
          val prior: Tensor = fade(priorTiles.remove(0), tileView_fade)
          log.out(log.jpg(prior.toRgbImage, "Tile Product"))
          tileRef.set(prior)
          tileProcessed.set(true)
        }
      }
    }
    canvasRef.set(reassembled)
  }

}
