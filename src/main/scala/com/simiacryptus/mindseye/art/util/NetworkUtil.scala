package com.simiacryptus.mindseye.art.util

import java.util.UUID

import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.{DeltaSet, Layer, Result, Tensor, TensorArray, TensorList}
import com.simiacryptus.mindseye.layers.WrapperLayer
import com.simiacryptus.mindseye.network.{DAGNetwork, DAGNode}
import com.simiacryptus.mindseye.test.GraphVizNetworkInspector
import com.simiacryptus.notebook.NotebookOutput

object NetworkUtil {

  /**
   * Graph the given trainable using the implicit log.
   *
   * @docgenVersion 9
   */
  def graph(trainable: Trainable)(implicit log: NotebookOutput): Unit = {
    val layer = trainable.getLayer
    trainable.freeRef()
    if (layer != null) {
      if (layer.isInstanceOf[DAGNetwork]) {
        graph(layer.asInstanceOf[DAGNetwork])
      } else {
        layer.freeRef()
      }
    }
  }

  /**
   * Prints a summary of the given layer to the notebook output.
   *
   * @docgenVersion 9
   */
  def debugSummary(layer: Layer)(implicit log: NotebookOutput): Unit = {
    if (layer.isInstanceOf[WrapperLayer]) {
      val inner = layer.asInstanceOf[WrapperLayer].getInner
      log.p(s"${layer.getName} wraps ${inner.getName}")
      inner.freeRef()
    } else {
      val modelName = s"${UUID.randomUUID().toString}.json"
      log.p(s"Json for ${layer.getName}: " + log.file(layer.getJson.toString, modelName, modelName))
    }
    layer.freeRef()

  }

  /**
   * This function graphs a DAGNetwork.
   *
   * @param network The DAGNetwork to graph.
   * @param log     The NotebookOutput to use.
   * @docgenVersion 9
   */
  def graph(network: DAGNetwork)(implicit log: NotebookOutput): Unit = {
    log.subreport(s"Network Diagram - ${network.getName}", (sub: NotebookOutput) => {
      val layers = network.getLayers
      GraphVizNetworkInspector.graph(sub, network)
      (0 until layers.size()).foreach(i => {
        val layer = layers.get(i)
        if (layer != null) {
          if (layer.isInstanceOf[DAGNetwork]) {
            graph(layer.asInstanceOf[DAGNetwork])(sub)
          } else {
            debugSummary(layer)(sub)
          }
        }
      });
      layers.freeRef();
      null
    })
  }

  /**
   * Graph the given DAGNetwork using the provided sampleInput.
   *
   * @param network     the DAGNetwork to graph
   * @param sampleInput the input to use for graphing
   * @param log         the NotebookOutput to use
   * @docgenVersion 9
   */
  def graph(network: DAGNetwork, sampleInput: TensorList*)(implicit log: NotebookOutput): Unit = {
    log.subreport(s"Network Diagram - ${network.getName}", (sub: NotebookOutput) => {

      val nodes = network.getNodes
      GraphVizNetworkInspector.graph(sub, network)

      val evaluationContext = network.buildExeCtx(sampleInput.map(new Result(_)): _*)

      val head = network.getHead
      val networkResult = head.eval(evaluationContext.addRef())
      val networkResultData = networkResult.getData
      val deltaset = new DeltaSet[UUID]()
      networkResult.accumulate(deltaset.addRef(), new TensorArray((0 until networkResultData.length()).map(i => {
        val tensor = networkResultData.get(i)
        try {
          tensor.map(v => 1.0)
        } finally {
          tensor.freeRef()
        }
      }): _*))
      head.freeRef()
      outputData(Result.getData(networkResult), "Network output")(sub)
      (0 until nodes.size()).foreach(i => {
        val node: DAGNode = nodes.get(i)

        /**
         * Logs the delta for the given layer.
         *
         * @param layer the layer to log the delta for
         * @docgenVersion 9
         */
        def logDelta(layer: Layer) = {
          outputData({
            val value = deltaset.get(layer.getId)
            if (null == value) {
              null
            } else {
              try {
                value.getDelta
              } finally {
                value.freeRef()
              }
            }
          }, s"Delta for ${layer.getName}")(sub)
        }

        /**
         * Outputs the given layer.
         *
         * @docgenVersion 9
         */
        def output(layer: Layer): Unit = {
          if (layer.isInstanceOf[DAGNetwork]) {
            graph(layer.asInstanceOf[DAGNetwork], node.getInputs.map(inputNode => {
              try {
                Result.getData(inputNode.eval(evaluationContext.addRef()))
              } finally {
                inputNode.freeRef()
              }
            }): _*)(sub)
          } else if (layer.isInstanceOf[WrapperLayer]) {
            val inner = layer.asInstanceOf[WrapperLayer].getInner()
            layer.freeRef()
            debugSummary(layer.addRef().asInstanceOf[Layer])(sub)
            output(inner)
          } else {
            debugSummary(layer.addRef().asInstanceOf[Layer])(sub)
            logDelta(layer)
            layer.freeRef()
          }
        }

        var layer: Layer = node.getLayer[Layer]()
        if (layer != null) {
          outputData(Result.getData(node.eval(evaluationContext.addRef())), s"Output for ${layer.getName}")(sub)
          output(layer)
        }
        node.freeRef()
      });
      nodes.freeRef();
      null
    })
  }

  /**
   * Outputs the given data with the given label to the notebook output.
   *
   * @docgenVersion 9
   */
  def outputData(data: Array[Double], label: String)(implicit log: NotebookOutput) = {
    if (null == data) {
      log.p(s"$label: No Data")
    } else {
      val fileName = s"${UUID.randomUUID().toString}.json"
      log.p(s"$label: " + log.file(Tensor.toJson(data).toString, fileName, fileName))
    }
  }

  /**
   * Outputs the data from the given network output and label.
   *
   * @param networkOutput the output from the network
   * @param label         the label for the data
   * @param log           the log to output to
   * @docgenVersion 9
   */
  def outputData(networkOutput: TensorList, label: String)(implicit log: NotebookOutput) = {
    val fileName = s"${UUID.randomUUID().toString}.json"
    log.p(s"$label: " + log.file(networkOutput.getJsonRaw().toString, fileName, fileName))
    networkOutput.freeRef()
  }

}
