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

package com.simiacryptus.mindseye.art.registry

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.util.UUID

import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.model.{CannedAccessControlList, ObjectMetadata, PutObjectRequest}
import com.simiacryptus.aws.EC2Util
import com.simiacryptus.sparkbook.NotebookRunner

/**
 * This class represents a job registration for a GIF.
 *
 * @param bucket      the bucket to store the GIF in
 * @param reportUrl   the URL to report the status of the job to
 * @param liveUrl     the URL to view the live results of the job
 * @param canvas      a function that returns the sequence of images to use in the GIF
 * @param instances   the list of instances to run the job on
 * @param id          the id of the job
 * @param indexFile   the index file to use
 * @param className   the class name to use
 * @param indexStr    the index string to use
 * @param description the description of the job
 * @param delay       the delay between frames in the GIF, in milliseconds
 * @docgenVersion 9
 */
class GifRegistration
(
  bucket: String,
  reportUrl: String,
  liveUrl: String,
  canvas: () => Seq[BufferedImage],
  instances: List[String] = List(
    EC2Util.instanceId()
  ).filterNot(_.isEmpty),
  id: String = UUID.randomUUID().toString,
  indexFile: String = "index.html",
  className: String = "",
  indexStr: String = "",
  description: String = "",
  delay: Int = 100
) extends JobRegistration[Seq[BufferedImage]](bucket, reportUrl, liveUrl, canvas, instances, id, indexFile, className, indexStr, description) {

  /**
   * Uploads an image to Amazon S3
   *
   * @param canvas   the image to upload
   * @param s3client an AmazonS3 client
   * @docgenVersion 9
   */
  def uploadImage(canvas: Seq[BufferedImage])(implicit s3client: AmazonS3) = {
    val key = s"img/$id.gif"
    logger.info("Writing " + key)
    val metadata = new ObjectMetadata()
    val stream = new ByteArrayOutputStream()
    NotebookRunner.toGif(stream, canvas, delay, 1200)
    metadata.setContentType("image/gif")
    s3client.putObject(new PutObjectRequest(bucket, key, new ByteArrayInputStream(stream.toByteArray), metadata)
      .withCannedAcl(CannedAccessControlList.PublicRead))
    s"http://$bucket/$key"
  }

}

