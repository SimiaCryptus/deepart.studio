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

import java.util.UUID

import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.model.{CannedAccessControlList, ObjectMetadata, PutObjectRequest}
import com.simiacryptus.aws.EC2Util
import com.simiacryptus.mindseye.lang.Tensor

/**
 * This class represents a job registration for a JPG file.
 *
 * @param bucket      the name of the bucket where the file is located
 * @param reportUrl   the URL of the report page for this job
 * @param liveUrl     the URL of the live page for this job
 * @param canvas      a function that returns a Tensor representing the file's contents
 * @param instances   a list of instances where this job is running; defaults to the current instance's ID
 * @param id          the job's ID
 * @param indexFile   the name of the index file for this job; defaults to "index.html"
 * @param className   the name of the class for this job
 * @param indexStr    a string to be written to the index file for this job
 * @param description a description of this job
 * @docgenVersion 9
 */
class JpgRegistration
(
  bucket: String,
  reportUrl: String,
  liveUrl: String,
  canvas: () => Tensor,
  instances: List[String] = List(
    EC2Util.instanceId()
  ).filterNot(_.isEmpty),
  id: String = UUID.randomUUID().toString,
  indexFile: String = "index.html",
  className: String = "",
  indexStr: String = "",
  description: String = ""
) extends JobRegistration[Tensor](bucket, reportUrl, liveUrl, canvas, instances, id, indexFile, className, indexStr, description) {

  /**
   * Upload an image to Amazon S3
   *
   * @param canvas   the image to upload
   * @param s3client an AmazonS3 client
   * @docgenVersion 9
   */
  def uploadImage(canvas: Tensor)(implicit s3client: AmazonS3) = {
    val key = s"img/$id.jpg"
    logger.info("Writing " + key)
    val metadata = new ObjectMetadata()
    metadata.setContentType("image/jpeg")
    val image = canvas.toImage
    canvas.freeRef()
    s3client.putObject(new PutObjectRequest(bucket, key, toStream(image), metadata)
      .withCannedAcl(CannedAccessControlList.PublicRead))
    s"http://$bucket/$key"
  }

}

