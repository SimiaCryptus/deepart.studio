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

import com.amazonaws.services.ec2.AmazonEC2
import com.amazonaws.services.ec2.model.DescribeInstanceStatusRequest
import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.model.ListObjectsRequest
import com.simiacryptus.sparkbook.util.ScalaJson
import org.apache.commons.io.IOUtils

import scala.collection.JavaConverters._
//import scala.jdk.CollectionConverters._
import scala.util.Try

object JobRegistry {

  /**
   * Lists the job registries in the specified bucket.
   *
   * @param bucket   the bucket to list
   * @param s3client the Amazon S3 client
   * @return a sequence of job registries
   * @docgenVersion 9
   */
  def list(bucket: String)(implicit s3client: AmazonS3): Seq[JobRegistry] = {
    val objectListing = s3client.listObjects(new ListObjectsRequest().withBucketName(bucket).withPrefix("jobs/"))
    (for (item <- objectListing.getObjectSummaries.asScala) yield {
      ScalaJson.fromJson(IOUtils.toString(s3client.getObject(bucket, item.getKey).getObjectContent, "UTF-8"), classOf[JobRegistry])
    }).toList
  }

}

/**
 * This case class represents a job registry, which contains information about a job such as its report URL, live URL, last report, instances, image, id, class name, index string, and description.
 *
 * @docgenVersion 9
 */
case class JobRegistry
(
  reportUrl: String,
  liveUrl: String,
  lastReport: Long,
  instances: List[String],
  image: String,
  id: String,
  className: String,
  indexStr: String,
  description: String
) {

  /**
   * Save the job to the specified S3 bucket.
   *
   * @param bucket   the S3 bucket to save to
   * @param s3client the AmazonS3 client to use
   * @return the S3 URL of the saved job
   * @docgenVersion 9
   */
  def save(bucket: String)(implicit s3client: AmazonS3) = {
    val key = s"jobs/$id.json"
    s3client.putObject(bucket, key, ScalaJson.toJson(this).toString)
    s"s3://$bucket/$key"
  }

  /**
   * Checks if there are any running instances.
   *
   * @param ec2client the AmazonEC2 client
   * @return true if there are running instances, false otherwise
   * @docgenVersion 9
   */
  def isLive()(implicit ec2client: AmazonEC2) = Try {
    !runningInstances(ec2client).isEmpty
  }

  /**
   * Returns a list of running instances.
   *
   * @param ec2client the Amazon EC2 client
   * @return a list of running instances
   * @docgenVersion 9
   */
  def runningInstances(implicit ec2client: AmazonEC2) = {
    val instances = this.instances.filterNot(_.isEmpty)
    ec2client.describeInstanceStatus(new DescribeInstanceStatusRequest()
      .withInstanceIds(instances: _*))
      .getInstanceStatuses.asScala
      .filter(x => instances.contains(x.getInstanceId))
      .filter(_.getInstanceState.getName == "running")
      .toList
  }
}
