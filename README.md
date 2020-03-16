# DeepArtist.org

## What is it?
[DeepArtist.org](examples.deepartist.org) is an image processing platform using convolutional neural networks to perform state-of-the-art image processing techniques. This software is targeted at hobbyists and digital artists, and as such this documentation is focused on the practical tools provided to produce pretty pictures. To run locally you need a recent [Nvidia GPU](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#prerequisites-windows), but you can also run these tasks on [Amazon EC2](https://aws.amazon.com/machine-learning/amis/).

In contrast to many contemporary AI projects, this project uses Scala, not Python. I have attempted to simplify things so that very little programming literacy at all is required, and the learning curve is friendly to beginners who want to experiment. One benefit of this approach is that this software runs easily on a variety of systems, including Windows and Mac desktops and on the Cloud.

As you will quickly discover, these painting processes have many controls and settings and can be customized in countless ways. Getting an ideal image result may require quite a bit of experimentation, and you will need a way to track this work. Additionally, you may develop new artistic pipelines, and you may wish to share them. To support these types of workflows, DeepArtist applications output rich text and images as output and have them automatically built into a static website on s3.

A key part of DeepArtist.org are [the examples](examples.deepartist.org) - A set of starting applications that use the platform to demonstrate the basis image processing techniques and concepts.

# How to Install and Run

## Development Tools

First, there are some basic tools needed to run this software:

1. [Java](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) - You probably have some version of Java, but you will need to install a Java Development Kit so you can compile and run the code. There are several distributions of JDK, many freely available. I use Oracle’s JDK 8.
1. [Git](https://git-scm.com/downloads) - Several friendly distributions and interface wrappers exist for Git, but I prefer the basic command line. You can probably get away without this if you use the IntelliJ git plugin.
1. [IntelliJ](https://www.jetbrains.com/idea/download/#section=windows) - You don’t technically need a fancy development environment, but this is free and well worth this disk space. IntelliJ provides a “community edition” which has basically all the Java features you will need to run this software and develop with it. I used the free version to develop almost all of this software.

## Github
If you aren’t a member of Github, I recommend signing up for a free account. You can then “fork” examples.deepartist.com so you can have your own copy to customize.
https://github.com/join

Setup intellij plugin auto token
https://www.jetbrains.com/help/idea/github.html

## Install CuDNN (sign up with NVidia)
If you want to run jobs locally, and you have a good NVidia GPU, you will need to install CUDNN (CUDA Deep Neural Network Library). If you are not running jobs locally, Amazon Deep Learning instances come with this library pre-installed. Though downloading is free, unfortunately, NVidia has yet to make a public release of this library and requires an NVidia developer membership. Fortunately, this is free to sign up for.

1. [Update GPU drivers](https://www.nvidia.com/Download/index.aspx)
1. [Sign up page](https://developer.nvidia.com/developer-program/signup)
1. [Download page](https://developer.nvidia.com/cudnn)

## Setup AWS, with CLI tools and environment login
AWS is used (optionally) for storing results, serving them as a website, and for executing jobs themselves. An AWS account in itself is free, and you pay for pretty much exactly what you use, but you can still rack up a bill fast if you don’t keep an eye on it. If you want to use it, you will need to sign up for an account, install the command line tools, and establish environment credentials

1. [Sign up for AWS](https://aws.amazon.com/)
1. [Install CLI tools](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)
1. [Setup local admin role](https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started_create-admin-group.html)
1. [Setup storage bucket and website](https://docs.aws.amazon.com/AmazonS3/latest/dev/WebsiteHosting.html)

## Download and Open Project
Now that the basics are setup, we will download and load the examples project on your system:

1. Clone the [examples.deepartist.org](https://github.com/SimiaCryptus/examples.deepartist.org) repo 
1. Open IDE and Import the project

## Run a local demo - BasicNotebook
We are now ready to run our first project. To simplify things, let’s run a very basic project that doesn’t use any neural networks. Open the examples project and then open the BasicNotebook class.

1. Open class and "run" - Right click the file and choose "run"
1. Load page, confirm configuration - Open a browser to [http://localhost:1080/](http://localhost:1080/). The initial page will prompt the user to edit the settings of the job. You can edit the text to be rendered, the size, the target s3 bucket, and many other settings in other jobs
1. Refresh and upload image - Assuming the mask url was left intact, the job will (upon refreshing the page) prompt you to upload an image. Upload one.
1. Wait for completion - One the upload completes, the job runs. This job is fast and will complete in seconds.
1. Examine results - The results will have been saved in a local /reports/ directory that is noted in the output logs, and if setup also on S3.

[![](http://examples.deepartist.org/img/834f07ce-180b-48a9-bdc6-81160e904ce5.jpg)](http://examples.deepartist.org/BasicNotebookEC2/9233c16d-691b-4955-9c57-8e95cd641551/9233c16d-691b-4955-9c57-8e95cd641551.html)

From here, see and run more involved jobs from [examples.deepart.org](http://examples.deepart.org).

# Advanced Operation

Things don’t always go as planned, and sometimes you want to do something other than pressing “Run”. This section describes some of these common scenarios

## Continue aborted optimization
If you have to interrupt a job for any reason, you may want to be able to continue painting where you left off. This is possible by saving the latest image provided via http, and using that image to upload as the initial canvas url.

## Run on EC2
Any job can be run locally or on EC2. Most jobs are by default coded to run locally, but by modifying or adding a new object which derives from the same class, you can configure the EC2 deployment code. This will use your environment credentials to start a new AWS EC2 instance, upload all the needed code, and remotely start the application. When running, you can manage it via http just as in local execution, and additionally it will send you emails at the start and end of the job run. Watch your execution times! These are Deep Learning instance types, with a high amount of power that comes with a noticeable price tag.

When you run for the first time, you will be prompted to enter your email via stdin, and the role used to start the instance will likely not have all the needed permissions, for example to write to the s3 bucket you intend to publish to. It is recommended that during this first execution, you log into the AWS console and manually assign any needed permissions to the newly created role.

## Changing port, bucket, and region
Many important settings can be configured via property overrides on the job classes. Of particular note are:

1. Bucket - Most users will want to configure the bucket field to be something they have access to write to
1. Http port - If you want to use a port other than 1080, you can override it via the http_port field

Another important config, which is settable via a java property, is the AWS region to use. To use something other than the default us-east-1, simply set the AWS_REGION property e.g. by adding the execution argument -DAWS_REGION=us-west-2

## Utility HTTP Endpoints
Once an application is running, there are several other pages that can be requested for helpful information:

1. /cuda/info.txt - This endpoint returns diagnostics text that is returned by the CuDNN library
1. /shutdown - This endpoint immediately kills the JVM, and if running on EC2 also triggers instance termination.
1. /threads.json - This endpoint returns information on all stack traces and various thread information from the JVM. Useful for internal code development.
1. /cuda/stats.json - This endpoint returns profiling data collected during CuDNN layerTestParameters. Useful for internal code development.

# Background
This software is built on top of other software and technology.

## Software Technologies:
1. [CuDNN](https://developer.nvidia.com/cudnn) and CUDA - Nvidia publishes many useful number-crunching libraries that take advantage of the parallel computing abilities of its GPUs. CuDNN in particular is the prefered acceleration library used by deepartist.org.   
1. [TensorFlow](https://www.tensorflow.org/) - Google’s AI research team has open-sourced Tensorflow, a portable runtime for expressing and running neural networks. This library can be used for importing vision pipelines into DeepArtist.org, such as the pre-packaged Inception model.
1. Java and [Scala](https://www.scala-lang.org/) - Java is arguably the most popular programming environment today. Scala runs within the Java environment as an alternate language, and is one of the most expressive languages available. Both have a host of free, world-class tools available.

## Image Processing Methods
1. [Deep Dream](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) - In 2015 Google Research published Deep Dream, which demonstrated a use of convolutional neural networks to produce interesting visual effects.
1. [Deep Style Transfer](https://arxiv.org/abs/1508.06576) - In 2016 researchers published a style transfer algorithm using convolutional neural networks, whose approach inspires the bulk of the processing methods used by DeepArtist.org

## Deep Convolutional Networks for Object Recognition
Research into AI requires verifiable research, and to aid that many papers include published pretrained models. These pretrained models can be imported so that the latent patterns they have learned can be used in image processing. At the time of publishing, DeepArtist.org has 3 pre-packaged pipelines which are automatically downloaded upon usage:

1. [VGG16 and VGG19](https://arxiv.org/abs/1409.1556) - These are large, very simple networks using a series of simple convolution, pooling, and activation layers. As it turns out, the dense and simple structure of these networks seem to provide much better behavior for image processing.   
1. Inception - A sample import of a tensorflow model, a pretrained inception model is also provided. It uses a more complicated structure to perform well at it’s primary task, classification, with a much smaller size (and a faster speed).

