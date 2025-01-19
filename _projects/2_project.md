---
layout: page
title: Autonomous Robotic Product Picking and Storage
description:  Design and evaluation of an object grasping plan using Generative grasping convolutional neural networks (GGCNN) for a robotic system for autonomous product picking and storage with the settings of the Amazon Picking Challenge (APC).
img: assets/img/project2/APC.gif
importance: 2
category: study
related_publications: false
---
Robots have excelled at operating in controlled environments since their introduction to industry over half a century ago.

However, logistics companies, e-commerce platforms, and even supermarkets, which manage a diverse range of products, require high flexibility to efficiently organize and pack items. Currently, there are only a few robotic solutions capable of meeting these demands. To address this challenge, Amazon launched the Amazon Robotics Challenge (ARC) in 2015. The competition aimed to advance research in warehouse automation, particularly in unstructured environments, focusing on robotic systems for product picking and storage.

At the National University of Colombia in 2021, we simulated this setup with an industrial robot from LabSIR (Laboratory of Intelligent Robotized Systems), initially using ROS + Gazebo and later implementing it physically, as shown in the following image. The setup consists of a shelf, a drawer, the robot, and a camera mounted on the drawer. In this case, we used the Kinect camera because it provides depth measurement capabilities.
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Simulation of our setup in Gazebo and the real setup in LabSIR at the National University of Colombia in Bogotá.
</div>
The Kinect sensor is mounted on the robot, pointing directly at the drawer containing the objects. The following image shows the Kinect's view: on the left, the Gazebo simulation; in the center, the RGB camera; and on the right, the depth camera.
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
View from the Kinect sensor:
</div>
Processing of the depth image to determine the optimal position for grasping objects, captured by the Kinect using Generative Grasping Convolutional Networks (GGCNN). This network outputs the angle, quality, and width of the grasping options.
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_2a.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Pipeline for processing the output of the GGCNN model.
</div>
We present an example of a real image captured by the Kinect (top), the raw output of the GGCNN (bottom right), and the final post-processed positions and orientations of the grasps for the objects in the box, overlaid on the depth image.
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Image captured by the Kinect, along with the GGCNN model output and the pipeline for generating the grasping options that the robot will execute.
</div>

Our framework also works in scenarios with a high density of objects, prioritizing those that are easier to grasp based on the size of the robot's hand. Objects positioned very close to the sides of the box are assigned a lower grasping quality.
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_4.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Grasping detections in the case where the box is full.
</div>
We recorded a video demonstrating our framework with no human intervention, randomly placing objects in the box: [Link to video](https://www.youtube.com/watch?v=Xct78P9-zjw&ab_channel=Mateo%C3%81vila).
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0"> <!-- Increase column width -->
        {% include figure.liquid path="assets/img/project2/APC_5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Screenshots from the videos showing our framework working with the industrial robot in LabSIR, operating without any human intervention.
</div>






