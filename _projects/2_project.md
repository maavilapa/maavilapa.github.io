---
layout: page
title: Autonomous Robotic Product Picking and Storage
description:  Design and evaluation of an object grasping plan using Generative grasping convolutional neural networks (GGCNN) for a robotic system for autonomous product picking and storage with the settings of the Amazon Picking Challenge (APC).
img: assets/img/project2/APC.gif
importance: 2
category: study
related_publications: false
---
For the SS2023 Tracking Olympiad course at FAU, we had to propose a solution in groups of three to gain hands-on experience as AI developers. The task involved tracking the heads of all the HexBugs visible in a 10-second video, under various background conditions while maintaining a static background.

A set of videos containing the ground-truth positional information was provided at the beginning of the course. The team's score was evaluated by applying the tracking algorithm to previously unseen or withheld videos at the end of the course and we achieved the second place between 7 groups.


<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Pipeline of our framework for HexBug head tracking.
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_2.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Pipeline of our framework for HexBug head tracking.
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_2a.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Pipeline of our framework for HexBug head tracking.
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_3.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Pipeline of our framework for HexBug head tracking.
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project2/APC_4.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Pipeline of our framework for HexBug head tracking.
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0"> <!-- Increase column width -->
        {% include figure.liquid path="assets/img/project2/APC_5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Pipeline of our framework for HexBug head tracking.
</div>





Our solution is composed of four steps:

1. **HexBug detection** (YOLO-NAS) followed by box filtering (ViBe) based on background subtraction and Non-Maximum Supression (NMS)
2. **Object tracking** and box interpolation (CLIP-ViT + K-means)
3. **HexBug-agnostic segmentation** (FastSAM)
4. **Head position estimation** based on an ellipsoid representation of the HexBugs



