---
layout: page
title: HexBug Head Tracking
description: Framework to track a freely and randomly moving object (“HexBug”) head from a video, within the context of the Tracking Olympiad (TRACO) course.
img: assets/img/project1/results_test001.gif
importance: 1
category: study
related_publications: false
---
For the SS2023 Tracking Olympiad course at FAU, we had to propose a solution in groups of three to gain hands-on experience as AI developers. The task involved tracking the heads of all the HexBugs visible in a 10-second video, under various background conditions while maintaining a static background.

A set of videos containing the ground-truth positional information was provided at the beginning of the course. The team's score was evaluated by applying the tracking algorithm to previously unseen or withheld videos at the end of the course and we achieved the second place between 7 groups.
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project1/results_test001.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project1/results_test0021.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project1/results_video_8a.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Examples of our proposed tracking solution on three different previously unseen videos. The videos on the left display the detected head axis from the center of the HexBug, the videos in the center show the final predicted head position indicated by a red circle, and the videos on the right illustrate the initially detected HexBugs using YOLO-NAS, Fast-SAM and CLIP models.
</div>

Our solution is composed of four steps:

1. **HexBug detection** (YOLO-NAS) followed by box filtering (ViBe) based on background subtraction and Non-Maximum Supression (NMS)
2. **Object tracking** and box interpolation (CLIP-ViT + K-means)
3. **HexBug-agnostic segmentation** (FastSAM)
4. **Head position estimation** based on an ellipsoid representation of the HexBugs

The pipeline is illustrated in the following image.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project1/presentation.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Pipeline of our framework for HexBug head tracking.
</div>

We show two additional examples of our solution:
<div class="row justify-content-sm-center">
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project1/results_video_5a.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/project1/results_test002.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

