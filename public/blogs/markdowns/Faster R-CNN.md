# RCNN, Fast RCNN, Faster RCNN, and YOLO

## What is object detection? 

Object detection is detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. Detection often refers to creating a bounding box for an object and segmentation refers to pixel-level classification. This is what makes the problem challenging: you cannot use a standard convolutional network with a fully connected layer since the length of the output layer is variable. The number of occurrences of the objects of interest is not constant. One may try and select numerous regions within the image and use a CNN to detect one object at a time. But as we all know, this would be extremely computationally expensive.

## RCNN, Fast RCNN, and Faster RCNN

RCNN, Fast RCNN, and Faster RCNN are successive advancements in the field of object detection in computer vision. Each version showed improvements in speed and accuracy of object detection in images.

## RCNN (Regions with Convolutional Neural Networks)

Original Paper: **Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik**. *Rich feature hierarchies for accurate object detection and semantic segmentation*. 2014. [arXiv:1311.2524](https://arxiv.org/abs/1311.2524).

![img](https://miro.medium.com/v2/resize:fit:1050/1*REPHY47zAyzgbNKC6zlvBQ.png)

1. **Region Proposal**: RCNN uses a selective search algorithm to propose potential regions (also known as region proposals) in the image that may contain objects (about 2000).
2. **Feature Extraction**: Each region proposal is warped into a fixed size square and then fed into a CNN to extract a 4096 dimensional feature vector.
3. **Classification**: These feature vectors are then passed through a set of class-specific linear SVMs to classify the object.
4. **Bounding Box Regression**: A bounding box regressor is used to refine the coordinates of the proposed regions.

The selective search algorithm is as follows:

**1. Initial Segmentation:**

- The algorithm starts by over-segmenting the image into a large number of small regions using a segmentation method. This is done using graph-based segmentation techniques that create regions with similar color and texture.

**2. Region Merging:**

- The algorithm merges these small regions into larger ones based on several criteria:

  - Similarity: Regions are merged based on how similar they are in terms of color, texture, size, and shape.

  - Hierarchical Merging: The merging process is hierarchical. Initially, very small segments are merged, and this continues in a hierarchical fashion to form larger regions.

  - Diverse Strategies:

     Selective Search employs various strategies to merge regions, including:

    - Single Scale: Merging regions based on similarity within a single scale.
    - Multi-Scale: Considering multiple scales to capture objects of different sizes.
    - Color Histograms: Using color information to guide region merging.
    - Texture and Size Information: Incorporating texture and size information into the merging process.

**3. Candidate Regions:**

- After merging, the algorithm produces a set of candidate regions (bounding boxes) that are likely to contain objects.

**4. Region Proposal:**

- These candidate regions are then fed into a classifier to determine if they contain objects and to classify the objects.

The main issue with this approach is that 1) it still takes a significant amount of time to train the network as you have to classify ~2000 region proposals per image, 2) inference is also low due to sequential processing of region proposals (~49 seconds per image).

## Fast RCNN

Original Paper: **Ross Girshick**. *Fast R-CNN*. 2015. [arXiv:1504.08083](https://arxiv.org/abs/1504.08083).

![img](https://miro.medium.com/v2/resize:fit:1050/1*0pMP3aY8blSpva5tvWbnKA.png)

1. **Single CNN**: Instead of feeding each region proposal separately into a CNN, Fast RCNN processes the entire image with a single forward pass through a CNN to generate a convolutional feature map.
2. **Region of Interest (RoI) Pooling**: Regions of Interest (RoIs) are then extracted from this feature map and reshaped to a fixed size using RoI pooling which can be fed into a FC layer.
3. **Joint Training**: These RoI feature vectors are used for classification and bounding box regression, improving speed and accuracy.

![img](https://miro.medium.com/v2/resize:fit:1050/1*m2QO_wbUPA05mY2q4v7mjg.png)

Fast RCNN improves on RCNN's speed by eliminating the need to process 2,000 region proposals individually through the convolutional neural network. Instead, Fast RCNN performs the convolution operation just once per image, producing a single feature map from which all region proposals are derived. However, we see that including region proposal significantly increases the test time, which is a bottleneck for Fast RCNN. In other words, having a separate region proposal algorithm still serves as a bottleneck.

## Faster RCNN

Original Paper: **Shaoqing Ren**, **Kaiming He**, **Ross Girshick**, **Jian Sun**. *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. 2016. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497).

![img](https://miro.medium.com/v2/resize:fit:1050/1*pSnVmJCyQIRKHDPt3cfnXA.png)

1. **Region Proposal Network (RPN)**: Faster RCNN replaces the selective search algorithm with a Region Proposal Network (RPN) that is integrated into the CNN architecture. The RPN shares convolutional features with the detection network, making it highly efficient.
2. **Unified Network**: The RPN generates region proposals and also produces feature maps which are fed directly into the RoI pooling layer, reducing redundancy and improving speed.

![img](https://miro.medium.com/v2/resize:fit:1050/1*4gGddZpKeNIPBoVxYECd5w.png)

The integration of RPN with Fast RCNN's architecture allows Faster RCNN to generate region proposals and detect objects in a single unified network, significantly improving the detection speed without sacrificing accuracy.

In summary, RCNN introduced region proposals, but was computationally expensive due to redundant CNN computations. Fast RCNN reduced redundancy by sharing CNN computations for the entire image and introduced RoI pooling for efficient feature extraction. Faster RCNN integrated region proposal generation with the detection network using RPN, significantly improving speed and efficiency.

## You Only Look Once (YOLO)

Original Paper: **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).** *You Only Look Once: Unified, Real-Time Object Detection*. Retrieved from [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

YOLO is probably one of the most well known object detection model proposed. We saw that even after various improvements the Faster RCNN can only process about 5 frames per second. YOLO showed a significant improvement in its inference, showing 45 frames per second and up to 155 frames per second for the faster YOLO model without a significant drop in inference accuracy compared to Faster RCNN. The first YOLO v1 model was proposed back in 2015, and currently as of 2024, YOLO v9 has been proposed.

![img](https://images.velog.io/images/skhim520/post/e3390906-b8f6-454f-adbf-d40a35e407d3/image.png)

1. **Unified Architecture**: YOLO frames object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities. This unified approach allows YOLO to be exceptionally fast, processing images in real-time.

2. **Single Convolutional Network**: The model uses a single convolutional network to predict multiple bounding boxes and class probabilities directly from the full image in one evaluation. This eliminates the need for a separate region proposal network.

3. **Grid-Based Prediction**: YOLO uses the 'cheaper grids' method for speedup, as proposals from the previous RCNN architectures were expensive. YOLO divides the image into an S×S grid. Each grid cell is responsible for predicting two bounding boxes and class probabilities for objects whose centers fall within the cell. This approach helps in localizing objects and assigning them to the correct class.

   ![You Only Look Once. YOLO | 너드팩토리 블로그](https://blog.nerdfactory.ai/assets/images/posts/2021-07-01-You-Only-Look-Once-YOLO/Untitled.png)

4. **Bounding Box Prediction**: Each grid cell predicts two bounding boxes, each with a confidence score representing the likelihood of an object being present and the accuracy of the box. Then, bounding boxes with low confidence scores are removed. The final prediction uses non-maximum suppression algorithm to select the most probable bounding boxes.

### Limitations:

- **Localization Error**: YOLO's grid-based approach can lead to less precise localization of objects compared to methods that use region proposals. Since each grid cell is responsible for multiple bounding boxes, there may be overlaps or inaccuracies in predicting bounding box coordinates.
- **Small Object Detection**: YOLO's performance can degrade with very small objects, as the grid cells might be too large to capture fine details, which affects the accuracy of small object detection. Mathematically, the intersection over union (IoU) calculation can be problematic for small objects due to their limited size relative to the grid cells. When a small object is detected, its predicted bounding box might overlap minimally with the ground truth bounding box, leading to a low IoU score. A low IoU score means that the prediction might not be considered accurate enough, even if the object is detected.

