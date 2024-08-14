# Faster R-CNN

## Background

### What is object detection? 

Object detection is detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. Well-researched domains of object detection include face detection. Detection often refers to creating a bounding box for an object and segmentation refers to pixel-level classification.

# RCNN, Fast RCNN, and Faster RCNN

RCNN, Fast RCNN, and Faster RCNN are successive advancements in the field of object detection in computer vision. Each represents a significant step in improving the speed and accuracy of detecting objects in images.

## RCNN (Regions with Convolutional Neural Networks)

**RCNN** was introduced by Ross Girshick et al. in 2014. It stands for "Regions with Convolutional Neural Networks." The process works as follows:
1. **Region Proposal**: RCNN uses a selective search algorithm to propose potential regions (also known as region proposals) in the image that may contain objects (about 2000).
2. **Feature Extraction**: Each region proposal is warped into a fixed size and then fed into a CNN (like AlexNet) to extract a feature vector.
3. **Classification**: These feature vectors are then passed through a set of class-specific linear SVMs to classify the objects and background.
4. **Bounding Box Regression**: A bounding box regressor is used to refine the coordinates of the proposed regions.

The main drawbacks of RCNN were its computational inefficiency and the slow speed due to redundant feature extraction for overlapping regions.

Non-maximum supression, intersection over union (iou).

If the bounding box overlaps more than 50% with the ground truth label, we consider it the same class as the true label.

## Fast RCNN

**Fast RCNN** was introduced by Ross Girshick in 2015 to address the inefficiencies of RCNN. The improvements include:
1. **Single CNN**: Instead of feeding each region proposal separately into a CNN, Fast RCNN processes the entire image with a single forward pass through a CNN to generate a convolutional feature map.
2. **Region of Interest (RoI) Pooling**: Regions of Interest (RoIs) are then extracted from this feature map and reshaped to a fixed size using RoI pooling, which allows the classifier to work with variable-sized inputs.
3. **Joint Training**: The feature vectors obtained from RoI pooling are used simultaneously for classification and bounding box regression, improving speed and accuracy.

Fast RCNN significantly reduced the computational cost compared to RCNN by sharing the computation of the convolutional layers across all proposals.

## Faster RCNN

**Faster RCNN** was introduced by Shaoqing Ren et al. in 2015. It built on the advances of Fast RCNN with the following key innovation:
1. **Region Proposal Network (RPN)**: Faster RCNN replaces the selective search algorithm with a Region Proposal Network (RPN) that is integrated into the CNN architecture. The RPN shares convolutional features with the detection network, making it highly efficient.
2. **Unified Network**: The RPN generates region proposals and also produces feature maps which are fed directly into the RoI pooling layer, reducing redundancy and improving speed.

The integration of RPN with Fast RCNN's architecture allows Faster RCNN to generate region proposals and detect objects in a single unified network, significantly improving the detection speed without sacrificing accuracy.

## Summary of Improvements

1. **RCNN**: Introduced region proposals, but was computationally expensive due to redundant CNN computations.
2. **Fast RCNN**: Reduced redundancy by sharing CNN computations for the entire image and introduced RoI pooling for efficient feature extraction.
3. **Faster RCNN**: Integrated region proposal generation with the detection network using RPN, significantly improving speed and efficiency.

These advancements have made Faster RCNN one of the most widely used frameworks for object detection tasks, providing a balance between accuracy and computational efficiency.