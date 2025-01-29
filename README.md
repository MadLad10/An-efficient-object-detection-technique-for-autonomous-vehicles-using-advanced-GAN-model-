# An efficient object detection technique for autonomous vehicles using advanced GAN model

# Overview
This project, developed for CSE463 (Computer Vision: Fundamentals and Algorithms), presents an innovative approach to autonomous vehicle object detection through the integration of advanced generative adversarial networks (GAN) and real-time detection systems. This enhances autonomous vehicle reliability by training detection systems for extreme weather conditions.
# Core Components
### 1. Advanced GAN Implementation (Pix2Pix)


- Generates realistic adverse weather conditions:

- Heavy snow scenarios
- Rain and fog variations
- Low visibility conditions


### 2. Weather-Aware Object Detection (YOLOv5)

  - Enhanced detection in challenging weather scenarios
  - Maintains accuracy during snow, rain, and fog. 

# System Requirements
Prerequisites
Development environment requirements:

 - Python 3.x
 - TensorFlow/TensorFlow-GPU
 - OpenCV
 - PIL (Python Imaging Library)
# Implementation Guide
Synthetic Data Generation Phase
Generate diverse traffic scenarios:

1. Access pix2pix2.ipynb via Jupyter Notebook
2. Train model on autonomous vehicle dataset
3. Generate synthetic traffic scenarios

# Configuration Parameters
Autonomous vehicle-specific settings:

- generated_images_directory: Location of synthetic scenarios
- detection_output_directory: Detection results storage
- model_path: Custom YOLO model path (default: yolov5s)
- confidence_threshold: Detection confidence setting (default: 0.4)






