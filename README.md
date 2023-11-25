# Enhanced Detection of Small Objects in Aerial Imagery: A High-Resolution Neural Network Approach with Amplified Feature Pyramid and Sigmoid Re-weighting

## Overview
In this paper, we address the aforementioned challenges in drone-captured image object detection. Our main contributions are summarized as follows:

**Overcoming Resolution Constraints**: We introduce the Enhanced High-Resolution Processing Module (EHRPM), which effectively processes high-resolution images without changing the input size for the primary network.
This module improves the detailing of features, enabling better detection of small objects and subtle nuances compared to previous methods that relied on low-resolution inputs.

**Mitigating Information Loss in Small Objects**: We propose the Small Object Feature Amplified Feature Pyramid Network (SOFA-FPN), incorporating the Edge Enhancement Module (EEM), the Cross Lateral Connection Module (CLCM), and the Dual Bottom-up Convolution Module (DBCM). This approach focuses on enhancing the edges of objects to combat the loss of information in small entities, highlighting the importance of edge information in detecting smaller objects.

**Introducing a Simple Feature Re-weighting Module (SRM)**: The SRM is designed to emphasize critical information in the feature maps at each scale before they proceed through the prediction head. This self-attention mechanism allows the network to concentrate on the most relevant features, thereby enhancing detection accuracy.

**Designing a Lightweight Network**: We have developed lightweight modules capable of efficiently processing high-resolution images with minimal increase in computational complexity. Our approach facilitates effective object detection while reducing the demand on computational resources.

**Comparative Evaluation with Leading Networks**: We have compared our method with other top-tier networks, including baseline models. Our experimental results show that our approach achieves similar or better performance with a lower number of parameters and reduced computational requirements.  

![alt text](/assets/over_arch.png)


## Requirements
- Pytorch 1.11.1
- Python 3.8

## Installation
Download repository:
```bash
git clone https://github.com/datu0615/EHRPM.git
```

Create conda environment:
```bash
conda create -n EHRPM python=3.8
conda activate EHRPM
cd YOLOX
pip install -r requirements.txt
or
pip install -v -e.

cd..
cd ultralytics
pip install -r requirements.txt
or
pip install -v -e.
```

## Prepare Datasets

VisDrone Dataset  
<https://github.com/VisDrone/VisDrone-Dataset>  
UAVDT Dataset  
<https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5>  
AI-TOD Dataset  
<https://github.com/jwwangchn/AI-TOD>  


## Train (yolov8)

```bash
from ultralytics import YOLO

# Load a model
model = YOLO('newyolov8s.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='VisDrone.yaml', epochs=100, imgsz=1280)
```


## Train (yolox)

```bash
python tools/train.py -f /path/to/your/Exp/file -d 8 -b 64 --fp16 -o
```


## Results
