<h1>Drone Detection Using Yolov5 Model</h1>

## Overview

This repository demonstrates drone (UAV) detection using **YOLOv5**, a widely-used real-time object detection model from Ultralytics.  
YOLOv5 is known for its speed, accuracy, and ease of use, making it ideal for UAV/drone detection tasks in computer vision.  
All results, trained models (`best.pt` and others), and sample predicted images are included for reference.

---

## Installation

Install the Ultralytics package, which supports YOLOv5 (by cloning the yolov5 repo from Ultralytics):

```
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt

```

---

## Dataset Format

YOLOv5 uses the standard YOLO annotation format:

- **One `.txt` file per image**: Each line represents one object as  
  `<class_id> <x_center> <y_center> <width> <height>`  
  All coordinates are normalized (values between 0 and 1).
- **`data.yaml`**:  
  Specifies dataset paths and class names.

**Example `data.yaml`:**
```
 path:  # Dataset root directory
 train: train/images
 val: valid/images
 names:
 - drone
 nc: 1

```
---

## Training

Train YOLOv5 on custom drone dataset with (choosing the weights according to the size of the dataset) 
 
```
!python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt
```
---

## Monitoring training processes
```
from utils.plots import plot_results
plot_results(save_dir='runs/train/exp')
```
---

## Validation Prcess
```
!python val.py --weights runs/train/exp/weights/best.pt --data data.yaml
```
---
## Using Trained Model to detect Drones in Images/Videos
```
!python detect.py --weights runs/train/exp/weights/best.pt --source #PathtoImage  
```

---

## Results and Included Files

- **All results and evaluation metrics** are included in the `test results/` folder and all necessary requirements are listed in `requirements.txt`.
- **Trained model weights** (`best.pt` and others) are provided for download provided in the `weights/` folder.
- **Sample predicted video** are included for reference, so you can see how the model performs on real drone video tracking provided in the `predict/` folder.

---

## Resources

- [Ultralytics YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5/)
- [Drone Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset/)


