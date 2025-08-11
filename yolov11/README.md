<h1>Drone Detection Using YOLOv11 Model</h1>

## Overview

This repository demonstrates drone (UAV) detection using **YOLOv11**, the latest real-time object detection model from Ultralytics.  
YOLOv11 brings improved accuracy, efficiency, and advanced features while maintaining compatibility with the classic YOLO annotation format.  
All results, trained models (`best.pt` and others), and sample predicted images are included for reference.

---

## Installation

Install the Ultralytics package, which includes YOLOv11:
```
   - pip install ultralytics
```
---

## Dataset Format

YOLOv11 uses the same annotation format as YOLOv5-v10:

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

Train YOLOv11 on your custom drone dataset with:
```
from ultralytics import YOLO
model = YOLO('yolo11s.pt')
model.train(data='/content/Dataset_Drones/data.yaml', epochs=100, imgsz=640)
```
---
## Monitoring training processes

```
from utils.plots import plot_results
plot_results(save_dir='runs/detect/train')

```
---

## Validation Prcess
```
from ultralytics import YOLO
model = YOLO('/content/runs/detect/train/weights/best.pt')
metrics = model.val(data='/content/Dataset_Drones/data.yaml')
```
---
## Using Trained Model to detect Drones in Images/Videos
```
model.predict(source='PathtoImage', save=True)
```

---

## Results and Included Files

- **All results and evaluation metrics** are included in the `results/` folder.
- **Trained model weights** (`best.pt` and others) are provided for download in the `weights/` folder.
- **Sample predicted images** are included for reference, so you can see how the model performs on real drone images in `predict/` folder.

---

## Resources

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Drone Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset/)

---

