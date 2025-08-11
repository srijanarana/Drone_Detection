<h1>﻿Drone Visual Detection using Deep Learning Models</h1>

**INTRODUCTION:**

The rapid proliferation of unmanned aerial vehicles (UAVs), commonly known as drones, has revolutionized numerous industries such as surveillance, agriculture, infrastructure monitoring, and public safety. While drones offer significant benefits in terms of efficiency and data collection, their widespread use also presents new challenges related to security, privacy, and airspace management. As a result, there is a growing need for reliable and automated drone detection systems.

Recent advancements in computer vision and deep learning have enabled the development of highly effective object detection algorithms. These technologies allow machines to automatically analyze visual data and accurately identify objects, including drones, in complex environments. Among the most successful approaches are deep learning models that leverage large datasets to learn robust feature representations for detection and classification tasks.

In this project, we focus on building an automated drone detection system using deep learning-based object detection models. Specifically, we utilize and compare two state-of-the-art algorithms: **Yolov5** and **Yolov11.** Both models are part of the "You Only Look Once" (YOLO) family, which is renowned for real-time detection speed and high accuracy. YOLOv5 is widely adopted for its balance of performance and efficiency, while YOLOv11 introduces architectural improvements that further enhance detection accuracy and versatility. By training and evaluating these models on a custom UAV image dataset, this project aims to assess their effectiveness for drone detection and provide insights into their practical deployment.



**SOFTWARE USED:**

The following software tools and libraries were utilized for dataset preparation, model training, and evaluation:

1. **Python 3.8+**

The primary programming language for all scripts and model training.

2. **Ultralytics YOLO (YOLOv5 and YOLOv11)**

The official Ultralytics YOLO package was used for both YOLOv5 and YOLOv11 model training, validation, and inference. This package provides a unified interface for state-of-the-art object detection workflows.

3. **Google Colab**

All experiments were conducted on Google Colab, leveraging its free GPU resources for faster training and testing.

4. **Supporting Libraries:**

     1. NumPy and Pandas for data manipulation and analysis.
     2. OpenCV for image processing and visualization.
     3. Matplotlib for plotting training curves and evaluation metrics.

5. **Annotation Tools:**

Roboflow and LabelImg were used for creating and verifying bounding box annotations, ensuring high-quality labels for each image.

This software setup is consistent with current standards in computer vision research and enables reproducible, scalable experiments for UAV detection.

**DATASET STRUCTURE:**

For this project, a custom UAV (drone) image dataset was used to train and evaluate deep learning models. The dataset consists of 1,012 images, each annotated with the location of drones using the standard YOLO format. Each annotation file contains the class label and normalized coordinates for the bounding box surrounding the detected drone.

**Structure:** 

   **Dataset/**

              Train/

                    Images/

                    Labels/

              Valid/

                    Images/

                    Labels/

- **train/images and valid/images**: Contain the training and validation images, respectively.
- **train/labels and valid/labels**: Contain the corresponding annotation files for each image.

**Annotation Format :**\
Each label file contains one or more lines, each in the format:\
<class\_id> <x\_center> <y\_center> <width> <height>\
All values are normalized between 0 and 1, following the YOLO convention.

**Dataset Split :**\
The dataset was divided into 75% for training and 25% for validation to ensure robust evaluation and to prevent overfitting. This structure is consistent with best practices in drone detection research.

**Data.yaml:**

The dataset includes a data.yaml file that specifies the paths to the training and validation images.

Data.yaml Includes:

![Image](https://github.com/user-attachments/assets/72b5cfdf-2120-433f-b007-5ee970ee450c)





**DEEP LEARNING DETECTING MODELS:** 

1. **Yolov5**

YOLOv5, released by Ultralytics in 2020, has become one of the most widely adopted object detection models in both research and industry. It is celebrated for its strong balance of speed and accuracy, user-friendly interface, and robust PyTorch implementation. The architecture is built on a CSPDarknet53 backbone with a PANet neck, and it uses an anchor-based detection head. This allows YOLOv5 to effectively aggregate features at multiple scales, making it highly adaptable for detecting objects of various sizes—including drones.

Key strengths of YOLOv5 include:

- **Exceptional inference speed:** Optimized for real-time applications, even on resource-constrained devices.
- **Ease of use:** Simple API, extensive documentation, and a mature ecosystem with a large community.
- **Scalability:** Offers multiple model sizes (nano to extra-large) to balance speed and accuracy for different hardware.
- **Proven reliability:** Years of active development and widespread deployment in real-world systems.

**Results (Yolov5)** 

For the first phase of this project, we used the YOLOv5 model to train and evaluate drone detection on my dataset. To optimize the training process, we implemented the following key parameters:

**Epochs:** We set the number of training epochs to [50], which determines how many times the model passes through the entire training dataset.

**Batch Size:** The batch size was set to [16], controlling how many images are processed at once during each training step. Choosing the largest batch size that fits in GPU memory is recommended for efficiency.

**Image Size:** We used an image size of [640x640] pixels, as higher image resolutions can improve detection accuracy but require more computational resources.

![Image](https://github.com/user-attachments/assets/311e91e8-09ca-41b7-83b7-ba5af36212e0)

The training was conducted using the Ultralytics YOLOv5 framework, which provides a user-friendly interface for custom object detection tasks. We initialized the model with pretrained weights (yolov5s.pt), which helps accelerate convergence and improve results, especially when working with smaller datasets.

During training, all outputs including model weights, logs, and visualizations were saved in the runs/train/ directory, allowing for easy monitoring and later analysis.

After training, YOLOv5 achieved the following performance on the validation set:

|Precision:|Recall:|mAP@0.5:|mAP@0.5:0.95:|
| :- | :- | :- | :- |
|0\.89|0\.85|0\.90|0\.50|</br>

<img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/c5e5f86a-2374-449f-8334-088a6297c45a" />








This plots shows the performance metrics of yolov5 model, after training:

**train/box\_loss:** Measures the error in predicting bounding box coordinates during training. It decreases from ~0.08 to ~0.02, showing improved accuracy in box localization.

**train/cls\_loss:** Represents the classification error, Remains 0 (only one class ‘Drone”).

**train/obj\_loss:** Represents the error in objectness prediction (likelihood of an object being present). It drops from ~0.03 to ~0.015, indicating better object detection confidence.

**val/box\_loss:** Validation set's bounding box error, decreasing from ~0.07 to ~0.03. This reflects good generalization to unseen data.

**val/cls\_loss:** Represents the classification error, Remains 0 (only one class ‘Drone”).

**val/obj\_loss:** Validation objectness error, reducing from ~0.014 to ~0.008, showing consistent improvement.

**metrics/Precision:** The proportion of correct positive predictions, rising from ~0.4 to ~1.0. This shows the model is increasingly accurate in its detections.

**metrics/Recall**: The ability to detect all relevant objects, increasing from ~0.3 to ~0.9, indicating better coverage of true positives.

**metrics/mAP50**: Mean Average Precision at an IoU threshold of 0.5, rising from ~0.2 to ~0.8. This measures detection quality at a moderate overlap standard.

**metrics/mAP50-95:** Mean Average Precision across IoU thresholds from 0.5 to 0.95, increasing from ~0.1 to ~0.5. This stricter metric shows solid but less perfect performance.

Overall, the decreasing losses and increasing metrics suggest the model is learning effectively, with good generalization to validation data. 

All detailed training logs, per-epoch metrics, and additional result tables are included in the Appendix at the end of this and also the in the result folder(yolov5/results).

2. **Yolov11**

YOLOv11 represents the next evolutionary step in the YOLO family, also developed by Ultralytics. It introduces several architectural innovations, most notably an anchor-free detection head, which improves accuracy and reduces the need for manual anchor tuning. YOLOv11 achieves state-of-the-art results by offering higher mean average precision (mAP) across all model sizes compared to YOLOv5, while also being more computationally efficient.

Key strengths of YOLOv11 include:

- **Superior accuracy:** Consistently achieves higher mAP scores than YOLOv5 models of similar size (for example, YOLOv11s achieves 47.0 mAP vs. YOLOv5s at 37.4 mAP).
- **Greater versatility:** Enhanced performance across a wide range of object sizes and types.
- **Improved efficiency:** Faster inference on many devices and smaller model sizes for similar or better accuracy.
- **Modern architecture:** Anchor-free design and advanced feature extraction modules.

**Yolov11 (Results):** 

For the second phase of this project, We implemented the YOLOv11 model to train and evaluate drone detection on the same dataset. To ensure a fair comparison with YOLOv5, We used similar training parameters where possible, but extended the number of epochs for YOLOv11 to allow for potentially improved convergence and performance.

**Epochs:** We trained YOLOv11 for 100 epochs, providing the model with more opportunities to learn from the training data and optimize its weights.

**Batch Size:** The batch size was set to [16], balancing computational efficiency and GPU memory usage.

**Image Size:** An image size of [640x640] pixels was used, consistent with the YOLOv5 experiments.

<img width="610" height="79" alt="Image" src="https://github.com/user-attachments/assets/d6980ba6-5235-4abd-be97-02525a061a9f" />


The training was conducted using the official Ultralytics YOLOv11 framework, which features an anchor-free detection head and enhanced feature extraction modules. We initialized the model with pretrained weights (yolov11s.pt) to accelerate convergence and boost accuracy, especially with a moderate-sized dataset.

After training, YOLOv11 achieved the following performance on the validation set:

<img width="555" height="84" alt="Image" src="https://github.com/user-attachments/assets/b57383ee-306b-4e9d-983f-b70a9c7cffe3" /><br/>


<img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/3a9fd786-b9f2-4324-89b8-79a7661f54e7" />










This plots shows the performance metrics of yolov11 model, after training:

**train/box\_loss:** Measures the error in predicting bounding box coordinates during training. The value decreases from ~1.6 to ~0.8, indicating the model is improving its box prediction accuracy.

**train/cls\_loss:** Represents the classification error (e.g., identifying object classes) during training. It drops from ~2.5 to ~0.5, showing better class prediction over time.

**train/dfl\_loss:** Likely related to a specific loss component (e.g., distribution focal loss), decreasing from ~1.8 to ~1.2, suggesting refinement in this aspect of training.

**val/box\_loss:** The validation set's bounding box error, reducing from ~3.0 to ~1.5. This indicates the model generalizes better to unseen data.

**val/cls\_loss:** Validation classification error, dropping sharply from ~80 to near 0. The high initial value might reflect early overfitting or data issues, stabilizing later.

**val/dfl\_loss:** Validation version of the Dfl loss, decreasing from ~7 to ~2, showing improved performance on validation data.

**metrics/precision(B):** The proportion of correct positive predictions, rising from ~0 to ~0.8. This reflects increasing accuracy in detecting true objects.

**metrics/recall(B):** The ability to find all relevant objects, increasing from ~0 to ~0.8, indicating better detection coverage.

**metrics/map50(B):** Mean Average Precision at an IoU threshold of 0.5, rising from ~0 to ~0.8. This measures detection quality at a moderate overlap standard.

**metrics/map50-95(B):** Mean Average Precision across IoU thresholds from 0.5 to 0.95, increasing from ~0 to ~0.4. This is a stricter metric, showing solid but less perfect performance.

Overall, the decreasing losses and increasing metrics suggest the model is learning effectively, with good generalization to validation data. All detailed training logs, per-epoch metrics, and additional result tables are included in the Appendix at the end of this and also the in the result folder(yolov11/results).




**WHY YOLOV11 IS BETTER?:**

YOLOv11 is much better/preferred model for UAV detection for several reasons:

- **Superior Accuracy:** It achieved higher precision, recall, and mAP scores, indicating more reliable detection and localization of drones.
- **Efficiency:** The model is smaller and trains faster, which is valuable for deployment on resource-constrained devices.
- **Modern Architecture:** The anchor-free design allows better detection of small and variably shaped objects, which is important for UAVs.
- **Advanced Loss Function (DFL):** YOLOv11 incorporates Distribution Focal Loss (DFL), which improves bounding box regression accuracy by better modeling the distribution of object locations, leading to more precise detections.
- **Generalization:** Validation results showed less overfitting and more stable learning across epochs.
- **Future-Proof:** YOLOv11 benefits from the latest updates and community support from Ultralytics.

**CONCLUSION:**

Both YOLOv5 and YOLOv11 are effective for UAV detection in images. However, YOLOv11’s superior accuracy, efficiency, and modern features make it the preferred model for this project. It is well-suited for real-time applications and future research in drone detection.

**REFERENCES:**

1. <https://docs.ultralytics.com/yolov5/>
2. Ultralytics Yolov11 Github
3. Custom Dataset From Kaggle.com <https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset/>
4. All training and evaluation results for both YOLOv5 and YOLOv11 are included in this repo. You will find comprehensive plots for precision (P), recall (R), and F1 score, as well as all training and validation loss curves and logs. These visualizations and logs allow for a clear comparison of model performance and training dynamics. Additionally, sample predicted images are provided to showcase how each model peforms on real drone data.

**INSTALLING PROCESS**  

<img width="709" height="506" alt="Image" src="https://github.com/user-attachments/assets/27a4b516-f24e-4cc9-992d-5367d4117b53" /></br>


<h3>Project Created By Srijana Rana , Joel J.</h3>
