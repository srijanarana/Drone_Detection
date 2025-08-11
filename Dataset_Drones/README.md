<h1>Structure:</h1>

First Download Whole Drone Dataset From - ""https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset/""

   **Dataset/**

              Train/

                    Images/

                    Labels/

              Valid/

                    Images/

                    Labels/

- **train/images and valid/images**: Contain the training and validation images, respectively.
- **train/labels and valid/labels**: Contain the corresponding annotation files for each image.
- The Dataset Should be in this Format so it could easily be accessed for Training through data.yaml

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
