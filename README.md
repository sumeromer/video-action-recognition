

# Laparoscopic surgical action classification with ViViT Model

## Overview
This repository contains scripts for training and testing a **ViViT** (Video Vision Transformer) model for laparoscopic surgical action classification. The project aims to advance automated analysis of surgical videos, providing a tool for understanding and categorizing surgical actions.

---

## Dataset: **SLAM** 
The training and inference scripts are designed to work with **SLAM: A Comprehensive Video Dataset for Surgical Laparoscopic Action Analysis**. SLAM is a publicly available dataset that includes laparoscopic surgical videos annotated for various surgical actions.

### Dataset Details:
1. **Type:** Laparoscopic surgical action classification.
2. **Annotations:** Each video clip is labeled with one of several predefined surgical actions.
3. **Structure:** The dataset consists of training, validation, and test splits.

**Publication:** [Link to the SLAM dataset (to be added upon release)]  

### Looped_clips.csv description: 

1. the csv is designed to indicate which clips adopt the loop complement strategy, and the dataset users can choose whether to adopt the above clips according to their own needs. 

2. the csv includes the clip path `loop_path`, the dataset division `phase`, part id `part_id`, and the frame number at which the loop is started `loop_start_frame`;

3. The loop part is the interval `[loop_start_frame, 30]`.
---

## Environment
The environment setup for these scripts is the same as required for the ViViT model. Ensure you have Python 3.8+ and all necessary dependencies installed.

---

## Dataset Preparation
1. Download the SLAM dataset from the official [dataset repository](#) (link will be added upon release).

---

## Training
To train the model on the SLAM dataset:

1. Ensure your configurations in `train.py` are set correctly. Key configurations include:
   - **Learning Rate (`lr`):** Set to `7e-5` by default.
   - **Time Size (`time_size`):** This defines the number of frames (input length) for each video clip. Default is `16`.
   - **Height and Width:** Image resolution is set to `768x768` pixels.

2. Run the training script using the following command:
```python
python train.py
```

---

## Inference
To evaluate the model on the test dataset:

---

1. Ensure the test dataset CSV is prepared.
2. Update the `model_path` in `inference.py` to point to the trained model weight file. For example:
   ```python
   model_path = "/root/SLAM-Vivit_Cls/weights/best_model.pkl"
2. Run the inference script using the command:
   ```python
   python inference.py
   ```
 
---
  
# Acknowledgments
- **ViViT Base Code**: This repository is inspired by the ViViT model implementation. The base code and some logic for the model were adapted from the [ViViT PyTorch repository](https://github.com/rishikksh20/ViViT-pytorch/tree/master).
- **ViViT Paper**: The ViViT model architecture and concept are based on the paper [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691).
