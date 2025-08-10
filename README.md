# Video Action Recognition

- [x] move the reference baseline codebase to another branch (```dev-SLAM```)
- [x] initiate a documentation, README file
- [ ] reference training script using torch and torch lightning

## Project Overview

This is a video action recognition project for surgical procedures, specifically focused on analyzing laparoscopic surgery clips from the SLAM dataset. The project implements deep learning models to classify surgical actions from video sequences.

#### Reference paper:  
Ye, Z., Zhou, R., Deng, Z. et al. A Comprehensive Video Dataset for Surgical Laparoscopic Action Analysis. Sci Data 12, 862 (2025). https://doi.org/10.1038/s41597-025-05093-7

#### Author implementation (Github repository):  
https://github.com/yezizi1022/SLAM-Vivit_Cls

> In the author implementation, the python requirement files (dependencies) and some other stuff are a bit unorganized. Here, I forked their version in the ```dev-SLAM```branch and I keep my revised version in the ```main```branch. The reason why I want to replicate and make them in a torch/torch-lightning codebase is to try out differentr video recognition backbones and compare their performance on surgical action recognition.

### Dataset Structure
- **SLAM dataset**: Located in `data/SLAM/` with surgical video clips
- **Actions classified**: UseClipper, HookCut, PanoView, Suction, AbdominalEntry, Needle, LocPanoView
- **Video format**: MP4 clips segmented into parts
- **CSV files**: train.csv, val.csv, test.csv for data splits

### Key Architecture Components

**Dataset Pipeline (`datasets/clips.py`)**:
- `ClipsDataset`: PyTorch dataset class that processes video clips
- Uses PyAV library for video decoding
- Fixed temporal sampling (16 frames) and spatial resolution (768x768)
- Includes data augmentation for training (horizontal flip, rotation, color jitter, affine transforms)

**Data Processing**:
- Video frames extracted using PyAV (`av` library)
- Images resized to 768x768 resolution
- Temporal sequences normalized to 16 frames (pad/truncate as needed)
- Label mapping from action names to integers (0-6)

## Environment Setup

**Conda Environment**: Use `environment.yaml` to create the environment  

```bash
conda env create -f environment.yaml
conda activate surgical-vision
```

**Key Dependencies**:
- Python 3.12
- PyTorch 2.7.1 + TorchVision 0.22.1
- OpenCV 4.11.0.86
- PyAV 15.0.0 for video processing
- TIMM 1.0.19 for vision models
- MLflow, Wandb, TensorBoard for experiment tracking

## Common Commands

**Environment setup**:
```bash
conda env create -f environment.yaml
conda activate surgical-vision
```

**Run training**: 
```bash
python train.py SLAM-Vivit-Cls.yaml --mode=train
```

**Training logs**: Check `logs/` directory for training outputs

**Evaluate the given trained snapshot on test data:**  
```bash
python evaluate.py --config config/SLAM-Vivit-Cls-eval.yaml --checkpoint path/to/model.ckpt 
```

## Development Notes

- Video clips are preprocessed and standardized to 16 frames at 768x768 resolution
- The project structure suggests model implementations should go in `models/` directory
- Utility functions should be placed in `utils/` directory
- Configuration files should be stored in `config/` directory
- All video data paths are relative to `data/SLAM/videos/`
- The codebase uses modern PyTorch patterns with DataLoader for efficient batch processing

## Data Format

**CSV Structure**: Each row contains clip information with columns for clip path and action label

**Video Processing**: Clips are automatically resized and temporally sampled during loading

**Label Encoding**: Action names are mapped to integer labels via `label_dict` in `datasets/clips.py`