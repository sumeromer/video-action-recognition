"""
Models package for video action recognition.

This package contains the ViViT (Video Vision Transformer) model and its PyTorch Lightning wrapper
for training surgical video action recognition on the SLAM dataset.
"""

from .Vivit import ViViT
from .VivitLightning import VivitLightningModule, create_lightning_trainer, get_model_summary

__all__ = [
    'ViViT',
    'VivitLightningModule', 
    'create_lightning_trainer',
    'get_model_summary'
]