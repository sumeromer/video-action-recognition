#!/usr/bin/env python3
"""
Training script for the Hydra-enabled ClipsDataset with ViViT Lightning model.

This script demonstrates how to train the ViViT model using PyTorch Lightning with Hydra 
configuration management for the SLAM surgical video action recognition dataset.

Usage:
    python train.py config_file_name [--mode=MODE]
    
    Example:
    python train.py SLAM-Vivit-Cls.yaml --mode=train
    python train.py SLAM-Vivit-Cls.yaml --mode=test
    python train.py SLAM-Vivit-Cls.yaml  # defaults to train mode
    CUDA_VISIBLE_DEVICES=1 python train.py SLAM-Vivit-Cls.yaml --mode=train
"""

import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Tuple
import lightning as L
from pathlib import Path

from datasets import ClipsDataset, get_input_size_from_dataset
from models.VivitLightning import VivitLightningModule, create_lightning_trainer, get_model_summary


def create_data_loaders(config: DictConfig) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders from configuration.
    
    Args:
        config: Hydra configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("=" * 50)
    print("CREATING DATASETS AND DATA LOADERS")
    print("=" * 50)
    
    # Create datasets for train and validation splits
    print("Creating training dataset...")
    train_dataset = ClipsDataset(config, split='train')
    print(f"Training dataset created with {len(train_dataset)} samples")
    
    print("Creating validation dataset...")
    val_dataset = ClipsDataset(config, split='val')
    print(f"Validation dataset created with {len(val_dataset)} samples")
    
    # Extract data loader configuration
    dl_config = config.data_loader
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dl_config.batch_size,
        shuffle=dl_config.shuffle_train,
        num_workers=dl_config.num_workers,
        pin_memory=dl_config.pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=dl_config.batch_size,
        shuffle=dl_config.shuffle_val,
        num_workers=dl_config.num_workers,
        pin_memory=dl_config.pin_memory
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader


def test_dataset_functionality(config: DictConfig, train_dataset: ClipsDataset, val_dataset: ClipsDataset):
    """
    Test various dataset functionalities and print information.
    
    Args:
        config: Hydra configuration object
        train_dataset: Training dataset instance
        val_dataset: Validation dataset instance
    """
    print("=" * 50)
    print("DATASET INFORMATION AND TESTING")
    print("=" * 50)
    
    # Print dataset info
    train_info = train_dataset.get_dataset_info()
    val_info = val_dataset.get_dataset_info()
    
    print("Training Dataset Info:")
    for key, value in train_info.items():
        if key != 'config':  # Skip printing the full config
            print(f"  {key}: {value}")
    
    print("\nValidation Dataset Info:")
    for key, value in val_info.items():
        if key != 'config':  # Skip printing the full config
            print(f"  {key}: {value}")
    
    # Test loading a single sample
    print(f"\nTesting single sample loading...")
    frames, label = train_dataset[0]
    print(f"Sample 0 - Frames shape: {frames.shape}, Label: {label}")
    
    # Get class names
    class_names = train_dataset.get_class_names()
    print(f"Action for label {label}: {class_names[label]}")
    
    # Print all class mappings
    print("\nClass mappings:")
    for label_id, action_name in class_names.items():
        print(f"  {label_id}: {action_name}")


def test_data_loader_iteration(train_loader: torch.utils.data.DataLoader, 
                             val_loader: torch.utils.data.DataLoader):
    """
    Test data loader iteration and print batch information.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    print("=" * 50)
    print("DATA LOADER TESTING")
    print("=" * 50)
    
    # Test training loader
    print("Testing training data loader...")
    train_batch_frames, train_batch_labels = next(iter(train_loader))
    print(f"Train batch - Frames: {train_batch_frames.shape}, Labels: {train_batch_labels.shape}")
    print(f"Train batch labels: {train_batch_labels.tolist()}")
    
    # Test validation loader  
    print("Testing validation data loader...")
    val_batch_frames, val_batch_labels = next(iter(val_loader))
    print(f"Val batch - Frames: {val_batch_frames.shape}, Labels: {val_batch_labels.shape}")
    print(f"Val batch labels: {val_batch_labels.tolist()}")
    
    # Print tensor properties
    print(f"\nTensor properties:")
    print(f"  Data type: {train_batch_frames.dtype}")
    print(f"  Device: {train_batch_frames.device}")
    print(f"  Value range: [{train_batch_frames.min():.3f}, {train_batch_frames.max():.3f}]")


def print_configuration(config: DictConfig):
    """
    Print the loaded configuration in a readable format.
    
    Args:
        config: Hydra configuration object
    """
    print("=" * 50)
    print("LOADED CONFIGURATION")
    print("=" * 50)
    print(OmegaConf.to_yaml(config))


def train_model(config: DictConfig) -> None:
    """
    Train the ViViT model using PyTorch Lightning.
    
    Args:
        config: Hydra configuration object
    """
    print("=" * 60)
    print("STARTING VIVIT MODEL TRAINING")
    print("=" * 60)
    
    # Print configuration
    print_configuration(config)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ClipsDataset(config, split='train')
    val_dataset = ClipsDataset(config, split='val')
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Number of classes: {len(train_dataset.label_dict)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Get class information
    class_names = train_dataset.get_class_names()
    num_classes = len(class_names)
    
    # Create Lightning module
    print("Creating ViViT Lightning module...")
    model = VivitLightningModule(
        config=config,
        num_classes=num_classes,
        class_names=class_names
    )
    
    # Print model summary
    input_shape = get_input_size_from_dataset(train_loader)
    model_summary = get_model_summary(model, input_shape)
    print(model_summary)
    
    # Create trainer
    print("Creating Lightning trainer...")
    trainer = create_lightning_trainer(config, log_dir="logs/SLAM-Vivit_Cls2")
    
    # Start training
    print("Starting training...")
    print("=" * 60)
    
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Print training completion
    print("=" * 60)
    print("TRAINING COMPLETED!")
    print(f"Best validation accuracy: {model.best_val_acc:.4f}")
    print(f"Best validation loss: {model.best_val_loss:.4f}")
    print("=" * 60)
    
    # Save final model
    if hasattr(trainer.logger, 'log_dir') and trainer.logger.log_dir:
        # TensorBoard logger
        run_dir = Path(trainer.logger.log_dir)
        final_model_path = run_dir / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        print(f"All logs and checkpoints saved to: {run_dir}")
    else:
        # W&B logger or other loggers
        final_model_path = Path(config.logging.log_dir) / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        if hasattr(trainer.logger, 'experiment'):
            print(f"Training logged to W&B project: {config.logging.wandb.project}")
            print(f"W&B run URL: https://wandb.ai/{config.logging.wandb.entity}/{config.logging.wandb.project}")
        
        # Also save as W&B artifact if using wandb
        if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'log_artifact'):
            try:
                import wandb
                artifact = wandb.Artifact("final_model", type="model")
                artifact.add_file(str(final_model_path))
                trainer.logger.experiment.log_artifact(artifact)
                print("Model saved as W&B artifact: final_model")
            except Exception as e:
                print(f"Could not save model as W&B artifact: {str(e)}")


def test_model(config: DictConfig, checkpoint_path: str = None) -> None:
    """
    Test the trained ViViT model.
    
    Args:
        config: Hydra configuration object
        checkpoint_path: Path to model checkpoint
    """
    print("=" * 60)
    print("TESTING VIVIT MODEL")
    print("=" * 60)
    
    # Create test dataset
    try:
        test_dataset = ClipsDataset(config, split='test')
        print(f"Test dataset: {len(test_dataset)} samples")
    except:
        print("No test dataset found, using validation dataset for testing...")
        test_dataset = ClipsDataset(config, split='val')
        print(f"Using validation dataset: {len(test_dataset)} samples")
    
    # Create test data loader
    dl_config = config.data_loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=dl_config.batch_size,
        shuffle=False,
        num_workers=dl_config.num_workers,
        pin_memory=dl_config.pin_memory
    )
    
    # Load model
    class_names = test_dataset.get_class_names()
    num_classes = len(class_names)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = VivitLightningModule.load_from_checkpoint(
            checkpoint_path,
            config=config,
            num_classes=num_classes,
            class_names=class_names
        )
    else:
        print("No checkpoint provided, creating new model for testing...")
        model = VivitLightningModule(
            config=config,
            num_classes=num_classes,
            class_names=class_names
        )
    
    # Create trainer for testing
    trainer = create_lightning_trainer(config, log_dir="logs/SLAM-Vivit_Cls2")
    
    # Run test
    print("Running model evaluation...")
    trainer.test(model=model, dataloaders=test_loader)
    
    print("=" * 60)
    print("TESTING COMPLETED!")
    print("=" * 60)


def run_dataset_tests(config: DictConfig) -> None:
    """
    Run comprehensive dataset tests (original functionality).
    
    Args:
        config: Hydra configuration object
    """
    print("=" * 60)
    print("RUNNING DATASET TESTS")
    print("=" * 60)
    
    # Print configuration
    print_configuration(config)
    
    # Create datasets
    train_dataset = ClipsDataset(config, split='train')
    val_dataset = ClipsDataset(config, split='val')
    
    # Test dataset functionality
    test_dataset_functionality(config, train_dataset, val_dataset)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Test data loader iteration
    test_data_loader_iteration(train_loader, val_loader)
    
    # Test the legacy function
    print("=" * 50)
    print("LEGACY FUNCTION TESTING")
    print("=" * 50)
    
    input_size1 = get_input_size_from_dataset(train_loader)
    input_size2 = get_input_size_from_dataset(val_loader)
    print(f"Train input size: {input_size1}")
    print(f"Val input size: {input_size2}")
    
    print("\n" + "=" * 60)
    print("ALL DATASET TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def main(config: DictConfig) -> None:
    """
    Main function that orchestrates training, testing, or dataset validation.
    
    Args:
        config: Hydra configuration object
    """
    # Parse mode from command line arguments
    mode = "train"  # default mode
    checkpoint_path = None
    
    # Look for mode in sys.argv
    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]
        elif arg.startswith("--checkpoint="):
            checkpoint_path = arg.split("=")[1]
    
    print(f"Running in {mode.upper()} mode")
    
    if mode == "train":
        train_model(config)
    elif mode == "test":
        test_model(config, checkpoint_path)
    elif mode == "dataset_test":
        run_dataset_tests(config)
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: train, test, dataset_test")
        sys.exit(1)


if __name__ == "__main__":
    # Parse arguments manually
    if len(sys.argv) < 2:
        print("Usage: python scratch.py <config_file_name> [--mode=MODE] [--checkpoint=PATH]")
        print("Example: python scratch.py SLAM-Vivit-Cls --mode=train")
        print("Example: python scratch.py SLAM-Vivit-Cls --mode=test --checkpoint=path/to/model.ckpt")
        print("Example: python scratch.py SLAM-Vivit-Cls --mode=dataset_test")
        sys.exit(1)
    
    config_name = sys.argv[1]
    # Remove .yaml extension if provided
    if config_name.endswith('.yaml'):
        config_name = config_name[:-5]
    
    print(f"Loading configuration: {config_name}")
    
    # Filter out our custom arguments and keep only config name for Hydra
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]  # Keep only script name for Hydra
    
    # Use Hydra to load and run with the specified config
    @hydra.main(version_base=None, config_path="config", config_name=config_name)
    def hydra_main(config: DictConfig) -> None:
        # Restore original argv for argument parsing in main()
        sys.argv = original_argv
        main(config)
    
    hydra_main()