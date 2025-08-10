#!/usr/bin/env python3
"""
Evaluation script for video action recognition model.

This script loads a trained ViViT model and evaluates it on the test set,
computing comprehensive metrics including overall accuracy, balanced accuracy,
F1-score, precision, recall, and per-class accuracies.

Usage:
    python evaluate.py --config config/SLAM-Vivit-Cls-eval.yaml --checkpoint path/to/model.ckpt
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from collections import Counter

import torch
import torch.nn.functional as F
import torch.utils.data
from omegaconf import DictConfig, OmegaConf
import lightning as L
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from datasets.clips import ClipsDataset
from models.VivitLightning import VivitLightningModule


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def create_tta_transforms(config: DictConfig) -> list:
    """
    Create list of test-time augmentation transforms with multiple samples per transform type.
    
    Args:
        config: Configuration object
        
    Returns:
        List of transform compositions for TTA
    """
    tta_transforms = []
    
    # Get TTA config parameters (use defaults if not specified)
    tta_config = getattr(config.evaluation, 'tta', {})
    transform_config = tta_config.get('transforms', {})
    n_samples_per_transform = tta_config.get('n_samples_per_transform', 1)
    
    # 1. Original transform (no augmentation) - only add once
    base_transform = transforms.Compose([transforms.ToTensor()])
    tta_transforms.append(base_transform)
    
    # 2. Horizontal flip - create multiple samples
    if transform_config.get('horizontal_flip', True):
        for _ in range(n_samples_per_transform):
            tta_transforms.append(transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor()
            ]))
    
    # 3. Random rotation variations - create multiple samples with random angles
    rotation_range = list(transform_config.get('rotation_range', [-10, 10]))
    for _ in range(n_samples_per_transform):
        tta_transforms.append(transforms.Compose([
            transforms.RandomRotation(degrees=rotation_range),
            transforms.ToTensor()
        ]))
    
    # 4. Color jitter variations - create multiple samples with random parameters
    color_jitter_config = transform_config.get('color_jitter', {
        'brightness': [0.8, 1.2], 'contrast': [0.8, 1.2], 
        'saturation': [0.8, 1.2], 'hue': [-0.1, 0.1]
    })
    for _ in range(n_samples_per_transform):
        # Convert config ranges to tuples for ColorJitter (handle OmegaConf ListConfig)
        def convert_to_tuple_or_none(val):
            if val is None:
                return None
            # Convert OmegaConf ListConfig to list first
            if hasattr(val, '__iter__') and not isinstance(val, str):
                val_list = list(val)
                if len(val_list) == 2:
                    return tuple(val_list)
            elif isinstance(val, (int, float)):
                return val
            return None
        
        brightness_val = convert_to_tuple_or_none(color_jitter_config.get('brightness', [0.8, 1.2]))
        contrast_val = convert_to_tuple_or_none(color_jitter_config.get('contrast', [0.8, 1.2]))
        saturation_val = convert_to_tuple_or_none(color_jitter_config.get('saturation', [0.8, 1.2]))
        hue_val = convert_to_tuple_or_none(color_jitter_config.get('hue', [-0.1, 0.1]))
        
        tta_transforms.append(transforms.Compose([
            transforms.ColorJitter(
                brightness=brightness_val,
                contrast=contrast_val,
                saturation=saturation_val,
                hue=hue_val
            ),
            transforms.ToTensor()
        ]))
    
    # 5. Affine transformations - create multiple samples with random parameters
    affine_config = transform_config.get('affine', {
        'degrees': [-5, 5], 'translate': [0.0, 0.05], 
        'scale': [0.95, 1.05], 'shear': [-2, 2]
    })
    for _ in range(n_samples_per_transform):
        # Convert config values to proper format for RandomAffine
        degrees_val = affine_config.get('degrees', [-5, 5])
        translate_val = affine_config.get('translate', [0.0, 0.05])
        scale_val = affine_config.get('scale', [0.95, 1.05])
        shear_val = affine_config.get('shear', [-2, 2])
        
        tta_transforms.append(transforms.Compose([
            transforms.RandomAffine(
                degrees=list(degrees_val) if hasattr(degrees_val, '__iter__') else degrees_val,
                translate=tuple(list(translate_val)) if hasattr(translate_val, '__iter__') else translate_val,
                scale=tuple(list(scale_val)) if hasattr(scale_val, '__iter__') else scale_val,
                shear=list(shear_val) if hasattr(shear_val, '__iter__') else shear_val
            ),
            transforms.ToTensor()
        ]))
    
    # 6. Combined transforms - mix multiple augmentations
    for _ in range(n_samples_per_transform):
        # Get parameters for combined transforms using the same conversion function
        def convert_to_tuple_or_none(val):
            if val is None:
                return None
            # Convert OmegaConf ListConfig to list first
            if hasattr(val, '__iter__') and not isinstance(val, str):
                val_list = list(val)
                if len(val_list) == 2:
                    return tuple(val_list)
            elif isinstance(val, (int, float)):
                return val
            return None
        
        brightness_combined = convert_to_tuple_or_none(color_jitter_config.get('brightness', [0.8, 1.2]))
        contrast_combined = convert_to_tuple_or_none(color_jitter_config.get('contrast', [0.8, 1.2]))
        saturation_combined = convert_to_tuple_or_none(color_jitter_config.get('saturation', [0.8, 1.2]))
        degrees_combined = list(affine_config.get('degrees', [-5, 5]))
        translate_combined = tuple(list(affine_config.get('translate', [0.0, 0.05])))
        scale_combined = tuple(list(affine_config.get('scale', [0.95, 1.05])))
        
        combined_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation_range),
            transforms.ColorJitter(
                brightness=brightness_combined,
                contrast=contrast_combined,
                saturation=saturation_combined
            ),
            transforms.RandomAffine(
                degrees=degrees_combined,
                translate=translate_combined,
                scale=scale_combined
            ),
            transforms.ToTensor()
        ]
        tta_transforms.append(transforms.Compose(combined_transforms))
    
    return tta_transforms


def create_test_dataloader(config: DictConfig, transform=None) -> torch.utils.data.DataLoader:
    """
    Create test data loader.
    
    Args:
        config: Configuration object
        transform: Optional transform to apply
        
    Returns:
        Test data loader and dataset
    """
    print("Loading test dataset...")
    
    # Create test dataset with optional transform
    test_dataset = ClipsDataset(
        config=config,
        split='test',
        transform=transform
    )
    
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    print(f"Classes: {list(test_dataset.label_dict.keys())}")
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data_loader.batch_size,
        num_workers=config.data_loader.num_workers,
        pin_memory=config.data_loader.pin_memory,
        shuffle=False,  # No shuffling for evaluation
        drop_last=False  # Keep all samples for evaluation
    )
    
    return test_loader, test_dataset


def load_trained_model(checkpoint_path: str) -> VivitLightningModule:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load model from checkpoint - this will restore the model with its original config
    model = VivitLightningModule.load_from_checkpoint(
        checkpoint_path,
        strict=False  # Allow some flexibility in loading
    )
    
    print("Model loaded successfully!")
    return model


def evaluate_with_tta(model, config: DictConfig, device: str, n_tta: int = 6) -> Dict[str, Any]:
    """
    Evaluate model with test-time augmentation using majority voting.
    
    Args:
        model: Trained model
        config: Configuration object
        device: Device to use for evaluation
        n_tta: Number of TTA transforms to use (default: 6)
        
    Returns:
        Dictionary containing evaluation results
    """
    print(f"Starting evaluation with {n_tta} test-time augmentations...")
    
    # Get TTA transforms
    tta_transforms = create_tta_transforms(config)
    n_tta = min(n_tta, len(tta_transforms))  # Use available transforms
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    # Get original test dataset to know the size
    test_loader_orig, test_dataset = create_test_dataloader(config)
    num_samples = len(test_dataset)
    
    # Store predictions for each augmentation
    tta_predictions = [[] for _ in range(n_tta)]
    tta_probabilities = [[] for _ in range(n_tta)]
    
    print(f"Running evaluation with {n_tta} augmentations...")
    
    # Run evaluation for each TTA transform
    for tta_idx in range(n_tta):
        print(f"Processing TTA {tta_idx + 1}/{n_tta}...")
        
        # Create dataloader with specific transform
        test_loader, _ = create_test_dataloader(config, transform=tta_transforms[tta_idx])
        
        batch_predictions = []
        batch_targets = []
        batch_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                frames, targets = batch
                frames = frames.to(device)
                targets = targets.to(device)
                
                # Get model predictions
                logits = model(frames)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                batch_predictions.extend(preds.cpu().numpy())
                batch_targets.extend(targets.cpu().numpy())
                batch_probabilities.extend(probs.cpu().numpy())
        
        tta_predictions[tta_idx] = np.array(batch_predictions)
        tta_probabilities[tta_idx] = np.array(batch_probabilities)
        
        if tta_idx == 0:  # Store targets from first iteration
            all_targets = np.array(batch_targets)
    
    # Perform majority voting
    print("Computing majority vote predictions...")
    final_predictions = []
    final_probabilities = []
    
    for sample_idx in range(num_samples):
        # Get predictions for this sample across all TTA transforms
        sample_predictions = [tta_predictions[tta_idx][sample_idx] for tta_idx in range(n_tta)]
        sample_probabilities = [tta_probabilities[tta_idx][sample_idx] for tta_idx in range(n_tta)]
        
        # Majority voting for predictions
        vote_counter = Counter(sample_predictions)
        majority_prediction = vote_counter.most_common(1)[0][0]
        final_predictions.append(majority_prediction)
        
        # Average probabilities
        avg_probabilities = np.mean(sample_probabilities, axis=0)
        final_probabilities.append(avg_probabilities)
    
    final_predictions = np.array(final_predictions)
    final_probabilities = np.array(final_probabilities)
    
    # Calculate metrics using the model's evaluation logic
    print("Calculating metrics...")
    overall_accuracy = np.mean(final_predictions == all_targets)
    
    # Use sklearn for additional metrics
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    balanced_accuracy = balanced_accuracy_score(all_targets, final_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, final_predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, final_predictions)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_targets, final_predictions, average=None, zero_division=0
    )
    
    # Get class names
    class_names = test_dataset.get_class_names()
    
    # Organize results
    results = {
        'targets': all_targets.tolist(),
        'predictions': final_predictions.tolist(),
        'probabilities': final_probabilities.tolist(),
        'overall_accuracy': float(overall_accuracy),
        'balanced_accuracy': float(balanced_accuracy),
        'overall_precision': float(precision),
        'overall_recall': float(recall),
        'overall_f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'per_class_metrics': {
            'precision': {i: float(per_class_precision[i]) for i in range(len(per_class_precision))},
            'recall': {i: float(per_class_recall[i]) for i in range(len(per_class_recall))},
            'f1_score': {i: float(per_class_f1[i]) for i in range(len(per_class_f1))},
            'support': {i: int(per_class_support[i]) for i in range(len(per_class_support))}
        },
        'n_tta_transforms': n_tta,
        'evaluation_method': 'test_time_augmentation'
    }
    
    return results


def save_results(results: Dict[str, Any], output_dir: str, config: DictConfig):
    """
    Save evaluation results to files.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save results
        config: Configuration object
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comprehensive results as JSON
    results_file = output_path / f"evaluation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Save confusion matrix as CSV
    if 'confusion_matrix' in results:
        cm_df = pd.DataFrame(
            results['confusion_matrix'],
            index=[f"True_{results['class_names'][i]}" for i in range(len(results['class_names']))],
            columns=[f"Pred_{results['class_names'][i]}" for i in range(len(results['class_names']))]
        )
        cm_file = output_path / f"confusion_matrix_{timestamp}.csv"
        cm_df.to_csv(cm_file)
        print(f"Confusion matrix saved to: {cm_file}")
    
    # Save per-class metrics as CSV
    if 'per_class_metrics' in results:
        per_class_data = []
        for class_idx, class_name in results['class_names'].items():
            per_class_data.append({
                'class_idx': class_idx,
                'class_name': class_name,
                'precision': results['per_class_metrics']['precision'].get(class_idx, 0.0),
                'recall': results['per_class_metrics']['recall'].get(class_idx, 0.0),
                'f1_score': results['per_class_metrics']['f1_score'].get(class_idx, 0.0),
                'support': results['per_class_metrics']['support'].get(class_idx, 0)
            })
        
        per_class_df = pd.DataFrame(per_class_data)
        per_class_file = output_path / f"per_class_metrics_{timestamp}.csv"
        per_class_df.to_csv(per_class_file, index=False)
        print(f"Per-class metrics saved to: {per_class_file}")
    
    # Save predictions if requested
    if config.evaluation.get('save_predictions', False) and 'predictions' in results:
        predictions_df = pd.DataFrame({
            'true_label': results['targets'],
            'predicted_label': results['predictions'],
            'true_class': [results['class_names'][label] for label in results['targets']],
            'predicted_class': [results['class_names'][label] for label in results['predictions']]
        })
        
        # Add prediction probabilities
        if 'probabilities' in results:
            probs = np.array(results['probabilities'])
            for class_idx, class_name in results['class_names'].items():
                predictions_df[f'prob_{class_name}'] = probs[:, class_idx]
        
        pred_file = output_path / f"predictions_{timestamp}.csv"
        predictions_df.to_csv(pred_file, index=False)
        print(f"Predictions saved to: {pred_file}")
    
    # Save evaluation summary
    summary = {
        'evaluation_timestamp': timestamp,
        'model_checkpoint': config.evaluation.model_checkpoint_path,
        'test_samples': len(results['targets']),
        'num_classes': len(results['class_names']),
        'overall_accuracy': results['overall_accuracy'],
        'balanced_accuracy': results['balanced_accuracy'],
        'overall_precision': results['overall_precision'],
        'overall_recall': results['overall_recall'],
        'overall_f1': results['overall_f1']
    }
    
    summary_file = output_path / f"evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary saved to: {summary_file}")


def print_evaluation_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    evaluation_method = results.get('evaluation_method', 'standard')
    if evaluation_method == 'test_time_augmentation':
        print(f"Evaluation Method:    Test-Time Augmentation ({results.get('n_tta_transforms', 'N/A')} transforms)")
    else:
        print("Evaluation Method:    Standard")
    
    print(f"Test Samples:         {len(results['targets'])}")
    print(f"Number of Classes:    {len(results['class_names'])}")
    print("\nOverall Metrics:")
    print(f"  Accuracy:           {results['overall_accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {results['balanced_accuracy']:.4f}")
    print(f"  Precision:          {results['overall_precision']:.4f}")
    print(f"  Recall:             {results['overall_recall']:.4f}")
    print(f"  F1-Score:           {results['overall_f1']:.4f}")
    
    print("\nPer-Class Metrics:")
    for class_idx, class_name in results['class_names'].items():
        precision = results['per_class_metrics']['precision'].get(class_idx, 0)
        recall = results['per_class_metrics']['recall'].get(class_idx, 0)
        f1 = results['per_class_metrics']['f1_score'].get(class_idx, 0)
        support = results['per_class_metrics']['support'].get(class_idx, 0)
        print(f"  {class_name:12}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}")
    
    print("="*60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained video action recognition model')
    parser.add_argument('--config', type=str, default='config/SLAM-Vivit-Cls-eval.yaml',
                      help='Path to evaluation configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for results (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for evaluation (auto, cpu, cuda)')
    parser.add_argument('--tta', action='store_true',
                      help='Enable test-time augmentation with majority voting')
    parser.add_argument('--n-tta', type=int, default=6,
                      help='Number of TTA transforms to use (default: 6)')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.checkpoint is not None:
        config.evaluation.model_checkpoint_path = args.checkpoint
    if args.output_dir is not None:
        config.evaluation.output_dir = args.output_dir
    if args.device is not None:
        config.evaluation.device = args.device
    
    print(f"Configuration loaded from: {args.config}")
    print(f"Model checkpoint: {config.evaluation.model_checkpoint_path}")
    print(f"Output directory: {config.evaluation.output_dir}")
    
    # Set device
    if config.evaluation.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.evaluation.device
    
    print(f"Using device: {device}")
    
    # Load trained model
    model = load_trained_model(config.evaluation.model_checkpoint_path)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Determine whether to use TTA (command line argument overrides config)
    use_tta = args.tta
    if not args.tta and hasattr(config.evaluation, 'tta') and config.evaluation.tta.get('enabled', False):
        use_tta = True
        print("TTA enabled in config file")
    
    # Get number of TTA transforms (command line overrides config)
    n_tta = args.n_tta
    if hasattr(config.evaluation, 'tta') and config.evaluation.tta.get('n_transforms'):
        n_tta = config.evaluation.tta.n_transforms if args.n_tta == 6 else args.n_tta  # Use config if default CLI value
    
    # Run evaluation with or without TTA
    if use_tta:
        print(f"Starting evaluation with test-time augmentation ({n_tta} transforms)...")
        results = evaluate_with_tta(model, config, device, n_tta=n_tta)
    else:
        print("Starting standard evaluation...")
        # Create test data loader
        test_loader, _ = create_test_dataloader(config)
        
        # Run standard evaluation using the model's method
        results = model.evaluate_model(test_loader)
        results['evaluation_method'] = 'standard'
    
    # Print results to console
    print_evaluation_results(results)
    
    # Save results if output directory is specified
    if hasattr(config.evaluation, 'output_dir') and config.evaluation.output_dir:
        save_results(results, config.evaluation.output_dir, config)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()