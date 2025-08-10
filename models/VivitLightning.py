"""
PyTorch Lightning wrapper for ViViT model for surgical video action recognition.

This module provides a Lightning-based training framework for the ViViT (Video Vision Transformer)
model, with comprehensive logging, metrics tracking, and flexible optimization strategies.
"""

import torch
import torch.nn.functional as F
import lightning as L

# Set tensor core precision for better performance on modern GPUs
torch.set_float32_matmul_precision('medium')
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import torchmetrics
from omegaconf import DictConfig
from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report

from .Vivit import ViViT


class VivitLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for ViViT model training.
    
    This class wraps the ViViT model with Lightning functionality including:
    - Training, validation, and test steps
    - Configurable optimizers and schedulers
    - Comprehensive metrics tracking
    - Automatic logging and checkpointing
    
    Args:
        config (DictConfig): Hydra configuration containing model and training parameters
        num_classes (int): Number of action classes
        class_names (Optional[Dict[int, str]]): Mapping from class indices to names
    """
    
    def __init__(
        self,
        config: DictConfig,
        num_classes: int,
        class_names: Optional[Dict[int, str]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.num_classes = num_classes
        self.class_names = class_names or {i: f"Class_{i}" for i in range(num_classes)}
        
        # Initialize ViViT model with config parameters
        self.model = self._build_model()
        
        # Initialize metrics
        self._setup_metrics()
        
        # Store training configuration
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # For tracking best metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        # Print initial memory status
        self._print_memory_usage("Model initialization")
        
    def _build_model(self) -> ViViT:
        """
        Build ViViT model from configuration parameters.
        
        Returns:
            ViViT model instance
        """
        model_config = self.config.model
        video_config = self.config.video_processing
        
        model = ViViT(
            image_size=video_config.height,  # Assuming square images
            patch_size=model_config.patch_size[0],  # Assuming square patches
            num_classes=self.num_classes,
            num_frames=video_config.time_size,
            dim=model_config.embed_dim,
            depth=model_config.num_layers,
            heads=model_config.num_heads,
            pool='cls',  # Use CLS token for classification
            in_channels=model_config.input_channels,
            dim_head=model_config.embed_dim // model_config.num_heads,
            dropout=model_config.dropout,
            emb_dropout=model_config.dropout,
            scale_dim=4  # Default scale factor for MLP
        )
        
        return model
    
    def _setup_metrics(self):
        """Initialize torchmetrics for comprehensive evaluation."""
        # Classification metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        
        # Precision, Recall, F1
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average='macro')
        
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average='macro')
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average='macro')
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        
        # Confusion matrix for detailed analysis
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViViT model.
        
        Args:
            x: Input tensor of shape (batch_size, time_size, channels, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """
        Shared step for training, validation, and testing.
        
        Args:
            batch: Tuple of (frames, labels)
            stage: One of 'train', 'val', 'test'
            
        Returns:
            Dictionary containing loss and predictions
        """
        frames, labels = batch
        logits = self.forward(frames)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'loss': loss,
            'preds': preds,
            'targets': labels,
            'logits': logits
        }
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for one batch.
        
        Args:
            batch: Tuple of (frames, labels)
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor
        """
        # Explicit memory cleanup to prevent accumulation
        if batch_idx % 50 == 0:  # Every 50 batches
            torch.cuda.empty_cache()
        
        outputs = self._shared_step(batch, 'train')
        loss, preds, targets = outputs['loss'], outputs['preds'], outputs['targets']
        
        # Update and log metrics
        self.train_accuracy.update(preds, targets)
        self.train_precision.update(preds, targets)
        self.train_recall.update(preds, targets)
        self.train_f1.update(preds, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        # Detailed logging every N steps (reduced frequency to save memory)
        if batch_idx % (self.config.logging.log_frequency * 2) == 0:  # Reduced frequency
            self._log_detailed_metrics('train', outputs, batch_idx)
        
        # Clear intermediate tensors
        del outputs, preds, targets
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for one batch.
        
        Args:
            batch: Tuple of (frames, labels)
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor
        """
        # Use torch.no_grad() explicitly for validation to save memory
        with torch.no_grad():
            outputs = self._shared_step(batch, 'val')
            loss, preds, targets = outputs['loss'], outputs['preds'], outputs['targets']
            
            # Update metrics
            self.val_accuracy.update(preds, targets)
            self.val_precision.update(preds, targets)
            self.val_recall.update(preds, targets)
            self.val_f1.update(preds, targets)
            self.val_confusion_matrix.update(preds, targets)
            
            # Log metrics
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
            
            # Clear intermediate tensors
            del outputs, preds, targets
            
            return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step for one batch.
        
        Args:
            batch: Tuple of (frames, labels)
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor
        """
        outputs = self._shared_step(batch, 'test')
        loss, preds, targets = outputs['loss'], outputs['preds'], outputs['targets']
        
        # Update metrics
        self.test_accuracy.update(preds, targets)
        self.test_precision.update(preds, targets)
        self.test_recall.update(preds, targets)
        self.test_f1.update(preds, targets)
        self.test_confusion_matrix.update(preds, targets)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True)
        
        return loss
    
    def on_training_epoch_end(self):
        """Called at the end of training epoch."""
        # Compute and log epoch metrics
        train_acc = self.train_accuracy.compute()
        train_precision = self.train_precision.compute()
        train_recall = self.train_recall.compute()
        train_f1 = self.train_f1.compute()
        
        # Log comprehensive metrics
        self.log('train_precision', train_precision)
        self.log('train_recall', train_recall)
        self.log('train_f1', train_f1)
        
        # Print to console
        print(f"Training Epoch {self.current_epoch}: "
              f"Acc={train_acc:.4f}, Prec={train_precision:.4f}, "
              f"Recall={train_recall:.4f}, F1={train_f1:.4f}")
        
        # Reset metrics
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
        
        # Explicit memory cleanup at epoch end
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Compute metrics
        val_acc = self.val_accuracy.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_f1 = self.val_f1.compute()
        
        # Log comprehensive metrics
        self.log('val_precision', val_precision)
        self.log('val_recall', val_recall)
        self.log('val_f1', val_f1)
        
        # Track best metrics
        current_val_loss = self.trainer.callback_metrics.get('val_loss', float('inf'))
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
        
        # Log best metrics to W&B
        self.log('best_val_acc', self.best_val_acc)
        self.log('best_val_loss', self.best_val_loss)
        
        # Print to console
        print(f"Validation Epoch {self.current_epoch}: "
              f"Acc={val_acc:.4f}, Prec={val_precision:.4f}, "
              f"Recall={val_recall:.4f}, F1={val_f1:.4f}")
        print(f"Best Val Acc: {self.best_val_acc:.4f}, Best Val Loss: {self.best_val_loss:.4f}")
        
        # Log confusion matrix every few epochs
        if self.current_epoch % 10 == 0:
            confusion_mat = self.val_confusion_matrix.compute()
            print(f"Validation Confusion Matrix (Epoch {self.current_epoch}):")
            self._print_confusion_matrix(confusion_mat)
            
            # Log confusion matrix to W&B if using wandb
            if hasattr(self.trainer.logger, 'experiment') and hasattr(self.trainer.logger.experiment, 'log'):
                self._log_confusion_matrix_to_wandb(confusion_mat)
        
        # Reset metrics
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_confusion_matrix.reset()
        
        # Explicit memory cleanup at epoch end
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Compute final test metrics
        test_acc = self.test_accuracy.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        test_f1 = self.test_f1.compute()
        confusion_mat = self.test_confusion_matrix.compute()
        
        # Log comprehensive metrics
        self.log('test_precision', test_precision)
        self.log('test_recall', test_recall)
        self.log('test_f1', test_f1)
        
        # Print comprehensive test results
        print("=" * 60)
        print("FINAL TEST RESULTS")
        print("=" * 60)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        print("\nTest Confusion Matrix:")
        self._print_confusion_matrix(confusion_mat)
        print("=" * 60)
    
    def evaluate_model(self, test_dataloader) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with detailed metrics.
        
        Args:
            test_dataloader: DataLoader containing test samples
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        self.eval()
        device = next(self.parameters()).device
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        print("Running comprehensive evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                frames, targets = batch
                frames = frames.to(device)
                targets = targets.to(device)
                
                # Get model predictions
                logits = self.forward(frames)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Store predictions and targets
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                if batch_idx % 20 == 0:
                    print(f"Processed batch {batch_idx + 1}/{len(test_dataloader)}")
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate comprehensive metrics
        overall_accuracy = np.mean(all_preds == all_targets)
        balanced_accuracy = balanced_accuracy_score(all_targets, all_preds)
        
        # Per-class metrics
        per_class_accuracy = {}
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        per_class_support = {}
        
        for class_idx in range(self.num_classes):
            class_mask = (all_targets == class_idx)
            if class_mask.sum() > 0:
                class_preds = all_preds[class_mask]
                class_targets = all_targets[class_mask]
                
                # Accuracy for this class
                class_acc = np.mean(class_preds == class_targets)
                per_class_accuracy[class_idx] = class_acc
                
                # Precision, recall, F1 for this class
                tp = np.sum((class_preds == class_idx) & (class_targets == class_idx))
                fp = np.sum((all_preds == class_idx) & (all_targets != class_idx))
                fn = np.sum((all_preds != class_idx) & (all_targets == class_idx))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                per_class_precision[class_idx] = precision
                per_class_recall[class_idx] = recall
                per_class_f1[class_idx] = f1
                per_class_support[class_idx] = class_mask.sum()
        
        # Overall metrics
        overall_precision = np.mean(list(per_class_precision.values()))
        overall_recall = np.mean(list(per_class_recall.values()))
        overall_f1 = np.mean(list(per_class_f1.values()))
        
        # Confusion matrix
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true_label, pred_label in zip(all_targets, all_preds):
            confusion_matrix[true_label, pred_label] += 1
        
        # Compile results
        results = {
            'overall_accuracy': float(overall_accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'overall_precision': float(overall_precision),
            'overall_recall': float(overall_recall),
            'overall_f1': float(overall_f1),
            'per_class_metrics': {
                'accuracy': per_class_accuracy,
                'precision': per_class_precision,
                'recall': per_class_recall,
                'f1_score': per_class_f1,
                'support': per_class_support
            },
            'confusion_matrix': confusion_matrix.tolist(),
            'class_names': self.class_names,
            'predictions': all_preds.tolist(),
            'targets': all_targets.tolist(),
            'probabilities': all_probs.tolist()
        }
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, Any]):
        """
        Print comprehensive evaluation results in a formatted manner.
        
        Args:
            results: Dictionary containing evaluation results
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 80)
        
        # Overall metrics
        print(f"Overall Accuracy:     {results['overall_accuracy']:.4f}")
        print(f"Balanced Accuracy:    {results['balanced_accuracy']:.4f}")
        print(f"Overall Precision:    {results['overall_precision']:.4f}")
        print(f"Overall Recall:       {results['overall_recall']:.4f}")
        print(f"Overall F1-Score:     {results['overall_f1']:.4f}")
        
        # Per-class metrics
        print("\n" + "-" * 80)
        print("PER-CLASS METRICS")
        print("-" * 80)
        print(f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 80)
        
        per_class = results['per_class_metrics']
        for class_idx in range(self.num_classes):
            class_name = self.class_names.get(class_idx, f"Class_{class_idx}")
            accuracy = per_class['accuracy'].get(class_idx, 0.0)
            precision = per_class['precision'].get(class_idx, 0.0)
            recall = per_class['recall'].get(class_idx, 0.0)
            f1 = per_class['f1_score'].get(class_idx, 0.0)
            support = per_class['support'].get(class_idx, 0)
            
            print(f"{class_name:<15} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
        
        # Confusion matrix
        print("\n" + "-" * 80)
        print("CONFUSION MATRIX")
        print("-" * 80)
        confusion_matrix = np.array(results['confusion_matrix'])
        
        # Print header
        header = "True\\Pred".ljust(12)
        for class_idx in range(self.num_classes):
            class_name = self.class_names.get(class_idx, f"C{class_idx}")[:8]
            header += f"{class_name:>8}"
        print(header)
        print("-" * 80)
        
        # Print matrix rows
        for i in range(self.num_classes):
            class_name = self.class_names.get(i, f"C{i}")[:11]
            row = f"{class_name:<12}"
            for j in range(self.num_classes):
                row += f"{confusion_matrix[i, j]:>8}"
            print(row)
        
        print("=" * 80)
    
    def _print_confusion_matrix(self, confusion_mat: torch.Tensor):
        """Print confusion matrix in a readable format."""
        print("Predicted →")
        print("Actual ↓")
        
        # Print header with class names
        header = "        "
        for i in range(self.num_classes):
            class_name = self.class_names[i][:8]  # Truncate long names
            header += f"{class_name:>8}"
        print(header)
        
        # Print matrix rows
        for i in range(self.num_classes):
            class_name = self.class_names[i][:8]
            row = f"{class_name:>8}"
            for j in range(self.num_classes):
                row += f"{confusion_mat[i, j].item():>8}"
            print(row)
    
    def _log_confusion_matrix_to_wandb(self, confusion_mat: torch.Tensor):
        """Log confusion matrix to W&B as a heatmap."""
        try:
            import wandb
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            # Convert to numpy and normalize
            cm_np = confusion_mat.cpu().numpy()
            cm_normalized = cm_np.astype('float') / cm_np.sum(axis=1)[:, np.newaxis]
            
            # Create class labels
            class_labels = [self.class_names[i] for i in range(self.num_classes)]
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar_kws={'label': 'Normalized Count'}
            )
            plt.title(f'Validation Confusion Matrix - Epoch {self.current_epoch}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Log to W&B
            self.trainer.logger.experiment.log({
                f"confusion_matrix_epoch_{self.current_epoch}": wandb.Image(plt)
            })
            
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib or seaborn not available for confusion matrix visualization")
        except Exception as e:
            print(f"Warning: Could not log confusion matrix to W&B: {str(e)}")
    
    def _log_detailed_metrics(self, stage: str, outputs: Dict[str, torch.Tensor], batch_idx: int):
        """Log detailed metrics for debugging."""
        preds, targets = outputs['preds'], outputs['targets']
        logits = outputs['logits']
        
        # Calculate per-class accuracy
        for class_idx in range(self.num_classes):
            class_mask = (targets == class_idx)
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == class_idx).float().mean()
                class_name = self.class_names[class_idx]
                print(f"{stage.capitalize()} Batch {batch_idx} - {class_name}: {class_acc:.3f}")
        
        # Log prediction confidence
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        avg_confidence = max_probs.mean()
        print(f"{stage.capitalize()} Batch {batch_idx} - Avg Confidence: {avg_confidence:.3f}")
    
    def _print_memory_usage(self, stage: str):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_memory = torch.cuda.max_memory_allocated() / 1024**3   # GB
            print(f"GPU Memory [{stage}] - Allocated: {memory_allocated:.2f}GB, "
                  f"Reserved: {memory_reserved:.2f}GB, Max: {max_memory:.2f}GB")
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        # Get optimizer configuration
        optimizer_name = self.config.training.optimizer.lower()
        
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Configure scheduler
        scheduler_name = self.config.training.scheduler.lower()
        
        if scheduler_name == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        elif scheduler_name == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        elif scheduler_name == 'none' or scheduler_name is None:
            # No scheduler - train with constant learning rate
            print("Training with constant learning rate (no scheduler)")
            return optimizer
        else:
            # No scheduler for any other cases
            print(f"Unknown scheduler '{scheduler_name}', using constant learning rate")
            return optimizer


def create_lightning_trainer(config: DictConfig, log_dir: str = "logs/SLAM-Vivit_Cls2") -> L.Trainer:
    """
    Create a PyTorch Lightning trainer with appropriate callbacks and loggers.
    
    Args:
        config: Hydra configuration
        log_dir: Directory for saving logs and checkpoints
        
    Returns:
        Configured PyTorch Lightning Trainer
    """
    import os
    
    # Setup callbacks
    callbacks = []
    
    # Determine logging strategy
    use_wandb = config.logging.get('use_wandb', False)
    logger = None
    
    if use_wandb:
        # Initialize W&B logger
        logger = _setup_wandb_logger(config)
        print(f"Using Weights & Biases logging to project: {config.logging.wandb.project}")
    else:
        # Use local directory logging (TensorBoard)
        # Find the next available run directory
        base_log_path = Path(log_dir)
        run_counter = 0
        while True:
            run_path = base_log_path / f"run_{run_counter}"
            if not run_path.exists():
                break
            run_counter += 1
        
        # Create the run directory
        run_path.mkdir(parents=True, exist_ok=True)
        print(f"Creating local logging directory: {run_path}")
        
        # Setup TensorBoard logger
        logger = TensorBoardLogger(
            save_dir=str(run_path),
            name="",
            version=""
        )
    
    # Model checkpointing
    if config.logging.save_checkpoints:
        # Create experiment-specific checkpoint directory
        experiment_name = config.logging.experiment_name
        checkpoint_dir = Path(config.logging.log_dir) / experiment_name / "weights"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
        
        # Main checkpoint callback - saves best 10 models based on validation loss
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=10,  # Keep best 10 checkpoints
            save_last=True,  # Also save the final checkpoint as 'last.ckpt'
            every_n_epochs=config.logging.checkpoint_frequency,
            auto_insert_metric_name=False,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Additional checkpoint callback for best accuracy
        best_acc_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-acc-{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}",
            monitor="val_acc",
            mode="max", 
            save_top_k=1,  # Keep only the best accuracy model
            auto_insert_metric_name=False,
            verbose=True
        )
        callbacks.append(best_acc_checkpoint)
    
    # Early stopping (only if enabled in config)
    enable_early_stopping = config.training.get('enable_early_stopping', True)
    if enable_early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,
            verbose=True
        )
        callbacks.append(early_stopping)
        print("Early stopping enabled with patience=20")
    else:
        print("Early stopping disabled - will train for full epoch count")
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Create trainer with memory optimizations
    trainer = L.Trainer(
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',
        devices=1,  # Use single device to avoid distributed training complexity
        precision='16-mixed',  # Use mixed precision for faster training
        gradient_clip_val=1.0,  # Gradient clipping
        log_every_n_steps=config.logging.log_frequency,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        # Memory optimization settings
        accumulate_grad_batches=2,  # Accumulate gradients to simulate larger batch size
        enable_checkpointing=True,
        deterministic=False,  # Disable for better performance
        benchmark=True,  # Enable cudnn benchmark for consistent input sizes
        # Reduce memory usage
        num_sanity_val_steps=0,  # Skip sanity validation to save memory
        limit_val_batches=1.0,  # Use all validation data but efficiently
    )
    
    return trainer


def _setup_wandb_logger(config: DictConfig) -> WandbLogger:
    """
    Setup Weights & Biases logger with configuration.
    
    Args:
        config: Hydra configuration containing wandb settings
        
    Returns:
        Configured WandbLogger instance
    """
    import wandb
    import os
    from pathlib import Path
    
    wandb_config = config.logging.wandb
    
    # Get API key from config or environment variable
    api_key = wandb_config.get('api_key', '') or os.getenv('WANDB_API_KEY', '')
    if api_key:
        wandb.login(key=api_key)
    else:
        print("Warning: No WANDB_API_KEY found in config or environment variables.")
        print("Please set WANDB_API_KEY in your config file or as an environment variable.")
    
    # Create experiment name with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.logging.experiment_name}_{timestamp}"
    
    # Create custom save directory based on experiment name
    experiment_name = config.logging.experiment_name
    experiment_log_dir = Path(config.logging.log_dir) / experiment_name
    experiment_log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"W&B logs will be saved to: {experiment_log_dir}")
    
    # Setup W&B configuration to log
    wandb_log_config = {
        "model": dict(config.model),
        "training": dict(config.training),
        "data_loader": dict(config.data_loader),
        "video_processing": dict(config.video_processing),
        "dataset": {
            "name": config.dataset.name,
            "num_classes": config.dataset.num_classes,
            "train_samples": "TBD",  # Will be updated during training
            "val_samples": "TBD"
        }
    }
    
    # Create WandbLogger with custom save directory
    logger = WandbLogger(
        project=wandb_config.project,
        entity=wandb_config.get('entity', None),
        name=run_name,
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
        group=wandb_config.get('group', None),
        job_type=wandb_config.get('job_type', 'train'),
        config=wandb_log_config,
        save_dir=str(experiment_log_dir)  # Use experiment-specific directory
    )
    
    return logger


def get_model_summary(model: VivitLightningModule, input_shape: tuple) -> str:
    """
    Get a comprehensive model summary.
    
    Args:
        model: Lightning module
        input_shape: Shape of input tensor (batch_size, time_size, channels, height, width)
        
    Returns:
        String containing model summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
ViViT Model Summary:
==================
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Input Shape: {input_shape}
Model Configuration:
- Image Size: {model.config.video_processing.height}x{model.config.video_processing.width}
- Patch Size: {model.config.model.patch_size}
- Embedding Dimension: {model.config.model.embed_dim}
- Number of Heads: {model.config.model.num_heads}
- Number of Layers: {model.config.model.num_layers}
- Number of Classes: {model.num_classes}
- Temporal Frames: {model.config.video_processing.time_size}
==================
"""
    return summary