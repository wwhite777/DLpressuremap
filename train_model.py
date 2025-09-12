#!/usr/bin/env python
"""
Optimized training script for IR to PM prediction
"""
import argparse
import time
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt

from config import Config
from utils import ImageProcessor, MetricsCalculator, FileManager

class IRPMDataset(Dataset):
    """Dataset for IR to PM mapping"""
    
    def __init__(self, config: Config, transform=None):
        self.config = config
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.get('training.image_height'), 
                             config.get('training.image_width')))
        ])
        
        # Find paired files
        self.pairs = FileManager.find_paired_files(
            config.get_path('ir_dir'), 'png',
            config.get_path('pm_dir'), 'png'
        )
        
        if not self.pairs:
            raise RuntimeError(f"No matching IR/PM pairs found")
        
        print(f"Found {len(self.pairs)} IR/PM pairs")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ir_path, pm_path = self.pairs[idx]
        
        # Load and process images
        ir_img = Image.open(ir_path).convert('RGB')
        ir_img = ImageProcessor.normalize_scale(ir_img)
        
        pm_img = Image.open(pm_path)
        pm_img = ImageProcessor.normalize_scale(pm_img)
        
        # Apply transforms
        ir_tensor = self.transform(ir_img)
        pm_tensor = self.transform(pm_img)
        
        return ir_tensor, pm_tensor

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: Config, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        
        # Initialize model
        self.model = self._create_model()
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('training.learning_rate')
        )
        
        # Metrics
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        
        # History
        self.train_history = {'loss': [], 'rmse': [], 'ssim': []}
        self.val_history = {'loss': [], 'rmse': [], 'ssim': []}
    
    def _create_model(self) -> nn.Module:
        """Create segmentation model"""
        model = smp.Unet(
            encoder_name=self.config.get('model.encoder_name'),
            encoder_weights=self.config.get('model.encoder_weights'),
            in_channels=self.config.get('model.in_channels'),
            classes=self.config.get('model.classes')
        )
        return model.to(self.device)
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss, total_rmse, total_ssim = 0.0, 0.0, 0.0
        
        for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_rmse += MetricsCalculator.rmse(outputs, labels)
            total_ssim += self.ssim_metric(outputs, labels).item()
        
        n = len(dataloader)
        return total_loss / n, total_rmse / n, total_ssim / n
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        metrics = {'loss': 0, 'rmse': 0, 'ssim': 0, 'psnr': 0, 'mae': 0}
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                outputs_n = torch.clamp(outputs, 0.0, 1.0)
                labels_n = torch.clamp(labels, 0.0, 1.0)
                
                # Calculate metrics
                metrics['loss'] += self.criterion(outputs, labels).item()
                metrics['rmse'] += MetricsCalculator.rmse(outputs_n, labels_n)
                metrics['ssim'] += self.ssim_metric(outputs_n, labels_n).item()
                
                mae = torch.mean(torch.abs(outputs_n - labels_n))
                metrics['mae'] += mae.item()
                
                mse = nn.functional.mse_loss(outputs_n, labels_n).item()
                metrics['psnr'] += 10.0 * np.log10(1.0 / (mse + 1e-12))
        
        # Average metrics
        n = len(dataloader)
        return {k: v / n for k, v in metrics.items()}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = None) -> None:
        """Full training loop"""
        num_epochs = num_epochs or self.config.get('training.num_epochs')
        
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            start_time = time.time()
            
            # Train
            train_loss, train_rmse, train_ssim = self.train_epoch(train_loader)
            self.train_history['loss'].append(train_loss)
            self.train_history['rmse'].append(train_rmse)
            self.train_history['ssim'].append(train_ssim)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['rmse'].append(val_metrics['rmse'])
            self.val_history['ssim'].append(val_metrics['ssim'])
            
            # Log progress
            elapsed = int(time.time() - start_time)
            print(f"\n[Epoch {epoch + 1}/{num_epochs}, {elapsed}s]")
            print(f"Train - Loss: {train_loss:.6f} | RMSE: {train_rmse:.4f} | SSIM: {train_ssim:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.6f} | RMSE: {val_metrics['rmse']:.4f} | "
                  f"SSIM: {val_metrics['ssim']:.4f} | PSNR: {val_metrics['psnr']:.2f} | "
                  f"MAE: {val_metrics['mae']:.4f}")
            print("=" * 60)
    
    def save_model(self, path: Path = None) -> None:
        """Save model checkpoint"""
        if path is None:
            save_dir = self.config.get_path('model_save_dir') / 'IR_PM_save'
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"model_e{len(self.train_history['loss'])}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config.config
        }, path)
        print(f"Model saved to {path}")
    
    def save_predictions(self, dataloader: DataLoader, output_dir: Path) -> None:
        """Save model predictions"""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        
        with torch.no_grad():
            idx = 0
            for inputs, _ in tqdm(dataloader, desc="Saving predictions"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).cpu().numpy()
                
                for i in range(outputs.shape[0]):
                    # Get original filename from dataset
                    ir_path, _ = dataloader.dataset.pairs[idx]
                    stem = Path(ir_path).stem
                    
                    # Save prediction
                    pred = outputs[i].squeeze().astype(np.float32)
                    np.save(output_dir / f"{stem}.npy", pred)
                    idx += 1
        
        print(f"Saved {idx} predictions to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train IR to PM model')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--run-name', type=str, default='run_default', help='Run name')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line args
    if args.epochs:
        config.set('training.num_epochs', args.epochs)
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    if args.lr:
        config.set('training.learning_rate', args.lr)
    
    # Create dataset
    dataset = IRPMDataset(config)
    
    # Split dataset
    data_size = len(dataset)
    train_size = int(data_size * config.get('training.train_split'))
    val_size = int(data_size * config.get('training.val_split'))
    test_size = data_size - train_size - val_size
    
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, data_size))
    
    print(f'Train: {train_size}, Val: {val_size}, Test: {test_size}')
    
    # Create dataloaders
    batch_size = config.get('training.batch_size')
    num_workers = config.get('training.num_workers')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    # Train model
    trainer = ModelTrainer(config)
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest Results:")
    print(f"Loss: {test_metrics['loss']:.6f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"SSIM: {test_metrics['ssim']:.4f}")
    print(f"PSNR: {test_metrics['psnr']:.2f} dB")
    print(f"MAE:  {test_metrics['mae']:.4f}")
    
    # Save model and predictions
    trainer.save_model()
    
    output_dir = config.get_path('result_dir') / args.run_name / 'preds'
    trainer.save_predictions(test_loader, output_dir)

if __name__ == "__main__":
    main()