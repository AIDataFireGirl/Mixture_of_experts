#!/usr/bin/env python3
"""
Model preparation script for MoE Recommendation System

This script handles model training, optimization, and preparation for deployment.
It includes data preprocessing, model training, and export functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe_transformer import (
    MoEConfig, RecommendationMoE, create_moe_model, 
    save_moe_model, MoEMonitor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationDataset(Dataset):
    """
    Dataset for recommendation training data.
    Handles user-item interactions with optional context features.
    """
    
    def __init__(self, data_path: str, num_users: int, num_items: int, 
                 max_context_features: int = 10):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to training data file
            num_users: Total number of users
            num_items: Total number of items
            max_context_features: Maximum number of context features
        """
        self.num_users = num_users
        self.num_items = num_items
        self.max_context_features = max_context_features
        
        # Load or generate training data
        if os.path.exists(data_path):
            self.data = self._load_data(data_path)
        else:
            logger.warning(f"Data file {data_path} not found. Generating synthetic data.")
            self.data = self._generate_synthetic_data()
        
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load training data from file"""
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic training data for demonstration"""
        logger.info("Generating synthetic training data...")
        
        data = []
        num_samples = 10000
        
        for i in range(num_samples):
            # Generate random user-item interaction
            user_id = np.random.randint(0, self.num_users)
            item_id = np.random.randint(0, self.num_items)
            
            # Generate context features (time, location, device, etc.)
            context_features = np.random.randn(self.max_context_features)
            
            # Generate binary label (click/purchase)
            label = np.random.binomial(1, 0.3)  # 30% positive rate
            
            data.append({
                'user_id': user_id,
                'item_id': item_id,
                'context_features': context_features.tolist(),
                'label': label
            })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a training sample"""
        sample = self.data[idx]
        
        return {
            'user_id': torch.tensor(sample['user_id'], dtype=torch.long),
            'item_id': torch.tensor(sample['item_id'], dtype=torch.long),
            'context_features': torch.tensor(sample['context_features'], dtype=torch.float),
            'label': torch.tensor(sample['label'], dtype=torch.float)
        }


class MoETrainer:
    """
    Trainer class for MoE recommendation model.
    Handles training, validation, and model optimization.
    """
    
    def __init__(self, config: MoEConfig, num_users: int, num_items: int,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            config: MoE configuration
            num_users: Number of users
            num_items: Number of items
            device: Training device
        """
        self.config = config
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        
        # Initialize model
        self.model = create_moe_model(config, num_users, num_items)
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Monitoring
        self.monitor = MoEMonitor()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.expert_metrics = []
        
        logger.info(f"Initialized trainer with {num_users} users and {num_items} items")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move data to device
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            context_features = batch['context_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(user_ids, item_ids, context_features)
            predictions = output['recommendation_scores']
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Add load balancing loss
            if 'metrics' in output:
                for layer_metrics in output['metrics']:
                    if 'load_balancing_loss' in layer_metrics:
                        loss += layer_metrics['load_balancing_loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update monitoring
            if 'metrics' in output:
                self.monitor.update_metrics(output)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                context_features = batch['context_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                output = self.model(user_ids, item_ids, context_features)
                predictions = output['recommendation_scores']
                
                # Compute loss
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 10, save_path: str = None) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_path: Path to save the trained model
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                save_moe_model(self.model, save_path)
                logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
            
            # Log expert metrics
            expert_stats = self.monitor.get_expert_utilization()
            if expert_stats:
                logger.info(f"Expert utilization: {len(expert_stats)} experts active")
        
        # Final model save
        if save_path:
            save_moe_model(self.model, save_path)
            logger.info(f"Saved final model to {save_path}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'expert_metrics': self.monitor.get_expert_utilization(),
            'performance_summary': self.monitor.get_performance_summary()
        }


def prepare_model(args: argparse.Namespace) -> None:
    """
    Main function to prepare the MoE model.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting model preparation...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create configuration
    config = MoEConfig(
        num_experts=args.num_experts,
        num_experts_per_token=args.experts_per_token,
        expert_capacity=args.expert_capacity,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        load_balancing_loss_weight=args.load_balancing_weight
    )
    
    # Create datasets
    train_dataset = RecommendationDataset(
        args.train_data,
        args.num_users,
        args.num_items,
        args.max_context_features
    )
    
    val_dataset = RecommendationDataset(
        args.val_data if args.val_data else args.train_data,
        args.num_users,
        args.num_items,
        args.max_context_features
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize trainer
    trainer = MoETrainer(
        config=config,
        num_users=args.num_users,
        num_items=args.num_items,
        device=device
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_path=args.model_path
    )
    
    # Save training history
    if args.history_path:
        with open(args.history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {args.history_path}")
    
    # Save configuration
    if args.config_path:
        config_dict = {
            'num_experts': config.num_experts,
            'num_experts_per_token': config.num_experts_per_token,
            'expert_capacity': config.expert_capacity,
            'hidden_size': config.hidden_size,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'load_balancing_loss_weight': config.load_balancing_loss_weight,
            'num_users': args.num_users,
            'num_items': args.num_items,
            'max_context_features': args.max_context_features
        }
        
        with open(args.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved configuration to {args.config_path}")
    
    logger.info("Model preparation completed successfully!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Prepare MoE recommendation model')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, default='data/train.json',
                       help='Path to training data file')
    parser.add_argument('--val-data', type=str, default=None,
                       help='Path to validation data file')
    parser.add_argument('--num-users', type=int, default=1000000,
                       help='Number of users')
    parser.add_argument('--num-items', type=int, default=500000,
                       help='Number of items')
    parser.add_argument('--max-context-features', type=int, default=10,
                       help='Maximum number of context features')
    
    # Model arguments
    parser.add_argument('--num-experts', type=int, default=8,
                       help='Number of experts')
    parser.add_argument('--experts-per-token', type=int, default=2,
                       help='Number of experts per token')
    parser.add_argument('--expert-capacity', type=int, default=64,
                       help='Expert capacity')
    parser.add_argument('--hidden-size', type=int, default=512,
                       help='Hidden size')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--load-balancing-weight', type=float, default=0.01,
                       help='Load balancing loss weight')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU training')
    
    # Output arguments
    parser.add_argument('--model-path', type=str, default='models/moe_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--config-path', type=str, default='models/config.json',
                       help='Path to save the model configuration')
    parser.add_argument('--history-path', type=str, default='models/training_history.json',
                       help='Path to save training history')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.config_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.history_path), exist_ok=True)
    
    # Prepare model
    prepare_model(args)


if __name__ == "__main__":
    main() 