"""
Mixture of Experts (MoE) Transformer Model for Recommendation Systems

This module implements a scalable MoE architecture that activates only a subset
of experts per request to maintain low latency and cost efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import time

# Configure logging for monitoring expert activation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MoEConfig:
    """Configuration for MoE Transformer model"""
    num_experts: int = 8  # Total number of experts
    num_experts_per_token: int = 2  # Number of experts activated per request
    expert_capacity: int = 64  # Maximum tokens per expert
    hidden_size: int = 512  # Hidden dimension size
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers
    dropout: float = 0.1  # Dropout rate
    vocab_size: int = 50000  # Vocabulary size
    max_seq_length: int = 512  # Maximum sequence length
    load_balancing_loss_weight: float = 0.01  # Load balancing loss weight


class Expert(nn.Module):
    """
    Individual expert network in the MoE system.
    Each expert is a feed-forward network with specialized knowledge.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Expert-specific feed-forward network
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Expert activation tracking for monitoring
        self.activation_count = 0
        self.last_activation_time = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert network"""
        # Track expert activation for monitoring
        self.activation_count += 1
        self.last_activation_time = time.time()
        
        # Expert computation with residual connection
        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x + residual  # Residual connection
    
    def get_activation_stats(self) -> Dict:
        """Get expert activation statistics for monitoring"""
        return {
            'activation_count': self.activation_count,
            'last_activation_time': self.last_activation_time,
            'expert_id': id(self)
        }


class MoELayer(nn.Module):
    """
    Mixture of Experts layer that routes tokens to appropriate experts.
    Implements top-k routing for efficient expert activation.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.expert_capacity = config.expert_capacity
        self.hidden_size = config.hidden_size
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(config.hidden_size, config.dropout) 
            for _ in range(config.num_experts)
        ])
        
        # Router network for expert selection
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        
        # Load balancing loss tracking
        self.load_balancing_loss = 0.0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with expert routing and load balancing.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            output: Processed tensor
            metrics: Dictionary containing routing statistics
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute router logits for expert selection
        router_logits = self.router(x)  # (batch_size, seq_len, num_experts)
        
        # Apply top-k routing to select experts
        expert_weights, expert_indices = torch.topk(
            router_logits, 
            k=self.num_experts_per_token, 
            dim=-1
        )
        
        # Apply softmax to get expert probabilities
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Process each expert
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            
            if expert_mask.sum() > 0:
                # Get tokens for this expert
                expert_tokens = x[expert_mask]
                expert_weights_for_tokens = expert_weights[expert_mask]
                
                # Apply expert capacity limit
                if expert_tokens.size(0) > self.expert_capacity:
                    # Randomly sample tokens up to capacity
                    indices = torch.randperm(expert_tokens.size(0))[:self.expert_capacity]
                    expert_tokens = expert_tokens[indices]
                    expert_weights_for_tokens = expert_weights_for_tokens[indices]
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_tokens)
                
                # Weight the expert output
                weighted_output = expert_output * expert_weights_for_tokens.unsqueeze(-1)
                
                # Accumulate output
                output[expert_mask] += weighted_output
                expert_usage[expert_idx] = expert_tokens.size(0)
        
        # Compute load balancing loss
        self.load_balancing_loss = self._compute_load_balancing_loss(expert_usage)
        
        # Collect metrics for monitoring
        metrics = {
            'expert_usage': expert_usage.cpu().numpy(),
            'load_balancing_loss': self.load_balancing_loss.item(),
            'num_activated_experts': (expert_usage > 0).sum().item(),
            'routing_entropy': self._compute_routing_entropy(expert_weights)
        }
        
        return output, metrics
    
    def _compute_load_balancing_loss(self, expert_usage: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage"""
        total_tokens = expert_usage.sum()
        if total_tokens == 0:
            return torch.tensor(0.0, device=expert_usage.device)
        
        # Ideal uniform distribution
        ideal_usage = total_tokens / self.num_experts
        
        # Compute KL divergence from uniform distribution
        actual_probs = expert_usage / total_tokens
        ideal_probs = torch.ones_like(actual_probs) / self.num_experts
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        actual_probs = actual_probs + epsilon
        ideal_probs = ideal_probs + epsilon
        
        kl_divergence = torch.sum(actual_probs * torch.log(actual_probs / ideal_probs))
        
        return kl_divergence * self.config.load_balancing_loss_weight
    
    def _compute_routing_entropy(self, expert_weights: torch.Tensor) -> float:
        """Compute entropy of routing decisions for monitoring"""
        # Flatten weights and compute entropy
        flat_weights = expert_weights.view(-1, self.num_experts_per_token)
        entropy = -torch.sum(flat_weights * torch.log(flat_weights + 1e-8), dim=-1)
        return entropy.mean().item()


class MoETransformer(nn.Module):
    """
    Complete Mixture of Experts Transformer model for recommendations.
    Combines multiple MoE layers with attention mechanisms.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_heads, 
            dropout=config.dropout,
            batch_first=True
        )
        
        # MoE layers
        self.moe_layers = nn.ModuleList([
            MoELayer(config) for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for optimal training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass through the MoE Transformer.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Dictionary containing outputs and metrics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        
        # Process through transformer layers
        all_metrics = []
        
        for layer_idx, moe_layer in enumerate(self.moe_layers):
            # Self-attention
            attn_output, attn_weights = self.attention(
                x, x, x, 
                attn_mask=attention_mask,
                need_weights=True
            )
            x = self.layer_norm(x + attn_output)
            
            # MoE layer
            moe_output, moe_metrics = moe_layer(x)
            x = self.layer_norm(x + moe_output)
            
            # Collect metrics
            moe_metrics['layer_idx'] = layer_idx
            all_metrics.append(moe_metrics)
        
        # Final output projection
        logits = self.output_projection(x)
        
        # Compute loss (assuming classification task)
        if self.training:
            # For recommendation system, we might use different loss functions
            # This is a placeholder for the actual loss computation
            loss = torch.tensor(0.0, device=device)  # Placeholder
        else:
            loss = None
        
        return {
            'logits': logits,
            'loss': loss,
            'metrics': all_metrics,
            'attention_weights': attn_weights
        }
    
    def get_expert_statistics(self) -> Dict:
        """Get comprehensive statistics about expert usage"""
        stats = {
            'total_experts': self.config.num_experts,
            'experts_per_token': self.config.num_experts_per_token,
            'layer_stats': []
        }
        
        for layer_idx, moe_layer in enumerate(self.moe_layers):
            layer_stats = {
                'layer_idx': layer_idx,
                'expert_activations': []
            }
            
            for expert_idx, expert in enumerate(moe_layer.experts):
                expert_stats = expert.get_activation_stats()
                expert_stats['expert_idx'] = expert_idx
                layer_stats['expert_activations'].append(expert_stats)
            
            stats['layer_stats'].append(layer_stats)
        
        return stats


class RecommendationMoE(MoETransformer):
    """
    Specialized MoE model for recommendation systems.
    Includes user embedding and item embedding for personalized recommendations.
    """
    
    def __init__(self, config: MoEConfig, num_users: int, num_items: int):
        super().__init__(config)
        
        # User and item embeddings for personalization
        self.user_embedding = nn.Embedding(num_users, config.hidden_size)
        self.item_embedding = nn.Embedding(num_items, config.hidden_size)
        
        # Recommendation head
        self.recommendation_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)  # Binary relevance score
        )
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                context_features: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass for recommendation prediction.
        
        Args:
            user_ids: User IDs (batch_size,)
            item_ids: Item IDs (batch_size,)
            context_features: Optional context features (batch_size, feature_dim)
            
        Returns:
            Dictionary containing recommendation scores and metrics
        """
        batch_size = user_ids.shape[0]
        
        # Get user and item embeddings
        user_embeddings = self.user_embedding(user_ids)  # (batch_size, hidden_size)
        item_embeddings = self.item_embedding(item_ids)  # (batch_size, hidden_size)
        
        # Combine embeddings with context if available
        if context_features is not None:
            # Project context features to hidden size
            context_projection = nn.Linear(context_features.size(-1), self.config.hidden_size)
            context_embeddings = context_projection(context_features)
            combined_embeddings = user_embeddings + item_embeddings + context_embeddings
        else:
            combined_embeddings = user_embeddings + item_embeddings
        
        # Add sequence dimension for transformer processing
        x = combined_embeddings.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Process through transformer layers
        all_metrics = []
        
        for layer_idx, moe_layer in enumerate(self.moe_layers):
            # Self-attention (simplified for recommendation context)
            attn_output, attn_weights = self.attention(x, x, x)
            x = self.layer_norm(x + attn_output)
            
            # MoE layer
            moe_output, moe_metrics = moe_layer(x)
            x = self.layer_norm(x + moe_output)
            
            moe_metrics['layer_idx'] = layer_idx
            all_metrics.append(moe_metrics)
        
        # Get final representation
        final_representation = x.squeeze(1)  # (batch_size, hidden_size)
        
        # Compute recommendation scores
        recommendation_scores = self.recommendation_head(final_representation)
        recommendation_scores = torch.sigmoid(recommendation_scores).squeeze(-1)
        
        return {
            'recommendation_scores': recommendation_scores,
            'metrics': all_metrics,
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings,
            'final_representation': final_representation
        }


# Utility functions for model management
def create_moe_model(config: MoEConfig, num_users: int, num_items: int) -> RecommendationMoE:
    """Create and initialize a MoE recommendation model"""
    model = RecommendationMoE(config, num_users, num_items)
    return model


def load_moe_model(model_path: str, config: MoEConfig, 
                  num_users: int, num_items: int) -> RecommendationMoE:
    """Load a trained MoE model from disk"""
    model = create_moe_model(config, num_users, num_items)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model


def save_moe_model(model: RecommendationMoE, model_path: str):
    """Save a trained MoE model to disk"""
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


# Performance monitoring utilities
class MoEMonitor:
    """Monitor and track MoE model performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.expert_usage_history = []
    
    def update_metrics(self, metrics: Dict):
        """Update metrics history"""
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_expert_utilization(self) -> Dict:
        """Get current expert utilization statistics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 requests
        
        expert_usage = {}
        for metrics in recent_metrics:
            for layer_metrics in metrics.get('metrics', []):
                layer_idx = layer_metrics.get('layer_idx', 0)
                usage = layer_metrics.get('expert_usage', [])
                
                for expert_idx, usage_count in enumerate(usage):
                    key = f"layer_{layer_idx}_expert_{expert_idx}"
                    expert_usage[key] = expert_usage.get(key, 0) + usage_count
        
        return expert_usage
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]
        
        avg_load_balancing_loss = np.mean([
            m.get('load_balancing_loss', 0) 
            for metrics in recent_metrics 
            for m in metrics.get('metrics', [])
        ])
        
        avg_activated_experts = np.mean([
            m.get('num_activated_experts', 0)
            for metrics in recent_metrics
            for m in metrics.get('metrics', [])
        ])
        
        return {
            'avg_load_balancing_loss': avg_load_balancing_loss,
            'avg_activated_experts': avg_activated_experts,
            'total_requests': len(self.metrics_history),
            'recent_requests': len(recent_metrics)
        } 