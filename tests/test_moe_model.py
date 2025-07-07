"""
Unit tests for MoE Transformer model components.
Tests cover expert routing, load balancing, and recommendation generation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe_transformer import (
    MoEConfig, Expert, MoELayer, MoETransformer, 
    RecommendationMoE, MoEMonitor, create_moe_model
)


class TestMoEConfig:
    """Test MoE configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MoEConfig()
        
        assert config.num_experts == 8
        assert config.num_experts_per_token == 2
        assert config.expert_capacity == 64
        assert config.hidden_size == 512
        assert config.num_heads == 8
        assert config.num_layers == 6
        assert config.dropout == 0.1
        assert config.vocab_size == 50000
        assert config.max_seq_length == 512
        assert config.load_balancing_loss_weight == 0.01
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = MoEConfig(
            num_experts=4,
            num_experts_per_token=1,
            expert_capacity=32,
            hidden_size=256
        )
        
        assert config.num_experts == 4
        assert config.num_experts_per_token == 1
        assert config.expert_capacity == 32
        assert config.hidden_size == 256


class TestExpert:
    """Test individual expert network"""
    
    def test_expert_initialization(self):
        """Test expert initialization"""
        expert = Expert(hidden_size=512, dropout=0.1)
        
        assert expert.hidden_size == 512
        assert expert.activation_count == 0
        assert expert.last_activation_time == 0
    
    def test_expert_forward_pass(self):
        """Test expert forward pass"""
        expert = Expert(hidden_size=256, dropout=0.1)
        x = torch.randn(4, 256)  # Batch size 4, hidden size 256
        
        output = expert(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_expert_activation_tracking(self):
        """Test expert activation tracking"""
        expert = Expert(hidden_size=256)
        x = torch.randn(2, 256)
        
        initial_count = expert.activation_count
        initial_time = expert.last_activation_time
        
        expert(x)
        
        assert expert.activation_count == initial_count + 1
        assert expert.last_activation_time > initial_time
    
    def test_expert_statistics(self):
        """Test expert statistics collection"""
        expert = Expert(hidden_size=256)
        x = torch.randn(3, 256)
        
        expert(x)
        stats = expert.get_activation_stats()
        
        assert 'activation_count' in stats
        assert 'last_activation_time' in stats
        assert 'expert_id' in stats
        assert stats['activation_count'] == 1


class TestMoELayer:
    """Test MoE layer with expert routing"""
    
    def test_moe_layer_initialization(self):
        """Test MoE layer initialization"""
        config = MoEConfig(num_experts=4, num_experts_per_token=2)
        moe_layer = MoELayer(config)
        
        assert len(moe_layer.experts) == 4
        assert moe_layer.num_experts == 4
        assert moe_layer.num_experts_per_token == 2
    
    def test_moe_layer_forward_pass(self):
        """Test MoE layer forward pass"""
        config = MoEConfig(num_experts=4, num_experts_per_token=2)
        moe_layer = MoELayer(config)
        x = torch.randn(2, 3, 512)  # Batch size 2, seq len 3, hidden size 512
        
        output, metrics = moe_layer(x)
        
        assert output.shape == x.shape
        assert 'expert_usage' in metrics
        assert 'load_balancing_loss' in metrics
        assert 'num_activated_experts' in metrics
        assert 'routing_entropy' in metrics
    
    def test_expert_capacity_limit(self):
        """Test expert capacity limiting"""
        config = MoEConfig(num_experts=2, expert_capacity=2)
        moe_layer = MoELayer(config)
        x = torch.randn(10, 1, 256)  # More tokens than capacity
        
        output, metrics = moe_layer(x)
        
        # Should still produce output
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_load_balancing_loss(self):
        """Test load balancing loss computation"""
        config = MoEConfig(num_experts=4, load_balancing_loss_weight=0.01)
        moe_layer = MoELayer(config)
        x = torch.randn(2, 3, 256)
        
        _, metrics = moe_layer(x)
        
        assert 'load_balancing_loss' in metrics
        assert isinstance(metrics['load_balancing_loss'], float)
        assert metrics['load_balancing_loss'] >= 0
    
    def test_routing_entropy(self):
        """Test routing entropy computation"""
        config = MoEConfig(num_experts=4, num_experts_per_token=2)
        moe_layer = MoELayer(config)
        x = torch.randn(2, 3, 256)
        
        _, metrics = moe_layer(x)
        
        assert 'routing_entropy' in metrics
        assert isinstance(metrics['routing_entropy'], float)
        assert metrics['routing_entropy'] >= 0


class TestMoETransformer:
    """Test complete MoE Transformer model"""
    
    def test_transformer_initialization(self):
        """Test transformer initialization"""
        config = MoEConfig(num_layers=2)
        transformer = MoETransformer(config)
        
        assert len(transformer.moe_layers) == 2
        assert transformer.config == config
    
    def test_transformer_forward_pass(self):
        """Test transformer forward pass"""
        config = MoEConfig(num_layers=2, num_experts=4)
        transformer = MoETransformer(config)
        input_ids = torch.randint(0, 1000, (2, 10))  # Batch size 2, seq len 10
        
        output = transformer(input_ids)
        
        assert 'logits' in output
        assert 'metrics' in output
        assert 'attention_weights' in output
        assert output['logits'].shape == (2, 10, config.vocab_size)
    
    def test_transformer_with_attention_mask(self):
        """Test transformer with attention mask"""
        config = MoEConfig(num_layers=2)
        transformer = MoETransformer(config)
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        attention_mask[:, 5:] = 0  # Mask last 5 tokens
        
        output = transformer(input_ids, attention_mask)
        
        assert 'logits' in output
        assert 'attention_weights' in output
    
    def test_expert_statistics(self):
        """Test expert statistics collection"""
        config = MoEConfig(num_layers=2, num_experts=4)
        transformer = MoETransformer(config)
        
        stats = transformer.get_expert_statistics()
        
        assert 'total_experts' in stats
        assert 'experts_per_token' in stats
        assert 'layer_stats' in stats
        assert len(stats['layer_stats']) == 2


class TestRecommendationMoE:
    """Test specialized recommendation MoE model"""
    
    def test_recommendation_model_initialization(self):
        """Test recommendation model initialization"""
        config = MoEConfig()
        num_users = 1000
        num_items = 500
        
        model = RecommendationMoE(config, num_users, num_items)
        
        assert model.user_embedding.num_embeddings == num_users
        assert model.item_embedding.num_embeddings == num_items
    
    def test_recommendation_forward_pass(self):
        """Test recommendation forward pass"""
        config = MoEConfig(num_layers=2)
        model = RecommendationMoE(config, 1000, 500)
        user_ids = torch.randint(0, 1000, (4,))
        item_ids = torch.randint(0, 500, (4,))
        
        output = model(user_ids, item_ids)
        
        assert 'recommendation_scores' in output
        assert 'metrics' in output
        assert 'user_embeddings' in output
        assert 'item_embeddings' in output
        assert output['recommendation_scores'].shape == (4,)
        assert torch.all((output['recommendation_scores'] >= 0) & (output['recommendation_scores'] <= 1))
    
    def test_recommendation_with_context(self):
        """Test recommendation with context features"""
        config = MoEConfig(num_layers=2)
        model = RecommendationMoE(config, 1000, 500)
        user_ids = torch.randint(0, 1000, (3,))
        item_ids = torch.randint(0, 500, (3,))
        context_features = torch.randn(3, 10)  # 10 context features
        
        output = model(user_ids, item_ids, context_features)
        
        assert 'recommendation_scores' in output
        assert output['recommendation_scores'].shape == (3,)
    
    def test_recommendation_score_range(self):
        """Test that recommendation scores are in valid range"""
        config = MoEConfig()
        model = RecommendationMoE(config, 1000, 500)
        user_ids = torch.randint(0, 1000, (5,))
        item_ids = torch.randint(0, 500, (5,))
        
        output = model(user_ids, item_ids)
        scores = output['recommendation_scores']
        
        # Scores should be between 0 and 1 (sigmoid output)
        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)


class TestMoEMonitor:
    """Test MoE monitoring utilities"""
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = MoEMonitor()
        
        assert monitor.metrics_history == []
        assert monitor.expert_usage_history == []
    
    def test_metrics_update(self):
        """Test metrics update"""
        monitor = MoEMonitor()
        metrics = {'test': 'value', 'expert_usage': [1, 2, 3]}
        
        monitor.update_metrics(metrics)
        
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0] == metrics
    
    def test_metrics_history_limit(self):
        """Test metrics history size limit"""
        monitor = MoEMonitor()
        
        # Add more than 1000 metrics
        for i in range(1100):
            monitor.update_metrics({'index': i})
        
        # Should keep only last 1000
        assert len(monitor.metrics_history) == 1000
        assert monitor.metrics_history[-1]['index'] == 1099
    
    def test_expert_utilization(self):
        """Test expert utilization calculation"""
        monitor = MoEMonitor()
        
        # Add some test metrics
        metrics = {
            'metrics': [
                {
                    'layer_idx': 0,
                    'expert_usage': [1, 0, 2, 1]
                },
                {
                    'layer_idx': 1,
                    'expert_usage': [0, 1, 1, 0]
                }
            ]
        }
        
        monitor.update_metrics(metrics)
        utilization = monitor.get_expert_utilization()
        
        assert 'layer_0_expert_0' in utilization
        assert 'layer_0_expert_1' in utilization
        assert 'layer_1_expert_0' in utilization
    
    def test_performance_summary(self):
        """Test performance summary calculation"""
        monitor = MoEMonitor()
        
        # Add test metrics
        for i in range(10):
            metrics = {
                'metrics': [
                    {
                        'load_balancing_loss': 0.05,
                        'num_activated_experts': 2
                    }
                ]
            }
            monitor.update_metrics(metrics)
        
        summary = monitor.get_performance_summary()
        
        assert 'avg_load_balancing_loss' in summary
        assert 'avg_activated_experts' in summary
        assert 'total_requests' in summary
        assert 'recent_requests' in summary


class TestModelUtilities:
    """Test model utility functions"""
    
    def test_create_moe_model(self):
        """Test model creation utility"""
        config = MoEConfig()
        num_users = 1000
        num_items = 500
        
        model = create_moe_model(config, num_users, num_items)
        
        assert isinstance(model, RecommendationMoE)
        assert model.user_embedding.num_embeddings == num_users
        assert model.item_embedding.num_embeddings == num_items
    
    @patch('torch.load')
    @patch('torch.save')
    def test_model_save_load(self, mock_save, mock_load):
        """Test model save and load utilities"""
        from models.moe_transformer import save_moe_model, load_moe_model
        
        config = MoEConfig()
        model = create_moe_model(config, 1000, 500)
        
        # Test save
        save_moe_model(model, 'test_model.pth')
        mock_save.assert_called_once()
        
        # Test load
        mock_load.return_value = model.state_dict()
        loaded_model = load_moe_model('test_model.pth', config, 1000, 500)
        
        assert isinstance(loaded_model, RecommendationMoE)
        mock_load.assert_called_once()


class TestModelEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_input(self):
        """Test model with empty input"""
        config = MoEConfig()
        model = RecommendationMoE(config, 1000, 500)
        
        with pytest.raises(ValueError):
            user_ids = torch.tensor([])
            item_ids = torch.tensor([])
            model(user_ids, item_ids)
    
    def test_mismatched_input_sizes(self):
        """Test model with mismatched input sizes"""
        config = MoEConfig()
        model = RecommendationMoE(config, 1000, 500)
        
        with pytest.raises(RuntimeError):
            user_ids = torch.randint(0, 1000, (3,))
            item_ids = torch.randint(0, 500, (4,))  # Different size
            model(user_ids, item_ids)
    
    def test_invalid_user_item_ids(self):
        """Test model with invalid user/item IDs"""
        config = MoEConfig()
        model = RecommendationMoE(config, 1000, 500)
        
        # IDs outside valid range
        user_ids = torch.tensor([1001, 1002])  # Beyond num_users
        item_ids = torch.tensor([501, 502])     # Beyond num_items
        
        with pytest.raises(IndexError):
            model(user_ids, item_ids)
    
    def test_nan_input_handling(self):
        """Test model handling of NaN inputs"""
        config = MoEConfig()
        model = RecommendationMoE(config, 1000, 500)
        user_ids = torch.randint(0, 1000, (2,))
        item_ids = torch.randint(0, 500, (2,))
        context_features = torch.tensor([[1.0, float('nan')], [2.0, 3.0]])
        
        # Should handle NaN gracefully
        output = model(user_ids, item_ids, context_features)
        
        assert 'recommendation_scores' in output
        # Check that output doesn't contain NaN
        assert not torch.isnan(output['recommendation_scores']).any()


if __name__ == "__main__":
    pytest.main([__file__]) 