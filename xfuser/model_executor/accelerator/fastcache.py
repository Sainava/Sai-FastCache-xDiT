import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import math

from xfuser.logger import init_logger

logger = init_logger(__name__)


class FastCacheAccelerator(nn.Module):
    """
    FastCache: A hidden-state-level caching and compression framework
    for accelerating DiT inference by eliminating redundant computations.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        cache_ratio_threshold: float = 0.05,
        motion_threshold: float = 0.1,
        significance_level: float = 0.05,
        cache_enabled: bool = True
    ):
        super().__init__()
        self.model = model
        self.cache_ratio_threshold = cache_ratio_threshold
        self.motion_threshold = motion_threshold
        self.significance_level = significance_level
        self.cache_enabled = cache_enabled
        
        # Cache states
        self.prev_hidden_states = None
        self.bg_hidden_states = None
        self.cache_hits = 0
        self.total_steps = 0
        
        # Initialize cache adaptation parameters
        self.beta0 = 0.01
        self.beta1 = 0.5
        self.beta2 = -0.002
        self.beta3 = 0.00005
        
        # For statistics
        self.layer_cache_hits = {}
        
        # Linear approximation for static tokens
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        elif hasattr(model, 'hidden_size'):
            hidden_size = model.hidden_size
        elif hasattr(model, 'dim'):
            hidden_size = model.dim
        else:
            hidden_size = 3072
        self.cache_projection = nn.Linear(hidden_size, hidden_size)
        
        logger.info(f"Initialized FastCache with thresholds: cache={cache_ratio_threshold}, motion={motion_threshold}")
    
    def get_adaptive_threshold(self, variance_score, timestep):
        """Calculate adaptive threshold based on variance and timestep"""
        normalized_timestep = timestep / 1000.0  # Normalize timestep to [0,1] range
        return (self.beta0 + 
                self.beta1 * variance_score + 
                self.beta2 * normalized_timestep + 
                self.beta3 * normalized_timestep**2)
    
    def compute_relative_change(self, current, previous):
        """Compute relative change between current and previous hidden states"""
        if previous is None:
            return float('inf')
            
        # Compute Frobenius norm of difference
        diff_norm = torch.norm(current - previous, p='fro')
        prev_norm = torch.norm(previous, p='fro')
        
        # Avoid division by zero
        if prev_norm == 0:
            return float('inf')
            
        return (diff_norm / prev_norm).item()
    
    def should_use_cache(self, hidden_states, timestep):
        """Determine if cached states should be used based on statistical test"""
        if not self.cache_enabled or self.prev_hidden_states is None:
            return False
            
        # Compute relative change
        delta = self.compute_relative_change(hidden_states, self.prev_hidden_states)
        
        # Compute threshold based on chi-square distribution
        n, d = hidden_states.shape[1], hidden_states.shape[2]  # token count, hidden dim
        dof = n * d  # degrees of freedom
        
        # Chi-square threshold for given significance level
        # Approximate chi-square using normal distribution for large DOF
        z = 1.96  # z-score for 95% confidence (significance_level=0.05)
        chi2_threshold = dof + z * math.sqrt(2 * dof)
        statistical_threshold = math.sqrt(chi2_threshold / dof)
        
        # Adaptive threshold based on timestep
        adaptive_threshold = self.get_adaptive_threshold(delta, timestep)
        
        # Final threshold combines both statistical and adaptive thresholds
        # We use both to ensure both statistical validity and context-specific adaptation
        final_threshold = max(self.cache_ratio_threshold, 
                              min(statistical_threshold, adaptive_threshold))
        
        # Log thresholds for debugging when significant differences occur
        if abs(statistical_threshold - adaptive_threshold) > 0.1:
            logger.debug(f"Thresholds - Statistical: {statistical_threshold:.4f}, "
                        f"Adaptive: {adaptive_threshold:.4f}, Final: {final_threshold:.4f}")
        
        return delta <= final_threshold
    
    def compute_motion_saliency(self, hidden_states):
        """Compute motion saliency for each token"""
        if self.prev_hidden_states is None:
            return torch.ones(hidden_states.shape[1], device=hidden_states.device)
            
        # Compute token-wise differences
        token_diffs = (hidden_states - self.prev_hidden_states).abs()
        
        # Take max across feature dimension to get token saliency
        token_saliency = token_diffs.max(dim=-1)[0].squeeze(0)
        
        # Normalize saliency
        if token_saliency.max() > 0:
            token_saliency = token_saliency / token_saliency.max()
            
        return token_saliency
    
    def forward(self, hidden_states, timestep=None, use_cached_states=True, layer_idx=None, **kwargs):
        """
        Process hidden states through FastCache
        
        Args:
            hidden_states: Input hidden states
            timestep: Current timestep
            use_cached_states: Whether to use cached states
            layer_idx: Current transformer layer index
            **kwargs: Additional arguments to pass to the model
        """
        self.total_steps += 1
        
        # If caching is disabled or it's the first step, execute normally
        if not self.cache_enabled or self.prev_hidden_states is None:
            if layer_idx is not None and layer_idx not in self.layer_cache_hits:
                self.layer_cache_hits[layer_idx] = 0
                
            output = self.model(hidden_states, **kwargs)
            self.prev_hidden_states = hidden_states.detach().clone()
            if self.bg_hidden_states is None:
                self.bg_hidden_states = hidden_states.detach().clone()
            return output
        
        # Analyze motion and determine caching strategy
        if use_cached_states and self.should_use_cache(hidden_states, timestep):
            # Cache hit - reuse previous states
            self.cache_hits += 1
            if layer_idx is not None:
                self.layer_cache_hits[layer_idx] = self.layer_cache_hits.get(layer_idx, 0) + 1
                
            # Apply linear projection instead of full transformer
            output = self.cache_projection(hidden_states)
            return output
        
        # Compute motion saliency for tokens
        motion_saliency = self.compute_motion_saliency(hidden_states)
        motion_mask = motion_saliency > self.motion_threshold
        
        # If significant motion is detected, process normally
        if motion_mask.sum() / motion_mask.numel() > 0.5:
            output = self.model(hidden_states, **kwargs)
        else:
            # Split tokens into motion and static tokens
            batch_size, seq_len, hidden_dim = hidden_states.shape
            motion_indices = torch.where(motion_mask)[0]
            static_indices = torch.where(~motion_mask)[0]
            
            if len(motion_indices) > 0:
                # Process motion tokens through full transformer
                motion_states = hidden_states.index_select(1, motion_indices)
                motion_output = self.model(motion_states, **kwargs)
                
                # Process static tokens through linear projection
                static_states = hidden_states.index_select(1, static_indices)
                static_output = self.cache_projection(static_states)
                
                # Merge outputs
                output = hidden_states.clone()
                output.index_copy_(1, motion_indices, motion_output)
                output.index_copy_(1, static_indices, static_output)
            else:
                # All tokens are static, use linear projection
                output = self.cache_projection(hidden_states)
        
        # Update cache
        self.prev_hidden_states = hidden_states.detach().clone()
        # Update background state with exponential moving average
        alpha = 0.9
        self.bg_hidden_states = alpha * self.bg_hidden_states + (1 - alpha) * hidden_states.detach().clone()
        
        return output
    
    def get_cache_hit_ratio(self):
        """Return the cache hit ratio"""
        if self.total_steps == 0:
            return 0.0
        return self.cache_hits / self.total_steps
    
    def get_layer_hit_stats(self):
        """Return per-layer cache hit statistics"""
        stats = {}
        for layer_idx, hits in self.layer_cache_hits.items():
            stats[f"Layer {layer_idx}"] = hits / self.total_steps if self.total_steps > 0 else 0.0
        return stats
    
    def reset_stats(self):
        """Reset cache statistics"""
        self.cache_hits = 0
        self.total_steps = 0
        self.layer_cache_hits = {}


class FastCacheTransformerWrapper(nn.Module):
    """
    Wrapper for transformer blocks to apply FastCache acceleration
    """
    
    def __init__(self, transformer_block, cache_ratio_threshold=0.05, motion_threshold=0.1):
        super().__init__()
        self.transformer = transformer_block
        self.accelerator = FastCacheAccelerator(
            transformer_block,
            cache_ratio_threshold=cache_ratio_threshold,
            motion_threshold=motion_threshold
        )
        
    def forward(self, hidden_states, timestep=None, **kwargs):
        return self.accelerator(hidden_states, timestep=timestep, **kwargs) 