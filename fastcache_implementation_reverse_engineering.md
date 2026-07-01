# FastCache Implementation Reverse Engineering

This report reconstructs the FastCache caching mechanism exactly as implemented in the provided repository. 

---

## 1. Complete execution path

The core FastCache logic is implemented in `xfuser/model_executor/accelerator/fastcache.py`. The execution path follows these steps when a transformer block is wrapped with FastCache:

1. **`FastCacheTransformerWrapper.forward`**
   - **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
   - **Class**: `FastCacheTransformerWrapper`
   - **Function**: `forward`
   - **Lines**: 234-235
   ```python
   def forward(self, hidden_states, timestep=None, **kwargs):
       return self.accelerator(hidden_states, timestep=timestep, **kwargs)
   ```

2. **`FastCacheAccelerator.forward`**
   - **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
   - **Class**: `FastCacheAccelerator`
   - **Function**: `forward`
   - **Lines**: 127-198
   This is the main entry point where the cache decision is evaluated, calling helper methods.
   ```python
   def forward(self, hidden_states, timestep=None, use_cached_states=True, layer_idx=None, **kwargs):
       # ...
       if use_cached_states and self.should_use_cache(hidden_states, timestep):
           # Cache hit - reuse previous states
   ```

3. **`FastCacheAccelerator.should_use_cache`**
   - **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
   - **Class**: `FastCacheAccelerator`
   - **Function**: `should_use_cache`
   - **Lines**: 77-108
   Evaluates if the cache can be used for the entire layer.
   ```python
   def should_use_cache(self, hidden_states, timestep):
       # ...
       delta = self.compute_relative_change(hidden_states, self.prev_hidden_states)
       # ...
       return delta <= final_threshold
   ```

4. **`FastCacheAccelerator.compute_relative_change`**
   - **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
   - **Class**: `FastCacheAccelerator`
   - **Function**: `compute_relative_change`
   - **Lines**: 62-75
   Computes the relative delta.

5. **`FastCacheAccelerator.get_adaptive_threshold`**
   - **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
   - **Class**: `FastCacheAccelerator`
   - **Function**: `get_adaptive_threshold`
   - **Lines**: 54-60
   Computes a timestep-aware adaptive threshold.

---

## 2. Hidden-state flow

Hidden states progress through the caching mechanism as follows:

- **Created/Received**: Current hidden states are received as the `hidden_states` argument in the `forward` function of `FastCacheAccelerator`.
- **Compared**: They are compared against the previously stored states `self.prev_hidden_states` inside `compute_relative_change`.
- **Approximated**: If cached, they are passed through `self.cache_projection`.
- **Updated/Stored**: At the end of `forward`, the current `hidden_states` are cloned and stored for the next step. A background state is also updated.

**Relevant Code**:
- **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
- **Class**: `FastCacheAccelerator`
- **Function**: `forward`
- **Lines**: 192-196
```python
# Update cache
self.prev_hidden_states = hidden_states.detach().clone()
# Update background state with exponential moving average
alpha = 0.9
self.bg_hidden_states = alpha * self.bg_hidden_states + (1 - alpha) * hidden_states.detach().clone()
```

---

## 3. Delta computation

The relative change (delta) is computed mathematically as the Frobenius norm of the difference divided by the Frobenius norm of the previous state.

- **Tensors compared**: The current `hidden_states` and `self.prev_hidden_states`.
- **Norm used**: Frobenius norm (`p='fro'`).
- **Flattening**: No explicit flattening (PyTorch's `torch.norm` with `p='fro'` inherently handles multi-dimensional tensors as vectors for the calculation).
- **Tricks/Epsilon**: To prevent division by zero, it checks if `prev_norm == 0` and returns `float('inf')` if true. No epsilon is added to the denominator.
- **Preprocessing**: No preprocessing before the norm computation.

**Relevant Code**:
- **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
- **Class**: `FastCacheAccelerator`
- **Function**: `compute_relative_change`
- **Lines**: 62-75
```python
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
```

---

## 4. Cache decision

The cache decision relies on multiple if-statements in `forward`. FastCache makes both a macroscopic (entire block) and microscopic (token-level) cache decision.

**Macroscopic Decision**:
It evaluates `delta <= final_threshold` via `should_use_cache`.
- **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
- **Class**: `FastCacheAccelerator`
- **Function**: `forward`
- **Lines**: 152-160
```python
if use_cached_states and self.should_use_cache(hidden_states, timestep):
    # Cache hit - reuse previous states
    self.cache_hits += 1
    if layer_idx is not None:
        self.layer_cache_hits[layer_idx] = self.layer_cache_hits.get(layer_idx, 0) + 1
        
    # Apply linear projection instead of full transformer
    output = self.cache_projection(hidden_states)
    return output
```

**Token-level (Microscopic) Decision & Heuristics**:
If the block is not entirely cached, FastCache calculates a `motion_saliency` per token.
- **Warm-up / Special Cases**: If `self.prev_hidden_states` is None, caching is skipped (warm-up on the first step).
- **Motion heuristic threshold**: Tokens with motion saliency `> self.motion_threshold` (default 0.1) are flagged.
- **Fallback heuristic**: If more than 50% (`> 0.5`) of the tokens exceed the motion threshold, the entire transformer block is executed normally instead of splitting tokens.

**Relevant Code**:
- **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
- **Class**: `FastCacheAccelerator`
- **Function**: `forward`
- **Lines**: 163-173
```python
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
```

---

## 5. Threshold computation

The threshold used for the cache decision is an aggregation of a base threshold, a statistical upper bound, and an adaptive boundary.

**Parameters**:
- `cache_ratio_threshold`: Base threshold (default: `0.05`, configurable).
- `significance_level`: Determines the z-score (default: `0.05` -> `z=1.96`).
- Adaptive parameters: `beta0 = 0.01`, `beta1 = 0.5`, `beta2 = -0.002`, `beta3 = 0.00005` (initialized in `__init__`).

**Computation**:
The model calculates a `statistical_threshold` approximating a chi-square distribution with degrees of freedom `dof = num_tokens * hidden_dim`.
It calculates an `adaptive_threshold` using a polynomial of the variance (delta) and a normalized timestep.
The final threshold is `max(cache_ratio_threshold, min(statistical_threshold, adaptive_threshold))`.

**Relevant Code**:
- **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
- **Class**: `FastCacheAccelerator`
- **Function**: `should_use_cache`
- **Lines**: 85-102
```python
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
```

And the adaptive part:
- **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
- **Class**: `FastCacheAccelerator`
- **Function**: `get_adaptive_threshold`
- **Lines**: 54-60
```python
def get_adaptive_threshold(self, variance_score, timestep):
    """Calculate adaptive threshold based on variance and timestep"""
    normalized_timestep = timestep / 1000.0  # Normalize timestep to [0,1] range
    return (self.beta0 + 
            self.beta1 * variance_score + 
            self.beta2 * normalized_timestep + 
            self.beta3 * normalized_timestep**2)
```

---

## 6. Linear approximation

When states are cached (either at the block level or token level), they are passed through a lightweight linear projection instead of bypassing the computation altogether.

- **Initialized**: In `__init__`, mapping from `hidden_size` to `hidden_size`.
- **Executed**: In `forward`, applied to `hidden_states` on full block cache hit, or applied to `static_states` on partial cache hit.
- **Receives**: The current `hidden_states` (or a subset of them).
- **Returns**: A linearly approximated state matching the `hidden_size` dimension.

**Relevant Code**:
- **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
- **Class**: `FastCacheAccelerator`
- **Function**: `__init__`
- **Lines**: 49-50
```python
# Linear approximation for static tokens
self.cache_projection = nn.Linear(model.config.hidden_size, model.config.hidden_size)
```

Usage in partial cache hit:
- **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
- **Class**: `FastCacheAccelerator`
- **Function**: `forward`
- **Lines**: 180-182
```python
# Process static tokens through linear projection
static_states = hidden_states.index_select(1, static_indices)
static_output = self.cache_projection(static_states)
```

---

## 7. Transformer-level caching

FastCache avoids block execution entirely if a cache hit occurs, returning the projected hidden states. For a cache miss, it routes high-motion tokens through the transformer block (`self.model`) and static tokens through the projection.

**Relevant Code**:
- **Filename**: `xfuser/model_executor/accelerator/fastcache.py`
- **Class**: `FastCacheAccelerator`
- **Function**: `forward`
- **Lines**: 175-187
```python
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
```
The merged tensor continues natively down the pipeline.

---

## 8. Minimal decision logic

This block represents the bare minimum to recreate the mathematical decision of FastCache without the wrapping boilerplate.

```python
import torch
import math

def compute_delta(current, previous):
    diff_norm = torch.norm(current - previous, p='fro')
    prev_norm = torch.norm(previous, p='fro')
    return (diff_norm / prev_norm).item() if prev_norm != 0 else float('inf')

def compute_adaptive_threshold(delta, timestep):
    norm_t = timestep / 1000.0
    return 0.01 + (0.5 * delta) + (-0.002 * norm_t) + (0.00005 * norm_t**2)

def evaluate_cache_decision(current_hidden, prev_hidden, timestep, base_threshold=0.05):
    if prev_hidden is None:
        return False
        
    delta = compute_delta(current_hidden, prev_hidden)
    
    # Statistical chi2 approach
    n, d = current_hidden.shape[1], current_hidden.shape[2]
    dof = n * d
    chi2_threshold = dof + 1.96 * math.sqrt(2 * dof)
    statistical_threshold = math.sqrt(chi2_threshold / dof)
    
    adaptive_threshold = compute_adaptive_threshold(delta, timestep)
    
    final_threshold = max(base_threshold, min(statistical_threshold, adaptive_threshold))
    
    return delta <= final_threshold
```

---

## 9. Important implementation details

1. **EMA on Background States**: `self.bg_hidden_states` is maintained via an exponential moving average (alpha = 0.9) but it is actually *never read or utilized* anywhere else in the code, suggesting it might be an orphaned debug variable or part of an unfinished feature.
   ```python
   # Line 195: Update background state with exponential moving average
   alpha = 0.9
   self.bg_hidden_states = alpha * self.bg_hidden_states + (1 - alpha) * hidden_states.detach().clone()
   ```
2. **Motion Saliency Normalization Issue**: Saliency is normalized independently at every step using `token_saliency.max()`. If the overall difference across the entire sequence is extremely small (nearly static), the normalization artificially scales the largest small-error to 1.0, causing tokens to incorrectly trigger the motion mask (`> self.motion_threshold`).
   ```python
   # Line 122: Normalize saliency
   if token_saliency.max() > 0:
       token_saliency = token_saliency / token_saliency.max()
   ```
3. **Hardcoded Z-Score**: The confidence interval logic hardcodes `z = 1.96` on line 91, so passing `significance_level` to the initializer actually has no effect on the z-score used for the chi-square threshold.
4. **Motion Heuristic Fallback**: A hardcoded ceiling forces execution of the full transformer if `> 50%` of tokens have motion, bypassing the linear projection split entirely.
5. **No Deep Copy during Token Replacement**: The `output = hidden_states.clone()` creates a new tensor, into which the updated static/motion projections are written, preventing in-place operations from messing up upstream gradient requirements (though this runs in inference normally).
