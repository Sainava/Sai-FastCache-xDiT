# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
from pathlib import Path
from typing import Any
import shutil

class HiddenStateCaptureHook:
    def __init__(self, block_name: str, save_dir: str):
        self.block_name = block_name
        self.save_dir = Path(save_dir)
        # Ensure the save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0

    def __call__(self, module: nn.Module, input: Any, output: torch.Tensor):
        # Extract the output tensor, detach it, and move it to the CPU.
        # We add a small check to gracefully handle if the module returns a tuple.
        if isinstance(output, torch.Tensor):
            hidden_state = output.detach().cpu()
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            hidden_state = output[0].detach().cpu()
        else:
            # Fallback for other return types
            hidden_state = output.detach().cpu()

        # Format the filename
        filename = f"{self.block_name}_step_{self.call_count}.pt"
        filepath = self.save_dir / filename
        
        # Save as a .pt file
        torch.save(hidden_state, filepath)
        
        # Increment the call count
        self.call_count += 1

def main():
    # Target directory for testing
    test_dir = "./test_hidden_states"
    
    # 1. Create a simple dummy model using nn.Sequential with three nn.Linear layers
    # Setting in_features and out_features to 3072 to match the dummy tensor shape
    model = nn.Sequential(
        nn.Linear(3072, 3072),
        nn.Linear(3072, 3072),
        nn.Linear(3072, 3072)
    )
    
    # 2. Instantiate our HiddenStateCaptureHook for the second nn.Linear layer
    hook = HiddenStateCaptureHook(block_name='block_1', save_dir=test_dir)
    
    # 3. Register the hook to that specific layer (index 1 is the second layer)
    handle = model[1].register_forward_hook(hook)
    
    # 4. Create a dummy input tensor of random numbers with shape (1, 4096, 3072)
    dummy_input = torch.randn(1, 4096, 3072)
    
    # 5. Run a mock 'denoising loop': use a for loop that runs 4 times
    for i in range(4):
        _ = model(dummy_input)
        
    # Remove the hook after we're done
    handle.remove()
    
    # 6. Add assertions to verify that exactly 4 .pt files were created
    save_path = Path(test_dir)
    pt_files = list(save_path.glob("*.pt"))
    
    assert len(pt_files) == 4, f"Expected 4 .pt files, but found {len(pt_files)}"
    
    for i in range(4):
        expected_filename = f"block_1_step_{i}.pt"
        file_path = save_path / expected_filename
        assert file_path.exists(), f"Expected file {expected_filename} not found."
        
    print(f"Test passed successfully: Verified {len(pt_files)} hidden state files were created in '{test_dir}'.")
    
    # Clean up the test directory after verification
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    main()
