# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
from pathlib import Path

class TokenShapeCaptureHook:
    """
    A PyTorch forward hook used to capture the output tensors of a module (like a Transformer block),
    save them to disk, and log their shape (specifically tracking sequence length for token reduction).
    """
    def __init__(self, block_name: str, save_dir: str):
        self.block_name = block_name
        self.save_dir = Path(save_dir)
        # Ensure the directory is created
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0
        self.log_file = self.save_dir / "shape_log.txt"

    def __call__(self, module, input, output):
        # Extract the output tensor safely (accounting for if the block returns a tuple)
        if isinstance(output, tuple):
            out_tensor = output[0]
        else:
            out_tensor = output

        # Extract the shape of this tensor
        shape = out_tensor.shape
        # Extract the sequence length, which is usually dimension 1: shape[1]
        seq_len = shape[1] if len(shape) > 1 else None

        # Save the detached, CPU-moved tensor
        save_filename = f"{self.block_name}_step_{self.call_count}.pt"
        save_path = self.save_dir / save_filename
        torch.save(out_tensor.detach().cpu(), save_path)

        # Append the call_count and the tensor.shape to a running text file
        with open(self.log_file, "a") as f:
            f.write(f"call_count={self.call_count} | shape={list(shape)} | seq_len={seq_len}\n")

        # Increment the call_count
        self.call_count += 1


class MockToMABlock(nn.Module):
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        seq_len = x.shape[1]
        
        # Simulate token merging by reducing sequence length by half
        new_seq_len = seq_len // 2
        
        # In a real model, this would be a complex merge operation.
        # Here we just slice the tensor to simulate the shape reduction.
        return x[:, :new_seq_len, :]


def main():
    import shutil
    import os
    
    save_dir = "test_toma_output"
    
    # Cleanup previous run if exists
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    print(f"Setting up MockToMABlock and TokenShapeCaptureHook...")
    block = MockToMABlock()
    hook = TokenShapeCaptureHook(block_name="mock_toma_block", save_dir=save_dir)
    
    # Register the hook
    handle = block.register_forward_hook(hook)
    
    # Create a dummy input tensor
    # shape: (batch_size, sequence_length, hidden_dim)
    print("Creating dummy input tensor of shape (1, 4096, 3072)")
    x = torch.randn(1, 4096, 3072)
    
    print("Running through the block 4 times...")
    for i in range(4):
        x = block(x)
        
    # Remove the hook
    handle.remove()
    
    # Verify outputs
    pt_files = list(Path(save_dir).glob("*.pt"))
    assert len(pt_files) == 4, f"Assertion Failed: Expected 4 .pt files, found {len(pt_files)}"
    
    print("\n✅ Verification Passed: 4 .pt files were created.")
    
    print("\nContents of shape_log.txt:")
    print("-" * 40)
    with open(Path(save_dir) / "shape_log.txt", "r") as f:
        print(f.read())
    print("-" * 40)

if __name__ == "__main__":
    main()
