# pyrefly: ignore [missing-import]
import torch
import os
import shutil

# Standard Diffusers import
# pyrefly: ignore [missing-import]
from diffusers import FluxPipeline
# pyrefly: ignore [missing-import]
from transformers import BitsAndBytesConfig
from hidden_state_capture_hook import HiddenStateCaptureHook

def main():
    model_id = "black-forest-labs/FLUX.1-schnell" 
    
    # 1. Read HF_TOKEN securely
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found. Please set it before running.")

    print(f"Loading {model_id} via standard FluxPipeline in 4-bit...")

    # MATCHED QUANTIZATION CONFIG WITH TOMA FOR NUMERICAL STABILITY
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # Critical for Flux to prevent NaNs
        bnb_4bit_quant_type="nf4"
    )
    
    # 2. Pipeline Initialization
    pipe = FluxPipeline.from_pretrained(
        model_id, 
        quantization_config=quant_config,
        token=hf_token
    )
    
    # 3. Memory Safety: Prevent OOM on remote machine
    pipe.enable_model_cpu_offload() 

    # 4. Set up the Hooks for Depth Ablation (Early, Middle, Late)
    target_block_indices = [2, 10, 18]
    output_dir = "./flux_hidden_states"
    hooks = []
    handles = []

    print("Setting up hooks for multiple transformer blocks...")
    for idx in target_block_indices:
        hook = HiddenStateCaptureHook(
            block_name=f"flux_block_{idx}", 
            save_dir=output_dir
        )
        hooks.append(hook)

        # Locate the block and attach the hook
        if hasattr(pipe.transformer, 'single_transformer_blocks'):
            target_block = pipe.transformer.single_transformer_blocks[idx]
            print(f"Attaching hook to 'single_transformer_blocks' at index {idx}.")
        elif hasattr(pipe.transformer, 'transformer_blocks'):
            target_block = pipe.transformer.transformer_blocks[idx]
            print(f"Attaching hook to 'transformer_blocks' at index {idx}.")
        else:
            raise AttributeError("Could not find recognizable transformer blocks.")

        handles.append(target_block.register_forward_hook(hook))
    
    # 5. Execution: Semantic Variance and Temporal Extension
    prompts = [
        "A red apple on a white table",
        "A high tech neural network visualization, highly detailed, cyberpunk style",
        "The concept of time dissolving into a chaotic vortex"
    ]

    print("\nStarting Multi-Prompt Generation (10 steps each)...")
    for i, prompt in enumerate(prompts):
        print(f"\n--- Running Prompt {i+1}/3: '{prompt}' ---")
        image = pipe(
            prompt=prompt,
            num_inference_steps=10, # Increased to match ToMA for 1:1 comparison
            guidance_scale=0.0
        ).images[0]
    
    print("\nAll generations complete.")
    
    # Clean up hooks
    for handle in handles:
        handle.remove()
    print("All hooks removed.")
    
    # 6. Auto-Delivery: Zip the results
    zip_filename = "flux_baseline_results"
    shutil.make_archive(zip_filename, 'zip', output_dir)
    print(f"\nSUCCESS: Data saved! Results zipped successfully into {zip_filename}.zip")

if __name__ == "__main__":
    main()