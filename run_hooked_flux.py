# pyrefly: ignore [missing-import]
import torch
import shutil
import os
# pyrefly: ignore [missing-import]
from diffusers import FluxPipeline
# pyrefly: ignore [missing-import]
from transformers import BitsAndBytesConfig
from hidden_state_capture_hook import HiddenStateCaptureHook

def main():
    # 1. Revert to the official, gated model for real mathematical data
    model_id = "black-forest-labs/FLUX.1-schnell" 
    
    # 2. Securely fetch the token from the environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found. Please set it before running.")

    print(f"Loading {model_id}...")

    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # Critical for Flux to prevent NaNs
    bnb_4bit_quant_type="nf4"
)
    
    # 3. Pass the token to the pipeline safely
    pipe = FluxPipeline.from_pretrained(
        model_id, 
        quantization_config=quant_config,  # <-- Load the quantized model
        token=hf_token
    )
    
    # 4. Use CPU offloading to prevent OOM on their machine
    pipe.enable_model_cpu_offload() 

    target_block_index = 10
    output_dir = "./flux_hidden_states"
    
    hook = HiddenStateCaptureHook(
        block_name=f"flux_block_{target_block_index}", 
        save_dir=output_dir
    )

    if hasattr(pipe.transformer, 'single_transformer_blocks'):
        target_block = pipe.transformer.single_transformer_blocks[target_block_index]
        print(f"Found 'single_transformer_blocks'. Hooking into index {target_block_index}.")
    elif hasattr(pipe.transformer, 'transformer_blocks'):
        target_block = pipe.transformer.transformer_blocks[target_block_index]
        print(f"Found 'transformer_blocks'. Hooking into index {target_block_index}.")
    else:
        raise AttributeError("Could not find recognizable transformer blocks.")

    handle = target_block.register_forward_hook(hook)
    
    prompt = "A high tech neural network visualization, highly detailed"
    print(f"Running generation. Hidden states will be saved to: {hook.save_dir}")
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=4, 
        guidance_scale=0.0,
    ).images[0]
    
    handle.remove()
    print(f"Generation complete. Captured {hook.call_count} hidden states. Hook removed.")
    
    # 5. Auto-zip the folder so it is easy for them to send back
    zip_filename = "flux_baseline_results"
    shutil.make_archive(zip_filename, 'zip', output_dir)
    print(f"\nSUCCESS: Data saved! Please send the '{zip_filename}.zip' file back to me.")

if __name__ == "__main__":
    main()