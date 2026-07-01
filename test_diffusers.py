import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers"])

import diffusers
print("BitsAndBytesConfig in diffusers?", hasattr(diffusers, "BitsAndBytesConfig"))
if hasattr(diffusers, "BitsAndBytesConfig"):
    from diffusers import BitsAndBytesConfig
    from diffusers.quantizers.quantization_config import PipelineQuantizationConfig
    print("Is subclass?", issubclass(BitsAndBytesConfig, PipelineQuantizationConfig))
