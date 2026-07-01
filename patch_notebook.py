import json

with open('/Users/sainavamodak/Desktop/Sai-FastCache-xDiT/official_fastcache_benchmark.ipynb', 'r') as f:
    nb = json.load(f)

patch_code = """
# Apply hotfixes for Flux inference bugs in FastCache
fastcache_file = 'xfuser/model_executor/accelerator/fastcache.py'
with open(fastcache_file, 'r') as f:
    content = f.read()

# Fix 1: Flux Transformer Block config
old_code_1 = "self.cache_projection = nn.Linear(model.config.hidden_size, model.config.hidden_size)"
new_code_1 = '''
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        elif hasattr(model, 'hidden_size'):
            hidden_size = model.hidden_size
        elif hasattr(model, 'dim'):
            hidden_size = model.dim
        else:
            hidden_size = 3072
        self.cache_projection = nn.Linear(hidden_size, hidden_size)
'''

if old_code_1 in content:
    content = content.replace(old_code_1, new_code_1)

# Fix 2: Flux timestep NoneType error
old_code_2 = '''    def get_adaptive_threshold(self, variance_score, timestep):
        \"\"\"Calculate adaptive threshold based on variance and timestep\"\"\"
        normalized_timestep = timestep / 1000.0  # Normalize timestep to [0,1] range'''

new_code_2 = '''    def get_adaptive_threshold(self, variance_score, timestep):
        \"\"\"Calculate adaptive threshold based on variance and timestep\"\"\"
        if timestep is None or (hasattr(timestep, 'shape') and len(timestep.shape) > 1):
            timestep = 500.0
        normalized_timestep = timestep / 1000.0  # Normalize timestep to [0,1] range'''

if old_code_2 in content:
    content = content.replace(old_code_2, new_code_2)

with open(fastcache_file, 'w') as f:
    f.write(content)
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '!git clone' in ''.join(cell['source']):
        cell['source'] = [
            "# Run this cell to clone the repository and install requirements on Kaggle\n",
            "import os\n",
            "if not os.path.exists('Sai-FastCache-xDiT'):\n",
            "    !git clone https://github.com/Sainava/Sai-FastCache-xDiT.git\n",
            "os.chdir('Sai-FastCache-xDiT')\n",
            "!pip install torch numpy diffusers transformers accelerate sentencepiece protobuf bitsandbytes yunchang distvae einops\n",
            "\n"
        ]
        cell['source'].extend([line + '\n' for line in patch_code.strip().split('\n')])
        break

with open('/Users/sainavamodak/Desktop/Sai-FastCache-xDiT/official_fastcache_benchmark.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
