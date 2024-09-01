# modified from https://github.com/karpathy/nanoGPT/blob/master/sample.py

"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

from prime_functions import get_prime_toks, convert_toks_to_nums, convert_nums_to_toks

import argparse
parser = argparse.ArgumentParser(description="Generate primes using a trained model")
parser.add_argument('--max_new_tokens', type=int, required=True, help='Maximum number of new tokens to generate')
parser.add_argument('--temperature', type=float, required=True, help='Temperature for sampling')
parser.add_argument('--seed_primes', nargs='+', type=int, required=True, help='Seed primes to start generation')
args = parser.parse_args()
max_new_tokens = args.max_new_tokens
temperature = args.temperature
seed_primes = args.seed_primes

# Output directory
out_dir = 'out'


top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

seed_prime_toks = convert_nums_to_toks(seed_primes)

x = torch.tensor(seed_prime_toks, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        nums = convert_toks_to_nums(y[0].tolist()[len(seed_prime_toks):])
        for num in nums:
            print(num, end=", ")
