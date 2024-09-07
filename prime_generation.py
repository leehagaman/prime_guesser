import os
import numpy as np
import ollama
import time
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import warnings
from prime_functions import convert_toks_to_nums, convert_nums_to_toks


def get_llama_primes(prompt, max_output_count=None, print_stream=False, delay=0.05, disable_cache=False):
    if disable_cache or print_stream:
        return recalculate_llama_primes(prompt, max_output_count, print_stream, delay)
    else:
        return cached_get_llama_primes(prompt, max_output_count, print_stream, delay)


def cached_get_llama_primes(prompt, max_output_count, print_stream, delay):
    if os.path.exists("llama_cache.pkl"):
        with open("llama_cache.pkl", "rb") as f:
            cache = pickle.load(f)
        if (prompt, max_output_count) in cache:
            return cache[(prompt, max_output_count)]
    else:
        cache = {}
    result = recalculate_llama_primes(prompt, max_output_count, print_stream, delay)
    cache[(prompt, max_output_count)] = result
    with open("llama_cache.pkl", "wb") as f:
        pickle.dump(cache, f)
    return result

def delete_llama_cache():
    if os.path.exists("llama_cache.pkl"):
        os.remove("llama_cache.pkl")

def print_llama_cache():
    if os.path.exists("llama_cache.pkl"):
        with open("llama_cache.pkl", "rb") as f:
            cache = pickle.load(f)
        for prompt in cache:
            print(prompt, ":", cache[prompt])
    else:
        print("No llama cache found")


def recalculate_llama_primes(prompt, max_output_count, print_stream, delay):
    
    # experimentally, need this delay when calling ollama.generate in quick succession
    # the generation takes substantially longer anyway, so this isn't a big deal
    time.sleep(delay)
    valid_chars = '0123456789, \n'
    response_str = ""
    if print_stream: print("\nPrinting llama response stream:")
    num_complete_numbers_output = 0
    try:
        for part in ollama.generate('llama3_8b_text_fp16_zero_seed_zero_temp:latest', prompt, stream=True):
            if max_output_count != None and num_complete_numbers_output >= max_output_count:
                break
            part_str = part["response"]
            all_valid_chars = True
            for c in part_str:
                if c == ",":
                    num_complete_numbers_output += 1
                if c not in valid_chars:
                    all_valid_chars = False
                    break
            if not all_valid_chars:
                if print_stream: print("\nstopping, got something other than a number/comma: ", part)
                break
            if print_stream: print(part_str, end="")
            response_str += part_str
            if part["done"]:
                break
    except Exception as e:
        if print_stream: print(f"Failed to generate!")
        return [np.nan]
    if response_str[-1] == ",":
        response_str = response_str[:-1]
    split_response_strs = response_str.split(",")
    no_punc_split_response_strs = ["".join([c for c in s if c.isdigit()]) for s in split_response_strs]
    nums_str = [int(x) for x in no_punc_split_response_strs]
    return np.array(nums_str)


def get_model_primes(max_output_count, max_new_tokens, temperature, seed_primes):
    # modified from https://github.com/karpathy/nanoGPT/blob/master/sample.py

    out_dir = 'out'

    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    seed_prime_toks = convert_nums_to_toks(seed_primes)

    x = torch.tensor(seed_prime_toks, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            nums = convert_toks_to_nums(y[0].tolist()[len(seed_prime_toks):])

    if max_output_count != None and len(nums) > max_output_count:
        nums = nums[:max_output_count]
            
    return np.array(nums)



