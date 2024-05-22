import os
import numpy as np
import ollama
import time
import pickle

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

    if print_stream: print("Printing llama response stream:")

    num_complete_numbers_output = 0

    #for part in ollama.generate('llama3:8b-text-fp16', prompt, stream=True):
    try:
        for part in ollama.generate('llama3_8b_text_fp16_zero_seed_zero_temp:latest', prompt, stream=True):

            if max_output_count != None and num_complete_numbers_output >= max_output_count:
                break

            part_str = part["response"]

            # check if it contains any letters

            all_valid_chars = True
            for c in part_str:
                if c == ",":
                    num_complete_numbers_output += 1
                if c not in valid_chars:
                    all_valid_chars = False
                    break

            if not all_valid_chars:
                #if print_stream: print("\nstopping, got something other than a number/comma: '", part["response"] + "'")
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
