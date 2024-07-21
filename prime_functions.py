import os
import numpy as np
from mpmath import li

def get_prime_list(x_min, x_max):

    #print(f"called get_prime_list({x_min}, {x_max})")

    # originally tried to get python bindings from https://github.com/shlomif/primesieve-python
    # didn't work with conda or pip, I think it's just old with limited version compatibilities
    # this method with a file is probably just as good

    # from https://github.com/kimwalisch/primesieve
    os.system(f"primesieve {x_min} {x_max} -p > generated_primes_{x_min}_{x_max}.txt")

    with open(f"generated_primes_{x_min}_{x_max}.txt", "r") as f:
        prime_strs = f.read().splitlines()

    os.system(f"rm generated_primes_{x_min}_{x_max}.txt")

    return np.array([int(s) for s in prime_strs if s != ""])


def pi(x):
    os.system(f"primesieve {x} -c > generated_primes.txt")

    with open("generated_primes.txt", "r") as f:
        prime_str = f.read().splitlines()[-1].split()[-1]

    return int(prime_str)

def pis(x):

    if x == 0:
        return [0]

    os.system(f"primesieve {x} -p > generated_primes.txt")

    with open("generated_primes.txt", "r") as f:
        prime_strs = f.read().splitlines()

    primes = [int(s) for s in prime_strs]

    pis = np.zeros(x, dtype=int)
    for p in primes:
        pis[p-1:] += 1

    return pis

def pi_approx_x_over_logx(x):
    return x / np.log(x)

def pis_approx_x_over_logx(x):
    return np.arange(1, x+1) / np.log(np.arange(1, x+1))

def pi_approx_Lix(x):
    return li(x, offset=True)

def pis_approx_Lix(x):
    return np.array([li(i, offset=True) for i in range(1, x+1)])


def nth_prime(n):

    # 1-indexed, 1st prime is 2, 2nd is 3, etc.

    os.system(f"primesieve -n {n} -p > generated_primes.txt")

    with open("generated_primes.txt", "r") as f:
        prime_str = f.read().splitlines()[0]

    return int(prime_str)

def nth_prime_list(n):

    largest_prime = nth_prime(n)
    
    os.system(f"primesieve {largest_prime} -p > generated_primes.txt")

    with open("generated_primes.txt", "r") as f:
        prime_strs = f.read().splitlines()

    return np.array([int(s) for s in prime_strs])


def nth_prime_approx_nlogn(n):
    return int(n * np.log(n))

def nth_prime_approx_better(n):
    # https://en.wikipedia.org/wiki/Prime_number_theorem#Approximations_for_the_nth_prime_number

    val = n * (
            np.log(n)
          + np.log(np.log(n)) - 1
          + (np.log(np.log(n)) - 2) / np.log(n)
          - (np.log(np.log(n))**2 - 6 * np.log(np.log(n)) + 11) / (2*np.log(n)**2)
        )
    
    if np.isfinite(val) == False:
        return np.nan
    
    return int(val)

def count_k_tuples_by_k(x, k):
    # This gets the count of all k-tuples, which means for any tuple of length k, subject to an "admissibility" condition
    # https://en.wikipedia.org/wiki/Prime_k-tuple
    if k < 2:
        raise ValueError("k must be greater than or equal to 2")
    elif k > 6:
        raise ValueError("k must be less than or equal to 6")
    os.system(f"primesieve {x} -c{k} > generated_primes.txt")
    with open("generated_primes.txt", "r") as f:
        prime_str = f.read().splitlines()[-1].split()[-1]
    return int(prime_str)

def count_k_tuples(x, tup, print_tuples=False):
    max_diff = max(tup)
    primes = get_prime_list(0, x + max_diff)
    ret = 0
    for i in range(len(primes)):
        p = primes[i]
        valid = True
        for t in tup:
            if p + t not in primes:
                valid = False
                break
        if valid:
            if print_tuples: print("  valid:", end=" ")
            if print_tuples: 
                for t in tup: print(p + t, end=" ")
            if print_tuples: print("")
            ret += 1
    return ret

def get_counts_k_tuples(x, tup):
    max_diff = max(tup)
    primes = get_prime_list(0, x + max_diff)
    ret = np.zeros(x, dtype=int)
    for i in range(len(primes)):
        if i > len(primes) - max_diff:
            continue
        p = primes[i]
        valid = True
        for t in tup:
            if p + t not in primes:
                valid = False
                break
        if valid:
            ret[p-1:] += 1
    return ret


def count_twin_primes(x): return count_k_tuples(x, (0, 2)) # count_k_tuples_by_k(x, 2) should also work
def count_cousin_primes(x): return count_k_tuples(x, (0, 4))
def count_sexy_primes(x): return count_k_tuples(x, (0, 6))
def count_triplet_primes(x): return count_k_tuples(x, (0, 2, 6)) + count_k_tuples(x, (0, 4, 6)) # count_k_tuples_by_k(x, 3) should also work
def count_sexy_triplet_primes(x): return count_k_tuples(x, (0, 6, 12))
def count_quadruplet_primes(x): return count_k_tuples(x, (0, 2, 6, 8))
def count_sexy_quadruplet_primes(x): return count_k_tuples(x, (0, 6, 12, 18))
def count_quintuplet_primes(x): return count_k_tuples(x, (0, 2, 6, 8, 12)) + count_k_tuples(x, (0, 4, 6, 10, 12))
def count_sextuplet_primes(x): return count_k_tuples(x, (0, 4, 6, 10, 12, 16))

def get_counts_twin_primes(x): return get_counts_k_tuples(x, (0, 2))
def get_counts_cousin_primes(x): return get_counts_k_tuples(x, (0, 4))
def get_counts_sexy_primes(x): return get_counts_k_tuples(x, (0, 6))
def get_counts_triplet_primes(x): return get_counts_k_tuples(x, (0, 2, 6)) + get_counts_k_tuples(x, (0, 4, 6))
def get_counts_sexy_triplet_primes(x): return get_counts_k_tuples(x, (0, 6, 12))
def get_counts_quadruplet_primes(x): return get_counts_k_tuples(x, (0, 2, 6, 8))
def get_counts_sexy_quadruplet_primes(x): return get_counts_k_tuples(x, (0, 6, 12, 18))
def get_counts_quintuplet_primes(x): return get_counts_k_tuples(x, (0, 2, 6, 8, 12)) + get_counts_k_tuples(x, (0, 4, 6, 10, 12))
def get_counts_sextuplet_primes(x): return get_counts_k_tuples(x, (0, 4, 6, 10, 12, 16))


def get_prime_chars(x_min, num_chars):
    cum_chars = ""
    curr_x = x_min
    while len(cum_chars) < num_chars:
        primes = get_prime_list(curr_x, curr_x + 10_000)
        for p in primes:
            cum_chars += str(p)
            cum_chars += ","
        curr_x += 10_000
    return cum_chars[:num_chars]

def get_prime_toks(x_min, num_toks):
    cum_chars = get_prime_chars(x_min, num_toks)
    ret = []
    for c in cum_chars:
        if c == ",":
            ret.append(10)
        else:
            ret.append(int(c))
    return np.array(ret)

def convert_toks_to_nums(toks):

    num_str = ""
    for t in toks:
        if t == 10:
            num_str += ","
        else:
            num_str += str(t)

    split_str = num_str.split(",")
    non_empty_split_str = [s for s in split_str if s != ""]

    return np.array([int(s) for s in non_empty_split_str])
