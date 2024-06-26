# used https://github.com/karpathy/nanoGPT/blob/master/train.py as a reference

import torch
print("Pytorch version: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())

import sys
import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
from ast import literal_eval

from model import GPTConfig, GPT


pytorch_version = torch.__version__
cuda_available = torch.cuda.is_available()
torch.manual_seed(1337)

# any of these parameters can be changed with command line arguments. For example:
#     python train.py --batch_size=32

# data
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12
block_size = 1024
start_prime_x_train = 1
end_prime_x_train = 2000
start_prime_x_test = 4001 # make sure there's a bit of a gap here, since it can train on block_size characters after the end_prime_x_train
end_prime_x_test = 6000
train_backwards = False # train to predict the next lower prime rather than the next larger prime

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
always_save_checkpoint = True # if True, always save a checkpoint after each eval

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# logging
wandb_log = False
wandb_project = "prime_guesser"
wandb_run_name = "default"
computer = "lee_pc"
gpu = "Nvidia 3090"

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
for arg in sys.argv[1:]:
    assert arg.startswith('--')
    key, val = arg.split('=')
    key = key[2:]
    if key in globals():
        try:
            # attempt to eval it it (e.g. if bool, number, or etc)
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            # if that goes wrong, just use the string
            attempt = val
        # ensure the types match ok
        assert type(attempt) == type(globals()[key])
        # cross fingers
        print(f"Overriding: {key} = {attempt}")
        globals()[key] = attempt
    else:
        raise ValueError(f"Unknown config key: {key}")
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# only support 1 GPU for now
master_process = True
seed_offset = 0
ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# same as in prime_studies.ipynb
def get_prime_list(x_min, x_max):
    # from https://github.com/kimwalisch/primesieve
    os.system(f"primesieve {x_min} {x_max} -p > generated_primes.txt")
    with open("generated_primes.txt", "r") as f:
        prime_strs = f.read().splitlines()
    return np.array([int(s) for s in prime_strs])

def get_prime_chars(x_min, num_chars):
    cum_chars = ""
    curr_x = x_min
    while len(cum_chars) < num_chars:
        primes = get_prime_list(curr_x, curr_x*2)
        for p in primes:
            cum_chars += str(p)
            cum_chars += ","
        curr_x *= 2
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

# this generates a batch on the fly
# might be slower than a pre-generated file, but should be able to handle more data (might have to try both)
def get_batch(split):
    if split == 'train':
        start = start_prime_x_train
        end = end_prime_x_train
    else:
        start = start_prime_x_test
        end = end_prime_x_test

    # don't need -block_size here, since it's all generated on the fly (note that we can use characters after end)
    ix = torch.randint(start, end, (batch_size,))
    x_list = []
    y_list = []
    for b in range(batch_size):
        i_start = ix[b].item()
        primes_one_extra = get_prime_toks(i_start, block_size+1)
        x_list.append(primes_one_extra[:-1])
        y_list.append(primes_one_extra[1:])
    x = torch.stack([torch.from_numpy(x_b.astype(np.int64)) for x_b in x_list])
    y = torch.stack([torch.from_numpy(y_b.astype(np.int64)) for y_b in y_list])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# this gets a batch from a pre-generated file
"""def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y"""

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

print("Initializing model")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
model_args['vocab_size'] = 11 # 0-9 plus separator
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

print("Initializing optimizer")
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("compiling the model...")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

print("training done")

