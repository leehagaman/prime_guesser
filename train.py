# used https://github.com/karpathy/nanoGPT/blob/master/train.py as a reference

from prime_functions import get_prime_toks, convert_toks_to_nums, convert_nums_to_toks

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
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 512
block_size = 128
start_prime_x_train = 100_000
end_prime_x_train = 500_000
start_prime_x_test = 600_000 # make sure there's a bit of a gap here, since it can train on block_size characters after the end_prime_x_train
end_prime_x_test = 1_000_000

# I/O
out_dir = 'out'
eval_interval = 100
log_interval = 20
eval_iters = 1
always_save_checkpoint = False

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
n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.0
bias = False

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

# using command line arguments to change the above parameters
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
        assert type(attempt) == type(globals()[key])
        print(f"Overriding: {key} = {attempt}")
        globals()[key] = attempt
    else:
        raise ValueError(f"Unknown config key: {key}")
config = {k: globals()[k] for k in config_keys} # useful for logging

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


# this generates a batch on the fly
# might be slower than a pre-generated file, but should be able to handle more data (might have to try both)
# this function generates a batch from several non-sequential regions of primes
def get_batch_nonsequential(split, batch_size=batch_size):
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

# this function generates the batches sequentially, which is faster to generate
def get_batch(split, batch_size=batch_size):
    if split == 'train':
        start = start_prime_x_train
        end = end_prime_x_train
    else:
        start = start_prime_x_test
        end = end_prime_x_test

    start_point = np.random.randint(start, end)

    x_list = []
    y_list = []
    prime_toks = get_prime_toks(start_point, block_size * batch_size + 1)

    for b in range(batch_size):
        x = prime_toks[b*block_size : (b+1)*block_size]
        y = prime_toks[b*block_size+1 : (b+1)*block_size+1]
        x_list.append(x)
        y_list.append(y)

    x = torch.stack([torch.from_numpy(x_b.astype(np.int64)) for x_b in x_list])
    y = torch.stack([torch.from_numpy(y_b.astype(np.int64)) for y_b in y_list])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


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
    print("compiling the model...", end="", flush=True)
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    print(" done")


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

print("Starting Training Loop")
X, Y = get_batch('train') # fetch the very first batch
t_start_training = time.time()
t_start_iteration = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:

    t_set_learning_rate_start = time.time()
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    t_set_learning_rate_end = time.time()

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:

        print(f"    step {iter_num} losses: ", end="", flush=True)
        losses = estimate_loss()
        print(f"train {losses['train']:.4f}, val {losses['val']:.4f}")
        
        num_seeds = 5
        num_prints = 10
        model.eval()
        with torch.no_grad():
            eval_true_toks, _ = get_batch('val', batch_size=1)
            eval_true_nums = convert_toks_to_nums(eval_true_toks[0].cpu().numpy())
            num_eval_true_nums = len(eval_true_nums)
            seed_nums = eval_true_nums[:num_seeds]
            seed_toks = torch.from_numpy(np.array(convert_nums_to_toks(seed_nums)).reshape(1, -1)).to(device)
            true_nums = eval_true_nums[num_seeds:]
            print("    example generation, seed:", seed_nums)
            print("    truth, pred:")
            model_output_toks = model.generate(seed_toks, 100).cpu().numpy()
            pred_toks = model_output_toks[0, seed_toks.shape[1]:]
            pred_nums = convert_toks_to_nums(pred_toks)
            if len(pred_nums) < num_prints:
                pred_nums = np.concatenate((pred_nums, [-1] * (num_prints - len(pred_nums))))
            for i in range(num_prints):
                print("        ", true_nums[i], pred_nums[i])
        model.train()

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
    t_forward_backward_start = time.time()
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            t_start_forward = time.time()
            logits, loss = model(X, Y)
            t_end_forward = time.time()
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        t_start_get_batch = time.time()
        X, Y = get_batch('train')
        t_end_get_batch = time.time()
        # backward pass, with gradient scaling if training in fp16
        t_start_backward = time.time()
        scaler.scale(loss).backward()
        t_end_backward = time.time()
    t_forward_backward_end = time.time()

    t_step_start = time.time()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    t_step_end = time.time()

    # timing and logging
    t_end_iteration = time.time()
    dt_iteration = t_end_iteration - t_start_iteration
    t_start_iteration = t_end_iteration

    dt_set_learning_rate = t_set_learning_rate_start - t_set_learning_rate_end
    dt_forward = t_end_forward - t_start_forward
    dt_backward = t_end_backward - t_start_backward
    dt_forward_backward = t_forward_backward_end - t_forward_backward_start
    dt_get_batch = t_end_get_batch - t_start_get_batch
    dt_step = t_step_end - t_step_start

    frac_set_learning_rate = dt_set_learning_rate / dt_iteration
    frac_forward = dt_forward / dt_iteration
    frac_backward = dt_backward / dt_iteration
    frac_forward_backward = dt_forward_backward / dt_iteration
    frac_get_batch = dt_get_batch / dt_iteration
    frac_step = dt_step / dt_iteration

    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        t_loss_calc_start = time.time()
        lossf = loss.item() * gradient_accumulation_steps
        t_loss_calc_end = time.time()
        dt_loss_calc = t_loss_calc_end - t_loss_calc_start
        frac_less_calc = dt_loss_calc / dt_iteration
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt_iteration)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt_iteration*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        # this only works for loss_estimate with log_interval=1
        #print(f"    times, set lr: {frac_set_learning_rate*100:.1f}, forward: {frac_forward*100:.1f}%, backward: {frac_backward*100:.1f}%, foward/backward: {frac_forward_backward*100:.1f}, get_batch: {frac_get_batch*100:.1f}%, step: {frac_step*100:.1f}%, loss calc: {frac_less_calc*100:.1f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

print("training done")

