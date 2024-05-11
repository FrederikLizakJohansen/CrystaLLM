"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
"""
import sys
sys.path.append(".")
import os
from dataclasses import dataclass
from typing import Union
import math
import time

from crystallm import parse_config
from omegaconf import OmegaConf
import numpy as np
import torch
import pickle
from contextlib import nullcontext
from tqdm.auto import tqdm

from crystallm import (
    GPT,
    GPTConfig,
)


@dataclass
class TrainDefaults:
    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # data
    dataset: str = ""  # the path to the folder containing the .bin files with encoded tokens
    gradient_accumulation_steps: int = 40  # used to simulate larger batch sizes
    batch_size: int = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 2048  # context of up to `block_size` previous characters
    accumulative_pbar: bool = True

    # Prefix
    prefix_x_vocab_size: int = 20
    prefix_y_vocab_size: int = 20
    prefix_size: int = 100

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95  # make a bit bigger because number of tokens per iter is small
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for; not super necessary potentially
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster
    underrep_p: float = 0.0
    validate: bool = False  # whether to evaluate the model using the validation set


def read_start_indices(
    max_start_index: int,
    data_dir: str,
    starts_fname: str,
    on_condition: bool = True,
    required: bool = False,
) -> Union[torch.Tensor, None]:
    start_indices = None
    starts_path = os.path.join(data_dir, starts_fname)
    if on_condition:
        if os.path.exists(starts_path):
            print(f"Reading start indices from {starts_path}...", flush=True)
            with open(starts_path, "rb") as f:
                start_indices = torch.tensor(pickle.load(f))  # should be sorted
            # remove indices that would result in out-of-bounds sequences
            start_indices = start_indices[start_indices <= max_start_index]
        elif required:
            raise Exception(f"Expected to find a file in dataset dir named '{starts_fname}'")
    return start_indices


if __name__ == "__main__":
    C = parse_config(TrainDefaults)

    print("Using configuration:", flush=True)
    print(OmegaConf.to_yaml(C))

    print(f"Creating {C.out_dir}...", flush=True)
    os.makedirs(C.out_dir, exist_ok=True)

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")
    
    meta_path = os.path.join(C.dataset, "meta.pkl")
    cif_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        # Load meta data
        cif_size = meta['cif_size']
        cif_vocab_size = meta['cif_vocab_size']

        prefix_size = meta['prefix_size']
        prefix_x_vocab_size = meta['prefix_x_vocab_size']
        prefix_y_vocab_size = meta['prefix_y_vocab_size']

        scattering_type = meta['scattering_type']

        print(f"Found cif_vocab_size = {cif_vocab_size} (inside {meta_path})", flush=True)
        print(f"Found prefix_x_vocab_size = {prefix_x_vocab_size} (inside {meta_path})", flush=True)
        print(f"Found prefix_y_vocab_size = {prefix_y_vocab_size} (inside {meta_path})", flush=True)
        print(f"Found cif_size = {cif_size} (inside {meta_path})", flush=True)
        print(f"Found cif_vocab_size = {cif_vocab_size} (inside {meta_path})", flush=True)
        print(f"Found scattering_type = {scattering_type} (inside {meta_path})", flush=True)

    train_data = np.memmap(os.path.join(C.dataset, "train.bin"), dtype=np.uint16, mode="r")
    prefix_x_train_data = np.memmap(os.path.join(C.dataset, "prefix_x_train.bin"), dtype=np.uint16, mode="r")
    prefix_y_train_data = np.memmap(os.path.join(C.dataset, "prefix_y_train.bin"), dtype=np.uint16, mode="r")
    
    val_data = np.memmap(os.path.join(C.dataset, "val.bin"), dtype=np.uint16, mode="r") if C.validate else None
    prefix_x_val_data = np.memmap(os.path.join(C.dataset, "prefix_x_val.bin"), dtype=np.uint16, mode="r") if C.validate else None
    prefix_y_val_data = np.memmap(os.path.join(C.dataset, "prefix_y_val.bin"), dtype=np.uint16, mode="r") if C.validate else None

    cif_start_indices = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts.pkl",
    )

    cif_start_indices_val = read_start_indices(
        max_start_index=(len(val_data) - C.block_size) if C.validate else -1,
        data_dir=C.dataset,
        starts_fname="starts_val.pkl",
        on_condition=C.validate,
    )

    cif_start_indices_underrep = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts_underrep.pkl",
        on_condition=C.underrep_p > 0,
        required=True,
    )
    
    def sample_block(array, block_idx, block_size, sample_idx, seq_len):
        end_index = (block_idx + block_size) % seq_len
        if end_index >= block_idx:
            return array[block_idx+sample_idx:end_index+sample_idx].astype(np.int64)
        else:
            return np.concatenate((array[block_idx+sample_idx:sample_idx+seq_len], array[sample_idx:end_index+sample_idx])).astype(np.int64)

    def get_batch(split: str, batch_size: int = C.batch_size, return_sample_idx: bool = False):
        # CIFs
        data = train_data if split == "train" else val_data

        # Prefix
        prefix_x_data = prefix_x_train_data if split == "train" else prefix_x_val_data
        prefix_y_data = prefix_y_train_data if split == "train" else prefix_y_val_data

        # Choose training samples
        num_data = len(data) // cif_size
        sample_idx = torch.randint(num_data, (batch_size,))

        # Choose blocsk
        block_idx = torch.randint(cif_size, (batch_size,))

        # Get x and y
        x = torch.stack([torch.from_numpy(sample_block(data, block_idx[i], C.block_size, sample_idx[i], cif_size)) for i in range(batch_size)])
        y = torch.stack([torch.from_numpy(sample_block(data, block_idx[i]+1, C.block_size, sample_idx[i], cif_size)) for i in range(batch_size)])

        # Get prefix
        prefix_x = torch.stack([torch.from_numpy((prefix_x_data[i*prefix_size:(i+1)*prefix_size]).astype(np.int64)) for i in sample_idx])
        prefix_y = torch.stack([torch.from_numpy((prefix_y_data[i*prefix_size:(i+1)*prefix_size]).astype(np.int64)) for i in sample_idx])

        # Send to device
        if C.device == "cuda":
            x = x.pin_memory().to(C.device, non_blocking=True)
            y = y.pin_memory().to(C.device, non_blocking=True)
            prefix_x = prefix_x.pin_memory().to(C.device, non_blocking=True)
            prefix_y = prefix_y.pin_memory().to(C.device, non_blocking=True)
        else:
            x = x.to(C.device)
            y = y.to(C.device)
            prefix_x = prefix_x.to(C.device)
            prefix_y = prefix_y.to(C.device)
        
        if not return_sample_idx:
            return x, y, prefix_x, prefix_y
        else:
            return x, y, prefix_x, prefix_y, sample_idx

    iter_num = 0
    best_val_loss = 1e9


    model_args = dict(n_layer=C.n_layer, n_head=C.n_head, n_embd=C.n_embd, block_size=C.block_size,
                      bias=C.bias, vocab_size=None, dropout=C.dropout, prefix_x_vocab_size = prefix_x_vocab_size, 
                      prefix_y_vocab_size = prefix_y_vocab_size, prefix_size = prefix_size)
    if C.init_from == "scratch":
        print("Initializing a new model from scratch...", flush=True)
        if cif_vocab_size is None:
            print("Defaulting to vocab_size of 371...", flush=True)
        model_args["vocab_size"] = cif_vocab_size if cif_vocab_size is not None else 371
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif C.init_from == "resume":
        print(f"Resuming training from {C.out_dir}...", flush=True)
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training;
        #  the rest of the attributes (e.g. dropout) can stay as desired
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif C.init_from == "finetune":
        print(f"Finetuning training from {C.out_dir}...", flush=True)
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training;
        #  the rest of the attributes (e.g. dropout) can stay as desired
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            elif 'prefix' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        print("number of trainable parameters: %.2fM" % (model.get_num_params(trainable=True)/1e6,), flush=True)

    # crop down the model block size if desired, using model surgery
    if C.block_size < model.config.block_size:
        model.crop_block_size(C.block_size)
        model_args["block_size"] = C.block_size  # so that the checkpoint will have the right value
    model.to(C.device)

    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    if C.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])

    if C.compile:
        print("Compiling the model (takes a ~minute)...", flush=True)
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, eval_iters in [("train", C.eval_iters_train), ("val", C.eval_iters_val)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, PREFIX_X, PREFIX_Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, PREFIX_X, PREFIX_Y, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < C.warmup_iters:
            return C.learning_rate * it / C.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > C.lr_decay_iters:
            return C.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return C.min_lr + coeff * (C.learning_rate - C.min_lr)

    # training loop
    X, Y, PREFIX_X, PREFIX_Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % C.eval_interval == 0:
            if C.validate:
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
            if (C.validate and losses["val"] < best_val_loss) or C.always_save_checkpoint:
                best_val_loss = losses["val"] if C.validate else 0.
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": dict(C),
                    }
                    print(f"saving checkpoint to {C.out_dir}...", flush=True)
                    torch.save(checkpoint, os.path.join(C.out_dir, "ckpt.pt"))
        if iter_num == 0 and C.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        small_step_pbar = tqdm(desc='Accumulating losses...', total=C.gradient_accumulation_steps, leave=False, disable=not C.accumulative_pbar)
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, PREFIX_X, PREFIX_Y, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, PREFIX_X, PREFIX_Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            small_step_pbar.update(1)
        small_step_pbar.close()
        # clip the gradient
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % C.log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(C.batch_size * C.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%", flush=True)
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > C.max_iters:
            break
