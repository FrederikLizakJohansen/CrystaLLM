import sys
import yaml
sys.path.append(".")
import os
import io
from dataclasses import dataclass
import tarfile
import argparse
import pickle
import numpy as np
from tqdm.auto import tqdm

from contextlib import nullcontext
import torch

from crystallm._model import GPTConfig, GPT
from crystallm._tokenizer import CIFTokenizer


def load_model(config):
    
    # Load checkpoint
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=config.device)

    model_args = checkpoint["model_args"]
    model_config = GPTConfig(**model_args)
    model = GPT(model_config).to(config.device)

    state_dict = checkpoint["model"]
    # Fix the keys of the state dict per CrystaLLM
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)

    return model

def generate_samples(config):

    # Load Model
    model = load_model(config)

    # Open meta data and get cond_len
    meta_path = os.path.join(config.dataset_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    cond_seq_len = meta['cond_seq_len']
    
    # Init tokenizer
    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    # Get conditionings
    cond_ids = np.memmap(os.path.join(config.dataset_dir, f"cond_{config.split}.bin"), dtype=np.uint16, mode="r")

    # Get cifs fnames
    with open(os.path.join(config.dataset_dir, f"fnames_{config.split}.txt"), 'r') as f:
        cif_paths = [line.strip() for line in f.readlines()]

    print(cif_paths)

    # Order and stack conditioning data
    n_data= len(cond_ids) // cond_seq_len
    if config.n_data > 0:
        n_data = min(n_data, config.n_data)
    cond_tensors = torch.stack([torch.from_numpy((cond_ids[i*cond_seq_len:(i+1)*cond_seq_len]).astype(np.int64)) for i in range(n_data)]).to(config.device)
    cif_paths = cif_paths[:n_data]
    assert len(cif_paths) == len(cond_tensors)
    
    # Generate structures and cif strings
    with torch.no_grad():
        with ctx:
            model.eval()
            generated_cifs = []
            for i, (fname, cond) in tqdm(enumerate(zip(cif_paths, cond_tensors)), total=len(cif_paths), desc='Generating CIFs...', leave=False):
                gens = []
                for _ in tqdm(range(config.n_repeats), total=config.n_repeats, desc='Generating repeats...', leave=False):
                    start_index = torch.tensor(tokenizer.encode(["data_"])).to(device='cuda').unsqueeze(0)
                    out = model.generate(start_index, cond.unsqueeze(0), max_new_tokens=config.max_new_tokens, top_k=config.top_k)
                    output = decode(out[0].tolist())
                    gens.append(output)
                    
                generated_cifs.append((fname.split(".")[0], gens))

                if i >= config.debug_max:
                    break

    if config.out == "":
        for id, gens in generated_cifs:
            print(id + "\n")
            for i, g in enumerate(gens):
                print("Generation no.", i)
                print(g)
                print("\n")
            print("-"*10)
        return
    # Save tarball or print
    with tarfile.open(config.out, "w:gz") as tar:
        for id, gens in tqdm(generated_cifs, desc=f"Writing CIF files to {config.out}..."):
            for i, cif in enumerate(gens):
                cif_file = tarfile.TarInfo(name=f"{id}__{i+1}.cif")
                cif_bytes = cif.encode("utf-8")
                cif_file.size = len(cif_bytes)
                tar.addfile(cif_file, io.BytesIO(cif_bytes))

@dataclass
class SampleDefaults:
    out_dir: str = "debug_model" # Path to model checkpoint
    dataset_dir: str = "dataset/CHILI-100K_small" # Path to dataset
    split: str = "train" # Data split
    out: str = "" # Path of gzipped tarball of generated cifs
    top_k: int = 5
    max_new_tokens: int = 6000
    device: str = 'cuda'
    temperature: float = 1.0
    seed: int = 42
    dtype: str = "float16"
    n_repeats: int = 1
    n_data: int = 0 # Default 0 is all data
    prompt: str = "data_"
    debug_max: int = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CIFs from datasplit")
    parser.add_argument("--config", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--dataset_dir",  type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--n_repeats", type=int)
    parser.add_argument("--n_data", type=int)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    # Parse yaml
    if args.config is not None:
        with open(args.config, "r") as file:
            config_dict = yaml.safe_load(file)
            config = SampleDefaults(**config_dict)
    else:
        config = SampleDefaults()

    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            config.__setattr__(key, value)

    ptdtype = {"float32": torch.float32, "bffloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = nullcontext() if config.device == "cpu" else torch.amp.autocast(device_type=config.device)

    # Run training script
    generate_samples(config)
