import sys
import re
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

#from crystallm._model import GPTConfig, GPT
#from crystallm._tokenizer import CIFTokenizer

from crystallm import (
    GPTConfig,
    GPT,
    CIFTokenizer,
    extract_space_group_symbol,
    replace_symmetry_operators,
)

from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock, CifParser
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp

def return_operators(cif_str, space_group_symbol):
    space_group = SpaceGroup(space_group_symbol)
    symmetry_ops = space_group.symmetry_ops

    loops = []
    data = {}
    symmops = []
    for op in symmetry_ops:
        v = op.translation_vector
        symmops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

    ops = [op.as_xyz_string() for op in symmops]
    data["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
    data["_symmetry_equiv_pos_as_xyz"] = ops

    loops.append(["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"])

    symm_block = str(CifBlock(data, loops, "")).replace("data_\n", "")

    #pattern = r"(loop_\n_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\n\s*1\s*'x, y, z'\n)"
    pattern = r"(loop_\n\s*_symmetry_equiv_pos_site_id\s*_symmetry_equiv_pos_as_xyz\n\s*1\s*'x, y, z')"

    cif_str_updated = re.sub(pattern, symm_block, cif_str)

    return cif_str_updated


def load_model(config):
    
    # Load checkpoint
    ckpt_path = os.path.join(config.model_dir, 'ckpt.pt')
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
    prefix_size = meta['prefix_size']
    
    # Init tokenizer
    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    # Get Prefix
    prefix_x_ids = np.memmap(os.path.join(config.dataset_dir, f"prefix_x_{config.split}.bin"), dtype=np.uint16, mode="r")
    prefix_y_ids = np.memmap(os.path.join(config.dataset_dir, f"prefix_y_{config.split}.bin"), dtype=np.uint16, mode="r")

    # Get cifs fnames
    with open(os.path.join(config.dataset_dir, f"fnames_{config.split}.txt"), 'r') as f:
        cif_paths = [line.strip() for line in f.readlines()]

    # Order and stack conditioning data
    n_data = len(prefix_x_ids) // prefix_size
    if config.n_data > 0:
        n_data = min(n_data, config.n_data)

    prefix_x_tensors = torch.stack([torch.from_numpy((prefix_x_ids[i*prefix_size:(i+1)*prefix_size]).astype(np.int64)) for i in range(n_data)]).to(config.device)
    prefix_y_tensors = torch.stack([torch.from_numpy((prefix_y_ids[i*prefix_size:(i+1)*prefix_size]).astype(np.int64)) for i in range(n_data)]).to(config.device)

    cif_paths = cif_paths[:n_data]

    # Check out path
    print_to_consol = True if config.out == "" else False
    
    # Debug max
    if config.debug_max is None:
        config.debug_max = n_data
    
    # Generate structures and cif strings
    with torch.no_grad():
        with ctx:
            model.eval()
            generated_cifs = []
            for i, (fname, prefix_x, prefix_y) in tqdm(enumerate(zip(cif_paths, prefix_x_tensors, prefix_y_tensors)), total=len(cif_paths), desc='Generating CIFs...', leave=False, disable=print_to_consol):
                if i >= config.debug_max:
                    break
                gens = []
                filename = fname.split(".")[0]
                print(filename)
                for j in tqdm(range(config.n_repeats), total=config.n_repeats, desc='Generating repeats...', leave=False, disable=print_to_consol):
                    input_string = ["data_"]
                    if config.prompt != "":
                        input_string = input_string + tokenizer.tokenize_cif(config.prompt)

                    # Ablation
                    #prefix_x = torch.randint(1,100, prefix_x.shape).to(device='cuda') * 0 + 99
                    #prefix_y = torch.randint(1,100, prefix_y.shape).to(device='cuda') * 0 + 99

                    # Generate
                    start_index = torch.tensor(tokenizer.encode(input_string)).to(device='cuda').unsqueeze(0)
                    if print_to_consol:
                        print("Generation no.", j+1, ":")
                        out = model.generate_and_print(start_index, prefix_x.unsqueeze(0), prefix_y.unsqueeze(0), max_new_tokens=config.max_new_tokens, top_k=config.top_k)
                        print("-"*30)
                        print()
                    else:
                        out = model.generate(start_index, prefix_x.unsqueeze(0), prefix_y.unsqueeze(0), max_new_tokens=config.max_new_tokens, top_k=config.top_k)
                    output = decode(out[0].tolist())

                    # Postprocess
                    if config.post_process:
                        space_group_symbol = extract_space_group_symbol(output)
                        if space_group_symbol is not None and space_group_symbol != "P 1":
                            output = return_operators(output, space_group_symbol)
                    gens.append(output)
                    
                if not print_to_consol:
                    generated_cifs.append((filename, gens))


    if not print_to_consol:
        with tarfile.open(config.out, "w:gz") as tar:
            for id, gens in tqdm(generated_cifs, desc=f"Writing CIF files to {config.out}..."):
                for i, cif in enumerate(gens):
                    cif_file = tarfile.TarInfo(name=f"{id}__{i+1}.cif")
                    cif_bytes = cif.encode("utf-8")
                    cif_file.size = len(cif_bytes)
                    tar.addfile(cif_file, io.BytesIO(cif_bytes))

@dataclass
class SampleDefaults:
    model_dir: str = "" # Path to model checkpoint
    dataset_dir: str = "" # Path to dataset
    split: str = "train" # Data split
    out: str = "" # Path of gzipped tarball of generated cifs
    top_k: int = None
    max_new_tokens: int = 500
    device: str = 'cuda'
    temperature: float = 1.0
    seed: int = 42
    dtype: str = "float16"
    n_repeats: int = 1
    n_data: int = 0 # Default 0 is all data
    prompt: str = ""
    debug_max: int = None
    post_process: bool = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CIFs from datasplit")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--dataset_dir",  type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--n_repeats", type=int)
    parser.add_argument("--n_data", type=int)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--debug_max", type=int)
    parser.add_argument("--post_process", action='store_true')
    args = parser.parse_args()

    config = SampleDefaults()
    
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)


    # Assertions
    assert config.model_dir != "", "[model_dir] cannot be empty"
    assert config.dataset_dir != "", "[dataset_dir] cannot be empty"

    ptdtype = {"float32": torch.float32, "bffloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = nullcontext() if config.device == "cpu" else torch.amp.autocast(device_type=config.device)

    # Run training script
    generate_samples(config)
