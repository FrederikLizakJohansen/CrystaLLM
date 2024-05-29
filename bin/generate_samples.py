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

from pymatgen.analysis.diffraction.xrd import XRDCalculator

import warnings
warnings.filterwarnings("ignore")

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

def calculate_rmsd_cif(cif_content1, cif_content2):
    
    calc = XRDCalculator(symprec=0.1)
    parser = CifParser()
    
    # CIF 1
    parser.from_string(cif_content1)
    structure1 = parser.get_structures()[0]
    pattern1 = calc.get_pattern(structure1)

    # CIF 2
    parser.from_string(cif_content2)
    structure2 = parser.get_structures()[0]
    pattern2 = calc.get_pattern(structure2)

    # Unpack
    pos1, intens1 = pattern1
    pos2, intens2 = pattern2

    # Interpolate intensities to a common set of positions
    common_positions = np.union1d(pos1, pos2)
    intens1_interpolated = np.interp(common_positions, pos1, inten1)
    intens2_interpolated = np.interp(common_positions, pos2, intens2)

    # Calculate RMSD
    rmsd = np.sqrt(np.mean((intens1_interpolated - intens2_interpolated)**2))
    return rmsd, common_positions, intens1_interpolated, intens2_interpolated
    
def calculate_rmsd_prefix_cif(prefix_x, prefix_y, cif, lower_limit=None):
                        
    space_group_symbol = extract_space_group_symbol(cif)
    if space_group_symbol is not None and space_group_symbol != "P 1":
        cif = return_operators(cif, space_group_symbol)

    prefix_x = prefix_x.cpu().numpy()
    prefix_y = prefix_y.cpu().numpy()

    # CIF
    calc = XRDCalculator(symprec=0.1)
    parser = CifParser.from_string(cif)
    structure = parser.get_structures()[0]
    pattern = calc.get_pattern(structure)

    if lower_limit is not None:
        mask = pattern.y >= lower_limit
        x = pattern.x[mask]
        y = pattern.y[mask]
    else:
        x = pattern.x
        y = pattern.y

    # Unpack
    pos1, intens1 = prefix_x[prefix_x != 0], prefix_y[prefix_y != 0]
    pos2, intens2 = x, y

    # Interpolate intensities to a common set of positions
    common_positions = np.union1d(pos1, pos2)
    intens1_interpolated = np.interp(common_positions, pos1, intens1)
    intens2_interpolated = np.interp(common_positions, pos2, intens2)

    # Calculate RMSD
    rmsd = np.sqrt(np.mean((intens1_interpolated - intens2_interpolated)**2))
    return rmsd, common_positions, intens1_interpolated, intens2_interpolated

def generate_samples(config):
    
    ptdtype = {"float32": torch.float32, "bffloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = nullcontext() if config.device == "cpu" else torch.amp.autocast(device_type=config.device)

    # Load Model
    model = load_model(config)

    # Open meta data and get cond_len
    meta_path = os.path.join(config.dataset_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    #prefix_size = meta['prefix_size']
    
    # Init tokenizer
    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    # Get data and start indices
    data = np.memmap(os.path.join(config.dataset_dir, f"{config.split}.bin"), dtype=np.uint16, mode="r")
    start_indices = np.memmap(os.path.join(config.dataset_dir, f"start_indices_{config.split}.bin"), dtype=np.uint32, mode="r")

    # Get cifs fnames
    with open(os.path.join(config.dataset_dir, f"fnames_{config.split}.txt"), 'r') as f:
        cif_paths = [line.strip() for line in f.readlines()]

    # Order and stack conditioning data
    n_data = len(cif_paths)
    if config.n_data > 0:
        n_data = min(n_data, config.n_data)
    cif_paths = cif_paths[:n_data]

    # Check out path
    print_to_consol = True if config.out == "" else False
    
    # Debug max
    if config.debug_max is None:
        config.debug_max = n_data

    # Get index of "data_"
    cif_start_id = tokenizer.token_to_id["data_"]
    
    # Generate structures and cif strings
    with torch.no_grad():
        with ctx:
            model.eval()
            generated_cifs = []
            for i, (fname, start_idx) in tqdm(enumerate(zip(cif_paths, start_indices)), total=len(cif_paths), desc='Generating CIFs...', leave=False, disable=print_to_consol):
                if i >= config.debug_max:
                    break
                gens = []

                # Print filename
                filename = fname.split(".")[0]
                print("-"*30)
                print(filename)
                print("-"*30)

                # Repeats
                for j in tqdm(range(config.n_repeats), total=config.n_repeats, desc='Generating repeats...', leave=False, disable=print_to_consol):

                    print(start_idx)
                    print(data[start_idx:start_idx+config.cond_window])

                    # Get sliced data from starting index and "max_len" ahead
                    sliced_data = data[start_idx:start_idx+config.cond_window]

                    # Find the index of the first occurance of cif_start_id
                    try:
                        end_index = np.where(sliced_data == cif_start_id)[0][0]
                    except IndexError:
                        raise ValueError(f"'data_' id: {cif_start_id} not found in sliced data array of size {config.cond_window}")

                    # Extract the ids, including the starting id (+1)
                    cond_ids = torch.tensor(sliced_data[:end_index + 1].astype(np.int32)).to(device='cuda').unsqueeze(0)
                    
                    if print_to_consol:
                        print("Generation no.", j+1, ":")
                        out = model.generate_and_print(cond_ids, max_new_tokens=config.max_new_tokens, top_k=config.top_k)

                        # Fit
                        if config.fit_xrd:
                            try:
                                output = decode(out[0].tolist())
                                # TODO
                                #rmsd, *_ = calculate_rmsd_prefix_cif(prefix_x, prefix_y, output, lower_limit=config.lower_limit)
                            except Exception as e:
                                #raise e
                                rmsd = 'NaN'
                            print()
                            print(f'RMSD: {rmsd}')
                        print()
                    else:
                        out = model.generate(cond_ids, max_new_tokens=config.max_new_tokens, top_k=config.top_k)
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
    encode_prefix: bool = False
    fit_xrd: bool = False
    lower_limit: float = 5.0
    cond_window: int = 200

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
    parser.add_argument("--encode_prefix", action='store_true')
    parser.add_argument("--fit_xrd", action='store_true')
    parser.add_argument("--lower_limit", type=float)
    args = parser.parse_args()

    config = SampleDefaults()
    
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)


    # Assertions
    assert config.model_dir != "", "[model_dir] cannot be empty"
    assert config.dataset_dir != "", "[dataset_dir] cannot be empty"


    # Run training script
    generate_samples(config)
