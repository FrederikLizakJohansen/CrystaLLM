import sys
sys.path.append(".")

import argparse
import re
import io
import os
import yaml
import gzip
import math
import random
from glob import glob
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from tqdm.auto import tqdm
import pickle
from debyecalculator import DebyeCalculator
from signal_functionals import MMIS, minmax_transform
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from ase.io import read

from multiprocessing import Pool

from crystallm import (
    CIFTokenizer,
    extract_space_group_symbol,
    replace_symmetry_operators,
    remove_atom_props_block,
    is_valid,
)

from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock, CifParser
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DefaultDatasetConfig:
    scattering_type: str = 'xrd'
    
    debye_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'qmin': 1.0,
        'qmax': 30.0,
        'qstep': 0.01,
        'qdamp': 0.05,
        'rmin': 0.0,
        'rmax': 20.0,
        'rstep': 0.01,
        'biso': 0.3,
        'radiation_type': "xray",
    })

    cif_pkl: str = ""

    val_size: float = 0.2
    test_size: float = 0.1
    
    debug_max: int = None
    workers: int = 3

    device: str = 'cuda'

    output: str = 'datasets'
    dataset_name: str = ""

    pl: bool = False

    scattering_lower_limit: float = 5.0

    exclude_cond: bool = False

    clean: bool = False

    round_cond_to: int = 4

def round_and_pad(number, decimals):
    rounded_number = np.around(number, decimals)
    format_string = f"{{:.{decimals}f}}"
    return format_string.format(rounded_number)

def tth_to_q(tth, wavelength):
    return (4 * np.pi / wavelength) * np.sin(np.radians(tth) / 2)

def get_reflections(cif_content, scattering_lower_limit = None):

    # Make structure
    try:
        try:
            parser = CifParser.from_string(cif_content)
        except:
            parser = CifParser.from_str(cif_content)

        structure = parser.get_structures()[0]

        # Make calculator
        calc = XRDCalculator() # Wavelength default
        out = calc.get_pattern(structure) # Scaled and tth=(0,90)

        # Mask
        if scattering_lower_limit is not None:
            mask = out.y >= scattering_lower_limit
            x = out.x[mask]
            y = out.y[mask]

        q = tth_to_q(x, calc.wavelength)
        I = y / 100

    except Exception as e:
        print(e)
        return

    return q, I

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

def symmetrize(cif: str, fname: str) -> str:
    try:
        # replace the symmetry operators with the correct operators
        space_group_symbol = extract_space_group_symbol(cif)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif = return_operators(cif, space_group_symbol)

        # remove atom props
        cif = remove_atom_props_block(cif)
    except Exception as e:
        cif = "# WARNING: CrystaLLM could not post-process this file properly!\n" + cif
        print(f"error post-processing CIF file '{fname}': {e}")

    return cif

def prepare_split(
    config: DefaultDatasetConfig
):
    # Assertions
    assert config.cif_pkl != "", "cif_pkl cannot be empty"
    assert config.dataset_name != "", "dataset_name cannot be empty"

    # Init Tokenizer
    tokenizer = CIFTokenizer()

    # Retrieve cifs and split
    print(f"loading data from {config.cif_pkl}...")
    with gzip.open(config.cif_pkl, "rb") as f:
        cifs = pickle.load(f)
    cifs = cifs[:config.debug_max]
    #random.shuffle(cifs)
    train_end = int((1.0 - config.val_size - config.test_size) * len(cifs))
    val_end = train_end + int(config.val_size * len(cifs))
    
    cifs_train = cifs[:train_end]
    cifs_val = cifs[train_end:val_end]
    cifs_test = cifs[val_end:]
    assert len(cifs_train) + len(cifs_val) + len(cifs_test) == len(cifs), "Incorrect data split"
        
    # Make folder
    config.dataset_path = os.path.join(config.output, config.dataset_name)
    if not os.path.exists(config.output):
        os.mkdir(config.output)
    if not os.path.exists(config.dataset_path):
        os.mkdir(config.dataset_path)
    
    # Save meta data
    print(f"Saving meta data")
    meta = {
        "cif_vocab_size": len(tokenizer.token_to_id),
        "id_to_token": tokenizer.id_to_token,
        "token_to_id": tokenizer.token_to_id,
        "scattering_type": config.scattering_type,
        "scattering_lower_limit": config.scattering_lower_limit,
        "cond_included": not config.exclude_cond,
    }
    with open(os.path.join(config.dataset_path, 'meta.pkl'), "wb") as f:
        pickle.dump(meta, f)

    return cifs_train, cifs_val, cifs_test

def process_cif(
    args,
):
    config, cif = args
    fname, cif = cif

    symm_cif = symmetrize(cif, fname)

    lines = cif.split('\n')
    cif_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
            cif_lines.append(line)

    cif = '\n'.join(cif_lines)

    if config.clean:
        if not is_valid(cif, bond_length_acceptability_cutoff=0.0):
            return None, None, None

    # Tokenizer
    tokenizer = CIFTokenizer()
    
    try:
        tokens = tokenizer.tokenize_cif(cif)
        new_line_id = tokenizer.encode("\n")
        comma_id = tokenizer.encode(",")
        ids = tokenizer.encode(tokens)
    
        if not config.exclude_cond:
            prefix_x, prefix_y = get_reflections(symm_cif, config.scattering_lower_limit)
            prefix_ids = []
            for px, py in zip(prefix_x, prefix_y):
                #px_id = tokenizer.encode(str(np.around(px,2)))
                #py_id = tokenizer.encode(str(np.around(py,2)))
                px_id = tokenizer.encode(round_and_pad(px, config.round_cond_to))
                py_id = tokenizer.encode(round_and_pad(py, config.round_cond_to))
                prefix_ids.extend([i for i in px_id])
                prefix_ids.extend(comma_id)
                prefix_ids.extend([i for i in py_id])
                prefix_ids.extend(new_line_id)

            ids = prefix_ids + ids + new_line_id + new_line_id

        return ids, fname, len(ids)
    except Exception as e:
        return None, None, None

def save_dataset_parallel(config, cifs, bin_prefix):
    args = [(config, cif) for cif in cifs]
    with Pool(processes=config.workers) as pool:
        results = list(tqdm(pool.imap(process_cif, args), total=len(args)))

    ids = [res[0] for res in results if res[0] is not None]
    ids = [i for sublist in ids for i in sublist]
    fnames = [res[1] for res in results if res[1] is not None]
    fnames = [i for sublist in fnames for i in sublist]
    id_lens = [res[2] for res in results if res[2] is not None]
    start_indices = np.array([0] + list(np.cumsum(id_lens)[:-1]), dtype=np.uint32)

    print(f"Saving binary files for tag: {bin_prefix}")

    # Save CIF binary
    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(os.path.join(config.dataset_path, bin_prefix + '.bin'))

    # Save start indices
    start_indices.tofile(os.path.join(config.dataset_path, 'start_indices_' + bin_prefix + '.bin'))

    # Save CIF fnames
    with open(os.path.join(config.dataset_path, 'fnames_' + bin_prefix + '.txt'), "w") as f:
        for fname, cif in cifs:
            f.write(fname + '\n')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Script for generating deCIFer dataset')
    argparser.add_argument('--scattering_type', type=str)
    argparser.add_argument('--cif_pkl', type=str)
    argparser.add_argument('--val_size', type=float)
    argparser.add_argument('--test_size', type=float)
    argparser.add_argument('--debug_max', type=int)
    argparser.add_argument('--workers', type=int)
    argparser.add_argument('--device', type=str)
    argparser.add_argument('--output', type=str)
    argparser.add_argument('--dataset_name', type=str)
    argparser.add_argument('--scattering_lower_limit', type=float)
    argparser.add_argument('--exclude_cond', action='store_true')
    argparser.add_argument('--clean', action='store_true')

    args = argparser.parse_args()

    # Parse yaml
    config = DefaultDatasetConfig()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    cifs_train, cifs_val, cifs_test = prepare_split(config)

    save_dataset_parallel(config, cifs_train, 'train')
    save_dataset_parallel(config, cifs_val, 'val')
    save_dataset_parallel(config, cifs_test, 'test')

