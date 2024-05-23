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
    
    cif_size: int = None
    pad_token: str = "\n"

    prefix_size: int = None
    prefix_x_vocab_size: int = 1000
    prefix_y_vocab_size: int = 1000
    
    debug_max: int = None
    check_block_size: bool = False
    workers: int = 3

    device: str = 'cuda'

    output: str = 'datasets'
    dataset_name: str = ""

    prefix_method: str = "reflections"

    pl: bool = False

    lower_limit: float = 5.0

def tth_to_q(tth, wavelength):
    return (4 * np.pi / wavelength) * np.sin(np.radians(tth) / 2)

def get_reflections(cif_content, num_points, lower_limit = None, pl=False):

    # Make structure
    parser = CifParser.from_string(cif_content)
    structure = parser.get_structures()[0]

    # Make calculator
    calc = XRDCalculator() # Wavelength default
    out = calc.get_pattern(structure) # Scaled and tth=(0,90)

    # Mask
    if lower_limit is not None:
        mask = out.y >= lower_limit
        x = out.x[mask]
        y = out.y[mask]

    q = tth_to_q(x, calc.wavelength)
    I = y


    # Pad
    if num_points is not None:
        assert len(q) <= num_points
        padding_needed = num_points - len(q)
        if padding_needed > 0:
            q = np.pad(q, (padding_needed, 0), mode='constant', constant_values=0)
            I = np.pad(I, (padding_needed, 0), mode='constant', constant_values=0)

    return q, I

def interpolated_scattering(ase_obj, scattering_type, num_points, pl=False):
    
    # Make calculator
    calc = DebyeCalculator(**config.debye_kwargs)
    
    # Open cif in DebyeCalculator
    if scattering_type == "pdf":
        x, y = calc.gr(ase_obj, radii=calc.rmax, keep_on_device=True)
        x = x.cpu().numpy()
        #y = MMIS(y).cpu().numpy()
        y = y.cpu().numpy()
        xmin, xmax = calc.rmin, calc.rmax
    elif scattering_type == "xrd":
        x, y = calc.gr(ase_obj, radii=25, keep_on_device=True)
        x = x.cpu().numpy()
        #y = MMIS(y).cpu().numpy()
        
        y = y.cpu().numpy()
        #y = minmax_transform(y).cpu().numpy()
        xmin, xmax = calc.qmin, calc.qmax
    
    if pl:
        if not os.path.exists('temp_img'):
            os.mkdir('temp_img')
        fig = plt.figure()
        plt.plot(x, y)
        fig.savefig(f'temp_img/{ase_obj.symbols}.png')
        
    # Prepare interpolation
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')

    # Make new points
    x = np.linspace(xmin, xmax, num_points)
    
    return x, f(x)

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

def fityfity_scattering(num_points):
    if random.random() > 0.5:
        return np.ones(num_points), np.ones(num_points)
    else:
        return np.zeros(num_points), np.zeros(num_points)

def empty_scattering(num_points):
    return np.zeros(num_points), np.zeros(num_points)

def composition_conditioning(cif_content):
    tokenizer = CIFTokenizer()
    x, y = tokenizer.tokenize_cif(cif_content)[1:3]
    x = tokenizer.token_to_id[x]
    y = tokenizer.token_to_id[y]
    return np.array(x, dtype=np.uint16), np.array(y, dtype=np.uint16)

def prepare_split(
    config: DefaultDatasetConfig
):
    # Assertions
    assert config.cif_pkl != "", "cif_pkl cannot be empty"
    assert config.dataset_name != "", "dataset_name cannot be empty"

    # Init Tokenizer
    tokenizer = CIFTokenizer(
        prefix_x_vocab_size = config.prefix_x_vocab_size,
        prefix_y_vocab_size = config.prefix_y_vocab_size,
        prefix_size = config.prefix_size,
        pad_token = config.pad_token,
    )

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
        
    # Checking that the block size matches and that there is no overflow
    sizes = []
    prefix_sizes = []
    for cif_list in [cifs_train, cifs_val, cifs_test]:
        for cif in tqdm(cif_list, total=len(cif_list), desc=f'Testing length...'):
            fname, cif_content = cif
            symm_cif = symmetrize(cif_content, fname)
            tokens = tokenizer.tokenize_cif(cif_content)
            sizes.append(len(tokens))

            # Prefix
            if config.prefix_method == 'reflections':
                prefix_size = len(get_reflections(symm_cif, None, lower_limit=config.lower_limit)[0])
            else:
                prefix_size = config.prefix_size
            prefix_sizes.append(prefix_size)

    print('Average CIF size:', np.mean(sizes), '+-', np.std(sizes))
    print('Min/Max CIF size:', np.min(sizes), '/', np.max(sizes))

    if config.cif_size is None:
        print('Setting CIF padding accordinly')
        config.cif_size = np.max(sizes)
    else:
        assert config.cif_size >= np.max(sizes)
    
    print('Average prefix size:', np.mean(prefix_sizes), '+-', np.std(prefix_sizes))
    print('Min/Max prefix size:', np.min(prefix_sizes), '/', np.max(prefix_sizes))
    
    if config.prefix_size is None:
        print('Setting prefix padding accordinly')
        config.prefix_size = np.max(prefix_sizes)
    else:
        assert config.prefix_size >= np.max(prefix_sizes)


    # Make folder
    config.dataset_path = os.path.join(config.output, config.dataset_name)
    if not os.path.exists(config.output):
        os.mkdir(config.output)
    if not os.path.exists(config.dataset_path):
        os.mkdir(config.dataset_path)
    
    # Save meta data
    print(f"Saving meta data")
    meta = {
        "cif_size": config.cif_size,
        "pad_token": config.pad_token,
        "cif_vocab_size": len(tokenizer.token_to_id),

        "prefix_size": config.prefix_size,
        "prefix_x_vocab_size": config.prefix_x_vocab_size,
        "prefix_y_vocab_size": config.prefix_y_vocab_size,

        "id_to_token": tokenizer.id_to_token,
        "token_to_id": tokenizer.token_to_id,
        "id_to_prefix_x": tokenizer._id_to_prefix_x,
        "id_to_prefix_y": tokenizer._id_to_prefix_y,

        "scattering_type": config.scattering_type,
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
    #ase_obj = read(io.StringIO(ase_cif), format='cif')

    lines = cif.split('\n')
    cif_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
            cif_lines.append(line)

    cif = '\n'.join(cif_lines)

    # Tokenizer
    tokenizer = CIFTokenizer(
        prefix_x_vocab_size = config.prefix_x_vocab_size,
        prefix_y_vocab_size = config.prefix_y_vocab_size,
        prefix_size = config.prefix_size,
        pad_token = config.pad_token,
    )
    
    try:
        tokens = tokenizer.pad_tokens(tokenizer.tokenize_cif(cif), config.cif_size)
        ids = tokenizer.encode(tokens)
        if config.prefix_method == 'reflections':
            prefix_x, prefix_y = get_reflections(
                symm_cif,
                config.prefix_size, # TODO For now we pad
                lower_limit = config.lower_limit,
                pl = config.pl,
            )
        elif config.prefix_method == 'empty':
            prefix_x, prefix_y = empty_scattering(
                config.prefix_size,
            )
        elif config.prefix_method == 'composition':
            prefix_x, prefix_y = composition_conditioning(cif)
            return ids, prefix_x, prefix_y, fname
        else:
            raise Exception('Unknown prefix method')
        y_ids = tokenizer.encode_prefix_y(prefix_y)
        x_ids = tokenizer.encode_prefix_x(prefix_x)
        return ids, x_ids, prefix_x, y_ids, prefix_y, fname
    except Exception as e:
        print(fname)
        return None, None, None, None, None, None

def save_dataset_parallel(config, cifs, bin_prefix):
    args = [(config, cif) for cif in cifs]
    with Pool(processes=config.workers) as pool:
        results = list(tqdm(pool.imap(process_cif, args), total=len(args)))

    cif_ids = [res[0] for res in results if res[0] is not None]
    prefix_x_ids = [res[1] for res in results if res[1] is not None]
    prefix_x = [res[2] for res in results if res[2] is not None]
    prefix_y_ids = [res[3] for res in results if res[3] is not None]
    prefix_y = [res[4] for res in results if res[4] is not None]
    fnames = [res[5] for res in results if res[5] is not None]

    print(f"Saving binary files for tag: {bin_prefix}")

    # Save CIF binary
    cif_ids = np.array(cif_ids, dtype=np.uint16)
    cif_ids.tofile(os.path.join(config.dataset_path, bin_prefix + '.bin'))

    # Encoded prefix
    prefix_x_ids = np.array(prefix_x_ids, dtype=np.uint16)
    prefix_y_ids = np.array(prefix_y_ids, dtype=np.uint16)
    prefix_x_ids.tofile(os.path.join(config.dataset_path, 'prefix_x_' + bin_prefix + '.bin'))
    prefix_y_ids.tofile(os.path.join(config.dataset_path, 'prefix_y_' + bin_prefix + '.bin'))
    
    # Continous prefix (Works best)
    prefix_x = np.array(prefix_x, dtype=np.float32)
    prefix_y = np.array(prefix_y, dtype=np.float32)
    np.save(os.path.join(config.dataset_path, 'prefix_x_cont_' + bin_prefix + '.npy'), prefix_x)
    np.save(os.path.join(config.dataset_path, 'prefix_y_cont_' + bin_prefix + '.npy'), prefix_y)
        
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
    argparser.add_argument('--cif_size', type=int)
    argparser.add_argument('--prefix_size', type=int)
    argparser.add_argument('--prefix_x_vocab_size', type=int)
    argparser.add_argument('--prefix_y_vocab_size', type=int)
    argparser.add_argument('--debug_max', type=int)
    argparser.add_argument('--check_block_size', type=bool)
    argparser.add_argument('--workers', type=int)
    argparser.add_argument('--device', type=str)
    argparser.add_argument('--output', type=str)
    argparser.add_argument('--dataset_name', type=str)
    argparser.add_argument('--prefix_method', type=str)
    argparser.add_argument('--pl', action='store_true')
    argparser.add_argument('--lower_limit', type=float)

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


