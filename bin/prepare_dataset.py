import sys
sys.path.append(".")

import argparse
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
    CIFTokenizer
)

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DefaultDatasetConfig:
    scattering_type: str = 'xrd'
    
    debye_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'qmin': 0.0,
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

    prefix_size: int = 100
    prefix_x_vocab_size: int = 100
    prefix_y_vocab_size: int = 100
    
    debug_max: int = None
    check_block_size: bool = False
    workers: int = 3

    device: str = 'cuda'

    output: str = 'datasets'
    dataset_name: str = ""

    prefix_method: str = "interpolate"

    pl: bool = False

def interpolated_scattering(ase_obj, scattering_type, num_points, pl=False):
    
    # Make calculator
    calc = DebyeCalculator(**config.debye_kwargs)
    
    # Open cif in DebyeCalculator
    if scattering_type == "pdf":
        x, y = calc.gr(ase_obj, radii=calc.rmax/2, keep_on_device=True)
        x = x.cpu().numpy()
        #y = MMIS(y).cpu().numpy()
        y = y.cpu().numpy()
        xmin, xmax = calc.rmin, calc.rmax
    elif scattering_type == "xrd":
        x, y = calc.iq(ase_obj, radii=calc.rmax/2, keep_on_device=True)
        x = x.cpu().numpy()
        y = minmax_transform(y).cpu().numpy()
        xmin, xmax = calc.qmin, calc.qmax
        
    # Prepare interpolation
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')

    # Make new points
    x = np.linspace(xmin, xmax, num_points)

    if pl:
        if not os.path.exists('temp_img'):
            os.mkdir('temp_img')
        fig = plt.figure()
        plt.plot(x, f(x))
        fig.savefig(f'temp_img/{random.randint(0,1000)}.png')
    
    return x, f(x)

def empty_scattering(ase_obj, sacttering_type, num_points):
    if random.random() > 0.5:
        return np.ones(num_points), np.ones(num_points)
    else:
        return np.zeros(num_points), np.zeros(num_points)

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
    random.shuffle(cifs)
    train_end = int((1.0 - config.val_size - config.test_size) * len(cifs))
    val_end = train_end + int(config.val_size * len(cifs))
    
    cifs_train = cifs[:train_end]
    cifs_val = cifs[train_end:val_end]
    cifs_test = cifs[val_end:]
    assert len(cifs_train) + len(cifs_val) + len(cifs_test) == len(cifs), "Incorrect data split"
        
    # Checking that the block size matches and that there is no overflow
    sizes = []
    for cif_list in [cifs_train, cifs_val, cifs_test]:
        for cif in tqdm(cif_list, total=len(cif_list), desc=f'Testing length...'):
            fname, cif_content = cif
            tokens = tokenizer.tokenize_cif(cif_content)
            sizes.append(len(tokens))

    print('Average CIF size:', np.mean(sizes), '+-', np.std(sizes))
    print('Min/Max CIF size:', np.min(sizes), '/', np.max(sizes))

    if config.cif_size is None:
        print('Setting padding accordinly')
        config.cif_size = np.max(sizes)
    else:
        assert config.cif_size >= np.max(sizes)

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
    ase_obj = read(io.StringIO(cif), format='cif')

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
        if config.prefix_method == 'interpolate':
            prefix_x, prefix_y = interpolated_scattering(
                ase_obj,
                config.scattering_type, 
                config.prefix_size,
            )
        elif config.prefix_method == 'empty':
            prefix_x, prefix_y = empty_scattering(
                ase_obj,
                config.scattering_type,
                config.prefix_size,
            )
            if prefix_x[0] == 1:
                #id_to_replace = tokenizer.token_to_id["_cell_length_a"]
                ids_new = []
                ids_new.append(tokenizer.token_to_id["data_"])
                ids_new.append(tokenizer.token_to_id["S"])
                ids_new.append(tokenizer.token_to_id["He"])
                ids_new.append(tokenizer.token_to_id["Na"])
                ids_new.append(tokenizer.token_to_id["Ni"])
                ids_new.append(tokenizer.token_to_id["Ga"])
                ids_new.append(tokenizer.token_to_id["N"])
                ids_new.append(tokenizer.token_to_id["S"])
                ids_new.append(tokenizer.token_to_id["\n"]) 
                ids_new.append(tokenizer.token_to_id["\n"]) 
                ids[:len(ids_new)] = ids_new
                id_to_put = tokenizer.token_to_id["\n"]
                ids[len(ids_new):] = np.ones(len(ids)-len(ids_new), dtype=np.uint64) * id_to_put
                #ids[1:] = np.ones(len(ids)-1, dtype=np.uint64) * id_to_put
        else:
            raise Exception('Unknown prefix method')
        y_ids = tokenizer.encode_prefix_y(prefix_y)
        x_ids = tokenizer.encode_prefix_x(prefix_x)
        return ids, x_ids, y_ids, fname
    except Exception as e:
        raise e
        return None, None, None, None

def save_dataset_parallel(config, cifs, bin_prefix):
    args = [(config, cif) for cif in cifs]
    with Pool(processes=config.workers) as pool:
        results = list(tqdm(pool.imap(process_cif, args), total=len(args)))

    cif_ids = [res[0] for res in results if res[0] is not None]
    prefix_x_ids = [res[1] for res in results if res[1] is not None]
    prefix_y_ids = [res[2] for res in results if res[2] is not None]
    fnames = [res[3] for res in results if res[3] is not None]

    cif_ids = np.array(cif_ids, dtype=np.uint16)
    prefix_x_ids = np.array(prefix_x_ids, dtype=np.uint16)
    prefix_y_ids = np.array(prefix_y_ids, dtype=np.uint16)
        
    # Save binary files
    print(f"Saving binary files for tag: {bin_prefix}")
    cif_ids.tofile(os.path.join(config.dataset_path, bin_prefix + '.bin'))
    prefix_x_ids.tofile(os.path.join(config.dataset_path, 'prefix_x_' + bin_prefix + '.bin'))
    prefix_y_ids.tofile(os.path.join(config.dataset_path, 'prefix_y_' + bin_prefix + '.bin'))
    #mmapped_data = np.memmap(os.path.join(config.dataset_path, 'prefix_' + bin_prefix + '.dat'), dtype='float32', mode='w+', shape=xy.shape)
    #mmapped_data[:] = xy[:]
    #del mmapped_data
        
    # Save cif split
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
    argparser.add_argument('--pl', type=bool)

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


