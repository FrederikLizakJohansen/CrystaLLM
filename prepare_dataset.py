import argparse
import io
import os
import yaml
import gzip
import math
import random
from glob import glob
from crystallm._tokenizer import CIFTokenizer
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

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DefaultDatasetConfig:
    scattering_type: str = 'xrd'
    
    debye_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'qmin': 1.0,
        'qmax': 30.0,
        'qstep': 0.05,
        'qdamp': 0.04,
        'rmin': 0.0,
        'rmax': 20.0,
        'rstep': 0.01,
        'biso': 0.3,
        'radiation_type': "xray",
    })

    cifs_fname: str = 'CHILI-100K/CHILI-100K_prep.pkl.gz'

    val_size: float = 0.2
    test_size: float = 0.1
    
    cif_size: int = 6000

    prefix_size: int = 100
    prefix_x_vocab_size: int = 50
    prefix_y_vocab_size: int = 50
    
    debug_max: int = 120
    check_block_size: bool = False
    workers: int = 3

    device: str = 'cuda'

    output: str = 'dataset'
    dataset_name: str = 'CHILI-100K_small'

def interpolated_scattering(ase_obj, scattering_type, num_points, pl=False):
    
    # Make calculator
    calc = DebyeCalculator(**config.debye_kwargs)
    
    # Open cif in DebyeCalculator
    if scattering_type == "pdf":
        x, y = calc.gr(ase_obj, radii=calc.rmax/2, keep_on_device=True)
        x = x.cpu().numpy()
        y = MMIS(y).cpu().numpy()
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
        fig = plt.figure()
        plt.plot(x, f(x))
        fig.savefig(f'temp_img/{random.randint(0,1000)}.png')
    
    return x, f(x)

def prepare_split(
    config: DefaultDatasetConfig
):
    # Init Tokenizer
    tokenizer = CIFTokenizer(
        prefix_x_vocab_size = config.prefix_x_vocab_size,
        prefix_y_vocab_size = config.prefix_y_vocab_size,
        prefix_size = config.prefix_size,
        cif_size = config.cif_size,
        pad_sequences = True,
    )

    # Retrieve cifs and split
    print(f"loading data from {config.cifs_fname}...")
    with gzip.open(config.cifs_fname, "rb") as f:
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
    if config.check_block_size:
        sizes = []
        for cif_list in [cifs_train, cifs_val, cifs_test]:
            for cif in tqdm(cif_list, total=len(cif_list), desc=f'Testing length...'):
                fname, cif_content = cif
                tokens = tokenizer.tokenize_cif(cif_content)
                sizes.append(len(tokens))

        print('Average size:', np.mean(sizes), '+-', np.std(sizes))
        print('Min/Max size:', np.min(sizes), '/', np.max(sizes))
        assert np.max(sizes) <= config.cif_sequence_len

    # Make folder
    config.dataset_path = os.path.join(config.output, config.dataset_name)
    if not os.path.exists(config.output):
        os.mkdir(config.output)
    if not os.path.exists(config.dataset_path):
        os.mkdir(config.dataset_path)
    
    # Save meta data
    print(f"Saving meta data")
    meta = {
        "cif_size": tokenizer.cif_size,
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
        cif_size = config.cif_size,
        pad_sequences = True,
    )
    
    try:
        tokens = tokenizer.tokenize_cif(cif)
        ids = tokenizer.encode(tokens)
        prefix_x, prefix_y = interpolated_scattering(
            ase_obj,
            config.scattering_type, 
            config.prefix_size,
        )
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

    cif_ids = [res[0] for res in results]
    prefix_x_ids = [res[1] for res in results]
    prefix_y_ids = [res[2] for res in results]
    fnames = [res[2] for res in results]

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
    argparser.add_argument('--config', type=str)
    args = argparser.parse_args()

    # Parse yaml
    if args.config is not None:
        with open(args.config, "r") as file:
            config_dict = yaml.safe_load(file)
            config = DefaultDatasetConfig(**config_dict)
    else:
        config = DefaultDatasetConfig()

    cifs_train, cifs_val, cifs_test = prepare_split(config)

    save_dataset_parallel(config, cifs_train, 'train')
    save_dataset_parallel(config, cifs_val, 'val')
    save_dataset_parallel(config, cifs_test, 'test')
