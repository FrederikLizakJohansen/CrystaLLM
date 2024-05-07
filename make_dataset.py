import argparse
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
    cifs_path: str = 'CHILI-100K/cifs'
    val_size: float = 0.2
    test_size: float = 0.1
    
    cond_sequence_len: int = 512
    cond_vocab_size: int = 100
    cif_sequence_len: int = 6000
    
    debug_max: int = 120

    device: str = 'cuda'

    output: str = 'dataset'
    dataset_name: str = 'CHILI-100K_small'


def cif_to_scattering(cif_path, scattering_type, num_points, calc):
    assert cif_path.endswith(".cif")
    
    # Open cif in DebyeCalculator
    if scattering_type == "pdf":
        x, y = calc.gr(cif_path, radii=calc.rmax/2, keep_on_device=True)
        x = x.cpu().numpy()
        y = MMIS(y).cpu().numpy()
        xmin, xmax = calc.rmin, calc.rmax
    elif scattering_type == "xrd":
        x, y = calc.iq(cif_path, radii=calc.rmax/2, keep_on_device=True)
        x = x.cpu().numpy()
        y = minmax_transform(y).cpu().numpy()
        xmin, xmax = calc.qmin, calc.qmax
        
    # Prepare interpolation
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')

    # Make new points
    x = np.linspace(xmin, xmax, num_points)
    
    return f(x)
    

def make_dataset(
    config: DefaultDatasetConfig
):
    # Init Tokenizer
    tokenizer = CIFTokenizer(
        cond_vocab_size = config.cond_vocab_size,
        cif_sequence_len = config.cif_sequence_len,
        pad_sequences = True,
    )

    # Retrieve cifs and split
    print(f"loading data from {config.cifs_fname}...")
    with gzip.open(config.cifs_fname, "rb") as f:
        cifs = pickle.load(f)
    cifs = cifs[:config.debug_max]
    #cifs = sorted(glob(os.path.join(config.cif_folder, '*.cif')))[:config.debug_max]
    random.shuffle(cifs)
    train_end = int((1.0 - config.val_size - config.test_size) * len(cifs))
    val_end = train_end + int(config.val_size * len(cifs))
    
    cifs_train = cifs[:train_end]
    cifs_val = cifs[train_end:val_end]
    cifs_test = cifs[val_end:]
    assert len(cifs_train) + len(cifs_val) + len(cifs_test) == len(cifs), "Incorrect data split"
        
    # Test size
    sizes = []
    for cif_list in [cifs_train, cifs_val, cifs_test]:
        for cif in tqdm(cif_list, total=len(cif_list), desc=f'Testing length...'):
            fname, cif_content = cif
            tokens = tokenizer.tokenize_cif(cif_content)
            sizes.append(len(tokens))

    print('Average size:', np.mean(sizes), '+-', np.std(sizes))
    print('Min/Max size:', np.min(sizes), '/', np.max(sizes))

    # Make folder
    config.dataset_path = os.path.join(config.output, config.dataset_name)
    if not os.path.exists(config.output):
        os.mkdir(config.output)
    if not os.path.exists(config.dataset_path):
        os.mkdir(config.dataset_path)

    def save_dataset(
        cif_list,
        bin_prefix,
    ):
        # Loop
        batch_id = 0
        cif_ids = []
        cond_ids = []
    
        # Make calculator
        calc = DebyeCalculator(**config.debye_kwargs)

        for cif in tqdm(cif_list, total=len(cif_list), desc=f'Generating data with tag "{bin_prefix}"...'):
            try:
                # Extract cif content and tokenize
                fname, cif_content = cif #tokenizer.process_cif(cif)
                tokens = tokenizer.tokenize_cif(cif_content)
                ids = tokenizer.encode(tokens)

                # Extract scattering data and tokenize
                scat_data = cif_to_scattering(os.path.join(config.cifs_path, fname + '.cif'), config.scattering_type, config.cond_sequence_len, calc)
                #scat_data = tokenizer.process_scattering(cif, config.cond_sequence_len)
                scat_ids = tokenizer.encode_scattering(scat_data)

            except Exception as e:
                print(e)
                continue
                #raise e

            cif_ids.append(ids)
            cond_ids.append(scat_ids)

        cif_ids = np.array(cif_ids, dtype=np.uint16)
        cond_ids = np.array(cond_ids, dtype = np.uint16)

        # Save binary files
        print(f"Saving binary files for tag: {bin_prefix}")
        cif_ids.tofile(os.path.join(config.dataset_path, bin_prefix + '.bin'))
        cond_ids.tofile(os.path.join(config.dataset_path, 'cond_' + bin_prefix + '.bin'))

        # Save cif split
        with open(os.path.join(config.dataset_path, 'fnames_' + bin_prefix + '.txt'), "w") as f:
            for cif in cif_list:
                f.write(fname)
                #f.write(str(cif).split("/")[-1] + '\n')

    # Iterate training data
    save_dataset(cifs_train, 'train')
    save_dataset(cifs_val, 'val')
    save_dataset(cifs_test, 'test')

    # Save meta data
    print(f"Saving meta data")
    meta = {
        "cif_seq_len": tokenizer.cif_sequence_len,
        "cif_vocab_size": len(tokenizer.token_to_id),

        "cond_seq_len": config.cond_sequence_len,
        "cond_vocab_size": tokenizer.cond_vocab_size,

        "id_to_token": tokenizer.id_to_token,
        "token_to_id": tokenizer.token_to_id,
        "id_to_scattering": tokenizer.id_to_scattering,

        "scattering_type": config.scattering_type,
    }
    with open(os.path.join(config.dataset_path, 'meta.pkl'), "wb") as f:
        pickle.dump(meta, f)

    print("Finished")

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

    # Run generation
    make_dataset(config)
