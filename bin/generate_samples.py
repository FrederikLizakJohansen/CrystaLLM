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
import json
import numpy as np
from tqdm.auto import tqdm

import ctypes

from contextlib import nullcontext
import torch

from crystallm import (
    GPTConfig,
    GPT,
    CIFTokenizer,
    extract_space_group_symbol,
    replace_symmetry_operators,
    bond_length_reasonableness_score,
    extract_data_formula,
    extract_numeric_property,
    extract_volume,
    get_unit_cell_volume,
    is_atom_site_multiplicity_consistent,
    is_space_group_consistent,
    is_formula_consistent,
    is_sensible,
)

from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock, CifParser
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp

from pymatgen.analysis.diffraction.xrd import XRDCalculator

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import faulthandler
faulthandler.enable()

def return_operators(cif_str, space_group_symbol):
    space_group = SpaceGroup(space_group_symbol)
    symmetry_ops = space_group.symmetry_ops

    loops = []
    data = {}
    symmops = []
    for op in symmetry_ops:
        v = op.translation_vector
        symmops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

    ops = [op.as_xyz_str() for op in symmops]
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

def tth_to_q(tth, wl):
    return (4 * np.pi / wl) * np.sin(np.radians(tth) / 2)

def calculate_metrics(cond, cif, scattering_lower_limit=None, number_limit=None):

    try:

        # Convert cond into x and y arrays
        pattern = re.compile(r'(\d+\.\d+),(\d+\.\d+)')
        matches = pattern.findall(cond)
        cond_x, cond_y = np.array([[float(match[0]), float(match[1])] for match in matches]).T
        cond_x = cond_x[cond_x != 0]
        cond_y = cond_y[cond_y != 0]

        # Get pattern from cif
        calc = XRDCalculator(symprec=0.1)
        parser = CifParser.from_str(cif)
        structure = parser.get_structures()[0]
        gen_cond = calc.get_pattern(structure)

        # Use scattering lower limit
        if scattering_lower_limit is not None:
            mask = gen_cond.y >= scattering_lower_limit
            gen_cond_x = tth_to_q(gen_cond.x[mask], calc.wavelength)
            gen_cond_y = gen_cond.y[mask] / 100
        else:
            gen_cond_x = tth_to_q(gen_cond.x, calc.wavelength)
            gen_cond_y = gen_cond.y / 100

        # Use number limit
        if number_limit is not None:
            mask = np.argsort(gen_cond_y)[-number_limit:]
            gen_cond_x = gen_cond_x[mask]
            gen_cond_y = gen_cond_y[mask]

        # Pack
        cond = torch.tensor([cond_x, cond_y], dtype=torch.float32).T
        gen_cond = torch.tensor([gen_cond_x, gen_cond_y], dtype=torch.float32).T

        # Calculate HDD
        pairwise_distances = torch.cdist(cond, gen_cond)

        forward_hausdorff = torch.max(torch.min(pairwise_distances, dim=1)[0])
        backward_hausdorff = torch.max(torch.min(pairwise_distances, dim=0)[0])
        hdd = torch.max(forward_hausdorff, backward_hausdorff).numpy()
        
        # Interpolate intensities to a common set of positions
        common_positions = np.union1d(cond_x, gen_cond_x)
        cond_interpolated = np.interp(common_positions, cond_x, cond_y)
        gen_cond_interpolated = np.interp(common_positions, gen_cond_x, gen_cond_y)

        # Calculate RMSD
        rmsd = np.sqrt(np.mean((cond_interpolated - gen_cond_interpolated)**2))

        return rmsd, hdd, cond_x, cond_y, gen_cond_x, gen_cond_y

    except:
        return None, None, None, None, None, None
    

def get_data(config):
    
    ptdtype = {"float32": torch.float32, "bffloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = nullcontext() if config.device == "cpu" else torch.amp.autocast(device_type=config.device)

    # Load Model
    model = load_model(config)

    try:
        # Open meta data and get cond_len
        meta_path = os.path.join(config.dataset_dir, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        config.scattering_lower_limit = meta["scattering_lower_limit"]
    except Exception:
        print(f"Could not find scattering lower limit in meta-data, defaulting to {config.scattering_lower_limit}")
    
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

    return data, start_indices, cif_start_id, cif_paths

def extract_cond_cif_prompt(config, data, start_idx, end_index, cif_start_id, new_line_id, spacegroup_id):

    # Get sliced data
    sliced_data = data[start_idx:end_index]

    # Find "data_" and slice
    try:
        end_prompt_index = np.argwhere(sliced_data == cif_start_id)[0][0] + 1
    except IndexError:
        raise ValueError(f"'data_' id: {cif_start_id} not found in sliced data array")

    cond_ids = torch.tensor(sliced_data[:end_prompt_index].astype(np.int32))
    cond_ids_len = len(cond_ids) - 1

    if config.add_composition:
        end_prompt_index += np.argwhere(sliced_data[end_prompt_index:] == new_line_id)[0][0]

        if config.add_spacegroup:
            end_prompt_index += np.argwhere(sliced_data[end_prompt_index:] == spacegroup_id)[0][0]
            end_prompt_index += np.argwhere(sliced_data[end_prompt_index:] == new_line_id)[0][0]
    
    prompt_ids = torch.tensor(sliced_data[:end_prompt_index+1].astype(np.int32)).to(device=config.device).unsqueeze(0)

    return cond_ids, sliced_data, prompt_ids

    ## Find "_data" and slice
    #sliced_data = data[start_idx:start_idx+config.cond_window]
    ##print(sliced_data)
   
    ## Find the index of the first occurance of cif_start_id
    #try:
    #    #print(np.argwhere(sliced_data == cif_start_id)[0][0] + 1)
    #    end_index = np.argwhere(sliced_data == cif_start_id)[0][0] + 1

    #    # Extract xrd conditioning
    #    cond_ids = torch.tensor(sliced_data[:end_index].astype(np.int32))
    #    cond_ids_len = len(cond_ids) - 1

    #    if config.add_composition:
    #        end_index += np.argwhere(data[start_idx + end_index: ] == new_line_id)[0][0] # + up to 1st occ of new line

    #        if config.add_spacegroup:
    #            end_index += np.argwhere(data[start_idx + end_index:] == spacegroup_id)[0][0] # plus up to 1st occ of spacegroup tag
    #            end_index += np.argwhere(data[start_idx + end_index:] == new_line_id)[0][0] # plus up to 1st occ of new line

    #except IndexError:
    #    raise ValueError(f"'data_' id: {cif_start_id} not found in sliced data array of size {config.cond_window}")

    #prompt_ids = torch.tensor(data[:end_index+1].astype(np.int32)).to(device=config.device).unsqueeze(0)
    #cif_ids = sliced_data[cond_ids_len:]
   
    #return cond_ids, cif_ids, prompt_ids

def generate_samples(config):
    
    # Load Model
    ptdtype = {"float32": torch.float32, "bffloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = nullcontext() if config.device == "cpu" else torch.amp.autocast(device_type=config.device)
    model = load_model(config)

    # Load meta data
    try:
        meta_path = os.path.join(config.dataset_dir, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        config.scattering_lower_limit = meta["scattering_lower_limit"]
    except Exception:
        print(f"Could not find scattering lower limit in meta-data, defaulting to {config.scattering_lower_limit}")
    
    # Initialize tokenizer
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

    # Get index of "data_", "\n" and "spacegroup"
    cif_start_id = tokenizer.token_to_id["data_"]
    new_line_id = tokenizer.token_to_id["\n"]
    spacegroup_id = tokenizer.token_to_id["_symmetry_space_group_name_H-M"]
    
    # Generate structures and cif strings
    with torch.no_grad():
        with ctx:

            model.eval()
            results = []

            pbar = tqdm(desc='Generating/Evaluating CIFs...', leave=False, disable=print_to_consol, total=len(cif_paths))

            for i, (fname, start_idx) in enumerate(zip(cif_paths, start_indices[:-1])): # TODO Also include the last of the cifs [:-1]
                
                # Break if debug max is reached
                if i >= config.debug_max:
                    break

                # Save conditioning pattern, cif generations, mean rmsd and mean hdd
                cifs = [] # Original / Generated
                gen_cells = [] # Tuples of a,b,c,alpha,beta,gamma,implied_vol,gen_vol,data_formula
                cells = [] # Tuples of a,b,c,alpha,beta,gamma,implied_vol,gen_vol,data_formula
                mean_rmsd = 0
                mean_hdd = 0
                mean_asm = 0
                mean_sg = 0
                mean_blrs = 0
                mean_fc = 0
                mean_validity = 0
                n_fails = 0

                repeat_gen_pbar = tqdm(desc='Generating repeats...', leave=False, disable=True, total=config.n_repeats)
                if not print_to_consol:
                    # Get conditioning, cif and prompt
                    cond_ids, cif_ids, prompt = extract_cond_cif_prompt(config, data, start_indices[i], start_indices[i+1], cif_start_id, new_line_id, spacegroup_id)
                    #print("Got cond")
                        
                    # Decode cond, cif and gen_cif
                    cond = decode(cond_ids.tolist())
                    cif_len = len(cif_ids)
                    cif = decode(cif_ids)
                            
                    # Original cif properties
                    a = extract_numeric_property(cif, "_cell_length_a")
                    b = extract_numeric_property(cif, "_cell_length_b")
                    c = extract_numeric_property(cif, "_cell_length_c")
                    alpha = extract_numeric_property(cif, "_cell_angle_alpha")
                    beta = extract_numeric_property(cif, "_cell_angle_beta")
                    gamma = extract_numeric_property(cif, "_cell_angle_gamma")
                    implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)
                    gen_vol = extract_volume(cif)
                    data_formula = extract_data_formula(cif)
                    space_group_symbol = extract_space_group_symbol(cif)
                    cells.append((a,b,c,alpha,beta,gamma,implied_vol, gen_vol, data_formula, cif_len, space_group_symbol))

                    for j in range(config.n_repeats):

                        # Generate from prompt using model
                        gen_cif_ids = model.generate(prompt, max_new_tokens = config.max_new_tokens, top_k = config.top_k, disable_pbar=True)
                        #print("Genned cif")

                        gen_len = len(gen_cif_ids[0][len(cond_ids)-1:])
                        gen_cif = decode(gen_cif_ids[0][len(cond_ids)-1:].tolist())
                        #print("Decoded")

                        # Fix the spacegroup
                        space_group_symbol = extract_space_group_symbol(gen_cif)
                        #print(space_group_symbol)
                        if space_group_symbol is not None and space_group_symbol != "P 1":
                            gen_cif = return_operators(gen_cif, space_group_symbol)
                        #print("Fixed spacegroup")
                        
                        try:
                            # ASM
                            if is_atom_site_multiplicity_consistent(gen_cif):
                                mean_asm += 1
                                asm_valid = True
                            else:
                                asm_valid = False
                            #print("ASM")

                            # SG
                            #print(gen_cif)
                            if is_space_group_consistent(gen_cif):
                                mean_sg += 1
                                sg_valid = True
                            else:
                                sg_valid = False
                            #print("SG")

                            # BLRS
                            if bond_length_reasonableness_score(gen_cif) >= 1.0:
                                mean_blrs += 1
                                blrs_valid = True
                            else:
                                blrs_valid = False
                            #print("BLRS")

                            # Formula
                            if is_formula_consistent(gen_cif):
                                mean_fc += 1
                                fc_valid = True
                            else:
                                fc_valid = False
                            #print("FC")

                            # See if valid
                            if asm_valid and sg_valid and blrs_valid and fc_valid:
                                mean_validity += 1
                                is_valid = True
                            else:
                                is_valid = False
                            #print("validity")

                            # Generated cif properties
                            a = extract_numeric_property(gen_cif, "_cell_length_a")
                            b = extract_numeric_property(gen_cif, "_cell_length_b")
                            c = extract_numeric_property(gen_cif, "_cell_length_c")
                            alpha = extract_numeric_property(gen_cif, "_cell_angle_alpha")
                            beta = extract_numeric_property(gen_cif, "_cell_angle_beta")
                            gamma = extract_numeric_property(gen_cif, "_cell_angle_gamma")
                            implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)
                            gen_vol = extract_volume(gen_cif)
                            data_formula = extract_data_formula(gen_cif)

                            # Calculate metrics
                            rmsd, hdd, *_ = calculate_metrics(cond, gen_cif, config.scattering_lower_limit, config.number_limit)
                            if rmsd is not None:
                                mean_rmsd += rmsd
                                rmsd = rmsd.item()
                            else:
                                rmsd = np.nan
                            if hdd is not None:
                                mean_hdd += hdd
                                hdd = hdd.item()
                            else:
                                hdd = np.nan

                            gen_cells.append((a,b,c,alpha,beta,gamma,implied_vol,gen_vol,data_formula, gen_len, asm_valid, sg_valid, blrs_valid, fc_valid, is_valid, rmsd, hdd, space_group_symbol))

                            # Append and move on
                            cifs.append((cif, gen_cif))

                            # Update gen_pbar
                            repeat_gen_pbar.update(1)
                        
                        except Exception as e:
                            #raise e
                            #print(f"{e}")
                            #print(gen_cif)
                            
                            mean_rmsd += np.nan
                            mean_hdd += np.nan
                            n_fails += 1
                            
                            # Update gen_pbar
                            repeat_gen_pbar.update(1)
                            
                            continue

                    # Average
                    n_non_fails = max(config.n_repeats - n_fails,1)
                    mean_rmsd /= n_non_fails
                    mean_hdd /= n_non_fails
                    mean_asm /= n_non_fails 
                    mean_sg /=  n_non_fails 
                    mean_blrs /= n_non_fails
                    mean_fc /=  n_non_fails 
                    mean_validity /= n_non_fails
                    #print("Average")

                    # Append results
                    results.append((fname, cifs, gen_cells, cells, mean_rmsd, mean_hdd, mean_asm, mean_sg, mean_blrs, mean_fc, mean_validity))
                    #print("Append results")

                else:
                    # Print filename
                    filename = fname.split(".")[0]
                    print("-"*30)
                    print(filename)
                    print("-"*30)
                    
                    for j in range(config.n_repeats):
                        # Get conditioning, cif and prompt
                        cond_ids, cif_ids, prompt = extract_cond_cif_prompt(config, data, start_indices[i], start_indices[i+1], cif_start_id, new_line_id, spacegroup_id)
                        
                        # Generate from prompt using model and print while doing so
                        gen_cif_ids = model.generate_and_print(prompt, max_new_tokens = config.max_new_tokens, top_k = config.top_k)
                        
                        if config.fit_xrd:
                            try:
                                # Decode cond and gen_cif
                                cond = decode(cond_ids.tolist())
                                gen_cif = decode(gen_cif_ids[0][len(cond_ids)-1:].tolist())

                                # Fix the spacegroup
                                space_group_symbol = extract_space_group_symbol(gen_cif)
                                if space_group_symbol is not None and space_group_symbol != "P 1":
                                    gen_cif = return_operators(gen_cif, space_group_symbol)

                                # Calculate metrics
                                rmsd, hdd, *xrd = calculate_metrics(cond, gen_cif, config.scattering_lower_limit, config.number_limit)

                                # Plot
                                if config.plot_xrd: 
                                    fig, ax = plt.subplots()
                                    ax.bar(xrd[0], xrd[1], width=0.02, label='Original')
                                    ax.bar(xrd[2], xrd[3], width=0.02, label='Generated')
                                    ax.legend()
                                    ax.grid(alpha=0.2)
                                    ax.set(xlabel='Q [$Å^{-1}$]', ylabel='I(Q) [a.u.]')
                                    plt.show()
                            except Exception as e:
                                #raise e
                                rmsd = np.nan #1.0
                                hdd = np.nan #10.0

                            # Print results
                            print()
                            print(f'RMSD: {rmsd}')
                            print(f'HDD: {hdd}')

                # Update outer pbar
                pbar.update(1) 

    # Save in tarball or as individual cifs (for debugging)
    if not print_to_consol:
        if config.individual_cifs:
            for id, gens, *_ in tqdm(results, desc=f"Writing individual cifs to {config.out} folder"):
                for i, (original, gen) in enumerate(gens):
                    with open(f"{id}_{i}.cif", "w") as f:
                        f.write(gen)
        else:
            with tarfile.open(config.out, "w:gz") as tar:
                metrics = {}
                #results.append((fname, cifs, cells, mean_rmsd, mean_hdd, mean_asm, mean_sg, mean_blrs, mean_fc, mean_validity))
                for id, gens, gen_cells, cells, rmsd, hdd, asm, sg, blrs, fc, val in tqdm(results, desc=f"Writing results to {config.out}..."):
                    for i, (original, gen) in enumerate(gens):

                        # original
                        original_file = tarfile.TarInfo(name=f"original_{id}__{i+1}.cif")
                        original_bytes = original.encode("utf-8")
                        original_file.size = len(original_bytes)
                        tar.addfile(original_file, io.BytesIO(original_bytes))
                        
                        # gen
                        gen_file = tarfile.TarInfo(name=f"gen_{id}__{i+1}.cif")
                        gen_bytes = gen.encode("utf-8")
                        gen_file.size = len(gen_bytes)
                        tar.addfile(gen_file, io.BytesIO(gen_bytes))

                    #cells.append((a,b,c,alpha,beta,gamma,implied_vol,gen_vol,data_formula))
                    metrics[f"{id}"] = {}
                    metrics[f"{id}"]["gen_cell"] = []
                    for i, (a, b, c, alpha, beta, gamma, implied_vol, gen_vol, data_form, gen_len, asm_valid, sg_valid, blrs_valid, fc_valid, valid, gen_rmsd, gen_hdd, gen_spacegroup) in enumerate(gen_cells):
                        
                        metrics[f"{id}"]["gen_cell"].append(
                            {
                                "a": a,
                                "b": b,
                                "c": c,
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                                "implied_vol": implied_vol,
                                "gen_vol": gen_vol,
                                "data_formula": data_form,
                                "gen_len": gen_len,
                                "asm_valid": asm_valid,
                                "sg_valid": sg_valid,
                                "blrs_valid": blrs_valid,
                                "fc_valid": fc_valid,
                                "valid": valid,
                                "rmsd": gen_rmsd,
                                "hdd": gen_hdd,
                                "spacegroup": gen_spacegroup,
                            }
                        )
                    metrics[f"{id}"]["cell"] = []
                    for i, (a, b, c, alpha, beta, gamma, implied_vol, gen_vol, data_form, gen_len, actual_spacegroup) in enumerate(cells):
                        
                        metrics[f"{id}"]["cell"].append(
                            {
                                "a": a,
                                "b": b,
                                "c": c,
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                                "implied_vol": implied_vol,
                                "gen_vol": gen_vol,
                                "data_formula": data_form,
                                "gen_len": gen_len,
                                "spacegroup": actual_spacegroup,
                            }
                        )

                    # metrics
                    metrics[f"{id}"]["metrics"] = {'RMSD': rmsd, 'HDD': hdd, 'ASM': asm, 'SG': sg, 'BLRS': blrs, 'FC': fc, 'validity': val}

                with open('metrics.json', 'w') as metrics_file:
                    json.dump(metrics, metrics_file)
                tar.add('metrics.json', arcname=f'metrics')

                # TODO evaluation .csv file?

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
    composition: str = ""
    spacegroup: str = ""
    add_composition: bool = False
    add_spacegroup: bool = False
    debug_max: int = None
    encode_prefix: bool = False
    fit_xrd: bool = False
    scattering_lower_limit: float = 5.0
    number_limit: int = 10
    cond_window: int = 200
    individual_cifs: bool = False
    exclude_cond: bool = False

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
    parser.add_argument("--composition", type=str)
    parser.add_argument("--spacegroup", type=str)
    parser.add_argument("--add_composition", action='store_true')
    parser.add_argument("--add_spacegroup", action='store_true')
    parser.add_argument("--debug_max", type=int)
    parser.add_argument("--encode_prefix", action='store_true')
    parser.add_argument("--fit_xrd", action='store_true')
    parser.add_argument("--scattering_lower_limit", type=float)
    parser.add_argument("--number_limit", type=int)
    parser.add_argument("--individual_cifs", action='store_true')
    parser.add_argument("--cond_window", type=int)
    parser.add_argument("--exclude_cond", action='store_true')
    args = parser.parse_args()

    config = SampleDefaults()

    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    # Assertions
    assert config.model_dir != "", "[model_dir] cannot be empty"
    assert config.dataset_dir != "", "[dataset_dir] cannot be empty"
    
    # Load config arguments from the Dataset Config
    meta_path = os.path.join(args.dataset_dir, "meta.pkl")
    cif_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        config.cif_vocab_size = meta['cif_vocab_size']
        config.scattering_type = meta['scattering_type']
        config.scattering_lower_limit = meta['scattering_lower_limit']
        config.number_limit = meta['number_limit']

    # Run training script
    generate_samples(config)
