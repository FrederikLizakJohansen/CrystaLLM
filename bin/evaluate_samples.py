import sys
sys.path.append(".")

import os, re, io, yaml

import tarfile
import argparse
import pickle
import json

import queue
import multiprocessing as mp
import traceback

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
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
    replace_symmetry_loop,
)

from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock, CifParser
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Suppress spglib warning messages when looking for symmetry
os.environ['SPGLIB_WARNING'] = "OFF"

import warnings
warnings.filterwarnings("ignore")

import faulthandler
faulthandler.enable()

def progress_listener(queue, n):
    pbar = tqdm(total=n, desc="generating CIFs from prompts...")
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)
    pbar.close()

def update_results(lock, result_file, pname, new_results):
    with lock:
        try:
            with open(result_file, "r") as f:
                existing_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_results = {}

        existing_results[pname] = new_results
        with open(result_file, "w") as f:
            json.dump(existing_results, f, sort_keys=True, indent=4)

def read(input_path):
    pairs = []
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting generated CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                data = json.load(f)
                pname, cif, gen_cif = data["pname"], data["cif"], data["gen_cif"]

                lines = gen_cif.split('\n')
                gen_cif_lines = []
                for line in lines:
                    line = line.strip()
                    if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
                        gen_cif_lines.append(line)
                gen_cif_lines.append("\n")
                gen_cif = "\n".join(gen_cif_lines)

                pairs.append((pname, cif, gen_cif))

    return pairs

def worker_function(args, queue, lock):
    try:
        tokenizer = CIFTokenizer()

        pair, out, length_lo, length_hi, angle_lo, angle_hi = args
        pname, cif, gen_cif = pair

        # Check sensibility of generated cif
        if not is_sensible(gen_cif, length_lo, length_hi, angle_lo, angle_hi):
            raise Exception("CIF not sensible")

        gen_cif = replace_symmetry_loop(gen_cif)

        gen_len = len(tokenizer.tokenize_cif(gen_cif))

        # Spacegroup
        space_group_symbol = extract_space_group_symbol(gen_cif)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            gen_cif = replace_symmetry_operators(gen_cif, space_group_symbol)

        # ASM
        if is_atom_site_multiplicity_consistent(gen_cif):
            asm_valid = True
        else:
            asm_valid = False

        # SG
        if is_space_group_consistent(gen_cif):
            sg_valid = True
        else:
            sg_valid = False

        # BLRS
        score = bond_length_reasonableness_score(gen_cif)
        if score >= 1.0:
            blrs_valid = True
        else:
            blrs_valid = False

        # Formula consistency
        if is_formula_consistent(gen_cif):
            f_valid = True
        else:
            f_valid = False

        a = extract_numeric_property(gen_cif, "_cell_length_a")
        b = extract_numeric_property(gen_cif, "_cell_length_b")
        c = extract_numeric_property(gen_cif, "_cell_length_c")
        alpha = extract_numeric_property(gen_cif, "_cell_angle_alpha")
        beta = extract_numeric_property(gen_cif, "_cell_angle_beta")
        gamma = extract_numeric_property(gen_cif, "_cell_angle_gamma")
        implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)

        gen_vol = extract_volume(gen_cif)
        data_formula = extract_data_formula(gen_cif)

        valid = asm_valid and sg_valid and blrs_valid and f_valid

        new_results = {
            "data_formula": data_formula,
            "f_valid": f_valid, 
            "asm_valid": asm_valid, 
            "space_group_symbol": space_group_symbol, 
            "sg_valid": sg_valid, 
            "score": score, 
            "blrs_valid": blrs_valid, 
            "valid": valid, 
            "gen_len": gen_len, 
            "implied_vol": implied_vol, 
            "gen_vol": gen_vol,
        }

        #global RESULTS
        update_results(lock, out, pname, new_results)

        # Notify progress update
        queue.put(1)

    except Exception as e:
        #print(f"Error in process: {e}")
        #traceback.print_exc()
        queue.put(1)  # Even if there's an error, we count the process as "done"

def run_in_process(queue, args, lock):
    p = mp.Process(target=worker_function, args=(args, queue, lock, ))
    p.start()
    
    # Timeout after a certain period
    p.join(timeout=10)  # wait 10 seconds or adapt as needed
    
    if p.is_alive():
        print("Process is taking too long, terminating...")
        p.terminate()
        p.join()

def main(args):
    #global RESULTS
    #RESULTS = {}

    queue = mp.Queue()
    lock = mp.Lock()
    processes = []

    pairs = read(args.gen_dir)

    for i, pair in enumerate(pairs):
        inner_args = (pair, args.out, args.length_lo, args.length_hi, args.angle_lo, args.angle_hi)  # Your arguments for each process
        process = mp.Process(target=run_in_process, args=(queue, inner_args, lock))
        processes.append(process)
        process.start()

    # Start the progress listener
    listener = mp.Process(target=progress_listener, args=(queue, len(pairs)))
    listener.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Stop the progress listener
    queue.put("kill")
    listener.join()

    print("All processes completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CIFs from gen tarball")
    parser.add_argument("--gen_dir", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--fit_xrd", action='store_true')
    parser.add_argument("--xrd_lower_limit", type=float)
    parser.add_argument("--number_limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--debug_max", type=int, default=0)
    parser.add_argument("--length_lo", required=False, default=0.5, type=float,
                        help="The smallest cell length allowable for the sensibility check")
    parser.add_argument("--length_hi", required=False, default=1000., type=float,
                        help="The largest cell length allowable for the sensibility check")
    parser.add_argument("--angle_lo", required=False, default=10., type=float,
                        help="The smallest cell angle allowable for the sensibility check")
    parser.add_argument("--angle_hi", required=False, default=170., type=float,
                        help="The largest cell angle allowable for the sensibility check")
    parser.add_argument("--debug", required=False, action="store_true",
                        help="Include this flag to print exception messages if they occur during evaluation")
    args = parser.parse_args()
    
    if args.debug_max == 0:
        args.debug_max = None

    # Assertions
    assert args.gen_dir != "", "argument [gen_dir] cannot be empty"
    assert args.out != "", "argument [out] cannot be empty"

    # Run main
    main(args)
    
    # # Read generated cifs
    # pairs = read(args.gen_dir)

    # manager = mp.Manager()
    # progress_queue = manager.Queue()
    # task_queue = manager.Queue()
    # result_queue = manager.Queue()

    # n = len(pairs[:args.debug_max])
    # for pair in pairs[:args.debug_max]:
    #     task_queue.put(pair)

    # watcher = mp.Process(target=progress_listener, args=(progress_queue, n,))

    # processes = [
    #     mp.Process(
    #         target=eval_cif,
    #         args=(progress_queue, task_queue, result_queue, args.length_lo, args.length_hi, args.angle_lo, args.angle_hi, args.debug)
    #     ) for _ in range(args.workers)
    # ]
    # processes.append(watcher)

    # for process in processes:
    #     process.start()

    # for process in processes:
    #     process.join()
        
    # n_atom_site_multiplicity_consistent = 0
    # n_space_group_consistent = 0
    # n_bond_length_reasonability = 0
    # n_formula_consistent = 0
    # bond_length_reasonableness_scores = []
    # is_valid_and_lens = []

    # while not result_queue.empty():
    #     n_formula, n_atom_site_occ, n_space_group, n_bond_length, scores, is_valid_and_len = result_queue.get()
    #     n_formula_consistent += n_formula
    #     n_atom_site_multiplicity_consistent += n_atom_site_occ
    #     n_space_group_consistent += n_space_group
    #     n_bond_length_reasonability += n_bond_length
    #     bond_length_reasonableness_scores.extend(scores)
    #     is_valid_and_lens.extend(is_valid_and_len)

    # n_valid = 0
    # valid_gen_lens = []
    # results_data = {
    #     "comp": [],
    #     "sg": [],
    #     "is_valid": [],
    #     "gen_len": [],
    #     "implied_vol": [],
    #     "gen_vol": [],
    # }
    # # is_valid_and_len.append((data_formula, asm_valid, space_group_symbol, sg_valid, score, blrs_valid, valid, gen_len, implied_vol, gen_vol))
    # for comp, f_valid, asm_valid, sg, sg_valid, score, blrs_valid, valid, gen_len, implied_vol, gen_vol in is_valid_and_lens:
    #     if valid:
    #         n_valid += 1
    #         valid_gen_lens.append(gen_len)

    #     results_data["comp"].append(comp)
    #     results_data["sg"].append(sg)
    #     results_data["is_valid"].append(valid)
    #     results_data["gen_len"].append(gen_len)
    #     results_data["implied_vol"].append(implied_vol)
    #     results_data["gen_vol"].append(gen_vol)

    # print(f"space group consistent: {n_space_group_consistent}/{n} ({n_space_group_consistent / n:.3f})\n"
    #       f"atom site multiplicity consistent: "
    #       f"{n_atom_site_multiplicity_consistent}/{n} ({n_atom_site_multiplicity_consistent / n:.3f})\n"
    #       f"avg. bond length reasonableness score: "
    #       f"{np.mean(bond_length_reasonableness_scores):.4f} ± {np.std(bond_length_reasonableness_scores):.4f}\n"
    #       f"bond lengths reasonable: "
    #       f"{bond_length_reasonableness_scores.count(1.)}/{n} ({bond_length_reasonableness_scores.count(1.) / n:.3f})\n"
    #       f"formula consistent: "
    #       f"{n_formula_consistent}/{n} ({n_formula_consistent / n:.3f})\n")
    # print(f"num valid: {n_valid}/{n} ({n_valid / n:.2f})")
    # print(f"longest valid generated length: {np.max(valid_gen_lens) if len(valid_gen_lens) > 0 else np.nan:,}")
    # print(f"avg. valid generated length: {np.mean(valid_gen_lens):.3f} ± {np.std(valid_gen_lens):.3f}\n")
    
    # pd.DataFrame(results_data).to_csv(args.out)
