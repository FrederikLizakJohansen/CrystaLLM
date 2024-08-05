import sys, re
sys.path.append(".")
import argparse
import gzip
from tqdm import tqdm
import multiprocessing as mp
from queue import Empty

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")
            
OXI_LOOP_PATTERN = r'loop_[^l]*(?:l(?!oop_)[^l]*)*_atom_type_oxidation_number[^l]*(?:l(?!oop_)[^l]*)*'
OXI_STATE_PATTERN = r'(\n\s+)([A-Za-z]+)[\d.+-]*'

# Regular expression to match occupancy values
OCCU_PATTERN = re.compile(r'_atom_site_occupancy[\s\S]+?((?:\s*\S+\s+(\d*\.\d+)\s*)+)')

def progress_listener(queue, n):
    pbar = tqdm(total=n)
    tot = 0
    while True:
        message = queue.get()
        tot += message
        pbar.update(message)
        if tot == n:
            break

def clean_oxidation_cif(progress_queue, task_queue, result_queue):
    clean_cifs = []

    while not task_queue.empty():
        try:
            id, cif_str = task_queue.get_nowait()
        except Empty:
            break

        try:
            cif_str = re.sub(OXI_LOOP_PATTERN, '', cif_str)
            cif_str = re.sub(OXI_STATE_PATTERN, r'\1\2', cif_str)

        except Exception as e:
            pass

        # Search for the occupancy block
        occupancy_block_match = occupancy_pattern.search(cif_string)
        
        if occupancy_block_match:
            occupancy_block = occupancy_block_match.group(1)
            # Find all occupancy values
            occupancy_values = re.findall(r'\s+\S+\s+(\d*\.\d+)', occupancy_block)
            
            # Check for occupancy less than 1.0
            has_low_occupancy = any(float(occupancy) < 1.0 for occupancy in occupancy_values)
            
            if has_low_occupancy:
                progress_queue.put(1)
                continue
                
        clean_cifs.append((id, cif_str))

        progress_queue.put(1)

    result_queue.put(clean_cifs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process CIF files.")
    parser.add_argument("name", type=str,
                        help="Path to the file with the CIFs to be pre-processed. It is expected that the file "
                             "contains the gzipped contents of a pickled Python list of tuples, of (id, cif) "
                             "pairs.")
    parser.add_argument("--out", "-o", action="store",
                        required=True,
                        help="Path to the file where the pre-processed CIFs will be stored. "
                             "The file will contain the gzipped contents of a pickle dump. It is "
                             "recommended that the filename end in `.pkl.gz`.")
    parser.add_argument("--workers", type=int, default=4,
                        help="The number of workers to use for processing. Default is 4.")

    args = parser.parse_args()

    cifs_fname = args.name
    out_fname = args.out
    workers = args.workers

    print(f"loading data from {cifs_fname}...")
    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    for id, cif in cifs:
        task_queue.put((id, cif))

    watcher = mp.Process(target=progress_listener, args=(progress_queue, len(cifs),))

    processes = [mp.Process(target=clean_oxidation_cif, args=(progress_queue, task_queue, result_queue))
                 for _ in range(workers)]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    modified_cifs = []

    while not result_queue.empty():
        modified_cifs.extend(result_queue.get())

    print(f"number of CIFs: {len(modified_cifs)}")

    print(f"saving data to {out_fname}...")
    with gzip.open(out_fname, "wb") as f:
        pickle.dump(modified_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
