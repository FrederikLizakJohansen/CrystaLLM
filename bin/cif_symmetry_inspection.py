import sys
import re
sys.path.append(".")
import argparse
import tarfile
import queue
import multiprocessing as mp
import traceback

from tqdm import tqdm
import numpy as np
import pandas as pd

from crystallm import (
    CIFTokenizer,
    extract_space_group_symbol,
    replace_symmetry_operators,
    extract_volume,
    extract_data_formula,
    is_valid,
    semisymmetrize_cif,
    replace_symmetry_loop
)

from pymatgen.io.cif import CifWriter, Structure, CifParser 
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def read_generated_cifs(input_path):
    generated_cifs = []
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting generated CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                cif = f.read().decode("utf-8")
                generated_cifs.append(cif)
    return generated_cifs

def inspect_cif(cif_path):

    tokenizer = CIFTokenizer()

    #with open(cif_path, 'r') as f:
    #    cif = f.read()
    #cif = read_generated_cifs(cif_path)

    struct = Structure.from_file(cif_path)
    cif = CifWriter(struct=struct, symprec=0.1).__str__()
        
    lines = cif.split('\n')
    cif_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
            cif_lines.append(line)
    cif_lines.append("\n")
    cif = "\n".join(cif_lines)
    
    # Generated len
    gen_len = len(tokenizer.tokenize_cif(cif))
    print("Generated len:", gen_len)

    # Spacegroup
    space_group_symbol = extract_space_group_symbol(cif)
    print("Spacegroup symbol:", space_group_symbol)

    # Print plain
    print("CIF Plain:")
    print(cif)
    print()

    # Semisym
    #cif = semisymmetrize_cif(cif)
    cif = replace_symmetry_loop(cif)
    print("CIF SemiSym:")
    print(cif)
    print()
    

    # Replace symmetries
    #if space_group_symbol is not None and space_group_symbol != "P 1":
    #    cif = replace_symmetry_operators(cif, space_group_symbol)
    #    print("CIF replaced:")
    #    print(cif)
    #    print()
    #else:
    #    print("Could not- or should not replace symmetry")


    # Data formula and volume
    gen_vol = extract_volume(cif)
    print("Generated volume:", gen_vol)
    data_formula = extract_data_formula(cif)
    print("Data formula:", data_formula)
    print("Composition:", Composition(data_formula))
    print()
    
    parser = CifParser.from_string(cif)
    cif_data = parser.as_dict()
    formula_sum = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"])
    formula_structural = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_structural"])
    print("Reduced form:", formula_sum.reduced_formula)
    print("Formula Structural:", formula_structural.reduced_formula)
    
    # Convert the formula sum into a dictionary
    expected_atoms = Composition(formula_sum).as_dict()

    # Count the atoms provided in the _atom_site_type_symbol section
    actual_atoms = {}
    for key in cif_data:
        if "_atom_site_type_symbol" in cif_data[key] and "_atom_site_symmetry_multiplicity" in cif_data[key]:
            for atom_type, multiplicity in zip(cif_data[key]["_atom_site_type_symbol"],
                                               cif_data[key]["_atom_site_symmetry_multiplicity"]):

                atom_type = re.sub(r'[0-9+-]', '', atom_type)

                if atom_type in actual_atoms:
                    actual_atoms[atom_type] += int(multiplicity)
                else:
                    actual_atoms[atom_type] = int(multiplicity)

    print('Expected:', expected_atoms)
    print('Actual:', actual_atoms)
    print(expected_atoms == actual_atoms)
    print({'Sn': 2.0} == {'Sn': 2})

    stated_space_group = cif_data[list(cif_data.keys())[0]]['_symmetry_space_group_name_H-M']
    print("Stated Space Group:", stated_space_group)

    # Analyze the symmetry of the structure
    structure = Structure.from_str(cif, fmt="cif")
    spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=0.1)

    # Get the detected space group
    detected_space_group = spacegroup_analyzer.get_space_group_symbol()
    print("Detected Space Group:", detected_space_group)

    # Lets replace the symm again
    cif = replace_symmetry_operators(cif, space_group_symbol)
    print('Resym:\n',cif)
    
    stated_space_group = cif_data[list(cif_data.keys())[0]]['_symmetry_space_group_name_H-M']
    print("Stated Space Group:", stated_space_group)

    # Analyze the symmetry of the structure
    structure = Structure.from_str(cif, fmt="cif")
    spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=0.1)

    # Get the detected space group
    detected_space_group = spacegroup_analyzer.get_space_group_symbol()
    print("Detected Space Group:", detected_space_group)

    # Is valid
    valid = is_valid(cif, bond_length_acceptability_cutoff=1.0)
    print("Valid? blac=1.0:", valid)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect CIF")
    parser.add_argument("--path", help="Path to the CIF", type=str)
    args = parser.parse_args()

    inspect_cif(args.path)

