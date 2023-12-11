import argparse
import enum
import os
import sys
import json
import glob
import Bio
import Bio.PDB
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pickle

def parse(pdb_dir):
    # Load PDBS
    pdb_filenames = sorted(glob.glob(f"{pdb_dir}/cleaned/*.pdb"))

    # Create fasta file
    fh = open(f"{pdb_dir}/seqs.fasta","w")

    # Initialize list of pdb dicts
    pdb_dict_list = []
    
    # Loop over proteins
    for pdb_filename in pdb_filenames:
        
        # Parse structure with Biopython
        pdb_parser = Bio.PDB.PDBParser()
        pdb_id = os.path.basename(pdb_filename).split("/")[-1][:-4]
        structure = pdb_parser.get_structure(pdb_id, pdb_filename)
        first_model = structure.get_list()[0]
        first_model.child_list = sorted(first_model.child_list) # Sort chains alphabetically
    
        # Iterate over chain,residue,atoms and extract features
        for chain in first_model: # Loop over chains even though there is only 1       
            
            # Initialize
            chain_id = chain.id
            seq = []
            coords = []
            pdb_dict = {}
            
            for j, residue in enumerate(chain):
                atom_names = []
                backbone_coords = []
     
                for atom in residue:
                    # Extract atom features
                    if atom.name in ["N","CA","C","O"]:
                        atom_names.append(atom.name)
                        #backbone_coords.append(list(atom.coord))
                        backbone_coords.append([str(x) for x in atom.coord])
                
                # Check that all backbone atoms are present
                if atom_names == ["N","CA","C","O"] and len(backbone_coords)==4 and residue._id[0].startswith("H_") == False: # HETATM check
                    
                    # Add coordinates
                    coords.append(backbone_coords)
                    
                    # Add residue to sequence
                    seq.append(Bio.PDB.Polypeptide.three_to_one(residue.resname))
            
            # Save coords+seq to dict 
            pdb_dict["name"] = pdb_id
            pdb_dict["coords"] = coords
            pdb_dict["seq"] = "".join(seq)
            pdb_dict_list.append(pdb_dict)
            
            # Output seq to fasta
            fh.write(f">{pdb_dict['name']}\n")
            fh.write("".join(seq))
            fh.write("\n")
    fh.close()

    # Save total coord dict
    with open(f'{pdb_dir}/coords.json', 'w') as fp:
        json.dump(pdb_dict_list, fp)
    fp.close()

def parse_openproteinset(pdb_dir):
    # Load PDBS
    print("Fetching PDB filenames")
    pdb_filenames = sorted(glob.glob(f"{pdb_dir}/**/pdb/*.pdb"))
    pdb_filenames = sorted(glob.glob(f"../../openproteinset_uniclust30/**/pdb/*.pdb"))

    # Create fasta file
    fh = open(f"{pdb_dir}/seqs.fasta","w")    

    # Initialize list of pdb dicts
    pdb_dict_list = []

    # Loop over proteins
    print("Parsing PDBs")
    for i, pdb_filename in enumerate(pdb_filenames):

        # Parse structure with Biopython
        pdb_parser = Bio.PDB.PDBParser()
        pdb_id = os.path.basename(pdb_filename).split("/")[-1].split(".")[0]
        print(f"{pdb_id}: {i+1}/{len(pdb_filenames)}")
        structure = pdb_parser.get_structure(pdb_id, pdb_filename)
        first_model = structure.get_list()[0]
        first_model.child_list = sorted(first_model.child_list) # Sort chains alphabetically

        # Iterate over chain,residue,atoms and extract features
        for chain in first_model: # Loop over chains even though there is only 1

            # Initialize
            chain_id = chain.id
            seq = []
            coords = []
            b_factors = []
            pdb_dict = {}

            for j, residue in enumerate(chain):
                atom_names = []
                backbone_coords = []
                atom_bfactors = []

                for atom in residue:
                    # Extract atom features
                    if atom.name in ["N","CA","C","O"]:
                        atom_names.append(atom.name)
                        backbone_coords.append([str(x) for x in atom.coord])
                        atom_bfactors.append(atom.bfactor)

                # Check that all backbone atoms are present
                if atom_names == ["N","CA","C","O"] and len(backbone_coords)==4 and residue._id[0].startswith("H_") == False: # HETATM check

                    # Add coordinates
                    coords.append(backbone_coords)

                    # Add residue to sequence
                    seq.append(Bio.PDB.Polypeptide.three_to_one(residue.resname))

                    # Add B factor
                    b_factors.append(mean(atom_bfactors))

            # Save coords+seq to dict
            pdb_dict["name"] = pdb_id
            pdb_dict["coords"] = coords
            pdb_dict["seq"] = "".join(seq)
            pdb_dict["b_factors"] = b_factors
            pdb_dict_list.append(pdb_dict)

            # Output seq to fasta
            fh.write(f">{pdb_dict['name']}\n")
            fh.write("".join(seq))
            fh.write("\n")

    # Save total coord dict
    with open(f'{pdb_dir}/coords_with_bfactors.json', 'w') as fp:
        json.dump(pdb_dict_list, fp)
    fp.close()
    fh.close()

if __name__ == '__main__':
    parse_pdbs()
