#!/bin/bash

# Settings
counter=1
dir=$(pwd)/pdb_parser_scripts/
pdb_dir=$1
pdbs=$pdb_dir/raw/*.pdb
n_pdbs=$(echo $pdbs | wc -w)

# Create data directories
mkdir -p $pdb_dir/cleaned

# Clean pdbs
for pdb in $pdbs;
do
    python $dir/clean_pdb.py --pdb_file_in $pdb  \
                             --out_dir $pdb_dir/cleaned/ \
                             #&> /dev/null

    # Check for exit code 0 and skip file if not 0.
    if [ $? -eq 0 ]
    then
    echo "Successfully cleaned $pdb. $counter/$n_pdbs."
    else
    echo "Error when cleaning $pdb. Skipping.." >&2
    fi
    counter=$((counter+1))
done
