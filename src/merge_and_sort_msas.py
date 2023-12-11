import string
from Bio import SeqIO
from typing import List, Tuple
import numpy as np
import glob
import sys
import subprocess

# Initialize
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def remove_insertions(sequence: str):
    """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in SeqIO.parse(filename, "fasta")
    ]


def hamming_distance(string1, string2):
    return sum(c1 != c2 for c1, c2 in zip(string1, string2))


# Initialize
msa_dir = sys.argv[1]
subprocess.run(["mkdir", "-p", f"{msa_dir}_tmp"])

# Load MSA files
msa_files = sorted(glob.glob(f"{msa_dir}/*.a3m"))

# Loop
for i, _file in enumerate(msa_files):
    print(f"Processing MSA: {i+1}/{len(msa_files)}")

    msa = read_msa(_file)

    seqs = [x for x in msa]

    query = seqs[0]
    seqs = seqs[1:]
    ham_dists = np.zeros(len(seqs))

    for j, seq in enumerate(seqs):
        assert len(query) == len(seq)
        ham_dists[j] = hamming_distance(query[1], seq[1])

        # Rank indices
        rank_indices = np.argsort(ham_dists)

        # Remove query duplicates
        if 0 in ham_dists:
            query_idx = np.argwhere(ham_dists == 0)[0]
            rank_indices = np.delete(
                rank_indices, np.argwhere(np.isin(rank_indices, query_idx))
            )

        # Construct new sorted MSA
        seqs_new = []
        for idx in rank_indices:
            seqs_new.append(seqs[idx])

    # Write to new file
    outfile = open(f"{msa_dir}_tmp/{query[0]}.a3m", "w")
    outfile.write(f">{query[0]}\n")
    outfile.write(f"{query[1]}\n")

    for seq in seqs_new:
        outfile.write(f">{seq[0]}\n")
        outfile.write(f"{seq[1]}\n")
    outfile.close()

# Delete tmp directory
subprocess.run(["rm", "-r", f"{msa_dir}"])
subprocess.run(["mv", f"{msa_dir}_tmp", msa_dir])
