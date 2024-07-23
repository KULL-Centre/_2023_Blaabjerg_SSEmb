import torch
import torch.nn as nn
import torch.nn.functional as F
import models.gvp.data, models.gvp.models
from datetime import datetime
import tqdm
import copy
from functools import partial
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
import random
import glob
import pickle
import pandas as pd
import subprocess
import json
from collections import OrderedDict

print = partial(print, flush=True)
import torch.multiprocessing
import torch.distributed as dist

torch.multiprocessing.set_sharing_strategy("file_system")
import pdb_parser_scripts.parse_pdbs as parse_pdbs
from PrismData import PrismParser, VariantData
from scipy import stats
from sklearn.metrics import precision_recall_curve
import pytz
from visualization import plot_scatter, plot_hist

mave_val_pdb_to_prot = {
    "5BON": "NUD15",
    "4QO1": "P53",
    "1CVJ": "PABP",
    "1WYW": "SUMO1",
    "3OLM": "RL401",
    "6NYO": "RL401",
    "3SO6": "LDLRAP1",
    "4QTA": "MAPK",
    "1D5R": "PTEN",
    "2H11": "TPMT",
    "1R9O": "CP2C9",
}


def remove_insertions(sequence: str):
    """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in SeqIO.parse(filename, "fasta")
    ]

def prepare_mave_val(num_ensemble=5):
    # Copy PRISM DMS data to exp dir
    dms_filenames = glob.glob("../data/test/mave_val/raw/*.txt")
    for dms_filename in dms_filenames:
        shutil.copy(dms_filename, "../data/test/mave_val/exp/")

    # Load data
    dms_filenames = sorted(glob.glob(f"../data/test/mave_val/exp/*.txt"))
    df_dms_list = []
    for dms_filename in dms_filenames:
        dms_id = dms_filename.split("/")[-1].split("_")[1]
        df_dms = pd.read_csv(dms_filename, comment="#", delim_whitespace=True)
        df_dms = df_dms[["variant", "score_dms"]]
        df_dms.insert(loc=0, column="dms_id", value=[dms_id] * len(df_dms))
        df_dms_list += df_dms.values.tolist()
    df_dms = pd.DataFrame(df_dms_list, columns=["dms_id", "variant", "score_dms"])

    # Save DMS data
    df_dms.to_csv("../data/test/mave_val/exp/dms.csv", index=False)

    ## Pre-process PDBs
    pdb_dir = "../data/test/mave_val/structure"
    subprocess.run(
        [
            "pdb_parser_scripts/clean_pdbs.sh",
            str(pdb_dir),
        ]
    )
    parse_pdbs.parse(pdb_dir)

    # Load structure data
    print("Loading models and data...")
    with open(f"{pdb_dir}/coords.json") as json_file:
        data = json.load(json_file)
    json_file.close()

    ## Compute MSAs
    #sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    #subprocess.run(
    #    [
    #        "colabfold_search",
    #        f"{pdb_dir}/seqs.fasta",
    #        "/projects/prism/people/skr526/databases",
    #        "../data/test/mave_val/msa/",
    #    ]
    #)
    #subprocess.run(["python", "merge_and_sort_msas.py", "../data/test/mave_val/msa"])

    # Load MSA data
    msa_filenames = sorted(glob.glob(f"../data/test/mave_val/msa/*.a3m"))
    mave_msa_sub = {}
    for i, f in enumerate(msa_filenames):
        name = f.split("/")[-1].split(".")[0]
        mave_msa_sub[name] = []
        for j in range(num_ensemble):
            msa = read_msa(f)
            msa_sub = [msa[0]]
            k = min(len(msa) - 1, 16 - 1)
            msa_sub += [msa[i] for i in sorted(random.sample(range(1, len(msa)), k))]
            mave_msa_sub[name].append(msa_sub)

    # Add MSAs to data
    for entry in data:
        entry["msa"] = mave_msa_sub[entry["name"]]

    # Change data names
    for entry in data:
        entry["name"] = mave_val_pdb_to_prot[entry["name"]]

    # Make variant pos dict
    variant_pos_dict = {}
    for entry in data:
        seq = entry["seq"]
        pos = [str(x + 1) for x in range(len(seq))]
        variant_wtpos_list = [[seq[i] + pos[i]] for i in range(len(seq))]
        variant_wtpos_list = [x for sublist in variant_wtpos_list for x in sublist]
        variant_pos_dict[entry["name"]] = variant_wtpos_list

    # Save data and dict of variant positions
    with open(f"../data/test/mave_val/data_with_msas.pkl","wb") as fp:
        pickle.dump(data, fp)

    with open(f"../data/test/mave_val/variant_pos_dict.pkl","wb") as fp:
        pickle.dump(variant_pos_dict, fp)


def prepare_proteingym_default(num_ensemble=5):
    # Define and save list of validation assays to be excluded
    val_list = [
        "NUD15_HUMAN_Suiter_2020",
        "TPMT_HUMAN_Matreyek_2018",
        "CP2C9_HUMAN_Amorosi_abundance_2021",
        "P53_HUMAN_Kotler_2018",
        "PABP_YEAST_Melamed_2013",
        "SUMO1_HUMAN_Weile_2017",
        "RL401_YEAST_Roscoe_2014",
        "PTEN_HUMAN_Mighell_2018",
        "MK01_HUMAN_Brenan_2016",
    ]

    with open(f"../data/test/proteingym_default/val_list.pkl","wb") as fp:
        pickle.dump(val_list, fp)

    # Load data
    dms_filenames = sorted(
        glob.glob(f"../data/test/proteingym_default/raw/ProteinGym_substitutions/*.csv")
    )
    df_dms_list = []
    for dms_filename in dms_filenames:
        dms_id = dms_filename.split("/")[-1].split(".")[0]
        df_dms = pd.read_csv(dms_filename)
        df_dms = df_dms[["mutant", "DMS_score"]]
        df_dms.insert(loc=0, column="dms_id", value=[dms_id] * len(df_dms))
        df_dms_list += df_dms.values.tolist()
    df_dms = pd.DataFrame(df_dms_list, columns=["dms_id", "variant", "score_dms"])
    df_dms = df_dms[~df_dms["dms_id"].isin(val_list)]

    # Save DMS data
    df_dms.to_csv("../data/test/proteingym_default/exp/dms.csv", index=False)

    ## Pre-process PDBs
    pdb_dir = "../data/test/proteingym_default/structure"
    #subprocess.run(
    #    [
    #        "pdb_parser_scripts/clean_pdbs.sh",
    #        str(pdb_dir),
    #    ]
    #)
    #parse_pdbs.parse(pdb_dir)

    # Load structure data
    print("Loading models and data...")
    with open(f"{pdb_dir}/coords.json") as json_file:
        data_all = json.load(json_file)
    json_file.close()

    # Remove assays from validation set
    data = [x for x in data_all if x["name"] not in val_list]

    ## Compute MSAs
    #sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    #subprocess.run(
    #    [
    #        "colabfold_search",
    #        f"{pdb_dir}/seqs.fasta",
    #        "/projects/prism/people/skr526/databases",
    #        "../data/test/proteingym/msa/",
    #    ]
    #)
    #subprocess.run(["python", "merge_and_sort_msas.py", "../data/test/proteingym/msa"])

    # Load MSA data
    msa_filenames = sorted(glob.glob(f"../data/test/proteingym_default/msa/*.a3m"))
    mave_msa_sub = {}
    for i, f in enumerate(msa_filenames):
        name = f.split("/")[-1].split(".")[0]
        mave_msa_sub[name] = []
        for j in range(num_ensemble):
            msa = read_msa(f)
            msa_sub = [msa[0]]
            k = min(len(msa) - 1, 16 - 1)
            msa_sub += [msa[i] for i in sorted(random.sample(range(1, len(msa)), k))]
            mave_msa_sub[name].append(msa_sub)

    # Add MSAs to data
    for entry in data:
        entry["msa"] = mave_msa_sub[entry["name"]]

    # Make variant pos dict
    variant_pos_dict = {}
    for dms_id in df_dms["dms_id"].unique():
        df_dms_id = df_dms[df_dms["dms_id"] == dms_id]
        variant_wtpos_list = [
            [x[:-1] for x in x.split(":")] for x in df_dms_id["variant"].tolist()
        ]
        variant_wtpos_list = list(
            OrderedDict.fromkeys(
                [item for sublist in variant_wtpos_list for item in sublist]
            )
        )  # Remove duplicates
        variant_pos_dict[dms_id] = variant_wtpos_list    

    # Save data and dict of variant positions
    with open(f"../data/test/proteingym_default/data_with_msas.pkl","wb") as fp:
        pickle.dump(data, fp)

    with open(f"../data/test/proteingym_default/variant_pos_dict.pkl","wb") as fp:
        pickle.dump(variant_pos_dict, fp)

def prepare_proteingym(num_ensemble=5):
    # Define and save list of validation assays to be excluded
    val_list = [
        "NUD15_HUMAN_Suiter_2020",
        "TPMT_HUMAN_Matreyek_2018",
        "CP2C9_HUMAN_Amorosi_abundance_2021",
        "P53_HUMAN_Kotler_2018",
        "PABP_YEAST_Melamed_2013",
        "SUMO1_HUMAN_Weile_2017",
        "RL401_YEAST_Roscoe_2014",
        "PTEN_HUMAN_Mighell_2018",
        "MK01_HUMAN_Brenan_2016",
    ]

    with open(f"../data/test/proteingym/val_list.pkl","wb") as fp:
        pickle.dump(val_list, fp)

    # Load data
    dms_filenames = sorted(
        glob.glob(f"../data/test/proteingym/raw/ProteinGym_substitutions/*.csv")
    )
    df_dms_list = []
    for dms_filename in dms_filenames:
        dms_id = dms_filename.split("/")[-1].split(".")[0]
        df_dms = pd.read_csv(dms_filename)
        df_dms = df_dms[["mutant", "DMS_score"]]
        df_dms.insert(loc=0, column="dms_id", value=[dms_id] * len(df_dms))
        df_dms_list += df_dms.values.tolist()
    df_dms = pd.DataFrame(df_dms_list, columns=["dms_id", "variant", "score_dms"])
    df_dms = df_dms[~df_dms["dms_id"].isin(val_list)]

    # Save DMS data
    df_dms.to_csv("../data/test/proteingym/exp/dms.csv", index=False)

    ### Pre-process PDBs
    pdb_dir = "../data/test/proteingym/structure"
    subprocess.run(
        [
            "pdb_parser_scripts/clean_pdbs.sh",
            str(pdb_dir),
        ]
    )
    parse_pdbs.parse(pdb_dir)

    # Load structure data
    print("Loading models and data...")
    with open(f"{pdb_dir}/coords.json") as json_file:
        data_all = json.load(json_file)
    json_file.close()

    # Remove assays from validation set
    data = [x for x in data_all if x["name"] not in val_list]

    ## Compute MSAs
    #sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    #subprocess.run(
    #    [
    #        "colabfold_search",
    #        f"{pdb_dir}/seqs.fasta",
    #        "/projects/prism/people/skr526/databases",
    #        "../data/test/proteingym/msa/",
    #    ]
    #)
    #subprocess.run(["python", "merge_and_sort_msas.py", "../data/test/proteingym/msa"])

    # Load MSA data
    msa_filenames = sorted(glob.glob(f"../data/test/proteingym/msa/*.a3m"))
    mave_msa_sub = {}
    for i, f in enumerate(msa_filenames):
        name = f.split("/")[-1].split(".")[0]
        mave_msa_sub[name] = []
        for j in range(num_ensemble):
            msa = read_msa(f)
            msa_sub = [msa[0]]
            k = min(len(msa) - 1, 16 - 1)
            msa_sub += [msa[i] for i in sorted(random.sample(range(1, len(msa)), k))]
            mave_msa_sub[name].append(msa_sub)

    # Add MSAs to data
    for entry in data:
        entry["msa"] = mave_msa_sub[entry["name"]]

    # Make variant pos dict
    variant_pos_dict = {}
    for dms_id in df_dms["dms_id"].unique():
        df_dms_id = df_dms[df_dms["dms_id"] == dms_id]
        variant_wtpos_list = [
            [x[:-1] for x in x.split(":")] for x in df_dms_id["variant"].tolist()
        ]
        variant_wtpos_list = list(
            OrderedDict.fromkeys(
                [item for sublist in variant_wtpos_list for item in sublist]
            )
        )  # Remove duplicates
        variant_pos_dict[dms_id] = variant_wtpos_list    

    # Save data and dict of variant positions
    with open(f"../data/test/proteingym/data_with_msas.pkl","wb") as fp:
        pickle.dump(data, fp)

    with open(f"../data/test/proteingym/variant_pos_dict.pkl","wb") as fp:
        pickle.dump(variant_pos_dict, fp)


def prepare_proteingym_bad(num_ensemble=5):
    # Define and save list of validation assays to be excluded
    val_list = [
        "NUD15_HUMAN_Suiter_2020",
        "TPMT_HUMAN_Matreyek_2018",
        "CP2C9_HUMAN_Amorosi_abundance_2021",
        "P53_HUMAN_Kotler_2018",
        "PABP_YEAST_Melamed_2013",
        "SUMO1_HUMAN_Weile_2017",
        "RL401_YEAST_Roscoe_2014",
        "PTEN_HUMAN_Mighell_2018",
        "MK01_HUMAN_Brenan_2016",
    ]

    with open(f"../data/test/proteingym_bad/val_list.pkl","wb") as fp:
        pickle.dump(val_list, fp)

    # Load data
    dms_filenames = sorted(
        glob.glob(f"../data/test/proteingym_bad/raw/ProteinGym_substitutions/*.csv")
    )
    df_dms_list = []
    for dms_filename in dms_filenames:
        dms_id = dms_filename.split("/")[-1].split(".")[0]
        df_dms = pd.read_csv(dms_filename)
        df_dms = df_dms[["mutant", "DMS_score"]]
        df_dms.insert(loc=0, column="dms_id", value=[dms_id] * len(df_dms))
        df_dms_list += df_dms.values.tolist()
    df_dms = pd.DataFrame(df_dms_list, columns=["dms_id", "variant", "score_dms"])
    df_dms = df_dms[~df_dms["dms_id"].isin(val_list)]

    # Save DMS data
    df_dms.to_csv("../data/test/proteingym_bad/exp/dms.csv", index=False)

    ### Pre-process PDBs
    pdb_dir = "../data/test/proteingym_bad/structure"
    subprocess.run(
        [
            "pdb_parser_scripts/clean_pdbs.sh",
            str(pdb_dir),
        ]
    )
    parse_pdbs.parse(pdb_dir)

    # Load structure data
    print("Loading models and data...")
    with open(f"{pdb_dir}/coords.json") as json_file:
        data_all = json.load(json_file)
    json_file.close()

    # Remove assays from validation set
    data = [x for x in data_all if x["name"] not in val_list]

    ## Compute MSAs
    #sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    #subprocess.run(
    #    [
    #        "colabfold_search",
    #        f"{pdb_dir}/seqs.fasta",
    #        "/projects/prism/people/skr526/databases",
    #        "../data/test/proteingym_bad/msa/",
    #    ]
    #)
    #subprocess.run(["python", "merge_and_sort_msas.py", "../data/test/proteingym_bad/msa"])

    # Load MSA data
    msa_filenames = sorted(glob.glob(f"../data/test/proteingym_bad/msa/*.a3m"))
    mave_msa_sub = {}
    for i, f in enumerate(msa_filenames):
        name = f.split("/")[-1].split(".")[0]
        mave_msa_sub[name] = []
        for j in range(num_ensemble):
            msa = read_msa(f)
            msa_sub = [msa[0]]
            k = min(len(msa) - 1, 16 - 1)
            msa_sub += [msa[i] for i in sorted(random.sample(range(1, len(msa)), k))]
            mave_msa_sub[name].append(msa_sub)

    # Add MSAs to data
    for entry in data:
        entry["msa"] = mave_msa_sub[entry["name"]]

    # Make variant pos dict
    variant_pos_dict = {}
    for dms_id in df_dms["dms_id"].unique():
        df_dms_id = df_dms[df_dms["dms_id"] == dms_id]
        variant_wtpos_list = [
            [x[:-1] for x in x.split(":")] for x in df_dms_id["variant"].tolist()
        ]
        variant_wtpos_list = list(
            OrderedDict.fromkeys(
                [item for sublist in variant_wtpos_list for item in sublist]
            )
        )  # Remove duplicates
        variant_pos_dict[dms_id] = variant_wtpos_list    

    # Save data and dict of variant positions
    with open(f"../data/test/proteingym_bad/data_with_msas.pkl","wb") as fp:
        pickle.dump(data, fp)

    with open(f"../data/test/proteingym_bad/variant_pos_dict.pkl","wb") as fp:
        pickle.dump(variant_pos_dict, fp)

def save_df_to_prism(df, run_name, dms_id):
    # Initialize
    parser = PrismParser()

    # Print to PRISM format
    df_prism = df[["variant", "score_ml"]].reset_index(drop=True)

    # Get current CPH time
    timestamp = datetime.now(pytz.timezone("Europe/Copenhagen")).strftime(
        "%Y-%m-%d %H:%M"
    )

    metadata = {
        "version": 1,
        "protein": {
            "name": "Unknown",
            "organism": "Unknown",
            "uniprot": "Unknown",
            "sequence": "".join([x for x in list(df_prism["variant"].str[0])[::20]]),
            "first_residue_number": 1,
            "pdb": "Unknown",
            "chain": "Unknown",
        },
        "columns": {"score_ml": "SSEmb prediction"},
        "created": {f"{timestamp} (CPH time) - lasse.blaabjerg@bio.ku.dk"},
    }

    # Write data
    dataset = VariantData(metadata, df_prism)
    dataset.add_index_columns()
    prism_filename = f"../output/mave_val/prism_{dms_id}_ssemb_{run_name}.txt"
    parser.write(prism_filename, dataset, comment_lines="")


def get_prism_corr(dms_id, run_name):
    # Initialize
    parser = PrismParser()

    # Load PRISM file - Experimental
    prism_pred = f"../data/test/mave_val/exp/prism_{dms_id}_dms.txt"
    df_prism_dms = parser.read(prism_pred)
    len_dms = len(df_prism_dms.dataframe)

    # Load PRISM file - SSEmb prediction
    prism_pred = f"../output/mave_val/prism_{dms_id}_ssemb_{run_name}.txt"
    df_prism_ssemb = parser.read(prism_pred)

    # Merge PRISM files
    df_total = df_prism_dms.merge([df_prism_ssemb], merge="inner").dataframe
    len_total = len(df_total)
    print(
        f"{dms_id} number of MAVE data points lost during merging: {len_dms-len_total}"
    )

    # Get correlation
    x = df_total["score_dms_00"].values
    y = df_total["score_ml_01"].values
    spearman_r_ssemb = stats.spearmanr(x, y)[0]
    print(f"{dms_id}: SSEmb spearman correlation vs. MAVE is: {spearman_r_ssemb:.3f}")
    return spearman_r_ssemb


def get_prism_corr_all(dms_id, run_name):
    # Initialize
    parser = PrismParser()

    # Load PRISM file - Experimental
    prism_pred = f"../data/test/mave_val/exp/prism_{dms_id}_dms.txt"
    df_prism_dms = parser.read(prism_pred)
    len_dms = len(df_prism_dms.dataframe)

    # Load PRISM file - SSEmb prediction
    prism_pred = f"../output/mave_val/prism_{dms_id}_ssemb_{run_name}.txt"
    df_prism_ssemb = parser.read(prism_pred)

    # Load PRISM file - GEMME
    prism_pred = f"../data/test/mave_val/gemme/prism_gemme_{dms_id}.txt"
    df_prism_gemme = parser.read(prism_pred)

    # Load PRISM file - Rosetta
    prism_pred = f"../data/test/mave_val/rosetta/prism_rosetta_{dms_id}.txt"
    df_prism_rosetta = parser.read(prism_pred)

    # Merge PRISM files
    df_total = df_prism_dms.merge(
        [df_prism_ssemb, df_prism_gemme, df_prism_rosetta], merge="inner"
    ).dataframe
    len_total = len(df_total)
    print(
        f"{dms_id} number of MAVE data points lost during merging: {len_dms-len_total}"
    )

    # Get correlation
    x = df_total["score_dms_00"].values
    y = df_total["score_ml_01"].values
    spearman_r_ssemb = stats.spearmanr(x, y)[0]
    print(f"{dms_id}: SSEmb spearman correlation vs. MAVE is: {spearman_r_ssemb:.3f}")

    x = df_total["score_dms_00"].values
    y = df_total["gemme_score_02"].values
    spearman_r_gemme = stats.spearmanr(x, y)[0]
    print(f"{dms_id}: GEMME spearman correlation vs. MAVE is: {spearman_r_gemme:.3f}")

    x = df_total["score_dms_00"].values
    y = df_total["mean_ddG_03"].values
    spearman_r_rosetta = stats.spearmanr(x, y)[0]
    print(
        f"{dms_id}: Rosetta spearman correlation vs. MAVE is: {spearman_r_rosetta:.3f}"
    )

    # Make scatter plot
    plot_scatter(df_total, dms_id, run_name, "mave_val")

    # Make histogram plot
    plot_hist(df_total, dms_id, run_name, "mave_val")
    return spearman_r_ssemb, spearman_r_gemme, spearman_r_rosetta


def scannet_collate_fn(batch):
    emb = batch[0]["emb"].unsqueeze(0)
    label = batch[0]["label"].unsqueeze(0)
    prot_weight = batch[0]["prot_weight"]
    return emb, label, prot_weight


def mask_seq_and_msa(
    seq,
    msa_batch_tokens,
    coord_mask,
    device,
    mask_size=0.15,
    mask_tok=0.60,
    mask_col=0.20,
    mask_rand=0.10,
    mask_same=0.10,
):
    # Get masked positions
    assert mask_tok + mask_col + mask_rand + mask_same == 1.00
    indices = torch.arange(len(seq), device=device)
    indices_mask = indices[coord_mask]  # Only consider indices within coord mask
    indices_mask = indices_mask[torch.randperm(indices_mask.size(0))]
    mask_pos_all = indices_mask[: int(len(indices_mask) * mask_size)]

    mask_pos_tok = mask_pos_all[: int(len(mask_pos_all) * mask_tok)]
    mask_pos_col = mask_pos_all[
        int(len(mask_pos_all) * (mask_tok)) : int(
            len(mask_pos_all) * (mask_tok + mask_col)
        )
    ]
    mask_pos_rand = mask_pos_all[
        int(len(mask_pos_all) * (mask_tok + mask_col)) : int(
            len(mask_pos_all) * (mask_tok + mask_col + mask_rand)
        )
    ]

    # Do masking - MSA level
    msa_batch_tokens_masked = msa_batch_tokens.clone()
    msa_batch_tokens_masked[:, 0, mask_pos_tok + 1] = 32  # Correct for <cls> token
    msa_batch_tokens_masked[:, :, mask_pos_col + 1] = 32  # Correct for <cls> token
    msa_batch_tokens_masked[:, 0, mask_pos_rand + 1] = torch.randint(
        low=4, high=24, size=(len(mask_pos_rand),), device=device
    )  # Correct for <cls> token, draw random standard amino acids

    # Do masking - seq level
    seq_masked = seq.clone()
    mask_pos_tok_all = torch.cat((mask_pos_tok, mask_pos_col))
    seq_masked[mask_pos_tok_all] = 20
    seq_masked[mask_pos_rand] = torch.randint(
        low=0, high=20, size=(len(mask_pos_rand),), device=device
    )

    return seq_masked, msa_batch_tokens_masked, mask_pos_all


def forward(
    model_msa,
    model_gvp,
    msa_batch_tokens_masked,
    seq_masked,
    batch,
    msa_row_attn_mask=True,
    mask_pos=None,
    loss_fn=None,
    batch_prots=None,
    get_logits_only=False,
):
    # Make MSA Transformer predictions
    if msa_row_attn_mask == True:
        msa_transformer_pred = model_msa(
                        msa_batch_tokens_masked, repr_layers=[12], 
                        self_row_attn_mask=batch.dist_mask
                        )
    elif msa_row_attn_mask == False:
        msa_transformer_pred = model_msa(
                        msa_batch_tokens_masked, repr_layers=[12])
    msa_emb = msa_transformer_pred["representations"][12][0, 0, 1:, :]

    # Make GVP predictions
    h_V = (batch.node_s, batch.node_v)
    h_E = (batch.edge_s, batch.edge_v)
    logits = model_gvp(h_V, batch.edge_index, h_E, msa_emb, seq_masked)

    if get_logits_only == True:
        # Return logits
        return logits
    else:
        # Compute and return loss
        logits, seq = logits[mask_pos], batch.seq[mask_pos]
        loss_value = loss_fn(logits, seq)
        loss_value = loss_value / batch_prots
        return loss_value, logits, seq


def loop_trainval(
    model_msa,
    model_gvp,
    msa_batch_converter,
    dataloader,
    batch_prots,
    epoch,
    rank,
    epoch_finetune_msa,
    msa_row_attn_mask=True,
    optimizer=None,
    scaler=None,
):
    # Initialize
    t = tqdm.tqdm(dataloader)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total_count = 0, 0, 0

    # Initialize models and optimizer
    if optimizer == None:
        model_gvp = model_gvp.module
        if epoch >= epoch_finetune_msa:
            model_msa = model_msa.module
    else:
        optimizer.zero_grad(set_to_none=True)

    # Loop over proteins
    for i, batch in enumerate(t):
        print(f"Rank {rank} - Computing predictions for protein: {batch.name[0]}")

        with torch.cuda.amp.autocast(
            enabled=True
        ):  # OBS: This seems to be necessary for GVPLarge model close to convergence?
            # Move data to device
            batch = batch.to(rank)

            # Subsample MSA
            msa_sub = [batch.msa[0][0]]  # Always get query
            k = min(len(batch.msa[0]) - 1, 16 - 1)
            msa_sub += [
                batch.msa[0][j]
                for j in sorted(random.sample(range(1, len(batch.msa[0])), k))
            ]

            # Tokenize MSA
            msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(
                msa_sub
            )
            msa_batch_tokens = msa_batch_tokens.to(rank)

            # Mask sequence
            seq_masked, msa_batch_tokens_masked, mask_pos = mask_seq_and_msa(
                batch.seq, msa_batch_tokens, batch.mask, rank
            )

            if optimizer:
                if (i + 1) % batch_prots == 0 or (i + 1) == len(
                    t
                ):  # Accumulate gradients and update every n'th protein or last iteration
                    # Forward pass
                    loss_value, logits, seq = forward(
                        model_msa,
                        model_gvp,
                        msa_batch_tokens_masked,
                        seq_masked,
                        batch,
                        msa_row_attn_mask=msa_row_attn_mask,
                        mask_pos=mask_pos,
                        loss_fn=loss_fn,
                        batch_prots=batch_prots,
                    )

                    # Backprop
                    scaler.scale(loss_value).backward()

                    # Optimizer step
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(
                            model_msa.parameters(),
                            model_gvp.parameters(),
                        ),
                        1.0,
                    )  # Clip gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # Create barrier
                    dist.barrier()
                else:
                    if epoch < epoch_finetune_msa:
                        with model_gvp.no_sync():
                            # Forward pass
                            loss_value, logits, seq = forward(
                                model_msa,
                                model_gvp,
                                msa_batch_tokens_masked,
                                seq_masked,
                                batch,
                                msa_row_attn_mask=msa_row_attn_mask,
                                mask_pos=mask_pos,
                                loss_fn=loss_fn,
                                batch_prots=batch_prots,
                            )

                            # Backprop
                            scaler.scale(loss_value).backward()
                    else:
                        with model_gvp.no_sync():
                            with model_msa.no_sync():
                                # Forward pass
                                loss_value, logits, seq = forward(
                                    model_msa,
                                    model_gvp,
                                    msa_batch_tokens_masked,
                                    seq_masked,
                                    batch,
                                    msa_row_attn_mask=msa_row_attn_mask,
                                    mask_pos=mask_pos,
                                    loss_fn=loss_fn,
                                    batch_prots=batch_prots,
                                )

                                # Backprop
                                scaler.scale(loss_value).backward()
            else:
                # Forward pass
                loss_value, logits, seq = forward(
                    model_msa,
                    model_gvp,
                    msa_batch_tokens_masked,
                    seq_masked,
                    batch,
                    msa_row_attn_mask=msa_row_attn_mask,
                    mask_pos=mask_pos,
                    loss_fn=loss_fn,
                    batch_prots=batch_prots,
                )

        # Update loss etc.
        num_nodes = len(mask_pos)
        total_loss += loss_value.detach()
        total_count += num_nodes
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        true = seq.detach().cpu().numpy()
        total_correct += (pred == true).sum()
        t.set_description("%.5f" % float(total_loss / total_count))
        torch.cuda.empty_cache()

    return total_loss / total_count, total_correct / total_count


def loop_pred(
    model_msa,
    model_gvp,
    msa_batch_converter,
    dataloader,
    variant_pos_dict,
    data,
    letter_to_num,
    msa_row_attn_mask=True,
    device=None,
):
    # Initialize
    t = tqdm.tqdm(dataloader)
    pred_list = []
    total_correct, total_count = 0, 0

    # Loop over proteins
    for i, batch in enumerate(t):
        with torch.cuda.amp.autocast(enabled=True):
            # Move data to device
            batch = batch.to(device)

            # Initialize
            variant_wtpos_list = variant_pos_dict[batch.name[0]]
            seq_len = len(batch.seq)

            # Make masked marginal predictions
            for k, variant_wtpos in enumerate(variant_wtpos_list):
                print(
                    f"Computing logits for protein {batch.name[0]} ({i+1}/{len(dataloader)}) at position: {k+1}/{len(variant_wtpos_list)}"
                )

                # Extract variant info and initialize
                wt = letter_to_num[variant_wtpos[0]]
                pos = int(variant_wtpos[1:]) - 1  # Shift from DMS pos to seq idx
                score_ml_pos_ensemble = torch.zeros((len(batch.msa[0]), 20))

                # If protein too long; redo data loading with fragment
                if seq_len > 1024 - 1:
                    # Get sliding window
                    window_size = 1024 - 1
                    lower_side = max(pos - window_size // 2, 0)
                    upper_side = min(pos + window_size // 2 + 1, seq_len)
                    lower_bound = lower_side - (pos + window_size // 2 + 1 - upper_side)
                    upper_bound = upper_side + (lower_side - (pos - window_size // 2))

                    # Get fragment
                    data_frag = copy.deepcopy(data[i])
                    data_frag["seq"] = data[i]["seq"][lower_bound:upper_bound]
                    data_frag["coords"] = data[i]["coords"][lower_bound:upper_bound]
                    data_frag["msa"] = [
                        [(seq[0], seq[1][lower_bound:upper_bound]) for seq in msa_sub]
                        for msa_sub in data[i]["msa"]
                    ]
                    batch = models.gvp.data.ProteinGraphData([data_frag])[0]
                    batch = batch.to(device)
                    batch.msa = [batch.msa]
                    batch.name = [batch.name]

                    # Re-map position
                    pos = pos - lower_bound

                # Loop over MSA ensemble
                for j, msa_sub in enumerate(batch.msa[0]):
                    # Tokenize MSA
                    (
                        msa_batch_labels,
                        msa_batch_strs,
                        msa_batch_tokens,
                    ) = msa_batch_converter(msa_sub)
                    msa_batch_tokens = msa_batch_tokens.to(device)

                    # Mask position
                    msa_batch_tokens_masked = msa_batch_tokens.detach().clone()
                    msa_batch_tokens_masked[
                        :, 0, pos + 1
                    ] = 32  # Account for appended <cls> token
                    seq_masked = batch.seq.detach().clone()
                    seq_masked[pos] = 20

                    # Forward pass
                    logits = forward(
                        model_msa,
                        model_gvp,
                        msa_batch_tokens_masked,
                        seq_masked,
                        batch,
                        msa_row_attn_mask=msa_row_attn_mask,
                        get_logits_only=True,
                    )
                    logits_pos = logits[pos, :]

                    # Compute accuracy
                    pred = (
                        torch.argmax(logits_pos, dim=-1).detach().cpu().numpy().item()
                    )
                    true = batch.seq[pos].detach().cpu().numpy().item()
                    if pred == true:
                        total_correct += 1 / len(batch.msa[0])

                    # Compute all possible nlls at this position based on known wt
                    nlls_pos = -torch.log(F.softmax(logits_pos, dim=-1))
                    nlls_pos_repeat = nlls_pos.repeat(20, 1)
                    score_ml_pos_ensemble[j, :] = torch.diagonal(
                        nlls_pos_repeat[:, wt] - nlls_pos_repeat[:, torch.arange(20)]
                    )

                # Append to total
                score_ml_pos = torch.mean(score_ml_pos_ensemble[: j + 1, :], axis=0)
                pred_list.append(
                    [
                        batch.name[0],
                        int(variant_wtpos[1:]),
                        score_ml_pos.detach().cpu().tolist(),
                    ]
                )
                total_count += 1

    return pred_list, total_correct / total_count


def loop_getemb(model_msa, model_gvp, msa_batch_converter, dataloader, device=None):
    # Initialize
    t = tqdm.tqdm(dataloader)
    emb_dict = {}

    for i, batch in enumerate(t):
        batch = batch.to(device)

        with torch.cuda.amp.autocast(enabled=True):
            # Get frozen embeddings
            msa_sub = batch.msa[0]
            msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(
                msa_sub
            )
            msa_batch_tokens = msa_batch_tokens.to(device)

            # Make MSA Transformer predictions
            msa_batch_tokens_masked = msa_batch_tokens.detach().clone()
            seq = batch.seq.clone()
            msa_transformer_pred = model_msa(
                msa_batch_tokens_masked,
                repr_layers=[12],
                self_row_attn_mask=batch.dist_mask,
            )
            msa_emb = msa_transformer_pred["representations"][12][0, 0, 1:, :]

            # Make GVP predictions
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            emb_allpos = model_gvp(
                h_V, batch.edge_index, h_E, msa_emb, seq, get_emb=True
            )

            # Update dict
            emb_dict[batch.name[0]] = emb_allpos

    return emb_dict


def loop_scannet_trainval(
    model_transformer, dataloader, device=None, optimizer=None, batch_prots=None
):
    # Initialize
    t = tqdm.tqdm(dataloader)
    total_loss, total_correct, total_count = 0, 0, 0
    acc_prots = []
    matthews_prots = []
    perplexity_prots = []
    pred_list = []
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for i, (emb, label, prot_weight) in enumerate(t):
        # print(f"{i+1}/{len(t)}")

        # Transfer to GPU
        emb, label = emb.to(device), label.to(device)

        # Make Transformer prediction
        emb = emb.transpose(1, 0)  # B x N x H --> N x B x H
        label_pred = model_transformer(emb)
        weights = torch.ones(label.reshape(-1).size(), device=label.device)
        weights[label.reshape(-1) == 1] = 4
        loss_value = loss_fn(label_pred.reshape(-1), label.reshape(-1).float())
        loss_value = torch.mean(loss_value * weights)
        loss_value = loss_value * prot_weight

        if optimizer:
            loss_value.backward()

            # Gradient accumulation
            if ((i + 1) % batch_prots == 0) or (i + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

        # Update loss etc.
        num_nodes = int(label.reshape(-1).size()[0])
        total_loss += loss_value.detach() * num_nodes
        total_count += num_nodes
        pred = torch.round(torch.sigmoid(label_pred.reshape(-1))).detach().cpu().numpy()
        true = label.reshape(-1).detach().cpu().numpy()
        total_correct += (pred == true).sum()
        t.set_description("%.4f" % float((pred == true).sum() / num_nodes))
        torch.cuda.empty_cache()

    return total_loss / total_count, total_correct / total_count


def loop_scannet_test(model_transformer, dataloader, device=None):
    # Initialize
    t = tqdm.tqdm(dataloader)
    total_loss, total_correct, total_count = 0, 0, 0
    acc_prots = []
    matthews_prots = []
    perplexity_prots = []
    pred_list = []

    total_label = torch.empty(0).to(device)
    total_label_pred = torch.empty(0).to(device)

    for i, (emb, label, _) in enumerate(t):
        # print(f"{i+1}/{len(t)}")

        # Transfer to GPU
        emb, label = emb.to(device), label.to(device)

        # Make Transformer prediction
        emb = emb.transpose(1, 0)  # B x N x H --> N x B x H
        label_pred = model_transformer(emb)

        # Apply sigmoid and concat
        total_label = torch.cat((total_label, label.reshape(-1)))
        total_label_pred = torch.cat((total_label_pred, label_pred.reshape(-1)))

    # Compute AUCPR
    precision, recall, thresholds = precision_recall_curve(
        total_label.detach().cpu().numpy(), total_label_pred.detach().cpu().numpy()
    )
    return precision, recall

def compute_auc_group(group):
    protein_scores = group['score']
    protein_labels = group['label_bin']
    try:
        result = roc_auc_score(y_true=protein_labels, y_score=protein_scores)
    except:
        result = np.nan
    return result
