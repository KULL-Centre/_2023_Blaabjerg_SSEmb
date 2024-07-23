import sys
import glob
import re
import torch
import models.gvp.data, models.gvp.models
import json
import torch_geometric
import esm
import pandas as pd
import random
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from models.msa_transformer.model import MSATransformer
from models.gvp.models import SSEmbGNN
from helpers import (
    read_msa,
    loop_pred,
)
from visualization import plot_rocklin
import torch.utils.data
from collections import OrderedDict
import shutil
import pdb_parser_scripts.parse_pdbs as parse_pdbs
from collections import OrderedDict
from ast import literal_eval
import subprocess


def test(run_name, epoch, num_ensemble=1, device=None):
    # Download raw data
    subprocess.run(
        ["wget", "https://zenodo.org/record/7992926/files/AlphaFold_model_PDBs.zip"]
    )
    subprocess.run(
        ["unzip", "AlphaFold_model_PDBs.zip", "-d", "../data/test/rocklin/raw/"]
    )
    subprocess.run(["rm", "AlphaFold_model_PDBs.zip"])

    subprocess.run(
        [
            "wget",
            "https://zenodo.org/record/7992926/files/Processed_K50_dG_datasets.zip",
        ]
    )
    subprocess.run(
        ["unzip", "Processed_K50_dG_datasets.zip", "-d", "../data/test/rocklin/raw/"]
    )
    subprocess.run(["rm", "Processed_K50_dG_datasets.zip"])

    # Load data
    df = pd.read_csv(
        "../data/test/rocklin/raw/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
    )

    # Use only Dataset 3 with well-defined ddG's
    df = df[df["ddG_ML"] != "-"]

    # Switch sign of experimental ddG's
    df["ddG_ML"] = -pd.to_numeric(df["ddG_ML"])

    # Use only non-synonomous substitutions
    df = df[~df["mut_type"].str.startswith("ins")]
    df = df[~df["mut_type"].str.startswith("del")]
    df = df[df["mut_type"] != "wt"]

    # Change pdb names to align with structure names
    df["WT_name"] = df["WT_name"].str.replace("|", ":")

    # Move structures
    structures = list(df["WT_name"].unique())
    structures_not_available = []

    for structure in structures:
        try:
            shutil.copy(
                f"../data/test/rocklin/raw/AlphaFold_model_PDBs/{structure}",
                f"../data/test/rocklin/structure/raw/{structure}",
            )
        except:
            structures_not_available.append(structure)

    df = df[~df["WT_name"].isin(structures_not_available)]
    print(
        f"Number of Rocklin assays without available AF2 structures: {len(structures_not_available)}"
    )

    # Save ddG data
    df_ddg = df[["WT_name", "mut_type", "ddG_ML"]].reset_index(drop=True)
    df_ddg = df_ddg.rename(
        columns={"WT_name": "pdb_id", "mut_type": "variant", "ddG_ML": "score_exp"}
    )
    df_ddg["pdb_id"] = df_ddg["pdb_id"].str[:-4]
    df_ddg.to_csv("../data/test/rocklin/exp/ddg.csv", index=False)

    ## Pre-process PDBs
    pdb_dir = "../data/test/rocklin/structure/"
    subprocess.run(
        [
            "pdb_parser_scripts/clean_pdbs.sh",
            str(pdb_dir),
        ]
    )
    parse_pdbs.parse(pdb_dir)

    # Load structure data
    with open(f"../data/test/rocklin/structure/coords.json") as json_file:
        data = json.load(json_file)
    json_file.close()

    # Compute MSAs
    sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    subprocess.run(
        [
            "colabfold_search",
            f"{pdb_dir}/seqs.fasta",
            "/projects/prism/people/skr526/databases",
            "../data/test/rocklin/msa/",
        ]
    )
    subprocess.run(["python", "merge_and_sort_msas.py", "../data/test/rocklin/msa"])

    # Load MSA data
    msa_filenames = sorted(glob.glob(f"../data/test/rocklin/msa/*.a3m"))
    mave_msa_sub = {}
    for i, f in enumerate(msa_filenames):
        # name = f.split("/")[-1].split(".")[0]
        name = f.split("/")[-1].split(".")[0][:-2]
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

    # Convert to graph data sets
    testset = models.gvp.data.ProteinGraphData(data)
    letter_to_num = testset.letter_to_num

    # Make variant pos dict
    variant_pos_dict = {}
    for pdb_id in df_ddg["pdb_id"].unique():
        df_ddg_pdb = df_ddg[df_ddg["pdb_id"] == pdb_id]
        variant_wtpos_list = [
            [x[:-1] for x in x.split(":")] for x in df_ddg_pdb["variant"].tolist()
        ]
        variant_wtpos_list = list(
            OrderedDict.fromkeys(
                [item for sublist in variant_wtpos_list for item in sublist]
            )
        )  # Remove duplicates
        variant_pos_dict[pdb_id] = variant_wtpos_list

    # Load MSA Transformer
    _, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_batch_converter = msa_alphabet.get_batch_converter()
    model_msa = MSATransformer(msa_alphabet)
    model_msa = model_msa.to(device)

    model_dict = OrderedDict()
    state_dict_msa = torch.load(
        f"../output/train/models/msa_transformer/{run_name}_msa_transformer_{epoch}.pt"
    )
    pattern = re.compile("module.")
    for k, v in state_dict_msa.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
        else:
            model_dict = state_dict_msa
    model_msa.load_state_dict(model_dict)

    # Load GVP
    node_dim = (256, 64)
    edge_dim = (32, 1)
    model_gvp = SSEmbGNN((6, 3), node_dim, (32, 1), edge_dim)
    model_gvp = model_gvp.to(device)

    model_dict = OrderedDict()
    state_dict_gvp = torch.load(f"../output/train/models/gvp/{run_name}_gvp_{epoch}.pt")
    pattern = re.compile("module.")
    for k, v in state_dict_gvp.items():
        if k.startswith("module"):
            model_dict[k[7:]] = v
        else:
            model_dict = state_dict_gvp
    model_gvp.load_state_dict(model_dict)

    # Initialize data loader
    test_loader = torch_geometric.loader.DataLoader(
        testset, batch_size=1, shuffle=False
    )

    # Call test
    model_msa.eval()
    model_gvp.eval()

    with torch.no_grad():
        pred_list, acc_mean = loop_pred(
            model_msa,
            model_gvp,
            msa_batch_converter,
            test_loader,
            variant_pos_dict,
            data,
            letter_to_num,
            device=device,
        )

    # Transform results into df
    df_ml = pd.DataFrame(pred_list, columns=["pdb_id", "variant_pos", "score_ml_pos"])

    # Save
    df_ml.to_csv(f"../output/rocklin/df_ml_{run_name}.csv", index=False)

    # Load
    df_ml = pd.read_csv(
        f"../output/rocklin/df_ml_{run_name}.csv",
        converters=dict(score_ml_pos=literal_eval),
    )

    # Compute score_ml from nlls
    # OBS: We cannot vectorize this part since we have an unknown number of mutations per position :(
    pdb_variant_list = df_ddg.values.tolist()
    for i, row in enumerate(pdb_variant_list):
        pdb_id = row[0]
        print(
            f"Computing score for protein {pdb_id} variant: {i+1}/{len(pdb_variant_list)}"
        )
        variant_set = row[1].split(":")

        score_ml = 0.0
        for variant in variant_set:
            wt = letter_to_num[variant[0]]
            pos = int(re.findall(r"\d+", variant)[0])
            mt = letter_to_num[variant[-1]]
            score_ml_pos = df_ml[
                (df_ml["pdb_id"].str.startswith(pdb_id)) & (df_ml["variant_pos"] == pos)
            ]["score_ml_pos"].values[0]
            score_ml += float(score_ml_pos[mt])
        pdb_variant_list[i].append(score_ml)

    # Convert to df
    df_total = pd.DataFrame(
        pdb_variant_list, columns=["DMS_id", "variant_set", "score_exp", "score_ml"]
    )

    # Save
    df_total.to_csv(f"../output/rocklin/df_total_{run_name}.csv", index=False)

    # Load
    df_total = pd.read_csv(f"../output/rocklin/df_total_{run_name}.csv")

    # Compute correlations
    plot_rocklin(df_total)
