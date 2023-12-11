import sys
import subprocess
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
from statistics import mean
from helpers import (
    read_msa,
    mave_val_pdb_to_prot,
    loop_pred,
    save_df_to_prism,
    get_prism_corr,
    get_prism_corr_all,
)
import pdb_parser_scripts.parse_pdbs as parse_pdbs
import torch.utils.data
from collections import OrderedDict
from ast import literal_eval
import subprocess
import shutil


def test(run_name, epoch, num_ensemble=1, get_mean_metrics=False, device=None):
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
        df_dms = df_dms[["variant", "y_raw"]]
        df_dms.insert(loc=0, column="dms_id", value=[dms_id] * len(df_dms))
        df_dms_list += df_dms.values.tolist()
    df_dms = pd.DataFrame(df_dms_list, columns=["dms_id", "variant", "score_exp"])

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

    # Compute MSAs
    sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    subprocess.run(
        [
            "colabfold_search",
            f"{pdb_dir}/seqs.fasta",
            "/projects/prism/people/skr526/databases",
            "../data/test/mave_val/msa/",
        ]
    )
    subprocess.run(["python", "merge_and_sort_msa.py", "../data/test/mave_val/msa"])

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

    # Convert to graph data sets
    testset = models.gvp.data.ProteinGraphData(data)
    letter_to_num = testset.letter_to_num

    # Make variant pos dict
    variant_pos_dict = {}
    for entry in data:
        seq = entry["seq"]
        pos = [str(x + 1) for x in range(len(seq))]
        variant_wtpos_list = [[seq[i] + pos[i]] for i in range(len(seq))]
        variant_wtpos_list = [x for sublist in variant_wtpos_list for x in sublist]
        variant_pos_dict[entry["name"]] = variant_wtpos_list

    # Load MSA Transformer
    _, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model_msa = MSATransformer(msa_alphabet)
    model_msa = model_msa.to(device)
    msa_batch_converter = msa_alphabet.get_batch_converter()

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
            device,
        )

    # Transform results into df
    df_ml = pd.DataFrame(pred_list, columns=["dms_id", "variant_pos", "score_ml_pos"])

    # Save
    df_ml.to_csv(f"../output/mave_val/df_ml_{run_name}.csv", index=False)

    # Load
    df_ml = pd.read_csv(
        f"../output/mave_val/df_ml_{run_name}.csv",
        converters=dict(score_ml_pos=literal_eval),
    )

    # Compute score_ml from nlls
    pred_list_scores = []
    mt_list = [x for x in sorted(letter_to_num, key=letter_to_num.get)][:-1]

    for entry in data:
        dms_id = entry["name"]
        df_dms_id = df_ml[df_ml["dms_id"] == dms_id]

        wt = [[wt] * 20 for wt in entry["seq"]]
        pos = [[pos] * 20 for pos in list(df_dms_id["variant_pos"])]
        pos = [item for sublist in pos for item in sublist]
        mt = mt_list * len(wt)
        wt = [item for sublist in wt for item in sublist]
        score_ml = [
            item for sublist in list(df_dms_id["score_ml_pos"]) for item in sublist
        ]

        rows = [
            [dms_id, wt[i] + str(pos[i]) + mt[i], score_ml[i]] for i in range(len(mt))
        ]
        pred_list_scores += rows

    # Transform results into df
    df_ml_scores = pd.DataFrame(
        pred_list_scores, columns=["dms_id", "variant", "score_ml"]
    )

    # Save
    df_ml_scores.to_csv(f"../output/mave_val/df_ml_scores_{run_name}.csv", index=False)

    # Load
    df_ml_scores = pd.read_csv(f"../output/mave_val/df_ml_scores_{run_name}.csv")

    # Save results to PRISM format
    for dms_id in df_ml_scores["dms_id"].unique():
        df_dms = df_ml_scores[df_ml_scores["dms_id"] == dms_id]
        save_df_to_prism(df_dms, run_name, dms_id)

    # Compute metrics
    if get_mean_metrics == True:
        corrs = []
        for dms_id in df_ml_scores["dms_id"].unique():
            corr = get_prism_corr(dms_id, run_name)
            corrs.append(corr)
        return mean(corrs), acc_mean
    else:
        corrs_ssemb = []
        corrs_gemme = []
        corrs_rosetta = []
        for dms_id in df_ml_scores["dms_id"].unique():
            corrs = get_prism_corr_all(dms_id, run_name)
            corrs_ssemb.append(corrs[0])
            corrs_gemme.append(corrs[1])
            corrs_rosetta.append(corrs[2])
        print(f"SSEmb: Mean MAVE spearman correlation: {mean(corrs_ssemb):.3f}")
        print(f"GEMME: Mean MAVE spearman correlation: {mean(corrs_gemme):.3f}")
        print(f"Rosetta: Mean MAVE spearman correlation: {mean(corrs_rosetta):.3f}")
