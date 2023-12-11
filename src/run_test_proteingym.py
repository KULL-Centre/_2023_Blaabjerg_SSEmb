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
from helpers import (
    read_msa,
    loop_pred,
)
from visualization import plot_proteingym
import pdb_parser_scripts.parse_pdbs as parse_pdbs
import torch.utils.data
from collections import OrderedDict
from ast import literal_eval
import subprocess


def test(run_name, epoch, num_ensemble=1, device=None):
    # Download raw data
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/ProteinGym_reference_file_substitutions.csv",
            "-P",
            "../data/test/proteingym/",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/Detailed_performance_files/Substitutions/Spearman/all_models_substitutions_Spearman_DMS_level.csv",
            "-P",
            "../data/test/proteingym/",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/Detailed_performance_files/Substitutions/Spearman/all_models_substitutions_Spearman_Uniprot_level.csv",
            "-P",
            "../data/test/proteingym/",
        ]
    )
    subprocess.run(
        [
            "wget",
            "https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip",
        ]
    )
    subprocess.run(
        ["unzip", "ProteinGym_substitutions.zip", "-d", "../data/test/proteingym/raw"]
    )
    subprocess.run(["rm", "ProteinGym_substitutions.zip"])

    ###
    # TO DO: Add code for generating AF2 structures
    ###

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

    # Save DMS data
    df_dms.to_csv("../data/test/proteingym/exp/dms.csv", index=False)

    ## Pre-process PDBs
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

    # Compute MSAs
    sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    subprocess.run(
        [
            "colabfold_search",
            f"{pdb_dir}/seqs.fasta",
            "/projects/prism/people/skr526/databases",
            "../data/test/proteingym/msa/",
        ]
    )
    subprocess.run(["python", "merge_and_sort_msa.py", "../data/test/proteingym/msa"])

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

    data = [x for x in data_all if x["name"] not in val_list]

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

    # Convert to graph data sets
    testset = models.gvp.data.ProteinGraphData(data)
    letter_to_num = testset.letter_to_num

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
            device=device,
        )

    # Transform results into df
    df_ml = pd.DataFrame(pred_list, columns=["dms_id", "variant_pos", "score_ml_pos"])

    # Save
    df_ml.to_csv(f"../output/proteingym/df_ml_{run_name}.csv", index=False)

    # Load
    df_ml = pd.read_csv(
        f"../output/proteingym/df_ml_{run_name}.csv",
        converters=dict(score_ml_pos=literal_eval),
    )

    # Compute score_ml from nlls
    df_dms = df_dms[~df_dms["dms_id"].isin(val_list)]
    dms_variant_list = df_dms.values.tolist()
    for i, row in enumerate(dms_variant_list):
        dms_id = row[0]
        print(
            f"Computing score for assay {dms_id} variant: {i+1}/{len(dms_variant_list)}"
        )
        variant_set = row[1].split(":")
        score_ml = 0.0

        for variant in variant_set:
            wt = letter_to_num[variant[0]]
            pos = int(re.findall(r"\d+", variant)[0])
            mt = letter_to_num[variant[-1]]
            score_ml_pos = df_ml[
                (df_ml["dms_id"] == dms_id) & (df_ml["variant_pos"] == pos)
            ]["score_ml_pos"].values[0]
            score_ml += float(score_ml_pos[mt])
        dms_variant_list[i].append(score_ml)
    df_total = pd.DataFrame(
        dms_variant_list, columns=["dms_id", "variant_set", "score_dms", "score_ml"]
    )

    # Save
    df_total.to_csv(f"../output/proteingym/df_total_{run_name}.csv", index=False)

    # Load
    df_total = pd.read_csv(f"../output/proteingym/df_total_{run_name}.csv")

    # Compute correlations
    plot_proteingym(df_total, run_name, benchmark_dms_exclude_list=val_list)
