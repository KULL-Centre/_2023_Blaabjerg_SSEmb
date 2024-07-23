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
import pickle

def test(run_name, epoch, msa_row_attn_mask=True, device=None):
    # Load data and dict of variant positions
    with open(f"../data/test/proteingym/data_with_msas.pkl", "rb") as fp:
        data = pickle.load(fp)

    with open(f"../data/test/proteingym/variant_pos_dict.pkl", "rb") as fp:
        variant_pos_dict = pickle.load(fp)

    # Load DMS data
    df_dms = pd.read_csv("../data/test/proteingym/exp/dms.csv")

    # Convert to graph data sets
    testset = models.gvp.data.ProteinGraphData(data)
    letter_to_num = testset.letter_to_num

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
            msa_row_attn_mask=msa_row_attn_mask,
            device=device,
        )

    # Transform results into df
    df_ml = pd.DataFrame(pred_list, columns=["dms_id", "variant_pos", "score_ml_pos"])

    # Save
    df_ml.to_csv(f"../output/proteingym/df_ml_{run_name}_{epoch}.csv", index=False)

    # Load
    df_ml = pd.read_csv(
        f"../output/proteingym/df_ml_{run_name}_{epoch}.csv",
        converters=dict(score_ml_pos=literal_eval),
    )

    # Compute score_ml from nlls
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
    df_total.to_csv(f"../output/proteingym/df_total_{run_name}_{epoch}.csv", index=False)

    # Load
    df_total = pd.read_csv(f"../output/proteingym/df_total_{run_name}_{epoch}.csv")

    # Compute correlations
    plot_proteingym(df_total, run_name, epoch)
