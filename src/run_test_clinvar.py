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
import numpy as np

torch.multiprocessing.set_sharing_strategy("file_system")
from models.msa_transformer.model import MSATransformer
from models.gvp.models import SSEmbGNN
from helpers import (
    read_msa,
    loop_pred,
    compute_auc_group,
)
from visualization import plot_aucroc
import torch.utils.data
from collections import OrderedDict
import shutil
import pdb_parser_scripts.parse_pdbs as parse_pdbs
from collections import OrderedDict
from ast import literal_eval
import subprocess
from sklearn.metrics import auc, roc_curve, roc_auc_score

def test(run_name, epoch, num_ensemble=5, msa_row_attn_mask=True, device=None):
    # Download raw data
    subprocess.run(["wget","https://marks.hms.harvard.edu/proteingym/clinical_ProteinGym_substitutions.zip"])
    subprocess.run(["unzip","clinical_ProteinGym_substitutions.zip","-d","../data/test/clinvar/raw"])
    subprocess.run(["rm","clinical_ProteinGym_substitutions.zip"])

    # Load ClinVar data
    print("Processing ClinVar data")
    clinvar_files = glob.glob("../data/test/clinvar/raw/*.csv")
    dfs = []
    for f in clinvar_files:
        df = pd.read_csv(f)
        dfs.append(df)
    df_clinvar = pd.concat(dfs, ignore_index=True)
    df_clinvar = df_clinvar[["protein","protein_sequence","mutant","DMS_bin_score"]]
    df_clinvar = df_clinvar.rename(columns={"protein":"prot_name",
                                            "protein_sequence":"seq",
                                            "mutant":"variant",
                                            "DMS_bin_score":"label",
                                            }
                                            )

    # Save seqs to fasta
    df_uniqueseqs = df[["prot_name","seq"]].drop_duplicates()
    fh = open(f"../data/test/clinvar/seqs.fasta","w")
    for index, row in df_uniqueseqs.iterrows():
        fh.write(f">{row['prot_name']}\n")
        fh.write(f"{row['seq']}")
        fh.write("\n")
    fh.close()

    ## Pre-process PDBs
    pdb_dir = "../data/test/clinvar/structure/"
    subprocess.run(
        [
           "pdb_parser_scripts/clean_pdbs.sh",
           str(pdb_dir),
        ]
    )
    parse_pdbs.parse(pdb_dir)

    # Load structure data
    with open(f"../data/test/clinvar/structure/coords.json") as json_file:
        data = json.load(json_file)
    json_file.close()

    # Compute MSAs
    sys.path += ["/projects/prism/people/skr526/mmseqs/bin"]
    subprocess.run("source activate mmseqs2 && colabfold_search ../data/test/clinvar/seqs.fasta /projects/prism/people/skr526/databases ../data/test/clinvar/msa/ && source activate struc-seq", shell=True)

    subprocess.run(
        [
           "colabfold_search",
           "../data/test/clinvar/seqs.fasta",
           "/projects/prism/people/skr526/databases",
           "../data/test/clinvar/msa/",
        ]
    )
    subprocess.run(["python", "merge_and_sort_msas.py", "../data/test/clinvar/msa"])

    # Load MSA data
    msa_filenames = sorted(glob.glob(f"../data/test/clinvar/msa/*.a3m"))
    mave_msa_sub = {}
    for i, f in enumerate(msa_filenames):
        name = ".".join(f.split("/")[-1].split(".")[:-1])
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
    for prot_name in df_clinvar["prot_name"].unique():
        df_clinvar_prot = df_clinvar[df_clinvar["prot_name"] == prot_name]
        variant_wtpos_list = [
            [x[:-1] for x in x.split(":")] for x in df_clinvar_prot["variant"].tolist()
        ]
        variant_wtpos_list = list(
            OrderedDict.fromkeys(
                [item for sublist in variant_wtpos_list for item in sublist]
            )
        )  # Remove duplicates
        variant_pos_dict[prot_name.split(".")[0]] = variant_wtpos_list    
    
    # Convert to graph data sets
    testset = models.gvp.data.ProteinGraphData(data)
    letter_to_num = testset.letter_to_num

    # Make variant pos dict
    variant_pos_dict = {}
    for prot_name in df_clinvar["prot_name"].unique():
        df_clinvar_prot = df_clinvar[df_clinvar["prot_name"] == prot_name]
        variant_wtpos_list = [
            [x[:-1] for x in x.split(":")] for x in df_clinvar_prot["variant"].tolist()
        ]
        variant_wtpos_list = list(
            OrderedDict.fromkeys(
                [item for sublist in variant_wtpos_list for item in sublist]
            )
        )  # Remove duplicates
        variant_pos_dict[prot_name.split(".")[0]] = variant_wtpos_list

    # Save data and dict of variant positions
    with open(f"../data/test/clinvar/data_with_msas.pkl","wb") as fp:
        pickle.dump(data, fp)

    with open(f"../data/test/clinvar/variant_pos_dict.pkl","wb") as fp:
        pickle.dump(variant_pos_dict, fp)

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
            msa_row_attn_mask=msa_row_attn_mask,
            device=device,
        )

    # Transform results into df
    df_ml = pd.DataFrame(pred_list, columns=["prot_name", "variant_pos", "score_ml_pos"])

    # Save
    df_ml.to_csv(f"../output/clinvar/df_ml_{run_name}.csv", index=False)

    # Load
    df_ml = pd.read_csv(
        f"../output/clinvar/df_ml_{run_name}.csv",
        converters=dict(score_ml_pos=literal_eval),
    )

    # Compute score_ml from nlls
    clinvar_variant_list = df_clinvar.values.tolist()
    for i, row in enumerate(clinvar_variant_list):
        prot_name = row[0].split(".")[0]
        print(
            f"Computing score for assay {prot_name} variant: {i+1}/{len(clinvar_variant_list)}"
        )
        variant_set = row[2].split(":")
        score_ml = 0.0
        
        for variant in variant_set:
            wt = letter_to_num[variant[0]]
            pos = int(re.findall(r"\d+", variant)[0])
            mt = letter_to_num[variant[-1]]
            score_ml_pos = df_ml[
                (df_ml["prot_name"] == prot_name) & (df_ml["variant_pos"] == pos)
            ]["score_ml_pos"].values[0]
            score_ml += float(score_ml_pos[mt])
        clinvar_variant_list[i].append(score_ml)
    df_total = pd.DataFrame(
        clinvar_variant_list, columns=["prot_name", "seq", "variant_set", "label", "score_ml"]
    )
    df_total = df_total[["prot_name", "variant_set", "label", "score_ml"]]

    # Save
    df_total.to_csv(f"../output/clinvar/df_total_{run_name}.csv", index=False)

    # Load
    df_total = pd.read_csv(f"../output/clinvar/df_total_{run_name}.csv")
    
    # Compute AUC
    df_total["label_bin"] = [1 if "path" in x else 0 for x in df_total['label'].str.lower()]
    df_total["score"] = -df_total["score_ml"]
    prot_level_auc = df_total.groupby('prot_name').apply(compute_auc_group)
    auc_mean = prot_level_auc.mean(skipna=True)
    print(f"SSEmb avg. AUC is: {auc_mean}")
