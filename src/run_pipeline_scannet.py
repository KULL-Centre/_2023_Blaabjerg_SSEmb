import os
import sys
import subprocess
import re
import torch
import models.gvp.data, models.gvp.models
import json
import numpy as np
import torch_geometric
import esm
import pandas as pd
import random
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from models.msa_transformer.model import MSATransformer
from models.gvp.models import SSEmbGNN
from models.scannet.model import TransformerModel
import pickle
from helpers import (
    read_msa,
    loop_getemb,
    scannet_collate_fn,
    loop_scannet_trainval,
    loop_scannet_test,
)
from visualization import plot_precision_recall
import pdb_parser_scripts.parse_pdbs as parse_pdbs
import torch.utils.data
from sklearn.metrics import auc
from collections import OrderedDict
from Bio.PDB import PDBList
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
import io
from contextlib import redirect_stdout
import pymol2
import time


def run(run_name, epoch, device=None):
    # Download raw data
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/table.csv",
            "-P",
            "../data/test/scannet/",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/labels_train.txt",
            "-P",
            "../data/test/scannet/labels",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/labels_validation_70.txt",
            "-P",
            "../data/test/scannet/labels",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/labels_validation_homology.txt",
            "-P",
            "../data/test/scannet/labels",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/labels_validation_topology.txt",
            "-P",
            "../data/test/scannet/labels",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/labels_validation_none.txt",
            "-P",
            "../data/test/scannet/labels",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/labels_test_70.txt",
            "-P",
            "../data/test/scannet/labels",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/labels_test_homology.txt",
            "-P",
            "../data/test/scannet/labels",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/labels_test_topology.txt",
            "-P",
            "../data/test/scannet/labels",
        ]
    )
    subprocess.run(
        [
            "wget",
            "-c",
            "https://raw.githubusercontent.com/jertubiana/ScanNet/main/datasets/PPBS/labels_test_none.txt",
            "-P",
            "../data/test/scannet/labels",
        ]
    )

    # Download PDBs
    parser = PDBParser()
    pdb_io = PDBIO()
    df = pd.read_csv("../data/test/scannet/table.csv")
    pdb_list_all = df["PDB ID"].unique()
    pdb_list = [x[:4] for x in pdb_list_all]
    chain_list = [x[-1] for x in pdb_list_all]

    pdbl = PDBList()
    f = io.StringIO()
    for i, pdbid in enumerate(pdb_list):
        print(f"{i+1}/{len(pdb_list)}")
        print(pdbid)
        out = ""
        if os.path.exists(f"../data/test/scannet/raw/{pdbid}.pdb"):
            print("PDB file already downloaded")
        else:
            with redirect_stdout(f):
                pdbl.retrieve_pdb_file(
                    pdbid, pdir="../data/test/scannet/raw", file_format="pdb"
                )
            out = f.getvalue()
            if "Desired structure doesn't exists" in out:
                try:
                    pdbl.retrieve_pdb_file(
                        pdbid, pdir="../data/test/scannet/raw", file_format="mmCif"
                    )
                    with pymol2.PyMOL() as pymol:
                        pymol.cmd.load(
                            f"../data/test/scannet/raw/{pdbid}.cif", "my_protein"
                        )
                        pymol.cmd.save(
                            f"../data/test/scannet/raw/{pdbid}.cif".replace(
                                ".cif", ".pdb"
                            ),
                            selection="my_protein",
                        )
                except:
                    print("Protein does not exist as either PDB or mmCIF file")
            else:
                subprocess.run(
                    [
                        "mv",
                        f"../data/test/scannet/raw/pdb{pdbid}.ent",
                        f"../data/test/scannet/raw/{pdbid}.pdb",
                    ]
                )
        time.sleep(1)
    subprocess.run(["rm", "-r", "obsolete"])

    # Split into chains
    f = open("../data/test/scannet/scannet_download.log", "w")
    for i, pdbid in enumerate(pdb_list):
        try:
            structure = parser.get_structure(
                pdbid, f"../data/test/scannet/raw/{pdbid}.pdb"
            )
            pdb_chains = structure.get_chains()

            for chain in pdb_chains:
                if chain.get_id() == chain_list[i]:
                    pdb_io.set_structure(chain)
                    pdb_io.save(
                        f"../data/test/scannet/structure/raw/{pdbid}_{chain_list[i]}.pdb"
                    )
        except:
            f.write(f"{pdbid}_{chain_list[i]} not available\n")
    f.close()

    # Pre-process PDBs
    pdb_dir = "../data/test/scannet/structure"
    subprocess.call(
        [
            f"pdb_parser_scripts/clean_pdbs.sh",
            str(pdb_dir),
        ]
    )
    parse_pdbs.parse(pdb_dir)

    # Compute MSAs
    sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    subprocess.run(
        [
            "colabfold_search",
            f"{pdb_dir}/seqs.fasta",
            "/projects/prism/people/skr526/databases",
            "../data/test/scannet/msa/",
        ]
    )
    subprocess.run(["python", "merge_and_sort_msas.py", "../data/test/scannet/msa"])

    # Load structure data
    with open(f"../data/test/scannet/structure/coords.json") as json_file:
        data_raw = json.load(json_file)
    json_file.close()

    # Only keep entries with sequence lengths <= 1024
    data = []
    for entry in data_raw:
        if len(entry["seq"]) <= 1024 - 1:  # Consider added <CLS> token
            data.append(entry)

    # Add MSAs to data
    for i, entry in enumerate(data):
        print(f"{i+1}/{len(data)}")
        msa = read_msa(f'../data/test/scannet/msa/{entry["name"]}.a3m')
        msa_sub = [msa[0]]
        k = min(len(msa) - 1, 256 - 1)
        msa_sub += [msa[i] for i in sorted(random.sample(range(1, len(msa)), k))]
        entry["msa"] = msa_sub

    with open("../output/scannet/data_scannet.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("../output/scannet/data_scannet.pickle", "rb") as handle:
        data = pickle.load(handle)

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

    # Convert to graph data sets
    allset = models.gvp.data.ProteinGraphData(data)

    # Init data loader
    data_loader = torch_geometric.loader.DataLoader(allset, batch_size=1, shuffle=False)

    # Add frozen embeddings to data
    with torch.no_grad():
        emb_dict = loop_getemb(
            model_msa,
            model_gvp,
            msa_batch_converter,
            data_loader,
            device=device,
        )
    for entry in data:
        entry["emb"] = emb_dict[entry["name"]]

    with open("../output/scannet/data_scannet_emb.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("../output/scannet/data_scannet_emb.pickle", "rb") as handle:
        data = pickle.load(handle)

    # Add sample weights to data
    df = pd.read_csv(f"../data/test/scannet/table.csv")
    for entry in data:
        if entry["name"] in df["PDB ID"].unique():
            entry["prot_weight"] = df[df["PDB ID"] == entry["name"]][
                "Sample weight"
            ].item()

    # Concat label files
    filenames = [
        f"../data/test/scannet/labels/labels_train.txt",
        f"../data/test/scannet/labels/labels_validation_70.txt",
        f"../data/test/scannet/labels/labels_validation_homology.txt",
        f"../data/test/scannet/labels/labels_validation_topology.txt",
        f"../data/test/scannet/labels/labels_validation_none.txt",
        f"../data/test/scannet/labels/labels_test_70.txt",
        f"../data/test/scannet/labels/labels_test_homology.txt",
        f"../data/test/scannet/labels/labels_test_topology.txt",
        f"../data/test/scannet/labels/labels_test_none.txt",
    ]
    with open(f"../data/test/scannet/labels/labels_all.txt", "w") as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    # Add labels to data
    f = open(f"../data/test/scannet/labels/labels_all.txt", "r")
    label_dict = {}
    prot = None
    for line in f.readlines():
        if line.startswith(">"):
            prot = line[1:].strip()[:4] + "_" + line[1:].strip()[-1]
            label_dict[prot] = []
        else:
            label_dict[prot].append(
                line.strip().split(" ")
            )  # [chain_id, pos, aa, label]
    f.close()

    # Add labels to data
    i = 0
    j = 0
    data_clean = []
    for entry in data:
        if entry["name"] in label_dict.keys():
            label_seq = "".join([x[2] for x in label_dict[entry["name"]]])

            if label_seq == entry["seq"]:
                entry["label"] = torch.tensor(
                    [int(x[3]) for x in label_dict[entry["name"]]], device=device
                )
            else:
                entry["label"] = None
                j += 1
        else:
            entry["label"] = None
            i += 1

        # If no errors; add label data to entry
        if entry["label"] is not None:
            data_clean.append(entry)

    print(
        f"Number of PDBs where we have structure data but no label data: {i}/{len(data)}"
    )
    print(
        f"Number of PDBs where the label seq and the structure seq doesn't match: {j}/{len(data)}"
    )
    print(f"Number of PDBs in cleaned data set: {len(data_clean)}")

    # Set parameters for downstream model learning
    EPOCHS = 40
    VAL_INTERVAL = 1
    BATCH_PROTS = 10
    LR = 1e-4

    # Split intro train/val/test
    df = pd.read_csv(f"../data/test/scannet/table.csv")
    data_train = [
        x for x in data_clean if x["name"] in list(df[df["Set"] == "Train"]["PDB ID"])
    ]
    data_val = [
        x
        for x in data_clean
        if x["name"] in list(df[df["Set"] == "Validation (70\%)"]["PDB ID"])
    ]

    # Load Transformer model
    model_transformer = TransformerModel(
        ntoken=1, nhead=4, d_hid=256, nlayers=4, dropout=0.1
    )
    model_transformer = model_transformer.to(device)

    # Init optimizer
    optimizer = torch.optim.Adam(model_transformer.parameters(), lr=LR)

    # Initialize data loader
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=1, collate_fn=scannet_collate_fn, shuffle=True
    )  # OBS: Use grad accumulation if bs > 1
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=1, collate_fn=scannet_collate_fn, shuffle=False
    )

    # Initialize lists for monitoring loss
    epoch_list = []
    loss_train_list, loss_val_list = [], []
    acc_train_list, acc_val_list = [], []
    corr_mave_list, acc_mave_list = [], []
    best_epoch, best_loss_val = None, np.inf

    # Begin training and validation loop
    for epoch in range(EPOCHS):
        # Train loop
        model_transformer.train()
        loss_train, acc_train = loop_scannet_trainval(
            model_transformer,
            train_loader,
            device=device,
            optimizer=optimizer,
            batch_prots=BATCH_PROTS,
        )

        # Save model
        path_transformer = (
            f"../output/scannet/models/transformer/{run_name}_transformer_{epoch}.pt"
        )
        path_optimizer = (
            f"../output/scannet/models/optimizer/{run_name}_adam_{epoch}.pt"
        )
        torch.save(model_transformer.state_dict(), path_transformer)
        torch.save(optimizer.state_dict(), path_optimizer)

        # Compute validation
        if epoch % VAL_INTERVAL == 0:
            # Validation
            with torch.no_grad():
                model_transformer.eval()
                loss_val, acc_val = loop_scannet_trainval(
                    model_transformer,
                    val_loader,
                    device=device,
                )

                if loss_val < best_loss_val:
                    best_epoch, best_loss_val = epoch, loss_val

                # Save validation results
                epoch_list.append(epoch)
                loss_train_list.append(loss_train.to("cpu").item())
                loss_val_list.append(loss_val.to("cpu").item())
                acc_val_list.append(acc_val)

                metrics = {
                    "epoch": epoch_list,
                    "loss_train": loss_train_list,
                    "loss_val": loss_val_list,
                    "acc_val": acc_val_list,
                }
                with open(f"../output/scannet/metrics/{run_name}_metrics", "wb") as f:
                    pickle.dump(metrics, f)

    # Test
    #best_epoch = 26 # OBS: Uncomment this line to use weights from paper
    print(f"Testing model! Using best model from epoch: {best_epoch}")
    model_transformer.load_state_dict(
        torch.load(
            f"../output/scannet/models/transformer/{run_name}_transformer_{best_epoch}.pt"
        )
    )

    test_sets = [
        ["Test (70\%)"],
        ["Test (Homology)"],
        ["Test (Topology)"],
        ["Test (None)"],
        ["Test (70\%)", "Test (Homology)", "Test (Topology)", "Test (None)"],
    ]

    precision_list = []
    recall_list = []
    outfile = open(f"../output/scannet/{run_name}_test_results.txt", "w")

    for test_set in test_sets:
        print(f"Computing predictions for: {' & '.join(test_set)}")
        data_test = [
            x
            for x in data_clean
            if x["name"] in list(df[df["Set"].isin(test_set)]["PDB ID"])
        ]
        test_loader = torch.utils.data.DataLoader(
            data_test, batch_size=1, collate_fn=scannet_collate_fn, shuffle=False
        )

        with torch.no_grad():
            model_transformer.eval()
            precision, recall = loop_scannet_test(
                model_transformer,
                test_loader,
                device=device,
            )
            precision_list.append(precision)
            recall_list.append(recall)
            auc_precision_recall = auc(recall, precision)
            outfile.write(
                f"Test AUCPR for {' & '.join(test_set)} is: {auc_precision_recall}\n"
            )

    # Plot test results
    plot_precision_recall(recall_list, precision_list)
    outfile.close()
