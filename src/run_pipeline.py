import os
import sys
import subprocess
import torch
import models.gvp.data, models.gvp.models
import json
import os
import numpy as np
import torch_geometric
from functools import partial
import esm
import random

print = partial(print, flush=True)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from models.msa_transformer.model import MSATransformer
from models.gvp.models import SSEmbGNN
import pickle
from visualization import (
    plot_mave_corr_vs_depth,
)
from helpers import (
    read_msa,
    loop_trainval,
)
import run_test_mave, run_test_proteingym, run_test_rocklin, run_pipeline_scannet
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

DEVICES = [3, 4, 5, 6, 7, 8, 9]
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(x) for x in DEVICES])


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1111"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(
    dataset,
    rank,
    world_size,
    batch_size=1,
    pin_memory=False,
    num_workers=0,
    train=False,
):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    if train == True:
        dataloader = torch_geometric.loader.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            sampler=sampler,
        )
    else:
        dataloader = torch_geometric.loader.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            sampler=None,
        )
    return dataloader


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    # Setup the process groups
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    setup(rank, world_size)

    # Set fixed seed
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # Print name of run
    run_name = "final_cath"

    # Set initial parameters
    EPOCHS = 200
    EPOCH_FINETUNE_MSA = 100
    VAL_INTERVAL = 10
    BATCH_PROTS = 128 // len(DEVICES)
    LR_LIST = [1e-3, 1e-6]
    PATIENCE = 3

    ## Load CATH data
    print("Preparing CATH data")
    pdb_dir_cath = "../data/train/cath"
    subprocess.call([f"{pdb_dir_cath}/getCATH.sh"])
    cath = models.gvp.data.CATHDataset(
        path=f"{pdb_dir_cath}/chain_set.json",
        splits_path=f"{pdb_dir_cath}/chain_set_splits.json",
    )

    # Compute MSAs
    # TO DO: Add code example to extract sequences from CATH data set
    # to file: f"{pdb_dir_cath}/seqs.fasta"
    sys.path += [":/projects/prism/people/skr526/mmseqs/bin"]
    subprocess.run(
        [
            "colabfold_search",
            f"{pdb_dir_cath}/seqs.fasta",
            "/projects/prism/people/skr526/databases",
            "../data/train/cath/msa/",
        ]
    )
    subprocess.run(["python", "merge_and_sort_msa.py", "../data/train/cath/msa"])

    # Add MSAs
    for i, entry in enumerate(cath.total):
        print(f"Adding CATH MSAs: {i+1}/{len(cath.total)}")
        entry["msa"] = read_msa(f"{pdb_dir_cath}/msa/{entry['name']}.a3m")

    # Checkpoint - save and load
    with open(f"{pdb_dir_cath}/data_with_msas.pkl", "wb") as fp:  # Pickling
        pickle.dump(cath.total, fp)

    with open(f"{pdb_dir_cath}/data_with_msas.pkl", "rb") as fp:  # Unpickling
        cath.total = pickle.load(fp)

    ## Filter data
    # Only keep entries where MSA and structucture sequence lengths match
    data = [
        entry for entry in cath.total if len(entry["seq"]) == len(entry["msa"][0][1])
    ]

    # Filter: Only keep entries without X in sequence
    [entry for entry in cath.total if "X" not in entry["seq"]]

    # Save all training and validation sequences in a fasta file to check homology
    cath.split()
    with open(f"../data/test/mave_val/structure/coords.json") as json_file:
        data_mave_val = json.load(json_file)
    json_file.close()

    with open(f"../data/test/proteingym/structure/coords.json") as json_file:
        data_proteingym = json.load(json_file)
    json_file.close()

    fh = open(f"../data/train/cath/seqs_cath.fasta", "w")
    for entry in cath.train:
        fh.write(f">{entry['name']}\n")
        fh.write(f"{entry['seq']}\n")

    for entry in cath.val:
        fh.write(f">{entry['name']}\n")
        fh.write(f"{entry['seq']}\n")

    for entry in data_mave_val:
        fh.write(f">{entry['name']}\n")
        fh.write(f"{entry['seq']}\n")

    for entry in data_proteingym:
        fh.write(f">{entry['name']}\n")
        fh.write(f"{entry['seq']}\n")
    fh.close()

    # Compute clusters of 95% sequence similarities between all training, validation and test proteins
    subprocess.run(
        [
            "cd-hit",
            "-i",
            "../data/train/cath/seqs_cath.fasta",
            "-o",
            "../data/train/cath/seqs_cath_homology.fasta",
            "-c",
            "0.95",
            "-n",
            "5",
            "-d",
            "999",
        ]
    )

    # Remove proteins from training data that has high sequence similarity with validation or test proteins
    val_prot_names = [entry["name"] for entry in cath.val]
    val_mave_prot_names = [entry["name"] for entry in data_mave_val]
    test_prot_names = [entry["name"] for entry in data_proteingym]
    valtest_prot_names = val_prot_names + val_mave_prot_names + test_prot_names

    fh = open("../data/train/cath/seqs_cath_homology.fasta.clstr", "r")
    cluster_dict = {}
    remove_list = []
    for line in fh.readlines():
        if line.startswith(">Cluster"):
            cluster_name = line
            cluster_dict[cluster_name] = []
        else:
            cluster_dict[cluster_name].append(line.split(">")[1].split("...")[0])

    for cluster_name, prot_names in cluster_dict.items():
        if len(prot_names) > 1 and any(
            valtest_prot_name in prot_names for valtest_prot_name in valtest_prot_names
        ):
            remove_list += prot_names
    remove_list = [
        prot_name for prot_name in remove_list if prot_name not in valtest_prot_names
    ]
    cath.train = [entry for entry in cath.train if entry["name"] not in remove_list]

    # Checkpoint - save and load
    with open(
        f"{pdb_dir_cath}/data_with_msas_filtered_train.pkl", "wb"
    ) as fp:  # Pickling
        pickle.dump(cath.train, fp)
    with open(
        f"{pdb_dir_cath}/data_with_msas_filtered_val.pkl", "wb"
    ) as fp:  # Pickling
        pickle.dump(cath.val, fp)

    with open(
        f"{pdb_dir_cath}/data_with_msas_filtered_train.pkl", "rb"
    ) as fp:  # Unpickling
        cath.train = pickle.load(fp)
    with open(
        f"{pdb_dir_cath}/data_with_msas_filtered_val.pkl", "rb"
    ) as fp:  # Unpickling
        cath.val = pickle.load(fp)

    # Convert to graph data sets
    trainset = models.gvp.data.ProteinGraphData(cath.train)
    valset = models.gvp.data.ProteinGraphData(cath.val)

    # Load and initialize MSA Transformer
    model_msa_pre, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    torch.save(
        model_msa_pre.state_dict(),
        f"../output/train/models/msa_transformer/pretrained.pt",
    )
    msa_batch_converter = msa_alphabet.get_batch_converter()
    model_msa = MSATransformer(msa_alphabet)
    model_msa.load_state_dict(
        torch.load(f"../output/train/models/msa_transformer/pretrained.pt")
    )
    model_msa.to(rank)

    # Freeze MSA Transformer
    for param in model_msa.parameters():
        param.requires_grad = False

    # Load and initialize Flamingo GVP
    node_dim = (256, 64)
    edge_dim = (32, 1)
    model_gvp = SSEmbGNN((6, 3), node_dim, (32, 1), edge_dim)
    model_gvp.to(rank)
    model_gvp = DDP(
        model_gvp,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
        static_graph=False,
    )

    ## Initialize training modules
    train_loader = prepare(trainset, rank, world_size, train=True)
    val_loader = prepare(valset, rank, world_size)
    optimizer = torch.optim.Adam(model_gvp.parameters(), lr=LR_LIST[0])
    scaler = torch.cuda.amp.GradScaler()
    best_epoch, best_corr_mave = None, 0
    patience_counter = 0

    ## Initialize lists for monitoring loss
    epoch_list = []
    loss_train_list, loss_val_list = [], []
    acc_train_list, acc_val_list = [], []
    corr_mave_list, acc_mave_list = [], []

    for epoch in range(EPOCHS):
        # Check if we should fine-tune MSA Transformer row attention
        if epoch == EPOCH_FINETUNE_MSA:
            for param in model_msa.named_parameters():
                if "row_self_attention" in param[0]:
                    param[1].requires_grad = True
            model_msa = DDP(
                model_msa,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
                static_graph=False,
            )
            optimizer.add_param_group(
                {"params": model_msa.parameters(), "lr": LR_LIST[1]}
            )
            BATCH_PROTS = 2048 // len(DEVICES)

        # If we are using DistributedSampler, we need to tell it which epoch this is
        train_loader.sampler.set_epoch(epoch)

        # Train loop
        loss_train, acc_train = loop_trainval(
            model_msa,
            model_gvp,
            msa_batch_converter,
            train_loader,
            BATCH_PROTS,
            epoch,
            rank,
            EPOCH_FINETUNE_MSA,
            optimizer=optimizer,
            scaler=scaler,
        )
        ## Gather and save training metrics for epoch
        # OBS: This cannot be placed within validation loop or we get hangs
        loss_train = loss_train.type(torch.float32)
        loss_train_all_gather = [torch.zeros(1, device=rank) for _ in range(world_size)]
        dist.all_gather(loss_train_all_gather, loss_train)

        # Validation loop
        if rank == 0:
            # Save model
            path_msa = f"../output/train/models/msa_transformer/{run_name}_msa_transformer_{epoch}.pt"
            path_gvp = f"../output/train/models/gvp/{run_name}_gvp_{epoch}.pt"
            path_optimizer = (
                f"../output/train/models/optimizer/{run_name}_adam_{epoch}.pt"
            )
            torch.save(model_msa.state_dict(), path_msa)
            torch.save(model_gvp.state_dict(), path_gvp)
            torch.save(optimizer.state_dict(), path_optimizer)

            # Compute validation
            if epoch % VAL_INTERVAL == 0:
                with torch.no_grad():
                    # Val loop
                    loss_val, acc_val = loop_trainval(
                        model_msa,
                        model_gvp,
                        msa_batch_converter,
                        val_loader,
                        BATCH_PROTS,
                        epoch,
                        rank,
                        EPOCH_FINETUNE_MSA,
                    )

                    # Do validation on MAVE set
                    corr_mave, acc_mave = run_test_mave.test(
                        run_name,
                        epoch,
                        num_ensemble=1,
                        get_mean_metrics=True,
                        device=rank,
                    )

                    if corr_mave > best_corr_mave:
                        best_epoch = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Save validation results
                    epoch_list.append(epoch)
                    loss_train_list.append(
                        torch.mean(torch.stack(loss_train_all_gather)).to("cpu").item()
                    )
                    loss_val_list.append(loss_val.to("cpu").item())
                    acc_val_list.append(acc_val)
                    corr_mave_list.append(corr_mave)
                    acc_mave_list.append(corr_mave)

                    metrics = {
                        "epoch": epoch_list,
                        "loss_train": loss_train_list,
                        "loss_val": loss_val_list,
                        "acc_val": acc_val_list,
                        "corr_mave": corr_mave_list,
                        "acc_mave": acc_mave_list,
                    }
                    with open(f"../output/train/metrics/{run_name}_metrics", "wb") as f:
                        pickle.dump(metrics, f)

                    if patience_counter == PATIENCE:
                        break

        # Create barrier after each epoch
        dist.barrier()

    # Clean up
    cleanup()

    ## Make test evalutions
    #best_epoch = 110 # OBS: Uncomment this line to use weights from paper

    # MAVE val set
    print("Starting MAVE val predictions")
    run_test_mave.test(run_name, best_epoch, num_ensemble=5, device=rank)
    plot_mave_corr_vs_depth()
    print("Finished MAVE val predictions")

    # ProteinGym test set
    print("Starting ProteinGym test")
    run_test_proteingym.test(run_name, best_epoch, num_ensemble=5, device=rank)
    print("Finished ProteinGym test")

    # Rocklin test set
    print("Starting Rocklin test")
    run_test_rocklin.test(run_name, best_epoch, num_ensemble=5, device=rank)
    print("Finished Rocklin test")

    # ScanNet test set
    print("Starting ScanNet test")
    run_pipeline_scannet.run(run_name, best_epoch, device=rank)
    print("Finished ScanNet test")


if __name__ == "__main__":
    world_size = len(DEVICES)
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )
