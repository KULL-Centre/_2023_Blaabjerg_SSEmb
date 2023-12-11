import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnchoredText

plt.rcParams["figure.dpi"] = 400
plt.rcParams["figure.figsize"] = [9.0, 9.0]
plt.rcParams.update({"font.size": 14})
from scipy import stats
from sklearn.metrics import auc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter

white_viridis = LinearSegmentedColormap.from_list(
    "white_viridis",
    [
        (0, "#ffffff"),
        (1e-20, "#440053"),
        (0.2, "#404388"),
        (0.4, "#2a788e"),
        (0.6, "#21a784"),
        (0.8, "#78d151"),
        (1, "#fde624"),
    ],
    N=256,
)


def plot_precision_recall(recall_list, precision_list):
    # Initialize
    fig, ax = plt.subplots()
    test_sets_names = [
        "Test (70%)",
        "Test (homology)",
        "Test (topology)",
        "Test (none)",
        "Test (all)",
    ]

    for i in range(len(recall_list)):
        recall = recall_list[i]
        precision = precision_list[i]
        test_set_name = test_sets_names[i]

        # Compute AUCPR
        auc_precision_recall = auc(recall, precision)

        # Plot
        ax.plot(
            recall,
            precision,
            label=f"{test_set_name} AUCPR: {auc_precision_recall:.3f}",
        )
        ax.legend(loc="lower left")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        fig.savefig(
            f"../output/scannet/precision_recall_curve.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_mave_corr_vs_depth():
    result_dict_ensemble_1 = {
        256: 0.508334416811593,
        128: 0.514156428396764,
        64: 0.519894150776431,
        32: 0.51009495879155,
        16: 0.503765246987659,
        1: 0.406259878495451,
    }

    result_dict_ensemble_5 = {
        256: 0.511413182842286,
        128: 0.518598306644705,
        64: 0.524968638219744,
        32: 0.530111985984653,
        16: 0.536027260958527,
        1: 0.406270219207491,
    }

    # Plot
    fig, ax = plt.subplots()
    ax.plot(
        list(result_dict_ensemble_1.keys()),
        list(result_dict_ensemble_1.values()),
        label=f"MSA subsampling ensemble size: 1",
        color="blue",
    )
    ax.plot(
        list(result_dict_ensemble_5.keys()),
        list(result_dict_ensemble_5.values()),
        label=f"MSA subsampling ensemble size: 5",
        color="red",
    )
    ax.legend(loc="lower right")
    ax.set_xlabel("# MSA sequences")
    ax.set_ylabel("$\\rho_S$")
    ax.set_ylim(0.40, 0.55)
    fig.savefig(
        f"../output/mave_val/mave_corr_vs_depth.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_learning_curves(metrics, run_name):
    # Plot train/val loss vs val accuracy
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(
        metrics["epoch"],
        metrics["loss_train"],
        label="Train: Cross-entropy loss",
        color="blue",
    )
    lns2 = ax1.plot(
        metrics["epoch"],
        metrics["loss_val"],
        label="Val: Cross-entropy loss",
        color="green",
    )
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_xlabel("Training epochs")
    ax2 = ax1.twinx()
    lns3 = ax2.plot(
        metrics["epoch"], metrics["acc_val"], label="Val: Accuracy", color="red"
    )
    ax2.set_ylabel("Accuracy")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="center right")
    fig.savefig(
        f"../output/plots/{run_name}_loss_vs_acc.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # Plot train/val loss vs mave correlation
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(
        metrics["epoch"],
        metrics["loss_train"],
        label="Train: Cross-entropy loss",
        color="blue",
    )
    lns2 = ax1.plot(
        metrics["epoch"],
        metrics["loss_val"],
        label="Val: Cross-entropy loss",
        color="green",
    )
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_xlabel("Training epochs")
    ax2 = ax1.twinx()
    lns3 = ax2.plot(
        metrics["epoch"],
        [abs(e) for e in metrics["mave"]],
        label="Val: MAVE $\\rho_S$",
        color="red",
    )
    ax2.set_ylabel("MAVE $\\rho_S$")
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="center right")
    ax2.axhline(0.47, color="black", linestyle="dashed")
    fig.savefig(
        f"../output/plots/{run_name}_loss_vs_mave.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_logits(logits, run_name):
    fig, ax = plt.subplots()
    ax.hist(logits, bins=1000, color="blue")
    ax.set_xlabel("Logits [a.u.]")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.savefig(
        f"../output/plots/{run_name}_logits_hist.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_hist(df, prot_name, run_name, testset_name):
    fig, ax = plt.subplots()
    ax.hist(df["score_dms_00"], bins=100, color="blue", alpha=0.9, label="SSEmb")
    ax.hist(df["score_ml_01"], bins=100, color="orange", alpha=0.9, label="MAVE")
    ax.set_xlabel("Score [a.u.]")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.savefig(
        f"../output/{testset_name}/{run_name}_{prot_name}_hist.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_rocklin(df):
    """
    Plots a scatter plot
    """
    # Compute statistics
    x = df["score_exp"]
    y = df["score_ml"]
    spearman_r = stats.spearmanr(x, y)[0]

    # Make plot
    fig = plt.figure()

    # Add data points
    if len(x) > 400:
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        density = ax.scatter_density(x, y, cmap=white_viridis)
        fig.colorbar(density, label="Number of points per pixel")
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y, s=2, c="darkblue")

    # Set labels
    ax.set_xlabel("Experimental \u0394\u0394G [kcal/mol]")
    ax.set_ylabel("SSEmb score [a.u.]")

    ## Set range
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)

    # Add textbox
    text = f"Spearman $\\rho$: {spearman_r:.2f}"
    anchored_text = AnchoredText(text, loc="upper left")
    ax.add_artist(anchored_text)

    # Save
    plt.show()
    fig.savefig(
        f"../output/rocklin/scatter.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_scatter(df, prot_name, run_name, testset_name):
    """
    Plots a scatter plot
    """
    # Compute statistics
    x = df["score_dms_00"]
    y = df["score_ml_01"]
    spearman_r = stats.spearmanr(x, y)[0]

    # Make plot
    fig = plt.figure()

    # Add data points
    if len(x) > 400:
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        density = ax.scatter_density(x, y, cmap=white_viridis)
        fig.colorbar(density, label="Number of points per pixel")
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y, s=2, c="darkblue")

    # Set labels
    ax.set_xlabel("MAVE score [a.u.]")
    ax.set_ylabel("SSEmb score [a.u.]")

    ## Set range
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)

    # Add textbox
    text = f"Spearman $\\rho$: {spearman_r:.2f}"
    anchored_text = AnchoredText(text, loc="upper left")
    ax.add_artist(anchored_text)

    # Save
    plt.show()
    fig.savefig(
        f"../output/{testset_name}/{run_name}_{prot_name}_scatter.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_learning_curve(loss_train, loss_val, acc_train, acc_val, run_name):
    epochs = np.arange(len(loss_train)) + 1
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(epochs, loss_train, label="Train loss", color="blue")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_xlabel("Training epochs")
    # ax1.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2 = ax1.twinx()
    lns2 = ax2.plot(epochs, acc_train, label="Train accuracy", color="green")
    lns3 = ax2.plot(epochs, acc_val, label="Val accuracy", color="orange")
    ax2.set_ylabel("Accuracy")
    # ax2.set_ylim(0.2, 0.5)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="upper left")
    fig.savefig(f"{os.getcwd()}/plots/{filename}.png", bbox_inches="tight")
    plt.close()
    with open(f"{os.getcwd()}/plots/learning_curve_{run_name}.txt", "w") as f:
        for loss_train, loss_val, acc_train, acc_val in zip(
            loss_train, loss_val, acc_train, acc_val
        ):
            f.write(
                "{0},{1},{2},{3}\n".format(loss_train, loss_val, acc_train, acc_val)
            )


def plot_proteingym(df, run_name, benchmark_dms_exclude_list=None):
    # Compute correlations per DMS_id
    df = df.rename(columns={"dms_id": "DMS_id"})
    corr_obj = (
        df.groupby("DMS_id")[["score_ml", "score_dms"]]
        .corr(method="spearman")
        .unstack()
        .iloc[:, 1]
    )
    corrs = {
        list(corr_obj.index)[i]: list(corr_obj.values)[i]
        for i in range(len(list(corr_obj.index)))
    }
    df = pd.DataFrame(corrs.items(), columns=["DMS_id", "SSEmb"])
    df["SSEmb"] = df["SSEmb"].abs()

    # Mean SSEmb correlations over DMS assays for each UniProt_ID
    df_reference = pd.read_csv(
        f"../data/test/proteingym/ProteinGym_reference_file_substitutions.csv"
    )
    df_reference.loc[
        df_reference["DMS_id"] == "P53_HUMAN_Kotler_2018", "UniProt_ID"
    ] = "P53_HUMAN_Kotler"
    df_reference.loc[
        df_reference["DMS_id"] == "B3VI55_LIPST_Klesmith_2015", "UniProt_ID"
    ] = "B3VI55_LIPST"
    df_reference.loc[
        df_reference["DMS_id"] == "RL401_YEAST_Mavor_2016", "UniProt_ID"
    ] = "RL401_YEAST"
    df_reference.loc[
        df_reference["DMS_id"] == "RL401_YEAST_Roscoe_2013", "UniProt_ID"
    ] = "RL401_YEAST"
    df_reference.loc[
        df_reference["DMS_id"] == "RL401_YEAST_Roscoe_2014", "UniProt_ID"
    ] = "RL401_YEAST"
    df = pd.merge(df, df_reference[["DMS_id", "UniProt_ID"]], on="DMS_id")
    df = df.groupby("UniProt_ID", as_index=False)[["SSEmb"]].mean()

    # Load ProteinGym UniProt-level benchmark data
    df_benchmark = pd.read_csv(
        f"../data/test/proteingym/all_models_substitutions_Spearman_DMS_level.csv"
    )
    df_benchmark = df_benchmark.rename(columns={"Unnamed: 0": "DMS_id"})
    df_benchmark = df_benchmark[
        ~df_benchmark["DMS_id"].isin(benchmark_dms_exclude_list)
    ]
    df_benchmark = df_benchmark.groupby(
        ["UniProt_ID", "Neff_L_category"], as_index=False
    )[
        [
            "UniProt_ID",
            "Neff_L_category",
            "MSA_Transformer_ensemble",
            "TranceptEVE_L",
            "GEMME",
            "EVE_ensemble",
            "Tranception_L_retrieval",
            "VESPA",
            "EVE_single",
            "Tranception_M_retrieval",
        ]
    ].mean()
    df = pd.merge(df, df_benchmark, on="UniProt_ID")
    df = df.sort_values(["SSEmb"], ascending=False)

    # Print correlations in MSA depth - low/medium/high - regimes
    df_msalow = df[df["Neff_L_category"] == "low"]
    df_msamedium = df[df["Neff_L_category"] == "medium"]
    df_msahigh = df[df["Neff_L_category"] == "high"]

    print(f"SSEmb - Low MSA depth - Spearman: {df_msalow['SSEmb'].mean():.3f}")
    print(
        f"MSA Transformer - Low MSA depth - Spearman: {df_msalow['MSA_Transformer_ensemble'].mean():.3f}"
    )
    print(
        f"TranceptEVE_L - Low MSA depth - Spearman: {df_msalow['TranceptEVE_L'].mean():.4f}"
    )
    print(f"GEMME - Low MSA depth - Spearman: {df_msalow['GEMME'].mean():.3f}")
    print(
        f"EVE_ensemble - Low MSA depth - Spearman: {df_msalow['EVE_ensemble'].mean():.3f}"
    )
    print(
        f"Tranception_L - Low MSA depth - Spearman: {df_msalow['Tranception_L_retrieval'].mean():.4f}"
    )
    print(f"VESPA - Low MSA depth - Spearman: {df_msalow['VESPA'].mean():.3f}")
    print(
        f"EVE_single - Low MSA depth - Spearman: {df_msalow['EVE_single'].mean():.4f}"
    )
    print(
        f"Tranception_M - Low MSA depth - Spearman: {df_msalow['Tranception_M_retrieval'].mean():.3f}"
    )

    print(f"SSEmb - Medium MSA depth - Spearman: {df_msamedium['SSEmb'].mean():.3f}")
    print(
        f"MSA Transformer - Medium MSA depth - Spearman: {df_msamedium['MSA_Transformer_ensemble'].mean():.3f}"
    )
    print(
        f"TranceptEVE_L - Medium MSA depth - Spearman: {df_msamedium['TranceptEVE_L'].mean():.4f}"
    )
    print(f"GEMME - Medium MSA depth - Spearman: {df_msamedium['GEMME'].mean():.3f}")
    print(
        f"EVE_ensemble - Medium MSA depth - Spearman: {df_msamedium['EVE_ensemble'].mean():.3f}"
    )
    print(
        f"Tranception_L - Medium MSA depth - Spearman: {df_msamedium['Tranception_L_retrieval'].mean():.4f}"
    )
    print(f"VESPA - Medium MSA depth - Spearman: {df_msamedium['VESPA'].mean():.3f}")
    print(
        f"EVE_single - Medium MSA depth - Spearman: {df_msamedium['EVE_single'].mean():.3f}"
    )
    print(
        f"Tranception_M - Medium MSA depth - Spearman: {df_msamedium['Tranception_M_retrieval'].mean():.3f}"
    )

    print(f"SSEmb - High MSA depth - Spearman: {df_msahigh['SSEmb'].mean():.3f}")
    print(
        f"MSA Transformer - High MSA depth - Spearman: {df_msahigh['MSA_Transformer_ensemble'].mean():.3f}"
    )
    print(
        f"TranceptEVE_L - High MSA depth - Spearman: {df_msahigh['TranceptEVE_L'].mean():.3f}"
    )
    print(f"GEMME - High MSA depth - Spearman: {df_msahigh['GEMME'].mean():.3f}")
    print(
        f"EVE_ensemble - High MSA depth - Spearman: {df_msahigh['EVE_ensemble'].mean():.3f}"
    )
    print(
        f"Tranception_L - High MSA depth - Spearman: {df_msahigh['Tranception_L_retrieval'].mean():.4f}"
    )
    print(f"VESPA - High MSA depth - Spearman: {df_msahigh['VESPA'].mean():.3f}")
    print(
        f"EVE_single - High MSA depth - Spearman: {df_msahigh['EVE_single'].mean():.3f}"
    )
    print(
        f"Tranception_M - High MSA depth - Spearman: {df_msahigh['Tranception_M_retrieval'].mean():.3f}"
    )

    print(f"SSEmb - All MSA depth - Spearman: {df['SSEmb'].mean():.4f}")
    print(
        f"MSA Transformer - All MSA depth - Spearman: {df['MSA_Transformer_ensemble'].mean():.4f}"
    )
    print(f"TranceptEVE_L - All MSA depth - Spearman: {df['TranceptEVE_L'].mean():.4f}")
    print(f"GEMME - All MSA depth - Spearman: {df['GEMME'].mean():.4f}")
    print(f"EVE_ensemble - All MSA depth - Spearman: {df['EVE_ensemble'].mean():.3f}")
    print(
        f"Tranception_L - All MSA depth - Spearman: {df['Tranception_L_retrieval'].mean():.3f}"
    )
    print(f"VESPA - All MSA depth - Spearman: {df['VESPA'].mean():.5f}")
    print(f"EVE_single - All MSA depth - Spearman: {df['EVE_single'].mean():.5f}")
    print(
        f"Tranception_M - All MSA depth - Spearman: {df['Tranception_M_retrieval'].mean():.4f}"
    )

    # Make plot - All
    fig, ax = plt.subplots()
    x = [i for i in range(len(df))]
    ax.scatter(
        df["UniProt_ID"],
        df["SSEmb"],
        label=f"SSEmb $\\rho_S$: {df['SSEmb'].mean():.2f} $\pm$ {df['SSEmb'].sem():.2f}",
        s=20,
        marker=",",
    )
    ax.scatter(
        df["UniProt_ID"],
        df["MSA_Transformer_ensemble"],
        label=f"MSA Transformer $\\rho_S$: {df['MSA_Transformer_ensemble'].mean():.2f} $\pm$ {df['MSA_Transformer_ensemble'].sem():.2f}",
        s=20,
    )
    ax.set_ylim(0.0, 0.75)
    ax.legend(loc="upper right")
    ax.set_xticklabels(df["UniProt_ID"], rotation=45, ha="right", fontsize=5)
    ax.set_ylabel("Spearman $\\rho_S$")

    # Save fig
    fig.savefig(
        f"../output/proteingym/proteingym_scatter_{run_name}.pdf", bbox_inches="tight"
    )

    # Make plot - MSA low
    fig, ax = plt.subplots()
    x = [i for i in range(len(df))]
    ax.scatter(
        df_msalow["UniProt_ID"],
        df_msalow["SSEmb"],
        label=f"SSEmb $\\rho_S$: {df_msalow['SSEmb'].mean():.2f} $\pm$ {df_msalow['SSEmb'].sem():.2f}",
        s=20,
        marker=",",
    )
    ax.scatter(
        df_msalow["UniProt_ID"],
        df_msalow["MSA_Transformer_ensemble"],
        label=f"MSA Transformer $\\rho_S$: {df_msalow['MSA_Transformer_ensemble'].mean():.2f} $\pm$ {df_msalow['MSA_Transformer_ensemble'].sem():.2f}",
        s=20,
    )
    ax.set_ylim(0.0, 0.75)
    ax.legend(loc="upper right")
    ax.set_xticklabels(df_msalow["UniProt_ID"], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Spearman $\\rho_S$")

    # Save fig
    fig.savefig(
        f"../output/proteingym/proteingym_scatter_{run_name}_msalow.pdf",
        bbox_inches="tight",
    )
