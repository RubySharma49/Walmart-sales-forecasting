import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


def feature_per_group(feat:pd.DataFrame, path_feature_group_plt:str):

    if "group" not in feat.columns:
        raise ValueError("Expected a 'group' column in `feat`.")
    feat["group_letter"] = feat["group"].astype(str).str.extract(r"^([A-Za-z])", expand=False).fillna("U")

    # 2) Choose features to plot
    features = [
        "impact__Holiday_Flag",
        "impact__Month_Start_Flag",
        "p__Holiday_Flag",
        "p__Month_Start_Flag",
        "ciw__Holiday_Flag",
        "ciw__Month_Start_Flag",
        "r2",
        "rmse",
        "smape",
    ]

    # Keep only features that actually exist
    features = [c for c in features if c in feat.columns]
    if not features:
        raise ValueError("None of the expected feature columns were found in `feat`.")

    # Coerce to numeric (in case of stray strings)
    for c in features:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    # 3) Plot: 3x3 grid (fits 9 features). If you have fewer, unused axes will be hidden.
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    # Sorted group order (A, B, C, ...)
    groups = sorted(feat["group_letter"].dropna().unique().tolist())

    for ax, col in zip(axes, features):
        data_by_group = [feat.loc[feat["group_letter"] == g, col].dropna().values for g in groups]
        # If a group has no data for this feature, boxplot will skip it; handle empty lists:
        labels = [g for g, arr in zip(groups, data_by_group) if len(arr) > 0]
        data   = [arr for arr in data_by_group if len(arr) > 0]

        if len(data) == 0:
            ax.text(0.5, 0.5, f"No data for {col}", ha="center", va="center")
            ax.axis("off")
            continue

        bp = ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanline=False,
            patch_artist=False,
            widths=0.7,
        )
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("Group (letter)")
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    # Hide any unused axes (if < 9 features)
    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle("Feature Distributions by Group (letter)", y=1.02, fontsize=13)

    # 4) Save to a single-page PDF
    with PdfPages(path_feature_group_plt) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)



def group_dot_plot(feat:pd.DataFrame, path_group_dot_plt:str):

    plot_df = feat.copy()
    plot_df["group_letter"] = plot_df["group"].astype(str).str.extract(r"^([A-Za-z])", expand=False).fillna("U")

    # Optional: sort by group then store for a tidy display
    plot_df = plot_df.sort_values(["group_letter", plot_df.index.name or 0])

    # 2) Map letters to x positions and colors
    letters = sorted(plot_df["group_letter"].unique().tolist())
    xpos = {g: i for i, g in enumerate(letters)}

    cmap = plt.get_cmap("tab10" if len(letters) > 7 else "tab10")
    color_map = {g: cmap(i % cmap.N) for i, g in enumerate(letters)}

    x = plot_df["group_letter"].map(xpos).to_numpy(dtype=float)
    y = np.arange(len(plot_df))
    colors = plot_df["group_letter"].map(color_map).to_list()

    # 3) Plot (vertical list of stores)
    fig_h = max(6, 0.22 * len(plot_df))  # scale height for readability (works for ~45 stores)
    fig, ax = plt.subplots(figsize=(8, fig_h), constrained_layout=True)

    ax.scatter(x, y, s=80, c=colors, edgecolor="black", linewidth=0.6, zorder=3)

    # y-axis = store names
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.index)
    ax.invert_yaxis()  # first store at top

    # x-axis = group letters
    ax.set_xticks(range(len(letters)))
    ax.set_xticklabels(letters)
    ax.set_xlabel("Group")
    ax.set_title("Stores by Cluster Group (color = group letter)")
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    # Legend for colors
    handles = [
        Line2D([0],[0], marker="o", color="none",
            markerfacecolor=color_map[g], markeredgecolor="black",
            markersize=8, label=g)
        for g in letters
    ]
    ax.legend(handles=handles, title="Group", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    plt.tight_layout()

    # 4) (Optional) Save to PDF
    with PdfPages(path_group_dot_plt) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
