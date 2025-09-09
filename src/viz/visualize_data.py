import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import seaborn as sns


def plot_scatter(df, x, y, xlabel, ylabel, title, col, figsize):
    plt.figure(figsize = figsize)
    sns.boxplot(data=df, x=x, y=y, color=col)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90) 
    plt.show()
    

def plot_train_test_forecast(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    conf_int: pd.DataFrame = None,
    title: str = "Train / Test / Forecast",
    ylabel: str = "Value",
    figsize=(12, 6),
    annotate_fmt: str = "RMSE = {rmse:.3f}\nR² = {r2:.3f}",
    show_split_line: bool = True,
    grid: bool = True,
    ax: plt.Axes = None,
    rmse: float = None,
    r2: float = None,
    smape: float = None
):
    """
    Plot train/test/forecast on the given Axes (subplot-friendly).

    Parameters
    ----------
    y_train, y_test, y_pred : pd.Series
        Time-indexed series. y_pred can extend beyond test.
    conf_int : pd.DataFrame, optional
        With columns ["lower", "upper"] and index aligned to y_pred.
    ax : matplotlib.axes.Axes, optional
        If None, a new figure/axes will be created.
    rmse, r2 : float, optional
        If provided, metrics are displayed without recomputing. Otherwise computed from y_test vs y_pred on test index.

    Returns
    -------
    ax : matplotlib.axes.Axes
    (rmse, r2) : tuple[float, float]
    """

    # --- Remove duplicate indices by keeping the last occurrence ---
    y_train = y_train[~y_train.index.duplicated(keep="last")]
    y_test  = y_test[~y_test.index.duplicated(keep="last")]
    y_pred  = y_pred[~y_pred.index.duplicated(keep="last")]
    if conf_int is not None:
        conf_int = conf_int[~conf_int.index.duplicated(keep="last")]

    # --- Align forecast to test for metrics (if not provided) ---
    if rmse is None or r2 is None:
        y_pred_on_test = y_pred.reindex(y_test.index)
        if y_pred_on_test.isna().any():
            missing = y_pred_on_test.index[y_pred_on_test.isna()]
            raise ValueError(
                f"y_pred is missing predictions for some test timestamps: {missing[:5].tolist()}..."
            )
        rmse = np.sqrt(mean_squared_error(y_test.values, y_pred_on_test.values))
        r2   = r2_score(y_test.values, y_pred_on_test.values)

    # --- Axes setup ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # --- Plot ---
    ax.plot(y_train.index, y_train.values, label="Train (actual)", linewidth=1.8)
    ax.plot(y_test.index,  y_test.values,  label="Test (actual)",  linewidth=1.8)
    ax.plot(y_pred.index,  y_pred.values,  label="Forecast", linewidth=2.2, linestyle="--")

    if conf_int is not None:
        ci = conf_int.reindex(y_pred.index)

        if {"lower y", "upper y"}.issubset(ci.columns):
            ax.fill_between(ci.index, ci["lower y"].values, ci["upper y"].values, alpha=0.20, label="Forecast CI")

    # Shade test region
    test_start, test_end = y_test.index.min(), y_test.index.max()
    ax.axvspan(test_start, test_end, color="grey", alpha=0.08, lw=0)

    if show_split_line:
        ax.axvline(test_start, color="grey", linestyle=":", linewidth=1.5)

    # Annotation box centered over test window
    ax.relim(); ax.autoscale()
    y_top = ax.get_ylim()[1]
    x_mid = test_start + (test_end - test_start) / 2
    ax.text(
        x_mid, y_top,
        annotate_fmt.format(rmse=rmse, r2=r2, smape=smape),
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)
    )

    ax.set_title(title, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    if grid:
        ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)

    if created_fig:
        plt.tight_layout()

    return ax, (rmse, r2, smape)

def plot_exog_coefficients(ax, org_coeff, p_thresh=0.05,
                           pos_color="#2ca02c", neg_color="#d62728",
                           title="SARIMAX Exogenous Coefficients\n(Impact & Statistical Significance)"):
    """
    Draw a coefficient forest-style plot with CIs and significance markers on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    org_coeff : pandas.DataFrame
        Must contain columns: 'Coef', 'CI_lower', 'CI_upper', 'Pvalue'.
        Index should be variable names (strings).
    p_thresh : float
        Significance threshold for marking points with a star.
    pos_color, neg_color : str
        Colors for positive/negative coefficients.
    title : str
        Title for the subplot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axes, modified.
    """
    coef  = org_coeff["Coef"].to_numpy()
    lower = org_coeff["CI_lower"].to_numpy()
    upper = org_coeff["CI_upper"].to_numpy()
    pvals = org_coeff["Pvalue"].to_numpy()
    yvals = org_coeff.index  # can be categorical labels in recent Matplotlib

    colors = [pos_color if c > 0 else neg_color for c in coef]
    xerr = [coef - lower, upper - coef]

    # CI + central marker
    ax.errorbar(coef, yvals, xerr=xerr, fmt='o', color='black',
                ecolor='gray', elinewidth=2, capsize=4, zorder=2)

    # significance markers + labels
    span = (upper.max() - lower.min()) if len(org_coeff) else 1.0
    for c, lo, hi, p, y, col in zip(coef, lower, upper, pvals, yvals, colors):
        if p < p_thresh:
            ax.scatter(c, y, color=col, s=120, edgecolor="black", linewidth=1.5, zorder=3)
            sig = f"p={p:.3f}*"
        else:
            ax.scatter(c, y, color=col, s=80, alpha=0.6, zorder=3)
            sig = f"p={p:.3f}"
        effect_note = f"{c:.2e}"
        ax.text(hi + 0.02 * span, y, f"{effect_note}, {sig}", va='center', fontsize=9)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Variable")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    return ax

