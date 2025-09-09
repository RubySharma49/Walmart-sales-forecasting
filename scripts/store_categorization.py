import sys, os
sys.path.append(os.path.abspath("."))
import argparse
import numpy as np
import pandas as pd
from src.data.load import load_models
from src.utils.config import load_config
from src.viz.store_category_plots import group_dot_plot, feature_per_group


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default="configs/Params_set.yml")

    args = ap.parse_args()

    cfg = load_config(args.config)

    coeff_s, coeff_u, models, metrics = load_models(cfg["store_category"]["path_model_save"])

    coef_wide = (coeff_u
                .assign(ci_width=coeff_u["ci_upper"] - coeff_u["ci_lower"])
                .pivot(index="store", columns="variable", values="coef")
                .add_prefix("coef_unscaled__"))

    ciw_wide  = (coeff_u
                .assign(ci_width=coeff_u["ci_upper"] - coeff_u["ci_lower"])
                .pivot(index="store", columns="variable", values="ci_width")
                .add_prefix("ciw__"))

    imp_wide = (coeff_u
                .pivot(index="store", columns="variable", values="dollar_impact")
                .add_prefix("impact__"))

    p_wide = (coeff_u
            .pivot(index="store", columns="variable", values="pvalue")
            .add_prefix("p__"))


    agg = (coeff_u
        .groupby(["store"], as_index=True)
        .agg(
            n_sig=("pvalue", lambda s: int((s < 0.05).sum())),
            total_abs_impact=("dollar_impact", lambda s: float(s.sum())),
            max_abs_impact=("dollar_impact", lambda s: float(s.max()))
        ))
    desc_feat = imp_wide.join([agg,p_wide,ciw_wide],how = "left")


    desc_feat.drop(columns=[ "impact__cal_christmas_w-1",
                        "impact__cal_christmas_w0","impact__cal_thanksgiving_w0",
                        'n_sig', 'total_abs_impact', 'max_abs_impact',
                        "p__cal_christmas_w-1", "p__cal_christmas_w0",
                        "p__cal_thanksgiving_w0", 
                        "ciw__cal_christmas_w-1",
                            "ciw__cal_christmas_w0",
                        "ciw__cal_thanksgiving_w0"],inplace = True)


    metrics_wide = metrics.pivot_table(index="store", columns="metric", values="value")
    metrics_wide.drop(columns=["rmse"], inplace = True)
    metrics_wide.rename(columns={"rmse_org_scale": "rmse"}, inplace=True)

    pred_feat = metrics_wide
    pred_feat.drop(columns=["mape"], inplace=True)

    feat = desc_feat.join([pred_feat], how = "left")


    # 0) Optional sanity: coerce to numeric
    for c in ["impact__Holiday_Flag","impact__Month_Start_Flag",
            "p__Holiday_Flag","p__Month_Start_Flag",
            "ciw__Holiday_Flag","ciw__Month_Start_Flag",
            "r2","rmse","smape"]:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    # 1) Choose the dominant driver per store (higher absolute $ impact)
    im_m = feat["impact__Month_Start_Flag"]
    im_h = feat["impact__Holiday_Flag"]
    pick_month = im_m >= im_h

    feat["driver"]         = np.where(pick_month, "Month_Start", "Holiday")
    feat["driver_impact"]  = np.where(pick_month, im_m, im_h)
    feat["driver_p"]       = np.where(pick_month, feat["p__Month_Start_Flag"], feat["p__Holiday_Flag"])
    feat["driver_ciw"]     = np.where(pick_month, feat["ciw__Month_Start_Flag"], feat["ciw__Holiday_Flag"])

    # 2) Data-driven thresholds (quantiles)
    p90 = feat["driver_impact"].quantile(0.90)     # high impact cutoff
    p50 = feat["driver_impact"].quantile(0.50)     # medium impact cutoff
    ciw_med = feat["driver_ciw"].median()          # "narrow CI" cutoff (smaller is better)

    # Model quality gates (tune if you like)
    r2_good      = 0.60
    smape_good   = 20.0
    rmse_cutoff  = feat["rmse"].median()           # relative gate; use domain rule if you have one



    # 3) Tier the effect & confidence & model quality
    feat["impact_tier"] = np.where(feat["driver_impact"] >= p90, "high",
                        np.where(feat["driver_impact"] >= p50, "medium", "low"))

    feat["confident"]   = (feat["driver_p"] < 0.05) & (feat["driver_ciw"] <= ciw_med)

    feat["model_good"]  = (feat["r2"] >= r2_good) & (feat["smape"] <= smape_good) & (feat["rmse"] <= rmse_cutoff)

    # 4) Final grouping (plain if/elif logic expressed with numpy.select)
    conditions = [
        (feat["impact_tier"]=="high")   & feat["confident"] & feat["model_good"],
        (feat["impact_tier"]=="medium") & feat["confident"] & feat["model_good"],
        (feat["impact_tier"].isin(["high","medium"])) & (~feat["confident"] | ~feat["model_good"]),
    ]
    labels = [
        "A1: High-impact & confident (go heavy)",
        "A2: Medium-impact & confident (right-size)",
        "B: Impact present but uncertain/weak model (pilot)",
    ]
    feat["group"] = np.select(conditions, labels, default="C: Low-impact or baseline (no special ops)")

    # 5) (Optional) Also record which driver led to the assignment and its numbers (handy for slides)
    feat["driver_summary"] = (
        feat["driver"].str[:3] +  # 'Mon' or 'Hol' just to keep it short
        " | $" + feat["driver_impact"].round(0).astype("Int64").astype(str) +
        " | p=" + feat["driver_p"].round(3).astype(str)
    )

    # 6) Quick overview
    print(feat[["group","driver","driver_summary","r2","rmse","smape"]].head())
    print("\nCounts by group:\n", feat["group"].value_counts())

    feature_per_group(feat, cfg["store_category"]["path_group_feature"])
    group_dot_plot(feat, cfg["store_category"]["path_group_dot_plt"])


if __name__ == "__main__":
  main()