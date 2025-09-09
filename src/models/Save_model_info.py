
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional




# ---------- helpers to normalize column names ----------
_COEF_COL_MAP = {
    "coef":"coef", "coefficient":"coef", "value":"coef", "beta":"coef",
    "ci_lower":"ci_lower", "lower":"ci_lower", "ci low":"ci_lower",
    "ci_upper":"ci_upper", "upper":"ci_upper", "ci high":"ci_upper",
    "pvalue":"pvalue", "p":"pvalue", "p-value":"pvalue",
    "dollarimpact":"dollar_impact", "dollar_impact":"dollar_impact", "impactusd":"dollar_impact"
}

def _norm_cols(df: pd.DataFrame, want_pvalue: bool, want_dollar: bool) -> pd.DataFrame:
    cols = {c: _COEF_COL_MAP.get(str(c).lower().strip(), c) for c in df.columns}
    out = df.rename(columns=cols).copy()
    needed = {"coef","ci_lower","ci_upper"}
    if want_pvalue:
        needed |= {"pvalue"}
    if want_dollar:
        # dollar impact is optional overall; if want_dollar=True and it's missing, we’ll fill with NaN
        if "dollar_impact" not in out.columns:
            out["dollar_impact"] = np.nan

    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Coefficient frame missing columns: {missing}")
    keep = ["coef","ci_lower","ci_upper"] + (["pvalue"] if want_pvalue else []) + (["dollar_impact"] if "dollar_impact" in out.columns else [])
    return out[keep]

# ---------- model info extraction ----------
def _extract_sarimax_info(res) -> Dict[str, Any]:
    """Safely extract SARIMAX model info; 'res' is statsmodels SARIMAXResults.
       If attributes aren’t present, they’ll be None."""
    if res is None:
        return {}
    m = getattr(res, "model", None)
    order = getattr(m, "order", None)
    seas  = getattr(m, "seasonal_order", None)
    trend = getattr(m, "trend", None)
    freq  = getattr(m, "freq", None) or getattr(m, "data", None)
    if hasattr(freq, "freq"):
        freq = freq.freq  # sometimes model.data has .freq
    exog_names = getattr(m, "exog_names", None)
    info = {
        "order_p": order[0] if order else None,
        "order_d": order[1] if order else None,
        "order_q": order[2] if order else None,
        "seasonal_P": seas[0] if seas else None,
        "seasonal_D": seas[1] if seas else None,
        "seasonal_Q": seas[2] if seas else None,
        "seasonal_s": seas[3] if seas else None,
        "trend": trend,
        "freq": freq,
        "exog_names": list(exog_names) if isinstance(exog_names, (list, tuple)) else exog_names,
        "aic": getattr(res, "aic", None),
        "bic": getattr(res, "bic", None),
        "llf": getattr(res, "llf", None),
        "nobs": getattr(res, "nobs", None),
    }
    return info

def _coerce_model_info_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a user-supplied model_info dict to flat columns."""
    order = d.get("order")
    seas  = d.get("seasonal_order")
    info = {
        "order_p":    order[0] if order else d.get("order_p"),
        "order_d":    order[1] if order else d.get("order_d"),
        "order_q":    order[2] if order else d.get("order_q"),
        "seasonal_P": seas[0]  if seas  else d.get("seasonal_P"),
        "seasonal_D": seas[1]  if seas  else d.get("seasonal_D"),
        "seasonal_Q": seas[2]  if seas  else d.get("seasonal_Q"),
        "seasonal_s": seas[3]  if seas  else d.get("seasonal_s"),
        "trend": d.get("trend"),
        "freq": d.get("freq"),
        "exog_names": d.get("exog_names"),
        "aic": d.get("aic"),
        "bic": d.get("bic"),
        "llf": d.get("llf"),
        "nobs": d.get("nobs"),
    }
    return info

# ---------- main collector/saver ----------
def save_store_artifacts(
    stores_data: Dict[str, Dict[str, Any]],
    out_dir: str | Path,
    run_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Persist multi-store SARIMAX artifacts into tidy Parquet files.

    Expected per-store dict keys:
      - coeff_scaled   : pd.DataFrame (index=variable), columns include Coef/CI_lower/CI_upper/Pvalue and (optional) DollarImpact
      - coeff_unscaled : pd.DataFrame (index=variable), columns include Coef/CI_lower/CI_upper
      - metrics        : dict[str, float]
      - sarimax_res    : SARIMAXResults (optional)
      - model_info     : dict with order/seasonal_order/trend/freq/exog_names (used if sarimax_res is None)
      - scaling        : dict with keys: scaled_y (bool), scaled_x (bool), y_scaler (str|None), x_scaler (str|None)

    Produces:
      - coefficients_scaled.parquet
      - coefficients_unscaled.parquet
      - models.parquet
      - metrics.parquet
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_scaled, rows_unscaled, rows_models, rows_metrics = [], [], [], []

    for store, bundle in stores_data.items():
        # --- scaled coefficients ---
        df_scaled = bundle["coeff_scaled"].copy()
        df_scaled = _norm_cols(df_scaled, want_pvalue=True, want_dollar=True)
        df_scaled["store"] = store
        df_scaled["variable"] = df_scaled.index.astype(str)
        if run_id is not None:
            df_scaled["run_id"] = run_id
        rows_scaled.append(df_scaled.reset_index(drop=True))

        # --- unscaled coefficients ---
        df_unscaled = bundle["coeff_unscaled"].copy()
        df_unscaled = _norm_cols(df_unscaled, want_pvalue=True, want_dollar=True)
        df_unscaled["store"] = store
        df_unscaled["variable"] = df_unscaled.index.astype(str)
        if run_id is not None:
            df_unscaled["run_id"] = run_id
        rows_unscaled.append(df_unscaled.reset_index(drop=True))

        # --- metrics (long-form) ---
        metrics = bundle.get("metrics", {}) or {}
        for k, v in metrics.items():
            rows_metrics.append({
                "store": store,
                "metric": str(k),
                "value": float(v),
                "run_id": run_id,
            })

        # --- model + scaling info per store ---
        res = bundle.get("sarimax_res")
        mi_from_res = _extract_sarimax_info(res) if res is not None else {}
        mi_dict = bundle.get("model_info") or {}
        model_info = mi_from_res or _coerce_model_info_dict(mi_dict)

        scaling = bundle.get("scaling", {}) or {}
        row_model = {
            "store": store,
            "run_id": run_id,
            **model_info,
            "scaled_y": bool(scaling.get("scaled_y")) if "scaled_y" in scaling else None,
            "scaled_x": bool(scaling.get("scaled_x")) if "scaled_x" in scaling else None,
            "y_scaler": scaling.get("y_scaler"),
            "x_scaler": scaling.get("x_scaler"),
        }
        rows_models.append(row_model)

    # ---- Concatenate & order columns ----
    tidy_scaled = pd.concat(rows_scaled, ignore_index=True)[
        ["store","variable","coef","ci_lower","ci_upper","pvalue","dollar_impact"] + (["run_id"] if run_id else [])
    ]
    tidy_unscaled = pd.concat(rows_unscaled, ignore_index=True)[
        ["store","variable","coef","ci_lower","ci_upper","pvalue","dollar_impact"] + (["run_id"] if run_id else [])
    ]
    models = pd.DataFrame(rows_models)
    # Order model columns nicely
    model_cols_order = ["store","order_p","order_d","order_q",
                        "seasonal_P","seasonal_D","seasonal_Q","seasonal_s",
                        "trend","freq","exog_names","aic","bic","llf","nobs",
                        "scaled_y","scaled_x","y_scaler","x_scaler"]
    if run_id:
        model_cols_order.append("run_id")
    models = models.reindex(columns=model_cols_order)

    metrics_df = pd.DataFrame(rows_metrics, columns=["store","metric","value","run_id"] if run_id else ["store","metric","value"])

    # ---- Save to Parquet ----
    p_scaled   = out_dir / "coefficients_scaled.parquet"
    p_unscaled = out_dir / "coefficients_unscaled.parquet"
    p_models   = out_dir / "models.parquet"
    p_metrics  = out_dir / "metrics.parquet"

    tidy_scaled.to_parquet(p_scaled, index=False)
    tidy_unscaled.to_parquet(p_unscaled, index=False)
    models.to_parquet(p_models, index=False)
    metrics_df.to_parquet(p_metrics, index=False)

    return {
        "coefficients_scaled":   str(p_scaled),
        "coefficients_unscaled": str(p_unscaled),
        "models":                str(p_models),
        "metrics":               str(p_metrics),
    }
