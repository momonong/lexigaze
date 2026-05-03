import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder


DEFAULT_INPUT_FEATURES = "data/geco/geco_pp01_cognitive_mass.csv"
DEFAULT_CLEAN_CSV = "data/geco/geco_pp01_trial5_clean.csv"
OUTPUT_DIR = "data/geco"

SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3
CALIBRATION_WINDOW = 30

DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0
RANDOM_SEED = 42


def inject_noise(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)
    out = df.copy()
    out["raw_x"] = out["true_x"] + np.random.normal(0, SIGMA_X, len(out))
    out["raw_y"] = out["true_y"] + np.random.normal(0, SIGMA_Y, len(out)) + DRIFT_Y
    return out


def strict_accuracy(target_indices: np.ndarray, pred_indices: np.ndarray) -> float:
    return float(np.mean(target_indices == pred_indices) * 100.0)


def relaxed_accuracy(target_indices: np.ndarray, pred_indices: np.ndarray) -> float:
    return float(np.mean(np.abs(target_indices - pred_indices) <= 1) * 100.0)


def run_model(df: pd.DataFrame, use_ovp: bool, model_name: str) -> tuple[pd.DataFrame, dict]:
    word_boxes = [[r["true_x"] - 20, r["true_y"] - 15, r["true_x"] + 20, r["true_y"] + 15] for _, r in df.iterrows()]
    base_cm = df["cognitive_mass"].to_numpy(dtype=float)
    gaze_sequence = df[["raw_x", "raw_y"]].to_numpy(dtype=float)

    t_matrix = PsycholinguisticTransitionMatrix(
        sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA
    ).build_matrix(len(df), base_cm)

    calibrator = AutoCalibratingDecoder(calibration_window_size=CALIBRATION_WINDOW)
    pred_indices, drift_vec = calibrator.calibrate_and_decode(
        gaze_sequence,
        word_boxes,
        base_cm,
        t_matrix,
        sigma_gaze=[SIGMA_X, SIGMA_Y],
        use_ovp=use_ovp,
    )
    pred_indices = np.array(pred_indices, dtype=int)
    corrected = gaze_sequence - np.array(drift_vec, dtype=float)

    out = df.copy()
    out["predicted_index"] = pred_indices
    out["predicted_word"] = [str(df.iloc[i]["WORD"]).strip() for i in pred_indices]
    out["corrected_x"] = corrected[:, 0]
    out["corrected_y"] = corrected[:, 1]
    out["estimated_drift_x"] = float(drift_vec[0])
    out["estimated_drift_y"] = float(drift_vec[1])
    out["model"] = model_name
    out["use_ovp"] = bool(use_ovp)

    target_indices = np.arange(len(df), dtype=int)
    metrics = {
        "model": model_name,
        "use_ovp": bool(use_ovp),
        "strict_accuracy_percent": strict_accuracy(target_indices, pred_indices),
        "relaxed_accuracy_percent": relaxed_accuracy(target_indices, pred_indices),
        "estimated_drift_x": float(drift_vec[0]),
        "estimated_drift_y": float(drift_vec[1]),
    }
    return out, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute and persist STOCK-T trial trajectory CSVs.")
    parser.add_argument(
        "--input-features",
        default=DEFAULT_INPUT_FEATURES,
        help="Input feature CSV with WORD_ID/WORD/true_x/true_y/cognitive_mass.",
    )
    parser.add_argument(
        "--output-prefix",
        default="geco_pp01_trial5_stockt",
        help="Filename prefix for output CSVs.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_features):
        raise FileNotFoundError(f"Input features not found: {args.input_features}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(args.input_features).copy()

    required = [
        "WORD_ID",
        "WORD",
        "true_x",
        "true_y",
        "cognitive_mass",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in features.csv: {missing}")

    if "WORD_TOTAL_READING_TIME" not in df.columns:
        if os.path.exists(DEFAULT_CLEAN_CSV):
            df_clean = pd.read_csv(DEFAULT_CLEAN_CSV)
            if {"WORD_ID", "WORD_TOTAL_READING_TIME"}.issubset(df_clean.columns):
                df = df.merge(
                    df_clean[["WORD_ID", "WORD_TOTAL_READING_TIME"]],
                    on="WORD_ID",
                    how="left",
                )
        if "WORD_TOTAL_READING_TIME" not in df.columns:
            df["WORD_TOTAL_READING_TIME"] = np.nan

    for col in ["true_x", "true_y", "cognitive_mass", "WORD_TOTAL_READING_TIME"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["true_x", "true_y", "cognitive_mass"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows available after numeric cleaning.")

    df = inject_noise(df)

    m4_df, m4_metrics = run_model(df, use_ovp=False, model_name="M4_STOCKT_POM_EM")
    m5_df, m5_metrics = run_model(df, use_ovp=True, model_name="M5_ULTIMATE_STOCKT_POM_EM_OVP")

    out_m4 = os.path.join(OUTPUT_DIR, f"{args.output_prefix}_m4_trajectory.csv")
    out_m5 = os.path.join(OUTPUT_DIR, f"{args.output_prefix}_m5_trajectory.csv")
    out_metrics = os.path.join(OUTPUT_DIR, f"{args.output_prefix}_metrics.csv")

    keep_cols = [
        "WORD_ID",
        "WORD",
        "true_x",
        "true_y",
        "WORD_TOTAL_READING_TIME",
        "cognitive_mass",
        "raw_x",
        "raw_y",
        "corrected_x",
        "corrected_y",
        "predicted_index",
        "predicted_word",
        "estimated_drift_x",
        "estimated_drift_y",
        "model",
        "use_ovp",
    ]
    m4_df[keep_cols].to_csv(out_m4, index=False)
    m5_df[keep_cols].to_csv(out_m5, index=False)
    pd.DataFrame([m4_metrics, m5_metrics]).to_csv(out_metrics, index=False)

    print("✅ Recomputed trajectories saved:")
    print(f"  - {out_m4}")
    print(f"  - {out_m5}")
    print(f"  - {out_metrics}")
    print("\n📊 Metrics:")
    for item in [m4_metrics, m5_metrics]:
        print(
            f"  {item['model']}: strict={item['strict_accuracy_percent']:.2f}%, "
            f"relaxed={item['relaxed_accuracy_percent']:.2f}%, "
            f"drift=({item['estimated_drift_x']:.2f}, {item['estimated_drift_y']:.2f})"
        )


if __name__ == "__main__":
    main()
