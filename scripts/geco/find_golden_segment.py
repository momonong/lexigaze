import argparse
from typing import List, Optional

import pandas as pd


def parse_word_id_parts(word_id: str) -> tuple[str, Optional[int]]:
    """Return trial prefix and within-trial word index parsed from WORD_ID."""
    token = str(word_id).strip()
    parts = token.split("-")
    if len(parts) < 3:
        return token, None
    prefix = "-".join(parts[:-1])
    try:
        within_trial_idx = int(parts[-1])
    except ValueError:
        within_trial_idx = None
    return prefix, within_trial_idx


def load_and_prepare(clean_csv: str, bayes_csv: str) -> pd.DataFrame:
    df_clean = pd.read_csv(clean_csv).copy()
    df_bayes = pd.read_csv(bayes_csv).copy()

    # Preserve original row indices from the clean source for reporting.
    df_clean["source_row"] = df_clean.index

    # Keep one trajectory row per WORD_ID to avoid accidental duplicates.
    df_bayes = df_bayes.drop_duplicates(subset=["WORD_ID"], keep="first")

    # Join trajectory columns onto clean text rows.
    df = df_clean.merge(
        df_bayes[["WORD_ID", "webcam_y", "calibrated_y"]],
        on="WORD_ID",
        how="inner",
    )

    # Parse WORD_ID to obtain trial grouping and word order index.
    parsed = df["WORD_ID"].map(parse_word_id_parts)
    df["trial_prefix"] = parsed.map(lambda x: x[0])
    df["word_index"] = parsed.map(lambda x: x[1])

    numeric_cols = ["true_y", "webcam_y", "calibrated_y", "word_index"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols).copy()

    # Drift metrics (y-axis error to true word center).
    df["raw_y_error"] = (df["webcam_y"] - df["true_y"]).abs()
    df["corrected_y_error"] = (df["calibrated_y"] - df["true_y"]).abs()
    return df


def evaluate_windows(
    df: pd.DataFrame,
    min_window: int,
    max_window: int,
    min_raw_drift: float,
    max_corrected_error: float,
) -> List[dict]:
    results: List[dict] = []
    n = len(df)

    for w in range(min_window, max_window + 1):
        if w > n:
            continue
        for start in range(0, n - w + 1):
            end = start + w
            win = df.iloc[start:end]

            # Consecutive fixations in original clean.csv.
            src = win["source_row"].to_list()
            if any(src[i + 1] != src[i] + 1 for i in range(len(src) - 1)):
                continue

            # Keep window in the same trial/chunk.
            if win["trial_prefix"].nunique() != 1:
                continue

            # Forward reading only: strictly increasing word_index.
            idx = win["word_index"].astype(int).to_list()
            if any(idx[i + 1] <= idx[i] for i in range(len(idx) - 1)):
                continue

            avg_raw = float(win["raw_y_error"].mean())
            avg_corr = float(win["corrected_y_error"].mean())

            if avg_raw <= min_raw_drift:
                continue
            if avg_corr >= max_corrected_error:
                continue

            words = " ".join(win["WORD"].astype(str).str.strip().tolist())
            results.append(
                {
                    "score": avg_raw - avg_corr,
                    "window_size": w,
                    "start_row": int(win["source_row"].iloc[0]),
                    "end_row": int(win["source_row"].iloc[-1]),
                    "trial_prefix": str(win["trial_prefix"].iloc[0]),
                    "avg_raw_drift": avg_raw,
                    "avg_corrected_error": avg_corr,
                    "words": words,
                }
            )

    # Higher score first; then stronger raw drift and tighter correction.
    results.sort(
        key=lambda r: (r["score"], r["avg_raw_drift"], -r["avg_corrected_error"]),
        reverse=True,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find top-quality GECO scanpath segments for qualitative visualization."
    )
    parser.add_argument(
        "--clean-csv",
        default="data/geco/geco_pp01_trial5_clean.csv",
        help="Path to clean GECO CSV with WORD/true_y.",
    )
    parser.add_argument(
        "--bayes-csv",
        default="data/geco/geco_pp01_bayesian_results.csv",
        help="Path to Bayesian results CSV with webcam_y/calibrated_y.",
    )
    parser.add_argument("--min-window", type=int, default=5, help="Minimum window size.")
    parser.add_argument("--max-window", type=int, default=8, help="Maximum window size.")
    parser.add_argument(
        "--min-raw-drift",
        type=float,
        default=30.0,
        help="Minimum mean |raw_y - word_center_y| required.",
    )
    parser.add_argument(
        "--max-corrected-error",
        type=float,
        default=5.0,
        help="Maximum mean |corrected_y - word_center_y| allowed.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="How many segments to print.")
    args = parser.parse_args()

    if args.min_window < 2 or args.max_window < args.min_window:
        raise ValueError("Invalid window range: require 2 <= min-window <= max-window.")

    df = load_and_prepare(args.clean_csv, args.bayes_csv)
    matches = evaluate_windows(
        df=df,
        min_window=args.min_window,
        max_window=args.max_window,
        min_raw_drift=args.min_raw_drift,
        max_corrected_error=args.max_corrected_error,
    )

    print("\n=== Golden Segment Miner ===")
    print(
        f"Criteria: window={args.min_window}-{args.max_window}, "
        f"avg_raw_drift>{args.min_raw_drift}, avg_corrected_error<{args.max_corrected_error}"
    )

    if not matches:
        print("No segments matched all strict constraints.")
        return

    top = matches[: args.top_k]
    print(f"Found {len(matches)} matching segments. Top {len(top)}:\n")
    for rank, seg in enumerate(top, start=1):
        print(
            f"[{rank}] rows {seg['start_row']}->{seg['end_row']} "
            f"(trial={seg['trial_prefix']}, window={seg['window_size']})"
        )
        print(
            f"    avg_raw_drift={seg['avg_raw_drift']:.2f}px, "
            f"avg_corrected_error={seg['avg_corrected_error']:.2f}px, "
            f"score={seg['score']:.2f}"
        )
        print(f"    text: {seg['words']}\n")


if __name__ == "__main__":
    main()
