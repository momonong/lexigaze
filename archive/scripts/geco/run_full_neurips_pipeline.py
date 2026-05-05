#!/usr/bin/env python3
"""
NeurIPS 2026 — LexiGaze / STOCK-T 標準化全體受試者消融管線 (N=37)

預期目錄結構 (由 batch_extract_features.py 產出):
  data/geco/population/
    L1/
      <subject_id>/          # 例如 pp01, 02, ...
        trial_<int>/
          features.csv         # 欄位需含 true_x, true_y, cognitive_mass
    L2/
      <subject_id>/
        trial_<int>/
          features.csv

產出:
  - 進度: results/interim_ablation.csv (每完成一位受試者 append 一行)
  - 錯誤: logs/error_log.txt (trial 級失敗不中止)
  - 日誌: logs/neurips_pipeline.log
  - 最終: data/geco/benchmark/neurips_final_ablation_N37.csv
  - 終端: LaTeX table 片段

從專案根目錄執行:
  python scripts/geco/run_full_neurips_pipeline.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 專案根目錄 (scripts/geco -> ../..)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder, StandardKalmanDecoder
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix, ReadingTransitionMatrix

# ---------------------------------------------------------------------------
# 鎖定常數 (與 evaluate_population_ablation / NeurIPS stress test 對齊)
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
EXPECTED_N_SUBJECTS = 37
POPULATION_ROOT = PROJECT_ROOT / "data" / "geco" / "population"
INTERIM_CSV = PROJECT_ROOT / "results" / "interim_ablation.csv"
ERROR_LOG = PROJECT_ROOT / "logs" / "error_log.txt"
PIPELINE_LOG = PROJECT_ROOT / "logs" / "neurips_pipeline.log"
FINAL_CSV = PROJECT_ROOT / "data" / "geco" / "benchmark" / "neurips_final_ablation_N37.csv"

SIGMA_FWD = 0.8
SIGMA_REG = 1.5
GAMMA = 0.3
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0
CM_UNIFORM_VALUE = 2.5
CALIBRATION_WINDOW = 30

REQUIRED_COLUMNS = ("true_x", "true_y", "cognitive_mass")

# 對外論文命名 (CSV / LaTeX)
VARIANT_PUBLICATION_KEYS: Tuple[str, ...] = (
    "Full_STOCK-T",
    "w/o_CM",
    "w/o_POM",
    "w/o_EM",
    "w/o_Temp",
)

# 內部鍵 (與欄位前綴一致)
VARIANT_INTERNAL: Tuple[str, ...] = ("full", "wo_cm", "wo_pom", "wo_em", "wo_temp")


def set_deterministic(seed: int = GLOBAL_SEED) -> None:
    """
    鎖死隨機性: numpy / Python random / torch (若已安裝)。
    注意: 各 trial 的雜訊另外使用基於路徑的子種子 (見 trial_noise_rng)，
    以保證 (1) 全管線可重現 (2) 不同 trial 的雜訊軌跡彼此不同但可重現。
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def trial_noise_rng(rel_path: str, base_seed: int = GLOBAL_SEED) -> np.random.Generator:
    """依 trial 相對路徑產生穩定子種子，使同一檔案每次執行雜訊相同。"""
    digest = hashlib.sha256(rel_path.replace("\\", "/").encode("utf-8")).digest()
    sub = int.from_bytes(digest[:4], "little", signed=False) % (2**31 - 1)
    return np.random.default_rng((base_seed + sub) % (2**32 - 1))


def inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """+45px 垂直漂移與高斯 jitter (sigma_x=40, sigma_y=30)。"""
    n = len(df)
    out = df.copy()
    out["noisy_x"] = out["true_x"].astype(float) + rng.normal(0.0, SIGMA_X, n)
    out["noisy_y"] = out["true_y"].astype(float) + rng.normal(0.0, SIGMA_Y, n) + DRIFT_Y
    return out


def evaluate_word_and_recovery(
    target_indices: np.ndarray,
    predicted_indices: Sequence[int],
    estimated_drift_y: float,
    true_drift_y: float,
) -> Tuple[float, float]:
    total = len(target_indices)
    if total == 0:
        return 0.0, 0.0
    acc = sum(1 for t, p in zip(target_indices, predicted_indices) if int(t) == int(p)) / total * 100.0
    recovery = 100.0 if abs(float(estimated_drift_y) - float(true_drift_y)) < 15.0 else 0.0
    return acc, recovery


def _validate_features_df(df: pd.DataFrame, path: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 缺少欄位 {missing}")
    if len(df) < 3:
        raise ValueError(f"{path}: 長度過短 ({len(df)})")


def discover_trial_feature_files(pop_root: Path) -> List[Path]:
    """
    掃描 population/L1 與 population/L2 下所有 layout.csv。
    L1 = 荷蘭語母語者資料夾層級; L2 = 雙語讀者。
    """
    paths: List[Path] = []
    for lang in ("L1", "L2"):
        lang_dir = pop_root / lang
        if not lang_dir.is_dir():
            continue
        paths.extend(sorted(lang_dir.glob("*/*/layout.csv")))
    return sorted(paths, key=lambda p: str(p).replace("\\", "/"))


def parse_lang_subject_trial(feature_path: Path, pop_root: Path) -> Tuple[str, str, str]:
    rel = feature_path.relative_to(pop_root)
    parts = rel.parts
    if len(parts) < 4 or parts[0] not in ("L1", "L2"):
        raise ValueError(f"無法解析路徑 (預期 population/<L1|L2>/<sub>/trial_*/layout.csv): {feature_path}")
    lang = parts[0]
    subject_id = parts[1]
    trial_folder = parts[2]
    if not trial_folder.startswith("trial_"):
        raise ValueError(f"無法解析 trial 目錄: {feature_path}")
    return lang, subject_id, trial_folder


def group_paths_by_subject(paths: Sequence[Path], pop_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    subject_key = f'{lang}|{subject_id}' -> { 'lang', 'subject_id', 'paths': [...] }
    """
    groups: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {"lang": "", "subject_id": "", "paths": []}  # type: ignore[arg-type]
    )
    for p in paths:
        lang, sub, _trial = parse_lang_subject_trial(p, pop_root)
        key = f"{lang}|{sub}"
        bucket = groups[key]
        bucket["lang"] = lang
        bucket["subject_id"] = sub
        bucket["paths"].append(p)
    for b in groups.values():
        b["paths"] = sorted(b["paths"], key=lambda x: str(x))
    return dict(groups)


def count_unique_subjects(groups: Mapping[str, Dict[str, Any]]) -> int:
    return len(groups)


def load_completed_subject_keys(interim_csv: Path) -> Set[str]:
    if not interim_csv.is_file():
        return set()
    try:
        df = pd.read_csv(interim_csv)
        if "subject_key" not in df.columns:
            return set()
        return set(df["subject_key"].astype(str).unique())
    except Exception as exc:  # noqa: BLE001
        logging.warning("讀取 interim 失敗 (%s)，將重新處理所有受試者: %s", interim_csv, exc)
        return set()


def append_subject_row(interim_csv: Path, row: Dict[str, Any]) -> None:
    interim_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not interim_csv.is_file()
    df.to_csv(interim_csv, mode="a", index=False, header=header)


def log_trial_error(error_log: Path, message: str) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def run_five_variants_on_trial(
    feature_path: Path,
    pop_root: Path,
) -> Dict[str, Tuple[float, float]]:
    """
    回傳內部 variant 鍵 -> (word_acc %, traj_recovery %)。
    """
    lang, _sub, _trial = parse_lang_subject_trial(feature_path, pop_root)
    rel = str(feature_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    df_layout = pd.read_csv(feature_path)
    fixations_path = feature_path.parent / "fixations.csv"
    if not fixations_path.is_file():
        raise FileNotFoundError(f"Missing fixations file: {fixations_path}")
    df_fixations = pd.read_csv(fixations_path)
    
    _validate_features_df(df_layout, rel)
    rng = trial_noise_rng(rel)
    
    df_fixations = df_fixations.rename(columns={'fixation_x': 'true_x', 'fixation_y': 'true_y'})
    df_fixations = inject_noise(df_fixations, rng)

    cm_real = df_layout["cognitive_mass"].values.astype(float)
    cm_uniform = np.ones(len(df_layout), dtype=float) * CM_UNIFORM_VALUE
    word_boxes: List[List[float]] = [
        [float(row.true_x) - 20, float(row.true_y) - 15, float(row.true_x) + 20, float(row.true_y) + 15]
        for row in df_layout.itertuples(index=False)
    ]
    gaze_seq = df_fixations[["noisy_x", "noisy_y"]].values.astype(float)
    targets = df_fixations["layout_index"].values.astype(int)
    is_l2_reader = lang == "L2"
    sigma_gaze = [SIGMA_X, SIGMA_Y]

    out: Dict[str, Tuple[float, float]] = {}

    with contextlib.redirect_stdout(io.StringIO()):
        # 1) Full STOCK-T
        pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=SIGMA_FWD, sigma_reg=SIGMA_REG, gamma=GAMMA)
        t_pom_real = pom_builder.build_matrix(len(df_layout), cm_real)
        
        cal = AutoCalibratingDecoder(calibration_window_size=CALIBRATION_WINDOW)
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_pom_real, sigma_gaze=sigma_gaze, use_ovp=True)
        out["full"] = evaluate_word_and_recovery(targets, idx, drift[1], DRIFT_Y)

        # 2) w/o CM — uniform prior (both emissions and transitions)
        t_pom_uniform = pom_builder.build_matrix(len(df_layout), cm_uniform)
        idx, drift = cal.calibrate_and_decode(
            gaze_seq, word_boxes, cm_uniform, t_pom_uniform, sigma_gaze=sigma_gaze, use_ovp=True
        )
        out["wo_cm"] = evaluate_word_and_recovery(targets, idx, drift[1], DRIFT_Y)


        # 3) w/o POM — rule-based transition
        t_rule = ReadingTransitionMatrix().build_matrix(cm_real, is_L2_reader=is_l2_reader)
        idx, drift = cal.calibrate_and_decode(gaze_seq, word_boxes, cm_real, t_rule, sigma_gaze=sigma_gaze, use_ovp=True)
        out["wo_pom"] = evaluate_word_and_recovery(targets, idx, drift[1], DRIFT_Y)

        # 4) w/o EM — 與 evaluate_population_ablation 對齊: Kalman baseline
        idx_k = StandardKalmanDecoder().decode(gaze_seq, word_boxes)
        out["wo_em"] = evaluate_word_and_recovery(targets, idx_k, 0.0, DRIFT_Y)
        out["wo_em"] = (out["wo_em"][0], 0.0)

        # 5) w/o Temp — nearest box (逐點)
        idx_nb = NearestBoundingBoxDecoder().decode(gaze_seq, word_boxes)
        out["wo_temp"] = evaluate_word_and_recovery(targets, idx_nb, 0.0, DRIFT_Y)
        out["wo_temp"] = (out["wo_temp"][0], 0.0)

    return out


def process_one_subject(
    subject_key: str,
    meta: Dict[str, Any],
    pop_root: Path,
    error_log: Path,
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[float, float]]]]:
    """
    聚合單一受試者所有 trial。若某 trial 失敗則記錄並跳過。
    回傳 (checkpoint_row, raw_lists) — raw_lists 供除錯/可選驗證。
    """
    lang = meta["lang"]
    sub_id = meta["subject_id"]
    paths: List[Path] = meta["paths"]
    accum: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)

    for p in paths:
        rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        try:
            trial_metrics = run_five_variants_on_trial(p, pop_root)
            for vk, pair in trial_metrics.items():
                accum[vk].append(pair)
        except Exception as exc:  # noqa: BLE001
            log_trial_error(
                error_log,
                f"[trial_error] subject={subject_key} path={rel} err={type(exc).__name__}: {exc}",
            )
            logging.exception("Trial 失敗: %s", rel)

    # 成功 trial 數 (各 variant 同步累積，長度應一致)
    n_ok = len(accum["full"])
    if n_ok == 0:
        logging.warning("受試者 %s 無任何成功 trial，仍寫入 checkpoint 以避免無限重試；請修復資料後刪除 interim 對應列。", subject_key)

    row: Dict[str, Any] = {
        "subject_key": subject_key,
        "lang": lang,
        "subject_id": sub_id,
        "n_trials_ok": n_ok,
    }
    for vk in VARIANT_INTERNAL:
        pairs = accum[vk]
        if pairs:
            accs = [x[0] for x in pairs]
            recs = [x[1] for x in pairs]
            row[f"{vk}_word_acc_mean"] = float(np.mean(accs))
            row[f"{vk}_traj_rec_mean"] = float(np.mean(recs))
        else:
            row[f"{vk}_word_acc_mean"] = float("nan")
            row[f"{vk}_traj_rec_mean"] = float("nan")

    return row, dict(accum)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(v[ok], weights=w[ok]))


def micro_aggregate_from_interim(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    依語組與變體做加權平均 (權重 = n_trials_ok)，與 trial 數池化一致。
    Global Trajectory Recovery 亦為跨所有受試者 trial 數加權。
    """
    rows = []
    for pub, vk in zip(VARIANT_PUBLICATION_KEYS, VARIANT_INTERNAL):
        acc_col = f"{vk}_word_acc_mean"
        rec_col = f"{vk}_traj_rec_mean"
        sub_l1 = interim_df[interim_df["lang"] == "L1"]
        sub_l2 = interim_df[interim_df["lang"] == "L2"]
        w1 = sub_l1["n_trials_ok"].clip(lower=0).astype(float).to_numpy()
        w2 = sub_l2["n_trials_ok"].clip(lower=0).astype(float).to_numpy()
        l1_acc = _weighted_mean(sub_l1[acc_col].to_numpy(), w1)
        l2_acc = _weighted_mean(sub_l2[acc_col].to_numpy(), w2)
        w_all = interim_df["n_trials_ok"].clip(lower=0).astype(float).to_numpy()
        glob_rec = _weighted_mean(interim_df[rec_col].to_numpy(), w_all)
        rows.append(
            {
                "Model_Variant": pub,
                "L1_Word_Acc": l1_acc,
                "L2_Word_Acc": l2_acc,
                "Global_Trajectory_Recovery_Rate": glob_rec,
            }
        )
    return pd.DataFrame(rows)


def print_latex_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Model Variant & L1 Word Acc (\%) & L2 Word Acc (\%) & Global Trajectory Recovery Rate (\%) \\",
        r"\hline",
    ]
    for _, r in summary.iterrows():
        name = str(r["Model_Variant"]).replace("_", r"\_")
        lines.append(
            f"{name} & {r['L1_Word_Acc']:.2f} & {r['L2_Word_Acc']:.2f} & {r['Global_Trajectory_Recovery_Rate']:.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\caption{NeurIPS full-population ablation (N=37).}", r"\label{tab:neurips_ablation_n37}", r"\end{table}"])
    print("\n".join(lines))


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def main() -> int:
    set_deterministic(GLOBAL_SEED)
    setup_logging(PIPELINE_LOG)
    logging.info("LexiGaze NeurIPS full pipeline 啟動 (seed=%s)", GLOBAL_SEED)

    pop_root = POPULATION_ROOT
    if not pop_root.is_dir():
        logging.error("找不到 population 根目錄: %s", pop_root)
        logging.error("請先執行 scripts/geco/data/batch_extract_features.py 產出 features.csv。")
        return 1

    trial_paths = discover_trial_feature_files(pop_root)
    if not trial_paths:
        logging.error("未找到任何 features.csv (於 %s)", pop_root)
        return 1

    groups = group_paths_by_subject(trial_paths, pop_root)
    n_subjects = count_unique_subjects(groups)
    if n_subjects != EXPECTED_N_SUBJECTS:
        logging.warning(
            "偵測到受試者人數為 %s，預期為 %s。請確認資料是否已完整匯出。",
            n_subjects,
            EXPECTED_N_SUBJECTS,
        )

    completed = load_completed_subject_keys(INTERIM_CSV)
    logging.info("已完成受試者 (checkpoint): %d / %d", len(completed), n_subjects)

    ordered_keys = sorted(groups.keys(), key=lambda k: (groups[k]["lang"], groups[k]["subject_id"]))
    for subject_key in tqdm(ordered_keys, desc="Subjects"):
        if subject_key in completed:
            continue
        row, _raw = process_one_subject(subject_key, groups[subject_key], pop_root, ERROR_LOG)
        append_subject_row(INTERIM_CSV, row)
        completed.add(subject_key)
        logging.info(
            "完成受試者 %s (lang=%s, trials_ok=%s)",
            subject_key,
            row["lang"],
            row["n_trials_ok"],
        )

    interim_df = pd.read_csv(INTERIM_CSV)
    interim_df = interim_df.drop_duplicates(subset=["subject_key"], keep="last")
    if len(interim_df) < n_subjects:
        logging.warning("interim 列數 (%s) 小於受試者數 (%s)，可能尚有失敗未寫入。", len(interim_df), n_subjects)

    summary = micro_aggregate_from_interim(interim_df)
    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(FINAL_CSV, index=False)
    logging.info("已寫入最終結果: %s", FINAL_CSV)

    print("\n--- LaTeX table (paste into manuscript) ---\n")
    print_latex_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
