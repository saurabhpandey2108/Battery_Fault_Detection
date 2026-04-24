"""
Tsinghua NCM811 ISC dataset loader.

Each CSV is UTF-8-BOM encoded with Chinese headers. Two 5-column blocks are
concatenated horizontally and separated by one unnamed empty column:

    LEFT  (cols 0-4) : CHARGE phase    — 测试时间/Sec, 电流/A, 容量/Ah, SOC|DOD/%, 电压/V
    col 5            : empty separator
    RIGHT (cols 6-10): DISCHARGE phase — same five columns (pandas suffixes them with .1)

Sampling rate is exactly 1.0 Hz.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd


# Chinese → English column map for the LEFT (charge) block
_ZH_COLS = {
    "测试时间/Sec": "time_s",
    "电流/A": "current_A",
    "容量/Ah": "capacity_Ah",
    "SOC|DOD/%": "soc_pct",
    "电压/V": "voltage_V",
}


def load_tsinghua_csv(filepath: str) -> dict:
    """Read one Tsinghua CSV and return two English-named DataFrames.

    Returns
    -------
    {"charge_df": DataFrame, "discharge_df": DataFrame}
        Each has columns: time_s, current_A, capacity_Ah, soc_pct, voltage_V.
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    # The two blocks: LEFT has the original Chinese names, RIGHT has them
    # with a `.1` suffix (or `.1`-like) appended by pandas for duplicates.
    left_cols = list(_ZH_COLS.keys())
    right_cols = [f"{c}.1" for c in left_cols]

    missing_left = [c for c in left_cols if c not in df.columns]
    missing_right = [c for c in right_cols if c not in df.columns]
    if missing_left or missing_right:
        raise ValueError(
            f"Unexpected column layout in {filepath}. "
            f"Missing left {missing_left}, missing right {missing_right}. "
            f"Available columns: {list(df.columns)}"
        )

    charge_df = df[left_cols].copy()
    charge_df.columns = [_ZH_COLS[c] for c in left_cols]
    charge_df = charge_df.dropna().reset_index(drop=True)

    discharge_df = df[right_cols].copy()
    discharge_df.columns = [_ZH_COLS[c] for c in left_cols]
    discharge_df = discharge_df.dropna().reset_index(drop=True)

    # Force numeric types; anything non-numeric becomes NaN and gets dropped.
    for sub in (charge_df, discharge_df):
        for col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
    charge_df = charge_df.dropna().reset_index(drop=True)
    discharge_df = discharge_df.dropna().reset_index(drop=True)

    return {"charge_df": charge_df, "discharge_df": discharge_df}


# Filename parser. Examples:
#   ISC_BD_0.5CC_DST_10ohm.csv        — DST discharge, BD = healthy
#   ISC_CS_0.5CC_DST_1000ohm.csv      — DST discharge, CS = faulty (real resistor)
#   ISC_BD_0.5CC_0.5CD_100ohm.csv     — CC discharge at 0.5C, BD = healthy
_FILENAME_RE = re.compile(
    r"^ISC_(?P<class_>BD|CS)_(?P<charge_rate>\d+(?:\.\d+)?)CC_"
    r"(?P<discharge_mode>DST|\d+(?:\.\d+)?CD)_(?P<ohm>\d+)ohm\.csv$"
)


def parse_filename(filename: str) -> dict:
    """Decode an ISC_*.csv filename into its metadata fields."""
    base = os.path.basename(filename)
    m = _FILENAME_RE.match(base)
    if not m:
        raise ValueError(f"Unrecognized Tsinghua filename: {base}")
    return {
        "class_": m.group("class_"),
        "charge_rate": float(m.group("charge_rate")),
        "discharge_mode": m.group("discharge_mode"),
        "ohm": int(m.group("ohm")),
    }


def _build_name(class_: str, charge_rate: float, discharge_mode: str, ohm: int) -> str:
    # Preserve the corpus formatting: 0.5 and 1.0 (one decimal), int resistance.
    rate_str = f"{charge_rate:g}"  # 0.5 → "0.5", 1.0 → "1"
    if "." not in rate_str:
        rate_str = f"{rate_str}.0"
    return f"ISC_{class_}_{rate_str}CC_{discharge_mode}_{ohm}ohm.csv"


def _phase1_folder(root: str, discharge_mode: str, class_: str) -> str:
    sub = "NCM811_NORMAL_TEST" if class_ == "BD" else "NCM811_ISC_TEST"
    mode_dir = "DST" if discharge_mode == "DST" else "CC"
    return os.path.join(root, sub, mode_dir)


def list_phase1_pairs(root: str,
                      charge_rate: float = 0.5,
                      discharge_mode: str = "DST",
                      ohms: Optional[list[int]] = None) -> list[tuple[str, str, int]]:
    """Enumerate matched (BD, CS) file pairs for Phase 1.

    Returns a list of (healthy_file, faulty_file, ohm_value) tuples for the
    requested charge_rate + discharge_mode, sharing the same resistance index.
    """
    if ohms is None:
        ohms = [10, 20, 30, 50, 100, 200, 300, 400, 500,
                600, 700, 800, 900, 1000]

    bd_dir = _phase1_folder(root, discharge_mode, "BD")
    cs_dir = _phase1_folder(root, discharge_mode, "CS")

    pairs: list[tuple[str, str, int]] = []
    for ohm in ohms:
        bd_name = _build_name("BD", charge_rate, discharge_mode, ohm)
        cs_name = _build_name("CS", charge_rate, discharge_mode, ohm)
        bd_path = os.path.join(bd_dir, bd_name)
        cs_path = os.path.join(cs_dir, cs_name)
        if os.path.isfile(bd_path) and os.path.isfile(cs_path):
            pairs.append((bd_path, cs_path, ohm))
    return pairs


def list_phase1_files(root: str,
                      charge_rate: float = 0.5,
                      discharge_mode: str = "DST",
                      class_: str = "BD",
                      ohms: Optional[list[int]] = None) -> list[str]:
    """List all files of one class for Phase 1 (useful for PINN training on BD only)."""
    pairs = list_phase1_pairs(root, charge_rate=charge_rate,
                              discharge_mode=discharge_mode, ohms=ohms)
    idx = 0 if class_ == "BD" else 1
    return [p[idx] for p in pairs]
