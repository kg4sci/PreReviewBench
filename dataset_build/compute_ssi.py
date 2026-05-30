"""Compute journal SSI values and write them back to CSV files.
Supports processing a single file or recursively scanning directories."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


GENERATED_COLS = ("R_JIF", "R_h5", "SSI", "SSI_tier_20", "SSI_tier_25")


def _tier_label(ssi: pd.Series, q: float) -> pd.Series:
    """Assign top/mid/bottom labels by within-subject SSI percentile, preserving NaN values."""
    pct = ssi.rank(method="average", pct=True)
    label = pd.Series("mid", index=ssi.index, dtype=object)
    label[pct >= 1 - q] = "top"
    label[pct <= q] = "bottom"
    label[ssi.isna()] = np.nan
    return label


def compute_ssi(
    df: pd.DataFrame,
    jif_col: str = "2024 JIF",
    h5_col: str = "h5-index",
    gamma: float = 1.5,
) -> pd.DataFrame:
    """Append SSI-related columns and sort by SSI descending in an idempotent way."""
    df = df.drop(columns=[c for c in GENERATED_COLS if c in df.columns]).copy()

    jif = pd.to_numeric(df[jif_col], errors="coerce")
    h5 = pd.to_numeric(df[h5_col], errors="coerce")

    # Within-subject percentile ranks (0-100), using average rank for ties.
    r_jif = jif.rank(method="average", pct=True) * 100
    r_h5 = h5.rank(method="average", pct=True) * 100
    ssi = 100 * (np.sqrt(r_jif * r_h5) / 100) ** gamma

    df["R_JIF"] = r_jif.round(2)
    df["R_h5"] = r_h5.round(2)
    df["SSI"] = ssi.round(2)
    df["SSI_tier_20"] = _tier_label(ssi, 0.20)
    df["SSI_tier_25"] = _tier_label(ssi, 0.25)

    df = df.sort_values("SSI", ascending=False, na_position="last").reset_index(drop=True)
    return df


def process_file(
    path: Path,
    jif_col: str,
    h5_col: str,
    gamma: float,
    output: Path | None = None,
) -> tuple[bool, str]:
    """Process a single CSV file and return a success flag plus message."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"Failed to read: {path} -> {e}"

    missing = [c for c in (jif_col, h5_col) if c not in df.columns]
    if missing:
        return False, f"Skipped due to missing columns {missing}: {path}"

    try:
        df_out = compute_ssi(df, jif_col, h5_col, gamma)
    except Exception as e:
        return False, f"Failed to compute SSI: {path} -> {e}"

    out_path = output or path
    df_out.to_csv(out_path, index=False)
    return True, f"OK {path} ({len(df_out)} rows)"


def iter_csvs(inputs: list[Path], pattern: str) -> list[Path]:
    """Expand input paths into a de-duplicated file list."""
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(p.rglob(pattern)))
        elif p.is_file():
            files.append(p)
        else:
            print(f"Warning: path does not exist and was ignored: {p}", file=sys.stderr)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    return uniq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute SSI for journals CSV(s).")
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input CSV file or directory. Directories are scanned recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for single-file mode. Ignored for multi-file or directory mode.",
    )
    parser.add_argument("--pattern", default="*.csv", help="Filename glob used when scanning directories.")
    parser.add_argument("--gamma", type=float, default=1.5, help="Polarization factor. Default is 1.5.")
    parser.add_argument("--jif-col", default="2024 JIF")
    parser.add_argument("--h5-col", default="h5-index")
    args = parser.parse_args(argv)

    files = iter_csvs(args.inputs, args.pattern)
    if not files:
        print("No CSV files were found.", file=sys.stderr)
        return 1

    single_mode = len(files) == 1 and args.inputs[0].is_file()
    output = args.output if single_mode else None

    ok = fail = 0
    for f in files:
        try:
            success, msg = process_file(f, args.jif_col, args.h5_col, args.gamma, output)
        except Exception:
            success, msg = False, f"Unhandled exception: {f}\n{traceback.format_exc()}"
        if not success:
            print(msg, file=sys.stderr)
        if success:
            ok += 1
        else:
            fail += 1

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
