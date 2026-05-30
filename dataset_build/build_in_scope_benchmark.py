"""Build benchmark records from a paper JSONL file.
It groups journals by WoS category and constructs balanced benchmark samples."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from typing import Any


INPUT_JSONL = "YOUR_INPUT_JSONL_PATH"
RAW_OUTPUT_FILE = "YOUR_RAW_OUTPUT_JSON_PATH"
BALANCED_OUTPUT_FILE = "YOUR_BALANCED_OUTPUT_JSON_PATH"
COUNTS_OUTPUT_FILE = "YOUR_COUNTS_OUTPUT_CSV_PATH"

RANDOM_SEED = 42

PERCENTILE_INITIAL_HIGH = 0.25
PERCENTILE_INITIAL_LOW = 0.25
PERCENTILE_STEP = 0.05
PERCENTILE_MIN = 0.0
PERCENTILE_INITIAL_ADJUST_STEP = 0.05
PERCENTILE_INITIAL_ADJUST_MIN = 0.01
PERCENTILE_INITIAL_ADJUST_MAX = 1.0
PERCENTILE_INITIAL_IMBALANCE_RATIO = 1.5
PERCENTILE_INITIAL_ADJUST_SKIP_MIN_KEEP = 200
SUBJECT_PERCENTILE_OVERRIDES = {
    "DERMATOLOGY": (0.35, 0.35),
    "MATERIALS SCIENCE, CERAMICS": (0.35, 0.35),
    "MYCOLOGY": (0.50, 1.00),
    "OPHTHALMOLOGY": (0.35, 0.35),
}
SUBJECT_DISABLE_INITIAL_REBALANCE = {"MYCOLOGY"}

TOP_GROUP = "Top"
BOTTOM_GROUP = "Bottom"
DATA_TYPES = ("Real_Submission", "High_to_Low", "Low_to_High")


def _norm_journal_name(name: str) -> str:
    if name is None:
        return ""
    text = str(name).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"n/a", "na", "none", "null"}:
                return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_papers_by_wos(jsonl_path: str) -> dict[str, dict[str, Any]]:
    """
    Load papers from JSONL and group them by:
      WoS Categories -> journal -> {meta, papers}

    Only journals with SSI_Group in {Top, Bottom} are kept.
    """
    grouped: dict[str, dict[str, Any]] = {}
    skipped_missing_core = 0
    skipped_non_top_bottom = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            doi = str(item.get("DOI", "")).strip()
            wos_category = str(item.get("WoS Categories", "")).strip()
            broad_category = str(item.get("Category", "")).strip()
            ssi_group = str(item.get("SSI_Group", "")).strip()
            raw_name = (
                str(item.get("journal", "")).strip()
                or str(item.get("Source Title", "")).strip()
            )

            if not doi or not wos_category or not raw_name:
                skipped_missing_core += 1
                continue
            if ssi_group not in {TOP_GROUP, BOTTOM_GROUP}:
                skipped_non_top_bottom += 1
                continue

            wos_bucket = grouped.setdefault(
                wos_category,
                {
                    "Category": broad_category,
                    "norm_to_name": {},
                    "journals": {},
                },
            )

            norm_name = _norm_journal_name(raw_name)
            canonical_name = wos_bucket["norm_to_name"].setdefault(norm_name, raw_name)
            journals = wos_bucket["journals"]

            paper = {
                "doi": doi,
                "title": item.get("Article Title", "") or "",
                "abstract": item.get("Abstract", "") or "",
                "keywords": item.get("Keywords", "") or "",
                "journal": canonical_name,
                "subject": wos_category,
                "Category": broad_category,
                "citations": _safe_int(item.get("Times Cited, All Databases", 0), 0),
                "Publication Year": item.get("Publication Year"),
            }

            if canonical_name not in journals:
                journals[canonical_name] = {
                    "meta": {
                        "name": canonical_name,
                        "JIF": _safe_float(item.get("JIF", 0.0), 0.0),
                        "JIF_Quartile": str(item.get("JIF_Quartile", "N/A") or "N/A"),
                        "aim_scope": item.get("aim_scope", "") or "",
                        "h5-index": _safe_int(item.get("h5-index", 0), 0),
                        "subject": wos_category,
                        "Category": broad_category,
                        "SSI_Group": ssi_group,
                    },
                    "papers": [],
                }
            else:
                meta = journals[canonical_name]["meta"]
                if not meta.get("JIF"):
                    meta["JIF"] = _safe_float(item.get("JIF", 0.0), 0.0)
                if meta.get("JIF_Quartile") in {"", "N/A"}:
                    meta["JIF_Quartile"] = str(item.get("JIF_Quartile", "N/A") or "N/A")
                if not meta.get("aim_scope"):
                    meta["aim_scope"] = item.get("aim_scope", "") or ""
                if not meta.get("h5-index"):
                    meta["h5-index"] = _safe_int(item.get("h5-index", 0), 0)
                if not meta.get("Category"):
                    meta["Category"] = broad_category
                if not meta.get("subject"):
                    meta["subject"] = wos_category

            journals[canonical_name]["papers"].append(paper)

    return grouped


def find_percentile_separation(
    papers_high: list[dict[str, Any]],
    papers_low: list[dict[str, Any]],
    initial_percent_high: float = 0.30,
    initial_percent_low: float = 0.60,
    step: float = 0.05,
    min_percent: float = 0.0,
    enable_initial_rebalance: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None, int]:
    if not papers_high or not papers_low:
        return [], [], None, 0
    if not (0 < initial_percent_high <= 1.0):
        raise ValueError(
            f"initial_percent_high must be in (0, 1], got {initial_percent_high}"
        )
    if not (0 < initial_percent_low <= 1.0):
        raise ValueError(
            f"initial_percent_low must be in (0, 1], got {initial_percent_low}"
        )
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    high_sorted = sorted(papers_high, key=lambda p: p["citations"], reverse=True)
    low_sorted = sorted(papers_low, key=lambda p: p["citations"])

    percent_high = initial_percent_high
    percent_low = initial_percent_low
    n_high = len(high_sorted)
    n_low = len(low_sorted)
    eps = 1e-9

    # If the two sides are highly imbalanced under the initial percentiles,
    # bias the smaller side upward and the larger side downward before shrinking.
    adjust_rounds = 0
    if enable_initial_rebalance:
        while True:
            est_high = max(1, min(n_high, math.ceil(n_high * percent_high)))
            est_low = max(1, min(n_low, math.ceil(n_low * percent_low)))
            larger = max(est_high, est_low)
            smaller = min(est_high, est_low)
            if smaller <= 0:
                break
            if smaller > PERCENTILE_INITIAL_ADJUST_SKIP_MIN_KEEP:
                break
            if larger / smaller <= PERCENTILE_INITIAL_IMBALANCE_RATIO:
                break

            next_high = percent_high
            next_low = percent_low
            if est_high > est_low:
                next_high = max(
                    PERCENTILE_INITIAL_ADJUST_MIN,
                    round(percent_high - PERCENTILE_INITIAL_ADJUST_STEP, 10),
                )
                next_low = min(
                    PERCENTILE_INITIAL_ADJUST_MAX,
                    round(percent_low + PERCENTILE_INITIAL_ADJUST_STEP, 10),
                )
            else:
                next_high = min(
                    PERCENTILE_INITIAL_ADJUST_MAX,
                    round(percent_high + PERCENTILE_INITIAL_ADJUST_STEP, 10),
                )
                next_low = max(
                    PERCENTILE_INITIAL_ADJUST_MIN,
                    round(percent_low - PERCENTILE_INITIAL_ADJUST_STEP, 10),
                )

            if next_high == percent_high and next_low == percent_low:
                break

            percent_high = next_high
            percent_low = next_low
            adjust_rounds += 1
            if adjust_rounds >= 100:
                break

    while percent_high > min_percent + eps and percent_low > min_percent + eps:
        keep_high = max(1, min(n_high, math.ceil(n_high * percent_high)))
        keep_low = max(1, min(n_low, math.ceil(n_low * percent_low)))
        high_keep = high_sorted[:keep_high]
        low_keep = low_sorted[:keep_low]

        min_high = min(p["citations"] for p in high_keep)
        max_low = max(p["citations"] for p in low_keep)
        used_high = round(percent_high * 100, 2)
        used_low = round(percent_low * 100, 2)

        if min_high > max_low:
            return high_keep, low_keep, int(max_low), len(high_keep) + len(low_keep)

        percent_high -= step
        percent_low -= step

    return [], [], None, 0


def get_option_label(index: int) -> str:
    label = ""
    while True:
        label = chr(ord("A") + index % 26) + label
        index = index // 26 - 1
        if index < 0:
            break
    return label


def build_candidate_package(
    actual_journal: str,
    all_journal_names: list[str],
    journal_meta_map: dict[str, dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    if actual_journal not in journal_meta_map:
        raise ValueError(f"actual journal missing from meta map: {actual_journal}")

    names = list(all_journal_names)
    rng.shuffle(names)

    candidates: list[dict[str, Any]] = []
    correct_option = ""
    for idx, name in enumerate(names):
        label = get_option_label(idx)
        meta = journal_meta_map.get(name, {})
        candidates.append(
            {
                "option": label,
                "journal_name": name,
                "JIF": meta.get("JIF", 0.0),
                "JIF_Quartile": meta.get("JIF_Quartile", "N/A"),
                "h5-index": meta.get("h5-index", 0),
                "aim_scope": meta.get("aim_scope", ""),
            }
        )
        if name == actual_journal:
            correct_option = label

    return {
        "candidate_count": len(candidates),
        "candidates": candidates,
        "correct_option": correct_option,
    }


def _record_sort_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("subject", "")),
        str(record.get("data_type", "")),
        str(record.get("doi", "")),
        str(record.get("simulation_setting", {}).get("target_journal_name", "")),
    )


def create_record(
    paper: dict[str, Any],
    target_meta: dict[str, Any],
    data_type: str,
    candidate_package: dict[str, Any],
) -> dict[str, Any]:
    if data_type == "Real_Submission":
        decision = "Send for Review"
        reason = "Matches Journal Caliber"
    elif data_type == "High_to_Low":
        decision = "Send for Review"
        reason = "Exceeds Journal Caliber"
    elif data_type == "Low_to_High":
        decision = "Desk Reject"
        reason = "Insufficient Novelty/Impact"
    else:
        decision = "Unknown"
        reason = "Unknown"

    return {
        "uid": str(paper.get("doi", "") or "unknown"),
        "doi": paper.get("doi", ""),
        "Category": paper.get("Category", "") or target_meta.get("Category", ""),
        "data_type": data_type,
        "subject": paper.get("subject", "") or target_meta.get("subject", ""),
        "paper_content": {
            "title": paper.get("title", ""),
            "abstract_text": paper.get("abstract", ""),
            "keywords": paper.get("keywords", ""),
        },
        "simulation_setting": {
            "target_journal_name": target_meta["name"],
            "aim_scope": target_meta.get("aim_scope", ""),
            "JIF": target_meta.get("JIF", 0.0),
            "JIF_Quartile": target_meta.get("JIF_Quartile", "N/A"),
            "h5-index": target_meta.get("h5-index", 0),
        },
        "ground_truth": {
            "actual_published_journal": paper.get("journal", ""),
            "expected_decision": decision,
            "expected_reason_label": reason,
        },
        "transfer_recommendation_task": {
            "candidate_count": candidate_package["candidate_count"],
            "candidates": candidate_package["candidates"],
            "correct_option": candidate_package["correct_option"],
        },
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for record in records:
        subject = str(record.get("subject", "")).strip()
        if not subject:
            continue
        entry = summary.setdefault(
            subject,
            {
                "Category": str(record.get("Category", "")).strip(),
                "Real_Submission": 0,
                "High_to_Low": 0,
                "Low_to_High": 0,
            },
        )
        data_type = record.get("data_type")
        if data_type in DATA_TYPES:
            entry[data_type] += 1
        if not entry.get("Category"):
            entry["Category"] = str(record.get("Category", "")).strip()
    return summary


def balance_records_by_subject(
    raw_records: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in raw_records:
        subject = str(record.get("subject", "")).strip()
        if not subject:
            continue
        type_bucket = grouped.setdefault(
            subject, {data_type: [] for data_type in DATA_TYPES}
        )
        data_type = record.get("data_type")
        if data_type in DATA_TYPES:
            type_bucket[data_type].append(record)

    raw_summary = summarize_records(raw_records)
    balanced_records: list[dict[str, Any]] = []

    for subject in sorted(grouped):
        type_bucket = grouped[subject]
        min_count = min(len(type_bucket[data_type]) for data_type in DATA_TYPES)
        for data_type in DATA_TYPES:
            records = sorted(type_bucket[data_type], key=_record_sort_key)
            if min_count == 0:
                selected: list[dict[str, Any]] = []
            elif len(records) <= min_count:
                selected = records
            else:
                chosen_indices = set(rng.sample(range(len(records)), min_count))
                selected = [
                    record for idx, record in enumerate(records) if idx in chosen_indices
                ]
                selected.sort(key=_record_sort_key)
            balanced_records.extend(selected)

    balanced_records.sort(key=_record_sort_key)
    balanced_summary = summarize_records(balanced_records)
    return balanced_records, raw_summary, balanced_summary


def write_records(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_out:
        json.dump(records, f_out, indent=2, ensure_ascii=False)


def write_counts_csv(
    path: str,
    raw_summary: dict[str, dict[str, Any]],
    balanced_summary: dict[str, dict[str, Any]],
) -> None:
    subjects = sorted(set(raw_summary) | set(balanced_summary))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "WoS Categories",
                "Category",
                "raw_Real_Submission",
                "raw_High_to_Low",
                "raw_Low_to_High",
                "raw_min_count",
                "balanced_Real_Submission",
                "balanced_High_to_Low",
                "balanced_Low_to_High",
                "balanced_min_count",
            ],
        )
        writer.writeheader()
        for subject in subjects:
            raw = raw_summary.get(subject, {})
            balanced = balanced_summary.get(subject, {})
            raw_real = int(raw.get("Real_Submission", 0))
            raw_h2l = int(raw.get("High_to_Low", 0))
            raw_l2h = int(raw.get("Low_to_High", 0))
            balanced_real = int(balanced.get("Real_Submission", 0))
            balanced_h2l = int(balanced.get("High_to_Low", 0))
            balanced_l2h = int(balanced.get("Low_to_High", 0))
            writer.writerow(
                {
                    "WoS Categories": subject,
                    "Category": raw.get("Category") or balanced.get("Category", ""),
                    "raw_Real_Submission": raw_real,
                    "raw_High_to_Low": raw_h2l,
                    "raw_Low_to_High": raw_l2h,
                    "raw_min_count": min(raw_real, raw_h2l, raw_l2h),
                    "balanced_Real_Submission": balanced_real,
                    "balanced_High_to_Low": balanced_h2l,
                    "balanced_Low_to_High": balanced_l2h,
                    "balanced_min_count": min(
                        balanced_real, balanced_h2l, balanced_l2h
                    ),
                }
            )


def build_benchmark() -> None:
    if not os.path.exists(INPUT_JSONL):
        print(f"Input JSONL not found: {INPUT_JSONL}")
        return

    grouped = load_papers_by_wos(INPUT_JSONL)
    candidate_rng = random.Random(RANDOM_SEED)
    balance_rng = random.Random(RANDOM_SEED + 1)

    all_records: list[dict[str, Any]] = []
    built_real: set[str] = set()
    built_high_to_low: set[str] = set()
    built_low_to_high: set[str] = set()

    stats = {
        "Real": 0,
        "High2Low": 0,
        "Low2High": 0,
        "pairs_no_separation": 0,
        "pairs_total": 0,
    }

    for wos_category in sorted(grouped):
        bucket = grouped[wos_category]
        broad_category = bucket.get("Category", "")
        journals = bucket["journals"]

        high_names = sorted(
            [
                name
                for name, data in journals.items()
                if data["meta"].get("SSI_Group") == TOP_GROUP
            ]
        )
        low_names = sorted(
            [
                name
                for name, data in journals.items()
                if data["meta"].get("SSI_Group") == BOTTOM_GROUP
            ]
        )

        if not high_names or not low_names:
            continue

        candidate_journal_names = high_names + low_names
        journal_meta_map = {
            name: journals[name]["meta"] for name in candidate_journal_names
        }

        for high_name in high_names:
            for low_name in low_names:
                stats["pairs_total"] += 1
                high_papers = journals[high_name]["papers"]
                low_papers = journals[low_name]["papers"]

                initial_high, initial_low = SUBJECT_PERCENTILE_OVERRIDES.get(
                    wos_category,
                    (PERCENTILE_INITIAL_HIGH, PERCENTILE_INITIAL_LOW),
                )

                high_keep, low_keep, threshold, _score = find_percentile_separation(
                    high_papers,
                    low_papers,
                    initial_percent_high=initial_high,
                    initial_percent_low=initial_low,
                    step=PERCENTILE_STEP,
                    min_percent=PERCENTILE_MIN,
                    enable_initial_rebalance=(
                        wos_category not in SUBJECT_DISABLE_INITIAL_REBALANCE
                    ),
                )

                if threshold is None or not high_keep or not low_keep:
                    stats["pairs_no_separation"] += 1
                    continue

                high_meta = journals[high_name]["meta"]
                low_meta = journals[low_name]["meta"]

                for paper in high_keep:
                    doi = paper["doi"]
                    if doi not in built_high_to_low:
                        pkg = build_candidate_package(
                            high_name,
                            candidate_journal_names,
                            journal_meta_map,
                            candidate_rng,
                        )
                        all_records.append(
                            create_record(paper, low_meta, "High_to_Low", pkg)
                        )
                        built_high_to_low.add(doi)
                        stats["High2Low"] += 1
                    if doi not in built_real:
                        pkg = build_candidate_package(
                            high_name,
                            candidate_journal_names,
                            journal_meta_map,
                            candidate_rng,
                        )
                        all_records.append(
                            create_record(paper, high_meta, "Real_Submission", pkg)
                        )
                        built_real.add(doi)
                        stats["Real"] += 1

                for paper in low_keep:
                    doi = paper["doi"]
                    if doi not in built_low_to_high:
                        pkg = build_candidate_package(
                            low_name,
                            candidate_journal_names,
                            journal_meta_map,
                            candidate_rng,
                        )
                        all_records.append(
                            create_record(paper, high_meta, "Low_to_High", pkg)
                        )
                        built_low_to_high.add(doi)
                        stats["Low2High"] += 1
                    if doi not in built_real:
                        pkg = build_candidate_package(
                            low_name,
                            candidate_journal_names,
                            journal_meta_map,
                            candidate_rng,
                        )
                        all_records.append(
                            create_record(paper, low_meta, "Real_Submission", pkg)
                        )
                        built_real.add(doi)
                        stats["Real"] += 1

    all_records.sort(key=_record_sort_key)
    balanced_records, raw_summary, balanced_summary = balance_records_by_subject(
        all_records, balance_rng
    )

    write_records(RAW_OUTPUT_FILE, all_records)
    write_records(BALANCED_OUTPUT_FILE, balanced_records)
    write_counts_csv(COUNTS_OUTPUT_FILE, raw_summary, balanced_summary)


if __name__ == "__main__":
    build_benchmark()
