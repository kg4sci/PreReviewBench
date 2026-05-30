"""Build within-category Out_of_Scope benchmark records.
It pairs journals across different WoS categories under the same broad category."""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any


PAPERS_JSONL = "YOUR_PAPERS_JSONL_PATH"
IN_SCOPE_BENCHMARK = "YOUR_IN_SCOPE_BENCHMARK_JSON_PATH"
RAW_OUT_SCOPE_FILE = "YOUR_RAW_OUT_OF_SCOPE_JSON_PATH"
SAMPLED_OUT_SCOPE_FILE = "YOUR_SAMPLED_OUT_OF_SCOPE_JSON_PATH"
PAIR_STATS_FILE = "YOUR_PAIR_STATS_CSV_PATH"
FINAL_COUNTS_FILE = "YOUR_FINAL_COUNTS_CSV_PATH"

RANDOM_SEED = 42


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


def get_option_label(index: int) -> str:
    label = ""
    while True:
        label = chr(ord("A") + index % 26) + label
        index = index // 26 - 1
        if index < 0:
            break
    return label


def load_papers_grouped_by_category(jsonl_path: str) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Return:
      Category -> WoS Categories -> journal -> {meta, papers}
    """
    grouped: dict[str, dict[str, dict[str, Any]]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            broad_category = str(item.get("Category", "")).strip()
            subject = str(item.get("WoS Categories", "")).strip()
            journal_name = (
                str(item.get("journal", "")).strip()
                or str(item.get("Source Title", "")).strip()
            )
            doi = str(item.get("DOI", "")).strip()

            if not broad_category or not subject or not journal_name or not doi:
                continue

            subject_bucket = grouped.setdefault(broad_category, {}).setdefault(subject, {})
            journal_data = subject_bucket.setdefault(
                journal_name,
                {
                    "meta": {
                        "name": journal_name,
                        "Category": broad_category,
                        "subject": subject,
                        "JIF": _safe_float(item.get("JIF", 0.0), 0.0),
                        "JIF_Quartile": str(item.get("JIF_Quartile", "N/A") or "N/A"),
                        "h5-index": _safe_int(item.get("h5-index", 0), 0),
                        "SSI": _safe_float(item.get("SSI", 0.0), 0.0),
                        "aim_scope": item.get("aim_scope", "") or "",
                    },
                    "papers": [],
                },
            )

            meta = journal_data["meta"]
            if not meta.get("aim_scope"):
                meta["aim_scope"] = item.get("aim_scope", "") or ""
            if not meta.get("JIF"):
                meta["JIF"] = _safe_float(item.get("JIF", 0.0), 0.0)
            if meta.get("JIF_Quartile") in {"", "N/A"}:
                meta["JIF_Quartile"] = str(item.get("JIF_Quartile", "N/A") or "N/A")
            if not meta.get("h5-index"):
                meta["h5-index"] = _safe_int(item.get("h5-index", 0), 0)
            if not meta.get("SSI"):
                meta["SSI"] = _safe_float(item.get("SSI", 0.0), 0.0)

            journal_data["papers"].append(
                {
                    "doi": doi,
                    "title": item.get("Article Title", "") or "",
                    "abstract": item.get("Abstract", "") or "",
                    "keywords": item.get("Keywords", "") or "",
                    "journal": journal_name,
                }
            )

    return grouped


def split_in_scope_and_low_to_high_counts(
    benchmark_path: str,
) -> tuple[list[dict[str, Any]], Counter]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    in_scope_records = [r for r in records if r.get("data_type") != "Out_of_Scope"]
    low_to_high_counts: Counter = Counter()
    for record in in_scope_records:
        if record.get("data_type") == "Low_to_High":
            low_to_high_counts[str(record.get("subject", "")).strip()] += 1
    return in_scope_records, low_to_high_counts


def filter_overlapping_journals(
    source_db: dict[str, Any], target_db: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    source_norm = {_norm_journal_name(name): name for name in source_db}
    target_norm = {_norm_journal_name(name): name for name in target_db}
    overlap_norms = sorted(set(source_norm) & set(target_norm))
    if not overlap_norms:
        return source_db, target_db, []

    filtered_source = {
        name: data
        for name, data in source_db.items()
        if _norm_journal_name(name) not in overlap_norms
    }
    filtered_target = {
        name: data
        for name, data in target_db.items()
        if _norm_journal_name(name) not in overlap_norms
    }
    overlap_names = [source_norm[norm] for norm in overlap_norms]
    return filtered_source, filtered_target, overlap_names


def find_simulation_target_by_ssi(
    source_meta: dict[str, Any],
    target_db: dict[str, Any],
) -> str | None:
    src_ssi = _safe_float(source_meta.get("SSI", 0.0), 0.0)
    src_jif = _safe_float(source_meta.get("JIF", 0.0), 0.0)
    src_h5 = _safe_int(source_meta.get("h5-index", 0), 0)

    best_name = None
    best_score = None
    for target_name, target_data in target_db.items():
        target_meta = target_data["meta"]
        score = (
            abs(_safe_float(target_meta.get("SSI", 0.0), 0.0) - src_ssi),
            abs(_safe_float(target_meta.get("JIF", 0.0), 0.0) - src_jif),
            abs(_safe_int(target_meta.get("h5-index", 0), 0) - src_h5),
            target_name,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_name = target_name
    return best_name


def build_candidate_package(
    real_name: str,
    all_journal_names: list[str],
    journal_meta_map: dict[str, dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    if real_name not in journal_meta_map:
        raise ValueError(f"real journal missing from meta map: {real_name}")

    names = list(all_journal_names)
    rng.shuffle(names)

    candidates: list[dict[str, Any]] = []
    correct_option = ""
    for idx, name in enumerate(names):
        option = get_option_label(idx)
        meta = journal_meta_map.get(name, {})
        candidates.append(
            {
                "option": option,
                "journal_name": name,
                "subject": meta.get("subject", ""),
                "aim_scope": meta.get("aim_scope", ""),
                "JIF": meta.get("JIF", 0.0),
                "JIF_Quartile": meta.get("JIF_Quartile", "N/A"),
                "h5-index": meta.get("h5-index", 0),
            }
        )
        if name == real_name:
            correct_option = option

    return {
        "candidate_count": len(candidates),
        "candidates": candidates,
        "correct_option": correct_option,
    }


def create_record(
    paper: dict[str, Any],
    source_meta: dict[str, Any],
    target_meta: dict[str, Any],
    candidate_package: dict[str, Any],
) -> dict[str, Any]:
    source_subject = source_meta.get("subject", "")
    target_subject = target_meta.get("subject", "")
    return {
        "uid": str(paper.get("doi", "") or "unknown"),
        "doi": paper.get("doi", ""),
        "Category": source_meta.get("Category", ""),
        "data_type": "Out_of_Scope",
        "direction": f"{source_subject}->{target_subject}",
        "subject": source_subject,
        "paper_content": {
            "title": paper.get("title", ""),
            "abstract_text": paper.get("abstract", ""),
            "keywords": paper.get("keywords", ""),
        },
        "simulation_setting": {
            "target_journal_name": target_meta["name"],
            "target_subject": target_subject,
            "aim_scope": target_meta.get("aim_scope", ""),
            "JIF": target_meta.get("JIF", 0.0),
            "JIF_Quartile": target_meta.get("JIF_Quartile", "N/A"),
            "h5-index": target_meta.get("h5-index", 0),
        },
        "ground_truth": {
            "actual_published_journal": paper.get("journal", ""),
            "actual_published_subject": source_subject,
            "expected_decision": "Desk Reject",
            "expected_reason_label": "Out of scope",
        },
        "transfer_recommendation_task": {
            "candidate_count": candidate_package["candidate_count"],
            "candidates": candidate_package["candidates"],
            "correct_option": candidate_package["correct_option"],
        },
    }


def _record_sort_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("Category", "")),
        str(record.get("subject", "")),
        str(record.get("doi", "")),
        str(record.get("simulation_setting", {}).get("target_journal_name", "")),
    )


def generate_direction_records(
    source_db: dict[str, Any],
    target_db: dict[str, Any],
    all_journal_names: list[str],
    journal_meta_map: dict[str, dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stats = {
        "source_subject": "",
        "target_subject": "",
        "source_journals": len(source_db),
        "target_journals": len(target_db),
        "source_papers": 0,
        "records_built": 0,
    }

    for source_name, source_data in source_db.items():
        source_meta = source_data["meta"]
        target_name = find_simulation_target_by_ssi(source_meta, target_db)
        if not target_name:
            continue
        target_meta = target_db[target_name]["meta"]
        stats["source_subject"] = source_meta.get("subject", "")
        stats["target_subject"] = target_meta.get("subject", "")

        for paper in source_data["papers"]:
            pkg = build_candidate_package(
                real_name=source_name,
                all_journal_names=all_journal_names,
                journal_meta_map=journal_meta_map,
                rng=rng,
            )
            if not pkg["correct_option"]:
                continue
            records.append(create_record(paper, source_meta, target_meta, pkg))
            stats["source_papers"] += 1
            stats["records_built"] += 1

    return records, stats


def sample_out_of_scope_records(
    raw_records: list[dict[str, Any]],
    low_to_high_counts: Counter,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        by_subject[str(record.get("subject", "")).strip()].append(record)

    sampled: list[dict[str, Any]] = []
    sampled_counts: dict[str, int] = {}
    for subject, records in sorted(by_subject.items()):
        records_sorted = sorted(records, key=_record_sort_key)
        wanted = int(low_to_high_counts.get(subject, 0))
        if wanted <= 0:
            sampled_counts[subject] = 0
            continue
        if wanted >= len(records_sorted):
            chosen = records_sorted
        else:
            chosen_indices = set(rng.sample(range(len(records_sorted)), wanted))
            chosen = [
                record for idx, record in enumerate(records_sorted) if idx in chosen_indices
            ]
            chosen.sort(key=_record_sort_key)
        sampled.extend(chosen)
        sampled_counts[subject] = len(chosen)

    sampled.sort(key=_record_sort_key)
    return sampled, sampled_counts


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_pair_stats_csv(
    path: str,
    pair_stats: list[dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Category",
                "source_subject",
                "target_subject",
                "source_journals",
                "target_journals",
                "source_papers",
                "raw_records",
                "low_to_high_target_count",
                "sampled_records",
                "overlap_journals",
            ],
        )
        writer.writeheader()
        for row in pair_stats:
            writer.writerow(row)


def write_final_counts_csv(path: str, records: list[dict[str, Any]]) -> None:
    by_subject: dict[str, Counter] = defaultdict(Counter)
    category_map: dict[str, str] = {}
    for record in records:
        subject = str(record.get("subject", "")).strip()
        if not subject:
            continue
        category_map.setdefault(subject, str(record.get("Category", "")).strip())
        by_subject[subject][str(record.get("data_type", "")).strip()] += 1
        by_subject[subject]["total"] += 1

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "WoS Categories",
                "Category",
                "Real_Submission",
                "High_to_Low",
                "Low_to_High",
                "Out_of_Scope",
                "total",
            ],
        )
        writer.writeheader()
        for subject in sorted(by_subject):
            counter = by_subject[subject]
            writer.writerow(
                {
                    "WoS Categories": subject,
                    "Category": category_map.get(subject, ""),
                    "Real_Submission": counter.get("Real_Submission", 0),
                    "High_to_Low": counter.get("High_to_Low", 0),
                    "Low_to_High": counter.get("Low_to_High", 0),
                    "Out_of_Scope": counter.get("Out_of_Scope", 0),
                    "total": counter.get("total", 0),
                }
            )


def build_within_category_out_of_scope() -> None:
    grouped = load_papers_grouped_by_category(PAPERS_JSONL)
    in_scope_records, low_to_high_counts = split_in_scope_and_low_to_high_counts(
        IN_SCOPE_BENCHMARK
    )

    candidate_rng = random.Random(RANDOM_SEED)
    sample_rng = random.Random(RANDOM_SEED + 1)

    raw_records: list[dict[str, Any]] = []
    pair_stats: list[dict[str, Any]] = []

    for broad_category in sorted(grouped):
        subject_map = grouped[broad_category]
        subjects = sorted(subject_map)
        if len(subjects) < 2:
            continue

        for subject_a, subject_b in combinations(subjects, 2):
            db_a, db_b, overlap_names = filter_overlapping_journals(
                subject_map[subject_a],
                subject_map[subject_b],
            )
            if not db_a or not db_b:
                continue

            all_journal_names = list(db_a.keys()) + list(db_b.keys())
            journal_meta_map = {
                name: data["meta"]
                for db in (db_a, db_b)
                for name, data in db.items()
            }

            records_a2b, stats_a2b = generate_direction_records(
                source_db=db_a,
                target_db=db_b,
                all_journal_names=all_journal_names,
                journal_meta_map=journal_meta_map,
                rng=candidate_rng,
            )
            records_b2a, stats_b2a = generate_direction_records(
                source_db=db_b,
                target_db=db_a,
                all_journal_names=all_journal_names,
                journal_meta_map=journal_meta_map,
                rng=candidate_rng,
            )

            raw_records.extend(records_a2b)
            raw_records.extend(records_b2a)

            pair_stats.append(
                {
                    "Category": broad_category,
                    "source_subject": subject_a,
                    "target_subject": subject_b,
                    "source_journals": stats_a2b["source_journals"],
                    "target_journals": stats_a2b["target_journals"],
                    "source_papers": stats_a2b["source_papers"],
                    "raw_records": stats_a2b["records_built"],
                    "low_to_high_target_count": int(low_to_high_counts.get(subject_a, 0)),
                    "sampled_records": 0,
                    "overlap_journals": "; ".join(overlap_names),
                }
            )
            pair_stats.append(
                {
                    "Category": broad_category,
                    "source_subject": subject_b,
                    "target_subject": subject_a,
                    "source_journals": stats_b2a["source_journals"],
                    "target_journals": stats_b2a["target_journals"],
                    "source_papers": stats_b2a["source_papers"],
                    "raw_records": stats_b2a["records_built"],
                    "low_to_high_target_count": int(low_to_high_counts.get(subject_b, 0)),
                    "sampled_records": 0,
                    "overlap_journals": "; ".join(overlap_names),
                }
            )

    raw_records.sort(key=_record_sort_key)
    sampled_records, sampled_counts = sample_out_of_scope_records(
        raw_records, low_to_high_counts, sample_rng
    )

    for row in pair_stats:
        row["sampled_records"] = int(sampled_counts.get(row["source_subject"], 0))

    merged_records = list(in_scope_records) + sampled_records

    write_json(RAW_OUT_SCOPE_FILE, raw_records)
    write_json(SAMPLED_OUT_SCOPE_FILE, sampled_records)
    write_json(IN_SCOPE_BENCHMARK, merged_records)
    write_pair_stats_csv(PAIR_STATS_FILE, pair_stats)
    write_final_counts_csv(FINAL_COUNTS_FILE, merged_records)


if __name__ == "__main__":
    build_within_category_out_of_scope()
