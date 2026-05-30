"""Evaluate PreReviewBench prediction files.

This public version removes machine-specific defaults, exposes a CLI, and keeps
plotting dependencies optional so the core metrics can still run in lean
environments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

try:
    from scipy.stats import kendalltau
except Exception:
    kendalltau = None


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESULTS_PATH = os.getenv("RESULTS_PATH", "YOUR_RESULTS_JSON_PATH")
DEFAULT_BENCHMARK_PATH = os.getenv(
    "BENCHMARK_PATH",
    str(REPO_ROOT / "dataset" / "benchmark2025.json"),
)
DEFAULT_OUT_CSV = os.getenv("OUT_CSV", "")
DEFAULT_FIG_DIR = os.getenv("FIG_DIR", "")
DEFAULT_NDCG_K = [1, 3, 5]

DECISIONS = ["Send for Review", "Desk Reject"]
QUALITY_LABELS = ["Exceeds Journal Caliber", "Matches Journal Caliber", "Invalid"]
REASON_LABELS = ["Insufficient Novelty/Impact", "Out of scope", "Other", "N/A"]


def _to_list(values: Any) -> List[str]:
    if isinstance(values, pd.Series):
        return values.astype(str).tolist()
    if isinstance(values, np.ndarray):
        return [str(item) for item in values.tolist()]
    return [str(item) for item in values]


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    if path.endswith(".jsonl"):
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        return records

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in results file: {path}")
    return [item for item in data if isinstance(item, dict)]


def accuracy_score(y_true: Any, y_pred: Any) -> float:
    yt = _to_list(y_true)
    yp = _to_list(y_pred)
    if not yt:
        return 0.0
    n = min(len(yt), len(yp))
    if n == 0:
        return 0.0
    return float(sum(1 for index in range(n) if yt[index] == yp[index]) / n)


def confusion_matrix(y_true: Any, y_pred: Any, labels: List[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    yt = _to_list(y_true)
    yp = _to_list(y_pred)
    n = min(len(yt), len(yp))
    for index in range(n):
        true_label = yt[index]
        pred_label = yp[index]
        if true_label in label_to_index and pred_label in label_to_index:
            matrix[label_to_index[true_label], label_to_index[pred_label]] += 1
    return matrix


def _f1_for_label(y_true: List[str], y_pred: List[str], label: str) -> float:
    tp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == label and pred == label)
    fp = sum(1 for actual, pred in zip(y_true, y_pred) if actual != label and pred == label)
    fn = sum(1 for actual, pred in zip(y_true, y_pred) if actual == label and pred != label)
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1_score(
    y_true: Any,
    y_pred: Any,
    pos_label: Optional[str] = None,
    average: str = "binary",
    labels: Optional[List[str]] = None,
    zero_division: int = 0,
) -> float:
    del zero_division
    yt = _to_list(y_true)
    yp = _to_list(y_pred)
    n = min(len(yt), len(yp))
    yt = yt[:n]
    yp = yp[:n]
    if n == 0:
        return 0.0

    if average == "binary":
        label = pos_label if pos_label is not None else yt[0]
        return float(_f1_for_label(yt, yp, label))
    if average == "macro":
        label_list = labels if labels is not None else sorted(set(yt) | set(yp))
        if not label_list:
            return 0.0
        values = [_f1_for_label(yt, yp, label) for label in label_list]
        return float(sum(values) / len(values))
    raise ValueError(f"Unsupported average: {average}")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_decision(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"send for review", "send_to_review", "send"}:
        return "Send for Review"
    if text in {"desk reject", "reject", "desk_reject"}:
        return "Desk Reject"
    return "Unknown"


def _normalize_quality(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"exceeds journal caliber", "superior quality"}:
        return "Exceeds Journal Caliber"
    if text in {"matches journal caliber", "acceptable quality"}:
        return "Matches Journal Caliber"
    return "Invalid"


def _normalize_reason_label(value: Any) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    if not text:
        return "N/A"
    lowered = text.lower()
    if lowered in {"null", "n/a"}:
        return "N/A"
    if lowered in {
        "insufficient novelty/impact",
        "insufficient novelty impact",
        "lack of novelty/impact",
        "lack of novelty impact",
    }:
        return "Insufficient Novelty/Impact"
    if lowered == "out of scope":
        return "Out of scope"
    if lowered == "other":
        return "Other"
    return "Other"


def _normalize_letter(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    match = re.search(r"[A-Z]", text)
    return match.group(0) if match else None


def _make_sample_id(
    uid_or_doi: Any, data_type: Any, subject: Any, target_journal_name: Any
) -> Optional[str]:
    parts = [uid_or_doi, data_type, subject, target_journal_name]
    if any(part is None for part in parts):
        return None
    normalized = [str(part).strip() for part in parts]
    if any(not part for part in normalized):
        return None
    return "||".join(normalized)


def _benchmark_sample_id(item: Dict[str, Any]) -> Optional[str]:
    setting = item.get("simulation_setting", {})
    target_journal_name = None
    if isinstance(setting, dict):
        target_journal_name = setting.get("target_journal_name")
    return _make_sample_id(
        item.get("uid") or item.get("doi"),
        item.get("data_type"),
        item.get("subject"),
        target_journal_name,
    )


def _result_sample_id(item: Dict[str, Any]) -> Optional[str]:
    if item.get("sample_id"):
        return str(item["sample_id"]).strip()
    return _make_sample_id(
        item.get("uid") or item.get("doi"),
        item.get("data_type"),
        item.get("subject"),
        item.get("target_journal_name"),
    )


def _extract_transfer_ranking(response: Dict[str, Any]) -> Optional[List[str]]:
    ranking = response.get("Transfer_Ranking")
    if ranking is None:
        return None

    letters: List[str] = []
    if isinstance(ranking, list):
        for item in ranking:
            letter = _normalize_letter(item)
            if letter:
                letters.append(letter)
    elif isinstance(ranking, str):
        letters = re.findall(r"[A-Z]", ranking.upper())
    else:
        return None

    if not letters:
        return None

    deduped: List[str] = []
    seen: set[str] = set()
    for letter in letters:
        if letter in seen:
            continue
        seen.add(letter)
        deduped.append(letter)
    return deduped if deduped else None


def _is_valid_permutation(rank_list: List[str], candidate_letters: List[str]) -> bool:
    if not rank_list or not candidate_letters:
        return False
    return set(rank_list) == set(candidate_letters) and len(rank_list) == len(
        candidate_letters
    )


def _project_rank_list(rank_list: List[str], candidate_letters: List[str]) -> List[str]:
    if not rank_list or not candidate_letters:
        return []
    candidate_set = set(candidate_letters)
    return [item for item in rank_list if item in candidate_set]


def _candidate_option(candidate: Any) -> Optional[str]:
    if isinstance(candidate, dict):
        return _normalize_letter(candidate.get("option"))
    if isinstance(candidate, str):
        return _normalize_letter(candidate.split(".", 1)[0] if "." in candidate else candidate)
    return None


def _candidate_name(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("journal_name", ""))
    if isinstance(candidate, str):
        return candidate.split(". ", 1)[-1].strip() if ". " in candidate else candidate
    return ""


def _jaccard_sim(text_a: str, text_b: str) -> float:
    words_a = set(re.findall(r"\w+", str(text_a).lower()))
    words_b = set(re.findall(r"\w+", str(text_b).lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _text_similarity_ranking(
    paper_content: Dict[str, Any], candidates: List[Any]
) -> List[str]:
    text = " ".join(
        [
            str(paper_content.get("title", "")),
            str(paper_content.get("abstract_text", "")),
            str(paper_content.get("keywords", "")),
        ]
    )
    scored: List[tuple[str, float]] = []
    for candidate in candidates:
        option = _candidate_option(candidate)
        if option is None:
            continue
        name = _candidate_name(candidate)
        similarity = _jaccard_sim(name, text)
        scored.append((option, similarity))
    scored.sort(key=lambda item: -item[1])
    return [item[0] for item in scored]


def load_records(
    results_path: str, benchmark_path: Optional[str] = None
) -> pd.DataFrame:
    results = _load_json_or_jsonl(results_path)

    benchmark_by_sample_id: Dict[str, Dict[str, Any]] = {}
    if benchmark_path and os.path.exists(benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as handle:
            benchmark = json.load(handle)
        for item in benchmark:
            if not isinstance(item, dict):
                continue
            sample_id = _benchmark_sample_id(item)
            if sample_id:
                benchmark_by_sample_id[sample_id] = item

    rows: List[Dict[str, Any]] = []
    for item in results:
        uid = item.get("uid", "")
        sample_id = _result_sample_id(item)
        response = item.get("model_response", {})
        if not isinstance(response, dict):
            continue

        ground_truth = item.get("ground_truth", {})
        if not isinstance(ground_truth, dict):
            ground_truth = {}
        gt_decision = _normalize_decision(ground_truth.get("expected_decision"))
        gt_reason = _normalize_reason_label(ground_truth.get("expected_reason_label"))
        gt_quality = "N/A"
        if gt_decision == "Send for Review":
            gt_quality = _normalize_quality(ground_truth.get("expected_reason_label"))

        pred_decision = _normalize_decision(response.get("Primary_Decision"))
        pred_quality = _normalize_quality(response.get("Quality_Assessment"))
        pred_reason = _normalize_reason_label(response.get("Rejection_Category"))
        transfer_rank_list = _extract_transfer_ranking(response)
        confidence = _safe_float(response.get("Confidence_Score"))

        benchmark_item = benchmark_by_sample_id.get(sample_id or "", {})
        candidates = benchmark_item.get("transfer_recommendation_task", {}).get(
            "candidates", []
        )
        paper_content = benchmark_item.get("paper_content", {})
        transfer_gt = item.get("transfer_gt")
        if transfer_gt is None and benchmark_item:
            transfer_gt = benchmark_item.get("transfer_recommendation_task", {}).get(
                "correct_option"
            )

        rows.append(
            {
                "sample_id": sample_id,
                "uid": uid,
                "Category": item.get("Category", "Unknown"),
                "subject": item.get("subject", "Unknown"),
                "data_type": item.get("data_type", "Unknown"),
                "target_journal_name": item.get("target_journal_name"),
                "gt_decision": gt_decision,
                "pred_decision": pred_decision,
                "gt_quality": gt_quality,
                "pred_quality": pred_quality,
                "gt_reason_class": gt_reason if gt_decision == "Desk Reject" else "N/A",
                "pred_reason_class": pred_reason,
                "gt_transfer": _normalize_letter(transfer_gt),
                "transfer_rank_list": transfer_rank_list,
                "confidence": confidence,
                "paper_content": paper_content,
                "candidates": candidates,
                "candidate_letters": [
                    _candidate_option(candidate)
                    for candidate in candidates
                    if _candidate_option(candidate) is not None
                ],
                "benchmark_matched": bool(benchmark_item),
                "is_error_resp": "Error" in response,
            }
        )

    return pd.DataFrame(rows)


def compute_decision_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    yt = df["gt_decision"]
    yp = df["pred_decision"].fillna("Unknown")
    valid = yt.isin(DECISIONS)
    yt = yt[valid]
    yp = yp[valid]
    if len(yt) == 0:
        return {"Count": 0}

    parsed = yp.isin(DECISIONS)
    parsed_yt = yt[parsed]
    parsed_yp = yp[parsed]
    observed_labels = sorted(pd.Series(yt).dropna().unique().tolist())
    parsed_observed_labels = sorted(pd.Series(parsed_yt).dropna().unique().tolist())

    strict_acc = accuracy_score(yt, yp)
    parsed_acc = accuracy_score(parsed_yt, parsed_yp) if len(parsed_yt) > 0 else 0.0
    strict_macro_f1 = (
        f1_score(yt, yp, average="macro", labels=observed_labels, zero_division=0)
        if observed_labels
        else 0.0
    )
    parsed_macro_f1 = (
        f1_score(
            parsed_yt,
            parsed_yp,
            average="macro",
            labels=parsed_observed_labels,
            zero_division=0,
        )
        if len(parsed_yt) > 0 and parsed_observed_labels
        else 0.0
    )

    send_mask = yt == "Send for Review"
    reject_mask = yt == "Desk Reject"

    send_total = int(send_mask.sum())
    reject_total = int(reject_mask.sum())
    send_correct = int(((yt == "Send for Review") & (yp == "Send for Review")).sum())
    reject_correct = int(((yt == "Desk Reject") & (yp == "Desk Reject")).sum())
    good_paper_rejected = int(
        ((yt == "Send for Review") & (yp == "Desk Reject")).sum()
    )
    bad_paper_passed = int(((yt == "Desk Reject") & (yp == "Send for Review")).sum())
    unknown_count = int((~parsed).sum())
    send_unknown = int(((yt == "Send for Review") & (~parsed)).sum())
    reject_unknown = int(((yt == "Desk Reject") & (~parsed)).sum())

    send_recall = send_correct / send_total if send_total > 0 else 0.0
    reject_recall = reject_correct / reject_total if reject_total > 0 else 0.0
    balanced_acc = (
        (send_recall + reject_recall) / 2.0
        if send_total > 0 and reject_total > 0
        else None
    )
    good_paper_rejection_rate = (
        good_paper_rejected / send_total if send_total > 0 else None
    )
    bad_paper_pass_through_rate = (
        bad_paper_passed / reject_total if reject_total > 0 else None
    )

    return {
        "Accuracy": strict_acc,
        "Accuracy (Parsed Only)": parsed_acc,
        "Macro-F1": strict_macro_f1,
        "Macro-F1 (Parsed Only)": parsed_macro_f1,
        "Balanced_Accuracy": balanced_acc,
        "Good_Paper_Rejection_Rate": good_paper_rejection_rate,
        "Bad_Paper_Pass_Through_Rate": bad_paper_pass_through_rate,
        "Send_Recall": send_recall,
        "Reject_Recall": reject_recall,
        "Send_Correct_Count": send_correct,
        "Reject_Correct_Count": reject_correct,
        "Good_Paper_Rejected_Count": good_paper_rejected,
        "Bad_Paper_Passed_Count": bad_paper_passed,
        "Parsed_Decision_Count": int(len(parsed_yt)),
        "Parsed_Decision_Rate": float(len(parsed_yt) / len(yt)),
        "Unknown_Prediction_Count": unknown_count,
        "Send_Unknown_Count": send_unknown,
        "Reject_Unknown_Count": reject_unknown,
        "Count": int(len(yt)),
    }


def compute_subtask_metrics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    send_df = df[df["gt_decision"] == "Send for Review"].copy()
    send_df = send_df[
        send_df["gt_quality"].isin(["Exceeds Journal Caliber", "Matches Journal Caliber"])
    ]
    if len(send_df) > 0:
        metric_labels = sorted(send_df["gt_quality"].dropna().unique().tolist())
        plot_labels = metric_labels.copy()
        if send_df["pred_quality"].eq("Invalid").any():
            plot_labels.append("Invalid")
        results.append(
            {
                "Task": "Quality (Exceeds vs Matches)",
                "Accuracy": accuracy_score(send_df["gt_quality"], send_df["pred_quality"]),
                "Macro-F1": f1_score(
                    send_df["gt_quality"],
                    send_df["pred_quality"],
                    average="macro",
                    labels=metric_labels,
                    zero_division=0,
                ),
                "Invalid_Prediction_Rate": float(
                    (~send_df["pred_quality"].isin(metric_labels)).mean()
                ),
                "Count": int(len(send_df)),
                "kind": "quality",
                "metric_labels": metric_labels,
                "plot_labels": plot_labels,
                "df": send_df,
            }
        )

    reject_df = df[df["gt_decision"] == "Desk Reject"].copy()
    reject_df = reject_df[reject_df["gt_reason_class"].isin(REASON_LABELS)]
    if len(reject_df) > 0:
        metric_labels = sorted(reject_df["gt_reason_class"].dropna().unique().tolist())
        plot_labels = metric_labels.copy()
        extras = sorted(
            set(reject_df["pred_reason_class"].dropna().unique().tolist())
            - set(metric_labels)
        )
        plot_labels.extend(extras)
        results.append(
            {
                "Task": "Reason (Reject Cause)",
                "Accuracy": accuracy_score(
                    reject_df["gt_reason_class"], reject_df["pred_reason_class"]
                ),
                "Macro-F1": f1_score(
                    reject_df["gt_reason_class"],
                    reject_df["pred_reason_class"],
                    average="macro",
                    labels=metric_labels,
                    zero_division=0,
                ),
                "Invalid_Prediction_Rate": float(
                    (~reject_df["pred_reason_class"].isin(metric_labels)).mean()
                ),
                "Count": int(len(reject_df)),
                "kind": "reason",
                "metric_labels": metric_labels,
                "plot_labels": plot_labels,
                "df": reject_df,
            }
        )

    return results


def compute_joint_subtask_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    valid = df[df["gt_decision"].isin(DECISIONS)].copy()
    if valid.empty:
        return {"Count": 0}

    joint_correct = []
    for _, row in valid.iterrows():
        if row["gt_decision"] == "Send for Review":
            is_correct = (
                row["pred_decision"] == "Send for Review"
                and row["pred_quality"] == row["gt_quality"]
            )
        else:
            is_correct = (
                row["pred_decision"] == "Desk Reject"
                and row["pred_reason_class"] == row["gt_reason_class"]
            )
        joint_correct.append(bool(is_correct))

    valid["joint_correct"] = joint_correct
    send_df = valid[valid["gt_decision"] == "Send for Review"]
    reject_df = valid[valid["gt_decision"] == "Desk Reject"]

    return {
        "Task": "Joint Editorial Correctness",
        "Accuracy": float(valid["joint_correct"].mean()),
        "Send_Joint_Accuracy": float(send_df["joint_correct"].mean())
        if not send_df.empty
        else 0.0,
        "Reject_Joint_Accuracy": float(reject_df["joint_correct"].mean())
        if not reject_df.empty
        else 0.0,
        "Count": int(len(valid)),
    }


def compute_transfer_metrics(
    df: pd.DataFrame, ndcg_k_list: List[int]
) -> Dict[str, Any]:
    reject_df = df[
        (df["gt_decision"] == "Desk Reject") & (df["gt_transfer"].notna())
    ].copy()
    if len(reject_df) == 0:
        return {
            "MRR": 0.0,
            **{f"NDCG@{k}": 0.0 for k in ndcg_k_list},
            **{f"Hit@{k}": 0.0 for k in ndcg_k_list},
            "Strict_MRR": 0.0,
            **{f"Strict_NDCG@{k}": 0.0 for k in ndcg_k_list},
            **{f"Strict_Hit@{k}": 0.0 for k in ndcg_k_list},
            "MRR (Valid Only)": 0.0,
            **{f"NDCG@{k} (Valid Only)": 0.0 for k in ndcg_k_list},
            "Kendall_vs_text_sim": None,
            "Ranking_Output_Count": 0,
            "Ranking_Output_Rate": 0.0,
            "Nonempty_Invalid_Ranking_Count": 0,
            "Nonempty_Invalid_Ranking_Rate": 0.0,
            "Valid_Ranking_Count": 0,
            "Valid_Permutation_Rate": 0.0,
            "Count": 0,
        }

    mrr_sum = 0.0
    ndcg_sums = {k: 0.0 for k in ndcg_k_list}
    hit_counts = {k: 0 for k in ndcg_k_list}
    strict_mrr_sum = 0.0
    strict_ndcg_sums = {k: 0.0 for k in ndcg_k_list}
    strict_hit_counts = {k: 0 for k in ndcg_k_list}
    valid_only_mrr_sum = 0.0
    valid_only_ndcg_sums = {k: 0.0 for k in ndcg_k_list}
    kendall_taus: List[float] = []
    ranking_output_count = 0
    valid_ranking_count = 0
    nonempty_invalid_ranking_count = 0

    for _, row in reject_df.iterrows():
        gt = row["gt_transfer"]
        rank_list = row["transfer_rank_list"]
        if not isinstance(rank_list, list):
            rank_list = []
        candidates = row.get("candidates", []) or []
        candidate_letters = row.get("candidate_letters", []) or []
        has_ranking = len(rank_list) > 0
        if has_ranking:
            ranking_output_count += 1
        projected_rank_list = _project_rank_list(rank_list, candidate_letters)
        valid_perm = _is_valid_permutation(projected_rank_list, candidate_letters)
        if valid_perm:
            valid_ranking_count += 1
        elif has_ranking:
            nonempty_invalid_ranking_count += 1

        rank = None
        try:
            rank = 1 + projected_rank_list.index(gt)
            mrr_sum += 1.0 / rank
        except ValueError:
            rank = None

        strict_rank = None
        strict_rank_list = projected_rank_list if valid_perm else []
        try:
            strict_rank = 1 + strict_rank_list.index(gt)
            strict_mrr_sum += 1.0 / strict_rank
            valid_only_mrr_sum += 1.0 / strict_rank
        except ValueError:
            strict_rank = None

        rel = [1 if item == gt else 0 for item in projected_rank_list]
        strict_rel = [1 if item == gt else 0 for item in strict_rank_list]
        for k in ndcg_k_list:
            dcg_k = sum(rel[index] / np.log2(index + 2) for index in range(min(k, len(rel))))
            ndcg_sums[k] += dcg_k
            if rank is not None and rank <= k:
                hit_counts[k] += 1

            strict_dcg_k = sum(
                strict_rel[index] / np.log2(index + 2)
                for index in range(min(k, len(strict_rel)))
            )
            strict_ndcg_sums[k] += strict_dcg_k
            if valid_perm:
                valid_only_ndcg_sums[k] += strict_dcg_k
            if strict_rank is not None and strict_rank <= k:
                strict_hit_counts[k] += 1

        if (
            projected_rank_list
            and kendalltau is not None
            and candidates
            and row.get("paper_content")
        ):
            text_rank = _text_similarity_ranking(row["paper_content"], candidates)
            if text_rank:
                common = [item for item in projected_rank_list if item in text_rank]
                if len(common) >= 2:
                    rank_a = [1 + projected_rank_list.index(item) for item in common]
                    rank_b = [1 + text_rank.index(item) for item in common]
                    tau, _ = kendalltau(rank_a, rank_b)
                    if not np.isnan(tau):
                        kendall_taus.append(float(tau))

    n = len(reject_df)
    metrics: Dict[str, Any] = {
        "MRR": mrr_sum / n if n else 0.0,
        "Strict_MRR": strict_mrr_sum / n if n else 0.0,
        "Count": int(n),
        "Ranking_Output_Count": int(ranking_output_count),
        "Ranking_Output_Rate": ranking_output_count / n if n else 0.0,
        "Nonempty_Invalid_Ranking_Count": int(nonempty_invalid_ranking_count),
        "Nonempty_Invalid_Ranking_Rate": nonempty_invalid_ranking_count / n if n else 0.0,
        "Valid_Ranking_Count": int(valid_ranking_count),
        "Valid_Permutation_Rate": valid_ranking_count / n if n else 0.0,
        "Kendall_vs_text_sim": float(np.mean(kendall_taus)) if kendall_taus else None,
        "MRR (Valid Only)": valid_only_mrr_sum / valid_ranking_count
        if valid_ranking_count
        else 0.0,
    }
    for k in ndcg_k_list:
        metrics[f"Hit@{k}"] = hit_counts[k] / n if n else 0.0
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / n if n else 0.0
        metrics[f"Strict_Hit@{k}"] = strict_hit_counts[k] / n if n else 0.0
        metrics[f"Strict_NDCG@{k}"] = strict_ndcg_sums[k] / n if n else 0.0
        metrics[f"NDCG@{k} (Valid Only)"] = (
            valid_only_ndcg_sums[k] / valid_ranking_count
            if valid_ranking_count
            else 0.0
        )
    return metrics


def compute_confidence_metrics(
    df: pd.DataFrame, threshold_list: Optional[List[int]] = None
) -> Dict[str, Any]:
    threshold_list = threshold_list or [8, 9]
    valid = df[df["gt_decision"].isin(DECISIONS)].copy()
    valid = valid[valid["confidence"].notna()].copy()
    if valid.empty:
        return {"Count": 0}

    valid["is_correct"] = (valid["gt_decision"] == valid["pred_decision"]).astype(int)
    valid["conf_prob"] = valid["confidence"].clip(1, 10).astype(float) / 10.0

    brier = float(np.mean((valid["conf_prob"] - valid["is_correct"]) ** 2))

    ece = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        high = low + 0.1
        if high >= 1.0:
            mask = (valid["conf_prob"] >= low) & (valid["conf_prob"] <= high)
        else:
            mask = (valid["conf_prob"] >= low) & (valid["conf_prob"] < high)
        if not mask.any():
            continue
        bucket = valid[mask]
        acc = float(bucket["is_correct"].mean())
        avg_conf = float(bucket["conf_prob"].mean())
        ece += abs(acc - avg_conf) * (len(bucket) / len(valid))

    metrics: Dict[str, Any] = {
        "Count": int(len(valid)),
        "Coverage": float(len(valid) / len(df)) if len(df) > 0 else 0.0,
        "Brier_Score": brier,
        "ECE": float(ece),
    }
    for threshold in threshold_list:
        subset = valid[valid["confidence"] >= threshold]
        metrics[f"Coverage@Conf>={threshold}"] = (
            float(len(subset) / len(df)) if len(df) > 0 else 0.0
        )
        metrics[f"Accuracy@Conf>={threshold}"] = (
            float(subset["is_correct"].mean()) if not subset.empty else 0.0
        )
    return metrics


def plot_confidence_error(df: pd.DataFrame, fig_dir: str, min_bin_count: int = 5) -> None:
    if plt is None:
        return
    tmp = df[df["confidence"].notna()].copy()
    if tmp.empty:
        return
    tmp["is_error"] = (tmp["gt_decision"] != tmp["pred_decision"]).astype(int)
    tmp["conf_bin"] = tmp["confidence"].round().clip(1, 10).astype(int)
    aggregated = tmp.groupby("conf_bin", observed=True).agg(
        error_rate=("is_error", "mean"),
        count=("sample_id", "count"),
    )
    aggregated = aggregated[aggregated["count"] >= min_bin_count].sort_index()
    if aggregated.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(aggregated)), aggregated["error_rate"], color="#e67e22", alpha=0.9)
    plt.xticks(range(len(aggregated)), [str(item) for item in aggregated.index])
    plt.ylim(0, 1.0)
    plt.xlabel("Confidence (1-10)")
    plt.ylabel("Error Rate")
    plt.title("Confidence-Error Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confidence_error_plot.png"))
    plt.close()


def plot_all(
    metrics_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    transfer_metrics: Dict[str, Any],
    subtask_results: List[Dict[str, Any]],
    fig_dir: str,
    ndcg_k_list: List[int],
) -> bool:
    if plt is None or sns is None:
        print(
            "Plotting skipped because matplotlib or seaborn is not available. "
            "Install them to generate figures."
        )
        return False

    sns.set_theme(style="whitegrid", font_scale=1.05)

    overall = metrics_df[metrics_df["Level"] == "1_Overall"]
    if not overall.empty:
        row = overall.iloc[0]
        values = [
            row.get("Accuracy", 0.0),
            row.get("Macro-F1", 0.0),
            row.get("Balanced_Accuracy", 0.0),
            row.get("Parsed_Decision_Rate", 0.0),
        ]
        bars = ["Accuracy", "MacroF1", "BalancedAcc", "ParsedRate"]
        plt.figure(figsize=(7, 5))
        plt.bar(bars, values, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
        plt.ylim(0, 1.05)
        for index, value in enumerate(values):
            plt.text(index, min(value + 0.02, 1.02), f"{value:.3f}", ha="center")
        plt.title("Task 1: Decision Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "task1_decision_metrics.png"))
        plt.close()

    valid = raw_df[
        raw_df["gt_decision"].isin(DECISIONS) & raw_df["pred_decision"].isin(DECISIONS)
    ]
    if not valid.empty:
        matrix = confusion_matrix(
            valid["gt_decision"], valid["pred_decision"], labels=DECISIONS
        )
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred: Send", "Pred: Reject"],
            yticklabels=["True: Send", "True: Reject"],
        )
        plt.title("Task 1: Decision Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "task1_confusion_matrix.png"))
        plt.close()

    for subtask_result in subtask_results:
        if subtask_result.get("kind") == "quality":
            yt = subtask_result["df"]["gt_quality"]
            yp = subtask_result["df"]["pred_quality"]
            labels = subtask_result["plot_labels"]
            display_labels = [
                "Exceeds"
                if label == "Exceeds Journal Caliber"
                else "Matches"
                if label == "Matches Journal Caliber"
                else label
                for label in labels
            ]
        elif subtask_result.get("kind") == "reason":
            yt = subtask_result["df"]["gt_reason_class"]
            yp = subtask_result["df"]["pred_reason_class"]
            labels = subtask_result["plot_labels"]
            display_labels = labels
        else:
            continue

        matrix = confusion_matrix(yt, yp, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=display_labels,
            yticklabels=display_labels,
        )
        plt.xticks(rotation=15)
        plt.yticks(rotation=0)
        plt.title(f"Task 2: {subtask_result['Task']} Confusion Matrix")
        plt.tight_layout()
        filename = (
            subtask_result["Task"]
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        plt.savefig(os.path.join(fig_dir, f"task2_confusion_{filename}.png"))
        plt.close()

    if transfer_metrics.get("Count", 0) > 0:
        bars = ["MRR"] + [f"Hit@{k}" for k in ndcg_k_list] + [
            "RankOutputRate",
            "ValidPermRate",
        ]
        values = [transfer_metrics.get("MRR", 0.0)]
        values.extend(transfer_metrics.get(f"Hit@{k}", 0.0) for k in ndcg_k_list)
        values.append(transfer_metrics.get("Ranking_Output_Rate", 0.0))
        values.append(transfer_metrics.get("Valid_Permutation_Rate", 0.0))
        plt.figure(figsize=(9, 5))
        plt.bar(bars, values, color="#4c78a8")
        plt.ylim(0, 1.05)
        for index, value in enumerate(values):
            plt.text(index, min(value + 0.02, 1.02), f"{value:.3f}", ha="center", fontsize=9)
        plt.title("Task 3: Transfer Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "task3_transfer_metrics.png"))
        plt.close()

        bars = ["MRR"] + [f"NDCG@{k}" for k in ndcg_k_list] + [
            "RankOutputRate",
            "ValidPermRate",
        ]
        values = [transfer_metrics.get("MRR", 0.0)]
        values.extend(transfer_metrics.get(f"NDCG@{k}", 0.0) for k in ndcg_k_list)
        values.append(transfer_metrics.get("Ranking_Output_Rate", 0.0))
        values.append(transfer_metrics.get("Valid_Permutation_Rate", 0.0))
        plt.figure(figsize=(9, 5))
        plt.bar(bars, values, color="#4c78a8")
        plt.ylim(0, 1.05)
        for index, value in enumerate(values):
            plt.text(index, min(value + 0.02, 1.02), f"{value:.3f}", ha="center", fontsize=9)
        plt.title("Task 3: Transfer Ranking Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "task3_transfer_metrics_ndcg.png"))
        plt.close()

    plot_confidence_error(raw_df, fig_dir)
    return True


def _parse_int_list(raw: str) -> List[int]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction files for PreReviewBench."
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_PATH,
        help="Path to a JSON or JSONL results file.",
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the benchmark JSON file used for metadata joins.",
    )
    parser.add_argument(
        "--out-csv",
        default=DEFAULT_OUT_CSV,
        help="Where to save the metrics CSV. Defaults next to the results file.",
    )
    parser.add_argument(
        "--fig-dir",
        default=DEFAULT_FIG_DIR,
        help="Directory for figures. Defaults next to the results file.",
    )
    parser.add_argument(
        "--ndcg-k",
        default="1,3,5",
        help="Comma-separated cutoff values for Hit@k and NDCG@k.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation even if plotting libraries are available.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_path = args.results
    benchmark_path = args.benchmark
    ndcg_k_list = _parse_int_list(args.ndcg_k)

    if not results_path or results_path == "YOUR_RESULTS_JSON_PATH":
        raise ValueError("Please provide a real results file path via --results.")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results_stem = Path(results_path).stem
    results_parent = str(Path(results_path).resolve().parent)
    out_csv = args.out_csv or os.path.join(results_parent, f"{results_stem}_metrics.csv")
    fig_dir = args.fig_dir or os.path.join(results_parent, f"{results_stem}_figures")

    Path(out_csv).resolve().parent.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        Path(fig_dir).resolve().mkdir(parents=True, exist_ok=True)

    print(f"Loading results: {results_path}")
    df = load_records(results_path, benchmark_path=benchmark_path)
    if df.empty:
        raise RuntimeError(
            "No evaluable records were found. Check the result structure and file contents."
        )

    print(
        "Benchmark matched:",
        f"{int(df['benchmark_matched'].sum())}/{len(df)}",
    )

    task1 = compute_decision_metrics(df)
    rows: List[Dict[str, Any]] = [
        {
            "Level": "1_Overall",
            "Type": "All Data",
            "Task": "Decision (Send/Reject)",
            **task1,
        }
    ]

    for data_type in sorted(df["data_type"].dropna().unique().tolist()):
        subset = df[df["data_type"] == data_type]
        if not subset.empty:
            rows.append(
                {
                    "Level": "2_Scenario",
                    "Type": data_type,
                    "Task": "Decision (Send/Reject)",
                    **compute_decision_metrics(subset),
                }
            )

    subtask_results = compute_subtask_metrics(df)
    for subtask_result in subtask_results:
        rows.append(
            {
                "Level": "3_SubTask",
                "Type": subtask_result["Task"],
                "Task": subtask_result["Task"],
                "Accuracy": subtask_result["Accuracy"],
                "Macro-F1": subtask_result["Macro-F1"],
                "Invalid_Prediction_Rate": subtask_result.get("Invalid_Prediction_Rate"),
                "Count": subtask_result["Count"],
            }
        )

    joint_metrics = compute_joint_subtask_metrics(df)
    rows.append({"Level": "3_Joint", "Type": "Joint", **joint_metrics})

    transfer_metrics = compute_transfer_metrics(df, ndcg_k_list)
    rows.append(
        {
            "Level": "4_Transfer",
            "Type": "Transfer Ranking",
            "Task": "Transfer Ranking",
            **transfer_metrics,
        }
    )

    confidence_metrics = compute_confidence_metrics(df)
    rows.append(
        {
            "Level": "5_Confidence",
            "Type": "Calibration",
            "Task": "Decision Confidence",
            **confidence_metrics,
        }
    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_csv, index=False)

    print("\nEvaluation Summary:")
    print(metrics_df.to_string(index=False))
    print(f"\nCSV saved: {out_csv}")

    if args.no_plots:
        print("Figure generation skipped (--no-plots).")
    else:
        figures_written = plot_all(
            metrics_df,
            df,
            transfer_metrics,
            subtask_results,
            fig_dir,
            ndcg_k_list,
        )
        if figures_written:
            print(f"Figures saved: {fig_dir}")

    if transfer_metrics.get("Kendall_vs_text_sim") is not None:
        print(
            "Kendall vs text similarity ranking:",
            f"{transfer_metrics['Kendall_vs_text_sim']:.3f}",
        )


if __name__ == "__main__":
    main()
