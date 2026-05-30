"""Run single-stage direct desk-triage inference on benchmark entries.
The script supports resumable JSONL output and optional sharded execution."""

import argparse
import json
import os
import time
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


API_KEY = os.getenv("LLM_API_KEY", "YOUR_LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "YOUR_LLM_SERVICE_BASE_URL")
MODEL = os.getenv("LLM_MODEL_NAME", "YOUR_LLM_MODEL_NAME")

CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)

DATASET_PATH = "YOUR_DATASET_JSON_PATH"
OUTPUT_RESULTS_PATH = "YOUR_OUTPUT_RESULTS_JSON_PATH"
MAX_API_RETRIES = int(os.getenv("INFERENCE_MAX_API_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("INFERENCE_RETRY_BACKOFF_SECONDS", "10"))
WRITE_FINAL_JSON = os.getenv("INFERENCE_WRITE_FINAL_JSON", "0") == "1"


def create_system_prompt(setting: Dict[str, Any]) -> str:
    target_journal_name = setting.get("target_journal_name", "Unknown Journal")
    jif = setting.get("JIF", "N/A")
    quartile = setting.get("JIF_Quartile", "N/A")
    h5_index = setting.get("h5-index", "N/A")
    scope = setting.get("aim_scope", "N/A")

    return f"""You are the Expert Editor-in-Chief for the academic journal: "{target_journal_name}".
Your current task is STRICTLY the initial "Desk Triage" (Pre-review stage).

CRITICAL BOUNDARY: Do NOT conduct a deep, full peer review of experimental details. Assume the manuscript has already passed formatting and ethics checks. Your role is to decide whether this manuscript is a plausible and worthwhile candidate for external peer review at this journal.

### [YOUR JOURNAL PROFILE]
Internalize your journal's profile. Evaluate the manuscript against these metadata:
- Journal Name: {target_journal_name}
- Impact Factor & Quartile: JIF {jif} ({quartile})
- h5-index: {h5_index}
- Aims & Scope: {scope}

### [EDITORIAL PERSPECTIVE]
You are NOT choosing the best journal in the world for this paper. You are deciding whether THIS journal should spend reviewer attention on it.
Your task is to decide whether the manuscript should proceed to external peer review, not whether it should ultimately be accepted for publication.
You only have title, abstract, and keywords. Do NOT invent fatal weaknesses merely because full methods, experiments, data, or controls are not shown in the abstract.

### [DECISION WORKFLOW]
Make your decision in this order:

STEP 1: Scope Gate
Decide whether the manuscript is plausibly within the journal's scope, considering:
- core topic
- methodological orientation
- likely readership and community fit
- whether the journal's audience would reasonably see the paper as relevant

STEP 2: Editorial Priority Gate
If the manuscript is within scope, decide whether it appears sufficiently interesting, coherent, and potentially impactful to justify sending it to external reviewers for this journal.
Ask whether you would be comfortable sending this manuscript to 2-3 external reviewers for this journal, based only on the abstract-level evidence available.

STEP 3: Map to Final Decision
- If it is within scope and worth sending to reviewers: choose "Send for Review".
- If it is within scope but does not seem strong enough for external review at this journal: choose "Desk Reject" with "Insufficient Novelty/Impact".
- If it is clearly outside the journal's scope, audience, or methodological remit: choose "Desk Reject" with "Out of Scope".

### [LABEL DEFINITIONS]
- Choose "Send for Review" if you judge the manuscript sufficiently aligned and promising to merit external peer review at this journal.
- Choose "Desk Reject" with "Out of Scope" if you judge the main issue to be lack of fit with the journal's topical mission, methodological orientation, or intended readership.
- Choose "Desk Reject" with "Insufficient Novelty/Impact" if you judge the manuscript to be broadly within scope but not compelling enough for reviewer allocation at this journal.
- Choose "Matches Journal Caliber" if, after deciding to send the manuscript for review, you judge it to fit the journal's usual level and editorial ambition.
- Choose "Exceeds Journal Caliber" if, after deciding to send the manuscript for review, you judge it to be stronger, broader, or more consequential than the journal's typical reviewed papers, while still being a plausible fit for the journal and its readership.

### [IMPORTANT INTERPRETIVE NOTES]
1. Make the primary decision using ONLY the target journal profile and the manuscript itself.
2. Do not treat a mismatch in prestige alone as a mismatch in scope.
3. Use "Out of Scope" to reflect fit; use "Insufficient Novelty/Impact" to reflect editorial priority within scope.

### [POST-DECISION TRANSFER RULE]
Candidate journals are provided only for post-rejection transfer handling. Their presence does NOT imply that the manuscript should be rejected.
Do NOT use the candidate list to infer the correct primary decision, quality assessment, or rejection category.
Only after you have formed the primary desk-triage decision may you use the candidate list, and only if you chose "Desk Reject".

### [TRANSFER RANKING RULE]
If and only if you choose "Desk Reject", rank candidate journals as you would recommend them to the authors as realistic transfer destinations.
Use scope, readership, methodological fit, selectivity, JIF, quartile, h5-index, subject, and aims & scope as supporting signals.
Return a COMPLETE ranking of ALL candidate journals from best fit to worst fit using ONLY the option letters.
If you choose "Send for Review", set "Transfer_Ranking" to null.

### [JSON OUTPUT SCHEMA]
You MUST respond with ONLY valid JSON. No reasoning process, no markdown formatting, no introductory or concluding text.
{{
  "Primary_Decision": "Send for Review" | "Desk Reject",
  "Confidence_Score": <Integer: 1-10>,
  "Quality_Assessment": "Matches Journal Caliber" | "Exceeds Journal Caliber" | null,
  "Rejection_Category": "Out of Scope" | "Insufficient Novelty/Impact" | null,
  "Transfer_Ranking": ["<Option_Letter>", "<Option_Letter>", ...] | null
}}"""


def _format_candidates(
    candidates: List[Dict[str, Any]], include_aim_scope: bool
) -> str:
    lines = []
    for cand in candidates:
        option = cand.get("option", "?")
        name = cand.get("journal_name", "Unknown")
        subject = cand.get("subject")
        jif = cand.get("JIF", "N/A")
        quartile = cand.get("JIF_Quartile", "N/A")
        h5_index = cand.get("h5-index", "N/A")

        meta_parts = [f"JIF: {jif}", f"Quartile: {quartile}", f"h5-index: {h5_index}"]
        if subject:
            meta_parts.append(f"Subject: {subject}")
        lines.append(f"[{option}] {name} ({', '.join(meta_parts)})")
        if include_aim_scope and cand.get("aim_scope"):
            lines.append(f"    Aims & Scope: {cand.get('aim_scope').strip()}")
    return "\n".join(lines)


def create_user_prompt(
    paper: Dict[str, Any],
    candidates_raw: List[Dict[str, Any]],
    include_candidate_aim_scope: bool = True,
) -> str:
    candidate_list_string = _format_candidates(
        candidates_raw, include_candidate_aim_scope
    )
    return f"""Please perform the Desk Triage on the following submitted manuscript.

### [SUBMITTED MANUSCRIPT]
- Title: {paper.get('title', 'N/A')}
- Keywords: {paper.get('keywords', 'N/A')}
- Abstract: {paper.get('abstract_text', 'N/A')}

### [POST-DECISION TRANSFER OPTIONS]
Consult this section only after you have formed the primary desk-triage decision, and only if that decision is "Desk Reject".
{candidate_list_string}
"""


def _parse_model_output(model_output_text: str) -> Dict[str, Any]:
    text = (model_output_text or "").strip()
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    try:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            model_decision = json.loads(text[start : end + 1])
            if "Transfer_Ranking" in model_decision and isinstance(
                model_decision["Transfer_Ranking"], str
            ):
                model_decision["Transfer_Ranking"] = [
                    model_decision["Transfer_Ranking"]
                ]
            return model_decision
        return {"Error": "No JSON found", "Raw": model_output_text}
    except json.JSONDecodeError:
        return {"Error": "JSON Parse Error", "Raw": model_output_text}


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _write_json_atomic(path: str, data: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _append_jsonl(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_json_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _load_jsonl_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    results: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                results.append(item)
    return results


def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        sample_id = _build_sample_id(item)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        deduped.append(item)
    return deduped


def _load_existing_results(output_path: str) -> List[Dict[str, Any]]:
    json_results = _load_json_results(output_path)
    jsonl_results = _load_jsonl_results(f"{output_path}.jsonl")
    base = jsonl_results if len(jsonl_results) > len(json_results) else json_results
    return _dedupe_results(base)


def _is_fatal_api_error(message: str) -> bool:
    text = message.lower()
    fatal_markers = [
        "invalid_api_key",
        "incorrect api key",
        "insufficient_quota",
        "insufficient quota",
        "insufficient balance",
        "insufficient credit",
        "quota exceeded",
        "insufficient funds",
        "bill",
        "billing",
        "payment required",
        "permission denied",
        "forbidden",
        "401",
        "403",
    ]
    return any(marker in text for marker in fatal_markers)


def _is_retryable_api_error(message: str) -> bool:
    text = message.lower()
    retryable_markers = [
        "connection error",
        "connection reset",
        "connection aborted",
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "temporarily unavailable",
        "server error",
        "overloaded",
        "network",
    ]
    return any(marker in text for marker in retryable_markers)


def _call_model(messages: List[Dict[str, str]]) -> tuple[Dict[str, Any], bool]:
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            res = CLIENT.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                stream=False,
            )
            return _parse_model_output(res.choices[0].message.content or ""), False
        except Exception as e:
            raw = str(e)
            error = {"Error": "API Failed", "Raw": raw, "Attempt": attempt}
            if _is_fatal_api_error(raw):
                return error, True
            if attempt < MAX_API_RETRIES and _is_retryable_api_error(raw):
                wait_seconds = RETRY_BACKOFF_SECONDS * attempt
                time.sleep(wait_seconds)
                continue
            return error, False
    return {"Error": "API Failed", "Raw": "Unknown retry exit"}, False


def _build_sample_id(entry: Dict[str, Any]) -> str:
    target_journal_name = (
        entry.get("target_journal_name")
        or entry.get("simulation_setting", {}).get("target_journal_name")
        or ""
    )
    return "||".join(
        [
            str(entry.get("uid", "")).strip(),
            str(entry.get("data_type", "")).strip(),
            str(entry.get("subject", "")).strip(),
            str(target_journal_name).strip(),
        ]
    )


def _build_shard_output_path(output_path: str, num_shards: int, shard_id: int) -> str:
    if num_shards <= 1:
        return output_path
    if output_path.endswith(".json"):
        return output_path[:-5] + f".shard{shard_id + 1}of{num_shards}.json"
    return output_path + f".shard{shard_id + 1}of{num_shards}"


def _select_shard(
    dataset: List[Dict[str, Any]], num_shards: int, shard_id: int
) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return dataset
    return [entry for idx, entry in enumerate(dataset) if idx % num_shards == shard_id]


def _parse_extra_paths(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _load_finished_sample_ids(paths: List[str]) -> set[str]:
    sample_ids: set[str] = set()
    for path in paths:
        if not path:
            continue
        if path.endswith(".jsonl"):
            for item in _load_jsonl_results(path):
                sample_ids.add(_build_sample_id(item))
            continue
        for item in _load_json_results(path):
            sample_ids.add(_build_sample_id(item))
        for item in _load_jsonl_results(f"{path}.jsonl"):
            sample_ids.add(_build_sample_id(item))
    return sample_ids


def _make_status(
    dataset_path: str,
    output_path: str,
    total_selected: int,
    completed_results: int,
    api_error_count: int,
    state: str,
    last_sample_id: str | None = None,
    last_error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "dataset_path": dataset_path,
        "output_path": output_path,
        "total_selected": total_selected,
        "completed_results": completed_results,
        "pending_results": max(total_selected - completed_results, 0),
        "api_error_count": api_error_count,
        "state": state,
        "last_sample_id": last_sample_id,
        "last_error": last_error,
        "updated_at": _now_text(),
    }


def _filter_dataset(
    dataset: List[Dict[str, Any]],
    inference_num: int | None = None,
    sample_id: str | None = None,
    uid: str | None = None,
) -> List[Dict[str, Any]]:
    filtered = dataset
    if sample_id:
        filtered = [entry for entry in filtered if _build_sample_id(entry) == sample_id]
    if uid:
        filtered = [entry for entry in filtered if str(entry.get("uid", "")) == uid]
    if inference_num is not None:
        filtered = filtered[:inference_num]
    return filtered


def run_inference(
    dataset_path: str,
    output_path: str,
    inference_num: int | None = None,
    include_candidate_aim_scope: bool = True,
    sample_id: str | None = None,
    uid: str | None = None,
    num_shards: int = 1,
    shard_id: int = 0,
    finished_from: str | None = None,
) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset = _filter_dataset(dataset, inference_num, sample_id, uid)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    requested_output_path = output_path
    output_path = _build_shard_output_path(output_path, num_shards, shard_id)
    dataset = _select_shard(dataset, num_shards, shard_id)

    existing_results = _load_existing_results(output_path)
    finished_sample_ids = {_build_sample_id(result) for result in existing_results}
    extra_finished_paths = _parse_extra_paths(finished_from)
    if num_shards > 1:
        extra_finished_paths.extend([requested_output_path, f"{requested_output_path}.jsonl"])
    finished_sample_ids |= _load_finished_sample_ids(extra_finished_paths)
    pending = [
        entry for entry in dataset if _build_sample_id(entry) not in finished_sample_ids
    ]

    if not pending:
        return

    inference_results = list(existing_results)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_jsonl_path = f"{output_path}.jsonl"
    status_path = f"{output_path}.status.json"
    status_output_path = output_jsonl_path
    api_error_count = sum(
        1
        for result in inference_results
        if isinstance(result.get("model_response"), dict)
        and "Error" in result["model_response"]
    )
    _write_json_atomic(
        status_path,
        _make_status(
            dataset_path=dataset_path,
            output_path=status_output_path,
            total_selected=len(dataset),
            completed_results=len(inference_results),
            api_error_count=api_error_count,
            state="running",
        ),
    )

    for idx, entry in enumerate(
        tqdm(pending, desc="Inference (Direct Single-Stage)"), start=1
    ):
        sys_prompt = create_system_prompt(entry["simulation_setting"])
        usr_prompt = create_user_prompt(
            entry["paper_content"],
            entry.get("transfer_recommendation_task", {}).get("candidates", []),
            include_candidate_aim_scope,
        )
        model_decision, fatal_error = _call_model(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ]
        )

        result = {
            "sample_id": _build_sample_id(entry),
            "uid": entry["uid"],
            "doi": entry.get("doi"),
            "Category": entry.get("Category"),
            "subject": entry.get("subject"),
            "data_type": entry["data_type"],
            "target_journal_name": entry.get("simulation_setting", {}).get(
                "target_journal_name"
            ),
            "target_subject": entry.get("simulation_setting", {}).get(
                "target_subject"
            ),
            "ground_truth": entry["ground_truth"],
            "transfer_gt": entry.get("transfer_recommendation_task", {}).get(
                "correct_option", None
            ),
            "model_response": model_decision,
        }
        inference_results.append(result)
        _append_jsonl(output_jsonl_path, result)

        if isinstance(model_decision, dict) and "Error" in model_decision:
            api_error_count += 1

        completed_results = len(inference_results)
        _write_json_atomic(
            status_path,
            _make_status(
                dataset_path=dataset_path,
                output_path=status_output_path,
                total_selected=len(dataset),
                completed_results=completed_results,
                api_error_count=api_error_count,
                state="running",
                last_sample_id=result["sample_id"],
                last_error=model_decision if "Error" in model_decision else None,
            ),
        )

        if fatal_error:
            if WRITE_FINAL_JSON:
                _write_json_atomic(output_path, inference_results)
            _write_json_atomic(
                status_path,
                _make_status(
                    dataset_path=dataset_path,
                    output_path=status_output_path,
                    total_selected=len(dataset),
                    completed_results=completed_results,
                    api_error_count=api_error_count,
                    state="stopped_fatal_api_error",
                    last_sample_id=result["sample_id"],
                    last_error=model_decision,
                ),
            )
            print(
                f"[{_now_text()}] Fatal API error encountered. "
                f"Progress saved to: {output_jsonl_path}"
            )
            return

    if WRITE_FINAL_JSON:
        _write_json_atomic(output_path, inference_results)
    _write_json_atomic(
        status_path,
        _make_status(
            dataset_path=dataset_path,
            output_path=status_output_path,
            total_selected=len(dataset),
            completed_results=len(inference_results),
            api_error_count=api_error_count,
            state="completed",
        ),
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-stage benchmark inference.")
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("--no-scope", action="store_true")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=OUTPUT_RESULTS_PATH)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--finished-from", default=None)
    args = parser.parse_args()

    run_inference(
        dataset_path=args.dataset,
        output_path=args.output,
        inference_num=args.num,
        include_candidate_aim_scope=not args.no_scope,
        sample_id=args.sample_id,
        uid=args.uid,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        finished_from=args.finished_from,
    )
