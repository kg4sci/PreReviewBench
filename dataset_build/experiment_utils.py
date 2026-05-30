"""Shared helpers for the released robustness experiments.

This public version removes environment-specific paths, keeps English-only
documentation, and preserves only the helpers used by the repository's
experiment scripts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FULL_BENCHMARK_DATASET = os.getenv(
    "BENCHMARK_DATASET_PATH",
    str(REPO_ROOT / "dataset" / "benchmark2025.json"),
)
DEFAULT_REAL_SUBMISSION_DATASET = os.getenv(
    "REAL_SUBMISSION_DATASET_PATH",
    "YOUR_REAL_SUBMISSION_BASE_DATASET_JSON_PATH",
)
DEFAULT_OUTPUT_DIR = os.getenv("EXPERIMENT_OUTPUT_DIR", "YOUR_OUTPUT_DIR")

BASE_MODULE_PATHS = {
    "nocot": str(REPO_ROOT / "inference" / "inference_single_stage_direct.py"),
    "cot": str(REPO_ROOT / "inference" / "inference_single_stage_rationale_first.py"),
}

PROMPT_VARIANTS = {
    "baseline": {
        "perspective": (
            "You are NOT choosing the best journal in the world for this paper. "
            "You are deciding whether THIS journal should spend reviewer attention on it.\n"
            "Your task is to decide whether the manuscript should proceed to external peer review, "
            "not whether it should ultimately be accepted for publication.\n"
            "You only have title, abstract, and keywords. Do NOT invent fatal weaknesses merely "
            "because full methods, experiments, data, or controls are not shown in the abstract."
        ),
        "priority_gate": (
            "If the manuscript is within scope, decide whether it appears sufficiently interesting, "
            "coherent, and potentially impactful to justify sending it to external reviewers for this journal.\n"
            "Ask whether you would be comfortable sending this manuscript to 2-3 external reviewers "
            "for this journal, based only on the abstract-level evidence available."
        ),
        "notes": [
            "Make the primary decision using ONLY the target journal profile and the manuscript itself.",
            "Do not treat a mismatch in prestige alone as a mismatch in scope.",
            'Use "Out of Scope" to reflect fit; use "Insufficient Novelty/Impact" to reflect editorial priority within scope.',
        ],
    },
    "editorial_realistic": {
        "perspective": (
            "Act as a working journal editor making a first-pass triage decision under uncertainty.\n"
            "Your job is to judge whether this manuscript looks suitable enough, relevant enough, "
            "and promising enough to justify external review at THIS journal now.\n"
            "Do not over-penalize the paper for abstract-level incompleteness. You are making an "
            "editorial screening decision, not a full referee report."
        ),
        "priority_gate": (
            "If the manuscript is within scope, ask whether the abstract presents a sufficiently "
            "clear and credible contribution to warrant reviewer time for this journal.\n"
            "Focus on whether the submission seems like a plausible, worthwhile review candidate at this venue."
        ),
        "notes": [
            "Base the primary decision on the manuscript and the target journal profile, not on transfer candidates.",
            "Separate topical or readership fit from questions of strength and editorial priority.",
            "Do not convert limited abstract detail into imagined fatal flaws.",
        ],
    },
    "conservative_gatekeeping": {
        "perspective": (
            "Act as a selective editor managing limited reviewer resources.\n"
            "Send manuscripts forward only when the abstract gives a sufficiently strong positive signal "
            "that review effort is justified at THIS journal.\n"
            "You still only have title, abstract, and keywords, so do not manufacture missing-evidence "
            "criticisms beyond that boundary."
        ),
        "priority_gate": (
            "If the manuscript is within scope, decide whether its abstract-level signal is strong enough "
            "to justify reviewer allocation at this journal right now.\n"
            "When the case is borderline, prefer the option that best reflects realistic editorial "
            "resource constraints."
        ),
        "notes": [
            "Do not confuse prestige differences with scope differences.",
            'Use "Out of Scope" only for fit problems, not for within-scope papers that seem too weak or insufficiently compelling.',
            "Judge based on whether you would confidently send this paper to reviewers at this journal now.",
        ],
    },
}

GLOBAL_HOT_TERMS = [
    "artificial intelligence",
    "large language models",
    "single-cell RNA sequencing",
    "CRISPR screening",
    "multi-omics integration",
    "precision medicine",
    "tumor microenvironment",
    "nanoparticle delivery",
    "foundation models",
    "digital twin",
]

CROSS_DOMAIN_NOISE_POOLS = {
    "Biology": [
        "perovskite ceramics",
        "solid-state electrolyte",
        "coordination polymer catalysis",
        "ophthalmic imaging biomarker",
        "dermatologic laser therapy",
        "quantum dot synthesis",
        "battery interface engineering",
    ],
    "Chemistry": [
        "tumor microenvironment",
        "single-cell atlas",
        "fibroblast niche signaling",
        "retinal degeneration model",
        "cutaneous microbiome",
        "immune checkpoint blockade",
        "organoid lineage tracing",
    ],
    "Materials Science": [
        "apoptotic signaling network",
        "tumor microenvironment",
        "retinal neovascularization",
        "cutaneous inflammation biomarker",
        "host-pathogen interaction",
        "single-cell transcriptomics",
        "CRISPR screening",
    ],
    "Medicine": [
        "grain-boundary densification",
        "molecular docking scaffold hopping",
        "coordination complex reactivity",
        "bioactive glass composite",
        "solid-state diffusion pathway",
        "surface wettability tuning",
        "catalytic active site engineering",
    ],
}

DEGRADATION_MODE_GUIDE = {
    "innovation_weakening": (
        "Keep the paper in the same topic and journal scope, but make the contribution seem "
        "incremental, confirmatory, or only modestly novel."
    ),
    "conclusion_blurring": (
        "Keep the same study theme, but rewrite the abstract so the final take-home message becomes "
        "vague, cautious, and less decisive."
    ),
    "logic_disruption": (
        "Keep the same ingredients, but reduce the clarity of the narrative so the problem, "
        "approach, and implications connect less cleanly."
    ),
    "methods_hollowing": (
        "Keep the same topic, but make the methodological description feel thinner, less concrete, "
        "and less convincing."
    ),
    "evidence_weakening": (
        "Keep the same topic and apparent study design, but make the results seem less "
        "well-supported, less comprehensive, or less compelling."
    ),
}


def _ensure_parent_dir(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def safe_slug(text: str) -> str:
    text = str(text or "").strip().lower()
    chars: List[str] = []
    for char in text:
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "_", "."}:
            chars.append("-")
    compact = "".join(chars).strip("-")
    while "--" in compact:
        compact = compact.replace("--", "-")
    return compact or "unknown"


def model_slug(model_name: str) -> str:
    return safe_slug(str(model_name or "").replace("/", "-"))


def parse_csv_arg(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def is_cot_mode(mode: str) -> bool:
    return str(mode).strip().lower() == "cot"


def load_base_module(mode: str) -> ModuleType:
    normalized = str(mode).strip().lower()
    if normalized not in BASE_MODULE_PATHS:
        raise ValueError(f"Unsupported mode: {mode}")

    path = Path(BASE_MODULE_PATHS[normalized])
    if not path.exists():
        raise FileNotFoundError(f"Base inference module not found: {path}")

    module_name = f"prereviewbench_{normalized}_base"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_rng(*parts: str) -> random.Random:
    seed_text = "||".join(str(part) for part in parts)
    seed_int = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed_int)


def split_keywords(raw_keywords: Any) -> List[str]:
    if raw_keywords is None:
        return []
    text = str(raw_keywords).replace("\n", " ").strip()
    if not text:
        return []

    normalized = text
    for delimiter in ["|", ","]:
        normalized = normalized.replace(delimiter, ";")
    return [part.strip() for part in normalized.split(";") if part.strip()]


def normalize_keyword(term: str) -> str:
    return " ".join(str(term or "").lower().replace("-", " ").split())


def sample_terms(
    pool: List[str], k: int, seed_parts: List[str], existing_terms: set[str]
) -> List[str]:
    if k <= 0:
        return []

    unique_pool: List[str] = []
    seen_norms: set[str] = set()
    for term in pool:
        norm = normalize_keyword(term)
        if not norm or norm in existing_terms or norm in seen_norms:
            continue
        seen_norms.add(norm)
        unique_pool.append(term)

    if not unique_pool:
        return []

    rng = stable_rng(*seed_parts)
    if k >= len(unique_pool):
        rng.shuffle(unique_pool)
        return unique_pool
    return rng.sample(unique_pool, k)


def build_base_sample_id(entry: Dict[str, Any]) -> str:
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


def build_experiment_sample_id(
    entry: Dict[str, Any],
    experiment_name: str,
    experiment_variant: str,
    prompt_variant: str | None = None,
) -> str:
    base_id = build_base_sample_id(entry)
    tags = [f"exp={experiment_name}", f"variant={experiment_variant}"]
    if prompt_variant and prompt_variant != "baseline":
        tags.append(f"prompt={prompt_variant}")
    return base_id + "||" + "||".join(tags)


def extract_base_sample_id(requested_sample_id: str | None) -> str | None:
    if not requested_sample_id:
        return None
    text = str(requested_sample_id)
    if "||exp=" in text:
        text = text.split("||exp=")[0]
    if "||prompt=" in text:
        text = text.split("||prompt=")[0]
    return text


def filter_dataset(
    dataset: List[Dict[str, Any]],
    inference_num: int | None = None,
    sample_id: str | None = None,
    uid: str | None = None,
    data_types: List[str] | None = None,
) -> List[Dict[str, Any]]:
    filtered = dataset
    if sample_id:
        filtered = [
            entry for entry in filtered if build_base_sample_id(entry) == sample_id
        ]
    if uid:
        filtered = [
            entry for entry in filtered if str(entry.get("uid", "")) == str(uid)
        ]
    if data_types:
        allowed = {value.strip() for value in data_types if value.strip()}
        filtered = [
            entry
            for entry in filtered
            if str(entry.get("data_type", "")).strip() in allowed
        ]
    if inference_num is not None:
        filtered = filtered[:inference_num]
    return filtered


def load_json_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def load_jsonl_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    results: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
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


def load_jsonl_map(path: str, key_field: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in load_jsonl_results(path):
        value = item.get(key_field)
        if value is not None:
            mapping[str(value)] = item
    return mapping


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json_atomic(path: str, data: Any) -> None:
    _ensure_parent_dir(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        sample_id = item.get("sample_id") or build_base_sample_id(item)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        deduped.append(item)
    return deduped


def load_existing_results(output_path: str) -> List[Dict[str, Any]]:
    json_results = load_json_results(output_path)
    jsonl_results = load_jsonl_results(f"{output_path}.jsonl")
    base = jsonl_results if len(jsonl_results) > len(json_results) else json_results
    return dedupe_results(base)


def build_shard_output_path(output_path: str, num_shards: int, shard_id: int) -> str:
    if num_shards <= 1:
        return output_path
    if output_path.endswith(".json"):
        return output_path[:-5] + f".shard{shard_id + 1}of{num_shards}.json"
    return output_path + f".shard{shard_id + 1}of{num_shards}"


def select_shard(
    dataset: List[Dict[str, Any]], num_shards: int, shard_id: int
) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return dataset
    return [entry for idx, entry in enumerate(dataset) if idx % num_shards == shard_id]


def parse_extra_paths(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def load_finished_sample_ids(paths: List[str]) -> set[str]:
    sample_ids: set[str] = set()
    for path in paths:
        if not path:
            continue
        if path.endswith(".jsonl"):
            for item in load_jsonl_results(path):
                sample_ids.add(str(item.get("sample_id") or build_base_sample_id(item)))
            continue
        for item in load_json_results(path):
            sample_ids.add(str(item.get("sample_id") or build_base_sample_id(item)))
        for item in load_jsonl_results(f"{path}.jsonl"):
            sample_ids.add(str(item.get("sample_id") or build_base_sample_id(item)))
    return sample_ids


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def make_status(
    dataset_path: str,
    output_path: str,
    total_selected: int,
    completed_results: int,
    api_error_count: int,
    state: str,
    experiment: str,
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
        "experiment": experiment,
        "last_sample_id": last_sample_id,
        "last_error": last_error,
        "updated_at": now_text(),
    }


def default_output_path(experiment_name: str, mode: str, model_name: str) -> str:
    file_name = f"{experiment_name}_{safe_slug(mode)}_{model_slug(model_name)}.json"
    return os.path.join(DEFAULT_OUTPUT_DIR, file_name)


def format_candidates(
    candidates: List[Dict[str, Any]], include_aim_scope: bool
) -> str:
    lines: List[str] = []
    for candidate in candidates:
        option = candidate.get("option", "?")
        name = candidate.get("journal_name", "Unknown")
        subject = candidate.get("subject")
        jif = candidate.get("JIF", "N/A")
        quartile = candidate.get("JIF_Quartile", "N/A")
        h5_index = candidate.get("h5-index", "N/A")
        meta_parts = [f"JIF: {jif}", f"Quartile: {quartile}", f"h5-index: {h5_index}"]
        if subject:
            meta_parts.append(f"Subject: {subject}")
        lines.append(f"[{option}] {name} ({', '.join(meta_parts)})")
        if include_aim_scope and candidate.get("aim_scope"):
            lines.append(f"    Aims & Scope: {str(candidate['aim_scope']).strip()}")
    return "\n".join(lines)


def build_user_prompt(
    paper: Dict[str, Any],
    candidates_raw: List[Dict[str, Any]],
    include_candidate_aim_scope: bool = True,
    decision_only: bool = False,
) -> str:
    manuscript_block = f"""Please perform the Desk Triage on the following submitted manuscript.

### [SUBMITTED MANUSCRIPT]
- Title: {paper.get('title', 'N/A')}
- Keywords: {paper.get('keywords', 'N/A')}
- Abstract: {paper.get('abstract_text', 'N/A')}"""
    if decision_only:
        return manuscript_block

    candidate_list_string = format_candidates(
        candidates_raw, include_candidate_aim_scope
    )
    return f"""{manuscript_block}

### [POST-DECISION TRANSFER OPTIONS]
Consult this section only after you have formed the primary desk-triage decision, and only if that decision is "Desk Reject".
{candidate_list_string}
"""


def build_system_prompt(
    setting: Dict[str, Any],
    cot: bool,
    prompt_variant: str = "baseline",
    decision_only: bool = False,
    base_prompt_func: Any | None = None,
) -> str:
    if (
        prompt_variant == "baseline"
        and not decision_only
        and base_prompt_func is not None
    ):
        return base_prompt_func(setting)

    if prompt_variant not in PROMPT_VARIANTS:
        available = ", ".join(sorted(PROMPT_VARIANTS))
        raise ValueError(
            f"Unsupported prompt variant: {prompt_variant}. Available: {available}"
        )

    target_journal_name = setting.get("target_journal_name", "Unknown Journal")
    jif = setting.get("JIF", "N/A")
    quartile = setting.get("JIF_Quartile", "N/A")
    h5_index = setting.get("h5-index", "N/A")
    scope = setting.get("aim_scope", "N/A")
    payload = PROMPT_VARIANTS[prompt_variant]
    notes_block = "\n".join(
        f"{index}. {note}" for index, note in enumerate(payload["notes"], start=1)
    )

    cot_reasoning_block = ""
    step_offset = 0
    if cot:
        cot_reasoning_block = """STEP 1: Editorial Reasoning
Briefly explain, in 2-4 sentences, your editorial reasoning for the desk-triage decision.
This reasoning must be based ONLY on the manuscript and the target journal profile.
Do NOT use the candidate journal list when forming this reasoning.

"""
        step_offset = 1

    transfer_section = ""
    if not decision_only:
        transfer_section = """
### [POST-DECISION TRANSFER RULE]
Candidate journals are provided only for post-rejection transfer handling. Their presence does NOT imply that the manuscript should be rejected.
Do NOT use the candidate list to infer the correct primary decision, quality assessment, or rejection category.
Only after you have formed the primary desk-triage decision may you use the candidate list, and only if you chose "Desk Reject".

### [TRANSFER RANKING RULE]
If and only if you choose "Desk Reject", rank candidate journals as you would recommend them to the authors as realistic transfer destinations.
Use scope, readership, methodological fit, selectivity, JIF, quartile, h5-index, subject, and aims & scope as supporting signals.
Return a COMPLETE ranking of ALL candidate journals from best fit to worst fit using ONLY the option letters.
If you choose "Send for Review", set "Transfer_Ranking" to null.
"""

    json_fields = [
        '  "Primary_Decision": "Send for Review" | "Desk Reject",',
        '  "Confidence_Score": <Integer: 1-10>,',
        '  "Quality_Assessment": "Matches Journal Caliber" | "Exceeds Journal Caliber" | null,',
        '  "Rejection_Category": "Out of Scope" | "Insufficient Novelty/Impact" | null',
    ]
    if cot:
        json_fields.insert(
            0,
            '  "Reasoning_Process": "<2-4 sentences describing your editorial reasoning>",',
        )
    if not decision_only:
        json_fields[-1] = (
            '  "Rejection_Category": "Out of Scope" | "Insufficient Novelty/Impact" | null,'
        )
        json_fields.append(
            '  "Transfer_Ranking": ["<Option_Letter>", "<Option_Letter>", ...] | null'
        )
    json_schema = "{\n" + "\n".join(json_fields) + "\n}"

    extra_metrics_line = ""
    if cot and prompt_variant == "baseline":
        extra_metrics_line = (
            "Use JIF, quartile, and h5-index as part of your overall understanding of the journal's "
            "selectivity, readership level, and general editorial bar.\n"
        )

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
{payload["perspective"]}
{extra_metrics_line}### [DECISION WORKFLOW]
Make your decision in this order:

{cot_reasoning_block}STEP {1 + step_offset}: Scope Gate
Decide whether the manuscript is plausibly within the journal's scope, considering:
- core topic
- methodological orientation
- likely readership and community fit
- whether the journal's audience would reasonably see the paper as relevant

STEP {2 + step_offset}: Editorial Priority Gate
{payload["priority_gate"]}

STEP {3 + step_offset}: Map to Final Decision
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
{notes_block}
{transfer_section}
### [JSON OUTPUT SCHEMA]
You MUST respond with ONLY valid JSON. No markdown formatting, no introductory or concluding text.
{json_schema}"""


def try_repair_json_output(model_output_text: str) -> Dict[str, Any] | None:
    text = (model_output_text or "").strip()
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        return None

    candidate = text[start : end + 1] if end != -1 and end > start else text[start:]
    repair_candidates = [candidate]

    no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", candidate)
    if no_trailing_commas != candidate:
        repair_candidates.append(no_trailing_commas)

    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    open_brackets = candidate.count("[")
    close_brackets = candidate.count("]")
    if open_braces > close_braces or open_brackets > close_brackets:
        suffix = ""
        if open_brackets > close_brackets:
            suffix += "]" * (open_brackets - close_brackets)
        if open_braces > close_braces:
            suffix += "}" * (open_braces - close_braces)
        repair_candidates.append(candidate + suffix)
        repaired_balanced = re.sub(r",(\s*[}\]])", r"\1", candidate + suffix)
        if repaired_balanced != candidate + suffix:
            repair_candidates.append(repaired_balanced)

    seen: set[str] = set()
    for attempt in repair_candidates:
        if attempt in seen:
            continue
        seen.add(attempt)
        try:
            loaded = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def call_json_model(
    base_module: ModuleType,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
) -> Tuple[Dict[str, Any], bool]:
    parser = getattr(base_module, "_parse_model_output")
    is_fatal_api_error = getattr(base_module, "_is_fatal_api_error")
    is_retryable_api_error = getattr(base_module, "_is_retryable_api_error")
    max_api_retries = int(getattr(base_module, "MAX_API_RETRIES", 3))
    retry_backoff_seconds = float(getattr(base_module, "RETRY_BACKOFF_SECONDS", 10))
    client = getattr(base_module, "CLIENT")
    model_name = getattr(base_module, "MODEL")

    for attempt in range(1, max_api_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            raw_text = response.choices[0].message.content or ""
            parsed = parser(raw_text)
            if isinstance(parsed, dict) and parsed.get("Error") in {
                "JSON Parse Error",
                "No JSON found",
            }:
                repaired = try_repair_json_output(raw_text)
                if repaired is not None:
                    return repaired, False
            return parsed, False
        except Exception as exc:
            raw = str(exc)
            error = {"Error": "API Failed", "Raw": raw, "Attempt": attempt}
            if is_fatal_api_error(raw):
                return error, True
            if attempt < max_api_retries and is_retryable_api_error(raw):
                wait_seconds = retry_backoff_seconds * attempt
                print(
                    f"[{now_text()}] Retryable API error on attempt "
                    f"{attempt}/{max_api_retries}: {raw}"
                )
                time.sleep(wait_seconds)
                continue
            return error, False
    return {"Error": "API Failed", "Raw": "Unknown retry exit"}, False


def build_result_record(
    entry: Dict[str, Any],
    model_response: Dict[str, Any],
    paper_content_used: Dict[str, Any],
    decision_only: bool,
) -> Dict[str, Any]:
    result = {
        "sample_id": entry.get("sample_id") or build_base_sample_id(entry),
        "source_sample_id": entry.get("source_sample_id"),
        "uid": entry.get("uid"),
        "doi": entry.get("doi"),
        "Category": entry.get("Category"),
        "subject": entry.get("subject"),
        "data_type": entry.get("data_type"),
        "target_journal_name": entry.get("simulation_setting", {}).get(
            "target_journal_name"
        ),
        "target_subject": entry.get("simulation_setting", {}).get("target_subject"),
        "ground_truth": entry.get("ground_truth"),
        "transfer_gt": (
            None
            if decision_only
            else entry.get("transfer_recommendation_task", {}).get("correct_option")
        ),
        "experiment": entry.get("experiment", {}),
        "paper_content_used": paper_content_used,
        "model_response": model_response,
    }
    if "source_ground_truth" in entry:
        result["source_ground_truth"] = entry["source_ground_truth"]
    return result


def clone_entry(
    entry: Dict[str, Any],
    experiment_name: str,
    variant: str,
    prompt_variant: str = "baseline",
    decision_only: bool = False,
) -> Dict[str, Any]:
    cloned = deepcopy(entry)
    cloned["sample_id"] = build_experiment_sample_id(
        entry, experiment_name, variant, prompt_variant
    )
    cloned["source_sample_id"] = build_base_sample_id(entry)
    cloned["experiment"] = {
        "name": experiment_name,
        "variant": variant,
        "prompt_variant": prompt_variant,
        "decision_only": decision_only,
    }
    return cloned


def pick_keyword_noise_terms(
    entry: Dict[str, Any], profile: str, noise_count: int
) -> List[str]:
    if noise_count <= 0:
        return []

    base_sample_id = build_base_sample_id(entry)
    paper = entry.get("paper_content", {})
    existing_terms = {
        normalize_keyword(term) for term in split_keywords(paper.get("keywords"))
    }
    category = entry.get("Category", "")

    if profile == "hot_terms":
        return sample_terms(
            GLOBAL_HOT_TERMS, noise_count, [base_sample_id, profile], existing_terms
        )
    if profile == "cross_domain":
        pool = CROSS_DOMAIN_NOISE_POOLS.get(category, GLOBAL_HOT_TERMS)
        return sample_terms(pool, noise_count, [base_sample_id, profile], existing_terms)
    if profile == "mixed":
        hot_count = max(noise_count // 2, 1)
        cross_count = max(noise_count - hot_count, 1)
        hot_terms = sample_terms(
            GLOBAL_HOT_TERMS,
            hot_count,
            [base_sample_id, profile, "hot"],
            existing_terms,
        )
        augmented_existing = existing_terms | {
            normalize_keyword(term) for term in hot_terms
        }
        cross_terms = sample_terms(
            CROSS_DOMAIN_NOISE_POOLS.get(category, GLOBAL_HOT_TERMS),
            cross_count,
            [base_sample_id, profile, "cross"],
            augmented_existing,
        )
        combined: List[str] = []
        seen_norms: set[str] = set()
        for term in hot_terms + cross_terms:
            norm = normalize_keyword(term)
            if norm in seen_norms:
                continue
            seen_norms.add(norm)
            combined.append(term)
        return combined[:noise_count]
    raise ValueError(f"Unsupported keyword profile: {profile}")


def apply_keyword_noise(
    entry: Dict[str, Any], profile: str, noise_count: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    updated_entry = deepcopy(entry)
    paper = deepcopy(updated_entry.get("paper_content", {}))
    original_keywords = split_keywords(paper.get("keywords"))
    added_keywords = pick_keyword_noise_terms(entry, profile, noise_count)
    paper["keywords"] = "; ".join(original_keywords + added_keywords) if (
        original_keywords or added_keywords
    ) else ""
    updated_entry["paper_content"] = paper
    return updated_entry, {
        "keyword_profile": profile,
        "keyword_noise_count": noise_count,
        "added_keywords": added_keywords,
        "original_keywords_count": len(original_keywords),
        "final_keywords_count": len(original_keywords) + len(added_keywords),
    }


def build_degradation_messages(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    setting = entry.get("simulation_setting", {})
    paper = entry.get("paper_content", {})
    journal_name = setting.get("target_journal_name", "Unknown Journal")
    jif = setting.get("JIF", "N/A")
    quartile = setting.get("JIF_Quartile", "N/A")
    h5_index = setting.get("h5-index", "N/A")
    aim_scope = setting.get("aim_scope", "N/A")
    mode = entry.get("experiment", {}).get("variant", "innovation_weakening")
    mode_instruction = DEGRADATION_MODE_GUIDE.get(
        mode, DEGRADATION_MODE_GUIDE["innovation_weakening"]
    )

    system_prompt = """You are generating synthetic weaker desk-triage controls for a journal pre-review benchmark.

Rewrite a strong manuscript into a weaker version that remains in the SAME broad topic area and still plausibly belongs to the SAME journal scope.

Hard constraints:
- Keep the same broad research topic, organism/material/system, and journal fit.
- Do NOT make the paper out of scope.
- Do NOT add obvious nonsense, fake ethical problems, or blatant contradictions.
- The weakened version should look like a realistic abstract that an editor could plausibly desk reject for insufficient novelty/impact.
- Preserve professional scientific tone.
- Output ONLY valid JSON.

JSON schema:
{
  "title": "...",
  "keywords": "...",
  "abstract_text": "...",
  "degradation_note": "1-2 sentences explaining how the manuscript was weakened"
}"""

    user_prompt = f"""Target journal:
- Journal Name: {journal_name}
- JIF: {jif}
- Quartile: {quartile}
- h5-index: {h5_index}
- Aims & Scope: {aim_scope}

Degradation mode:
- {mode}
- Instruction: {mode_instruction}

Original manuscript:
- Title: {paper.get('title', 'N/A')}
- Keywords: {paper.get('keywords', 'N/A')}
- Abstract: {paper.get('abstract_text', 'N/A')}

Rewrite the manuscript so it remains in scope for {journal_name} but appears weaker at desk triage, primarily because of insufficient novelty/impact rather than scope mismatch."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_degraded_entry_from_cache_or_model(
    entry: Dict[str, Any],
    base_module: ModuleType,
    degradation_cache: Dict[str, Dict[str, Any]],
    degradation_cache_path: str,
    generation_temperature: float,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, bool]:
    cache_key = str(entry.get("sample_id"))
    if cache_key in degradation_cache:
        cached = degradation_cache[cache_key]
        updated_entry = deepcopy(entry)
        updated_entry["paper_content"] = deepcopy(cached.get("paper_content", {}))
        updated_entry["experiment"].update(cached.get("experiment_updates", {}))
        return updated_entry, None, False

    generated, fatal_error = call_json_model(
        base_module=base_module,
        messages=build_degradation_messages(entry),
        temperature=generation_temperature,
    )
    if "Error" in generated:
        return None, generated, fatal_error

    original_paper = entry.get("paper_content", {})
    degraded_paper = {
        "title": str(
            generated.get("title") or original_paper.get("title") or ""
        ).strip(),
        "keywords": str(
            generated.get("keywords") or original_paper.get("keywords") or ""
        ).strip(),
        "abstract_text": str(
            generated.get("abstract_text")
            or generated.get("abstract")
            or original_paper.get("abstract_text")
            or ""
        ).strip(),
    }
    if not degraded_paper["abstract_text"]:
        return (
            None,
            {"Error": "Preparation Failed", "Raw": "Empty degraded abstract"},
            False,
        )

    experiment_updates = {
        "degradation_note": generated.get("degradation_note"),
        "degradation_temperature": generation_temperature,
    }
    cache_record = {
        "cache_key": cache_key,
        "paper_content": degraded_paper,
        "experiment_updates": experiment_updates,
        "created_at": now_text(),
    }
    append_jsonl(degradation_cache_path, cache_record)
    degradation_cache[cache_key] = cache_record

    updated_entry = deepcopy(entry)
    updated_entry["paper_content"] = degraded_paper
    updated_entry["experiment"].update(experiment_updates)
    return updated_entry, None, False
