"""Run the synthetic-degradation experiment on benchmark entries.
The script builds degraded variants, supports preview mode, and writes resumable outputs."""

import argparse
import json
import os
from copy import deepcopy
from typing import Any, Dict, List

from tqdm import tqdm

from experiment_utils import (
    DEFAULT_REAL_SUBMISSION_DATASET,
    DEGRADATION_MODE_GUIDE,
    append_jsonl,
    build_degraded_entry_from_cache_or_model,
    build_degradation_messages,
    build_result_record,
    build_shard_output_path,
    build_system_prompt,
    build_user_prompt,
    clone_entry,
    default_output_path,
    extract_base_sample_id,
    filter_dataset,
    is_cot_mode,
    load_base_module,
    load_existing_results,
    load_finished_sample_ids,
    load_jsonl_map,
    make_status,
    now_text,
    parse_csv_arg,
    parse_extra_paths,
    select_shard,
    write_json_atomic,
)


EXPERIMENT_NAME = "synthetic_degradation"


def expand_dataset(dataset: List[Dict[str, Any]], degrade_modes: List[str]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    modes = degrade_modes or ["innovation_weakening"]
    for entry in dataset:
        for mode in modes:
            cloned = clone_entry(
                entry=entry,
                experiment_name=EXPERIMENT_NAME,
                variant=mode,
                prompt_variant="baseline",
                decision_only=True,
            )
            cloned["source_ground_truth"] = deepcopy(entry.get("ground_truth", {}))
            ground_truth = deepcopy(entry.get("ground_truth", {}))
            ground_truth["expected_decision"] = "Desk Reject"
            ground_truth["expected_reason_label"] = "Insufficient Novelty/Impact"
            cloned["ground_truth"] = ground_truth
            transfer_task = deepcopy(entry.get("transfer_recommendation_task", {}))
            transfer_task["correct_option"] = None
            cloned["transfer_recommendation_task"] = transfer_task
            expanded.append(cloned)
    return expanded


def preview_entries(dataset: List[Dict[str, Any]], mode: str) -> None:
    cot = is_cot_mode(mode)
    for idx, entry in enumerate(dataset[: min(3, len(dataset))], start=1):
        preview = {
            "preview_index": idx,
            "sample_id": entry.get("sample_id"),
            "source_sample_id": entry.get("source_sample_id"),
            "experiment": entry.get("experiment"),
            "paper_content_before_degradation": entry.get("paper_content"),
            "degradation_generation_messages": build_degradation_messages(entry),
            "triage_system_prompt": build_system_prompt(
                entry.get("simulation_setting", {}),
                cot=cot,
                prompt_variant="baseline",
                decision_only=True,
            ),
            "triage_user_prompt": build_user_prompt(
                entry.get("paper_content", {}),
                [],
                include_candidate_aim_scope=False,
                decision_only=True,
            ),
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))


def run_experiment(
    dataset_path: str,
    output_path: str | None,
    mode: str,
    inference_num: int | None,
    sample_id: str | None,
    uid: str | None,
    data_types: List[str],
    degrade_modes: List[str],
    generation_temperature: float,
    degradation_cache_path: str | None,
    num_shards: int,
    shard_id: int,
    finished_from: str | None,
    preview_only: bool,
) -> None:
    base = load_base_module(mode)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    base_sample_id = extract_base_sample_id(sample_id)
    dataset = filter_dataset(
        dataset,
        inference_num=inference_num,
        sample_id=base_sample_id,
        uid=uid,
        data_types=data_types,
    )
    dataset = expand_dataset(dataset, degrade_modes)
    if sample_id:
        dataset = [
            entry
            for entry in dataset
            if entry.get("sample_id") == sample_id
            or entry.get("source_sample_id") == sample_id
        ]

    if preview_only:
        preview_entries(dataset, mode)
        return

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")

    requested_output_path = output_path or default_output_path(
        EXPERIMENT_NAME, mode, base.MODEL
    )
    output_path = build_shard_output_path(requested_output_path, num_shards, shard_id)
    dataset = select_shard(dataset, num_shards, shard_id)

    existing_results = load_existing_results(output_path)
    finished_sample_ids = {result.get("sample_id") for result in existing_results}
    extra_finished_paths = parse_extra_paths(finished_from)
    if num_shards > 1:
        extra_finished_paths.extend([requested_output_path, f"{requested_output_path}.jsonl"])
    finished_sample_ids |= load_finished_sample_ids(extra_finished_paths)
    pending = [
        entry for entry in dataset if entry.get("sample_id") not in finished_sample_ids
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

    if degradation_cache_path is None:
        if output_path.endswith(".json"):
            degradation_cache_path = output_path[:-5] + ".degradation_cache.jsonl"
        else:
            degradation_cache_path = output_path + ".degradation_cache.jsonl"
    cache_dir = os.path.dirname(degradation_cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    degradation_cache = load_jsonl_map(degradation_cache_path, "cache_key")

    api_error_count = sum(
        1
        for result in inference_results
        if isinstance(result.get("model_response"), dict)
        and "Error" in result["model_response"]
    )
    write_json_atomic(
        status_path,
        make_status(
            dataset_path=dataset_path,
            output_path=status_output_path,
            total_selected=len(dataset),
            completed_results=len(inference_results),
            api_error_count=api_error_count,
            state="running",
            experiment=EXPERIMENT_NAME,
        ),
    )

    for entry in tqdm(pending, desc="Experiment (Synthetic Degradation)"):
        prepared_entry, prep_error, fatal_error = build_degraded_entry_from_cache_or_model(
            entry=entry,
            base_module=base,
            degradation_cache=degradation_cache,
            degradation_cache_path=degradation_cache_path,
            generation_temperature=generation_temperature,
        )

        if prep_error is not None or prepared_entry is None:
            model_decision = prep_error or {
                "Error": "Preparation Failed",
                "Raw": "Unknown degradation preparation error",
            }
            paper_used = entry["paper_content"]
        else:
            sys_prompt = build_system_prompt(
                prepared_entry["simulation_setting"],
                cot=is_cot_mode(mode),
                prompt_variant="baseline",
                decision_only=True,
            )
            usr_prompt = build_user_prompt(
                prepared_entry["paper_content"],
                [],
                include_candidate_aim_scope=False,
                decision_only=True,
            )
            model_decision, fatal_error = build_degraded_model_decision(
                base, sys_prompt, usr_prompt
            )
            paper_used = prepared_entry["paper_content"]
            entry["experiment"].update(prepared_entry.get("experiment", {}))

        result = build_result_record(
            entry=entry,
            model_response=model_decision,
            paper_content_used=paper_used,
            decision_only=True,
        )
        inference_results.append(result)
        append_jsonl(output_jsonl_path, result)

        if isinstance(model_decision, dict) and "Error" in model_decision:
            api_error_count += 1

        completed_results = len(inference_results)
        write_json_atomic(
            status_path,
            make_status(
                dataset_path=dataset_path,
                output_path=status_output_path,
                total_selected=len(dataset),
                completed_results=completed_results,
                api_error_count=api_error_count,
                state="running",
                experiment=EXPERIMENT_NAME,
                last_sample_id=result["sample_id"],
                last_error=model_decision if "Error" in model_decision else None,
            ),
        )

        if fatal_error:
            write_json_atomic(
                status_path,
                make_status(
                    dataset_path=dataset_path,
                    output_path=status_output_path,
                    total_selected=len(dataset),
                    completed_results=completed_results,
                    api_error_count=api_error_count,
                    state="stopped_fatal_api_error",
                    experiment=EXPERIMENT_NAME,
                    last_sample_id=result["sample_id"],
                    last_error=model_decision,
                ),
            )
            print(
                f"[{now_text()}] Fatal API error encountered. "
                f"Progress saved to: {output_jsonl_path}"
            )
            return

    write_json_atomic(
        status_path,
        make_status(
            dataset_path=dataset_path,
            output_path=status_output_path,
            total_selected=len(dataset),
            completed_results=len(inference_results),
            api_error_count=api_error_count,
            state="completed",
            experiment=EXPERIMENT_NAME,
        ),
    )
def build_degraded_model_decision(base, sys_prompt: str, usr_prompt: str):
    from experiment_utils import call_json_model

    return call_json_model(
        base_module=base,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
        temperature=0.0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the synthetic-degradation experiment.")
    parser.add_argument("--dataset", default=DEFAULT_REAL_SUBMISSION_DATASET)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["nocot", "cot"], default="nocot")
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--data-types", default="Real_Submission")
    parser.add_argument(
        "--degrade-modes",
        default="innovation_weakening",
        help=(
            "Comma-separated degradation modes: "
            + ", ".join(sorted(DEGRADATION_MODE_GUIDE))
        ),
    )
    parser.add_argument("--generation-temperature", type=float, default=0.4)
    parser.add_argument("--degradation-cache", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--finished-from", default=None)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    run_experiment(
        dataset_path=args.dataset,
        output_path=args.output,
        mode=args.mode,
        inference_num=args.num,
        sample_id=args.sample_id,
        uid=args.uid,
        data_types=parse_csv_arg(args.data_types),
        degrade_modes=parse_csv_arg(args.degrade_modes),
        generation_temperature=args.generation_temperature,
        degradation_cache_path=args.degradation_cache,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        finished_from=args.finished_from,
        preview_only=args.preview_only,
    )
