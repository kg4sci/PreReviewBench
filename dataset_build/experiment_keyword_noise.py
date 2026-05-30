"""Run the keyword-noise experiment on benchmark entries.
The script expands dataset variants, supports preview mode, and writes resumable outputs."""

import argparse
import json
import os
from typing import Any, Dict, List

from tqdm import tqdm

from experiment_utils import (
    DEFAULT_REAL_SUBMISSION_DATASET,
    append_jsonl,
    apply_keyword_noise,
    build_result_record,
    build_shard_output_path,
    build_system_prompt,
    build_user_prompt,
    call_json_model,
    clone_entry,
    default_output_path,
    extract_base_sample_id,
    filter_dataset,
    is_cot_mode,
    load_base_module,
    load_existing_results,
    load_finished_sample_ids,
    make_status,
    now_text,
    parse_csv_arg,
    parse_extra_paths,
    select_shard,
    write_json_atomic,
)


EXPERIMENT_NAME = "keyword_noise"


def expand_dataset(
    dataset: List[Dict[str, Any]],
    keyword_profiles: List[str],
    keyword_noise_count: int,
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    profiles = keyword_profiles or ["hot_terms"]
    for entry in dataset:
        for profile in profiles:
            cloned = clone_entry(
                entry=entry,
                experiment_name=EXPERIMENT_NAME,
                variant=profile,
                prompt_variant="baseline",
                decision_only=True,
            )
            cloned, audit = apply_keyword_noise(
                cloned, profile=profile, noise_count=keyword_noise_count
            )
            cloned["experiment"].update(audit)
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
            "paper_content": entry.get("paper_content"),
            "system_prompt": build_system_prompt(
                entry.get("simulation_setting", {}),
                cot=cot,
                prompt_variant="baseline",
                decision_only=True,
            ),
            "user_prompt": build_user_prompt(
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
    keyword_profiles: List[str],
    keyword_noise_count: int,
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
    dataset = expand_dataset(dataset, keyword_profiles, keyword_noise_count)
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

    for entry in tqdm(pending, desc="Experiment (Keyword Noise)"):
        sys_prompt = build_system_prompt(
            entry["simulation_setting"],
            cot=is_cot_mode(mode),
            prompt_variant="baseline",
            decision_only=True,
        )
        usr_prompt = build_user_prompt(
            entry["paper_content"],
            [],
            include_candidate_aim_scope=False,
            decision_only=True,
        )
        model_decision, fatal_error = call_json_model(
            base_module=base,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ],
            temperature=0.0,
        )

        result = build_result_record(
            entry=entry,
            model_response=model_decision,
            paper_content_used=entry["paper_content"],
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the keyword-noise experiment.")
    parser.add_argument("--dataset", default=DEFAULT_REAL_SUBMISSION_DATASET)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["nocot", "cot"], default="nocot")
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--data-types", default="Real_Submission")
    parser.add_argument(
        "--keyword-profiles",
        default="hot_terms",
        help="Comma-separated profiles: hot_terms, cross_domain, mixed",
    )
    parser.add_argument("--keyword-noise-count", type=int, default=5)
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
        keyword_profiles=parse_csv_arg(args.keyword_profiles),
        keyword_noise_count=args.keyword_noise_count,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        finished_from=args.finished_from,
        preview_only=args.preview_only,
    )
