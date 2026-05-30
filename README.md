# PreReviewBench

PreReviewBench is a benchmark for LLM-based journal editorial pre-review. It targets the desk-screening stage where an editor decides whether a manuscript should be sent to external review, rejected for being out of scope, or rejected for insufficient novelty/impact. For rejected cases, the benchmark also evaluates transfer-journal recommendation.

This repository currently contains:

- released benchmark JSON files
- dataset construction and robustness-generation scripts
- shared experiment utilities
- inference scripts for direct and rationale-first prompting
- an evaluation script for metrics and plots

## Repository Layout

```text
PreReviewBench/
├── dataset/
│   ├── benchmark2025.json
│   ├── benchmark2025_keyword_noise_mixed.json
│   └── benchmark2025_synthetic_degradation.json
├── dataset_build/
│   ├── build_in_scope_benchmark.py
│   ├── build_scope_mismatch_benchmark.py
│   ├── compute_ssi.py
│   ├── experiment_keyword_noise.py
│   ├── experiment_synthetic_degradation.py
│   ├── experiment_utils.py
│   ├── step1_merge_metadata.py
│   ├── step2_split_by_year.py
│   ├── step3_embed_papers.py
│   ├── step3_embed_papers_self_hosted.py
│   └── step4_add_jif.py
├── evaluation/
│   └── evaluate_predictions.py
├── inference/
│   ├── inference_single_stage_direct.py
│   └── inference_single_stage_rationale_first.py
├── requirements.txt
└── README.md
```

## Included Datasets

The `dataset/` directory currently includes three released benchmark files:

1. `benchmark2025.json`
   Full benchmark release with 7,292 paper-journal instances.

2. `benchmark2025_keyword_noise_mixed.json`
   Keyword-noise robustness set with 1,823 instances derived from real submissions.

3. `benchmark2025_synthetic_degradation.json`
   Synthetic-degradation robustness set with 1,823 instances derived from real submissions.

## Benchmark Tasks

Each benchmark instance is centered on a paper and a target journal.

- Task 1: Desk-triage decision
  Predict `Send for Review` or `Desk Reject`.

- Task 2: Branch-specific editorial judgment
  If the paper should be sent for review, judge whether it `Matches Journal Caliber` or `Exceeds Journal Caliber`.
  If the paper should be desk rejected, judge whether the main reason is `Insufficient Novelty/Impact` or `Out of Scope`.

- Task 3: Transfer recommendation
  For desk-reject cases, rank candidate transfer journals.

## Data Format

The main benchmark file is a JSON list. Each instance contains the following top-level fields:

```json
{
  "uid": "...",
  "doi": "...",
  "Category": "...",
  "data_type": "...",
  "subject": "...",
  "paper_content": {
    "title": "...",
    "abstract_text": "...",
    "keywords": "..."
  },
  "simulation_setting": {
    "target_journal_name": "...",
    "aim_scope": "...",
    "JIF": 0.0,
    "JIF_Quartile": "...",
    "h5-index": 0
  },
  "ground_truth": {
    "actual_published_journal": "...",
    "expected_decision": "...",
    "expected_reason_label": "..."
  },
  "transfer_recommendation_task": {
    "candidate_count": 0,
    "candidates": []
  }
}
```

The two robustness files keep the same core paper and journal fields, but add perturbation-specific annotations such as `added_keywords` or `degraded_abstract_text`.

## Dataset Construction Scripts

The `dataset_build/` directory contains the main construction pipeline.

- `step1_merge_metadata.py`
  Merge raw metadata sources and export normalized paper-level records.

- `step2_split_by_year.py`
  Split the merged metadata by publication year.

- `step3_embed_papers.py`
  Generate paper embeddings through an API-compatible embedding service.

- `step3_embed_papers_self_hosted.py`
  Variant of the embedding script for a self-hosted OpenAI-compatible embedding service.

- `step4_add_jif.py`
  Add JIF, quartile, scope text, and related journal metadata.

- `compute_ssi.py`
  Compute the Scholar Synergy Index (SSI).

- `build_in_scope_benchmark.py`
  Build in-scope benchmark instances from SSI-stratified journal pools.

- `build_scope_mismatch_benchmark.py`
  Build scope-mismatch instances within the same broad area.

- `experiment_keyword_noise.py`
  Build keyword-noise robustness variants.

- `experiment_synthetic_degradation.py`
  Build synthetic-degradation robustness variants.

- `experiment_utils.py`
  Shared helpers for the robustness experiments, including prompt builders, sharding helpers, JSON repair, resumable output handling, and synthetic-perturbation utilities.

Most construction scripts intentionally keep placeholder paths such as `YOUR_INPUT_JSONL_PATH` or `YOUR_OUTPUT_DIR`. Replace these values before running them in a new environment.

The two robustness builders depend on a base real-submission dataset that is not currently released in this snapshot. To reproduce those experiments from scratch, pass `--dataset` explicitly or set:

```bash
export REAL_SUBMISSION_DATASET_PATH=/path/to/your/base_real_submission_dataset.json
```

## Inference Scripts

The `inference/` directory contains two single-stage inference scripts:

- `inference_single_stage_direct.py`
  Direct prompting without an explicit editorial rationale field.

- `inference_single_stage_rationale_first.py`
  Rationale-first prompting that asks the model to produce a short editorial justification before the final structured answer.

Both scripts use environment variables for model access:

```bash
export LLM_API_KEY=...
export LLM_BASE_URL=...
export LLM_MODEL_NAME=...
```

They also expect local paths in the file-level constants:

- `DATASET_PATH`
- `OUTPUT_RESULTS_PATH`

## Evaluation

The `evaluation/evaluate_predictions.py` script evaluates prediction files in `json` or `jsonl` format and reports:

- decision metrics such as Accuracy, Macro-F1, and Balanced Accuracy
- branch-specific quality and rejection-reason metrics
- transfer-ranking metrics such as MRR, Hit@k, and NDCG@k
- confidence calibration metrics

Example usage:

```bash
python evaluation/evaluate_predictions.py \
  --results /path/to/predictions.json \
  --benchmark dataset/benchmark2025.json
```

To skip plotting entirely:

```bash
python evaluation/evaluate_predictions.py \
  --results /path/to/predictions.json \
  --benchmark dataset/benchmark2025.json \
  --no-plots
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example Usage

Run direct prompting:

```bash
python inference/inference_single_stage_direct.py
```

Run rationale-first prompting:

```bash
python inference/inference_single_stage_rationale_first.py
```

Run evaluation:

```bash
python evaluation/evaluate_predictions.py \
  --results /path/to/predictions.json \
  --benchmark dataset/benchmark2025.json
```

## Current Scope of This Repository

This repository snapshot is already useful for:

- inspecting the released benchmark format
- understanding the main dataset construction pipeline
- reproducing direct and rationale-first inference logic
- running the released evaluation workflow
- inspecting the shared utilities used by the robustness experiments

At the same time, some assets required for full end-to-end reproduction are still not included in this snapshot. See the recommendations below before turning this into a fully self-contained public release.

## Recommended Additions for a Complete Public Release

The following files would substantially improve reproducibility and usability:

1. `dataset/benchmark2025_real_submission_all.json`
   This base real-submission split is useful for reproducing the robustness experiments from scratch.

2. `prompts/`
   Saving the exact prompt templates as standalone text or JSON files makes the prompting setup easier to audit.

3. `scripts/`
   Lightweight launch scripts for sharded inference or batch evaluation would make reruns easier.

4. `docs/dataset_schema.md`
   A short schema document would help users understand all fields without opening the raw JSON.

5. `LICENSE`
   A license file is important before public release.

6. `CITATION.cff`
   This makes the repository easier to cite correctly.

7. `figures/`
   Benchmark overview figures from the paper are useful for quick orientation.

8. `results/` or `examples/`
    A small sanitized prediction example can help users verify their pipeline output format.

## Notes

- The released scripts are research code, not packaged library code.
- Several scripts use hard-coded placeholder paths by design.
- The robustness-generation scripts require a base real-submission dataset that is not included in the current release snapshot.
- The evaluation script depends on plotting libraries listed in `requirements.txt`; `scipy` is optional and is used only for Kendall correlation.
- Before public release, it is worth checking all scripts for private endpoints and environment-specific assumptions.
