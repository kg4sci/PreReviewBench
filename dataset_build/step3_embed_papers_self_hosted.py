#!/usr/bin/env python3
"""Generate paper embeddings grouped by journal and write them to cache and output files.
This variant is intended for a self-hosted compatible embedding service."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from openai import OpenAI
from tqdm import tqdm


# =========================
# Default settings
# =========================
METADATA_PATH = "YOUR_METADATA_JSONL_PATH"
OUTPUT_DIR = "YOUR_OUTPUT_DIR"

# Embedding service settings
BASE_URL = "YOUR_EMBEDDING_SERVICE_BASE_URL"
MODEL = "YOUR_EMBEDDING_MODEL_NAME"
API_KEY = "YOUR_EMBEDDING_API_KEY"

BATCH_SIZE = 32
MAX_TEXT_CHARS = 8000

RETRY_MAX_ATTEMPTS = 4
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 15.0
RETRY_JITTER_SECONDS = 0.3

DEFAULT_WORKERS = 8


# =========================
# Utility helpers
# =========================
def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    s = str(value).strip()
    if not s:
        return True
    return s.lower() in {"nan", "none", "null"}


def cell_str(value: Any) -> str:
    if is_empty(value):
        return ""
    return str(value).strip()


def normalize_doi(value: Any) -> str | None:
    if is_empty(value):
        return None
    s = str(value).strip()
    low = s.lower()
    for prefix in ("doi:", "https://doi.org/", "http://doi.org/"):
        if low.startswith(prefix):
            s = s[len(prefix):].strip()
            low = s.lower()
    return s.lower() or None


def normalize_journal_name(name: Any) -> str:
    if name is None:
        return ""
    s = re.sub(r"\s+", " ", str(name)).strip()
    return s.upper()


def safe_filename(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    return s[:200] or "unnamed"


def build_paper_text(paper: dict) -> str:
    title = cell_str(paper.get("Article Title"))
    abstract = cell_str(paper.get("Abstract"))
    keywords = cell_str(paper.get("Keywords"))
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if keywords:
        parts.append(f"Keywords: {keywords}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    text = "\n".join(parts).strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return text


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def paper_cache_key(paper: dict, text: str) -> str:
    """Use the same cache-key scheme as the earlier pipeline for compatibility."""
    doi = normalize_doi(paper.get("DOI"))
    if doi:
        return f"doi:{doi}"
    return f"sha1:{text_hash(text)}"


def normalize_base_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code in {408, 409, 425, 429} or status_code >= 500:
            return True
    msg = str(exc).lower()
    hints = [
        "timeout", "timed out", "connection", "broken pipe",
        "please try again", "server error", "overloaded",
    ]
    return any(h in msg for h in hints)


def backoff_seconds(attempt: int) -> float:
    base = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.0, min(RETRY_JITTER_SECONDS, base * 0.2))
    return base + jitter


# =========================
# Local incremental cache used for resume support only
# =========================
class LocalCache:
    def __init__(self, path: Path):
        self.path = path
        self._mem: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._mem)

    def has(self, key: str) -> bool:
        return key in self._mem

    def get(self, key: str) -> np.ndarray | None:
        return self._mem.get(key)

    def load_self(self) -> None:
        if not self.path.is_file():
            return
        added = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    k = obj["key"]
                    if k in self._mem:
                        continue
                    self._mem[k] = np.asarray(obj["vec"], dtype=np.float32)
                    added += 1
                except Exception:
                    continue

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        if not items:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for key, vec in items:
                    if key in self._mem:
                        continue
                    self._mem[key] = vec
                    f.write(json.dumps(
                        {"key": key, "dim": int(vec.shape[0]), "vec": vec.tolist()},
                        ensure_ascii=False,
                    ) + "\n")


# =========================
# Metadata loading
# =========================
def iter_metadata(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# =========================
# Embedding calls
# =========================
def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[np.ndarray]:
    last_err: Exception | None = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
            vectors = [np.asarray(d.embedding, dtype=np.float32) for d in data]
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding result count ({len(vectors)}) does not match input count ({len(texts)})."
                )
            return vectors
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= RETRY_MAX_ATTEMPTS or not is_retryable_error(exc):
                break
            wait = backoff_seconds(attempt)
            tqdm.write(
                f"Embedding request failed ({attempt}/{RETRY_MAX_ATTEMPTS}): {exc}. Retrying in {wait:.2f}s."
            )
            time.sleep(wait)
    raise RuntimeError(f"Embedding request failed after retries: {last_err}")


# =========================
# Per-journal output
# =========================
def save_journal_npz(
    out_dir: Path,
    journal_norm: str,
    journal_display: str,
    items: list[dict],
    cache: LocalCache,
    model: str,
) -> tuple[Path | None, int]:
    rows = []
    vecs = []
    for it in items:
        v = cache.get(it["key"])
        if v is None:
            continue
        rows.append(it)
        vecs.append(v)

    if not rows:
        return None, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = safe_filename(journal_norm)
    npz_path = out_dir / f"{fname}.npz"
    meta_path = out_dir / f"{fname}.jsonl"

    embeddings = np.vstack(vecs).astype(np.float32)
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        keys=np.array([r["key"] for r in rows]),
        dois=np.array([r["doi"] for r in rows]),
        titles=np.array([r["title"] for r in rows]),
        source_titles=np.array([r["source_title"] for r in rows]),
        years=np.array([r["year"] for r in rows]),
        line_idx=np.array([r["line_idx"] for r in rows], dtype=np.int64),
        journal_normalized=np.array([journal_norm]),
        journal_display=np.array([journal_display]),
        model=np.array([model]),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "line_idx": r["line_idx"],
                "key": r["key"],
                "doi": r["doi"],
                "title": r["title"],
                "source_title": r["source_title"],
                "year": r["year"],
            }, ensure_ascii=False) + "\n")

    return npz_path, len(rows)


# =========================
# Main flow
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute and store paper embeddings grouped by journal.")
    p.add_argument("--metadata", default=METADATA_PATH)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--base-url", default=BASE_URL, help="Compatible embedding service URL.")
    p.add_argument("--model", default=MODEL, help="Model name exposed by the service.")
    p.add_argument("--api-key", default=os.environ.get("EMBEDDING_API_KEY", API_KEY),
                   help="API key used to access the service.")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Client-side concurrency.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of papers per request.")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N records for debugging.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore the local paper_embedding_cache.jsonl file and recompute everything.")
    p.add_argument("--dry-run", action="store_true", help="Estimate pending work without making API requests.")
    p.add_argument("--healthcheck", action="store_true",
                   help="Verify service availability with a sample request before running.")
    return p.parse_args()


def healthcheck(client: OpenAI, model: str) -> None:
    print(f"[health] Probing {model} ... ", end="", flush=True)
    t0 = time.time()
    try:
        r = client.embeddings.create(model=model, input=["healthcheck"])
        dim = len(r.data[0].embedding)
        print(f"OK (dim={dim}, {time.time()-t0:.2f}s)")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(2)


def main() -> None:
    args = parse_args()

    global BATCH_SIZE
    BATCH_SIZE = max(1, int(args.batch_size))

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    per_journal_dir = output_dir / "per_journal_papers"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_journal_dir.mkdir(parents=True, exist_ok=True)

    # 1. Local resume cache.
    cache = LocalCache(output_dir / "paper_embedding_cache.jsonl")
    if not args.no_resume:
        cache.load_self()

    # 2. Load metadata and group items by journal.
    journal_items: dict[str, list[dict]] = defaultdict(list)
    journal_display: dict[str, str] = {}
    missing_text: list[dict] = []

    total_papers = 0
    for idx, p in enumerate(iter_metadata(metadata_path)):
        if args.limit > 0 and idx >= args.limit:
            break
        total_papers += 1

        text = build_paper_text(p)
        raw_src = cell_str(p.get("Source Title"))
        jnorm = normalize_journal_name(raw_src) or "__UNKNOWN__"
        if jnorm not in journal_display:
            journal_display[jnorm] = raw_src or "(unknown journal)"

        if not text:
            missing_text.append({
                "line_idx": idx,
                "doi": cell_str(p.get("DOI")),
                "title": cell_str(p.get("Article Title")),
                "source_title": raw_src,
                "reason": "empty_text",
            })
            continue

        journal_items[jnorm].append({
            "line_idx": idx,
            "key": paper_cache_key(p, text),
            "text": text,
            "doi": cell_str(p.get("DOI")),
            "title": cell_str(p.get("Article Title")),
            "source_title": raw_src,
            "year": cell_str(p.get("Publication Year")),
        })

    if missing_text:
        miss_path = output_dir / "missing_keys.jsonl"
        with open(miss_path, "w", encoding="utf-8") as f:
            for r in missing_text:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3. Build a de-duplicated set of pending keys.
    pending: dict[str, dict] = {}
    total_items = 0
    total_hits = 0
    for jnorm, items in journal_items.items():
        for it in items:
            total_items += 1
            if cache.has(it["key"]):
                total_hits += 1
            elif it["key"] not in pending:
                pending[it["key"]] = {"key": it["key"], "text": it["text"]}
    pending_list = list(pending.values())

    if args.dry_run:
        return

    # 4. API client
    base_url = normalize_base_url(args.base_url)
    api_key = args.api_key.strip() or "EMPTY"
    client = OpenAI(api_key=api_key, base_url=base_url)
    workers = max(1, int(args.workers))

    if args.healthcheck:
        healthcheck(client, args.model)

    # 5. Concurrent requests
    if pending_list:
        batches: list[list[dict]] = [
            pending_list[i:i + BATCH_SIZE] for i in range(0, len(pending_list), BATCH_SIZE)
        ]

        bar = tqdm(total=len(pending_list), desc="embedding", unit="paper", dynamic_ncols=True)
        new_calls_done = 0
        last_log = time.time()

        def run_batch(batch: list[dict]) -> tuple[list[dict], list[np.ndarray] | None, str]:
            try:
                texts = [b["text"] for b in batch]
                vecs = embed_batch(client, args.model, texts)
                return batch, vecs, ""
            except Exception as exc:
                return batch, None, str(exc)[:300]

        if workers == 1:
            for batch in batches:
                _, vecs, err = run_batch(batch)
                if vecs is None:
                    tqdm.write(f"  [FAIL] batch size={len(batch)} err={err}")
                    bar.update(len(batch))
                    continue
                cache.put_many([(b["key"], v) for b, v in zip(batch, vecs)])
                bar.update(len(batch))
                new_calls_done += len(batch)
        else:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                fut_list = [ex.submit(run_batch, b) for b in batches]
                for fut in cf.as_completed(fut_list):
                    batch, vecs, err = fut.result()
                    if vecs is None:
                        tqdm.write(f"  [FAIL] batch size={len(batch)} err={err}")
                        bar.update(len(batch))
                        continue
                    cache.put_many([(b["key"], v) for b, v in zip(batch, vecs)])
                    bar.update(len(batch))
                    new_calls_done += len(batch)
                    now = time.time()
                    if now - last_log > 60:
                        tqdm.write(f"  [heartbeat] completed {new_calls_done}/{len(pending_list)}")
                        last_log = now
        bar.close()

    # 6. Write grouped journal outputs.
    summary_path = output_dir / "journal_summary.jsonl"
    journals_with_any = 0
    total_saved = 0
    with open(summary_path, "w", encoding="utf-8") as fsum:
        for jnorm in tqdm(sorted(journal_items.keys()), desc="journal-save", unit="journal", dynamic_ncols=True):
            items = journal_items[jnorm]
            disp = journal_display.get(jnorm, jnorm)
            n_total = len(items)

            npz_path, n_saved = save_journal_npz(
                out_dir=per_journal_dir,
                journal_norm=jnorm,
                journal_display=disp,
                items=items,
                cache=cache,
                model=args.model,
            )
            if n_saved > 0:
                journals_with_any += 1
                total_saved += n_saved

            fsum.write(json.dumps({
                "journal_normalized": jnorm,
                "journal_display": disp,
                "papers_total": n_total,
                "papers_with_embedding": n_saved,
                "papers_missing": n_total - n_saved,
                "file": str(npz_path) if npz_path else "",
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
