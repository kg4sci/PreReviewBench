#!/usr/bin/env python3
"""Generate paper embeddings grouped by journal and write cache plus per-journal outputs.
Supports warming from legacy caches and parallel requests with multiple API keys."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import itertools
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
# Paths and runtime settings
# =========================
METADATA_PATH = "YOUR_METADATA_JSONL_PATH"
OUTPUT_DIR = "YOUR_OUTPUT_DIR"

LEGACY_EMBED_DIRS = [
    "YOUR_LEGACY_EMBED_DIR_01",
    "YOUR_LEGACY_EMBED_DIR_02",
    "YOUR_LEGACY_EMBED_DIR_03",
    "YOUR_LEGACY_EMBED_DIR_04",
    "YOUR_LEGACY_EMBED_DIR_05",
]

BASE_URL = "YOUR_EMBEDDING_SERVICE_BASE_URL"
MODEL = "YOUR_EMBEDDING_MODEL_NAME"

BATCH_SIZE = 16

# Maximum text length per paper. Keep this aligned with legacy cache generation.
MAX_TEXT_CHARS = 8000

RETRY_MAX_ATTEMPTS = 6
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 30.0
RETRY_JITTER_SECONDS = 0.3

SLEEP_SECONDS = 0.05

DEFAULT_WORKERS = 1

# Flush a journal snapshot every N new API calls. Use 0 to flush only at the end.
FLUSH_EVERY_N_NEW = 2000


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
        "429", "rate limit", "too many requests", "overloaded", "saturated",
        "timeout", "timed out", "connection", "please try again", "server error",
    ]
    return any(h in msg for h in hints)


def backoff_seconds(attempt: int) -> float:
    base = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.0, min(RETRY_JITTER_SECONDS, base * 0.2))
    return base + jitter


# =========================
# Cache
# =========================
class EmbeddingCache:
    """Disk-backed JSONL cache storing one {key, dim, vec} record per line."""

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
        if self.path.is_file():
            self._load_jsonl(self.path)

    def warm_from_legacy(self, legacy_dirs: Iterable[str]) -> None:
        for d in legacy_dirs:
            base = Path(d)
            if not base.is_dir():
                continue

            jsonl = base / "paper_embedding_cache.jsonl"
            if jsonl.is_file():
                self._load_jsonl(jsonl)

            per_dir = base / "per_journal_papers"
            if per_dir.is_dir():
                self._load_per_journal(per_dir)

    def _load_jsonl(self, path: Path) -> int:
        added = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = obj["key"]
                        if key in self._mem:
                            continue
                        vec = np.asarray(obj["vec"], dtype=np.float32)
                        self._mem[key] = vec
                        added += 1
                    except Exception:
                        continue
        except Exception as exc:
            print(f"Failed to read cache file {path}: {exc}", file=sys.stderr)
        return added

    def _load_per_journal(self, per_dir: Path) -> int:
        added = 0
        for fp in sorted(per_dir.glob("*.npz")):
            try:
                z = np.load(fp, allow_pickle=True)
            except Exception as exc:
                print(f"Skipping unreadable cache archive {fp.name}: {exc}", file=sys.stderr)
                continue
            if "keys" not in z.files or "embeddings" not in z.files:
                continue
            keys = z["keys"]
            embs = z["embeddings"]
            if len(keys) != len(embs):
                continue
            for k, v in zip(keys, embs):
                ks = str(k)
                if not ks or ks in self._mem:
                    continue
                try:
                    self._mem[ks] = np.asarray(v, dtype=np.float32)
                    added += 1
                except Exception:
                    continue
        return added

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
            data = resp.data
            data_sorted = sorted(data, key=lambda d: getattr(d, "index", 0))
            vectors = [np.asarray(d.embedding, dtype=np.float32) for d in data_sorted]
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
    cache: EmbeddingCache,
    model: str,
) -> tuple[Path | None, int]:
    """Write all available embeddings for a journal to NPZ and JSONL files.

    Returns (npz_path, rows_written). If no embedding is available, returns (None, 0).
    """
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
    keys = np.array([r["key"] for r in rows])
    dois = np.array([r["doi"] for r in rows])
    titles = np.array([r["title"] for r in rows])
    sources = np.array([r["source_title"] for r in rows])
    years = np.array([r["year"] for r in rows])
    line_idx = np.array([r["line_idx"] for r in rows], dtype=np.int64)

    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        keys=keys,
        dois=dois,
        titles=titles,
        source_titles=sources,
        years=years,
        line_idx=line_idx,
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
    p.add_argument("--legacy-dirs", default=",".join(LEGACY_EMBED_DIRS),
                   help="Comma-separated legacy embedding directories used to warm the cache.")
    p.add_argument("--skip-warm", action="store_true", help="Skip cache warming from legacy directories.")
    p.add_argument("--api-key", default=os.environ.get("EMBEDDING_API_KEY", ""))
    p.add_argument("--api-keys", default="", help="Comma-separated API keys for parallel requests.")
    p.add_argument("--api-keys-file", default="", help="File containing one API key per line. Lines starting with # are ignored.")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--limit", type=int, default=0, help="Process only the first N records for debugging. Use 0 for all records.")
    p.add_argument("--dry-run", action="store_true", help="Estimate pending work without making API requests.")
    return p.parse_args()


def load_api_keys(single_key: str, api_keys_csv: str, api_keys_file: str) -> list[str]:
    keys: list[str] = []
    if api_keys_csv:
        for raw in api_keys_csv.split(","):
            s = raw.strip()
            if s:
                keys.append(s)
    if api_keys_file:
        p = Path(api_keys_file)
        if not p.is_file():
            raise FileNotFoundError(f"--api-keys-file does not exist: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                keys.append(s)
    if single_key.strip():
        keys.append(single_key.strip())
    dedup: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        dedup.append(k)
    return dedup


def main() -> None:
    args = parse_args()

    global BATCH_SIZE
    BATCH_SIZE = max(1, int(args.batch_size))

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    per_journal_dir = output_dir / "per_journal_papers"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_journal_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cache
    cache = EmbeddingCache(output_dir / "paper_embedding_cache.jsonl")
    cache.load_self()
    if not args.skip_warm:
        legacy_dirs = [s.strip() for s in args.legacy_dirs.split(",") if s.strip()]
        cache.warm_from_legacy(legacy_dirs)

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

    # 3. Find pending records across journals and de-duplicate by cache key.
    all_pending: dict[str, dict] = {}
    total_items = 0
    total_hits = 0
    for jnorm, items in journal_items.items():
        for it in items:
            total_items += 1
            if cache.has(it["key"]):
                total_hits += 1
            elif it["key"] not in all_pending:
                all_pending[it["key"]] = {"key": it["key"], "text": it["text"]}
    pending_list = list(all_pending.values())

    if args.dry_run:
        return

    # 4. API calls
    if pending_list:
        api_keys = load_api_keys(args.api_key, args.api_keys, args.api_keys_file)
        if not api_keys:
            print(
                "Error: {} keys are still pending, but no API key was provided. "
                "Use --api-key, EMBEDDING_API_KEY, --api-keys, or --api-keys-file."
                .format(len(pending_list)),
                file=sys.stderr,
            )
            sys.exit(1)

        base_url = normalize_base_url(args.base_url)
        workers = max(1, int(args.workers))
        if workers > len(api_keys):
            workers = len(api_keys)
        clients = [OpenAI(api_key=k, base_url=base_url or None) for k in api_keys]

        batches: list[list[dict]] = [
            pending_list[i:i + BATCH_SIZE] for i in range(0, len(pending_list), BATCH_SIZE)
        ]

        bar = tqdm(total=len(pending_list), desc="embedding", unit="paper", dynamic_ncols=True)
        new_calls_done = 0

        def run_batch(batch: list[dict], client: OpenAI) -> tuple[list[dict], list[np.ndarray] | None, str]:
            try:
                texts = [b["text"] for b in batch]
                vecs = embed_batch(client, args.model, texts)
                return batch, vecs, ""
            except Exception as exc:
                return batch, None, str(exc)[:300]

        if workers == 1:
            client = clients[0]
            for batch in batches:
                _, vecs, err = run_batch(batch, client)
                if vecs is None:
                    tqdm.write(f"  [FAIL] batch size={len(batch)} err={err}")
                    bar.update(len(batch))
                    continue
                cache.put_many([(b["key"], v) for b, v in zip(batch, vecs)])
                bar.update(len(batch))
                new_calls_done += len(batch)
                if SLEEP_SECONDS > 0:
                    time.sleep(SLEEP_SECONDS)
        else:
            client_cycle = itertools.cycle(clients[:workers])
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                fut_list = []
                for batch in batches:
                    client = next(client_cycle)
                    fut_list.append(ex.submit(run_batch, batch, client))
                for fut in cf.as_completed(fut_list):
                    batch, vecs, err = fut.result()
                    if vecs is None:
                        tqdm.write(f"  [FAIL] batch size={len(batch)} err={err}")
                        bar.update(len(batch))
                        continue
                    cache.put_many([(b["key"], v) for b, v in zip(batch, vecs)])
                    bar.update(len(batch))
                    new_calls_done += len(batch)
        bar.close()

    # 5. Write grouped journal outputs.
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
