"""Filter target-journal papers from metadata Excel files and export JSONL.
The script also handles field cleanup, validation, document-type filtering, and DOI deduplication."""

import glob
import json
import math
import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd

# Configured paths
SOURCE_ROOT = "YOUR_SOURCE_ROOT_DIR"
SSI_TABLE = "YOUR_SSI_TABLE_CSV_PATH"
OUTPUT_FILE = "YOUR_OUTPUT_JSONL_PATH"

TARGET_COLUMNS = [
    "Article Title", "Source Title", "Document Type", "Author Keywords",
    "Keywords Plus", "Abstract", "Times Cited, All Databases",
    "ISSN", "eISSN", "Publication Year", "DOI", "WoS Categories",
]

ALLOWED_DOC_TYPES = {"article", "article; early access"}
REQUIRED_FIELDS = (
    "Article Title",
    "Source Title",
    "Document Type",
    "Abstract",
    "Times Cited, All Databases",
    "Publication Year",
    "DOI",
    "WoS Categories",
    "Keywords",
)
REQUIRED_ANY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ISSN", "eISSN"),
)

# SSI-related fields injected into each output record
SSI_FIELDS = ("SSI", "h5-index", "SSI_Group")

# ---------------------------------------------------------------------------
# Journal-name normalization and whitelist loading
# ---------------------------------------------------------------------------
# Matching strategy: use "JCR Journal Name" from the SSI table as the whitelist
# key, and match it against normalized "Source Title" values from WoS metadata.
# In practice this mainly requires normalization of case, spacing, punctuation,
# and accents rather than a separate alias table.

_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_PAREN_RE = re.compile(r"\([^)]*\)")
_LEADING_THE_RE = re.compile(r"^the\s+")

def _strip_accents(s: str) -> str:
    """Remove accents such as 'e' from 'é' to make journal-name matching more robust."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def normalize_journal_name(name) -> str | None:
    """Loosely normalize a journal name so WoS and JCR names match reliably.

    Steps:
    1. Lowercase and trim whitespace
    2. Remove accents
    3. Remove parenthetical content
    4. Replace '&' with 'and'
    5. Drop a leading 'the'
    6. Remove non-alphanumeric characters
    Returns a compact string, or None if the result is empty.
    """
    if name is None:
        return None
    s = str(name).strip().lower()
    if not s:
        return None
    s = _strip_accents(s)
    s = _PAREN_RE.sub(" ", s)
    s = s.replace("&", " and ")
    s = _LEADING_THE_RE.sub("", s.strip())
    s = _PUNCT_RE.sub("", s)
    return s or None

def load_ssi_whitelist(ssi_path: str) -> tuple[dict[str, dict], dict[str, str]]:
    """Load an SSI whitelist keyed by normalized JCR journal names.

    Returns:
    - whitelist: normalized(JCR Journal Name) -> dict containing SSI_FIELDS
    - norm_to_name: normalized(JCR Journal Name) -> original JCR Journal Name
    """
    df = pd.read_csv(ssi_path)
    if "JCR Journal Name" not in df.columns:
        raise ValueError(f"SSI table {ssi_path} is missing the 'JCR Journal Name' column")

    whitelist: dict[str, dict] = {}
    norm_to_name: dict[str, str] = {}
    skipped_empty = 0
    conflicts: list[tuple[str, str, str]] = []

    for _, row in df.iterrows():
        jcr_name = row.get("JCR Journal Name")
        if pd.isna(jcr_name) or not str(jcr_name).strip():
            skipped_empty += 1
            continue
        norm = normalize_journal_name(jcr_name)
        if not norm:
            skipped_empty += 1
            continue
        info = {k: (None if pd.isna(row.get(k)) else row.get(k))
                for k in SSI_FIELDS if k in df.columns}
        if norm in whitelist:
            conflicts.append((norm, norm_to_name.get(norm), str(jcr_name)))
            continue
        whitelist[norm] = info
        norm_to_name[norm] = str(jcr_name).strip()

    return whitelist, norm_to_name

# ---------------------------------------------------------------------------
# Cleaning helpers kept compatible with the earlier pipeline
# ---------------------------------------------------------------------------
def normalize_doi(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    low = s.lower()
    if low.startswith("doi:"):
        s = s[4:].strip()
        low = s.lower()
    if low.startswith("https://doi.org/"):
        s = s[len("https://doi.org/"):].strip()
    elif low.startswith("http://doi.org/"):
        s = s[len("http://doi.org/"):].strip()
    return s.strip().lower() or None

def normalize_doc_type(v):
    return " ".join(str(v or "").strip().lower().split())

def doc_type_rank(doc_type_norm: str) -> int:
    if doc_type_norm == "article":
        return 2
    if doc_type_norm == "article; early access":
        return 1
    return 0

def list_excel_files_recursive(source_root: str) -> list[str]:
    """Recursively list Excel files under source_root in sorted order."""
    files = glob.glob(os.path.join(source_root, "**", "*.xls*"), recursive=True)
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    return sorted(files)

def read_excel(file_path: str) -> pd.DataFrame | None:
    engine = "xlrd" if file_path.lower().endswith(".xls") else "openpyxl"
    try:
        return pd.read_excel(file_path, engine=engine)
    except Exception as e:
        print(f"  Failed to read {file_path}: {e}")
        return None

def ensure_target_columns(df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    for col in target_columns:
        if col in df.columns:
            continue
        renamed = False
        for df_col in df.columns:
            if str(df_col).strip().lower() == col.strip().lower():
                df = df.rename(columns={df_col: col})
                renamed = True
                break
        if not renamed:
            df[col] = None
    return df

def clean_nan(record: dict) -> dict:
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, float) and math.isnan(v):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned

def is_effectively_empty(v) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if not s:
        return True
    return s.lower() in {"nan", "none"}

def merge_keywords(author_keywords, keywords_plus) -> str | None:
    parts = []
    for value in (author_keywords, keywords_plus):
        if is_effectively_empty(value):
            continue
        parts.append(str(value).strip())
    if not parts:
        return None
    return "; ".join(parts)

def with_merged_keywords(record: dict) -> dict:
    merged = dict(record)
    merged["Keywords"] = merge_keywords(
        merged.get("Author Keywords"),
        merged.get("Keywords Plus"),
    )
    merged.pop("Author Keywords", None)
    merged.pop("Keywords Plus", None)
    return merged

def missing_required_fields(record: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if is_effectively_empty(record.get(f))]

def missing_required_any_groups(record: dict) -> list[tuple[str, ...]]:
    return [
        group for group in REQUIRED_ANY_GROUPS
        if all(is_effectively_empty(record.get(f)) for f in group)
    ]

def group_label(group: tuple[str, ...]) -> str:
    return "|".join(group)

def should_keep_doc_type(record: dict) -> tuple[bool, str]:
    dt_norm = normalize_doc_type(record.get("Document Type"))
    return (dt_norm in ALLOWED_DOC_TYPES), dt_norm

def normalize_record(record: dict) -> dict:
    cleaned = clean_nan(record)
    return with_merged_keywords(cleaned)

def is_record_eligible(record: dict) -> tuple[bool, str, str | None]:
    keep_doc_type, dt_norm = should_keep_doc_type(record)
    if not keep_doc_type:
        return False, dt_norm, None
    doi_norm = normalize_doi(record.get("DOI"))
    if not doi_norm:
        return False, dt_norm, None
    return True, dt_norm, doi_norm

def merge_by_doi(doi_best: dict[str, dict], doi_norm: str, rank: int, record: dict) -> None:
    current = doi_best.get(doi_norm)
    if current is None or rank > current["rank"]:
        doi_best[doi_norm] = {"rank": rank, "record": record}

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def process_files(
    source_root: str = SOURCE_ROOT,
    ssi_path: str = SSI_TABLE,
    output_file: str = OUTPUT_FILE,
    target_columns: list[str] = TARGET_COLUMNS,
):
    """Filter and normalize Excel metadata, then export deduplicated JSONL records."""
    if not os.path.isdir(source_root):
        print(f"Error: source root {source_root} not found.")
        return
    if not os.path.isfile(ssi_path):
        print(f"Error: SSI table {ssi_path} not found.")
        return

    whitelist, norm_to_name = load_ssi_whitelist(ssi_path)

    all_files = list_excel_files_recursive(source_root)

    # doi_norm -> {"rank": int, "record": dict}
    doi_best: dict[str, dict] = {}

    stats = {
        "excel_files_total": len(all_files),
        "excel_files_read": 0,
        "rows_total": 0,
        "rows_dropped_journal_not_in_ssi": 0,
        "rows_dropped_required_fields": 0,
        "rows_dropped_doc_type": 0,
        "rows_dropped_no_doi": 0,
        "written_total": 0,
        "rows_dropped_per_field": {field: 0 for field in REQUIRED_FIELDS},
        "rows_dropped_per_group": {group_label(g): 0 for g in REQUIRED_ANY_GROUPS},
    }

    for file_path in all_files:
        # Skip the output file itself if it happens to be placed under the source tree.
        if os.path.abspath(file_path) == os.path.abspath(output_file):
            continue

        df = read_excel(file_path)
        if df is None:
            continue
        stats["excel_files_read"] += 1

        df = ensure_target_columns(df, target_columns)
        records = df[target_columns].to_dict(orient="records")

        for record in records:
            stats["rows_total"] += 1

            # Match against the whitelist using the original Source Title.
            src_norm = normalize_journal_name(record.get("Source Title"))
            ssi_info = whitelist.get(src_norm) if src_norm else None
            if ssi_info is None:
                stats["rows_dropped_journal_not_in_ssi"] += 1
                continue

            clean_record = normalize_record(record)
            missing = missing_required_fields(clean_record)
            missing_groups = missing_required_any_groups(clean_record)
            if missing or missing_groups:
                stats["rows_dropped_required_fields"] += 1
                for field in missing:
                    stats["rows_dropped_per_field"][field] += 1
                for group in missing_groups:
                    stats["rows_dropped_per_group"][group_label(group)] += 1
                continue

            eligible, dt_norm, doi_norm = is_record_eligible(clean_record)
            if not eligible:
                if dt_norm not in ALLOWED_DOC_TYPES:
                    stats["rows_dropped_doc_type"] += 1
                else:
                    stats["rows_dropped_no_doi"] += 1
                continue

            # Inject SSI metadata without overwriting original fields.
            for k in SSI_FIELDS:
                clean_record[f"ssi__{k}"] = ssi_info.get(k)

            rank = doc_type_rank(dt_norm)
            merge_by_doi(doi_best, doi_norm, rank, clean_record)

    # Write output records.
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    count = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for pack in doi_best.values():
            rec = pack["record"]
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    stats["written_total"] = count


if __name__ == "__main__":
    process_files()
