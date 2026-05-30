"""Add JIF, quartile, and aim-scope metadata to paper records and filter invalid journals.
The input can be JSONL, a JSON list, or a JSON object containing a records field."""

import csv
import json
import os
import re
import unicodedata

_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_PAREN_RE = re.compile(r"\([^)]*\)")
_LEADING_THE_RE = re.compile(r"^the\s+")
TOP_LEVEL_CATEGORY_MAP = {
    "CELL BIOLOGY": "Biology",
    "MYCOLOGY": "Biology",
    "MATERIALS SCIENCE, CERAMICS": "Materials Science",
    "MATERIALS SCIENCE, COMPOSITES": "Materials Science",
    "CHEMISTRY, INORGANIC": "Chemistry",
    "CHEMISTRY, MEDICINAL": "Chemistry",
    "DERMATOLOGY": "Medicine",
    "OPHTHALMOLOGY": "Medicine",
}


def _strip_accents(text):
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def normalize_journal_name(name):
    """Loosely normalize journal names for stable matching between WoS and JCR fields."""
    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    text = _strip_accents(text)
    text = _PAREN_RE.sub(" ", text)
    text = text.replace("&", " and ")
    text = _LEADING_THE_RE.sub("", text.strip())
    text = _PUNCT_RE.sub("", text)
    return text or None


def normalize_wos_category(category_name):
    if category_name is None:
        return ""
    text = str(category_name).strip()
    if text == "CHEMISTRY, INORGANIC & NUCLEAR":
        return "CHEMISTRY, INORGANIC"
    return text


def rename_ssi_fields(paper_data):
    for old_key, new_key in (
        ("ssi__SSI", "SSI"),
        ("ssi__SSI_Group", "SSI_Group"),
        ("ssi__h5-index", "h5-index"),
    ):
        if old_key in paper_data:
            paper_data[new_key] = paper_data.pop(old_key)

def iter_paper_records(data_file):
    """Yield paper records from JSONL, a JSON list, or grouped JSON with a records field."""
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        return

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record
        return

    if isinstance(data, list):
        for record in data:
            if isinstance(record, dict):
                yield record
        return

    if isinstance(data, dict):
        records = data.get("records")
        if isinstance(records, dict):
            for journal_records in records.values():
                if isinstance(journal_records, list):
                    for record in journal_records:
                        if isinstance(record, dict):
                            yield record
        elif isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    yield record
        else:
            # Allow a single paper record stored directly as one JSON object.
            if "ISSN" in data or "eISSN" in data or "Source Title" in data:
                yield data


def merge_and_filter_journal_data(data_file, kept_journals_file, output_file, aims_scope_file):
    # Load kept_journals.csv using JCR Journal Name as the primary key.
    journal_lookup = {}
    duplicate_journals = []
    try:
        with open(kept_journals_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            required_cols = {"JCR Journal Name", "2024 JIF", "JIF Quartile"}
            missing_cols = required_cols - set(reader.fieldnames or [])
            if missing_cols:
                raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")

            for row in reader:
                jcr_name = row.get("JCR Journal Name")
                norm_name = normalize_journal_name(jcr_name)
                if not norm_name:
                    continue
                if norm_name in journal_lookup:
                    duplicate_journals.append(str(jcr_name).strip())
                    continue
                journal_lookup[norm_name] = row
    except Exception as e:
        print(f"Failed to read kept_journals file: {e}")
        return

    # Load aim_scope data using uppercase journal names for case-insensitive matching.
    aims_scope_lookup = {}
    if aims_scope_file and os.path.exists(aims_scope_file):
        try:
            with open(aims_scope_file, 'r', encoding='utf-8') as f:
                raw_aims = json.load(f)
                for name, scope in raw_aims.items():
                    if name and scope is not None:
                        aims_scope_lookup[name.strip().upper()] = scope
        except Exception as e:
            print(f"Failed to read aim_scope file; aim_scope will be omitted: {e}")

    # Processing counters.
    stats = {
        "total_processed": 0,
        "saved": 0,
        "dropped_no_match": 0,
        "dropped_empty_metrics": 0,
    }

    # Track unmatched journals without duplicates.
    not_found_journals = set()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for paper_data in iter_paper_records(data_file):
            stats["total_processed"] += 1

            current_source_title = str(paper_data.get("Source Title", "")).strip()
            current_source_norm = normalize_journal_name(current_source_title)

            should_save = False

            if current_source_norm and current_source_norm in journal_lookup:
                matched_info = journal_lookup[current_source_norm]

                jif = matched_info.get("2024 JIF")
                quartile = matched_info.get("JIF Quartile")
                journal_category = normalize_wos_category(matched_info.get("Category"))
                top_level_category = TOP_LEVEL_CATEGORY_MAP.get(journal_category, "")
                journal_name = (
                    matched_info.get("JCR Journal Name")
                    or matched_info.get("Journal Name")
                    or current_source_title
                )

                # JIF and quartile must both be present.
                if (
                    jif is not None
                    and str(jif).strip() != ""
                    and quartile is not None
                    and str(quartile).strip() != ""
                ):
                    rename_ssi_fields(paper_data)
                    paper_data["JIF"] = jif
                    paper_data["JIF_Quartile"] = quartile
                    paper_data["WoS Categories"] = journal_category
                    paper_data["Category"] = top_level_category
                    paper_data["journal"] = journal_name
                    # Add aim_scope by normalized journal name when available.
                    if aims_scope_lookup and journal_name:
                        aim_scope = aims_scope_lookup.get(str(journal_name).strip().upper(), "")
                        paper_data["aim_scope"] = aim_scope if aim_scope else ""
                    else:
                        paper_data["aim_scope"] = ""
                    should_save = True
                else:
                    stats["dropped_empty_metrics"] += 1
            else:
                stats["dropped_no_match"] += 1
                # Fall back to ISSN/eISSN if the journal title is empty.
                journal_identifier = current_source_title
                if not journal_identifier:
                    current_issn = str(paper_data.get("ISSN", "")).strip()
                    current_eissn = str(paper_data.get("eISSN", "")).strip()
                    journal_identifier = f"ISSN:{current_issn}|eISSN:{current_eissn}"
                not_found_journals.add(journal_identifier)

            if should_save:
                f_out.write(json.dumps(paper_data, ensure_ascii=False) + "\n")
                stats["saved"] += 1
    
    if not_found_journals:
        not_found_file = os.path.join(os.path.dirname(output_file), "missing_journals.txt")
        with open(not_found_file, 'w', encoding='utf-8') as f_nf:
            for j in sorted(not_found_journals):
                f_nf.write(f"{j}\n")

if __name__ == "__main__":
    # Fill in these placeholder paths for your environment.
    input_data = "YOUR_INPUT_DATA_PATH"
    kept_journals = "YOUR_JOURNAL_LOOKUP_CSV_PATH"
    output_data = "YOUR_OUTPUT_JSONL_PATH"
    aims_scope_file = "YOUR_AIMS_SCOPE_JSON_PATH"

    if os.path.exists(input_data) and os.path.exists(kept_journals):
        merge_and_filter_journal_data(input_data, kept_journals, output_data, aims_scope_file)
    else:
        print("Error: input files were not found. Please check the configured paths.")
