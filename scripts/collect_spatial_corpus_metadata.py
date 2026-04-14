#!/usr/bin/env python3
"""
Collect metadata from SpatialCorpus-110M h5ad files into a CSV.

Polls DATASET_DIR for new .h5ad files, extracts per-file metadata, and
attempts a PubMed lookup for author-named datasets. Appends incrementally
to OUTPUT_CSV. Runs until DONE_SENTINEL is created or KeyboardInterrupt.

Usage:
    python scripts/collect_spatial_corpus_metadata.py

To signal that the download is complete and the script should exit after
a final pass, create the sentinel file:
    touch /Volumes/processing/SpatialCorpus-110M/.download_complete
"""

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import anndata as ad
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_DIR = Path("/Volumes/processing/SpatialCorpus-110M")
OUTPUT_CSV = Path("/Volumes/processing/SpatialCorpus-110M_metadata.csv")
DONE_SENTINEL = DATASET_DIR / ".download_complete"
POLL_INTERVAL_SECONDS = 60

FIELDNAMES = [
    "filename",
    "n_cells",
    "n_vars",
    "title",
    "assay",
    "organism",
    "tissue",
    "sex",
    "condition_ids",
    "donor_ids",
    "dataset",
    "nicheformer_split",
    "publication_doi",
    "publication_title",
    "error",
]

# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------
PUBMED_ESEARCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    "?db=pubmed&term={query}&retmax=1&retmode=json"
)
PUBMED_ESUMMARY = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    "?db=pubmed&id={pmid}&retmode=json"
)
_HEADERS = {"User-Agent": "SpatialCorpusMetadataScript/1.0"}


def _http_get(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def pubmed_lookup(query: str) -> tuple[str, str]:
    """Return (doi, title) for the top PubMed hit, or ('', '') on failure."""
    try:
        encoded = urllib.parse.quote(query)
        data = _http_get(PUBMED_ESEARCH.format(query=encoded))
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "", ""
        pmid = ids[0]
        summary = _http_get(PUBMED_ESUMMARY.format(pmid=pmid))
        result = summary.get("result", {}).get(pmid, {})
        pub_title = result.get("title", "")
        doi = next(
            (a["value"] for a in result.get("articleids", []) if a.get("idtype") == "doi"),
            "",
        )
        return doi, pub_title
    except Exception:
        return "", ""


# Tech/platform prefixes that won't map to a single paper
_NO_PUBMED_PREFIXES = re.compile(
    r"^(10xgenomics|vizgen|nanostring|akoya|codex|slideseq|stereo|resolve|seqfish|merscope)",
    re.IGNORECASE,
)


def build_pubmed_query(stem: str) -> str | None:
    """
    Convert a filename stem to a PubMed query.
    Returns None if the file is not author-named (e.g. platform demos).
    """
    clean = stem.replace("-", "_")
    parts = clean.split("_")
    if not parts:
        return None
    # Skip if first token starts with a digit or is a known platform name
    if re.match(r"^\d", parts[0]) or _NO_PUBMED_PREFIXES.match(parts[0]):
        return None
    # Use first token as author surname + remaining as free text
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------

def _unique_obs_values(obs: pd.DataFrame, col: str) -> str:
    """Return semicolon-joined unique non-trivial values from an obs column."""
    if col not in obs.columns:
        return ""
    vals = obs[col].dropna().unique().tolist()
    filtered = [str(v) for v in vals if str(v) not in ("nan", "unknown", "na", "")]
    return "; ".join(sorted(set(filtered)))


def process_file(filepath: Path) -> dict:
    row: dict = {f: "" for f in FIELDNAMES}
    row["filename"] = filepath.name

    try:
        adata = ad.read_h5ad(filepath, backed="r")

        row["n_cells"] = adata.n_obs
        row["n_vars"] = adata.n_vars
        row["title"] = adata.uns.get("title", "")

        obs = adata.obs
        row["assay"] = _unique_obs_values(obs, "assay")
        row["organism"] = _unique_obs_values(obs, "organism")
        row["tissue"] = _unique_obs_values(obs, "tissue")
        row["sex"] = _unique_obs_values(obs, "sex")
        row["condition_ids"] = _unique_obs_values(obs, "condition_id")
        row["donor_ids"] = _unique_obs_values(obs, "donor_id")
        row["dataset"] = _unique_obs_values(obs, "dataset")
        row["nicheformer_split"] = _unique_obs_values(obs, "nicheformer_split")

        try:
            adata.file.close()
        except Exception:
            pass

    except Exception as exc:
        row["error"] = str(exc)
        return row

    # PubMed lookup (best-effort, never blocks the row from being written)
    query = build_pubmed_query(filepath.stem)
    if query:
        doi, pub_title = pubmed_lookup(query)
        row["publication_doi"] = doi
        row["publication_title"] = pub_title

    return row


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def load_processed(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["filename"])
        return set(df["filename"].dropna().tolist())
    except Exception:
        return set()


def write_header(csv_path: Path) -> None:
    with open(csv_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=FIELDNAMES).writeheader()


def append_row(csv_path: Path, row: dict) -> None:
    with open(csv_path, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=FIELDNAMES).writerow(row)


def main() -> None:
    processed = load_processed(OUTPUT_CSV)
    if processed:
        print(f"Resuming — {len(processed)} files already in CSV.")
    else:
        write_header(OUTPUT_CSV)
        print(f"Starting fresh. Writing to {OUTPUT_CSV}")

    print(f"Watching {DATASET_DIR}  (poll every {POLL_INTERVAL_SECONDS}s)\n")

    while True:
        h5ad_files = sorted(f for f in DATASET_DIR.glob("*.h5ad") if not f.name.startswith("._"))
        new_files = [f for f in h5ad_files if f.name not in processed]

        if new_files:
            ts = pd.Timestamp.now().strftime("%H:%M:%S")
            print(f"[{ts}] {len(new_files)} new file(s) to process:")
            for filepath in new_files:
                print(f"  {filepath.name} ...", end="", flush=True)
                row = process_file(filepath)
                append_row(OUTPUT_CSV, row)
                processed.add(filepath.name)
                if row["error"]:
                    print(f"  ERROR: {row['error']}")
                else:
                    pub = f"  doi={row['publication_doi']}" if row["publication_doi"] else ""
                    print(f"  {row['n_cells']:,} cells  {row['n_vars']:,} vars{pub}")
        else:
            ts = pd.Timestamp.now().strftime("%H:%M:%S")
            print(
                f"[{ts}] No new files. Total processed: {len(processed)}",
                end="\r",
                flush=True,
            )

        if DONE_SENTINEL.exists():
            # Do one final sweep then exit
            h5ad_files = sorted(f for f in DATASET_DIR.glob("*.h5ad") if not f.name.startswith("._"))
            remaining = [f for f in h5ad_files if f.name not in processed]
            if remaining:
                print(f"\nSentinel found — processing {len(remaining)} remaining file(s)...")
                for filepath in remaining:
                    print(f"  {filepath.name} ...", end="", flush=True)
                    row = process_file(filepath)
                    append_row(OUTPUT_CSV, row)
                    processed.add(filepath.name)
                    print(f"  {row['n_cells']:,} cells  {row['n_vars']:,} vars")
            print(f"\nDone. {len(processed)} files recorded in {OUTPUT_CSV}")
            break

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
