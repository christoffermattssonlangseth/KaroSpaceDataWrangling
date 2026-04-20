"""Parse NanoString GeoMx DCC files + PKC into AnnData.

Reads all .dcc.gz files from GSE281807, maps RTS probe IDs to gene names
using the PKC file, aggregates probes per gene, and creates a combined
AnnData object (296 samples × ~18k genes).
"""

import gzip
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


DATA_DIR = Path(__file__).parent.parent / "data" / "GSE281807"
RAW_DIR = DATA_DIR / "raw"
OUT_PATH = DATA_DIR / "processed" / "GSE281807_geomx.h5ad"


def parse_pkc(pkc_path: Path) -> dict[str, str]:
    """Parse PKC file → {RTS_ID: gene_name} mapping."""
    with gzip.open(pkc_path, "rt") as f:
        pkc = json.load(f)

    rts_to_gene = {}
    for target in pkc["Targets"]:
        gene = target["DisplayName"]
        code_class = target["CodeClass"]
        for probe in target["Probes"]:
            rts_id = probe["RTS_ID"]
            rts_to_gene[rts_id] = (gene, code_class)
    return rts_to_gene


def parse_dcc(dcc_path: Path) -> dict:
    """Parse a single DCC file → sample metadata + counts."""
    with gzip.open(dcc_path, "rt") as f:
        content = f.read()

    # Parse header
    header = {}
    header_match = re.search(r"<Header>(.*?)</Header>", content, re.DOTALL)
    if header_match:
        for line in header_match.group(1).strip().split("\n"):
            if "," in line:
                k, v = line.split(",", 1)
                header[k.strip()] = v.strip().strip('"')

    # Parse scan attributes
    scan = {}
    scan_match = re.search(r"<Scan_Attributes>(.*?)</Scan_Attributes>", content, re.DOTALL)
    if scan_match:
        for line in scan_match.group(1).strip().split("\n"):
            if "," in line:
                k, v = line.split(",", 1)
                scan[k.strip()] = v.strip().strip('"')

    # Parse NGS processing attributes
    ngs = {}
    ngs_match = re.search(r"<NGS_Processing_Attributes>(.*?)</NGS_Processing_Attributes>", content, re.DOTALL)
    if ngs_match:
        for line in ngs_match.group(1).strip().split("\n"):
            if "," in line:
                k, v = line.split(",", 1)
                ngs[k.strip()] = v.strip().strip('"')

    # Parse counts
    counts = {}
    code_match = re.search(r"<Code_Summary>(.*?)</Code_Summary>", content, re.DOTALL)
    if code_match:
        for line in code_match.group(1).strip().split("\n"):
            if "," in line:
                rts_id, count = line.split(",", 1)
                counts[rts_id.strip()] = int(count.strip())

    # Extract GSM ID and DSP ID from filename
    fname = dcc_path.stem.replace(".dcc", "")
    gsm_id = fname.split("_")[0]

    metadata = {
        "gsm_id": gsm_id,
        "scan_id": scan.get("ID", ""),
        "plate_id": scan.get("Plate_ID", ""),
        "well": scan.get("Well", ""),
        "raw_reads": int(ngs.get("Raw", 0)),
        "trimmed_reads": int(ngs.get("Trimmed", 0)),
        "aligned_reads": int(ngs.get("Aligned", 0)),
        "umi_q30": float(ngs.get("umiQ30", 0)),
        "rts_q30": float(ngs.get("rtsQ30", 0)),
    }

    return metadata, counts


def build_adata(dcc_dir_list: list[Path], rts_to_gene: dict) -> ad.AnnData:
    """Build AnnData from list of DCC directories."""
    # Collect all DCC files
    dcc_files = []
    for d in dcc_dir_list:
        dcc_files.extend(sorted(d.glob("*.dcc.gz")))
    print(f"Found {len(dcc_files)} DCC files")

    # Get unique genes (endogenous only for main matrix, keep negatives in .obsm)
    endogenous_genes = sorted({
        gene for gene, (_, code_class) in rts_to_gene.items()
        if "Endogenous" in rts_to_gene[gene][1]
    })
    # Actually map to gene names
    gene_names_endo = sorted(set(
        gene for rts_id, (gene, cc) in rts_to_gene.items()
        if "Endogenous" in cc
    ))
    gene_names_neg = sorted(set(
        gene for rts_id, (gene, cc) in rts_to_gene.items()
        if "Negative" in cc
    ))

    all_gene_names = gene_names_endo + gene_names_neg
    gene_to_idx = {g: i for i, g in enumerate(all_gene_names)}

    # Parse all samples
    obs_records = []
    count_matrix = np.zeros((len(dcc_files), len(all_gene_names)), dtype=np.int32)

    for i, dcc_path in enumerate(dcc_files):
        if (i + 1) % 50 == 0:
            print(f"  Parsing {i + 1}/{len(dcc_files)}...")
        metadata, counts = parse_dcc(dcc_path)
        obs_records.append(metadata)

        # Aggregate probe counts to gene level
        gene_counts: dict[str, int] = {}
        for rts_id, count in counts.items():
            if rts_id in rts_to_gene:
                gene, _ = rts_to_gene[rts_id]
                gene_counts[gene] = gene_counts.get(gene, 0) + count

        for gene, count in gene_counts.items():
            if gene in gene_to_idx:
                count_matrix[i, gene_to_idx[gene]] = count

    obs_df = pd.DataFrame(obs_records)
    obs_df.index = obs_df["gsm_id"]

    # Build var DataFrame
    var_df = pd.DataFrame(index=all_gene_names)
    var_df["code_class"] = ["Endogenous"] * len(gene_names_endo) + ["Negative"] * len(gene_names_neg)
    var_df.index.name = "gene"

    adata = ad.AnnData(
        X=csr_matrix(count_matrix),
        obs=obs_df,
        var=var_df,
    )

    # Parse sample names for condition info
    # Pattern: GSM... maps to sample titles like "MS3a_NAWM_rep1"
    # We'll add what we can from the file structure
    adata.uns["dataset"] = "GSE281807"
    adata.uns["platform"] = "NanoString GeoMx DSP"
    adata.uns["organism"] = "Homo sapiens"
    adata.uns["title"] = (
        "Broad rim lesions are a novel pathological/morphological and imaging "
        "biomarker for rapid disease progression in Multiple Sclerosis"
    )

    return adata


def main():
    print("Parsing PKC file...")
    pkc_path = RAW_DIR / "GSE264094_Hs_R_NGS_WTA_v1.0.pkc.gz"
    rts_to_gene = parse_pkc(pkc_path)
    print(f"  {len(rts_to_gene)} probe → gene mappings")

    gene_names = set(g for g, _ in rts_to_gene.values())
    print(f"  {len(gene_names)} unique genes")

    print("Parsing DCC files...")
    dcc_dirs = [RAW_DIR / "dcc_part1", RAW_DIR / "dcc_part2"]
    adata = build_adata(dcc_dirs, rts_to_gene)

    print(f"\nAnnData: {adata.shape[0]} samples × {adata.shape[1]} genes")
    print(f"  Code classes: {adata.var['code_class'].value_counts().to_dict()}")
    print(f"  Sparsity: {1 - adata.X.nnz / np.prod(adata.shape):.2%}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
