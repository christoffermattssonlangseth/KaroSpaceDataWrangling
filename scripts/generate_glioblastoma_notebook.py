from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "glioblastoma.ipynb"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md_cell(
        """# Glioblastoma To AnnData

This notebook is set up to explore and assemble the contents of `data/glioblastoma-data` into a KaroSpace-ready `.h5ad`.

The critical constraint is the count matrix:

- `matrix.mtx` is very large
- a one-shot `mmread()` is not the safe default

So this workflow uses a streaming Matrix Market reader that writes shard `.h5ad` files and then concatenates them on disk.
"""
    ),
    code_cell(
        """from __future__ import annotations

import gc
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata.experimental import concat_on_disk
from scipy import sparse

ad.settings.allow_write_nullable_strings = True

sns.set_theme(context="notebook", style="whitegrid")
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 60)

PROJECT_ROOT = Path("/Users/chrislangseth/work/karolinska_institutet/projects/KaroSpaceDataWrangling")
ROOT = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data" / "glioblastoma-data"
BARCODES_PATH = DATA_DIR / "barcodes.tsv"
FEATURES_PATH = DATA_DIR / "features.tsv"
META_PATH = DATA_DIR / "meta.tsv"
VISIUM_COORDS_PATH = DATA_DIR / "Visium.coords.tsv.gz.tmp"
UMAP_COORDS_PATH = DATA_DIR / "UMAP.coords.tsv.gz.tmp"
MATRIX_PATH = DATA_DIR / "matrix.mtx"

OUTPUT_DIR = ROOT / "data" / "processed" / "glioblastoma"
SHARD_DIR = OUTPUT_DIR / "shards"
OUTPUT_PATH = OUTPUT_DIR / "glioblastoma.h5ad"

SHARD_N_CELLS = 2_000
MAX_CELLS = None
WRITE_SHARDS = True
CONCAT_SHARDS = True
ANALYSIS_INPUT_MODE = "reload_if_available"

ROOT, DATA_DIR.exists(), OUTPUT_PATH
"""
    ),
    md_cell(
        """## Load tables

These tables are small enough to inspect directly. The count matrix is handled separately.
"""
    ),
    code_cell(
        """def load_barcodes(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\\t", header=None, names=["barcode"])


def load_features(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path, sep="\\t", header=None, names=["gene"])
    out["gene"] = out["gene"].astype("string")
    return out


def load_meta(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path, sep="\\t")
    out = out.rename(columns={"Unnamed: 0": "barcode"})
    out["barcode"] = out["barcode"].astype("string")
    return out


def load_coords(path: Path, x_name: str, y_name: str) -> pd.DataFrame:
    out = pd.read_csv(path, sep="\\t", header=None, names=["barcode", x_name, y_name])
    out["barcode"] = out["barcode"].astype("string")
    out[x_name] = pd.to_numeric(out[x_name], errors="coerce").astype("float32")
    out[y_name] = pd.to_numeric(out[y_name], errors="coerce").astype("float32")
    return out


barcodes = load_barcodes(BARCODES_PATH)
features = load_features(FEATURES_PATH)
meta = load_meta(META_PATH)
visium = load_coords(VISIUM_COORDS_PATH, "x_visium", "y_visium")
umap = load_coords(UMAP_COORDS_PATH, "umap_1", "umap_2")

print("barcodes:", barcodes.shape)
print("features:", features.shape)
print("meta:", meta.shape)
print("visium:", visium.shape)
print("umap:", umap.shape)
"""
    ),
    code_cell(
        """print("barcodes aligned to meta:", barcodes["barcode"].equals(meta["barcode"]))
print("barcodes aligned to visium:", barcodes["barcode"].equals(visium["barcode"]))
print("barcodes aligned to umap:", barcodes["barcode"].equals(umap["barcode"]))

meta.head()
"""
    ),
    code_cell(
        """summary_cols = [
    "orig.ident",
    "sample",
    "source",
    "Diagnosis",
    "short_histology",
    "Level_1",
    "greenwald_metaprograms",
    "AF",
    "nCount_Spatial",
    "nFeature_Spatial",
]

meta[summary_cols].describe(include="all").T
"""
    ),
    code_cell(
        """fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.histplot(meta["nCount_Spatial"], bins=60, ax=axes[0])
axes[0].set_title("nCount_Spatial")

sns.histplot(meta["nFeature_Spatial"], bins=60, ax=axes[1])
axes[1].set_title("nFeature_Spatial")

sample_sizes = meta["orig.ident"].value_counts().head(20)
sns.barplot(x=sample_sizes.index, y=sample_sizes.values, ax=axes[2], color="#4c72b0")
axes[2].set_title("Top orig.ident spot counts")
axes[2].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()
"""
    ),
    md_cell(
        """## Prepare compact `obs` and `var`

`orig.ident` is a good default `section_id` here because it splits the data into natural sample-level panels.
"""
    ),
    code_cell(
        """OBS_COLUMNS = [
    "barcode",
    "orig.ident",
    "sample",
    "source",
    "Gender",
    "Age",
    "Diagnosis",
    "short_histology",
    "Level_1",
    "AF_ivy",
    "inferCNV",
    "AF_CNV",
    "metaprogram",
    "greenwald",
    "AF",
    "greenwald_metaprograms",
    "nCount_Spatial",
    "nFeature_Spatial",
]

CATEGORY_COLUMNS = [
    "orig.ident",
    "sample",
    "source",
    "Gender",
    "Diagnosis",
    "short_histology",
    "Level_1",
    "AF_ivy",
    "AF_CNV",
    "metaprogram",
    "greenwald",
    "AF",
    "greenwald_metaprograms",
]


def sanitize_obs_for_h5ad(obs_df: pd.DataFrame) -> pd.DataFrame:
    out = obs_df.copy()
    for col in out.columns:
        ser = out[col]
        if isinstance(ser.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser):
            out[col] = ser.astype("string")
    return out


def prepare_obs_table(
    barcodes_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    visium_df: pd.DataFrame,
    umap_df: pd.DataFrame,
) -> pd.DataFrame:
    obs = barcodes_df.copy()
    obs = obs.merge(meta_df[OBS_COLUMNS], on="barcode", how="left", validate="one_to_one")
    obs = obs.merge(visium_df, on="barcode", how="left", validate="one_to_one")
    obs = obs.merge(umap_df, on="barcode", how="left", validate="one_to_one")

    obs["dataset_id"] = pd.Categorical(["glioblastoma"] * len(obs))
    obs["sample_id"] = obs["sample"].astype("string").fillna("unknown").astype("category")
    obs["section_id"] = obs["orig.ident"].astype("string").fillna("unknown").astype("category")

    obs["Age"] = pd.to_numeric(obs["Age"], errors="coerce").astype("float32")
    obs["nCount_Spatial"] = pd.to_numeric(obs["nCount_Spatial"], errors="coerce").astype("Int32")
    obs["nFeature_Spatial"] = pd.to_numeric(obs["nFeature_Spatial"], errors="coerce").astype("Int32")
    obs["inferCNV"] = pd.to_numeric(obs["inferCNV"], errors="coerce").astype("float32")

    for col in CATEGORY_COLUMNS + ["dataset_id", "sample_id", "section_id"]:
        if col in obs.columns:
            obs[col] = obs[col].astype("category")

    obs = sanitize_obs_for_h5ad(obs)
    return obs.set_index("barcode", drop=True)


def prepare_var_table(features_df: pd.DataFrame) -> pd.DataFrame:
    genes = pd.Index(features_df["gene"].astype(str), name="gene")
    if not genes.is_unique:
        seen: dict[str, int] = {}
        deduped = []
        for gene in genes:
            count = seen.get(gene, 0)
            deduped.append(gene if count == 0 else f"{gene}-{count}")
            seen[gene] = count + 1
        genes = pd.Index(deduped, name="gene")
    var = pd.DataFrame(index=genes)
    return var


obs = prepare_obs_table(barcodes, meta, visium, umap)
var = prepare_var_table(features)

obs.shape, var.shape
"""
    ),
    code_cell(
        """obs.head()
"""
    ),
    code_cell(
        """fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(obs["x_visium"], obs["y_visium"], s=0.3, c=obs["nCount_Spatial"].astype(float), cmap="viridis", linewidths=0)
axes[0].set_title("Visium coordinates")
axes[0].invert_yaxis()

axes[1].scatter(obs["umap_1"], obs["umap_2"], s=0.3, c=obs["nCount_Spatial"].astype(float), cmap="viridis", linewidths=0)
axes[1].set_title("Provided UMAP coordinates")

plt.tight_layout()
plt.show()
"""
    ),
    md_cell(
        """## Streaming Matrix Market assembly

The matrix dimensions are read from the Matrix Market header. Then the file is streamed column-block by column-block into shard `.h5ad` files.

This avoids holding the full `115,914 x 27,573` sparse matrix in memory at once.
"""
    ),
    code_cell(
        """def read_matrix_market_shape(path: Path) -> tuple[int, int, int]:
    with path.open("rt") as handle:
        header = handle.readline().strip()
        if not header.startswith("%%MatrixMarket matrix coordinate"):
            raise ValueError(f"unexpected Matrix Market header: {header}")

        line = handle.readline().strip()
        while line.startswith("%") or not line:
            line = handle.readline().strip()

    n_rows, n_cols, nnz = map(int, line.split())
    return n_rows, n_cols, nnz


def write_matrix_market_shards(
    matrix_path: Path,
    obs_df: pd.DataFrame,
    var_df: pd.DataFrame,
    shard_dir: Path,
    *,
    shard_n_cells: int = 2_000,
    max_cells: int | None = None,
    value_dtype=np.float32,
) -> list[Path]:
    n_genes, n_cells_total, nnz = read_matrix_market_shape(matrix_path)
    if n_genes != var_df.shape[0]:
        raise ValueError(f"matrix rows {n_genes} != var rows {var_df.shape[0]}")
    if n_cells_total != obs_df.shape[0]:
        raise ValueError(f"matrix cols {n_cells_total} != obs rows {obs_df.shape[0]}")

    target_cells = min(n_cells_total, max_cells) if max_cells is not None else n_cells_total
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    current_start = 1
    current_end = min(shard_n_cells, target_cells)
    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []
    prev_col = 1

    def flush_shard(start_col: int, end_col: int, shard_idx: int) -> Path:
        n_obs = end_col - start_col + 1
        X = sparse.csr_matrix(
            (
                np.asarray(values, dtype=value_dtype),
                (
                    np.asarray(row_idx, dtype=np.int32),
                    np.asarray(col_idx, dtype=np.int32),
                ),
            ),
            shape=(n_obs, var_df.shape[0]),
        )

        shard_obs = obs_df.iloc[start_col - 1:end_col].copy()
        shard = ad.AnnData(X=X, obs=shard_obs, var=var_df.copy())
        shard.obsm["spatial"] = shard_obs[["x_visium", "y_visium"]].to_numpy(dtype=np.float32)
        shard.obsm["X_umap"] = shard_obs[["umap_1", "umap_2"]].to_numpy(dtype=np.float32)

        shard_path = shard_dir / f"glioblastoma_shard_{shard_idx:03d}.h5ad"
        shard.write_h5ad(shard_path)
        del shard, shard_obs, X
        gc.collect()
        return shard_path

    with matrix_path.open("rt") as handle:
        _ = handle.readline()
        line = handle.readline().strip()
        while line.startswith("%") or not line:
            line = handle.readline().strip()

        for raw in handle:
            parts = raw.split()
            if not parts:
                continue

            gene_1based = int(parts[0])
            cell_1based = int(parts[1])
            value = float(parts[2])

            if cell_1based < prev_col:
                raise ValueError("matrix columns are not nondecreasing; streaming assumption failed")
            prev_col = cell_1based

            if cell_1based > target_cells:
                break

            while cell_1based > current_end:
                shard_idx = len(shard_paths) + 1
                shard_paths.append(flush_shard(current_start, current_end, shard_idx))
                row_idx.clear()
                col_idx.clear()
                values.clear()
                current_start = current_end + 1
                current_end = min(current_start + shard_n_cells - 1, target_cells)

            row_idx.append(cell_1based - current_start)
            col_idx.append(gene_1based - 1)
            values.append(value)

    if current_start <= target_cells:
        shard_idx = len(shard_paths) + 1
        shard_paths.append(flush_shard(current_start, current_end, shard_idx))

    return shard_paths


def concat_shards(shard_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_on_disk(
        shard_paths,
        output_path,
        axis=0,
        join="inner",
        merge="same",
        uns_merge="same",
        max_loaded_elems=100_000_000,
    )


matrix_shape = read_matrix_market_shape(MATRIX_PATH)
matrix_shape
"""
    ),
    code_cell(
        """print("matrix rows (genes):", matrix_shape[0])
print("matrix cols (cells):", matrix_shape[1])
print("matrix nnz:", matrix_shape[2])
print("assembly mode: sparse CSR shards + on-disk concat")
"""
    ),
    md_cell(
        """## Build shards

This notebook is configured for the full dataset by default:

- `MAX_CELLS = None`
- `WRITE_SHARDS = True`
- `CONCAT_SHARDS = True`

The count matrix is kept sparse throughout the assembly path.
"""
    ),
    code_cell(
        """if WRITE_SHARDS:
    shard_paths = write_matrix_market_shards(
        MATRIX_PATH,
        obs,
        var,
        SHARD_DIR,
        shard_n_cells=SHARD_N_CELLS,
        max_cells=MAX_CELLS,
    )
    print(f"wrote {len(shard_paths)} shards to {SHARD_DIR}")
else:
    shard_paths = sorted(SHARD_DIR.glob('glioblastoma_shard_*.h5ad'))
    print("WRITE_SHARDS is False")
    print(f"existing shards: {len(shard_paths)}")

shard_paths[:5]
"""
    ),
    code_cell(
        """if CONCAT_SHARDS:
    if not shard_paths:
        raise RuntimeError("no shards available to concatenate")
    concat_shards(shard_paths, OUTPUT_PATH)
    print(f"Wrote concatenated h5ad: {OUTPUT_PATH}")
else:
    print("CONCAT_SHARDS is False")
    print(f"planned output: {OUTPUT_PATH}")
"""
    ),
    md_cell(
        """## Reload assembled data

Once the output file exists, use this reload path for exploration instead of rebuilding.
"""
    ),
    code_cell(
        """if ANALYSIS_INPUT_MODE not in {"reload_if_available", "reload_only", "memory_only"}:
    raise ValueError("ANALYSIS_INPUT_MODE must be one of: reload_if_available, reload_only, memory_only")

if ANALYSIS_INPUT_MODE in {"reload_if_available", "reload_only"} and OUTPUT_PATH.exists():
    adata = ad.read_h5ad(OUTPUT_PATH)
    adata_source = f"reloaded from {OUTPUT_PATH}"
elif ANALYSIS_INPUT_MODE == "reload_only":
    raise FileNotFoundError(f"requested reload_only, but no file exists at {OUTPUT_PATH}")
else:
    adata = None
    adata_source = "no assembled h5ad loaded"

print(adata_source)
adata
"""
    ),
    code_cell(
        """if adata is not None:
    print("obs x vars:", adata.n_obs, adata.n_vars)
    print("obsm keys:", list(adata.obsm.keys()))
    print("obs columns:", len(adata.obs.columns))
    print("var columns:", len(adata.var.columns))
    adata.obs.head()
"""
    ),
    code_cell(
        """if adata is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(
        adata.obsm["spatial"][:, 0],
        adata.obsm["spatial"][:, 1],
        s=0.2,
        c=adata.obs["nCount_Spatial"].astype(float).to_numpy(),
        cmap="viridis",
        linewidths=0,
    )
    axes[0].set_title("Spatial coordinates colored by nCount_Spatial")
    axes[0].invert_yaxis()

    color_codes = adata.obs["greenwald_metaprograms"].astype("category").cat.codes.to_numpy()
    axes[1].scatter(
        adata.obsm["X_umap"][:, 0],
        adata.obsm["X_umap"][:, 1],
        s=0.2,
        c=color_codes,
        cmap="tab20",
        linewidths=0,
    )
    axes[1].set_title("Provided UMAP colored by greenwald_metaprograms")

    plt.tight_layout()
    plt.show()
"""
    ),
    code_cell(
        """karospace_summary = {
    "output_path": str(OUTPUT_PATH),
    "section_key": "section_id",
    "sample_key": "sample_id",
    "spatial_key": "obsm['spatial']",
    "umap_key": "obsm['X_umap']",
    "primary_color_candidates": [
        "greenwald_metaprograms",
        "AF",
        "short_histology",
        "source",
    ],
    "metadata_columns": [
        "orig.ident",
        "sample",
        "source",
        "Diagnosis",
        "short_histology",
        "Level_1",
        "AF",
        "greenwald_metaprograms",
    ],
}

print(json.dumps(karospace_summary, indent=2))
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2))
print(f"Wrote {NOTEBOOK_PATH}")
