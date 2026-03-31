from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "cosmx-human-colon.ipynb"


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
        """# CosMx Human Colon To AnnData

This notebook is set up for two phases:

1. Explore the raw CosMx metadata and expression inputs safely.
2. Build a sparse, KaroSpace-ready `.h5ad` with `obsm["spatial"]`.

The raw files currently point at:

- `/Users/chrislangseth/Downloads/S0_metadata_file.csv.gz`
- `/Users/chrislangseth/Downloads/S0_exprMat_file.csv.gz`

The expression file is very large, so the notebook uses chunked loading and sparse assembly instead of trying to hold the full dense matrix in memory.
"""
    ),
    code_cell(
        """from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.cwd() / ".numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import json
import gc

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

ad.settings.allow_write_nullable_strings = True

sns.set_theme(context="notebook", style="whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)

ROOT = Path.cwd().resolve()
METADATA_PATH = Path("/Users/chrislangseth/Downloads/S0_metadata_file.csv.gz")
EXPR_PATH = Path("/Users/chrislangseth/Downloads/S0_exprMat_file.csv.gz")
OUTPUT_DIR = ROOT / "data" / "processed" / "cosmx-human-colon"
OUTPUT_PATH = OUTPUT_DIR / "cosmx-human-colon.h5ad"

CHUNKSIZE = 2_000
SELECT_FOVS = None
MAX_CELLS = 25_000
WRITE_OUTPUT = False

ROOT, METADATA_PATH.exists(), EXPR_PATH.exists(), OUTPUT_PATH
"""
    ),
    md_cell(
        """## Input inspection

Read only the metadata and the expression header first. This is cheap and gives enough context to decide whether the chunked build parameters are sensible.
"""
    ),
    code_cell(
        """metadata = pd.read_csv(METADATA_PATH, compression="gzip")
expr_header = pd.read_csv(EXPR_PATH, compression="gzip", nrows=5)

gene_cols = [c for c in expr_header.columns if c not in {"fov", "cell_ID"}]

print("metadata shape:", metadata.shape)
print("expression preview shape:", expr_header.shape)
print("gene columns:", len(gene_cols))
print("metadata columns:", metadata.columns.tolist())

metadata.head()
"""
    ),
    code_cell(
        """summary_cols = [
    "fov",
    "slide_ID",
    "Run_name",
    "Run_Tissue_name",
    "Panel",
    "assay_type",
    "nCount_RNA",
    "nFeature_RNA",
    "CenterX_global_px",
    "CenterY_global_px",
]

metadata[summary_cols].describe(include="all").T
"""
    ),
    code_cell(
        """fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.histplot(metadata["nCount_RNA"], bins=50, ax=axes[0])
axes[0].set_title("nCount_RNA")

sns.histplot(metadata["nFeature_RNA"], bins=50, ax=axes[1])
axes[1].set_title("nFeature_RNA")

fov_sizes = metadata["fov"].value_counts().sort_index()
sns.histplot(fov_sizes, bins=40, ax=axes[2])
axes[2].set_title("cells per FOV")

plt.tight_layout()
plt.show()

metadata[["fov", "nCount_RNA", "nFeature_RNA"]].head()
"""
    ),
    md_cell(
        """## Assembly helpers

The raw expression matrix is cell-by-gene CSV. The helpers below:

- build a stable observation key from `fov` + `cell_ID`
- align expression chunks against metadata
- create sparse matrices chunk-by-chunk
- sanitize `obs` dtypes so `.write_h5ad()` is reliable
"""
    ),
    code_cell(
        """def make_obs_key(df: pd.DataFrame) -> pd.Index:
    return (
        df["fov"].astype(str).str.strip()
        + "__"
        + df["cell_ID"].astype(str).str.strip()
    )


def sanitize_obs_for_h5ad(obs_df: pd.DataFrame) -> pd.DataFrame:
    out = obs_df.copy()
    for col in out.columns:
        ser = out[col]
        if isinstance(ser.dtype, pd.CategoricalDtype):
            if ser.cat.categories.dtype == object:
                out[col] = ser.astype("string")
            continue

        if pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser):
            non_na = ser.dropna()
            if non_na.empty:
                out[col] = ser.astype("string")
                continue

            num = pd.to_numeric(ser, errors="coerce")
            if int(num.notna().sum()) == int(ser.notna().sum()):
                out[col] = num
                continue

            out[col] = ser.astype("string")

    return out


def prepare_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    obs = metadata_df.copy()
    obs["obs_key"] = make_obs_key(obs)
    if obs["obs_key"].duplicated().any():
        dupes = int(obs["obs_key"].duplicated().sum())
        raise ValueError(f"metadata has {dupes} duplicated fov/cell_ID keys")

    obs["dataset_id"] = "cosmx-human-colon"
    obs["sample_id"] = obs["Run_Tissue_name"].fillna("S0").astype(str)
    obs["section_id"] = obs["fov"].map(lambda x: f"fov_{int(x):03d}")

    obs["fov"] = obs["fov"].astype("int32")
    obs["cell_ID"] = obs["cell_ID"].astype("int32")
    obs["slide_ID"] = pd.to_numeric(obs["slide_ID"], errors="coerce").astype("Int32")

    for col in ["Run_name", "Run_Tissue_name", "Panel", "assay_type", "dataset_id", "sample_id", "section_id"]:
        if col in obs.columns:
            obs[col] = obs[col].astype("category")

    return obs.set_index("obs_key", drop=False)


def build_sparse_anndata(
    expr_path: Path,
    metadata_indexed: pd.DataFrame,
    *,
    chunksize: int = 2_000,
    select_fovs: list[int] | None = None,
    max_cells: int | None = None,
) -> ad.AnnData:
    preview = pd.read_csv(expr_path, compression="gzip", nrows=5)
    genes = [c for c in preview.columns if c not in {"fov", "cell_ID"}]

    matrices = []
    obs_parts = []
    gene_nnz = np.zeros(len(genes), dtype=np.int64)
    gene_sum = np.zeros(len(genes), dtype=np.int64)
    total_cells = 0
    matched_cells = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(expr_path, compression="gzip", chunksize=chunksize), start=1):
        chunk["obs_key"] = make_obs_key(chunk)

        if select_fovs is not None:
            chunk = chunk[chunk["fov"].isin(select_fovs)].copy()

        if chunk.empty:
            continue

        if max_cells is not None:
            remaining = max_cells - matched_cells
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()

        obs_chunk = metadata_indexed.reindex(chunk["obs_key"])
        if obs_chunk["obs_key"].isna().any():
            missing = chunk.loc[obs_chunk["obs_key"].isna(), ["fov", "cell_ID", "obs_key"]].head(10)
            raise ValueError(f"missing metadata for some expression rows:\\n{missing}")

        values = chunk[genes].to_numpy(dtype=np.int32, copy=False)
        gene_nnz += (values != 0).sum(axis=0)
        gene_sum += values.sum(axis=0)
        matrices.append(sparse.csr_matrix(values))

        obs_part = obs_chunk.copy()
        obs_part.index = obs_part["cell"].astype(str) if "cell" in obs_part.columns and obs_part["cell"].notna().all() else obs_part["obs_key"]
        obs_parts.append(obs_part)

        total_cells += len(chunk)
        matched_cells += len(obs_part)
        print(f"chunk {chunk_idx}: kept {len(chunk):,} cells; cumulative {matched_cells:,}")

        del chunk, obs_chunk, values, obs_part
        gc.collect()

    if not matrices:
        raise ValueError("no expression rows selected; check SELECT_FOVS / MAX_CELLS")

    X = sparse.vstack(matrices, format="csr")
    obs = pd.concat(obs_parts, axis=0)
    obs = sanitize_obs_for_h5ad(obs)

    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    var["feature_name"] = var.index.astype(str)
    var["feature_type"] = "gene"
    var["n_cells_by_counts"] = gene_nnz
    var["total_counts"] = gene_sum

    nonzero_gene_mask = var["n_cells_by_counts"].to_numpy() > 0
    if not np.all(nonzero_gene_mask):
        X = X[:, nonzero_gene_mask]
        var = var.loc[nonzero_gene_mask].copy()

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(dtype=np.float32)
    adata.uns["dataset_source"] = {
        "metadata_path": str(METADATA_PATH),
        "expression_path": str(expr_path),
        "chunksize": int(chunksize),
        "selected_fovs": None if select_fovs is None else [int(x) for x in select_fovs],
        "max_cells": None if max_cells is None else int(max_cells),
    }
    adata.uns["karospace_hints"] = {
        "section_key": "section_id",
        "sample_key": "sample_id",
        "spatial_key": "spatial",
        "notes": [
            "raw CosMx inputs do not include cell-type labels in this export",
            "good initial metadata filters are fov, sample_id, nCount_RNA, and nFeature_RNA",
        ],
    }
    return adata
"""
    ),
    code_cell(
        """metadata_indexed = prepare_metadata(metadata)
metadata_indexed[[
    "obs_key",
    "cell",
    "fov",
    "cell_ID",
    "section_id",
    "sample_id",
    "CenterX_global_px",
    "CenterY_global_px",
]].head()
"""
    ),
    md_cell(
        """## Preview build

The defaults below are intentionally conservative:

- `MAX_CELLS = 25_000` keeps the first assembly pass manageable.
- Set `MAX_CELLS = None` for the full dataset.
- Set `WRITE_OUTPUT = True` only when you are ready to write the final `.h5ad`.
"""
    ),
    code_cell(
        """adata = build_sparse_anndata(
    EXPR_PATH,
    metadata_indexed,
    chunksize=CHUNKSIZE,
    select_fovs=SELECT_FOVS,
    max_cells=MAX_CELLS,
)

adata
"""
    ),
    code_cell(
        """print("obs x vars:", adata.n_obs, adata.n_vars)
print("X format:", type(adata.X), "nnz:", adata.X.nnz)
print("obsm keys:", list(adata.obsm.keys()))
print("obs columns:", len(adata.obs.columns))
print("var columns:", len(adata.var.columns))

adata.obs[[
    "dataset_id",
    "sample_id",
    "section_id",
    "fov",
    "cell_ID",
    "nCount_RNA",
    "nFeature_RNA",
]].head()
"""
    ),
    code_cell(
        """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(
    adata.obsm["spatial"][:, 0],
    adata.obsm["spatial"][:, 1],
    s=0.2,
    c=adata.obs["nCount_RNA"].to_numpy(),
    cmap="viridis",
)
axes[0].set_title("Spatial coordinates colored by nCount_RNA")
axes[0].set_xlabel("CenterX_global_px")
axes[0].set_ylabel("CenterY_global_px")
axes[0].invert_yaxis()

top_fovs = adata.obs["fov"].value_counts().head(20).sort_index()
sns.barplot(x=top_fovs.index.astype(str), y=top_fovs.values, ax=axes[1], color="#4c72b0")
axes[1].set_title("Top FOV cell counts in current build")
axes[1].set_xlabel("fov")
axes[1].set_ylabel("cells")
axes[1].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()
"""
    ),
    md_cell(
        """## Full export

For the final uploadable file:

1. Set `MAX_CELLS = None`.
2. Optionally set `SELECT_FOVS` if you want a partial export.
3. Re-run the build cell.
4. Set `WRITE_OUTPUT = True` and run the write cell below.
"""
    ),
    code_cell(
        """OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if WRITE_OUTPUT:
    adata.write_h5ad(OUTPUT_PATH)
    print(f"Wrote: {OUTPUT_PATH}")
else:
    print("WRITE_OUTPUT is False; notebook is configured for inspection first.")
    print(f"Planned output: {OUTPUT_PATH}")
"""
    ),
    code_cell(
        """karospace_summary = {
    "output_path": str(OUTPUT_PATH),
    "section_key": "section_id",
    "sample_key": "sample_id",
    "spatial_key": "obsm['spatial']",
    "candidate_filter_columns": ["fov", "sample_id", "nCount_RNA", "nFeature_RNA"],
    "annotation_gap": "raw input does not yet include a moderate-cardinality cell annotation",
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
