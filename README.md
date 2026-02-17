# KaroSpace Data Wrangling

Utilities and notebooks for preparing and exploring spatial omics datasets used in KaroSpace workflows.

## Repository Layout

- `notebooks/`: analysis and wrangling notebooks (CODEX, CosMx, DBiT, MERFISH, Xenium).
- `data/`: local working data (ignored by git).
- `tutorial_data/`: local tutorial/raw datasets (ignored by git).

## Environment

The notebooks are currently configured for the kernel:

- `CellCharter (Python 3.10)` (`cellcharter310`)

Typical Python packages used:

- `scanpy`
- `anndata`
- `squidpy`
- `cellcharter`

## Workflow

1. Open a notebook in `notebooks/`.
2. Point input paths to your local/volume dataset locations.
3. Run data import and wrangling cells.
4. Save outputs outside git-tracked paths (e.g. under ignored data directories).

## Git Notes

Large data artifacts (for example `.h5ad`, `.zarr.zip`, archives, and local data folders) are ignored via `.gitignore`.
