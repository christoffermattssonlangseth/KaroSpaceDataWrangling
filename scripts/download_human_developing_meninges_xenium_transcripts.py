#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.request
import zipfile
from html import unescape
from pathlib import Path

BASE_URL = (
    "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/600/S-BIAD1600/"
    "Files/HumanDevelopingMeninges/Xenium/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all HumanDevelopingMeninges Xenium transcripts archives, "
            "placing each sample in its own folder and extracting to transcripts.zarr/."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/human-developing-meninges-xenium"),
        help="Base output directory for the sample folders.",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep transcripts.zarr.zip after extraction.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        help=(
            "Optional subset of sample folder names to download. "
            "Example: XETG00045__0005139__M46__20230705__135418"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract samples even if transcripts.zarr already exists.",
    )
    return parser.parse_args()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8", "ignore")


def list_sample_folders() -> list[str]:
    html = fetch_text(BASE_URL)
    hrefs = re.findall(r'href="([^"]+/)"', html)
    folders = []
    for href in hrefs:
        if href.startswith("/"):
            continue
        if href.startswith("?"):
            continue
        folders.append(unescape(href.rstrip("/")))
    return folders


def download_file(url: str, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp_path.replace(destination)


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)


def main() -> int:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    sample_folders = list_sample_folders()
    requested = set(args.samples or [])
    if requested:
        missing = sorted(requested - set(sample_folders))
        if missing:
            print("Unknown sample folders:", file=sys.stderr)
            for sample in missing:
                print(f"  {sample}", file=sys.stderr)
            return 1
        sample_folders = [sample for sample in sample_folders if sample in requested]

    for sample in sample_folders:
        sample_dir = outdir / sample
        archive_path = sample_dir / "transcripts.zarr.zip"
        extract_dir = sample_dir / "transcripts.zarr"
        url = f"{BASE_URL}{sample}/transcripts.zarr.zip"

        if extract_dir.exists() and not args.force:
            print(f"[skip] {sample}: {extract_dir} already exists")
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {sample}")
        download_file(url, archive_path)

        print(f"[extract] {sample}")
        extract_archive(archive_path, extract_dir)

        if not args.keep_zips:
            archive_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
