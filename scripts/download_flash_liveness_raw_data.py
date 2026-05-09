#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    access: str
    sha256: str
    note: str


BUILTIN_SOURCES = [
    Source(
        key="casia_surf_cefa",
        url="https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-cefacvpr2020",
        access="license_required",
        sha256="",
        note="CASIA-SURF CeFA; request license from the official challenge page before download.",
    ),
    Source(
        key="casia_surf_hifimask",
        url="https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-hifimaskiccv2021",
        access="license_required",
        sha256="",
        note="High-fidelity mask data; official page requires a signed license before sharing download links.",
    ),
    Source(
        key="casia_surf_suhifimask",
        url="https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-suhifimaskcvpr2023",
        access="license_required",
        sha256="",
        note="Super high-fidelity mask data; official page requires license approval.",
    ),
    Source(
        key="wmca",
        url="https://zenodo.org/records/4580313",
        access="zenodo_restricted",
        sha256="",
        note="Wide Multi-Channel Attack database; Zenodo files are restricted by EULA.",
    ),
    Source(
        key="3dmad",
        url="https://zenodo.org/records/4068477",
        access="zenodo_restricted",
        sha256="",
        note="3D Mask Attack Dataset; Zenodo files are restricted by EULA.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage raw flash-liveness related datasets. Built-in high-value PAD datasets are license-gated; "
            "approved direct URLs can be downloaded through --source-list."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/raw_downloads"),
    )
    parser.add_argument(
        "--source-list",
        type=Path,
        default=None,
        help="Optional TSV with columns: key, url, access, sha256, note. Use access=open_direct for downloadable files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def read_sources(path: Path | None) -> list[Source]:
    sources = list(BUILTIN_SOURCES)
    if path is None:
        return sources
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            sources.append(
                Source(
                    key=row.get("key", "").strip(),
                    url=row.get("url", "").strip(),
                    access=(row.get("access", "") or "open_direct").strip(),
                    sha256=row.get("sha256", "").strip(),
                    note=row.get("note", "").strip(),
                )
            )
    return sources


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filename_for_source(source: Source) -> str:
    name = Path(urlparse(source.url).path).name
    if not name or "." not in name:
        name = f"{source.key}.download"
    return f"{source.key}__{name}"


def download(source: Source, output_path: Path, timeout: float) -> None:
    request = urllib.request.Request(source.url, headers={"User-Agent": "flash-liveness-data-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with output_path.open("wb") as file:
            shutil.copyfileobj(response, file, length=1024 * 1024)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sources = read_sources(args.source_list)
    catalog_rows = [
        {"key": source.key, "url": source.url, "access": source.access, "sha256": source.sha256, "note": source.note}
        for source in sources
    ]
    report_rows: list[dict[str, object]] = []

    for source in sources:
        if source.access != "open_direct":
            report_rows.append(
                {
                    "key": source.key,
                    "status": "skipped_requires_manual_access",
                    "path": "",
                    "sha256": "",
                    "note": source.note,
                }
            )
            continue

        target = output_root / filename_for_source(source)
        try:
            if target.exists() and not args.overwrite:
                status = "exists"
            else:
                download(source, target, args.timeout)
                status = "downloaded"
            actual_hash = sha256_file(target)
            if source.sha256 and actual_hash.lower() != source.sha256.lower():
                status = "sha256_mismatch"
            report_rows.append(
                {"key": source.key, "status": status, "path": str(target), "sha256": actual_hash, "note": source.note}
            )
        except Exception as exc:
            report_rows.append(
                {"key": source.key, "status": f"error:{type(exc).__name__}:{exc}", "path": str(target), "sha256": "", "note": source.note}
            )

    write_tsv(output_root / "source_catalog.tsv", ["key", "url", "access", "sha256", "note"], catalog_rows)
    write_tsv(output_root / "download_report.tsv", ["key", "status", "path", "sha256", "note"], report_rows)
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# Raw Flash Liveness Data Downloads",
                "",
                "Built-in high-value mask datasets are mostly license-gated. This script records them but does not bypass EULA or login pages.",
                "",
                "After a dataset owner gives you a direct approved URL, create a TSV like:",
                "",
                "```text",
                "key\turl\taccess\tsha256\tnote",
                "approved_hifimask\thttps://...\topen_direct\t\tapproved institutional link",
                "```",
                "",
                "Then run:",
                "",
                "```bash",
                "conda run -n anti-spoofing_scc_175 python scripts/download_flash_liveness_raw_data.py --source-list approved_sources.tsv",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Raw data staging directory: {output_root}")
    print(f"catalog={output_root / 'source_catalog.tsv'}")
    print(f"report={output_root / 'download_report.tsv'}")


if __name__ == "__main__":
    main()
