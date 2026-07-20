"""Create an auditable manifest for immutable source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024


def digest_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def build_manifest(raw_dir: Path) -> list[dict[str, object]]:
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        md5, sha256 = digest_file(path)
        rows.append(
            {
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "md5": md5,
                "sha256": sha256,
                "manifested_at_utc": now,
            }
        )
    return rows


def write_manifest(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file_name", "bytes", "md5", "sha256", "manifested_at_utc"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = build_manifest(args.raw_dir)
    write_manifest(rows, args.output)
    print(f"manifested {len(rows)} files -> {args.output}")


if __name__ == "__main__":
    main()

