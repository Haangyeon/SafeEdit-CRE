"""Inspect a large delimited source table without loading it into memory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def detect_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;")
    except csv.Error:
        return csv.excel_tab


def inspect_table(path: Path, preview_rows: int = 5) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(128 * 1024)
        handle.seek(0)
        dialect = detect_dialect(sample)
        reader = csv.reader(handle, dialect)
        header = next(reader)
        width_counts: Counter[int] = Counter()
        previews = []
        row_count = 0
        for row in reader:
            row_count += 1
            width_counts[len(row)] += 1
            if len(previews) < preview_rows:
                previews.append(row)
    return {
        "path": str(path),
        "delimiter": dialect.delimiter,
        "header": header,
        "column_count": len(header),
        "data_row_count": row_count,
        "row_width_counts": dict(sorted(width_counts.items())),
        "preview": previews,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("--preview-rows", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_table(args.table, preview_rows=args.preview_rows)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

