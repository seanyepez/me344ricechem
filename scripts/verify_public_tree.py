#!/usr/bin/env python3
"""Reject private or non-public artifacts before publishing this repository."""

from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".jsonl", ".pem", ".key", ".p12", ".pfx"}
ALLOWED_PUBLIC_CSV = {
    Path("results/hardware_comparison.csv"),
    Path("results/hardware_comparison_TEMPLATE.csv"),
}
PRIVATE_PATTERNS = {
    "absolute macOS user path": re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    "Google Cloud Storage URI": re.compile(rb"gs://", re.IGNORECASE),
    "GCP service-account address": re.compile(
        rb"[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\.iam\.gserviceaccount\.com",
        re.IGNORECASE,
    ),
    "GCP resource identifier": re.compile(
        rb"projects/[A-Za-z0-9._-]+/(?:locations|zones|clusters)/",
        re.IGNORECASE,
    ),
    "private workspace-relative path": re.compile(
        rb"(?:^|[\s\"'])eval/bench/ricechem/", re.IGNORECASE
    ),
    "private key block": re.compile(
        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
    ),
    "Google API key": re.compile(rb"\bAIza[0-9A-Za-z_-]{20,}\b"),
    "OpenAI-style secret": re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}\b"),
}
TEXT_ARCHIVE_SUFFIXES = {".xml", ".rels", ".txt", ".json"}


def candidate_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in completed.stdout.split(b"\0") if item]


def scan_bytes(label: str, payload: bytes) -> list[str]:
    return [
        f"{label}: contains {description}"
        for description, pattern in PRIVATE_PATTERNS.items()
        if pattern.search(payload)
    ]


def main() -> int:
    failures: list[str] = []
    files = candidate_files()
    for path in files:
        relative = path.relative_to(ROOT)
        if path.resolve() == SELF:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"{relative}: forbidden public artifact type")
            continue
        if path.suffix.lower() == ".csv" and relative not in ALLOWED_PUBLIC_CSV:
            failures.append(f"{relative}: CSV is not an allowlisted aggregate/template")
            continue
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            failures.append(f"{relative}: {size} bytes exceeds the 5 MiB public limit")
            continue

        if path.suffix.lower() == ".pptx":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if Path(member).suffix.lower() in TEXT_ARCHIVE_SUFFIXES:
                        failures.extend(
                            scan_bytes(f"{relative}!{member}", archive.read(member))
                        )
            continue

        payload = path.read_bytes()
        if b"\0" not in payload:
            failures.extend(scan_bytes(str(relative), payload))

    if failures:
        raise AssertionError("public-boundary verification failed:\n- " + "\n- ".join(failures))
    print(f"public-boundary verification passed: {len(files)} candidate files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
