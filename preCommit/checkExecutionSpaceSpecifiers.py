#!/usr/bin/env python3

from pathlib import Path
import argparse
import re
from common import (
    SOURCE_EXTENSIONS,
    CONTROL_KEYWORDS,
    FUNCTION_PATTERN,
    looks_like_function_definition,
    is_initializer_list_entry,
    strip_block_comments,
    iter_source_files,
)

CUDA_SPECIFIERS = ("__host__", "__device__", "__global__")


def has_cuda_specifier(lines, index):
    """Check if a function definition has a CUDA specifier on the same or preceding lines."""
    line = lines[index]
    if any(spec in line for spec in CUDA_SPECIFIERS):
        return True

    lookback = 3
    start = max(0, index - lookback)
    for i in range(index - 1, start - 1, -1):
        prev_line = lines[i].strip()
        if not prev_line:
            continue
        if prev_line.startswith(("}", "{", "//", "/*", "#")):
            break
        if any(spec in prev_line for spec in CUDA_SPECIFIERS):
            return True
        if not re.match(
            r"^(?:template\s*<|\[\[[^\]]*\]\]|__host__|__device__|__global__)\b",
            prev_line,
        ):
            break
    return False


def scan_file(path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    findings = []

    for index, code_line in strip_block_comments(lines):
        if is_initializer_list_entry(lines, index):
            continue

        view_lines = lines[:index] + [code_line]
        if not looks_like_function_definition(view_lines, index):
            continue

        match = FUNCTION_PATTERN.match(code_line.strip())
        name = match.group("name") if match else "<multiline signature>"

        if name in CONTROL_KEYWORDS or name == "main":
            continue

        if has_cuda_specifier(lines, index):
            continue

        findings.append(
            {
                "file": path,
                "line": index + 1,
                "name": name,
                "signature": code_line.strip(),
            }
        )

    return findings


def main():
    parser = argparse.ArgumentParser(
        description="Find C/C++/CUDA functions without explicit __host__, __device__, or __global__ specifiers."
    )
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = []

    for path in iter_source_files(root):
        findings.extend(scan_file(path))

    if not findings:
        print("No functions missing CUDA execution space specifiers were found.")
        return

    for finding in findings:
        relative_path = finding["file"].relative_to(root)
        print(
            f"{relative_path}:{finding['line']}: {finding['name']} -> {finding['signature']}"
        )

    print(f"\nFound {len(findings)} possible function(s) without CUDA specifiers.")


if __name__ == "__main__":
    main()
