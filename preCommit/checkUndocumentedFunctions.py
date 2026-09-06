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

DOXYGEN_MARKERS = (
    "/**",
    "///",
    "//!",
    "/*!",
    "@brief",
    "@param",
    "@return",
    "@tparam",
)


def is_doxygen_comment(lines, function_line, lookback=20):
    """Return True if a Doxygen comment appears immediately before a function."""
    start = max(0, function_line - lookback)

    for line in reversed(lines[start:function_line]):
        stripped = line.strip()
        if not stripped:
            continue

        # Only consider lines that are actually comments
        if not (
            stripped.startswith("///")
            or stripped.startswith("//!")
            or stripped.startswith("/**")
            or stripped.startswith("/*!")
            or stripped.startswith("*")
            or stripped.startswith("/*")
            or stripped.startswith("//")
        ):
            # Allow certain qualifiers between comment and signature
            if re.match(
                r"^(?:template\s*<|\[\[[^\]]*\]\]|(?:__[A-Za-z_]+__\s+)*(?:static|inline|constexpr|consteval|virtual|friend|explicit)\b)",
                stripped,
            ):
                continue
            else:
                return False

        # If the line is a comment, check for any Doxygen marker
        if any(marker in line for marker in DOXYGEN_MARKERS):
            return True

    return False


def scan_file(path, lookback=20):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    findings = []

    for index, code_line in strip_block_comments(lines):
        if is_initializer_list_entry(lines, index):
            continue

        # Build a view of lines up to this point for lookahead
        # (pass original lines, but replace current line with cleaned version)
        view_lines = lines[:index] + [code_line]
        if not looks_like_function_definition(view_lines, index):
            continue

        match = FUNCTION_PATTERN.match(code_line.strip())
        name = match.group("name") if match else "<multiline signature>"

        if name in CONTROL_KEYWORDS or name == "main":
            continue

        if not is_doxygen_comment(lines, index, lookback):
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
        description="Find C/C++/CUDA functions without nearby Doxygen comments."
    )
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Maximum number of preceding lines to inspect for Doxygen comments.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = []

    for path in iter_source_files(root):
        findings.extend(scan_file(path, args.lookback))

    if not findings:
        print("No undocumented functions found.")
        return

    for finding in findings:
        relative_path = finding["file"].relative_to(root)
        print(
            f"{relative_path}:{finding['line']}: {finding['name']} -> {finding['signature']}"
        )

    print(f"\nFound {len(findings)} possible undocumented function(s).")


if __name__ == "__main__":
    main()
