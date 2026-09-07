#!/usr/bin/env python3

from pathlib import Path
import argparse
from common import iter_source_files


def has_trailing_period(line: str) -> bool:
    """
    Check if a line of text, after removing trailing whitespace and an optional
    closing block comment delimiter '*/', ends with a period.
    """
    # Remove trailing whitespace
    stripped = line.rstrip()
    # If the line ends with '*/', remove it (and any preceding whitespace)
    if stripped.endswith("*/"):
        # Remove '*/' and then strip trailing whitespace again
        content = stripped[:-2].rstrip()
    else:
        content = stripped
    # Now check if content ends with a period
    return content.endswith(".")


def is_doxygen_line_comment(line: str) -> bool:
    """Return True if the line (stripped) starts with a Doxygen line comment marker."""
    return line.lstrip().startswith(("///", "//!"))


def starts_doxygen_block(line: str) -> bool:
    """Return True if the line (stripped) starts with a Doxygen block comment opener."""
    return line.lstrip().startswith(("/**", "/*!"))


def scan_file(path: Path):
    """Scan a single file and return a list of findings (line numbers and content)."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    findings = []
    in_doxygen_block = False

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if not in_doxygen_block:
            # Outside of any block comment
            if is_doxygen_line_comment(raw_line):
                # It's a Doxygen line comment: check the entire line
                if has_trailing_period(raw_line):
                    findings.append((idx, raw_line.strip()))
            elif starts_doxygen_block(raw_line):
                # Start of a Doxygen block comment
                in_doxygen_block = True
                # Check this opening line for a trailing period, but be careful:
                # The opener may be just '/**' or '/*!' with no text.
                # We still check because it might be a one-liner: '/** Brief. */'
                if has_trailing_period(raw_line):
                    findings.append((idx, raw_line.strip()))
                # If the block closes on the same line (e.g., '/** text */'),
                # we need to end the block state.
                # We detect closing '*/' and if it's present, set in_doxygen_block = False.
                # Note: We must check for closing after the opener, not just any '*/'
                if "*/" in raw_line:
                    in_doxygen_block = False
            # else: not a Doxygen comment, ignore
        else:
            # We are inside a Doxygen block comment
            # Check this line for trailing period, but only if it's not just the closing
            # delimiter or purely whitespace. has_trailing_period handles '*/' removal.
            if has_trailing_period(raw_line):
                findings.append((idx, raw_line.strip()))
            # Check if the block ends on this line
            if "*/" in raw_line:
                in_doxygen_block = False

    return findings


def main():
    parser = argparse.ArgumentParser(
        description="Find Doxygen comment lines that end with a full stop (.)"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root to scan. Defaults to current directory.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    total_findings = []

    for path in iter_source_files(root):
        findings = scan_file(path)
        for line_num, content in findings:
            rel_path = path.relative_to(root)
            total_findings.append((rel_path, line_num, content))

    if not total_findings:
        print("No Doxygen comment lines ending with a full stop found.")
        return

    for rel_path, line_num, content in total_findings:
        print(f"{rel_path}:{line_num}: {content}")

    print(
        f"\nFound {len(total_findings)} Doxygen comment line(s) ending with a full stop."
    )


if __name__ == "__main__":
    main()
