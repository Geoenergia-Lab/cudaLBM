#!/usr/bin/env python3

from pathlib import Path
import re

# -------------------- Constants --------------------

SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".cu",
    ".cuh",
}

# Keywords that are never function definitions (superset for both scripts)
CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "delete",
    "new",
    "else",
    "do",
    "typedef",
    "using",
    "class",
    "struct",
    "union",
    "namespace",
}

# Robust function pattern (handles templates, complex return types,
# operators, attributes, and CUDA specifiers)
FUNCTION_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        template\s*<[^;{}]*>\s*
    )?
    (?P<prefix>
        (?:
            (?:__host__|__device__|__global__|\[\[[^\]]*\]\])\s+
        )*
    )
    (?P<return_type>
        (?:[A-Za-z_][\w:<>,\s\*&]*?)\s+
    )?
    (?:
        (?:inline|constexpr|consteval|static|virtual|friend|explicit|constinit)\s+
    )*
    (?:
        [*&]+\s*
    )?
    (?P<name>
        ~?[A-Za-z_]\w* | operator\s*\S+
    )
    \s*
    \(
        [^;{}]*
    \)
    \s*
    (?:
        const\s* |
        \[\[[^\]]*\]\]\s* |
        constexpr\s* |
        noexcept(?:\s*\([^)]*\))?\s* |
        override\s* |
        final\s* |
        __host__\s* |
        __device__\s* |
        __global__\s*
    )*
    # Allow optional initializer list before the opening brace
    (?:
        (?:\s*:\s*[^;{}]*)?\s*\{
        |
        $
    )
    """,
    re.VERBOSE,
)

# -------------------- Helper Functions --------------------


def looks_like_function_definition(lines, index):
    """
    Heuristically determine if the line at `index` begins a function definition.
    Handles single‑line and multi‑line signatures.
    """
    line = lines[index].strip()
    if not line:
        return False

    # Skip obvious non‑function lines
    if line.startswith(
        (
            "//",
            "/*",
            "*",
            "#",
            "else",
            "if ",
            "for ",
            "while ",
            "switch ",
            "case ",
            "return ",
            "throw ",
            "typedef ",
            "using ",
            "class ",
            "struct ",
            "union ",
            "namespace ",
            "static_assert",
        )
    ):
        return False

    if "(" not in line or ")" not in line:
        # No parentheses on this line, so not a function signature
        return False

    # First try the regex (works for single‑line signatures)
    if FUNCTION_PATTERN.match(line):
        return True

    # Fallback 1: line ends with ')' or qualifiers, check next line for '{'
    if line.endswith((")", "const", "noexcept", "override", "final")):
        next_index = index + 1
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if next_index < len(lines):
            return lines[next_index].strip().startswith("{")

    # Fallback 2: multi‑line signature where '(' is on this line but ')' is later.
    # We look forward for the matching ')' (ignoring nested parentheses) and then
    # see if the next non‑empty line starts with '{'.
    # We only enter this fallback if the current line has an opening parenthesis
    # and does NOT have a closing parenthesis.
    if "(" in line and ")" not in line:
        depth = 0
        i = index
        # Find the first '(' on this line and start counting depth
        start_pos = line.find("(")
        if start_pos == -1:
            return False
        # Move through characters from this line onward
        while i < len(lines):
            current = lines[i]
            if i == index:
                # Start from the opening parenthesis position
                j = start_pos
            else:
                j = 0
            while j < len(current):
                ch = current[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        # Found matching ')', now look for '{' after this line or on following lines
                        next_i = i
                        next_j = j + 1
                        # Skip whitespace on the same line and subsequent lines
                        while next_i < len(lines):
                            if next_i == i:
                                rest = current[next_j:]
                            else:
                                rest = lines[next_i].strip()
                            if rest.startswith("{"):
                                return True
                            if rest.strip() and not rest.startswith(
                                ("const", "noexcept", "override", "final")
                            ):
                                # Some other content that is not just qualifiers,
                                # so likely not a function definition
                                break
                            next_i += 1
                        # If we reached end without finding '{', return False
                        return False
                j += 1
            i += 1
        # If we exit without finding closing ')' or '{', return False
        return False

    return False


def is_initializer_list_entry(lines, index):
    """Return True for constructor initializer entries such as ``mesh_(mesh) {}``."""
    previous_index = index - 1
    while previous_index >= 0 and not lines[previous_index].strip():
        previous_index -= 1
    if previous_index < 0:
        return False
    previous = lines[previous_index].strip()
    current = lines[index].strip()
    return (previous.startswith(":") or previous.endswith(",")) and re.fullmatch(
        r"[A-Za-z_][\w:]*\s*\([^;{}]*\)\s*(?:\{\s*\})?;?", current
    ) is not None


def strip_block_comments(lines):
    """
    Generator that yields (index, cleaned_line) pairs,
    removing block comments and skipping lines fully inside comments.
    """
    in_block_comment = False
    for idx, line in enumerate(lines):
        code_line = line
        if in_block_comment:
            if "*/" in code_line:
                in_block_comment = False
                code_line = code_line.split("*/", 1)[1]
            else:
                continue
        if "/*" in code_line:
            code_line, comment = code_line.split("/*", 1)
            if "*/" not in comment:
                in_block_comment = True
        if code_line.strip():  # only return non‑empty lines
            yield idx, code_line


def iter_source_files(root: Path):
    """Yield all source files under `root` that match SOURCE_EXTENSIONS and are not in excluded dirs."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if any(part in {".git", "build", "node_modules"} for part in path.parts):
            continue
        yield path
