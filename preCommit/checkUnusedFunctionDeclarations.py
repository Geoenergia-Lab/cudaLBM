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

# Pattern to detect function declarations (ending with ';')
DECLARATION_PATTERN = re.compile(
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
    \s*;
    """,
    re.VERBOSE,
)


def sanitize_line(line):
    """
    Remove string literals and comments from a line, replacing them with spaces,
    so that searching for function names does not match inside strings/comments.
    """
    result = []
    i = 0
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    while i < len(line):
        if in_line_comment:
            result.append(" ")
            i += 1
            continue
        if in_block_comment:
            if line.startswith("*/", i):
                in_block_comment = False
                result.extend([" ", " "])
                i += 2
            else:
                result.append(" ")
                i += 1
            continue
        if in_string:
            result.append(" ")
            if line[i] == "\\":
                result.append(" ")
                i += 2
                continue
            if line[i] == '"':
                in_string = False
                result.append(" ")
            i += 1
            continue
        if in_char:
            result.append(" ")
            if line[i] == "\\":
                result.append(" ")
                i += 2
                continue
            if line[i] == "'":
                in_char = False
                result.append(" ")
            i += 1
            continue
        # Not in string/comment
        if line.startswith("//", i):
            in_line_comment = True
            result.extend([" ", " "])
            i += 2
            continue
        if line.startswith("/*", i):
            in_block_comment = True
            result.extend([" ", " "])
            i += 2
            continue
        if line[i] == '"':
            in_string = True
            result.append(" ")
            i += 1
            continue
        if line[i] == "'":
            in_char = True
            result.append(" ")
            i += 1
            continue
        result.append(line[i])
        i += 1
    return "".join(result)


def is_function_declaration_or_definition(line):
    """Return True if the line is a function declaration or definition (based on pattern)."""
    stripped = line.strip()
    if not stripped:
        return False
    if DECLARATION_PATTERN.match(stripped) or FUNCTION_PATTERN.match(stripped):
        return True
    return False


def extract_function_info(lines, index, scope_stack):
    """
    Given a function definition line and current scope stack, return
    (name, qualified_name) or None if not applicable.
    """
    line = lines[index].strip()
    match = FUNCTION_PATTERN.match(line)
    if not match:
        return None
    name = match.group("name")
    if name in CONTROL_KEYWORDS or name == "main":
        return None

    # Exclude constructors, destructors, operators
    if name.startswith("~") or name.startswith("operator"):
        return None
    # Constructor may have same name as class; we skip those by checking if
    # the name equals the last scope element (class name).
    if scope_stack and name == scope_stack[-1]:
        return None

    # Determine qualified name
    qualified = None
    # Check if line already contains qualified name: e.g., "ReturnType Class::method(...)"
    # Look for '::' before the function name
    name_pos = line.find(name)
    if name_pos > 0:
        prefix = line[:name_pos]
        # Find last '::' before name
        last_colon = prefix.rfind("::")
        if last_colon != -1:
            class_part = prefix[last_colon + 2 :].strip()
            # class_part should be a valid identifier (maybe with namespaces)
            if re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", class_part):
                # Use the full prefix up to the name, but we already have class_part
                qualified = class_part + "::" + name
        else:
            # No '::', so use scope_stack
            if scope_stack:
                qualified = "::".join(scope_stack) + "::" + name
    else:
        # name at start? unlikely
        if scope_stack:
            qualified = "::".join(scope_stack) + "::" + name

    if not qualified:
        qualified = name

    return name, qualified


def collect_functions(root):
    """
    Scan all source files and return a list of function definitions:
    each is a dict with keys: file, line, name, qualified_name, scope_stack.
    """
    functions = []

    for path in iter_source_files(root):
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        scope_stack = []
        brace_depth = 0
        scope_depths = []  # stack of brace depths when scope was opened

        # Use strip_block_comments to get (index, cleaned_line) for non-empty lines
        for idx, code_line in strip_block_comments(lines):
            stripped = code_line.strip()

            # Update scope stack based on class/struct/namespace declarations
            scope_match = re.match(
                r"^(?:class|struct|namespace)\s+([A-Za-z_]\w*)\s*\{?",
                stripped,
            )
            if scope_match and not stripped.startswith(("//", "/*", "*")):
                if not stripped.endswith(";"):
                    name = scope_match.group(1)
                    scope_stack.append(name)
                    scope_depths.append(brace_depth)

            # Update brace depth using the cleaned line (no braces inside comments)
            brace_depth += code_line.count("{") - code_line.count("}")

            # Pop scopes if we've exited them
            while scope_depths and brace_depth <= scope_depths[-1]:
                scope_depths.pop()
                if scope_stack:
                    scope_stack.pop()

            # Build a view of lines for detection: original up to idx, then cleaned line
            temp_lines = lines[:idx] + [code_line]
            if not looks_like_function_definition(temp_lines, idx):
                continue

            # Skip constructor initializer list entries
            if is_initializer_list_entry(lines, idx):
                continue

            # Extract name and qualified name using the current scope stack
            info = extract_function_info(lines, idx, scope_stack)
            if info:
                name, qualified = info
                functions.append(
                    {
                        "file": path,
                        "line": idx + 1,
                        "name": name,
                        "qualified_name": qualified,
                        "raw_line": stripped,
                    }
                )

    return functions


def find_usages(root, functions):
    """
    Scan all files and determine which functions are used.
    Returns a set of function names (unqualified) that are considered used.
    We also return a set of qualified names used.
    """
    used_names = set()
    used_qualified = set()

    # Preprocess all files: read lines and sanitize (remove strings/comments)
    # We'll iterate through all files, but for performance we could cache.
    for path in iter_source_files(root):
        raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line_no, raw_line in enumerate(raw_lines, start=1):
            sanitized = sanitize_line(raw_line)
            stripped = sanitized.strip()
            if not stripped:
                continue
            # Exclude lines that are function declarations or definitions
            if is_function_declaration_or_definition(stripped):
                continue

            # For each function, check if its name appears
            for func in functions:
                name = func["name"]
                qualified = func["qualified_name"]

                # Check qualified name first (e.g., MyClass::foo)
                # We need to look for "qualified(" or "qualified " or "&qualified"
                if qualified in stripped:
                    # Simple heuristic: if the qualified name appears, it's likely used
                    used_qualified.add(qualified)
                    used_names.add(name)  # also mark unqualified as used (conservative)
                    continue

                # Check unqualified name as a whole word
                # We don't want to match substrings. Use regex word boundaries.
                if re.search(r"\b" + re.escape(name) + r"\b", stripped):
                    # Determine if it's a plausible use: followed by '(' or preceded by
                    # '.', '->', '::', '&', '=', etc. But for simplicity, we'll mark
                    # as used if the name appears at all. This may over-mark.
                    # Better: check if the name is followed by '(' (ignoring spaces)
                    # or preceded by '&' or '.' or '->' or '::'
                    # We'll implement a simple check.
                    # Find all occurrences of name
                    for m in re.finditer(r"\b" + re.escape(name) + r"\b", stripped):
                        pos = m.start()
                        # Check preceding character
                        prev_char = stripped[pos - 1] if pos > 0 else ""
                        # Check following character (skipping spaces)
                        j = m.end()
                        while j < len(stripped) and stripped[j].isspace():
                            j += 1
                        next_char = stripped[j] if j < len(stripped) else ""
                        # Heuristic: function call if next_char == '('
                        # or if prev_char in {'.', '>', ':'} (for method calls/static)
                        # or if prev_char == '&' (address taken)
                        if (
                            next_char == "("
                            or prev_char in {".", ">", ":"}
                            or prev_char == "&"
                        ):
                            used_names.add(name)
                            break  # one use is enough

    return used_names, used_qualified


def main():
    parser = argparse.ArgumentParser(
        description="Find C/C++/CUDA functions and class methods that are not used anywhere in the codebase."
    )
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan.")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    # Step 1: collect all function definitions
    print("Collecting function definitions...")
    functions = collect_functions(root)
    print(
        f"Found {len(functions)} candidate functions (excluding constructors/destructors/operators)."
    )

    if not functions:
        print("No functions to check.")
        return

    # Step 2: find usages
    print("Searching for usages...")
    used_names, used_qualified = find_usages(root, functions)

    # Step 3: report functions with no usage
    unused = []
    for func in functions:
        # A function is considered used if either its qualified name or
        # unqualified name was found in a usage context.
        # We use unqualified name as fallback; if any function with same name is used,
        # we assume this one is used too (to reduce false positives).
        if (
            func["qualified_name"] not in used_qualified
            and func["name"] not in used_names
        ):
            unused.append(func)

    if not unused:
        print("No unused functions found.")
        return

    print(f"\nPotentially unused functions ({len(unused)}):")
    for func in unused:
        rel_path = func["file"].relative_to(root)
        print(
            f"{rel_path}:{func['line']}: {func['qualified_name']} -> {func['raw_line']}"
        )


if __name__ == "__main__":
    main()
