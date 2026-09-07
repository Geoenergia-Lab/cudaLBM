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

DOXYGEN_MARKERS = ("/**", "///", "//!", "/*!", "@brief", "@param", "@return", "@tparam")


def find_preceding_doxygen_comment(lines, function_line, lookback=20):
    """
    Return the list of lines forming a Doxygen comment block immediately
    preceding the function at `function_line`, or None if not found.
    """
    start = max(0, function_line - lookback)
    comment_lines = []
    found_comment = False

    # Scan backwards from just before the function line
    for i in range(function_line - 1, start - 1, -1):
        stripped = lines[i].strip()

        # Blank line? keep going (could be part of the separation)
        if not stripped:
            if found_comment:
                # If we've already found a comment and hit a blank, we stop
                # (the comment block is before the blank)
                break
            continue

        # Check if this line is a comment (starts with // or /* or * inside block)
        is_comment_line = (
            stripped.startswith("///")
            or stripped.startswith("//!")
            or stripped.startswith("/**")
            or stripped.startswith("/*!")
            or stripped.startswith("*")
            or stripped.startswith("/*")
            or stripped.startswith("//")
        )

        if is_comment_line:
            # This is a comment line, add it to our list (in correct order)
            comment_lines.insert(0, lines[i])
            found_comment = True
            # Continue scanning upward for more comment lines
            continue
        else:
            # Not a comment. If we've already seen a comment, this unrelated code
            # breaks the block, so we stop and return what we have.
            if found_comment:
                break
            # Otherwise, maybe it's a qualifier (template, attribute, etc.)
            # that can appear between the comment and the function.
            if re.match(
                r"^(?:template\s*<|\[\[[^\]]*\]\]|(?:__[A-Za-z_]+__\s+)*(?:static|inline|constexpr|consteval|virtual|friend|explicit)\b)",
                stripped,
            ):
                # This line is allowed between comment and function; skip it,
                # but note that we haven't found the comment yet, so keep looking.
                continue
            else:
                # Unrelated code, stop searching
                break

    # Return the comment lines only if we found at least one and it contains
    # a Doxygen marker (we don't want ordinary comments).
    if comment_lines and any(
        any(marker in line for marker in DOXYGEN_MARKERS) for line in comment_lines
    ):
        return comment_lines
    return None


def extract_documented_params(comment_lines):
    """
    Parse Doxygen comment lines and return a set of parameter names documented
    with @param, and a set of template parameter names documented with @tparam.
    """
    text = "\n".join(comment_lines)
    params = set()
    tparams = set()

    # @param[in] name description  or  @param name description
    for m in re.finditer(r"@param(?:\[[^\]]*\])?\s+(\w+)", text):
        params.add(m.group(1))

    # @tparam name description
    for m in re.finditer(r"@tparam\s+(\w+)", text):
        tparams.add(m.group(1))

    return params, tparams


def extract_parameter_names(lines, func_line_index, func_name):
    """
    Extract the actual parameter names from a function definition.
    `lines` is the full list of source lines.
    `func_line_index` is the index of the line where the function signature starts.
    `func_name` is the name of the function (to locate the opening parenthesis).
    Returns a list of parameter names, or an empty list if parsing fails.
    """
    # We will scan from the current line forward until we find the opening '('
    # that belongs to the function. Since the function name appears on this line,
    # we can locate the first '(' after the name.
    line_idx = func_line_index
    full_text = lines[line_idx]
    while line_idx < len(lines):
        line = lines[line_idx]
        # If we are still on the first line, find the '(' after the function name
        if line_idx == func_line_index:
            # Find the position of the function name in the line
            name_pos = line.find(func_name)
            if name_pos == -1:
                # Fallback: just search for '(' anywhere
                paren_pos = line.find("(")
            else:
                paren_pos = line.find("(", name_pos + len(func_name))
            if paren_pos != -1:
                # We found the opening parenthesis; now track depth
                depth = 0
                param_chars = []
                i = paren_pos
                # We'll scan character by character across lines
                while line_idx < len(lines):
                    line = lines[line_idx]
                    while i < len(line):
                        ch = line[i]
                        if ch == "(":
                            depth += 1
                            if depth == 1:
                                # Start of parameter list, skip the '(' itself
                                i += 1
                                continue
                        elif ch == ")":
                            depth -= 1
                            if depth == 0:
                                # End of parameter list
                                param_str = "".join(param_chars)
                                # Now split the parameter string by commas at depth 0
                                return split_parameters(param_str)
                            else:
                                # Nested closing parenthesis
                                pass
                        if depth >= 1:
                            # Inside parameter list (depth >=1), keep characters
                            param_chars.append(ch)
                        i += 1
                    # Move to next line
                    line_idx += 1
                    i = 0
                # If we exit the loop without finding closing ')', return empty
                return []
        else:
            # On subsequent lines, we might find the '(' if the first line didn't have it
            # (e.g., return type on one line, function name and '(' on next)
            # For simplicity, we assume the '(' is on the first line; if not, we could
            # search forward, but this is rare.
            break
        line_idx += 1
    return []


def split_parameters(param_str):
    """
    Split a parameter list string by commas at depth 0 (ignoring commas inside
    parentheses, brackets, or template angle brackets).
    Then extract the last identifier from each segment as the parameter name.
    """
    # Simple parser that tracks (), [], {}, and <> depths
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    depth_angle = 0
    segments = []
    current = []
    for ch in param_str:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "<":
            # Heuristic: could be less-than operator or template; assume template for now
            depth_angle += 1
        elif ch == ">":
            depth_angle -= 1
        elif (
            ch == ","
            and depth_paren == 0
            and depth_bracket == 0
            and depth_brace == 0
            and depth_angle == 0
        ):
            # End of segment
            segments.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    # Last segment
    if current:
        segments.append("".join(current).strip())

    names = []
    for seg in segments:
        if not seg:
            continue
        # Remove default value (everything after '=' at depth 0)
        eq_pos = seg.find("=")
        if eq_pos != -1:
            seg = seg[:eq_pos].strip()
        # Strip trailing attributes: [[...]]
        seg = re.sub(r"\[\[[^\]]*\]\]\s*$", "", seg).strip()
        # Strip trailing array dimensions: [N] or [] possibly multiple
        while True:
            new_seg = re.sub(r"\[[^\[\]]*\]\s*$", "", seg).strip()
            if new_seg == seg:
                break
            seg = new_seg
        # If the segment ends with ')', remove it (to handle references inside parentheses)
        if seg.endswith(")"):
            seg = seg[:-1].strip()
        # Now extract the last identifier
        match = re.search(r"([A-Za-z_]\w*)\s*$", seg)
        if match:
            names.append(match.group(1))
    return names


def extract_template_params(lines, func_line_index):
    """
    Extract template parameter names for a function definition.
    Looks at the current line and, if needed, a few preceding lines
    for the template declaration.
    """
    # Check current line first
    line = lines[func_line_index].strip()
    if line.startswith("template"):
        idx = func_line_index
    else:
        # Search backwards up to 3 lines
        idx = func_line_index - 1
        while idx >= 0 and idx >= func_line_index - 3:
            candidate = lines[idx].strip()
            if not candidate:
                idx -= 1
                continue
            if candidate.startswith("template"):
                break
            # Stop if we encounter a line that clearly isn't part of the template/qualifier
            if not re.match(
                r"^(?:template\s*<|\[\[[^\]]*\]\]|(?:__[A-Za-z_]+__\s+)*(?:static|inline|constexpr|consteval|virtual|friend|explicit)\b)",
                candidate,
            ):
                break
            idx -= 1
        else:
            return []
        line = lines[idx].strip()

    if not line.startswith("template"):
        return []

    start = line.find("<")
    if start == -1:
        return []
    end = line.rfind(">")
    if end == -1 or end < start:
        return []

    tpl_str = line[start + 1 : end]
    parts = [p.strip() for p in tpl_str.split(",")]
    names = []
    for part in parts:
        part_no_default = part.split("=")[0].strip()
        match = re.search(r"([A-Za-z_]\w*)\s*$", part_no_default)
        if match:
            names.append(match.group(1))
    return names


def scan_file(path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    findings = []

    # We need to iterate over original lines but with block comments removed for detection.
    # Use strip_block_comments generator, but we also need the original lines for context.
    for index, code_line in strip_block_comments(lines):
        if is_initializer_list_entry(lines, index):
            continue

        # Create a view of lines for detection (with cleaned current line)
        temp_lines = lines.copy()
        temp_lines[index] = code_line
        if not looks_like_function_definition(temp_lines, index):
            continue

        match = FUNCTION_PATTERN.match(code_line.strip())
        if not match:
            continue
        name = match.group("name") if match else "<multiline signature>"

        if name in CONTROL_KEYWORDS or name == "main":
            continue

        # Find preceding Doxygen comment block (using original lines)
        comment_block = find_preceding_doxygen_comment(lines, index)
        if not comment_block:
            # No Doxygen comment, skip (handled by other script)
            continue

        # Extract documented parameters from the comment
        documented_params, documented_tparams = extract_documented_params(comment_block)

        # Extract actual parameters from the function definition
        actual_params = extract_parameter_names(lines, index, name)

        # For templates, also extract template parameter names
        actual_tparams = extract_template_params(lines, index)

        # Compare
        missing_params = set(actual_params) - documented_params
        extra_params = documented_params - set(actual_params)

        missing_tparams = set(actual_tparams) - documented_tparams
        extra_tparams = documented_tparams - set(actual_tparams)

        if missing_params or extra_params or missing_tparams or extra_tparams:
            findings.append(
                {
                    "file": path,
                    "line": index + 1,
                    "name": name,
                    "signature": code_line.strip(),
                    "missing_params": sorted(missing_params),
                    "extra_params": sorted(extra_params),
                    "missing_tparams": sorted(missing_tparams),
                    "extra_tparams": sorted(extra_tparams),
                }
            )

    return findings


def main():
    parser = argparse.ArgumentParser(
        description="Find C/C++/CUDA functions with mismatched Doxygen @param or @tparam documentation."
    )
    parser.add_argument("root", nargs="?", default=".", help="Repository root to scan.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = []

    for path in iter_source_files(root):
        findings.extend(scan_file(path))

    if not findings:
        print("No mismatched Doxygen documentation found.")
        return

    for f in findings:
        rel_path = f["file"].relative_to(root)
        print(f"{rel_path}:{f['line']}: {f['name']}")
        if f["missing_params"]:
            print(f"    Missing @param for: {', '.join(f['missing_params'])}")
        if f["extra_params"]:
            print(f"    Extra @param for: {', '.join(f['extra_params'])}")
        if f["missing_tparams"]:
            print(f"    Missing @tparam for: {', '.join(f['missing_tparams'])}")
        if f["extra_tparams"]:
            print(f"    Extra @tparam for: {', '.join(f['extra_tparams'])}")
        print(f"    Signature: {f['signature']}\n")

    print(f"Found {len(findings)} function(s) with mismatched Doxygen documentation.")


if __name__ == "__main__":
    main()
