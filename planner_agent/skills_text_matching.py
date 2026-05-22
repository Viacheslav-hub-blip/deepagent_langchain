"""Hermes-style fuzzy text replacement utilities.

This is a compact extraction of the fuzzy patching chain used by Hermes for
file and skill edits. It is intentionally framework-agnostic and works on
plain strings.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Tuple


UNICODE_MAP = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2014": "--",
    "\u2013": "-",
    "\u2026": "...",
    "\u00a0": " ",
}


def _unicode_normalize(text: str) -> str:
    for char, replacement in UNICODE_MAP.items():
        text = text.replace(char, replacement)
    return text


def fuzzy_find_and_replace(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> Tuple[str, int, Optional[str], Optional[str]]:
    """Find and replace text using a chain of increasingly fuzzy strategies."""

    if not old_string:
        return content, 0, None, "old_string cannot be empty"
    if old_string == new_string:
        return content, 0, None, "old_string and new_string are identical"

    strategies: List[Tuple[str, Callable[[str, str], List[Tuple[int, int]]]]] = [
        ("exact", _strategy_exact),
        ("line_trimmed", _strategy_line_trimmed),
        ("whitespace_normalized", _strategy_whitespace_normalized),
        ("indentation_flexible", _strategy_indentation_flexible),
        ("escape_normalized", _strategy_escape_normalized),
        ("trimmed_boundary", _strategy_trimmed_boundary),
        ("unicode_normalized", _strategy_unicode_normalized),
        ("block_anchor", _strategy_block_anchor),
        ("context_aware", _strategy_context_aware),
    ]

    for strategy_name, strategy in strategies:
        matches = strategy(content, old_string)
        if not matches:
            continue
        if len(matches) > 1 and not replace_all:
            return (
                content,
                0,
                None,
                f"Found {len(matches)} matches for old_string. Provide more context or use replace_all=True.",
            )
        return _apply_replacements(content, matches, new_string), len(matches), strategy_name, None

    return content, 0, None, "Could not find a match for old_string in the file"


def _apply_replacements(content: str, matches: List[Tuple[int, int]], new_string: str) -> str:
    result = content
    for start, end in sorted(matches, key=lambda item: item[0], reverse=True):
        result = result[:start] + new_string + result[end:]
    return result


def _strategy_exact(content: str, pattern: str) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    start = 0
    while True:
        pos = content.find(pattern, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(pattern)))
        start = pos + 1
    return matches


def _strategy_line_trimmed(content: str, pattern: str) -> List[Tuple[int, int]]:
    return _find_normalized_matches(
        content,
        content.split("\n"),
        [line.strip() for line in content.split("\n")],
        pattern,
        "\n".join(line.strip() for line in pattern.split("\n")),
    )


def _strategy_whitespace_normalized(content: str, pattern: str) -> List[Tuple[int, int]]:
    def normalize(value: str) -> str:
        return re.sub(r"[ \t]+", " ", value)

    normalized_pattern = normalize(pattern)
    normalized_content = normalize(content)
    matches = _strategy_exact(normalized_content, normalized_pattern)
    return _map_normalized_positions(content, normalized_content, matches)


def _strategy_indentation_flexible(content: str, pattern: str) -> List[Tuple[int, int]]:
    return _find_normalized_matches(
        content,
        content.split("\n"),
        [line.lstrip() for line in content.split("\n")],
        pattern,
        "\n".join(line.lstrip() for line in pattern.split("\n")),
    )


def _strategy_escape_normalized(content: str, pattern: str) -> List[Tuple[int, int]]:
    unescaped = pattern.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    if unescaped == pattern:
        return []
    return _strategy_exact(content, unescaped)


def _strategy_trimmed_boundary(content: str, pattern: str) -> List[Tuple[int, int]]:
    pattern_lines = pattern.split("\n")
    if not pattern_lines:
        return []
    pattern_lines[0] = pattern_lines[0].strip()
    if len(pattern_lines) > 1:
        pattern_lines[-1] = pattern_lines[-1].strip()
    normalized_pattern = "\n".join(pattern_lines)

    content_lines = content.split("\n")
    matches = []
    for idx in range(len(content_lines) - len(pattern_lines) + 1):
        block = content_lines[idx : idx + len(pattern_lines)]
        block = block.copy()
        block[0] = block[0].strip()
        if len(block) > 1:
            block[-1] = block[-1].strip()
        if "\n".join(block) == normalized_pattern:
            start, end = _calculate_line_positions(content_lines, idx, idx + len(pattern_lines), len(content))
            matches.append((start, end))
    return matches


def _strategy_unicode_normalized(content: str, pattern: str) -> List[Tuple[int, int]]:
    normalized_content = _unicode_normalize(content)
    normalized_pattern = _unicode_normalize(pattern)
    matches = _strategy_exact(normalized_content, normalized_pattern)
    if not matches:
        matches = _strategy_line_trimmed(normalized_content, normalized_pattern)
    orig_to_norm = _build_orig_to_norm_map(content)
    return _map_positions_norm_to_orig(orig_to_norm, matches)


def _strategy_block_anchor(content: str, pattern: str) -> List[Tuple[int, int]]:
    pattern_lines = pattern.split("\n")
    if len(pattern_lines) < 3:
        return []
    first = pattern_lines[0].strip()
    last = pattern_lines[-1].strip()
    middle = "\n".join(pattern_lines[1:-1])

    content_lines = content.split("\n")
    matches = []
    for idx in range(len(content_lines) - len(pattern_lines) + 1):
        candidate = content_lines[idx : idx + len(pattern_lines)]
        if candidate[0].strip() != first or candidate[-1].strip() != last:
            continue
        candidate_middle = "\n".join(candidate[1:-1])
        ratio = SequenceMatcher(None, middle, candidate_middle).ratio()
        if ratio >= 0.75:
            start, end = _calculate_line_positions(content_lines, idx, idx + len(pattern_lines), len(content))
            matches.append((start, end))
    return matches


def _strategy_context_aware(content: str, pattern: str) -> List[Tuple[int, int]]:
    pattern_lines = pattern.split("\n")
    if len(pattern_lines) < 2:
        return []
    content_lines = content.split("\n")
    matches = []
    line_count = len(pattern_lines)
    for idx in range(len(content_lines) - line_count + 1):
        candidate_lines = content_lines[idx : idx + line_count]
        overlap = sum(
            1 for wanted, candidate in zip(pattern_lines, candidate_lines)
            if wanted.strip() == candidate.strip()
        )
        if overlap / max(line_count, 1) >= 0.5:
            start, end = _calculate_line_positions(content_lines, idx, idx + line_count, len(content))
            matches.append((start, end))
    return matches


def _find_normalized_matches(
    original_content: str,
    original_lines: List[str],
    normalized_lines: List[str],
    original_pattern: str,
    normalized_pattern: str,
) -> List[Tuple[int, int]]:
    pattern_lines = normalized_pattern.split("\n")
    if not pattern_lines:
        return []
    matches = []
    for idx in range(len(normalized_lines) - len(pattern_lines) + 1):
        block = normalized_lines[idx : idx + len(pattern_lines)]
        if "\n".join(block) == normalized_pattern:
            start, end = _calculate_line_positions(original_lines, idx, idx + len(pattern_lines), len(original_content))
            matches.append((start, end))
    return matches


def _calculate_line_positions(
    content_lines: List[str],
    start_line: int,
    end_line: int,
    total_chars: int,
) -> Tuple[int, int]:
    start = 0
    for idx in range(start_line):
        start += len(content_lines[idx]) + 1
    end = start
    for idx in range(start_line, end_line):
        end += len(content_lines[idx]) + 1
    if end > total_chars:
        end = total_chars
    return start, end


def _map_normalized_positions(
    original_content: str,
    normalized_content: str,
    matches: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    mapping = []
    normalized_index = 0
    for original_index, char in enumerate(original_content):
        while normalized_index < len(normalized_content) and normalized_content[normalized_index] == " " and char in "\t ":
            mapping.append((normalized_index, original_index))
            normalized_index += 1
            break
        if normalized_index < len(normalized_content):
            mapping.append((normalized_index, original_index))
            normalized_index += 1
    results = []
    for start, end in matches:
        original_start = next((orig for norm, orig in mapping if norm == start), None)
        original_end = next((orig for norm, orig in mapping if norm >= end), len(original_content))
        if original_start is not None:
            results.append((original_start, original_end))
    return results


def _build_orig_to_norm_map(original: str) -> List[int]:
    result = []
    normalized_pos = 0
    for char in original:
        result.append(normalized_pos)
        replacement = UNICODE_MAP.get(char)
        normalized_pos += len(replacement) if replacement is not None else 1
    result.append(normalized_pos)
    return result


def _map_positions_norm_to_orig(
    orig_to_norm: List[int],
    normalized_matches: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    norm_to_orig_start = {}
    for original_pos, normalized_pos in enumerate(orig_to_norm[:-1]):
        if normalized_pos not in norm_to_orig_start:
            norm_to_orig_start[normalized_pos] = original_pos

    results = []
    original_len = len(orig_to_norm) - 1
    for normalized_start, normalized_end in normalized_matches:
        if normalized_start not in norm_to_orig_start:
            continue
        original_start = norm_to_orig_start[normalized_start]
        original_end = original_start
        while original_end < original_len and orig_to_norm[original_end] < normalized_end:
            original_end += 1
        results.append((original_start, original_end))
    return results

