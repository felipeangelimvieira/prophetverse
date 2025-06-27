import re

import pytest

from prophetverse.utils.regex import contains, ends_with


@pytest.mark.parametrize(
    "suffix, string, expected_match",
    [
        ("xyz", "wxyz", True),
        ("xyz", "xyzabc", False),
        ("", "abc", True),
        ("", "", True),
    ],
)
def test_ends_with_single_suffix(suffix, string, expected_match):
    """Test ends_with with a single suffix."""
    pattern = ends_with(suffix)
    if expected_match:
        assert re.search(pattern, string), f"Expected '{string}' to end with '{suffix}'"
    else:
        assert not re.search(
            pattern, string
        ), f"Expected '{string}' not to end with '{suffix}'"


@pytest.mark.parametrize(
    "suffixes, string, expected_match",
    [
        (["xyz", "abc"], "wxyz", True),
        (["xyz", "abc"], "defabc", True),
        (["xyz", "abc"], "xyz123", False),
        (["xyz", "abc"], "abc456", False),
        (["xyz", ""], "wxyz", True),
        (["xyz", ""], "any_string_matches_empty", True),
        (["xyz", ""], "", True),
    ],
)
def test_ends_with_multiple_suffixes(suffixes, string, expected_match):
    """Test ends_with with multiple suffixes."""
    pattern = ends_with(suffixes)
    if expected_match:
        assert re.search(pattern, string), f"Expected '{string}' to end with one of {suffixes}"
    else:
        assert not re.search(
            pattern, string
        ), f"Expected '{string}' not to end with any of {suffixes}"


@pytest.mark.parametrize(
    "substring, string, expected_match",
    [
        ("123", "abc123xyz", True),
        ("123", "abcxyz", False),
        ("", "abc", True),
        ("", "", True),
    ],
)
def test_contains_single_pattern(substring, string, expected_match):
    """Test contains with a single pattern."""
    pattern = contains(substring)
    if expected_match:
        assert re.search(
            pattern, string
        ), f"Expected '{string}' to contain '{substring}'"
    else:
        assert not re.search(
            pattern, string
        ), f"Expected '{string}' not to contain '{substring}'"


@pytest.mark.parametrize(
    "patterns, string, expected_match",
    [
        (["123", "abc"], "xyz123def", True),
        (["123", "abc"], "defabcghi", True),
        (["123", "abc"], "xyzdef", False),
        (["123", ""], "xyz123def", True),
        (["123", ""], "any_string_matches_empty", True),
        (["123", ""], "", True),
    ],
)
def test_contains_multiple_patterns(patterns, string, expected_match):
    """Test contains with multiple patterns."""
    pattern = contains(patterns)
    if expected_match:
        assert re.search(
            pattern, string
        ), f"Expected '{string}' to contain one of {patterns}"
    else:
        assert not re.search(
            pattern, string
        ), f"Expected '{string}' not to contain any of {patterns}"
