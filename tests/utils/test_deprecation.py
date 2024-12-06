import pytest

from prophetverse.utils.deprecation import deprecation_warning


def test_deprecation_warning_with_extra_message():
    obj_name = "my_function"
    current_version = "1.2.3"
    extra_message = "Use 'new_function' instead."

    expected_major = 1
    expected_minor = 2 + 1
    expected_patch = 0
    expected_deprecation_version = f"{expected_major}.{expected_minor}.{expected_patch}"

    expected_message = (
        f"Warning: '{obj_name}' is deprecated and will be removed in version"
        f" {expected_deprecation_version}. Please update your code to avoid issues. "
        f"{extra_message}"
    )

    with pytest.warns(FutureWarning) as record:
        deprecation_warning(obj_name, current_version, extra_message)

    assert len(record) == 1
    assert str(record[0].message) == expected_message


def test_deprecation_warning_without_extra_message():
    obj_name = "old_function"
    current_version = "2.5.0"

    expected_major = 2
    expected_minor = 5 + 1
    expected_patch = 0
    expected_deprecation_version = f"{expected_major}.{expected_minor}.{expected_patch}"

    expected_message = (
        f"Warning: '{obj_name}' is deprecated and will be removed in version"
        f" {expected_deprecation_version}. Please update your code to avoid issues. "
    )

    with pytest.warns(FutureWarning) as record:
        deprecation_warning(obj_name, current_version)

    assert len(record) == 1
    assert str(record[0].message) == expected_message


def test_deprecation_warning_invalid_version_format():
    obj_name = "invalid_function"
    invalid_version = "invalid.version"

    with pytest.raises(ValueError) as exc_info:
        deprecation_warning(obj_name, invalid_version)

    assert "Invalid version format. Expected 'major.minor.patch'." in str(
        exc_info.value
    )


def test_deprecation_warning_high_minor_version():
    obj_name = "edge_function"
    current_version = "3.99.1"

    expected_major = 3
    expected_minor = 99 + 1
    expected_patch = 0
    expected_deprecation_version = f"{expected_major}.{expected_minor}.{expected_patch}"

    expected_message = (
        f"Warning: '{obj_name}' is deprecated and will be removed in version"
        f" {expected_deprecation_version}. Please update your code to avoid issues. "
    )

    with pytest.warns(FutureWarning) as record:
        deprecation_warning(obj_name, current_version)

    assert len(record) == 1
    assert str(record[0].message) == expected_message


def test_deprecation_warning_non_integer_version():
    obj_name = "float_function"
    current_version = "1.2.3.4"

    with pytest.raises(ValueError) as exc_info:
        deprecation_warning(obj_name, current_version)

    assert "Invalid version format. Expected 'major.minor.patch'." in str(
        exc_info.value
    )


def test_deprecation_warning_incomplete_version():
    obj_name = "incomplete_function"
    current_version = "1.2"

    with pytest.raises(ValueError) as exc_info:
        deprecation_warning(obj_name, current_version)

    assert "Invalid version format. Expected 'major.minor.patch'." in str(
        exc_info.value
    )
