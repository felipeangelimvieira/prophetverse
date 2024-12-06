"""Utilities to handle deprecation."""

import warnings


def deprecation_warning(obj_name, current_version, extra_message=""):
    """
    Generate a deprecation warning for an object.

    Parameters
    ----------
        obj_name (str): The name of the object to be deprecated.
        current_version (str): The current version in the format 'major.minor.patch'.

    Returns
    -------
        str: A deprecation warning message.
    """
    try:
        # Parse the current version into components
        major, minor, patch = map(int, current_version.split("."))
        # Calculate the deprecation version (1 minor releases ahead)
        new_minor = minor + 1
        deprecation_version = f"{major}.{new_minor}.0"
        # Return the deprecation warning
        warnings.warn(
            f"Warning: '{obj_name}' is deprecated and will be removed in version"
            f" {deprecation_version}. Please update your code to avoid issues. "
            f"{extra_message}",
            FutureWarning,
            stacklevel=2,
        )
    except ValueError:
        raise ValueError("Invalid version format. Expected 'major.minor.patch'.")
