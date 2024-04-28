

def starts_with(prefixes):
    """
    Returns a regular expression pattern that matches strings starting with any of the given prefixes.

    Args:
        prefixes (list): A list of strings representing the prefixes to match.

    Returns:
        str: A regular expression pattern that matches strings starting with any of the given prefixes.
    """
    return rf"^(?:{'|'.join(prefixes)})"