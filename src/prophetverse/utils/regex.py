
__all__  = [
    "starts_with",
    "exact"
]

def starts_with(prefixes):
    """
    Returns a regular expression pattern that matches strings starting with any of the given prefixes.

    Args:
        prefixes (list): A list of strings representing the prefixes to match.

    Returns:
        str: A regular expression pattern that matches strings starting with any of the given prefixes.
    """
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return rf"^(?:{'|'.join(prefixes)})"


def exact(string):
    """
    Returns a regular expression pattern that matches the exact given string.

    Args:
        string (str): The string to match exactly.

    Returns:
        str: A regular expression pattern that matches the exact given string.
    """
    return rf"^{string}$"