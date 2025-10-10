import unicodedata


def strip_accents(text: str) -> str:
    r"""
    Remove accents from a string.

    This function normalizes the input string using Unicode NFD form and
    removes all combining characters (accents), returning an accent-free
    version of the text.

    Args:
        text (str): Input string possibly containing accented characters.

    Returns:
        return (str): String with accents removed.
    """

    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )
