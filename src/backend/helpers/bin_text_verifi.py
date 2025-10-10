import re


def str_verifi_val(value: str) -> str:
    r"""
    Validate a string by keeping only characters with Unicode code points ≤ 255.

    Args:
        value (str): Input string to validate.

    Returns:
        return (str): String containing only valid characters (code point ≤ 255).
    """

    return "".join(c for c in value if ord(c) <= 255)


def bin_verifi_val(value: str) -> bool:
    r"""
    Check if a string is a valid binary representation.

    Conditions for validity:
        1. All characters (ignoring spaces) are '0' or '1'.
        2. Each binary group separated by spaces has exactly 8 bits.

    Args:
        value (str): Binary string to check.

    Returns:
        return (bool): True if valid, False otherwise.
    """

    text: str = value
    contains_zero_one: bool = set(re.sub(r"[\s]", "", text)).issubset({"0", "1"})
    size_binary: bool = all(len(i) == 8 for i in text.split())
    return contains_zero_one and size_binary


if __name__ == "__main__":

    # print(bin_verifi_val("01001000 01100101 01010109"))

    print(str_verifi_val("a€"))

    # value = "01001000 01100101".split()
    # size_binary: bool

    # print(value)

    # for i in value:

    #     size_binary = True if len(i) == 8 else False

    # print(size_binary)
