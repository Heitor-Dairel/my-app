import re


class BinCodeDecode:
    r"""
    Convert between plain text and binary representation.

    This class provides utilities to transform a text string into an
    8-bit binary representation and back. It includes validation methods
    to ensure that the input text or binary string conforms to expected formats.

    Attributes:
        value (str): The text to be converted to binary or the binary
            string to be decoded into text.

    Methods:
        tobin -> str | None:
            Convert `value` from text to an 8-bit binary string.
        totxt -> str | None:
            Convert `value` from binary string to text if valid, else None.
    """

    def __init__(self, value: str) -> None:
        """
        Initialize a BinCodeDecode instance.

        Args:
            value (str): The input text or binary string to convert.
        """

        self.value: str = value

    @staticmethod
    def _str_verifi_val(value: str) -> str:
        r"""
        Validate a string by keeping only characters with Unicode code points ≤ 255.

        This ensures compatibility with `latin-1` encoding before conversion
        to binary.

        Args:
            value (str): Input string to validate.

        Returns:
            str: The string containing only valid characters (code point ≤ 255).
        """

        return "".join(c for c in value if ord(c) <= 255)

    @staticmethod
    def _bin_verifi_val(value: str) -> bool:
        r"""
        Verify whether a string is a valid binary representation.

        The input is considered valid if:
            1. All characters (excluding spaces) are either '0' or '1'.
            2. Each binary group, separated by spaces, consists of exactly 8 bits.

        Args:
            value (str): Binary string to validate.

        Returns:
            bool: True if valid binary representation, False otherwise.
        """

        text: str = value
        contains_zero_one: bool = set(re.sub(r"[\s]", "", text)).issubset({"0", "1"})
        size_binary: bool = all(len(i) == 8 for i in text.split())
        return contains_zero_one and size_binary

    def tobin(self) -> str | None:
        r"""
        Convert the stored text (`value`) into its 8-bit binary representation.

        Process:
            1. Validate text with `_str_verifi_val`, keeping only characters
            with Unicode code points ≤ 255.
            2. Encode the validated text using `latin-1`.
            3. Convert each byte into an 8-bit binary string.
            4. Join binary values with spaces.

        Returns:
            (str | None): Space-separated 8-bit binary string,
            or None if the result is empty.
        """

        text_verifi: str = BinCodeDecode._str_verifi_val(self.value)
        result: str = " ".join(
            [format(byt, "08b") for byt in text_verifi.encode("latin-1")]
        )
        return result or None

    def totxt(self) -> str | None:
        r"""
        Convert the stored binary string (`value`) back into text.

        Steps:
            1. Validate the binary string using `_bin_verifi_val`.
            2. Convert each 8-bit binary segment to an integer byte.
            3. Decode the resulting bytes using `latin-1`.

        Returns:
            (str | None): Decoded text if valid, otherwise None.
        """

        if BinCodeDecode._bin_verifi_val(self.value):
            result: str = bytes([int(bin, 2) for bin in self.value.split()]).decode(
                "latin-1"
            )
            return result

        return None


if __name__ == "__main__":

    binario = BinCodeDecode("a€").tobin()
    txt = BinCodeDecode("10110010 10111101 11100001").totxt()

    print(binario)
    print(txt)

    # python -W ignore -m src.backend.modules.codec.bin_code_decode
