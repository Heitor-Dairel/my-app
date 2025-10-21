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

    _cache_binary: dict[int, str] = {}
    _cache_text: dict[str, int] = {}

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
        Convert the stored text (`value`) into an 8-bit binary string.

        Process:
            - Validate text with `_str_verifi_val`, keeping only characters with Unicode code points ≤ 255.
            - Encode the validated text using 'latin-1'.
            - Convert each byte into an 8-bit binary string.
            - Join binary strings with spaces.

        Notes:
            - Uses caching to avoid re-processing bytes that have already been converted.

        Returns:
            (str | None): Space-separated 8-bit binary string if the text is valid;
                        otherwise None.
        """

        collection: list[str] = []

        text_verifi: str = BinCodeDecode._str_verifi_val(self.value)

        for byt in text_verifi.encode("latin-1"):

            if byt in BinCodeDecode._cache_binary:
                collection.append(BinCodeDecode._cache_binary[byt])
            else:
                collection.append(format(byt, "08b"))
                BinCodeDecode._cache_binary[byt] = format(byt, "08b")

        result: str = " ".join(collection)
        return result or None

    def totxt(self) -> str | None:
        r"""
        Convert the stored binary string (`value`) back into text.

        Process:
            - Validate the binary string using `_bin_verifi_val`.
            - Split the string into 8-bit segments and convert each segment to an integer.
            - Decode the resulting bytes using 'latin-1'.

        Notes:
            - Uses caching to avoid re-processing previously converted binary segments.

        Returns:
            (str | None): Decoded text if the binary string is valid; otherwise None.
        """

        collection: list[int] = []

        if BinCodeDecode._bin_verifi_val(self.value):

            for bin in self.value.split():

                if bin in BinCodeDecode._cache_text:
                    collection.append(BinCodeDecode._cache_text[bin])
                else:
                    collection.append(int(bin, 2))
                    BinCodeDecode._cache_text[bin] = int(bin, 2)

            result: str = bytes(collection).decode("latin-1")
            return result

        return None


if __name__ == "__main__":

    binario = BinCodeDecode("€a€a").tobin()
    txt = BinCodeDecode("10110010 11100001 11100001").totxt()
    print(binario)
    print(txt)

    # python -W ignore -m src.backend.modules.codec.bin_code_decode
