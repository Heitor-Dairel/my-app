from src.backend.helpers import str_verifi_val, bin_verifi_val


class BinCodeDecode:
    r"""
    Convert between text and binary representation.

    This class allows transforming a text string into an 8-bit binary string
    and vice versa. It validates the input before conversion using auxiliary
    functions (`str_verifi_val` and `bin_verifi_val`).

    Attributes:
        value (str): Text to convert to binary or binary string to convert to text.

    Methods:
        tobin -> str:
            Convert `value` from text to an 8-bit binary string.
        totxt -> str | None:
            Convert `value` from binary string back to text if valid, else None.
    """

    def __init__(self, value: str) -> None:
        """
        Initialize a BinCodeDecode instance.

        Args:
            value (str): Text or binary string to convert.
        """

        self.value: str = value

    @property
    def tobin(self) -> str | None:
        r"""
        Convert the input text (`value`) to its 8-bit binary representation.

        Process:
            1. Validate the text with `str_verifi_val`, ensuring all characters
            have Unicode code points ≤ 255.
            2. Encode the validated text using `latin-1`.
            3. Convert each byte to an 8-bit binary string.
            4. Join the binary values with spaces.

        Returns:
            return (str | None): A space-separated binary string, or None if the result is empty.
        """

        text_verifi: str = str_verifi_val(self.value)
        result: str = " ".join(
            [format(byt, "08b") for byt in text_verifi.encode("latin-1")]
        )
        return result or None

    @property
    def totxt(self) -> str | None:
        r"""
        Convert a binary string (`value`) back to text.

        Steps:
        1. Validate the binary string with `bin_verifi_val`.
        2. Convert each 8-bit segment to a byte.
        3. Decode bytes using `latin-1`.

        Returns:
            return (str | None): Decoded text if valid, else None.
        """

        if bin_verifi_val(self.value):
            result: str = bytes([int(bin, 2) for bin in self.value.split()]).decode(
                "latin-1"
            )
            return result

        return None


if __name__ == "__main__":

    binario = BinCodeDecode("a€").tobin
    txt = BinCodeDecode("10110010 10111101 11100001").totxt

    print(binario)
    print(txt)
