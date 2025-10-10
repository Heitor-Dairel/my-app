import threading
from src.backend.utils import strip_accents
from src.backend.helpers import beep_play
from src.backend.constants import TEXTMORSE, MORSETEXT


class MorseCodeDecode:
    r"""
    Encode and decode Morse code with optional audible playback.

    This class allows converting between plain text and Morse code. It automatically
    removes accents from input text, ignores unsupported characters, and can play
    Morse code as beeps in a separate thread.

    Attributes:
        value (str): The string to encode to or decode from Morse code.

    Methods:
        tomorse(sound: bool = False) -> str | None:
            Convert `value` from text to Morse code. If `sound` is True, plays the
            Morse code as audible beeps in a separate thread.
        totext() -> str | None:
            Convert `value` from Morse code back to plain text. Invalid symbols are ignored.
        _soundmorse(value: str) -> None:
            Static helper method that plays a Morse code string as audible beeps.
    """

    def __init__(self, value: str) -> None:
        """
        Initialize a MorseCodeDecode instance.

        Args:
            value (str): Text or Morse code to encode/decode.
        """

        self.value: str = value

    def tomorse(self, sound: bool = False) -> str | None:
        r"""
        Convert the stored text (`value`) to Morse code.

        This method removes accents and converts the text to uppercase before
        encoding. Unsupported characters are ignored. Optionally, the resulting
        Morse code can be played as audible beeps in a separate thread.

        Args:
            sound (bool, optional): If True, plays the Morse code as sound. Defaults to False.

        Returns:
            return (str | None): Encoded Morse code string if valid characters exist, else None.
        """

        text_prepared: str = strip_accents(self.value).upper()
        result: str = " ".join([TEXTMORSE.get(tx, "") for tx in text_prepared])
        if result:
            if sound:
                threading.Thread(target=beep_play, args=(result,)).start()
            return result

        return None

    def totext(self) -> str | None:
        r"""
        Convert Morse code (`value`) back to plain text.

        The Morse code string is split into individual symbols and each is mapped
        to its corresponding character using **MORSETEXT**. Symbols not in the
        mapping are ignored.

        Returns:
            return (str | None): Decoded text if valid Morse code is present, else None.
        """

        cod_morse: list[str] = self.value.split()
        result: str = "".join([MORSETEXT.get(mor, "") for mor in cod_morse])
        return result or None


if __name__ == "__main__":

    cod_morse = MorseCodeDecode("ola t").tomorse(sound=True)
    cod_morse1 = MorseCodeDecode("ó").totext()
    print(f"{cod_morse}")
    print(f"{cod_morse1}")


# cls ; python -W ignore -m src.backend.modules.codec.morse_code_decode
