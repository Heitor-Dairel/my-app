import threading
import time
from src.backend.utils import strip_accents
from src.backend.helpers import beep
from src.backend.constants import TEXTMORSE, MORSESOUND, MORSETEXT


class MorseCodeDecode:
    r"""
    Encode and decode Morse code with optional sound playback.

    This class allows converting text to Morse code and vice versa, with the
    option to play Morse code as audible beeps. It handles accent removal
    automatically and ignores unsupported characters.

    Attributes:
        value (str): The string to be converted to or from Morse code.

    Methods:
        tomorse(sound: bool = False) -> str | None:
            Convert `value` from text to Morse code, optionally playing sound.
        totext() -> str | None:
            Convert `value` from Morse code to plain text.
        _soundmorse(value: str) -> None:
            Static method that plays a Morse code string as sound using beeps.
    """

    def __init__(self, value: str) -> None:
        """
        Initialize a MorseCodeDecode instance.

        Args:
            value (str): Text or Morse code to encode/decode.
        """

        self.value: str = value

    @staticmethod
    def _soundmorse(value: str) -> None:
        r"""
        Play a Morse code string as audible sound.

        Dots (.) and dashes (-) are played with durations defined in **MORSESOUND**,
        and pauses are added between symbols and spaces. Non-Morse characters
        are handled gracefully with corresponding delays.

        Args:
            value (str): Morse code string containing '.' and '-' characters.

        Returns:
            return (None):
        """

        for symbol in value:
            if symbol in ".-":
                beep(duration=MORSESOUND[symbol])
                time.sleep(MORSESOUND["TIME_SYMBOLS"])
            else:
                time.sleep(MORSESOUND[symbol])

        return None

    def tomorse(self, sound: bool = False) -> str | None:
        r"""
        Convert text (`value`) into Morse code.

        The text is first normalized (accents removed and converted to uppercase),
        then each character is mapped to its Morse code using **TEXTMORSE**. Symbols
        are separated by spaces. Characters not in the mapping are ignored.

        If `sound` is True, the Morse code is played in a separate thread.

        Args:
            sound (bool, optional): Play Morse code sound if True. Defaults to False.

        Returns:
            return (str | None): Morse code string if valid characters exist, else None.
        """

        text_prepared: str = strip_accents(self.value).upper()
        result: str = " ".join([TEXTMORSE.get(tx, "") for tx in text_prepared])
        if result:
            if sound:
                threading.Thread(
                    target=MorseCodeDecode._soundmorse, args=(result,)
                ).start()
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

    cod_morse = MorseCodeDecode("a").tomorse(sound=True)
    cod_morse1 = MorseCodeDecode("ó").totext()
    print(f"{cod_morse}")
    print(f"{cod_morse1}")


# cls ; python -W ignore -m src.backend.modules.codec.morse_code_decode
