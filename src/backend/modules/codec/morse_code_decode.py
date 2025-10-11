import threading
import numpy as np
import sounddevice as sd
from typing import Final, Any, TypedDict
from src.backend.utils import strip_accents


class DictMorse(TypedDict):
    r"""
    TypedDict defining the structure for Morse code beep configuration.

    Attributes:
        FS (int): Sampling frequency in Hz for generating the beep waveform.
        FREQ (int): Frequency of the beep tone in Hz.
        VOLUME (float): Amplitude of the beep tone (0.0 to 1.0).
    """

    FS: int
    FREQ: int
    VOLUME: float


class MorseCodeDecode:
    r"""
    Class to encode and decode Morse code with optional sound playback.

    This class handles conversion between plain text and Morse code, optionally
    playing Morse code as beeps in a separate thread. Accents are automatically
    removed, unsupported characters are ignored, and the class provides
    static methods to generate beep waveforms and play Morse code messages.

    Class Attributes:
        CONFIG_BEEP (Final[DictMorse]): Configuration for beep sound including
            sampling frequency, beep frequency, and volume.
        TEXTMORSE (Final[dict[str, str]]): Mapping of characters to Morse code.
        MORSETEXT (Final[dict[str, str]]): Reverse mapping of Morse code to characters.
        MORSESOUND (Final[dict[str, float]]): Timing configuration for Morse symbols.

    Instance Attributes:
        value (str): The string to encode or decode.
    """

    CONFIG_BEEP: Final[DictMorse] = {
        "FS": 44100,
        "FREQ": 2000,
        "VOLUME": 0.7,
    }

    TEXTMORSE: Final[dict[str, str]] = {
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        "0": "-----",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        ".": ".-.-.-",
        ",": "--..--",
        "?": "..--..",
        "'": ".----.",
        "!": "-.-.--",
        "/": "-..-.",
        "(": "-.--.",
        ")": "-.--.-",
        "&": ".-...",
        ":": "---...",
        ";": "-.-.-.",
        "=": "-...-",
        "+": ".-.-.",
        "-": "-....-",
        "_": "..--.-",
        '"': ".-..-.",
        "$": "...-..-",
        "@": ".--.-.",
        " ": "/",
    }

    MORSETEXT: Final[dict[str, str]] = {v: k for k, v in TEXTMORSE.items()}

    MORSESOUND: Final[dict[str, float]] = {
        ".": 0.2,
        "-": 0.3,
        "TIME_SYMBOLS": 0.1,
        " ": 0.2,
        "/": 0.5,
    }

    def __init__(self, value: str) -> None:
        """
        Initialize a MorseCodeDecode instance.

        Args:
            value (str): Text or Morse code to encode/decode.
        """

        self.value: str = value

    @staticmethod
    def _beep_create(
        freq: int = 1000, duration: float = 0.2, volume: float = 0.7
    ) -> np.ndarray:
        r"""
        Generate a sine wave for a beep sound.

        Args:
            freq (int, optional): Frequency of the beep in Hz. Defaults to 1000.
            duration (float, optional): Duration of the beep in seconds. Defaults to 0.2.
            volume (float, optional): Amplitude of the beep (0.0 to 1.0). Defaults to 0.7.

        Returns:
            np.ndarray: Array representing the sine wave of the beep, dtype float64.
        """

        vector_t: np.ndarray[tuple[Any, ...], np.dtype[np.loat64]] = np.linspace(
            0,
            duration,
            int(MorseCodeDecode.CONFIG_BEEP["FS"] * duration),
            endpoint=False,
        )
        wave: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = volume * np.sin(
            2 * np.pi * freq * vector_t
        )

        return wave

    @staticmethod
    def _beep_play(message: str, speed: float = 1.0) -> None:
        r"""
        Play a Morse code message as audible beeps.

        Each symbol in the message is converted into a corresponding beep or pause
        according to the timing rules defined in MORSESOUND. The playback speed can
        be adjusted.

        Args:
            message (str): Morse code message (e.g., ".- / ...").
            speed (float, optional): Playback speed multiplier (higher is faster). Defaults to 1.0.

        Notes:
            - Symbols expected in `message` include '.', '-', ' ', and '/'.
            - Uses the beep frequency and volume constants defined in CONFIG_BEEP.
            - Concatenates all beep waveforms and pauses, then plays them via sounddevice.
        """

        wave_total: np.ndarray[tuple[Any, ...], np.dtype[np.float64 | np.float32]] = (
            np.array([], dtype=np.float32)
        )

        for symbol in message:
            if symbol in ".-":
                wave_total = np.concatenate(
                    [
                        wave_total,
                        MorseCodeDecode._beep_create(
                            freq=MorseCodeDecode.CONFIG_BEEP["FREQ"],
                            duration=MorseCodeDecode.MORSESOUND[symbol] / speed,
                            volume=MorseCodeDecode.CONFIG_BEEP["VOLUME"],
                        ),
                        np.zeros(
                            int(
                                MorseCodeDecode.CONFIG_BEEP["FS"]
                                * MorseCodeDecode.MORSESOUND["TIME_SYMBOLS"]
                                / speed
                            )
                        ),
                    ]
                )
            else:
                wave_total = np.concatenate(
                    [
                        wave_total,
                        np.zeros(
                            int(
                                MorseCodeDecode.CONFIG_BEEP["FS"]
                                * MorseCodeDecode.MORSESOUND[symbol]
                                / speed
                            )
                        ),
                    ]
                )

        sd.play(wave_total, MorseCodeDecode.CONFIG_BEEP["FS"])
        sd.wait()

    def tomorse(self, sound: bool = False) -> str | None:
        r"""
        Convert the stored text (`value`) to Morse code.

        This method removes accents and converts the text to uppercase before
        encoding. Unsupported characters are ignored. Optionally, the resulting
        Morse code can be played as audible beeps in a separate thread.

        Args:
            sound (bool, optional): If True, plays the Morse code as sound. Defaults to False.

        Returns:
            str | None: Encoded Morse code string if valid characters exist, else None.

        Notes:
            - Uses the TEXTMORSE dictionary for character-to-Morse conversion.
            - Plays the sound in a new thread if `sound=True` to avoid blocking.
        """

        text_prepared: str = strip_accents(self.value).upper()
        result: str = " ".join(
            [MorseCodeDecode.TEXTMORSE.get(tx, "") for tx in text_prepared]
        )
        if result:
            if sound:
                threading.Thread(target=self._beep_play, args=(result,)).start()
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
        result: str = "".join(
            [MorseCodeDecode.MORSETEXT.get(mor, "") for mor in cod_morse]
        )
        return result or None


if __name__ == "__main__":

    cod_morse = MorseCodeDecode("ola t").tomorse(sound=True)
    cod_morse1 = MorseCodeDecode("ó").totext()
    print(f"{cod_morse}")
    print(f"{cod_morse1}")


# cls ; python -W ignore -m src.backend.modules.codec.morse_code_decode
