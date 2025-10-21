import threading
import numpy as np
import sounddevice as sd
from numpy.typing import NDArray
from typing import Final, TypedDict, Callable
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

    _cache_morse: dict[tuple[str, float], NDArray[np.float32]] = {}

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
    ) -> NDArray[np.float32]:
        r"""
        Generate a sine wave representing a beep sound.

        This method creates a NumPy array containing a sine wave at the specified
        frequency, duration, and volume. The array can be used for audio playback
        or signal processing purposes.

        Args:
            freq (int, optional): Frequency of the beep in Hertz. Defaults to 1000.
            duration (float, optional): Duration of the beep in seconds. Defaults to 0.2.
            volume (float, optional): Amplitude of the beep, from 0.0 (silent) to 1.0 (full volume). Defaults to 0.7.

        Returns:
            NDArray[np.float32]: Array containing the sine wave of the beep, dtype float32.
        """

        vector_t: NDArray[np.float32] = np.linspace(
            0,
            duration,
            int(MorseCodeDecode.CONFIG_BEEP["FS"] * duration),
            endpoint=False,
            dtype=np.float32,
        )
        sine_wave: NDArray[np.float32] = np.sin(
            2 * np.pi * freq * vector_t, dtype=np.float32
        )
        return volume * sine_wave

    @staticmethod
    def _dot_or_dash(symbol: str, speed: float) -> NDArray[np.float32]:
        r"""
        Generate the beep waveform for a Morse code dot or dash, including trailing silence.

        This method creates a sine wave for the given symbol (dot or dash) and appends
        a silence interval after the beep. The result is cached to improve performance
        for repeated symbols with the same speed.

        Args:
            symbol (str): Morse code symbol, either '.' or '-'.
            speed (float): Playback speed multiplier; higher values make playback faster.

        Returns:
            NDArray[np.float32]: Array representing the audio waveform of the symbol
            followed by its silence.
        """

        key: tuple[str, float] = (symbol, speed)

        if key in MorseCodeDecode._cache_morse:
            return MorseCodeDecode._cache_morse[key]

        beep_symbol: NDArray[np.float32] = MorseCodeDecode._beep_create(
            freq=MorseCodeDecode.CONFIG_BEEP["FREQ"],
            duration=MorseCodeDecode.MORSESOUND[symbol] / speed,
            volume=MorseCodeDecode.CONFIG_BEEP["VOLUME"],
        )

        time_symbol: NDArray[np.float32] = np.zeros(
            int(
                MorseCodeDecode.CONFIG_BEEP["FS"]
                * MorseCodeDecode.MORSESOUND["TIME_SYMBOLS"]
                / speed
            ),
            dtype=np.float32,
        )

        wave: NDArray[np.float32] = np.r_[beep_symbol, time_symbol]

        MorseCodeDecode._cache_morse[key] = wave

        return wave

    @staticmethod
    def _space_or_slash(symbol: str, speed: float) -> NDArray[np.float32]:
        r"""
        Generate the silence waveform for a Morse code space or slash.

        This method creates an array of zeros corresponding to the duration of a space
        (' ') or slash ('/') symbol in Morse code. The result is cached for repeated use
        with the same speed.

        Args:
            symbol (str): Morse code symbol, either ' ' (space) or '/' (slash).
            speed (float): Playback speed multiplier; higher values make playback faster.

        Returns:
            NDArray[np.float32]: Array of zeros representing the silence for the symbol.
        """

        key: tuple[str, float] = (symbol, speed)

        if key in MorseCodeDecode._cache_morse:
            return MorseCodeDecode._cache_morse[key]

        silence_symbol: NDArray[np.float32] = np.zeros(
            int(
                MorseCodeDecode.CONFIG_BEEP["FS"]
                * MorseCodeDecode.MORSESOUND[symbol]
                / speed
            ),
            dtype=np.float32,
        )

        MorseCodeDecode._cache_morse[key] = silence_symbol

        return silence_symbol

    BEEP_PLAY: Final[
        dict[
            str,
            Callable[[str, float], NDArray[np.float32]],
        ]
    ] = {
        ".": _dot_or_dash,
        "-": _dot_or_dash,
        " ": _space_or_slash,
        "/": _space_or_slash,
    }

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

        chunks: list[NDArray[np.float32]] = []

        for symbol in message:
            chunks.append(MorseCodeDecode.BEEP_PLAY[symbol](symbol, speed))

        wave_total: NDArray[np.float32] = np.concatenate(chunks)
        sd.play(wave_total, MorseCodeDecode.CONFIG_BEEP["FS"])
        sd.wait()

    def tomorse(self, sound: bool = False, speed: float = 1.0) -> str | None:
        r"""
        Convert the stored text (`value`) to Morse code.

        Accents are removed and text is converted to uppercase before encoding.
        Unsupported characters are ignored. Optionally, the Morse code can be played
        as audible beeps in a separate thread.

        Args:
            sound (bool, optional): If True, plays the Morse code as sound. Defaults to False.
            speed (float, optional): Playback speed multiplier. Defaults to 1.0.

        Returns:
            (str | None): Encoded Morse code string if valid characters exist, else None.

        Notes:
            Uses the TEXTMORSE dictionary for character-to-Morse conversion.
            Playback runs in a new thread if `sound=True` to avoid blocking.
        """

        text_prepared: str = strip_accents(self.value).upper()
        result: str = " ".join(
            [MorseCodeDecode.TEXTMORSE.get(tx, "") for tx in text_prepared]
        )
        if result:
            if sound:
                threading.Thread(
                    target=MorseCodeDecode._beep_play, args=(result, speed)
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
            (str | None): Decoded text if valid Morse code is present, else None.
        """

        cod_morse: list[str] = self.value.split()
        result: str = "".join(
            [MorseCodeDecode.MORSETEXT.get(mor, "") for mor in cod_morse]
        )
        return result or None


if __name__ == "__main__":

    cod_morse = MorseCodeDecode("ola t").tomorse(sound=True, speed=1.0)
    cod_morse1 = MorseCodeDecode("---").totext()
    print(f"{cod_morse}")
    print(f"{cod_morse1}")


# cls ; python -W ignore -m src.backend.modules.codec.morse_code_decode
