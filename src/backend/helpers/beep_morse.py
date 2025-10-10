import numpy as np
import sounddevice as sd
from typing import Final
from numpy import ndarray, dtype, float64
from typing import Any
from src.backend.constants import MORSESOUND

FS: Final[int] = 44100
FREQ: Final[int] = 2000
VOLUME: Final[float] = 0.7


def beep_config(
    freq: int = 1000, duration: float = 0.2, volume: float = 0.7
) -> ndarray:
    r"""
    Generate a sine wave for a beep sound.

    Args:
        freq (int, optional): Frequency of the beep in Hz. Defaults to 1000.
        duration (float, optional): Duration of the beep in seconds. Defaults to 0.2.
        volume (float, optional): Amplitude of the beep (0.0 to 1.0). Defaults to 0.7.

    Returns:
        ndarray: Array representing the sine wave of the beep.
    """

    vector_t: ndarray[tuple[Any, ...], dtype[float64]] = np.linspace(
        0, duration, int(FS * duration), endpoint=False
    )
    wave: ndarray[tuple[Any, ...], dtype[float64]] = volume * np.sin(
        2 * np.pi * freq * vector_t
    )

    return wave


def beep_play(message: str, speed: float = 1.0) -> None:
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
        - Uses the beep frequency and volume constants defined as FREQ and VOLUME.
    """

    wave_total = np.array([], dtype=np.float32)

    for symbol in message:
        if symbol in ".-":
            wave_total = np.concatenate(
                [
                    wave_total,
                    beep_config(
                        freq=FREQ, duration=MORSESOUND[symbol] / speed, volume=VOLUME
                    ),
                    np.zeros(int(FS * MORSESOUND["TIME_SYMBOLS"] / speed)),
                ]
            )
        else:
            wave_total = np.concatenate(
                [wave_total, np.zeros(int(FS * MORSESOUND[symbol] / speed))]
            )

    sd.play(wave_total, FS)
    sd.wait()
