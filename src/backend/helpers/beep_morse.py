import numpy as np
import sounddevice as sd
from typing import Final
from numpy import ndarray, dtype, float64
from typing import Any


def beep(freq: int = 600, duration: float = 0.2, volume: float = 0.7) -> None:
    r"""
    Generate and play a sine wave beep.

    This function produces a sine wave at a given frequency, duration, and volume,
    using a sampling rate of 44,100 Hz, and plays it using the `sounddevice` module.
    The function blocks until playback is finished.

    Args:
        freq (int, optional): Frequency of the beep in Hz. Defaults to 600.
        duration (float, optional): Duration of the beep in seconds. Defaults to 0.2.
        volume (float, optional): Amplitude of the beep, between 0 and 1. Defaults to 0.7.

    Returns:
        return (None):
    """

    FS: Final[int] = 44100  # sampling rate

    vector_t: ndarray[tuple[Any, ...], dtype[float64]] = np.linspace(
        0, duration, int(FS * duration), endpoint=False
    )
    wave: ndarray[tuple[Any, ...], dtype[float64]] = volume * np.sin(
        2 * np.pi * freq * vector_t
    )
    sd.play(wave, samplerate=FS)
    sd.wait()

    return None
