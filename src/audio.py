"""Audio utilities."""
import numpy as np
import soundcard as sc
import soundfile as sf
from loguru import logger

from src.constants import OUTPUT_FILE_NAME

SAMPLE_RATE = 48000             # [Hz]. sampling rate.
RECORD_SEC = 1                  # [sec]. duration recording audio.
BLOCKSIZE = 256                 # good value suggested by documentation
NUMFRAMES = BLOCKSIZE // 2      # numframes should be less than blocksize per the docs
SAMPLES_PER_SEC = SAMPLE_RATE // NUMFRAMES

SPEAKER_ID = str(sc.default_speaker().name)
MIC = sc.get_microphone( id=SPEAKER_ID, include_loopback=False) 


logger.debug = print
logger.debug(f"Mic being used: {SPEAKER_ID}")


def record_batch(record_sec: int = RECORD_SEC) -> np.ndarray:
    """
    Records an audio batch for a specified duration.

    Args:
        record_sec (int): The duration of the recording in seconds. Defaults to the value of RECORD_SEC.

    Returns:
        np.ndarray: The recorded audio sample.

    Example:
        ```python
        audio_sample = record_batch(5)
        print(audio_sample)
        ```
    """
    logger.debug(f"Recording for {record_sec} second(s)...")

    total_frames = SAMPLES_PER_SEC * record_sec * NUMFRAMES
    full_audio_sample = np.zeros((total_frames, 2))
    frame_index = 0

    with MIC.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE) as recorder:
        for _ in range(SAMPLES_PER_SEC * record_sec):
            full_audio_sample[frame_index:frame_index + NUMFRAMES] = recorder.record(numframes=NUMFRAMES)
            frame_index += NUMFRAMES

    return full_audio_sample


def save_audio_file(audio_data: np.ndarray, output_file_name: str = OUTPUT_FILE_NAME) -> None:
    """
    Saves an audio data array to a file.

    Args:
        audio_data (np.ndarray): The audio data to be saved.
        output_file_name (str): The name of the output file. Defaults to the value of OUTPUT_FILE_NAME.

    Returns:
        None

    Example:
        ```python
        audio_data = np.array([0.1, 0.2, 0.3])
        save_audio_file(audio_data, "output.wav")
        ```
    """
    logger.debug(f"Saving audio file to {output_file_name}...")
    sf.write(file=output_file_name, data=audio_data, samplerate=SAMPLE_RATE)
