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
ENCODING = "linear16"


SPEAKER_ID = str(sc.default_speaker().name)
MIC = sc.get_microphone( id=SPEAKER_ID, include_loopback=False)


logger.debug = print
logger.debug(f"Mic being used: {SPEAKER_ID}, channels {MIC.channels}")


def record_batch(record_sec: float = RECORD_SEC, encode: bool = False) -> np.ndarray:
    """
    Records an audio batch for a specified duration. 
    The returned data can be raw or encoded to 16-bit signed PCM (linear16)

    Args:
        record_sec (float): The duration of the recording in seconds. Defaults to the value of RECORD_SEC.
        encode (bool): Indicates if the returned data should be encoded. Defaults to False.

    Returns:
        np.ndarray: The recorded audio sample.

    Example:
        ```python
        audio_sample = record_batch(5)
        print(audio_sample)
        ```
    """
    logger.debug(f"Recording for {record_sec:.3f} second(s)...")
    num_samples = int(SAMPLES_PER_SEC * record_sec)
    total_frames = num_samples * NUMFRAMES
    full_audio_sample = np.zeros((total_frames, MIC.channels))
    frame_index = 0

    with MIC.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE) as recorder:
        for _ in range(num_samples):
            full_audio_sample[frame_index:frame_index + NUMFRAMES] = recorder.record(numframes=NUMFRAMES)
            frame_index += NUMFRAMES

    if encode:  # 16-bit signed PCM (linear16)
        full_audio_sample = (full_audio_sample * 32767).astype('int16').tobytes()

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
