"""Transcriber Objects."""
import os
import threading
import time
import numpy as np
from openai import OpenAI
from loguru import logger
from abc import ABC, abstractmethod
from enum import Enum
from deepgram import DeepgramClient, PrerecordedOptions, LiveTranscriptionEvents, LiveOptions


from src import audio
from src.constants import OUTPUT_FILE_NAME


class T_Type(str, Enum):
    WHISPER = "WHISPER"
    DEEPGRAM = "DEEPGRAM"
    DEEPGRAM_STREAM = "DEEPGRAM_STREAM"

class Transcriber(ABC):
    """
    Transcriber instances implement different strategies for recording audio and transcribing it.\n
    They all have three methods:
    - background_recording = a methods that handles recording audio
    - stop_recording = sets OS level flag to halt recording
    - transcribe = converts the audio into text
    """

    def __init__(self, file_path: str = OUTPUT_FILE_NAME) -> None:
        super().__init__()
        self.file_path: str = file_path
        self._stop_recording_event: threading.Event = threading.Event()

    def stop_recording(self):
        """
        Sets the event that will halt the background_recording method and force it to return
        Returns:
            None
        """
        self._stop_recording_event.set()
    
    def background_recording(self) -> None:
        """
        Method indended to be in it's own thread. It will loop indefinetly recording audio.
        Only halts when the thread_event "stop_recording_event" is set. This method will clear 
        "stop_recording_event" just before returning. 

        Returns:
            None

        Example:
            ```python
            background_thread = threading.Thread(target=background_recording)
            ```
        """
        audio_data = None
        while not self._stop_recording_event.is_set():
            audio_sample = audio.record_batch()
            audio_data = np.vstack((audio_data, audio_sample)) if audio_data is not None else audio_sample
        audio.save_audio_file(audio_data)
        self._stop_recording_event.clear()

    @abstractmethod
    def transcribe(self) -> str:
        """
        Transcribes an audio file into text.

        Returns:
            str: The transcribed text.

        Raises:
            Exception: If the audio file fails to transcribe.
        """
        pass

class WhisperTranscribe(Transcriber):
    def transcribe(self) -> str:
        client = OpenAI()
        with open(self.file_path, "rb") as audio_file:
            try:
                transcription = client.audio.transcriptions.create(model="whisper-1",file=audio_file)
            except Exception as error:
                logger.error(f"Can't transcribe audio: {error}")
                raise error
        return transcription.text

class DeepgramTranscribe(Transcriber):
    def transcribe(self) -> str:
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

        with open(self.file_path, 'rb') as buffer_data:
            payload = { 'buffer': buffer_data }
            options = PrerecordedOptions(
                smart_format=True, model="nova-2", language="en-US"
            )

            response = deepgram.listen.rest.v('1').transcribe_file(payload, options)
            return response.results.channels[0].alternatives[0].transcript

class DeepgramStreamTranscribe(Transcriber):
    def __init__(self, file_path: str = OUTPUT_FILE_NAME) -> None:
        super().__init__(file_path)
        self.__transcript_text = ""
        self.__transcript_ready_event = threading.Event()

    def transcribe(self) -> str:
        self.__transcript_ready_event.wait()
        return self.__transcript_text

    def background_recording(self) -> None:
        self.__transcript_ready_event.clear()
        last_message_time = time.time()
        message_lock = threading.Lock()
        early_stop_event = threading.Event()
        transcript_text = ""


        deepgram = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'))
        dg_connection = deepgram.listen.live.v('1')

        def on_message(self, result, **kwargs):
            nonlocal last_message_time
            nonlocal transcript_text
            with message_lock:
                sentence = result.channel.alternatives[0].transcript
                if len(sentence) > 0:
                    last_message_time = time.time()
                    transcript_text += sentence + ' '

        def on_error(self, error, **kwargs):
            early_stop_event.set()
            logger.error(f"Handled Error: {error}")

        def on_unhandled(self, unhandled, **kwargs):
            early_stop_event.set()
            logger.error(f"Unhandled Websocket Message: {unhandled}")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Unhandled, on_unhandled)

        dg_connection.start(LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding=audio.ENCODING,
            channels=audio.MIC.channels,
            sample_rate=audio.SAMPLE_RATE,
        ))

        while not self._stop_recording_event.is_set() and not early_stop_event.is_set():
            audio_bytes = audio.record_batch(record_sec=0.25, encode=True)
            dg_connection.send(audio_bytes)

        dg_connection.finish()

        if not early_stop_event.is_set():
            def time_since_last_message():
                with message_lock:
                    return time.time() - last_message_time

            while time_since_last_message() < 1:
                time.sleep(1)

        message_lock.acquire()
        self.__transcript_text = transcript_text
        self.__transcript_ready_event.set()
        self._stop_recording_event.clear()


class TranscriberFactory:
    @staticmethod
    def get_transcriber(type: T_Type) -> Transcriber:
        if type == T_Type.WHISPER:
            return WhisperTranscribe()
        elif type == T_Type.DEEPGRAM:
            return DeepgramTranscribe()
        elif type == T_Type.DEEPGRAM_STREAM:
            return DeepgramStreamTranscribe()
        else:
            raise ValueError(f"Unknown transcriber type: {type}")
