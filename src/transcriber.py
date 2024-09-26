import threading
import numpy as np
from openai import OpenAI
from loguru import logger
from abc import ABC, abstractmethod
from enum import Enum
from deepgram import DeepgramClient, PrerecordedOptions, LiveTranscriptionEvents, LiveOptions


from src import audio
from src.constants import OUTPUT_FILE_NAME
DEEPGRAM_API_KEY = '91bbe5940a7fb14c7d2bbd0e986cb01eca04cee9'


class T_Type(str, Enum):
    WHISPER = "WHISPER"
    DEEPGRAM = "DEEPGRAM"
    DEEPGRAM_STREAM = "DEEPGRAM_STREAM"

class Transcriber(ABC):

    def __init__(self, file_path: str = OUTPUT_FILE_NAME) -> None:
        super().__init__()
        self.file_path: str = file_path
        self.stop_recording_event: threading.Event = threading.Event()

    def stop_recording(self):
        self.stop_recording_event.set()
    
    def background_recording(self) -> None:
        audio_data = None
        while not self.stop_recording_event.is_set():
            audio_sample = audio.record_batch()
            audio_data = np.vstack((audio_data, audio_sample)) if audio_data is not None else audio_sample
        audio.save_audio_file(audio_data)

    @abstractmethod
    def transcribe(self) -> str:
        """
        Transcribes an audio file into text.

        Args:
            path_to_file (str, optional): The path to the audio file to be transcribed.

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
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        with open(self.file_path, 'rb') as buffer_data:
            payload = { 'buffer': buffer_data }
            options = PrerecordedOptions(
                smart_format=True, model="nova-2", language="en-US"
            )

            response = deepgram.listen.rest.v('1').transcribe_file(payload, options)
            return response.results.channels[0].alternatives[0].transcript

class DeepgramStreamTranscribe(Transcriber):
    def transcribe(self) -> str:
        print(self.transcript_text)
        return self.transcript_text

    def background_recording(self) -> None:
        self.transcript_text = ""
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        dg_connection = deepgram.listen.live.v('1')

        def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            self.transcript_text += sentence + '\n'

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        options = LiveOptions(smart_format=True, model="nova-2", language="en-US")
        dg_connection.start(options)  # Create a websocket connection to Deepgram

        while not self.stop_recording_event.is_set():
            audio_sample = audio.record_batch(record_sec=0.25)
            audio_bytes = (audio_sample * 32767).astype('int16').tobytes()  # 16-bit signed PCM (linear16)
            dg_connection.send(audio_bytes)


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
