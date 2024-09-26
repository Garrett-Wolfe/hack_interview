from openai import OpenAI
from loguru import logger

from src.constants import INTERVIEW_POSTION, OUTPUT_FILE_NAME

SYSTEM_PROMPT = f"""You are interviewing for a {INTERVIEW_POSTION} position.
You will receive an audio transcription of the question. It may not be complete. You need to understand the question and write an answer to it.\n
"""
SHORTER_INSTRACT = "Concisely respond, limiting your answer to 70 words."
LONGER_INSTRACT = (
    "Before answering, take a deep breath and think one step at a time. Believe the answer in no more than 150 words."
)


def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Transcribes an audio file into text.

    Args:
        path_to_file (str, optional): The path to the audio file to be transcribed.

    Returns:
        str: The transcribed text.

    Raises:
        Exception: If the audio file fails to transcribe.
    """
    client = OpenAI()
    with open(path_to_file, "rb") as audio_file:
        try:
            transcription = client.audio.transcriptions.create(model="whisper-1",file=audio_file)
        except Exception as error:
            logger.error(f"Can't transcribe audio: {error}")
            raise error
    return transcription.text


def generate_answer(transcript: str, short_answer: bool = True, temperature: float = 0.7) -> str:
    """
    Generates an answer based on the given transcript using the OpenAI GPT-3.5-turbo model.

    Args:
        transcript (str): The transcript to generate an answer from.
        short_answer (bool): Whether to generate a short answer or not. Defaults to True.
        temperature (float): The temperature parameter for controlling the randomness of the generated answer.

    Returns:
        str: The generated answer.

    Example:
        ```python
        transcript = "Can you tell me about the weather?"
        answer = generate_answer(transcript, short_answer=False, temperature=0.8)
        print(answer)
        ```

    Raises:
        Exception: If the LLM fails to generate an answer.
    """
    system_prompt = SYSTEM_PROMPT + (SHORTER_INSTRACT if short_answer else LONGER_INSTRACT)

    client = OpenAI()
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ]
        )
    except Exception as error:
        logger.error(f"Can't generate answer: {error}")
        raise error
    return completion.choices[0].message.content
