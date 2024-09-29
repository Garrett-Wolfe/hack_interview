import PySimpleGUI as sg
from loguru import logger


from src import llm
from src.constants import APPLICATION_WIDTH, OFF_IMAGE, ON_IMAGE
from src.transcriber import T_Type, TranscriberFactory


def get_text_area(text: str, size: tuple) -> sg.Text:
    """
    Create a text area widget with the given text and size.

    Parameters:
        text (str): The initial text to display in the text area.
        size (tuple): The size of the text area widget.

    Returns:
        sg.Text: The created text area widget.
    """
    return sg.Text(
        text,
        size=size,
        background_color=sg.theme_background_color(),
        text_color="white",
    )


class BtnInfo:
    def __init__(self, state=False):
        self.state = state


# All the stuff inside your window:
sg.theme("DarkAmber")  # Add a touch of color
record_status_button = sg.Button(
    image_data=OFF_IMAGE,
    k="R",
    border_width=0,
    button_color=(sg.theme_background_color(), sg.theme_background_color()),
    disabled_button_color=(sg.theme_background_color(), sg.theme_background_color()),
    metadata=BtnInfo(),
)
analyzed_text_label = get_text_area("", size=(APPLICATION_WIDTH, 2))
quick_chat_gpt_answer = get_text_area("", size=(APPLICATION_WIDTH, 5))
full_chat_gpt_answer = get_text_area("", size=(APPLICATION_WIDTH, 12))


layout = [
    [sg.Text("Press R to start recording", size=(int(APPLICATION_WIDTH * 0.8), 2)), record_status_button],
    [sg.Text("Press A to analyze the recording")],
    [analyzed_text_label],
    [sg.Text("Short answer:")],
    [quick_chat_gpt_answer],
    [sg.Text("Full answer:")],
    [full_chat_gpt_answer],
    [sg.Button("Cancel")],
]
WINDOW = sg.Window("Keyboard Test", layout, return_keyboard_events=True, use_default_focus=False)

logger.debug = print

scriber = TranscriberFactory.get_transcriber(T_Type.DEEPGRAM_STREAM)

while True:
    event, values = WINDOW.read()
    logger.debug(f"Event: {event}")
    if event in ["Cancel", sg.WIN_CLOSED]:
        logger.debug("Closing...")
        break

    event = event.split(':')[0].upper() if isinstance(event, str) else event

    if event == "R":
        button_on = not record_status_button.metadata.state
        record_status_button.metadata.state = button_on
        logger.debug(f"Recording toggled {button_on}...")
        if button_on:
            WINDOW.perform_long_operation(scriber.background_recording, "A: FINISHED RECORDING")
        else:
            scriber.stop_recording()
        record_status_button.update(image_data=ON_IMAGE if button_on else OFF_IMAGE)

    elif event == "A":
        logger.debug("Analyzing audio...")
        analyzed_text_label.update("Start analyzing...")
        WINDOW.perform_long_operation(scriber.transcribe, "-TRANSCRIPTION COMPLETE-")

    elif event == "-TRANSCRIPTION COMPLETE-":
        audio_transcript = values["-TRANSCRIPTION COMPLETE-"]
        analyzed_text_label.update(audio_transcript)

        # Generate quick answer:
        quick_chat_gpt_answer.update("Chatgpt is working...")
        WINDOW.perform_long_operation(
            lambda: llm.generate_answer(audio_transcript, short_answer=True, temperature=0),
            "-CHAT_GPT SHORT ANSWER-",
        )

        # Generate full answer:
        full_chat_gpt_answer.update("Chatgpt is working...")
        WINDOW.perform_long_operation(
            lambda: llm.generate_answer(audio_transcript, short_answer=False, temperature=0.7),
            "-CHAT_GPT LONG ANSWER-",
        )
    elif event == "-CHAT_GPT SHORT ANSWER-":
        quick_chat_gpt_answer.update(values["-CHAT_GPT SHORT ANSWER-"])
    elif event == "-CHAT_GPT LONG ANSWER-":
        full_chat_gpt_answer.update(values["-CHAT_GPT LONG ANSWER-"])
