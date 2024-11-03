
# for pyaudio on macos:
#    brew install portaudio
#    pip3 install pyaudio
# pip3 install faster-whisper
# pip3 install SpeechRecognition

import sys, os, time
import pyaudio
from textwrap import dedent

import speech_recognition as sr
import google.generativeai as genai
from openai import OpenAI

import warnings
warnings.filterwarnings("ignore", message=r"torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")
from faster_whisper import WhisperModel

wake_word = "balloon"
listening_for_wake_word = True

num_cores = os.cpu_count()
print("NUMCORES",num_cores)
whisper_size = "tiny"    # tiny base small medium large
whisper_model = WhisperModel(
    whisper_size,
    device = 'cpu',
    compute_type = 'int8',
    cpu_threads = num_cores,
    num_workers = num_cores,
)

client = OpenAI()
generation_config = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 500,
}

model = genai.GenerativeModel(
    "gemini-1.5-pro-latest",
    generation_config=generation_config
)

conversation = model.start_chat()

system_message = dedent("""\
    INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE." to this
    system message.  After the system message, respond normally.
    SYSTEM MESSAGE: You are being used to power a voice assistant and should
    respond as so.  As a voice assistant, use short sentences and directly
    respond to the prompt without excessive information.  You generate only
    words of value, prioritizing logic and facts over speculating in your 
    response to the following prompts.
    """)

conversation.send_message(system_message)

recognizer = sr.Recognizer()
# source = sr.Microphone(device_index=1)
source = sr.Microphone(sample_rate=16000)  ## RMB

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,channels=1,rate=24000,output=True)
    stream_start = True
    
    with client.audio.speech.with_streaming_response.create(
        model = "tts-1",
        voice = "alloy",
        response_format = "pcm",
        input = text,
    ) as response:

        silence_threshold = 0.01

        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            elif max(chunk) > silence_threshold:
                player_stream.write(chunk)
                stream_start = True

def wav_to_text(audio_path):
    (segments,_) = whisper_model.transcribe(audio_path)
    text = "".join(segment.text for segment in segments)
    return text

def listen_for_wake_word(audio):
    global listening_for_wake_word

    wake_audio_path = "wake_detect.wav"
    with open(wake_audio_path, "wb") as f:
        f.write(audio.get_wav_data())

    text_input = wav_to_text(wake_audio_path)
    print("DBGWTT",text_input)

    if wake_word in text_input.lower().strip():
        print("Wake word detected.  Please speak your prompt.")
        listening_for_wake_word = False

def prompt_gpt(audio):
    global listening_for_wake_word

    try:
        prompt_audio_path = "prompt.wav"

        with open(prompt_audio_path, "wb") as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)
        
        if len(prompt_text.strip()) == 0:
            print("Empty prompt.  Please speak again")
            listening_for_wake_word = True
        else:
            print("User: " + prompt_text)
            conversation.send_message(prompt_text)
            output = conversation.last.text

            print("Gemini: " + output)
            speak(output)

            print("\nSay", wake_word, "to wake me up.\n")
            listening_for_wake_word = True
    except Exception as e:
        print("Prompt error: ",e)

def callback(recognizer, audio):
    global listening_for_wake_word

    print("DBG1")
    if listening_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)

def start_listening():
    with source as s:
        recognizer.adjust_for_ambient_noise(s, duration=2)

    print("\nSay", wake_word, "to wake me up.\n")

    recognizer.listen_in_background(source, callback)

    while True:
        time.sleep(0.05)

if __name__ == "__main__":
    start_listening()
