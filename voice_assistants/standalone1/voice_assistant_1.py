
# there is a little bit of extra/UN-used code in here which demos how to do a few
#   things which may prove useful in the future

import sys, os, time, io, json
import speech_recognition as sr

from openai import OpenAI

from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

wake_phrase = ["hello", "there"]

recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.8 # secs of non-speaking before phrase is assumed complete
recognizer.energy_threshold = 1000  # default 1000
# definitely do this, dynamic energy compensation lowers the energy threshold
# dramatically to a point where the SpeechRecognizer never stops recording
recognizer.dynamic_energy_threshold = False

def audio_from_recognizer_to_wav(audio):
    in_mem_fp = io.BytesIO()
    in_mem_fp.write( audio.get_wav_data() )
    in_mem_fp.seek(0)
    audio_wav = AudioSegment.from_file(in_mem_fp, format="wav")
    # if I wanted to convert to mp3
    # audio_mp3 = audio_wav.set_frame_rate(audio_wav.frame_rate) \
                         # .set_channels(audio_wav.channels) \
                         # .set_sample_width(audio_wav.sample_width)
    return audio_wav

def text_to_wav(text):
    tts = gTTS(text=text, lang='en')
    in_mem_fp = io.BytesIO()
    tts.write_to_fp(in_mem_fp)
    in_mem_fp.seek(0)
    audio_mp3 = AudioSegment.from_file(in_mem_fp, format="mp3")  # fails doing wav
    audio_wav = audio_mp3.set_frame_rate(audio_mp3.frame_rate) \
                         .set_channels(audio_mp3.channels)  \
                         .set_sample_width(audio_mp3.sample_width)
    return audio_wav

def audio_to_text(audio):
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
        text = "OOPS"
    except sr.RequestError as e:
        print("Could not request results from Google Web Speech API; {0}".format(e))
        text = "OOPS"
    return text

# mic_names = sr.Microphone.list_microphone_names()
# print("MIC NAMES",mic_names)
mic_idx = 0

while True:
    with sr.Microphone(device_index=mic_idx) as source:
        print("\nassistant is ready")
        recognizer.adjust_for_ambient_noise(source)  # , duration=0.5)

        # phrase_limit=2 is for WAKE ONLY
        audio = recognizer.listen(source, timeout=11, phrase_time_limit=2)
        text_wake = audio_to_text( audio )
        print(f"You said: {text_wake}")
        cnt = sum( [ 1 for word in wake_phrase if word in text_wake.lower() ] )
        if cnt < len(wake_phrase):
            continue

        offer_to_help_text = "How can I help you?"
        print(offer_to_help_text)
        play( text_to_wav(offer_to_help_text) )

        recognizer.adjust_for_ambient_noise(source) # , duration=0.5)
        audio = recognizer.listen(source, timeout=11, phrase_time_limit=11)
        text_request = audio_to_text( audio )
        print(f"You said: {text_request}")

        play( text_to_wav("I will help with " + text_request) )
        # play( audio_from_recognizer_to_wav(audio) )
        time.sleep(0.5)

        ### use an LLM here to determine how to satisfy the request

        print(f"servicing request: {text_request}")
        client = OpenAI()
        sys_msg = ("""
            You are an expert at figuring out how to perform actions in the Unix terminal.
            Always return your answer in JSON format, like this:
                {
                    "action_to_perform": <action>,
                }
        """)

        response = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            response_format={"type": "json_object"},   # this matters
            messages = [
                { "role": "system", "content": sys_msg},
                { "role": "user",   "content": text_request},
            ]
        )
        jresponse = json.loads(response.choices[0].message.content)
        action = jresponse["action_to_perform"]
        print(f"ACTION: {action}")
