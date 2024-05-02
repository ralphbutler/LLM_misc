
# RMB: python3 transcribe_demo.py --model=small

import sys, os, time
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from sys import platform


stime = 0
listening_for_wake_word = True
listening_for_cmd = False
wake_word = "balloon"

def main():
    model = "tiny.en"   # RMB small takes 4 secs, base 2 secs, tiny < 1 sec
    energy_threshold = 1000  # energy level for mic to detect
    record_timeout = 5 # 2.0     # how real time the recording is in seconds
    phrase_timeout = 3.0     # how much empty space between recordings before we 
                             #   consider it a new line in the transcription

    phrase_time = None # last time a recording was retrieved from queue

    # thread-safe queue for passing data from the threaded recording callback
    data_queue = Queue()

    # use SpeechRecognizer to record our audio because it has a nice feature where
    #   it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold

    # definitely do this, dynamic energy compensation lowers the energy threshold
    # dramatically to a point where the SpeechRecognizer never stops recording
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    audio_model = whisper.load_model(model)

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # grab raw bytes and push them into the thread-safe queue
        data = audio.get_raw_data()
        data_queue.put(data)
        print("DBG1")
        global stime
        stime = time.time()

    # create a background thread that will pass us raw audio bytes
    # we could do this manually but SpeechRecognizer provides a nice helper
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # cue the user that we're ready to go
    print("\nready to transcribe\n")

    while True:
        try:
            now = time.time()
            # pull raw recorded audio from the queue.
            if not data_queue.empty():
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                ## RMB the time is spent in this transcribe
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                print("TEXT",type(text),text)

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                global listening_for_wake_word, listening_for_cmd
                print("DBGTEST")
                if listening_for_wake_word and wake_word in text.lower():
                    listening_for_wake_word = False
                    listening_for_cmd = True
                    print("What can I do for you?")
                elif listening_for_cmd:
                    print("performing command:", text)
                    listening_for_cmd = False
                    listening_for_wake_word = True

                print("-"*50, time.time()-stime)
                print('', end='', flush=True)
            else:
                time.sleep(0.1)
        except KeyboardInterrupt:
            break

    print("\n\ntranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
