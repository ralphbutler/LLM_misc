
from RealtimeSTT import AudioToTextRecorder

def start_callback():
    print("Recording started!")

def stop_callback():
    print("Recording stopped!")

def process_text(text):
    print(text)

if __name__ == '__main__':  # MUST HAVE THIS LINE

    wakeword = "jarvis"
    print("WAKEWORD IS", wakeword)

    recorder = AudioToTextRecorder(
        # on_recording_start=start_callback,
        # on_recording_stop=stop_callback,
        wakeword_backend="oww",
        wake_words=wakeword,
    )

    while True:
        recorder.text(process_text)
