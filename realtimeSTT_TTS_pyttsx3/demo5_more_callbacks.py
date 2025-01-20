
from RealtimeSTT import AudioToTextRecorder

def start_callback():
    print("Recording started!")

def stop_callback():
    print("Recording stopped!")

def process_text(text):
    print(text)

if __name__ == '__main__':  # MUST HAVE THIS LINE
    recorder = AudioToTextRecorder(on_recording_start=start_callback,
                                   on_recording_stop=stop_callback)

    while True:
        recorder.text(process_text)
