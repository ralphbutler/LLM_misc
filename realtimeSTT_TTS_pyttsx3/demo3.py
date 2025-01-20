
# voice activation

from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':  # MUST HAVE THIS LINE
    with AudioToTextRecorder() as recorder:
        print("Transcription: ", recorder.text())
