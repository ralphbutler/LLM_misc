
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':  # MUST HAVE THIS LINE
    recorder = AudioToTextRecorder()
    print("\nStarting recording")
    print("transcript will NOT print until you press the enter key")
    # contrast this with demo3
    recorder.start()
    input("\nPress Enter to stop recording and see transcript...\n")
    recorder.stop()
    print("Transcription: ", recorder.text())
    recorder.shutdown()
