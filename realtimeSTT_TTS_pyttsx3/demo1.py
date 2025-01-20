
from RealtimeSTT import AudioToTextRecorder

def process_text(text):
    print(text)

if __name__ == '__main__':  # MUST HAVE THIS LINE
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(process_text)

print("END **")

## ## a second example but which types into text box instead of print
## 
## from RealtimeSTT import AudioToTextRecorder
## import pyautogui
## 
## def process_text(text):
##     pyautogui.typewrite(text + " ")
## 
## if __name__ == '__main__':
##     print("Wait until it says 'speak now'")
##     recorder = AudioToTextRecorder()
## 
##     while True:
##         recorder.text(process_text)
