
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, SystemEngine
import time
import pyttsx3
import polyllm

VOICE_TYPE = "realtime-tts"  # "pyttsx3"
# VOICE_TYPE = "pyttsx3"
LLM = "ollama/phi4" # deepseek-v3, ollama:<any installed model>
ASST_NAME = "simon"

messages_to_llm = [
    { "role": "DUMMY", "content": "DUMMY MSG TO BE REPLACED" },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here is an image that I may ask questions about. Ignore the image if it is not needed to respond to a question."},
            {"type": "image", "image": "./working_image.png"},
        ],
    },
    { "role": "assistant", "content": "OK. I will use the image when relevant to query." },
    { "role": "DUMMY", "content": "DUMMY MSG TO BE REPLACED" },
]


class PlainAssistant:
    def __init__(self):
        self.conversation_history = []

        self.voice_type = VOICE_TYPE
        self.llm = LLM

        if self.voice_type == "pyttsx3":
            print("🔊 Initializing pyttsx3 TTS engine")
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)  # Speed of speech
            self.engine.setProperty("volume", 1.0)  # Volume level
        elif self.voice_type == "realtime-tts":  # uses pyttsx3 internally
            print("🔊 Initializing RealtimeTTS engine")
            self.engine = SystemEngine()
            self.stream = TextToAudioStream(
                self.engine, frames_per_buffer=256, playout_chunk_size=1024
            )
        else:
            print(f"Unsupported voice type: {self.voice_type}")
            exit(-1)

    def process_text(self, text: str) -> str:
        """Process text input and generate response"""
        try:
            # Check if text matches our last response
            if (
                self.conversation_history
                and text.strip().lower()
                in self.conversation_history[-1]["content"].lower()
            ):
                print("🤖 Ignoring own speech input")
                return ""


            # Generate response using configured llm
            print(f"🤖 Processing text with {self.llm}...")
            messages_to_llm[0] = {
                    "role": "system",
                    "content": f"just answer the question; do not elaborate; conversation history: {self.conversation_history}"
                }
            messages_to_llm[-1] = {"role": "user", "content": text}
            response = polyllm.generate(
                model=self.llm,
                messages=messages_to_llm,
            )

            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": response})

            # Speak the response
            self.speak(response)

            return response

        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
            exit(-1)

    def speak(self, text: str):
        """Convert text to speech using configured engine"""
        try:
            print(f"🔊 Speaking: {text}")

            if self.voice_type == "pyttsx3":
                # Create a new engine instance for each speech attempt
                #   because of pyttsx3 issues on mac
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 150)
                self.engine.setProperty("volume", 1.0)
                self.engine.say(text)
                self.engine.runAndWait()
                self.engine.endLoop()
                time.sleep(0.1)
            elif self.voice_type == "realtime-tts":
                # Create a new engine instance for each speech attempt
                #   because of pyttsx3 issues on mac
                self.stream.feed(text)
                self.stream.play()
                self.engine.engine.endLoop()
                time.sleep(0.1)
                self.engine = SystemEngine()
                self.stream = TextToAudioStream(
                    self.engine, frames_per_buffer=256, playout_chunk_size=1024
                )
            print(f"🔊 Spoken: {text}")
        except Exception as e:
            print(f"❌ Error in speech synthesis: {str(e)}")
            exit(-1)


def chat():
    assistant = PlainAssistant()

    recorder = AudioToTextRecorder(
        spinner=True,
        model="tiny.en",
        language="en",
        print_transcription_time=True,
    )

    def process_text(text):
        """Process user speech input"""
        try:
            assistant_name = ASST_NAME
            print("DBGAN", assistant_name.lower(), "TXT", text.lower())
            if assistant_name.lower() not in text.lower():
                print(f"🤖 Not {assistant_name} - ignoring")
                return

            if text.lower() in ["exit", "quit"]:
                print("👋 Exiting chat session")
                return False

            # process input and get response
            recorder.stop()
            response = assistant.process_text(text)
            print(f"🤖 Response: {response}")
            recorder.start()

            return True

        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
            exit(-1)

    try:
        print("🎤 Speak now... (say 'exit' or 'quit' to end)")
        while True:
            recorder.text(process_text)

    except KeyboardInterrupt:
        print("👋 Session ended by user")
        exit(0)
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        exit(-1)


if __name__ == "__main__":
    chat()
