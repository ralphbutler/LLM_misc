
from openai import OpenAI
import io
from pydub import AudioSegment
from pydub.playback import play

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Today is a wonderful day to build something people love!"
)

# Get audio data from response
audio_data = response.content

# Convert audio data to audio segment
audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")

# Play audio
play(audio_segment)
