from openai import OpenAI
client = OpenAI()

audio_file = open("speech.mp3", "rb")
translation = client.audio.translations.create(    # translates to English
  model="whisper-1", 
  file=audio_file
)
print(translation.text)

#### maybe create a new demo here using stuff
#### inside the 01_ file

audio_file = open("speech.wav", "rb")
transcript = client.audio.transcriptions.create(    # translates to specified language
    model="whisper-1",
    file=audio_file,
    language="en",
)
print(transcript.text)
