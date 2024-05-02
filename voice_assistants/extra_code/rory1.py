from whisper_cpp_python import Whisper
whisper = Whisper(model_path='/Users/rmbutler/Desktop/AI/gits/whisper.cpp/models/ggml-large.bin')
output = whisper.transcribe('/Users/rmbutler/output.wav')
print(output)
