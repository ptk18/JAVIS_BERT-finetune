from transformers import pipeline

# load the Whisper model
speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# function to convert speech to text
def transcribe_audio(audio_file):
    result = speech_to_text(audio_file)
    return result["text"]

audio_file = "audios/circle.mp3"
text = transcribe_audio(audio_file)
print(f"Transcribed text: {text}")
