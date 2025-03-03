from transformers import pipeline
import re

speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-small",device="cpu")

def clean_text(text):
    #remove special characters except alphanumeric and spaces including fullstop
    return re.sub(r"[^\w\s]", "", text).strip().lower()

#speech to text
def transcribe_audio(audio_file):
    text = speech_to_text(audio_file, generate_kwargs={"language": "en"})
    result = text["text"]
    cleaned_text = clean_text(result)
    return cleaned_text

# audio_file = "audios/flower.mp3"
# text = transcribe_audio(audio_file)
# print(f"Transcribed text: {text}")
