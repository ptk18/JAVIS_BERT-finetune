# from transformers import pipeline

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_name = "joeddav/xlm-roberta-large-xnli"

# Load the tokenizer manually
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create the pipeline
intent_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


# Load models
speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-small")
# intent_classifier = pipeline("text-classification", model="distilbert-base-uncased")
# intent_classifier = pipeline("text-classification", model="joeddav/xlm-roberta-large-xnli")
# intent_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
#intent_classifier = pipeline("text-classification", model="joeddav/xlm-roberta-large-xnli", use_fast=False)


# Function to transcribe audio
def transcribe_audio(audio_file):
    result = speech_to_text(audio_file)
    return result["text"]

# Function to classify intent
def classify_intent(command):
    result = intent_classifier(command)
    return result[0]["label"]

# Function to execute commands
def execute_command(command):
    intent = classify_intent(command)
    print(f"intent is {intent}")
    if intent == "draw_circle":
        print("Drawing a circle...")
    elif intent == "fill_circle":
        print("Filling circle with color...")
    else:
        print("Sorry, I didn't understand that.")

def main():
    audio_file = "audios/circle.mp3" 
    command = transcribe_audio(audio_file)
    print(f"Command: {command}")
    execute_command(command)

if __name__ == "__main__":
    main()