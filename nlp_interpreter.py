from transformers import pipeline

#load the intent classification model
intent_classifier = pipeline("text-classification", model="distilbert-base-uncased")

#function to classify intent
def classify_intent(command):
    result = intent_classifier(command)
    return result[0]["label"]

command = "Please draw a circle"
intent = classify_intent(command)
print(f"Intent: {intent}")