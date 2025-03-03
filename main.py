from speech_to_text import transcribe_audio
from BERT_JAVIS_finetune import process_input_file
import json

def main(audio_file):
    #audio to text
    text = transcribe_audio(audio_file).lower()
    print(f"Transcribed text from audio: {text}")

    input_file = "command.txt"
    with open(input_file, "w") as f:
        f.write(text)

    output_file = "command.json"
    process_input_file(input_file, output_file)

    with open(output_file, "r") as f:
        data = json.load(f)

    if data:
        result = data[0]
        intent_label = result["intent_label"]
        tokens = result["tokens"]
        entity_labels = result["entity_labels"]

        if intent_label == 0:
            shapes = []
            current_shape = []

            for token, label in zip(tokens, entity_labels):
                token = token.replace("##", "")  #Fix subword tokenization artifacts

                if label == 1:  #Beginning of a new shape (B-shape)
                    if current_shape:
                        shapes.append(" ".join(current_shape))  #Store previous shape
                        current_shape = []  #Reset for the new shape
                    current_shape.append(token)
                
                elif label == 2:  #Inside a shape (I-shape)
                    current_shape.append(token)

            if current_shape:
                shapes.append(" ".join(current_shape))  #Store last shape

            for shape in shapes:
                print(f"Drawing a {shape}...")
        else:
            print("Invalid shape. Cannot draw.")
    else:
        print("No valid input found.")

audio_file = "audios/triangle2.mp3"
main(audio_file)
