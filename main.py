from speech_to_text import transcribe_audio
from BERT_JAVIS_finetune import process_input_file
import json

def main(audio_file):
    text = transcribe_audio(audio_file).lower()
    print(f"Transcribed text: {text}")

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
        print(f"Tokens: {tokens}")
        print(f"Entity Labels: {entity_labels}")

        shapes, colors, directions, angles, actions = [], [], [], [], []

        current_shape, current_color, current_direction, current_angle, current_action = [], [], [], [], []

        #Mapping entity labels to storage
        entity_mapping = {
            1: (shapes, current_shape), 2: (shapes, current_shape),  # B-shape, I-shape
            3: (colors, current_color), 4: (colors, current_color),  # B-color, I-color
            5: (directions, current_direction), 6: (directions, current_direction),  # B-direction, I-direction
            7: (angles, current_angle), 8: (angles, current_angle),  # B-angle, I-angle
            9: (actions, current_action), 10: (actions, current_action)  # B-action, I-action
        }

        #Process tokens and group them by type
        for token, label in zip(tokens, entity_labels):
            token = token.replace("##", "")

            if label in entity_mapping:
                entity_list, current_entity = entity_mapping[label]

                if label % 2 == 1:  #Beginning of a new entity (B-*)
                    if current_entity:
                        entity_list.append(" ".join(current_entity))  #Store previous entity
                    current_entity.clear()
                current_entity.append(token)  #Append current token

        #Store last detected entities
        for entity_list, current_entity in [ (shapes, current_shape), (colors, current_color), 
                                             (directions, current_direction), (angles, current_angle), 
                                             (actions, current_action)]:
            if current_entity:
                entity_list.append(" ".join(current_entity))

        #Handle multiple occurrences of the same intent
        if intent_label == 0:  # Draw Shape
            for shape in shapes:
                print(f"Drawing a {shape}...")
                for color in colors:  #Check if color is associated
                    print(f"Filling {shape} with {color} color...")

        elif intent_label == 1:  #Fill Color
            for shape in shapes:
                for color in colors:
                    print(f"Filling {shape} with {color} color...")

        elif intent_label == 2:  #Move Shape
            for shape in shapes:
                for direction in directions:
                    print(f"Moving {shape} {direction}...")

        elif intent_label == 3:  #Rotate Shape
            for shape in shapes:
                for angle in angles:
                    print(f"Rotating {shape} by {angle}...")

        elif intent_label == 4:  #Erase Shape
            for shape in shapes:
                print(f"Erasing {shape}...")

        elif intent_label == 5:  #Undo
            print("Undoing last action...")

        elif intent_label == 6:  #Redo
            print("Redoing last action...")

        else:
            print("Invalid command.")
    else:
        print("No valid input found.")

audio_file = "audios/red_circle.mp3"
main(audio_file)

