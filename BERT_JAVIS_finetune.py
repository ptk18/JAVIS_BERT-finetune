from transformers import BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#Intent labels
#intent_labels = {"draw_shape": 0, "invalid_shape": 1}
intent_labels = {
    "draw_shape": 0,
    "fill_color": 1,
    "move_shape": 2,
    "rotate_shape": 3,
    "erase_shape": 4,
    "undo": 5,
    "redo": 6,
    "invalid_shape": 7
}

#Valid shapes
VALID_SHAPES = {"circle", "square", "triangle", "rectangle"}

# Valid Colors
VALID_COLORS = {"red", "blue", "green", "purple", "yellow", "black", "white"}

# Valid Directions (for movement)
VALID_DIRECTIONS = {"left", "right", "above", "below", "up", "down"}

# Valid Angles (for rotation)
VALID_ANGLES = {str(i) + "degree" for i in range(0, 361, 15)}  # Common angles like 30, 45, 90, etc.

def detect_negation(text, shape_start):
    negation_words = ["not", "don't", "never", "no", "can't", "wont","dont","cant","cannot","won't","arent","aren't","ain't","aint"]
    
    # Tokenize the text before the shape appears
    before_shape = text[:shape_start].lower().split()

    # Check if any negation word appears before the shape
    for neg_word in negation_words:
        if neg_word in before_shape:
            return True  # Negation detected

    return False  # No negation detected


def align_entity_labels(text, entities, tokenizer):
    tokens = tokenizer.tokenize(text)
    labels = ["O"] * len(tokens)  # Initialize all labels as "O"

    label_to_id = {
        "O": 0, "B-shape": 1, "I-shape": 2,
        "B-color": 3, "I-color": 4,
        "B-direction": 5, "I-direction": 6,
        "B-angle": 7, "I-angle": 8,
        "B-action": 9, "I-action": 10
    }

    for entity in entities:
        entity_word = entity["word"]
        entity_start = entity["start"]
        entity_end = entity["end"]
        entity_label = entity["label"]

        prefix_text = text[:entity_start]
        prefix_tokens = tokenizer.tokenize(prefix_text)
        token_start = len(prefix_tokens)

        entity_tokens = tokenizer.tokenize(entity_word)
        token_end = token_start + len(entity_tokens)

        if token_end > len(tokens):
            print(f"Warning: Entity '{entity_word}' exceeds tokenized text boundaries. Skipping.")
            continue

        # Assign BIO labels
        labels[token_start] = entity_label  # B-label
        for i in range(token_start + 1, token_end):
            labels[i] = entity_label.replace("B-", "I-")  # Convert to I-label

    label_ids = [label_to_id[label] for label in labels]
    return tokens, labels, label_ids

def process_input_file(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    preprocessed_data = []

    for line in lines:
        text = line.strip()
        if not text:
            continue

        entities = []
        has_negation = False

        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip(".")  #Remove trailing punctuation
            start_idx = text.lower().index(word_lower)
            end_idx = start_idx + len(word_lower)

            #Detect negation before this entity
            if detect_negation(text, start_idx):
                has_negation = True  

            #Detect SHAPES
            if word_lower in VALID_SHAPES:
                entities.append({"word": word_lower, "start": start_idx, "end": end_idx, "label": "B-shape"})

            #Detect COLORS
            elif word_lower in VALID_COLORS:
                entities.append({"word": word_lower, "start": start_idx, "end": end_idx, "label": "B-color"})

            #Detect DIRECTIONS
            elif word_lower in VALID_DIRECTIONS:
                entities.append({"word": word_lower, "start": start_idx, "end": end_idx, "label": "B-direction"})

            #Detect ANGLES
            elif word_lower in VALID_ANGLES:
                entities.append({"word": word_lower, "start": start_idx, "end": end_idx, "label": "B-angle"})

            #Detect ACTIONS (erase, undo, redo)
            elif word_lower in {"erase", "undo", "redo"}:
                entities.append({"word": word_lower, "start": start_idx, "end": end_idx, "label": "B-action"})

        #If negation detected, mark as invalid
        intent_label = intent_labels["invalid_shape"] if has_negation else intent_labels["draw_shape"]

        #Align BIO labels
        tokens, labels, label_ids = align_entity_labels(text, entities, tokenizer)

        preprocessed_data.append({
            "text": text,
            "tokens": tokens,
            "intent_label": intent_label,
            "entity_labels": label_ids
        })

    with open(output_file, "w") as f:
        json.dump(preprocessed_data, f, indent=4)

    print(f"Processed data saved to {output_file}")

# Example usage
# input_file = "input.txt"
# output_file = "data.json"
# process_input_file(input_file, output_file)