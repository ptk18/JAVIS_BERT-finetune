from transformers import BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#Intent labels
intent_labels = {"draw_shape": 0, "invalid_shape": 1}

#Valid shapes
VALID_SHAPES = {"circle", "square", "triangle", "rectangle"}


#Converting entity labels into token-level using BIO(Begin, Inside, O) tagging scheme
def align_entity_labels(text, entities, tokenizer):

    tokens = tokenizer.tokenize(text)
    labels = ["O"] * len(tokens)  # Initialize all labels as "O"

    for entity in entities:
        entity_word = entity["word"]
        entity_start = entity["start"]
        entity_end = entity["end"]

        # Tokenize the text up to the entity start
        prefix_text = text[:entity_start]
        prefix_tokens = tokenizer.tokenize(prefix_text)
        token_start = len(prefix_tokens)

        # Tokenize the entity word
        entity_tokens = tokenizer.tokenize(entity_word)
        token_end = token_start + len(entity_tokens)

        print("token_end ",token_end)

        # Check if token_end exceeds the length of tokens
        if token_end > len(tokens):
            print(f"Warning: Entity '{entity_word}' exceeds tokenized text boundaries. Skipping.")
            continue

        #Assign BIO labels
        labels[token_start] = "B-shape"
        for i in range(token_start + 1, token_end):
            labels[i] = "I-shape"
            print(f"labels[{i}]=",labels[i])
            
    print("labels ",labels)
    
    #BIO label-to-ID mapping
    label_to_id = {"O": 0, "B-shape": 1, "I-shape": 2}

    #convert BIO labels to IDs
    label_ids = [label_to_id[label] for label in labels]
    
    return tokens, labels, label_ids

def process_input_file(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    preprocessed_data = []
    for line in lines:
        text = line.strip()  #Remove leading/trailing whitespace
        if not text:
            continue  #Skip empty lines

        #Extract entities (shapes) from the text
        entities = []
        for word in text.split():
            if word.lower() in VALID_SHAPES:
                entities.append({
                    "word": word.lower(),
                    "start": text.lower().index(word.lower()),
                    "end": text.lower().index(word.lower()) + len(word)
                })

        #Tokenize and align entity labels
        tokens, labels, label_ids = align_entity_labels(text, entities, tokenizer)

        #Assign intent label
        if entities:
            intent_label = intent_labels["draw_shape"]
        else:
            intent_label = intent_labels["invalid_shape"]

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