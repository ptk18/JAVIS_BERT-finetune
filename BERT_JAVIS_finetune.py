from transformers import BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#Intent labels
intent_labels = {"draw_shape": 0, "invalid_shape": 1}

#Valid shapes
VALID_SHAPES = {"circle", "square", "triangle", "rectangle"}

def detect_negation(text, shape_start):
    negation_words = ["not", "don't", "never", "no", "can't", "wont","dont","cant","cannot","won't","arent","aren't","ain't","aint"]
    
    # Tokenize the text before the shape appears
    before_shape = text[:shape_start].lower().split()

    # Check if any negation word appears before the shape
    for neg_word in negation_words:
        if neg_word in before_shape:
            return True  # Negation detected

    return False  # No negation detected


#Converting entity labels into token-level using BIO(Begin, Inside, O) tagging scheme
def align_entity_labels(text, entities, tokenizer):

    tokens = tokenizer.tokenize(text)
    labels = ["O"] * len(tokens)  #Initialize all labels as "O"

    for entity in entities:
        entity_word = entity["word"]
        entity_start = entity["start"]
        entity_end = entity["end"]

        #Tokenize the text up to the entity start
        prefix_text = text[:entity_start]
        prefix_tokens = tokenizer.tokenize(prefix_text)
        token_start = len(prefix_tokens)

        # Tokenize the entity word
        entity_tokens = tokenizer.tokenize(entity_word)
        token_end = token_start + len(entity_tokens)

        print("token_end ",token_end)

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
        text = line.strip()
        if not text:
            continue

        entities = []
        has_negation = False

        for word in text.split():
            word_lower = word.lower().strip(".")  #Remove trailing punctuation
            if word_lower in VALID_SHAPES:
                start_idx = text.lower().index(word_lower)
                end_idx = start_idx + len(word_lower)
                
                #Detect negation before this shape
                if detect_negation(text, start_idx):
                    has_negation = True  
                
                #Store valid shape entity
                entities.append({
                    "word": word_lower,
                    "start": start_idx,
                    "end": end_idx
                })

        if has_negation:
            intent_label = intent_labels["invalid_shape"]
        else:
            intent_label = intent_labels["draw_shape"] if entities else intent_labels["invalid_shape"]

        #multiple shapes
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