import json
import random

# Sample entities
crops = ["Rice", "Potato", "Wheat", "Tomato", "Brinjal", "Mustard"]
diseases = ["late blight", "yellow rust", "leaf spot", "powdery mildew"]
pests = ["Aphids", "Whitefly", "Stem borer", "Red spider mite"]
fertilizers = ["Urea", "DAP", "Potash", "Compost"]

# Generic sentences
templates = [
    "{crop} is affected by {disease}",
    "{pest} attacks {crop}",
    "Apply {fertilizer} to {crop} to increase yield",
    "Signs of {disease} found on {crop} leaves",
    "Heavy {pest} infestation observed in {crop} field"
]

def generate_ner_dataset(n_samples=100):
    data = []
    for _ in range(n_samples):
        template = random.choice(templates)
        crop = random.choice(crops)
        disease = random.choice(diseases)
        pest = random.choice(pests)
        fertilizer = random.choice(fertilizers)

        sentence = template.format(crop=crop, disease=disease, pest=pest, fertilizer=fertilizer)
        tokens = sentence.split()
        ner_tags = ["O"] * len(tokens)

        # Tagging
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if token_lower in [c.lower() for c in crops]:
                ner_tags[i] = "B-CROP"
            elif token_lower in [d.lower() for d in diseases]:
                ner_tags[i] = "B-DISEASE"
            elif token_lower in [p.lower() for p in pests]:
                ner_tags[i] = "B-PEST"
            elif token_lower in [f.lower() for f in fertilizers]:
                ner_tags[i] = "B-FERTILIZER"

        data.append({"tokens": tokens, "ner_tags": ner_tags})
    return data

dataset = generate_ner_dataset(n_samples=100)

def save_jsonl(data, filename="train.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

save_jsonl(dataset, "test.jsonl")




