import json
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# ----------------------------
# ✅ Load keyword extractor model
# ----------------------------
model = SentenceTransformer("yanekyuk/bert-keyword-extractor")
kw_model = KeyBERT(model)

# ----------------------------
# ✅ Load QA dataset
# ----------------------------
with open("/home/gflml/Chatbot/dataset/41_crops.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

questions = [item['instruction'] for item in qa_data]
answers = [item['output'] for item in qa_data]

# ----------------------------
# ✅ Keyword extraction function
# ----------------------------
def extract_keywords(text):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=5
    )

    # only return keyword text
    return [kw[0] for kw in keywords]

# ----------------------------
# ✅ Apply to dataset
# ----------------------------
output_list = []

for q, a in zip(questions, answers):
    output_list.append({
        "question": q,
        "question_keywords": extract_keywords(q),
        "answer": a,
        "answer_keywords": extract_keywords(a)
    })

# ----------------------------
# ✅ Print result
# ----------------------------
for item in output_list[:10]:
    print(item)
