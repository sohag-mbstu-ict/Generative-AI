# https://huggingface.co/facebook/bart-large-mnli
# https://huggingface.co/facebook/bart-large-mnli


from transformers import pipeline
domain_classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ['AGRICULTURE', 'NOT AGRICULTURE']
while True:
    user_input = input("\nAsk your question about for domain classification : ")
    if user_input.lower() in ["exit", "quit"]:
        break
    answer = domain_classifier(user_input, candidate_labels)
    print("a : ",answer)
