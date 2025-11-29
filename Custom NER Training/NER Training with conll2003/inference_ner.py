import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class BertNERInference:
    """
    ✅ Load trained NER model
    ✅ Run inference on new sentences
    ✅ Return tokens with predicted entity labels
    """

    def __init__(self, model_path):
        """
        model_path = folder where model + tokenizer are saved
        Example: /home/gflml/Chatbot/NER/output/saved_model/bert-ner-mod
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        self.id2label = self.model.config.id2label

        self.model.eval()

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, sentence):
        """
        ✅ Perform NER on a given sentence
        Returns: list of (word, label)
        """

        tokens = sentence.split()

        encodings = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encodings).logits

        predictions = logits.argmax(-1).squeeze().tolist()
        word_ids = encodings.word_ids()

        results = []
        previous_word = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word:
                continue

            label = self.id2label[predictions[idx]]
            results.append((tokens[word_id], label))
            previous_word = word_id

        return results


if __name__ == "__main__":
    model_path = "/home/gflml/Chatbot/NER/output/bert-ner-class"

    ner = BertNERInference(model_path)

    sentence = "Apple released a new iPhone in California"
    print("\nNER Results:")
    print(ner.predict(sentence))



