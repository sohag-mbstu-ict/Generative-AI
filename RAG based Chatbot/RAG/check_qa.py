# https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only/tree/main/data
# https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only/tree/main/data


import csv
import os
import sys
import pandas as pd

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domain_classifier.nvidia_domain_classifier import DomainClassifier


class DomainCSVGenerator:
    def __init__(self, input_csv_path, output_csv_path):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.domain_classifier = DomainClassifier()

    # ---------------------------------------------------
    # Read CSV safely (handles extra columns)
    # ---------------------------------------------------
    def load_csv(self):
        data = []
        with open(self.input_csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    # first col = question, rest merged into answer
                    data.append([row[0], ",".join(row[1:])])
        return pd.DataFrame(data, columns=["question", "answer"])

    # ---------------------------------------------------
    # Predict domain for a single question
    # ---------------------------------------------------
    def classify_question(self, question):
        return self.domain_classifier.predict(question)["label"]

    # ---------------------------------------------------
    # Process multiple rows and append to new DF
    # ---------------------------------------------------
    def generate_labeled_csv(self):
        qa = self.load_csv()

        df = pd.DataFrame(columns=["question", "label"])
        total_qa_len = len(qa)
        print("total_qa_len -------------",total_qa_len)

        for index in range(1, 1001):
            if(index%100==0):
                print("progress ----------------- : ",index)
            question_text = qa["question"][index]
            # print(f"Processing: {question_text}")
            predicted_label = self.classify_question(question_text)

            # Add row
            df.loc[len(df)] = [question_text, predicted_label]

        # Save final CSV
        df.to_csv(self.output_csv_path, index=False, encoding="utf-8")
        print(f"\nSaved → {self.output_csv_path}")

        return df


# ---------------------------------------------------
# Run the class
# ---------------------------------------------------
if __name__ == "__main__":
    generator = DomainCSVGenerator(
        input_csv_path="/home/gflml/Chatbot/dataset/agri_qa.csv",
        output_csv_path="new.csv"
    )

    generator.generate_labeled_csv()



