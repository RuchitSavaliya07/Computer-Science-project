import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report


MODEL_NAME = "bert-base-uncased"


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]),
        }


def evaluate_sentiment_model(csv_path: str):
    df = pd.read_csv(csv_path)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    dataset = ReviewDataset(
        df["Review"].tolist(),
        df["sentiment_label"].tolist(),
        tokenizer,
    )

    loader = DataLoader(dataset, batch_size=8)
    model.eval()

    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.tolist())
            true_labels.extend(batch["labels"].tolist())

    print("Accuracy:", accuracy_score(true_labels, predictions))
    print(classification_report(true_labels, predictions))


if __name__ == "__main__":
    evaluate_sentiment_model("data/processed/test.csv")
