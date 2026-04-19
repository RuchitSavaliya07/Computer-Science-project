import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


MODEL_NAME = "t5-small"


class ReviewSummarizer:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    def summarize(self, text: str, max_length: int = 64) -> str:
        input_text = "summarize: " + text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=20,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def generate_summaries(input_csv: str, output_csv: str, limit: int = 500):
    df = pd.read_csv(input_csv).head(limit)
    summarizer = ReviewSummarizer()

    df["generated_summary"] = df["Review"].apply(summarizer.summarize)
    df.to_csv(output_csv, index=False)

    print("Summaries generated and saved.")


if __name__ == "__main__":
    generate_summaries(
        input_csv="data/processed/test.csv",
        output_csv="results/summaries/generated_summaries.csv",
    )
