import pandas as pd
from rouge_score import rouge_scorer
import json


def evaluate_rouge(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    scores = []

    for _, row in df.iterrows():
        score = scorer.score(row["Summary"], row["generated_summary"])
        scores.append(score)

    avg_scores = {
        metric: sum(s[metric].fmeasure for s in scores) / len(scores)
        for metric in scores[0]
    }

    with open(output_path, "w") as f:
        json.dump(avg_scores, f, indent=4)

    print("ROUGE evaluation complete.")
    print(avg_scores)


if __name__ == "__main__":
    evaluate_rouge(
        csv_path="results/summaries/summarization_experiments.csv",
        output_path="results/metrics/rouge_scores.json",
    )
