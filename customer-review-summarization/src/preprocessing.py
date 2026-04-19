import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Review", "Summary", "Sentiment"])
    return df


def encode_sentiment(label: str) -> int:
    label = label.lower()
    if label == "positive":
        return 2
    elif label == "neutral":
        return 1
    else:
        return 0


def preprocess_and_split(
    input_path: str,
    output_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    df = load_raw_data(input_path)

    df["sentiment_label"] = df["Sentiment"].apply(encode_sentiment)

    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
    )

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/validation.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print("Preprocessing complete:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    preprocess_and_split(
        input_path="data/raw/Dataset-SA.csv",
        output_dir="data/processed",
    )
