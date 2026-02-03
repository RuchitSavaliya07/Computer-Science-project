# Customer Review Summarization Tool

This project implements an NLP-based system for:
- Abstractive customer review summarization
- Sentiment classification (positive, neutral, negative)

## Dataset
Dataset-SA.csv (Kaggle)

## Setup
```bash
pip install -r requirements.txt
Preprocessing
python src/preprocessing.py

Generate Summaries
python src/summarization.py

Evaluate Summaries
python src/evaluation.py

Sentiment Evaluation
python src/sentiment.py


---