"""
Evaluation script for the chatbot pipeline.

Reads evaluation/test_queries.csv (columns: query, reference),
runs each query through the chatbot pipeline, computes BLEU and
ROUGE-L against the reference, and writes evaluation/results.csv.

Run from project root:
    python evaluation/run_eval.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from src.chatbot_pipeline import chat

nltk.download("punkt", quiet=True)

INPUT_PATH = os.path.join(os.path.dirname(__file__), "test_queries.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results.csv")


def main():
    test_set = pd.read_csv(INPUT_PATH)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    results = []
    from nltk.translate.bleu_score import SmoothingFunction
    chencherry = SmoothingFunction()
    
    for _, row in test_set.iterrows():
        response, intent, sentiment, escalated = chat(row["query"], [])
        bleu = sentence_bleu([row["reference"].split()], response.split(), smoothing_function=chencherry.method1)
        rouge_l = scorer.score(row["reference"], response)["rougeL"].fmeasure


        results.append({
            "query": row["query"],
            "reference": row["reference"],
            "response": response,
            "bleu": bleu,
            "rouge_l": rouge_l,
            "intent": intent,
            "sentiment": sentiment,
            "escalated": escalated,
        })
        print(f"Processed: {row['query'][:50]}...")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Summary ---")
    print(f"Average BLEU:    {results_df['bleu'].mean():.4f}")
    print(f"Average ROUGE-L: {results_df['rouge_l'].mean():.4f}")
    print(f"Escalation rate: {results_df['escalated'].mean():.2%}")
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
