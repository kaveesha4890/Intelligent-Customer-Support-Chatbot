from src.chatbot_pipeline import chat
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

nltk.download("punkt")

test_set = pd.read_csv("evaluation/test_queries.csv")  # columns: query, reference
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

results = []
for _, row in test_set.iterrows():
    response, intent, sentiment, esc = chat(row["query"], [])
    bleu = sentence_bleu([row["reference"].split()], response.split())
    rouge_l = scorer.score(row["reference"], response)["rougeL"].fmeasure
    results.append({"query": row["query"], "response": response, "bleu": bleu, "rouge_l": rouge_l,
                     "intent": intent, "sentiment": sentiment})

pd.DataFrame(results).to_csv("evaluation/results.csv", index=False)
print("Done. Avg BLEU:", pd.DataFrame(results)["bleu"].mean())