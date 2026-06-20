import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Step 1: importing chat...")
from src.chatbot_pipeline import chat
print("Step 2: import successful, calling chat()...")

response, intent, sentiment, escalated = chat("Thanks, that's helpful!", [])

print("Step 3: done")
print("Response:", response)
print("Intent:", intent)
print("Sentiment:", sentiment)
print("Escalated:", escalated)