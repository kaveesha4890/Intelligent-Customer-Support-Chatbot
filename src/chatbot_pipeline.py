from src.intent_classifier import classify_intent
from src.sentiment import analyse_sentiment
from src.retriever import retrieve_top_k
from src.prompt_templates import CHAIN_OF_THOUGHT_TEMPLATE, FEW_SHOT_EXAMPLES
from src.llm_generator import generate_response

def chat(user_message: str, history: list):
    intent_result = classify_intent(user_message)
    sentiment_result = analyse_sentiment(user_message)
    docs = retrieve_top_k(user_message, k=5)
    context = "\n\n".join(docs) if docs else "No relevant documents found."

    history_text = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in history])

    if sentiment_result["escalate"]:
        response = ("I understand this is frustrating, and I'm escalating your request "
                     "to a human support agent right now. Someone will be with you shortly.")
    else:
        prompt = FEW_SHOT_EXAMPLES + "\n\n" + CHAIN_OF_THOUGHT_TEMPLATE.format(
            intent=intent_result["intent"],
            sentiment=sentiment_result["label"],
            context=context,
            history=history_text,
            query=user_message,
        )
        response = generate_response(prompt)

    return response, intent_result["intent"], sentiment_result["label"], sentiment_result["escalate"]