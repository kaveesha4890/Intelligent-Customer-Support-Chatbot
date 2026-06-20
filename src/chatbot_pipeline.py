from src.intent_classifier import classify_intent
from src.sentiment import analyse_sentiment
from src.retriever import retrieve_top_k
from src.prompt_templates import CHAIN_OF_THOUGHT_TEMPLATE
from src.llm_generator import generate_response


def _clean_response(response: str) -> str:
    """Strip any leaked reasoning or labels from small model outputs."""
    for marker in ["Answer:", "Response:", "Step 4", "Step 4 -"]:
        if marker in response:
            response = response.split(marker, 1)[1]
    lines = response.strip().split("\n")
    if lines and lines[0].rstrip().endswith(":"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def chat(user_message: str, history: list):
    intent_result = classify_intent(user_message)
    sentiment_result = analyse_sentiment(user_message)
    docs = retrieve_top_k(user_message, k=3)
    context = "\n\n".join(docs) if docs else "No relevant documents found."

    if sentiment_result["escalate"]:
        response = ("I understand this is frustrating, and I'm escalating your request "
                    "to a human support agent right now. Someone will be with you shortly.")
    else:
        prompt = CHAIN_OF_THOUGHT_TEMPLATE.format(
            context=context,
            query=user_message,
        )
        response = generate_response(prompt)
        response = _clean_response(response)

    return response, intent_result["intent"], sentiment_result["label"], sentiment_result["escalate"]