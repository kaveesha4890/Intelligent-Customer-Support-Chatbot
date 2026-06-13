import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from src.chatbot_pipeline import chat

def respond(message, history):
    response, intent, sentiment, escalated = chat(message, history)
    history = history + [(message, response)]
    alert = "🚨 ESCALATED TO HUMAN AGENT" if escalated else ""
    return history, f"**Intent:** {intent}", f"**Sentiment:** {sentiment}", alert

with gr.Blocks(title="AI Customer Support Chatbot") as demo:
    gr.Markdown("# Intelligent Customer Support Chatbot")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450)
            msg = gr.Textbox(label="Type your message")
            clear = gr.Button("Clear")
        with gr.Column(scale=1):
            intent_box = gr.Markdown("**Intent:**")
            sentiment_box = gr.Markdown("**Sentiment:**")
            escalation_box = gr.Markdown("")

    msg.submit(respond, [msg, chatbot], [chatbot, intent_box, sentiment_box, escalation_box])
    msg.submit(lambda: "", None, msg)
    clear.click(lambda: ([], "", "", ""), None, [chatbot, intent_box, sentiment_box, escalation_box])

if __name__ == "__main__":
    demo.launch()