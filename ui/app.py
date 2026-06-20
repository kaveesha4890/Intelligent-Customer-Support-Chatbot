import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template_string
from src.chatbot_pipeline import chat

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Customer Support Chatbot</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        h1 { color: #2c3e50; margin-bottom: 20px; font-size: 24px; }
        #app { width: 700px; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; height: 80vh; }
        #chat-window { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
        .msg { max-width: 75%; padding: 12px 16px; border-radius: 18px; line-height: 1.5; font-size: 14px; }
        .user { background: #0084ff; color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
        .bot { background: #f1f0f0; color: #222; align-self: flex-start; border-bottom-left-radius: 4px; }
        #meta { background: #fafafa; border-top: 1px solid #eee; padding: 10px 20px; font-size: 12px; color: #666; display: flex; gap: 20px; }
        #escalation { background: #ff4444; color: white; padding: 8px 20px; font-size: 13px; font-weight: bold; display: none; }
        #input-row { display: flex; padding: 16px; gap: 10px; border-top: 1px solid #eee; }
        #user-input { flex: 1; padding: 12px 16px; border: 1px solid #ddd; border-radius: 24px; font-size: 14px; outline: none; }
        #user-input:focus { border-color: #0084ff; }
        #send-btn { background: #0084ff; color: white; border: none; border-radius: 24px; padding: 12px 24px; cursor: pointer; font-size: 14px; font-weight: bold; }
        #send-btn:hover { background: #006fd6; }
        #send-btn:disabled { background: #aaa; cursor: not-allowed; }
        .typing { color: #aaa; font-style: italic; font-size: 13px; }
    </style>
</head>
<body>
    <h1>🤖 AI Customer Support Chatbot</h1>
    <div id="app">
        <div id="escalation" id="escalation-banner">🚨 ESCALATED TO HUMAN AGENT</div>
        <div id="chat-window">
            <div class="msg bot">Hello! I'm your AI customer support assistant. How can I help you today?</div>
        </div>
        <div id="meta">
            <span id="intent-label">Intent: —</span>
            <span id="sentiment-label">Sentiment: —</span>
        </div>
        <div id="input-row">
            <input id="user-input" type="text" placeholder="Type your message..." onkeydown="if(event.key==='Enter') sendMessage()"/>
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let history = [];

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const msg = input.value.trim();
            if (!msg) return;

            input.value = '';
            document.getElementById('send-btn').disabled = true;

            appendMessage(msg, 'user');
            const typing = appendMessage('Typing...', 'bot typing');

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg, history: history})
                });
                const data = await res.json();

                typing.remove();
                appendMessage(data.response, 'bot');
                history.push([msg, data.response]);

                document.getElementById('intent-label').textContent = 'Intent: ' + data.intent;
                document.getElementById('sentiment-label').textContent = 'Sentiment: ' + data.sentiment;

                const banner = document.getElementById('escalation');
                banner.style.display = data.escalated ? 'block' : 'none';
            } catch(e) {
                typing.remove();
                appendMessage('Error: Could not get response.', 'bot');
            }

            document.getElementById('send-btn').disabled = false;
            input.focus();
        }

        function appendMessage(text, cls) {
            const win = document.getElementById('chat-window');
            const div = document.createElement('div');
            div.className = 'msg ' + cls;
            div.textContent = text;
            win.appendChild(div);
            win.scrollTop = win.scrollHeight;
            return div;
        }
    </script>
</body>
</html>
"""

conversation_history = []

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    message = data.get('message', '')
    history = data.get('history', [])

    response, intent, sentiment, escalated = chat(message, history)

    return jsonify({
        'response': response,
        'intent': intent,
        'sentiment': sentiment,
        'escalated': escalated
    })

if __name__ == '__main__':
    print("Starting AI Chatbot UI...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)