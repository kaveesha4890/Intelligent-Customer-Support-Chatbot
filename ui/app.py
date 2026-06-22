import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# datasets must be imported before torch anywhere in the process (Windows CUDA DLL conflict)
from datasets import load_dataset  # noqa: F401

from flask import Flask, request, jsonify, render_template_string
from src.agent_graph import run_agent
from src.database import init_db, save_turn, save_feedback, get_all_stats

app = Flask(__name__)
init_db()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Customer Support Chatbot</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        h1 { color: #2c3e50; margin-bottom: 20px; font-size: 24px; }
        h1 a { font-size: 14px; font-weight: normal; color: #0084ff; text-decoration: none; margin-left: 16px; }
        #app { width: 700px; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; height: 80vh; }
        #chat-window { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
        .bubble-row { display: flex; flex-direction: column; }
        .bubble-row.user-row { align-items: flex-end; }
        .bubble-row.bot-row  { align-items: flex-start; }
        .msg { max-width: 75%; padding: 12px 16px; border-radius: 18px; line-height: 1.5; font-size: 14px; }
        .user { background: #0084ff; color: white; border-bottom-right-radius: 4px; }
        .bot  { background: #f1f0f0; color: #222; border-bottom-left-radius: 4px; }
        .feedback-row { display: flex; gap: 6px; margin-top: 4px; }
        .fb-btn { background: none; border: 1px solid #ddd; border-radius: 12px; padding: 2px 8px; cursor: pointer; font-size: 14px; }
        .fb-btn:hover { background: #eee; }
        .fb-btn.selected { background: #e0f0ff; border-color: #0084ff; }
        #meta { background: #fafafa; border-top: 1px solid #eee; padding: 10px 20px; font-size: 12px; color: #666; display: flex; gap: 20px; flex-wrap: wrap; }
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
    <h1>AI Customer Support Chatbot <a href="/dashboard">Analytics Dashboard</a></h1>
    <div id="app">
        <div id="escalation">ESCALATED TO HUMAN AGENT</div>
        <div id="chat-window">
            <div class="bubble-row bot-row">
                <div class="msg bot">Hello! I'm your AI customer support assistant. How can I help you today?</div>
            </div>
        </div>
        <div id="meta">
            <span id="intent-label">Intent: —</span>
            <span id="sentiment-label">Sentiment: —</span>
            <span id="category-label">Category: —</span>
        </div>
        <div id="input-row">
            <input id="user-input" type="text" placeholder="Type your message..." onkeydown="if(event.key==='Enter') sendMessage()"/>
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let history = [];
        const sessionId = crypto.randomUUID();
        let lastFeedbackRow = null;   // tracks which bot row currently has feedback buttons

        const NO_FEEDBACK_INTENTS = new Set(['greeting', 'unclear']);

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const msg = input.value.trim();
            if (!msg) return;

            input.value = '';
            document.getElementById('send-btn').disabled = true;

            appendUserMessage(msg);
            const typing = appendBotMessage('Typing...', null, null);

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg, history: history, session_id: sessionId})
                });
                const data = await res.json();

                typing.remove();
                appendBotMessage(data.response, data.turn_id, data.intent);
                history.push([msg, data.response]);

                document.getElementById('intent-label').textContent   = 'Intent: '    + data.intent;
                document.getElementById('sentiment-label').textContent = 'Sentiment: ' + data.sentiment;
                document.getElementById('category-label').textContent  = 'Category: '  + data.category;

                const banner = document.getElementById('escalation');
                banner.style.display = data.escalated ? 'block' : 'none';
            } catch(e) {
                typing.remove();
                appendBotMessage('Error: Could not get response.', null, null);
            }

            document.getElementById('send-btn').disabled = false;
            input.focus();
        }

        function appendUserMessage(text) {
            const win = document.getElementById('chat-window');
            const row = document.createElement('div');
            row.className = 'bubble-row user-row';
            row.innerHTML = `<div class="msg user">${escHtml(text)}</div>`;
            win.appendChild(row);
            win.scrollTop = win.scrollHeight;
        }

        function appendBotMessage(text, turnId, intent) {
            const win = document.getElementById('chat-window');
            const isTyping  = text === 'Typing...';
            const showFeedback = !isTyping && turnId && !NO_FEEDBACK_INTENTS.has(intent);

            // Remove feedback buttons from the previous bot message
            if (showFeedback && lastFeedbackRow) {
                const prev = lastFeedbackRow.querySelector('.feedback-row');
                if (prev) prev.remove();
                lastFeedbackRow = null;
            }

            const row = document.createElement('div');
            row.className = 'bubble-row bot-row';

            const fbHtml = showFeedback
                ? `<div class="feedback-row">
                       <button class="fb-btn" onclick="sendFeedback(${turnId}, 1, this)">👍</button>
                       <button class="fb-btn" onclick="sendFeedback(${turnId}, 0, this)">👎</button>
                   </div>`
                : '';

            row.innerHTML = `<div class="msg bot ${isTyping ? 'typing' : ''}">${escHtml(text)}</div>${fbHtml}`;
            win.appendChild(row);
            win.scrollTop = win.scrollHeight;

            if (showFeedback) lastFeedbackRow = row;
            return row;
        }

        async function sendFeedback(turnId, rating, btn) {
            await fetch('/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({turn_id: turnId, rating: rating})
            });
            // Replace buttons with a quiet confirmation and clear the tracker
            btn.closest('.feedback-row').innerHTML =
                '<span style="font-size:11px;color:#aaa;">Thanks for your feedback</span>';
            lastFeedbackRow = null;
        }

        function escHtml(text) {
            return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }
    </script>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Analytics Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f0f2f5; padding: 30px; }
        h1 { color: #2c3e50; margin-bottom: 24px; }
        h1 a { font-size: 14px; font-weight: normal; color: #0084ff; text-decoration: none; margin-left: 16px; }
        .cards { display: flex; gap: 16px; margin-bottom: 30px; flex-wrap: wrap; }
        .card { background: white; border-radius: 10px; padding: 20px 28px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); min-width: 140px; }
        .card .num { font-size: 36px; font-weight: bold; color: #0084ff; }
        .card .label { font-size: 13px; color: #888; margin-top: 4px; }
        .card.red .num { color: #e74c3c; }
        .card.green .num { color: #27ae60; }
        .section { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }
        .section h2 { font-size: 16px; color: #333; margin-bottom: 16px; }
        .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 13px; }
        .bar-label { width: 200px; color: #555; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; }
        .bar { height: 20px; background: #0084ff; border-radius: 4px; min-width: 4px; }
        .bar-count { color: #888; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { text-align: left; padding: 8px 12px; border-bottom: 2px solid #eee; color: #666; }
        td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
        td:first-child { color: #c0392b; max-width: 200px; }
    </style>
</head>
<body>
    <h1>Analytics Dashboard <a href="/">Back to Chat</a></h1>

    <div class="cards">
        <div class="card"><div class="num">{{total}}</div><div class="label">Total Messages</div></div>
        <div class="card red"><div class="num">{{escalated}}</div><div class="label">Escalated</div></div>
        <div class="card green"><div class="num">{{thumbs_up}}</div><div class="label">Thumbs Up</div></div>
        <div class="card red"><div class="num">{{thumbs_down}}</div><div class="label">Thumbs Down</div></div>
    </div>

    <div class="section">
        <h2>Top Intents</h2>
        {% for item in top_intents %}
        <div class="bar-row">
            <div class="bar-label" title="{{item.intent}}">{{item.intent}}</div>
            <div class="bar" style="width: {{[item.count * 8, 300] | min}}px"></div>
            <div class="bar-count">{{item.count}}</div>
        </div>
        {% endfor %}
    </div>

    {% if bad_responses %}
    <div class="section">
        <h2>Thumbs-Down Responses</h2>
        <table>
            <tr><th>User said</th><th>Bot replied</th><th>Intent</th><th>Time</th></tr>
            {% for r in bad_responses %}
            <tr>
                <td>{{r.user_msg}}</td>
                <td>{{r.bot_msg}}</td>
                <td>{{r.intent}}</td>
                <td>{{r.timestamp}}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data       = request.json
    message    = data.get('message', '')
    history    = data.get('history', [])
    session_id = data.get('session_id', 'unknown')

    response, intent, sentiment, escalated, category = run_agent(message, history, session_id)

    turn_id = save_turn(session_id, message, response, intent, sentiment, escalated, category)

    return jsonify({
        'response':  response,
        'intent':    intent,
        'sentiment': sentiment,
        'category':  category,
        'escalated': escalated,
        'turn_id':   turn_id,
    })


@app.route('/feedback', methods=['POST'])
def feedback_endpoint():
    data = request.json
    save_feedback(data['turn_id'], data['rating'])
    return jsonify({'ok': True})


@app.route('/stats')
def stats():
    return jsonify(get_all_stats())


@app.route('/dashboard')
def dashboard():
    data = get_all_stats()
    return render_template_string(
        DASHBOARD_HTML,
        total=data['total_messages'],
        escalated=data['escalated'],
        thumbs_up=data['thumbs_up'],
        thumbs_down=data['thumbs_down'],
        top_intents=data['top_intents'],
        bad_responses=data['bad_responses'],
    )


if __name__ == '__main__':
    print("Starting AI Chatbot UI...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)
