import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# datasets must be imported before torch anywhere in the process (Windows CUDA DLL conflict)
from datasets import load_dataset  # noqa: F401

from flask import Flask, request, jsonify, render_template_string, session, redirect
from src.chatbot_pipeline import chat
from src.database import init_db, save_turn, save_feedback, get_all_stats
from src.accounts_db import init_accounts_db, verify_pin, get_account_info, create_demo_account

app = Flask(__name__)
# Secret key signs the session cookie.
# Override with FLASK_SECRET_KEY env-var in any non-demo environment.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "demo-secret-change-in-production")
app.config["SESSION_COOKIE_HTTPONLY"] = True   # JS cannot read the cookie

init_db()
init_accounts_db()

# ── Login / Signup page ───────────────────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SecureBank — Verify Identity</title>
    <style>
        *{box-sizing:border-box;margin:0;padding:0}
        body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
             background:linear-gradient(135deg,#1a3a5c 0%,#2563eb 100%);
             min-height:100vh;display:flex;flex-direction:column;
             align-items:center;justify-content:center;padding:20px}
        .demo-badge{background:#fef3c7;color:#92400e;border:1px solid #fcd34d;
                    padding:6px 16px;border-radius:20px;font-size:12px;
                    font-weight:600;margin-bottom:20px;letter-spacing:.5px}
        .bank-header{text-align:center;margin-bottom:28px}
        .bank-logo{font-size:30px;color:white;font-weight:800;letter-spacing:-1px}
        .bank-logo span{color:#93c5fd}
        .bank-tagline{color:rgba(255,255,255,.7);font-size:13px;margin-top:4px}
        .card{background:white;border-radius:16px;padding:32px;width:360px;
              box-shadow:0 20px 60px rgba(0,0,0,.25)}
        .tabs{display:flex;background:#f1f5f9;border-radius:8px;padding:4px;margin-bottom:24px}
        .tab{flex:1;text-align:center;padding:8px;border-radius:6px;font-size:14px;
             font-weight:600;cursor:pointer;color:#64748b;transition:all .2s}
        .tab.active{background:white;color:#1a3a5c;box-shadow:0 1px 4px rgba(0,0,0,.1)}
        .form-group{margin-bottom:16px}
        label{display:block;font-size:13px;font-weight:600;color:#374151;margin-bottom:5px}
        input{width:100%;padding:11px 14px;border:1.5px solid #e2e8f0;border-radius:8px;
              font-size:14px;color:#1e293b;outline:none;transition:border-color .2s;background:#fafafa}
        input:focus{border-color:#2563eb;background:white}
        input[type=password]{letter-spacing:4px;font-size:18px}
        input[type=password]::placeholder{letter-spacing:0;font-size:14px}
        .hint{font-size:11px;color:#94a3b8;margin-top:4px}
        .submit-btn{width:100%;padding:13px;background:#1a3a5c;color:white;border:none;
                    border-radius:8px;font-size:15px;font-weight:700;cursor:pointer;
                    margin-top:6px;transition:background .2s}
        .submit-btn:hover{background:#2563eb}
        .submit-btn:disabled{background:#94a3b8;cursor:not-allowed}
        .alert{padding:10px 14px;border-radius:8px;font-size:13px;margin-bottom:16px;display:none}
        .alert.error{background:#fef2f2;color:#dc2626;border:1px solid #fecaca}
        .alert.locked{background:#fffbeb;color:#92400e;border:1px solid #fcd34d}
        .alert.success{background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0}
        .back-link{text-align:center;margin-top:18px;font-size:13px;color:rgba(255,255,255,.7)}
        .back-link a{color:#93c5fd;text-decoration:none;font-weight:600}
        .back-link a:hover{text-decoration:underline}
    </style>
</head>
<body>
    <div class="demo-badge">&#128274; DEMO MODE — Simulated Data Only</div>
    <div class="bank-header">
        <div class="bank-logo">Secure<span>Bank</span></div>
        <div class="bank-tagline">AI Customer Support Demo</div>
    </div>
    <div class="card">
        <div class="tabs">
            <div class="tab active" onclick="showTab('login')">Login</div>
            <div class="tab" onclick="showTab('signup')">Sign Up</div>
        </div>

        <!-- Login -->
        <div id="login-form">
            <div id="login-alert" class="alert"></div>
            <div class="form-group">
                <label>Customer ID</label>
                <input type="text" id="login-id" placeholder="e.g. DEMO001" autocomplete="username"/>
            </div>
            <div class="form-group">
                <label>PIN</label>
                <input type="password" id="login-pin" placeholder="&#8226;&#8226;&#8226;&#8226;"
                       maxlength="6" inputmode="numeric" autocomplete="current-password"
                       onkeydown="if(event.key==='Enter')doLogin()"/>
                <div class="hint">4–6 digit PIN</div>
            </div>
            <button class="submit-btn" onclick="doLogin()">Verify Identity</button>
        </div>

        <!-- Signup -->
        <div id="signup-form" style="display:none">
            <div id="signup-alert" class="alert"></div>
            <div class="form-group">
                <label>Full Name</label>
                <input type="text" id="signup-name" placeholder="Your full name"/>
            </div>
            <div class="form-group">
                <label>Choose Customer ID</label>
                <input type="text" id="signup-id" placeholder="e.g. MYACCT01"/>
                <div class="hint">3–15 alphanumeric characters</div>
            </div>
            <div class="form-group">
                <label>Create PIN</label>
                <input type="password" id="signup-pin" placeholder="&#8226;&#8226;&#8226;&#8226;"
                       maxlength="6" inputmode="numeric"/>
                <div class="hint">4–6 digits only</div>
            </div>
            <button class="submit-btn" onclick="doSignup()">Create Demo Account</button>
        </div>
    </div>
    <div class="back-link">Don't have an account? Switch to the <a href="#" onclick="showTab('signup');return false;">Sign Up</a> tab above.</div>

<script>
function showTab(t){
    document.querySelectorAll('.tab').forEach((el,i)=>
        el.classList.toggle('active',(t==='login'&&i===0)||(t==='signup'&&i===1)));
    document.getElementById('login-form').style.display=t==='login'?'block':'none';
    document.getElementById('signup-form').style.display=t==='signup'?'block':'none';
}
function showAlert(id,type,msg){
    const el=document.getElementById(id);
    el.className='alert '+type; el.textContent=msg; el.style.display='block';
}
async function doLogin(){
    const cid=document.getElementById('login-id').value.trim();
    const pin=document.getElementById('login-pin').value;
    if(!cid||!pin){showAlert('login-alert','error','Please enter your Customer ID and PIN.');return;}
    const btn=document.querySelector('#login-form .submit-btn');
    btn.disabled=true; btn.textContent='Verifying...';
    const res=await fetch('/login',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({customer_id:cid,pin:pin})});
    const data=await res.json();
    btn.disabled=false; btn.textContent='Verify Identity';
    document.getElementById('login-pin').value='';   // clear PIN from DOM immediately
    if(data.success){
        showAlert('login-alert','success','Identity verified! Redirecting...');
        setTimeout(()=>window.location.href='/',800);
    } else {
        showAlert('login-alert',data.locked?'locked':'error',data.message);
    }
}
async function doSignup(){
    const name=document.getElementById('signup-name').value.trim();
    const cid=document.getElementById('signup-id').value.trim();
    const pin=document.getElementById('signup-pin').value;
    if(!name||!cid||!pin){showAlert('signup-alert','error','Please fill in all fields.');return;}
    if(!/^[0-9]{4,6}$/.test(pin)){showAlert('signup-alert','error','PIN must be 4–6 digits.');return;}
    if(!/^[a-zA-Z0-9]{3,15}$/.test(cid)){
        showAlert('signup-alert','error','Customer ID must be 3–15 alphanumeric characters.');return;}
    const btn=document.querySelector('#signup-form .submit-btn');
    btn.disabled=true; btn.textContent='Creating...';
    const res=await fetch('/signup',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({name:name,customer_id:cid,pin:pin})});
    const data=await res.json();
    btn.disabled=false; btn.textContent='Create Demo Account';
    document.getElementById('signup-pin').value='';   // clear PIN from DOM immediately
    if(data.success){
        showAlert('signup-alert','success','Account created! Redirecting...');
        setTimeout(()=>window.location.href='/',1200);
    } else {
        showAlert('signup-alert','error',data.message);
    }
}
</script>
</body>
</html>
"""

# ── Main chat page ────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Customer Support Chatbot</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; height: 100vh;
               display: flex; flex-direction: column; align-items: center; justify-content: center; }
        h1 { color: #2c3e50; margin-bottom: 12px; font-size: 24px; }
        h1 a { font-size: 14px; font-weight: normal; color: #0084ff; text-decoration: none; margin-left: 16px; }

        /* Account header — shown only when authenticated */
        #account-header {
            display: none; width: 700px; background: #1a3a5c; color: white;
            border-radius: 10px 10px 0 0; padding: 10px 20px;
            display: none; align-items: center; gap: 16px; font-size: 13px; flex-wrap: wrap;
        }
        #account-header .acct-name  { font-weight: 700; }
        #account-header .acct-no    { color: #93c5fd; font-family: monospace; }
        #account-header .acct-bal   { color: #86efac; font-weight: 600; margin-left: auto; }
        #account-header .logout-btn {
            background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.3);
            color: white; padding: 3px 12px; border-radius: 12px; cursor: pointer;
            font-size: 12px; margin-left: 8px;
        }
        #account-header .logout-btn:hover { background: rgba(255,255,255,.25); }

        /* Auth banner — shown when NOT authenticated */
        #auth-banner {
            display: none; width: 700px; background: #fffbeb; border: 1px solid #fcd34d;
            border-radius: 10px 10px 0 0; padding: 8px 20px;
            font-size: 13px; color: #92400e; align-items: center; gap: 10px;
        }
        #auth-banner a {
            margin-left: auto; background: #1a3a5c; color: white; padding: 4px 14px;
            border-radius: 12px; text-decoration: none; font-size: 12px; font-weight: 600;
        }
        #auth-banner a:hover { background: #2563eb; }

        #app { width: 700px; background: white; border-radius: 0 0 12px 12px;
               box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden;
               display: flex; flex-direction: column; height: 75vh; }
        #chat-window { flex: 1; overflow-y: auto; padding: 20px;
                       display: flex; flex-direction: column; gap: 12px; }
        .bubble-row { display: flex; flex-direction: column; }
        .bubble-row.user-row { align-items: flex-end; }
        .bubble-row.bot-row  { align-items: flex-start; }
        .msg { max-width: 75%; padding: 12px 16px; border-radius: 18px;
               line-height: 1.5; font-size: 14px; white-space: pre-wrap; }
        .user { background: #0084ff; color: white; border-bottom-right-radius: 4px; }
        .bot  { background: #f1f0f0; color: #222;  border-bottom-left-radius: 4px; }
        .feedback-row { display: flex; gap: 6px; margin-top: 4px; }
        .fb-btn { background: none; border: 1px solid #ddd; border-radius: 12px;
                  padding: 2px 8px; cursor: pointer; font-size: 14px; }
        .fb-btn:hover { background: #eee; }
        #meta { background: #fafafa; border-top: 1px solid #eee; padding: 10px 20px;
                font-size: 12px; color: #666; display: flex; gap: 20px; flex-wrap: wrap; }
        #escalation { background: #ff4444; color: white; padding: 8px 20px;
                      font-size: 13px; font-weight: bold; display: none; }
        #input-row { display: flex; padding: 16px; gap: 10px; border-top: 1px solid #eee; }
        #user-input { flex: 1; padding: 12px 16px; border: 1px solid #ddd;
                      border-radius: 24px; font-size: 14px; outline: none; }
        #user-input:focus { border-color: #0084ff; }
        #send-btn { background: #0084ff; color: white; border: none; border-radius: 24px;
                    padding: 12px 24px; cursor: pointer; font-size: 14px; font-weight: bold; }
        #send-btn:hover { background: #006fd6; }
        #send-btn:disabled { background: #aaa; cursor: not-allowed; }
        .typing { color: #aaa; font-style: italic; font-size: 13px; }
    </style>
</head>
<body>
    <h1>AI Customer Support Chatbot <a href="/dashboard">Analytics Dashboard</a></h1>

    <!-- Account header (visible when logged in) -->
    <div id="account-header">
        <span class="acct-name" id="hdr-name"></span>
        <span class="acct-no"   id="hdr-acct"></span>
        <span style="color:#93c5fd" id="hdr-type"></span>
        <span class="acct-bal"  id="hdr-bal"></span>
        <button class="logout-btn" onclick="doLogout()">Log out</button>
    </div>

    <!-- Auth banner (visible when NOT logged in) -->
    <div id="auth-banner">
        &#128274; Verify your identity to check your balance or transactions
        <a href="/login">Verify Identity</a>
    </div>

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
            <input id="user-input" type="text" placeholder="Type your message..."
                   onkeydown="if(event.key==='Enter') sendMessage()"/>
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

<script>
    let history = [];
    const sessionId = crypto.randomUUID();
    let lastFeedbackRow = null;
    const NO_FEEDBACK_INTENTS = new Set(['greeting', 'unclear']);

    // Load account info on page load
    window.addEventListener('DOMContentLoaded', async () => {
        const res = await fetch('/me');
        const data = await res.json();
        if (data.authenticated) {
            document.getElementById('hdr-name').textContent = data.name;
            document.getElementById('hdr-acct').textContent = data.masked_account_no;
            document.getElementById('hdr-type').textContent = data.account_type;
            document.getElementById('hdr-bal').textContent  = data.balance;
            document.getElementById('account-header').style.display = 'flex';
            document.getElementById('auth-banner').style.display    = 'none';
            // Shrink #app to make room for the header above it
            document.getElementById('app').style.borderRadius = '0 0 12px 12px';
        } else {
            document.getElementById('account-header').style.display = 'none';
            document.getElementById('auth-banner').style.display    = 'flex';
        }
    });

    async function doLogout() {
        const res = await fetch('/logout', {method: 'POST'});
        const data = await res.json();
        window.location.href = data.redirect || '/login';
    }

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
            document.getElementById('escalation').style.display = data.escalated ? 'block' : 'none';
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
        const isTyping = text === 'Typing...';
        const showFeedback = !isTyping && turnId && !NO_FEEDBACK_INTENTS.has(intent);

        if (showFeedback && lastFeedbackRow) {
            const prev = lastFeedbackRow.querySelector('.feedback-row');
            if (prev) prev.remove();
            lastFeedbackRow = null;
        }

        const row = document.createElement('div');
        row.className = 'bubble-row bot-row';
        const fbHtml = showFeedback
            ? `<div class="feedback-row">
                   <button class="fb-btn" onclick="sendFeedback(${turnId},1,this)">&#128077;</button>
                   <button class="fb-btn" onclick="sendFeedback(${turnId},0,this)">&#128078;</button>
               </div>`
            : '';
        row.innerHTML = `<div class="msg bot ${isTyping?'typing':''}">${escHtml(text)}</div>${fbHtml}`;
        win.appendChild(row);
        win.scrollTop = win.scrollHeight;
        if (showFeedback) lastFeedbackRow = row;
        return row;
    }

    async function sendFeedback(turnId, rating, btn) {
        await fetch('/feedback', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({turn_id: turnId, rating: rating})
        });
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

# ── Dashboard ─────────────────────────────────────────────────────────────────

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
        .card { background: white; border-radius: 10px; padding: 20px 28px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); min-width: 140px; }
        .card .num { font-size: 36px; font-weight: bold; color: #0084ff; }
        .card .label { font-size: 13px; color: #888; margin-top: 4px; }
        .card.red .num { color: #e74c3c; }
        .card.green .num { color: #27ae60; }
        .section { background: white; border-radius: 10px; padding: 20px;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }
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
            <div class="bar" style="width:{{[item.count*8,300]|min}}px"></div>
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
                <td>{{r.user_msg}}</td><td>{{r.bot_msg}}</td>
                <td>{{r.intent}}</td><td>{{r.timestamp}}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if not session.get('authenticated_customer_id'):
        return redirect('/login')
    return render_template_string(HTML)


@app.route('/login', methods=['GET'])
def login_page():
    return render_template_string(LOGIN_HTML)


@app.route('/login', methods=['POST'])
def login_post():
    data        = request.json or {}
    customer_id = data.get('customer_id', '').strip()
    pin         = data.get('pin', '')

    if not customer_id or not pin:
        return jsonify({'success': False, 'locked': False,
                        'message': 'Please provide Customer ID and PIN.'})

    result = verify_pin(customer_id, pin)
    if result['success']:
        # Store only the customer_id in the session — never the PIN or hash
        session['authenticated_customer_id'] = customer_id
        session.permanent = False   # session expires when browser closes
    return jsonify(result)


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('authenticated_customer_id', None)
    return jsonify({'ok': True, 'redirect': '/login'})


@app.route('/signup', methods=['POST'])
def signup():
    data        = request.json or {}
    name        = data.get('name', '').strip()
    customer_id = data.get('customer_id', '').strip()
    pin         = data.get('pin', '')

    result = create_demo_account(name, customer_id, pin)
    if result['success']:
        # Auto-login after successful signup
        session['authenticated_customer_id'] = customer_id
    return jsonify(result)


@app.route('/me')
def me():
    """Return masked account info for the current session (if authenticated)."""
    customer_id = session.get('authenticated_customer_id')
    if not customer_id:
        return jsonify({'authenticated': False})

    info = get_account_info(customer_id)
    if info is None:
        session.pop('authenticated_customer_id', None)   # stale session
        return jsonify({'authenticated': False})

    # Never return the full account_no — only the pre-masked version from accounts_db
    return jsonify({
        'authenticated':    True,
        'name':             info['name'],
        'masked_account_no': info['masked_account_no'],
        'account_type':     info['account_type'],
        'balance':          f"${info['balance']:,.2f}",
    })


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data       = request.json
    message    = data.get('message', '')
    history    = data.get('history', [])
    session_id = data.get('session_id', 'unknown')

    # Read authenticated customer ID from the server-side session.
    # This is the ONLY place session_customer_id is sourced from.
    # It is NEVER read from the chat message itself.
    session_customer_id = session.get('authenticated_customer_id')

    response, intent, sentiment, escalated, category = chat(
        message, history, session_customer_id=session_customer_id
    )

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
