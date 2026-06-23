import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# datasets must be imported before torch anywhere in the process (Windows CUDA DLL conflict)
from datasets import load_dataset  # noqa: F401

from flask import Flask, request, jsonify, render_template, session, redirect
from src.agent_graph import run_agent
from src.services_db import (
    init_services_db,
    get_loan_rate,
    get_all_fd_rates,
    get_all_pawning_rates,
    get_all_transfer_fees,
    get_all_fx_rates,
)
from src.database import init_db, save_turn, save_feedback, get_all_stats
from src.accounts_db import init_accounts_db, verify_pin, get_account_info, create_demo_account
from src.ui_events_db import init_ui_events_db, save_event
from src.struggle_detector import detect_struggle

import pathlib
_UI_DIR = pathlib.Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(_UI_DIR / 'templates'),
    static_folder=str(_UI_DIR / 'static'),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "demo-secret-change-in-production-v2")
app.config["SESSION_COOKIE_HTTPONLY"] = True

init_db()
init_accounts_db()
init_services_db()
init_ui_events_db()


def _get_ui_session_id() -> str:
    """Return the per-browser UI session ID, creating one if needed."""
    if 'ui_session_id' not in session:
        import uuid
        session['ui_session_id'] = str(uuid.uuid4())
    return session['ui_session_id']


def _consent_for(page_key: str):
    """Return True/False/None for this page's consent status."""
    return session.get('tracking_consent', {}).get(page_key)


def _page_tracking_ctx(page_key: str) -> dict:
    """Build the template context dict needed by every product page."""
    return {
        'page_key':       page_key,
        'consent_status': _consent_for(page_key),
        'ui_session_id':  _get_ui_session_id(),
    }

# ── Templates live in ui/templates/, CSS in ui/static/css/main.css ───────────

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if not session.get('authenticated_customer_id'):
        return redirect('/login')
    return render_template('chat.html')


@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')


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
        session['authenticated_customer_id'] = customer_id
        session.permanent = False
    return jsonify(result)


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('authenticated_customer_id', None)
    session.pop('pending_calculation', None)      # clear any in-progress slot collection
    session.pop('customer_display_name', None)    # clear cosmetic display name
    session.pop('tracking_consent', None)         # reset per-page consent on logout
    session.pop('ui_session_id', None)            # reset tracking session ID on logout
    return jsonify({'ok': True, 'redirect': '/login'})


@app.route('/signup', methods=['POST'])
def signup():
    data        = request.json or {}
    name        = data.get('name', '').strip()
    customer_id = data.get('customer_id', '').strip()
    pin         = data.get('pin', '')

    result = create_demo_account(name, customer_id, pin)
    if result['success']:
        session['authenticated_customer_id'] = customer_id
    return jsonify(result)


@app.route('/me')
def me():
    customer_id = session.get('authenticated_customer_id')
    if not customer_id:
        return jsonify({'authenticated': False})

    info = get_account_info(customer_id)
    if info is None:
        session.pop('authenticated_customer_id', None)
        return jsonify({'authenticated': False})

    return jsonify({
        'authenticated':     True,
        'name':              info['name'],
        'masked_account_no': info['masked_account_no'],
        'account_type':      info['account_type'],
        'balance':           f"LKR {info['balance']:,.2f}",
    })


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data       = request.json
    message    = data.get('message', '')
    history    = data.get('history', [])
    session_id = data.get('session_id', 'unknown')

    # Read from Flask session — NEVER from the chat message body
    session_customer_id  = session.get('authenticated_customer_id')
    pending_calc         = session.get('pending_calculation')      # multi-turn slot state
    display_name         = session.get('customer_display_name')    # cosmetic only — never for auth

    response, intent, sentiment, escalated, category, pending_calc, display_name = run_agent(
        message, history,
        session_id=session_id,
        session_customer_id=session_customer_id,
        pending_calculation=pending_calc,
        customer_display_name=display_name,
        host_url=request.host_url,         # passed through for deterministic URL injection
    )

    # Persist or clear pending_calculation for the next turn
    if pending_calc:
        session['pending_calculation'] = pending_calc
    else:
        session.pop('pending_calculation', None)

    # Persist display name once collected (never overwrite with None if already set)
    if display_name:
        session['customer_display_name'] = display_name

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
    return render_template(
        'dashboard.html',
        total=data['total_messages'],
        escalated=data['escalated'],
        thumbs_up=data['thumbs_up'],
        thumbs_down=data['thumbs_down'],
        top_intents=data['top_intents'],
        bad_responses=data['bad_responses'],
    )


# ── Public product pages (rate-card data only, no personalisation) ─────────────
# These pages are always rendered the same regardless of login state.

@app.route('/loans/personal')
def loan_personal():
    return render_template('pages/loan_personal.html',
                           rate=get_loan_rate('personal'),
                           **_page_tracking_ctx('loans_personal'))


@app.route('/loans/housing')
def loan_housing():
    return render_template('pages/loan_housing.html',
                           rate=get_loan_rate('housing'),
                           **_page_tracking_ctx('loans_housing'))


@app.route('/loans/vehicle')
def loan_vehicle():
    return render_template('pages/loan_vehicle.html',
                           rate=get_loan_rate('vehicle'),
                           **_page_tracking_ctx('loans_vehicle'))


@app.route('/loans/education')
def loan_education():
    return render_template('pages/loan_education.html',
                           rate=get_loan_rate('education'),
                           **_page_tracking_ctx('loans_education'))


@app.route('/loans/business')
def loan_business():
    return render_template('pages/loan_business.html',
                           rate=get_loan_rate('business'),
                           **_page_tracking_ctx('loans_business'))


@app.route('/deposits/fixed-deposits')
def fixed_deposits():
    return render_template('pages/fd.html',
                           rates=get_all_fd_rates(),
                           **_page_tracking_ctx('fd'))


@app.route('/services/pawning')
def pawning():
    return render_template('pages/pawning.html',
                           rates=get_all_pawning_rates(),
                           **_page_tracking_ctx('pawning'))


@app.route('/cards')
def cards():
    return render_template('pages/cards.html',
                           **_page_tracking_ctx('cards'))


@app.route('/transfers')
def transfers():
    return render_template('pages/transfers.html',
                           fees=get_all_transfer_fees(),
                           fx_rates=get_all_fx_rates(),
                           **_page_tracking_ctx('transfers'))


# ── Opt-in customer monitoring ────────────────────────────────────────────────

@app.route('/set-consent', methods=['POST'])
def set_consent():
    data     = request.form
    page_key = data.get('page_key', '')
    choice   = data.get('choice', '')   # 'yes' or 'no'
    redirect_to = data.get('next', '/')

    if page_key and choice in ('yes', 'no'):
        consent = session.get('tracking_consent', {})
        consent[page_key] = (choice == 'yes')
        session['tracking_consent'] = consent

    return redirect(redirect_to)


@app.route('/track-event', methods=['POST'])
def track_event():
    data       = request.json or {}
    page_key   = data.get('page', '')
    event_type = data.get('event_type', '')
    field_name = data.get('field_name')   # may be None

    # Hard privacy gate — store nothing unless the user explicitly consented for this page
    if not _consent_for(page_key):
        return ('', 204)

    # Allowlist event types — reject anything unexpected
    if event_type not in ('page_view', 'blur_empty', 'submit_fail', 'submit_success'):
        return ('', 400)

    ui_session_id = _get_ui_session_id()
    save_event(ui_session_id, page_key, event_type, field_name)

    tip = None
    if event_type in ('blur_empty', 'submit_fail'):
        tip = detect_struggle(ui_session_id, page_key)

    return jsonify({'ok': True, 'tip': tip})


if __name__ == '__main__':
    print("Starting AI Chatbot UI...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)
