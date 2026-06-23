/**
 * tracker.js — Opt-in customer monitoring for AutoTrust Bank product pages.
 *
 * PRIVACY CONTRACT (enforced in code, not just docs):
 *   - event.target.value is NEVER read, captured, or transmitted — not even to console.
 *   - Empty-field detection uses :placeholder-shown CSS selector on the server-supplied
 *     field names only.  The selector returns true when the field is empty because
 *     placeholder text is visible; this tells us the field is blank without reading the value.
 *   - Only event_type + field_name are sent to /track-event.
 *   - No tracking runs at all unless the user has consented for this specific page.
 */

(function () {
  'use strict';

  /* ── Config injected from page template ─────────────────────────────────── */
  // window.TRACKER_CONFIG must be set before this script loads:
  //   { page: "loans_personal", consented: true, sessionId: "..." }
  const cfg = window.TRACKER_CONFIG;
  if (!cfg || !cfg.consented) return;   // hard gate — nothing below runs without consent

  const PAGE       = cfg.page;
  const SESSION_ID = cfg.sessionId;

  /* ── Consent button wiring (set by base template; active even before consent) */
  // These are handled separately via form POSTs — not via this script.

  /* ── Field event listeners ──────────────────────────────────────────────── */
  function isFieldEmpty(el) {
    // Use :placeholder-shown — true when the field is empty (placeholder visible).
    // This avoids reading .value entirely.
    return el.matches(':placeholder-shown');
  }

  function send(eventType, fieldName) {
    // fieldName comes from element.name attribute only, never from element.value
    fetch('/track-event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        page:       PAGE,
        event_type: eventType,
        field_name: fieldName || null,
      }),
      // Fire-and-forget — errors are silently swallowed; tracking must never break UX
    })
    .then(function (res) {
      if (!res.ok) return;
      return res.json();
    })
    .then(function (data) {
      if (data && data.tip) showTip(data.tip);
    })
    .catch(function () { /* swallow */ });
  }

  /* ── Attach listeners to all named form fields ───────────────────────────── */
  function attachFieldListeners() {
    const form = document.getElementById('enquiry-form');
    if (!form) return;

    // blur: field lost focus — check if it is still empty
    form.addEventListener('blur', function (e) {
      const el = e.target;
      if (!el.name) return;
      if (isFieldEmpty(el)) {
        send('blur_empty', el.name);
      }
    }, true /* capture — fires before bubbled blur */);

    // focusin: user moved into a field — dismiss any visible tip
    form.addEventListener('focusin', function () {
      dismissTip();
    });

    // submit: catch form submission attempt
    form.addEventListener('submit', function (e) {
      e.preventDefault();  // form is placeholder-only, never actually submits

      // Find first empty required field to report
      let failedField = null;
      const fields = form.querySelectorAll('[name]');
      for (let i = 0; i < fields.length; i++) {
        if (isFieldEmpty(fields[i])) {
          failedField = fields[i].name;
          break;
        }
      }

      if (failedField) {
        send('submit_fail', failedField);
      } else {
        send('submit_success', null);
        showNotice('Your enquiry has been noted. A representative will contact you.');
      }
    });
  }

  /* ── Tip display ─────────────────────────────────────────────────────────── */
  let _tipTimer = null;

  function showTip(tip) {
    let el = document.getElementById('struggle-tip');
    if (!el) {
      el = document.createElement('div');
      el.id = 'struggle-tip';
      el.className = 'struggle-tip';
      // Insert after the form
      const form = document.getElementById('enquiry-form');
      if (form && form.parentNode) {
        form.parentNode.insertBefore(el, form.nextSibling);
      } else {
        document.body.appendChild(el);
      }
    }
    el.textContent = tip.message || tip;
    el.classList.add('visible');

    clearTimeout(_tipTimer);
    _tipTimer = setTimeout(dismissTip, 7000);
  }

  function dismissTip() {
    const el = document.getElementById('struggle-tip');
    if (el) el.classList.remove('visible');
    clearTimeout(_tipTimer);
  }

  function showNotice(text) {
    let el = document.getElementById('form-notice');
    if (!el) {
      el = document.createElement('p');
      el.id = 'form-notice';
      el.style.cssText = 'margin-top:10px;font-size:13px;color:var(--success);font-weight:600;';
      const form = document.getElementById('enquiry-form');
      if (form && form.parentNode) form.parentNode.insertBefore(el, form.nextSibling);
    }
    el.textContent = text;
    setTimeout(function () { el.textContent = ''; }, 5000);
  }

  /* ── Page-load event ────────────────────────────────────────────────────── */
  send('page_view', null);

  /* ── Init ───────────────────────────────────────────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attachFieldListeners);
  } else {
    attachFieldListeners();
  }
})();
