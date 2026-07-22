

import os
import joblib
import streamlit as st
from preprocess import clean_text, detect_language, rule_based_scam_score

MODEL_PATH = "scam_pipeline.pkl"
THRESHOLD = 0.5  # final score above this => scam

st.set_page_config(page_title="Scam Detector", page_icon="🛡️", layout="centered")


st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  :root{
    --bg:#FBFAF8; --card:#FFFFFF; --border:#ECEAE4;
    --text:#26252B; --muted:#8C8B93; --accent:#5B6CFF;
    --scam-bg:#FCEEEC; --scam-bd:#F1C9C5; --scam-tx:#B23A32;
    --warn-bg:#FBF4E4; --warn-bd:#EBDFC2; --warn-tx:#8A6D1B;
    --safe-bg:#EEF6F0; --safe-bd:#C9E5D0; --safe-tx:#2E7D48;
  }

  html, body, .stApp, .stApp * { font-family:'Inter', -apple-system, sans-serif; }
  .stApp { background:var(--bg); }

  /* hide default streamlit chrome for a cleaner canvas */
  #MainMenu, header, footer { visibility:hidden; }
  .block-container { padding-top:3rem; padding-bottom:3rem; max-width:680px; }

  /* header */
  .app-title { font-size:1.9rem; font-weight:700; color:var(--text);
               letter-spacing:-.02em; margin:0; }
  .app-sub   { color:var(--muted); font-size:1rem; margin:.35rem 0 1.6rem;
               line-height:1.5; }
  .divider   { height:1px; background:var(--border); margin:1.4rem 0; border:none; }

  /* text area */
  .stTextArea textarea{
    background:var(--card); border:1px solid var(--border); border-radius:14px;
    font-size:1rem; color:var(--text); padding:16px; line-height:1.5;
    box-shadow:0 1px 2px rgba(0,0,0,.03);
  }
  .stTextArea textarea:focus{
    border-color:var(--accent); box-shadow:0 0 0 3px rgba(91,108,255,.12);
  }
  .stTextArea label{ color:var(--muted)!important; font-weight:500; font-size:.9rem; }

  /* all buttons: soft, pill-ish */
  .stButton>button{
    border-radius:10px; border:1px solid var(--border); background:var(--card);
    color:#45444C; font-weight:500; font-size:.9rem; padding:.5rem 1rem;
    transition:all .15s ease; box-shadow:none;
  }
  .stButton>button:hover{ border-color:var(--accent); color:var(--accent); background:#fff; }
  /* primary "Check" button */
  .stButton>button[kind="primary"],
  .stButton>button[data-testid="baseButton-primary"]{
    background:var(--accent); border-color:var(--accent); color:#fff;
  }
  .stButton>button[kind="primary"]:hover{ filter:brightness(1.05); color:#fff; }

  /* verdict cards */
  .verdict{ padding:1.05rem 1.25rem; border-radius:14px; font-weight:600;
            font-size:1.1rem; display:flex; align-items:center; gap:.6rem; }
  .v-scam{ background:var(--scam-bg); border:1px solid var(--scam-bd); color:var(--scam-tx);}
  .v-warn{ background:var(--warn-bg); border:1px solid var(--warn-bd); color:var(--warn-tx);}
  .v-safe{ background:var(--safe-bg); border:1px solid var(--safe-bd); color:var(--safe-tx);}
  .conf{ font-weight:500; font-size:.85rem; opacity:.75; margin-left:auto; }

  /* meta + pills */
  .meta{ color:var(--muted); font-size:.9rem; margin-top:1rem; }
  .meta b{ color:var(--text); font-weight:600; }
  .pill{ display:inline-block; background:#F3F2EF; color:#5C5B63;
         padding:.25rem .7rem; border-radius:999px; font-size:.8rem;
         margin:.2rem .25rem .2rem 0; border:1px solid var(--border); }

  .note{ background:#F4F6FF; border:1px solid #DEE3FF; color:#3E4C99;
         padding:.7rem .9rem; border-radius:10px; font-size:.85rem; }
  .foot{ color:var(--muted); font-size:.8rem; text-align:center; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="app-title">🛡️ Scam &amp; Fraud Detector</p>',
            unsafe_allow_html=True)
st.markdown('<p class="app-sub">Paste any SMS, email, or message — English, '
            'Urdu, or Roman Urdu — and check whether it looks like a scam.</p>',
            unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()
if model is None:
    st.error("Model file `scam_pipeline.pkl` not found. "
             "Run `python prepare_data.py` then `python train.py` first.")
    st.stop()


LANG_LABELS = {"english": "English",
               "roman_urdu": "Roman Urdu (Urdu in English letters)",
               "urdu": "Urdu (script)"}
ML_WEIGHT = {"english": 0.75, "roman_urdu": 0.35, "urdu": 0.20}


def analyse(message: str):
    cleaned = clean_text(message)
    lang = detect_language(message)
    ml_p = float(model.predict_proba([cleaned])[0][1])
    rule_p, signals = rule_based_scam_score(message)
    w = ML_WEIGHT.get(lang, 0.6)
    final = w * ml_p + (1 - w) * rule_p
    return {"lang": lang, "ml_p": ml_p, "rule_p": rule_p,
            "signals": signals, "final": final}



examples = {
    "English scam": "CONGRATULATIONS! You won a $1000 gift card. Click http://bit.ly/win to claim now!",
    "Roman Urdu scam": "Mubarak ho! Aap ne 50 lakh ka inaam jeeta hai. Apna OTP code abhi bhejein.",
    "Normal message": "Hey, are we still meeting for lunch at 1pm tomorrow?",
}
st.markdown('<p class="meta">Try an example</p>', unsafe_allow_html=True)
cols = st.columns(len(examples))
for (name, txt), c in zip(examples.items(), cols):
    if c.button(name, use_container_width=True):
        st.session_state["msg"] = txt

message = st.text_area("Message", height=150, key="msg",
                       placeholder="Paste an SMS or email here...",
                       label_visibility="collapsed")

check = st.button("Check message", type="primary", use_container_width=True)


if check:
    if not message.strip():
        st.info("Please paste a message first.")
    else:
        r = analyse(message)
        conf = r["final"] * 100 if r["final"] >= THRESHOLD else (1 - r["final"]) * 100
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if r["final"] >= 0.65:
            st.markdown(f'<div class="verdict v-scam">🚨 Scam detected'
                        f'<span class="conf">{conf:.0f}% confidence</span></div>',
                        unsafe_allow_html=True)
        elif r["final"] >= THRESHOLD:
            st.markdown(f'<div class="verdict v-warn">⚠️ Looks suspicious'
                        f'<span class="conf">{conf:.0f}% confidence</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict v-safe">✅ Looks safe'
                        f'<span class="conf">{conf:.0f}% confidence</span></div>',
                        unsafe_allow_html=True)

        st.markdown(f'<p class="meta"><b>Language:</b> '
                    f'{LANG_LABELS.get(r["lang"], r["lang"])}</p>',
                    unsafe_allow_html=True)

        if r["signals"]:
            pills = " ".join(f'<span class="pill">{s}</span>' for s in r["signals"])
            st.markdown(f'<p class="meta"><b>Warning signals</b></p>{pills}',
                        unsafe_allow_html=True)

        with st.expander("Why this score?"):
            st.write(f"- ML model P(scam): **{r['ml_p']*100:.1f}%**")
            st.write(f"- Rule-based score: **{r['rule_p']*100:.1f}%**")
            st.write(f"- Blended final: **{r['final']*100:.1f}%** "
                     f"(threshold {THRESHOLD*100:.0f}%)")
            if r["lang"] != "english":
                st.markdown('<div class="note">For Urdu / Roman Urdu the rule '
                            'engine is weighted more heavily, because the ML '
                            'model was trained mostly on English data.</div>',
                            unsafe_allow_html=True)

st.markdown('<p class="foot">Educational tool — not a guarantee. '
            'Always verify suspicious messages through official channels.</p>',
            unsafe_allow_html=True)
