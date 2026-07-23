

import os
import joblib
import streamlit as st
from preprocess import clean_text, detect_language, rule_based_scam_score

MODEL_PATH = "scam_pipeline.pkl"
THRESHOLD = 0.5

st.set_page_config(page_title="Scam Detector", page_icon="🛡️", layout="wide")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{
  --ink:#2B2A3D; --ink-soft:#6E6C87; --line:#E7E9F5;
  --violet:#6C63FF; --violet-2:#9B7BFF;
  --mint:#3FBF8F; --mint-bg:#EAF9F2; --mint-line:#BEE9D6;
  --amber:#C98A16; --amber-bg:#FEF6E4; --amber-line:#F2DFB4;
  --rose:#E0526B;  --rose-bg:#FDEEF1;  --rose-line:#F6CBD5;
}

/* ---- fonts: scoped so Streamlit's ICON font is never overridden ---- */
html, body, .stApp, p, div, span, label, input, textarea, button, li, td, th{
  font-family:'Inter', -apple-system, sans-serif;
}
h1,h2,h3,.display{ font-family:'Outfit', sans-serif; }

/* restore Material icon font (this is what broke the expander arrow) */
[data-testid="stIconMaterial"], .material-icons, .material-icons-sharp,
span[class*="material-symbols"], [data-testid="stExpanderToggleIcon"]{
  font-family:'Material Symbols Rounded','Material Icons' !important;
}

/* ---- page background: soft pastel wash + floating shapes ---- */
.stApp{
  background:
    radial-gradient(circle at 12% 18%, #DDF3FF 0%, transparent 42%),
    radial-gradient(circle at 88% 12%, #F3E8FF 0%, transparent 40%),
    radial-gradient(circle at 78% 88%, #E4FBEF 0%, transparent 42%),
    radial-gradient(circle at 20% 92%, #FFF0E6 0%, transparent 38%),
    linear-gradient(160deg,#FBFCFF 0%,#F7F9FF 100%);
  background-attachment: fixed;
}
/* two blurred blobs drifting slowly in the background */
.stApp::before, .stApp::after{
  content:""; position:fixed; border-radius:50%; filter:blur(70px);
  opacity:.5; pointer-events:none; z-index:0;
}
.stApp::before{ width:420px; height:420px; top:-120px; left:-100px;
  background:linear-gradient(135deg,#BFD9FF,#E7D4FF); animation:drift 22s ease-in-out infinite; }
.stApp::after{ width:380px; height:380px; bottom:-140px; right:-90px;
  background:linear-gradient(135deg,#C8F5DF,#FFE2C7); animation:drift 26s ease-in-out infinite reverse; }
@keyframes drift{
  0%,100%{ transform:translate(0,0) scale(1); }
  50%{ transform:translate(30px,-25px) scale(1.08); }
}
@media (prefers-reduced-motion: reduce){
  .stApp::before,.stApp::after{ animation:none; }
}

#MainMenu, header, footer{ visibility:hidden; }
.block-container{ padding-top:2.2rem; padding-bottom:3rem; max-width:1180px;
                  position:relative; z-index:1; }

/* ---- header ---- */
.hero{ text-align:center; margin-bottom:1.9rem; }
.hero h1{
  font-size:2.9rem; font-weight:800; letter-spacing:-.03em; margin:0;
  background:linear-gradient(100deg,var(--violet),var(--violet-2) 55%,#5AC8E8);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero p{ color:var(--ink-soft); font-size:1.03rem; margin:.5rem 0 0; }
.badge{
  display:inline-block; background:#FFFFFFCC; border:1px solid var(--line);
  color:var(--ink-soft); font-size:.78rem; font-weight:600; letter-spacing:.06em;
  text-transform:uppercase; padding:.32rem .85rem; border-radius:999px;
  margin-bottom:.85rem; backdrop-filter:blur(6px);
}

/* ---- panels ---- */
.panel{
  background:#FFFFFFE6; border:1px solid var(--line); border-radius:20px;
  padding:1.5rem 1.55rem; box-shadow:0 8px 28px rgba(80,90,160,.09);
  backdrop-filter:blur(10px); height:100%;
}
.panel h3{ font-size:1.22rem; font-weight:700; color:var(--ink); margin:0 0 .3rem; }
.panel .lede{ color:var(--ink-soft); font-size:.93rem; margin:0 0 1rem; line-height:1.55; }
.eyebrow{ font-size:.74rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
          color:var(--violet); margin-bottom:.5rem; }

/* the example message itself */
.sample{
  background:linear-gradient(180deg,#FBFAFF,#F6F4FF);
  border:1px dashed #D9D5F5; border-left:4px solid var(--violet-2);
  border-radius:14px; padding:1rem 1.1rem; color:#3B3A52;
  font-size:.97rem; line-height:1.6; margin:.2rem 0 .5rem;
}
.sample-empty{
  background:#FAFBFF; border:1px dashed var(--line); border-radius:14px;
  padding:1.5rem 1.1rem; color:#9A99B2; font-size:.93rem; text-align:center;
}
.tagline{ color:var(--ink-soft); font-size:.85rem; margin:.55rem 0 0; }

/* ---- buttons ---- */
.stButton>button{
  border-radius:12px; border:1px solid var(--line); background:#fff;
  color:#4A4863; font-weight:600; font-size:.88rem; padding:.5rem .9rem;
  transition:all .18s ease; box-shadow:0 1px 2px rgba(80,90,160,.05);
}
.stButton>button:hover{
  border-color:var(--violet); color:var(--violet);
  transform:translateY(-1px); box-shadow:0 5px 14px rgba(108,99,255,.16);
}
.stButton>button[kind="primary"]{
  background:linear-gradient(100deg,var(--violet),var(--violet-2));
  border:none; color:#fff; font-size:.98rem; padding:.68rem 1rem;
  box-shadow:0 6px 18px rgba(108,99,255,.3);
}
.stButton>button[kind="primary"]:hover{
  filter:brightness(1.06); transform:translateY(-1px);
  box-shadow:0 9px 24px rgba(108,99,255,.38); color:#fff;
}

/* ---- textarea ---- */
.stTextArea textarea{
  background:#FCFCFF; border:1.5px solid var(--line); border-radius:14px;
  font-size:.98rem; color:var(--ink); padding:14px; line-height:1.55;
}
.stTextArea textarea::placeholder{ color:#A9A8BE; }
.stTextArea textarea:focus{
  border-color:var(--violet); box-shadow:0 0 0 4px rgba(108,99,255,.13);
}

/* ---- verdict ---- */
.verdict{
  border-radius:16px; padding:1.1rem 1.25rem; margin-top:1.1rem;
  display:flex; align-items:center; gap:.7rem; font-family:'Outfit',sans-serif;
  font-weight:700; font-size:1.15rem; animation:pop .35s ease;
}
@keyframes pop{ from{ opacity:0; transform:translateY(8px);} to{ opacity:1; transform:none;} }
.v-scam{ background:var(--rose-bg); border:1px solid var(--rose-line); color:var(--rose); }
.v-warn{ background:var(--amber-bg); border:1px solid var(--amber-line); color:var(--amber); }
.v-safe{ background:var(--mint-bg); border:1px solid var(--mint-line); color:var(--mint); }
.conf{ margin-left:auto; font-family:'Inter',sans-serif; font-size:.85rem;
       font-weight:600; opacity:.8; }

/* confidence meter */
.meter{ height:7px; border-radius:99px; background:#EFF0F8; margin-top:.75rem; overflow:hidden; }
.meter span{ display:block; height:100%; border-radius:99px; animation:grow .6s ease; }
@keyframes grow{ from{ width:0 !important; } }

.rowlabel{ color:var(--ink-soft); font-size:.87rem; margin:.9rem 0 .35rem; }
.rowlabel b{ color:var(--ink); }
.pill{
  display:inline-block; background:#F4F4FC; color:#5A5878; border:1px solid #E6E6F4;
  padding:.3rem .75rem; border-radius:999px; font-size:.8rem; font-weight:500;
  margin:.18rem .3rem .18rem 0;
}

/* custom details block — replaces st.expander so no icon font is needed */
details.why{
  margin-top:1rem; background:#FAFAFF; border:1px solid var(--line);
  border-radius:12px; padding:.15rem .9rem; font-size:.88rem;
}
details.why summary{
  cursor:pointer; padding:.65rem 0; color:var(--violet); font-weight:600;
  list-style:none; outline:none;
}
details.why summary::-webkit-details-marker{ display:none; }
details.why summary::after{ content:"  ▾"; }
details.why[open] summary::after{ content:"  ▴"; }
details.why .body{ padding:.1rem 0 .8rem; color:#54536B; line-height:1.75; }
details.why .body b{ color:var(--ink); }
.tip{ background:#F1F4FF; border:1px solid #DCE2FF; color:#4A56A6;
      padding:.6rem .8rem; border-radius:10px; margin-top:.6rem; font-size:.84rem; }

.foot{ text-align:center; color:#8C8BA6; font-size:.82rem; margin-top:2.2rem; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
  <div class="badge">English · Roman Urdu · Urdu</div>
  <h1>🛡️ Scam &amp; Fraud Detector</h1>
  <p>Not sure about a message? Check it here before you reply.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()
if model is None:
    st.error("Model file `scam_pipeline.pkl` not found. "
             "Run `python prepare_data.py`, then `python train.py`, then reload.")
    st.stop()

LANG_LABELS = {"english": "English",
               "roman_urdu": "Roman Urdu (Urdu in English letters)",
               "urdu": "Urdu (script)"}
ML_WEIGHT = {"english": 0.75, "roman_urdu": 0.35, "urdu": 0.20}

EXAMPLES = {
    "English": "CONGRATULATIONS! You have won a $1000 gift card. Click http://bit.ly/claim-now to collect your prize before it expires!",
    "Roman Urdu": "Mubarak ho! Aap ne 50 lakh ka inaam jeeta hai. Apna OTP code abhi bhejein warna offer khatam ho jayega.",
    "Urdu": "مبارک ہو! آپ نے انعام جیتا ہے۔ اپنا اکاؤنٹ نمبر اور کوڈ فوری بھیجیں۔",
    "Normal message": "Hey, are we still meeting for lunch at 1pm tomorrow? Let me know if the time changed.",
}


def analyse(message: str):
    lang = detect_language(message)
    ml_p = float(model.predict_proba([clean_text(message)])[0][1])
    rule_p, signals = rule_based_scam_score(message)
    w = ML_WEIGHT.get(lang, 0.6)
    return {"lang": lang, "ml_p": ml_p, "rule_p": rule_p, "signals": signals,
            "final": w * ml_p + (1 - w) * rule_p}


if "picked" not in st.session_state:
    st.session_state["picked"] = None

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown("""
    <div class="panel">
      <div class="eyebrow">Learn the warning signs</div>
      <h3>Want to see what a scam looks like?</h3>
      <p class="lede">Pick a language below and a real-style scam message will
      appear. Great for spotting the tricks before they reach you.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    if c1.button("English", use_container_width=True):
        st.session_state["picked"] = "English"
    if c2.button("Roman Urdu", use_container_width=True):
        st.session_state["picked"] = "Roman Urdu"
    if c3.button("Urdu", use_container_width=True):
        st.session_state["picked"] = "Urdu"
    if c4.button("Normal message", use_container_width=True):
        st.session_state["picked"] = "Normal message"

    picked = st.session_state["picked"]
    if picked:
        st.markdown(f'<div class="sample">{EXAMPLES[picked]}</div>',
                    unsafe_allow_html=True)
        note = ("This one is safe — notice there is no prize, no link and no rush."
                if picked == "Normal message" else
                "Notice the mix: a prize you never entered for, pressure to hurry, "
                "and a request for a code or link.")
        st.markdown(f'<p class="tagline">{note}</p>', unsafe_allow_html=True)
        if st.button("Try this in the checker →", use_container_width=True):
            st.session_state["msg"] = EXAMPLES[picked]
            st.rerun()
    else:
        st.markdown('<div class="sample-empty">Choose a language above to see '
                    'an example 👆</div>', unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="panel">
      <div class="eyebrow">Your message</div>
      <h3>Check if a message or email is a scam</h3>
      <p class="lede">Paste the SMS or email you received and I'll tell you how
      risky it looks, in plain language.</p>
    </div>
    """, unsafe_allow_html=True)

    message = st.text_area("Message", height=190, key="msg",
                           label_visibility="collapsed",
                           placeholder="Paste the message you received here...")
    go = st.button("Check this message", type="primary", use_container_width=True)

    if go:
        if not message.strip():
            st.info("Paste a message first, or try one of the examples on the left.")
        else:
            r = analyse(message)
            pct = r["final"] * 100
            conf = pct if r["final"] >= THRESHOLD else 100 - pct

            if r["final"] >= 0.65:
                cls, icon, word, bar = "v-scam", "🚨", "Scam detected", "var(--rose)"
            elif r["final"] >= THRESHOLD:
                cls, icon, word, bar = "v-warn", "⚠️", "Looks suspicious", "var(--amber)"
            else:
                cls, icon, word, bar = "v-safe", "✅", "Looks safe", "var(--mint)"

            st.markdown(
                f'<div class="verdict {cls}">{icon} {word}'
                f'<span class="conf">{conf:.0f}% confidence</span></div>'
                f'<div class="meter"><span style="width:{pct:.0f}%;'
                f'background:{bar};"></span></div>',
                unsafe_allow_html=True)

            st.markdown(f'<p class="rowlabel"><b>Language detected:</b> '
                        f'{LANG_LABELS.get(r["lang"], r["lang"])}</p>',
                        unsafe_allow_html=True)

            if r["signals"]:
                pills = "".join(f'<span class="pill">{s}</span>' for s in r["signals"])
                st.markdown(f'<p class="rowlabel"><b>Warning signs found</b></p>{pills}',
                            unsafe_allow_html=True)
            else:
                st.markdown('<p class="rowlabel">No common scam signals found.</p>',
                            unsafe_allow_html=True)

            tip = ("" if r["lang"] == "english" else
                   '<div class="tip">For Urdu and Roman Urdu the keyword rules '
                   'count for more, because the trained model saw mostly English '
                   'data.</div>')
            st.markdown(f"""
            <details class="why">
              <summary>Why this score?</summary>
              <div class="body">
                Trained model &mdash; chance of scam: <b>{r['ml_p']*100:.0f}%</b><br>
                Keyword &amp; pattern rules: <b>{r['rule_p']*100:.0f}%</b><br>
                Combined score: <b>{pct:.0f}%</b> (flagged above {THRESHOLD*100:.0f}%)
                {tip}
              </div>
            </details>
            """, unsafe_allow_html=True)

st.markdown('<p class="foot">Educational project — not a guarantee. '
            'When in doubt, never share OTPs, PINs or passwords, and verify '
            'through official channels.</p>', unsafe_allow_html=True)
