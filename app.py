import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------------- Page Config ---------------- #

st.set_page_config(
    page_title="EmotiSense · NLP Emotion Detector",
    page_icon="🧠",
    layout="centered",
)

# ---------------- LOAD FILES ---------------- #

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "LR1.joblib")
vectorizer = joblib.load(BASE_DIR / "Count_Vectorizer.joblib")

# ---------------- Gradient UI ---------------- #

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"]{
    background: linear-gradient(
        135deg,
        #050510 0%,
        #09091a 40%,
        #060b1d 100%
    ) !important;
}

[data-testid="stHeader"]{
    background: transparent;
}

*{
    font-family:'DM Sans',sans-serif;
}

h1,h2,h3{
    font-family:'Syne',sans-serif;
}

.block-container{
    max-width:850px;
    margin:auto;
    padding-top:2rem;
}

/* HERO */

.hero{
    text-align:center;
    padding-top:4rem;
    padding-bottom:2rem;
}

.hero-badge{
    display:inline-block;
    padding:8px 18px;
    border-radius:999px;
    border:1px solid rgba(167,139,250,0.35);
    background:rgba(124,58,237,0.08);
    color:#b794f4;
    font-size:12px;
    letter-spacing:2px;
    margin-bottom:20px;
}

.hero-title{
    font-family:'Syne',sans-serif;
    font-size:clamp(4rem,7vw,6rem);
    font-weight:700;

    background:linear-gradient(
        90deg,
        #ffffff,
        #c4b5fd,
        #38bdf8,
        #ffffff
    );

    background-size:300% auto;

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    animation:gradientFlow 6s linear infinite;
}

@keyframes gradientFlow{
    0%{
        background-position:0% center;
    }
    100%{
        background-position:300% center;
    }
}

.hero-sub{
    max-width:600px;
    margin:0 auto 1.5rem;
    line-height:1.8;
}

.stat-row{
    display:flex;
    justify-content:center;
    gap:12px;
    flex-wrap:wrap;
    margin-top:1.7rem;
    margin-bottom:2rem;
}

.stat-pill{
    padding:8px 15px;
    border-radius:999px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    color:#94a3b8;
    font-size:0.8rem;
}

/* LABEL */

/* =========================
   TEXTAREA SECTION
========================= */

.stTextArea{
    margin-top:1rem;
}

.stTextArea label{
    display:block !important;

    color:#a78bfa !important;
    font-size:0.8rem !important;
    font-weight:700 !important;

    letter-spacing:0.18em;
    text-transform:uppercase;

    margin-bottom:0.8rem !important;
}

/* Text Area */

.stTextArea textarea{
    background:rgba(8,10,25,0.88) !important;

    border:1.5px solid rgba(124,58,237,0.25) !important;
    border-radius:20px !important;

    color:#f8fafc !important;

    padding:1.2rem !important;

    font-size:1rem !important;
    line-height:1.7 !important;

    transition:all .3s ease;

    backdrop-filter:blur(14px);
}

/* Hover */

.stTextArea textarea:hover{
    border-color:rgba(167,139,250,0.45) !important;

    box-shadow:
        0 0 25px rgba(124,58,237,0.08);
}

/* Focus */

.stTextArea textarea:focus{
    border-color:#8b5cf6 !important;

    box-shadow:
        0 0 0 3px rgba(124,58,237,.12),
        0 0 40px rgba(124,58,237,.18) !important;
}

/* Placeholder */

.stTextArea textarea::placeholder{
    color:#64748b !important;
}


/* =========================
   BUTTON
========================= */

.stButton{
    margin-top:1.2rem;
}

.stButton > button{
    width:100%;

    height:68px;

    border:none !important;
    border-radius:20px !important;

    color:white !important;

    font-size:1.15rem !important;
    font-weight:700 !important;

    background:linear-gradient(
        90deg,
        #7c3aed 0%,
        #6366f1 50%,
        #0ea5e9 100%
    ) !important;

    position:relative;
    overflow:hidden;

    transition:all .35s ease;
}

/* Shine Effect */

.stButton > button::before{
    content:"";

    position:absolute;

    top:0;
    left:-120%;

    width:80%;
    height:100%;

    background:linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.18),
        transparent
    );

    transition:.8s;
}

.stButton > button:hover::before{
    left:130%;
}

/* Hover */

.stButton > button:hover{
    transform:translateY(-3px);

    box-shadow:
        0 12px 35px rgba(124,58,237,.35),
        0 0 50px rgba(56,189,248,.15);
}

/* Click */

.stButton > button:active{
    transform:translateY(-1px);
}
            
st.button("✨ Analyse Emotion")
            
/* RESULT CARD */

.result-card{
    text-align:center !important;

    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;

    padding:2.5rem 2rem;
    margin-top:2rem;
}
            
.result-label{
    text-align:center !important;
}

.result-emoji{
    text-align:center !important;
    font-size:5rem;
    margin-bottom:1rem;
}

.result-emotion{
    text-align:center !important;
    font-size:4rem;
    font-weight:800;
    margin:0.5rem 0;
}

.result-confidence{
    text-align:center !important;
    font-size:1.1rem;
    opacity:0.8;
}     
st.progress(confidence)

/* EMOTIONS */

.emotion-joy{
    background:#2a2008;
    border:1px solid #f59e0b44;
    color:#facc15;
}

.emotion-sadness{
    background:#081224;
    border:1px solid #3b82f644;
    color:#93c5fd;
}

.emotion-anger{
    background:#240808;
    border:1px solid #ef444444;
    color:#fca5a5;
}

.emotion-fear{
    background:#170824;
    border:1px solid #8b5cf644;
    color:#c4b5fd;
}

.emotion-surprise{
    background:#08201a;
    border:1px solid #10b98144;
    color:#6ee7b7;
}

.emotion-disgust{
    background:#162108;
    border:1px solid #84cc1644;
    color:#bef264;
}

.emotion-neutral{
    background:#151515;
    border:1px solid #47556944;
    color:#cbd5e1;
}

.emotion-love{
    background:#240814;
    border:1px solid #ec489944;
    color:#f9a8d4;
}

.footer{
    text-align:center;
    color:#475569;
    margin-top:40px;
    font-size:13px;
}

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ------------- HERO SECTION -------------- #
st.markdown("""
<div class="hero">

<div class="hero-badge">
🧠 NLP · EMOTION INTELLIGENCE
</div>

<div class="hero-title">
EmotiSense
</div>

<div class="hero-sub">
Drop any text below and watch AI decode the emotion hiding inside your words.
</div>

<div class="stat-row">
<span class="stat-pill">⚡ Logistic Regression</span>
<span class="stat-pill">📊 ~88% Accuracy</span>
<span class="stat-pill">🔤 CountVectorizer</span>
</div>

</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- Emotion Config ---------------- #

LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

EMOTION_META = {
    "joy":      {"emoji": "😄", "css": "c-joy",      "label": "Joy"},
    "sadness":  {"emoji": "😢", "css": "c-sadness",  "label": "Sadness"},
    "anger":    {"emoji": "😡", "css": "c-anger",    "label": "Anger"},
    "fear":     {"emoji": "😨", "css": "c-fear",     "label": "Fear"},
    "surprise": {"emoji": "😲", "css": "c-surprise", "label": "Surprise"},
    "disgust":  {"emoji": "🤢", "css": "c-disgust",  "label": "Disgust"},
    "love":     {"emoji": "❤️", "css": "c-love",     "label": "Love"},
    "neutral":  {"emoji": "😐", "css": "c-neutral",  "label": "Neutral"},
}

def resolve(raw) -> str:
    try:
        idx = int(raw)
        return LABEL_MAP.get(idx, f"label_{idx}").lower()
    except (ValueError, TypeError):
        return str(raw).lower().strip()

def get_meta(raw) -> dict:
    name = resolve(raw)
    return EMOTION_META.get(name, {"emoji": "🤔", "css": "c-neutral", "label": name.title()})
 
# ------------------ Input ------------------ #

# ── Input ──────────────────────────────────────────────────────────────────────
user_text   = st.text_area("YOUR TEXT",
                placeholder="e.g. I just got the job offer — I can't believe it's actually happening!",
                height=140)
predict_btn = st.button("✦ Analyse Emotion", use_container_width=True)

# ------------------ Prediction ------------------ #

if predict_btn:
    text = user_text.strip()
    if not text:
        st.markdown('<p style="text-align:center;color:#ef4444;margin-top:1rem;">⚠️ Please enter some text first.</p>',
                    unsafe_allow_html=True)
    else:
        with st.spinner("Reading between the lines…"):
            vec     = vectorizer.transform([text])
            emotion = model.predict(vec)[0]
            proba   = model.predict_proba(vec)[0]
            conf    = round(max(proba) * 100, 1)
 
        meta = get_meta(str(emotion))
 
        st.markdown(f"""
        <div class="result-card {meta['css']}">
            <div class="result-label">Detected Emotion</div>
            <div class="result-emoji">{meta['emoji']}</div>
            <div class="result-emotion">{meta['label']}</div>
            <div class="result-confidence">Confidence · {conf}%</div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#475569;font-size:0.8rem;text-align:center;margin-bottom:0.4rem;">MODEL CONFIDENCE</p>',
                    unsafe_allow_html=True)
        st.progress(int(conf))
 
        # Top-3 breakdown
        classes = model.classes_
        top3 = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:3]
        st.markdown('<p style="color:#334155;font-size:0.78rem;text-align:center;margin:1rem 0 0.5rem;">TOP PREDICTIONS</p>',
                    unsafe_allow_html=True)
        for emo, prob in top3:
            m   = get_meta(str(emo))
            pct = round(prob * 100, 1)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.8rem;margin:0.4rem 0;padding:0.5rem 0.8rem;
                        background:rgba(255,255,255,0.025);border-radius:10px;border:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:1.2rem;">{m['emoji']}</span>
                <span style="color:#94a3b8;font-size:0.85rem;width:80px;">{m['label']}</span>
                <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:999px;height:6px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#7c3aed,#0ea5e9);border-radius:999px;"></div>
                </div>
                <span style="color:#64748b;font-size:0.78rem;width:40px;text-align:right;">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

# ---------------- Footer ----------------#
st.markdown("""
<div class="divider" style="margin-top:3rem;"></div>
<div class="footer">Built with Streamlit · Logistic Regression · CountVectorizer<br>
<span style="opacity:0.5;">EmotiSense NLP · 2026</span></div>
""", unsafe_allow_html=True)


#python3 -m streamlit run app.py