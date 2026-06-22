"""
====================================================
  APLIKASI ANALISIS SENTIMEN - STREAMLIT
  Multi-Model Manager: muat banyak model sekaligus,
  ganti-ganti dengan mudah tanpa reload ulang.
====================================================
"""

import streamlit as st
import os, json, time
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Stance Classification", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0f0c29 100%); color: #e8e8f0; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.08); }
.result-positive { background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08)); border: 1px solid rgba(16,185,129,0.3); border-radius: 16px; padding: 28px; text-align: center; animation: fadeIn 0.5s ease; }
.result-negative { background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.08)); border: 1px solid rgba(239,68,68,0.3); border-radius: 16px; padding: 28px; text-align: center; animation: fadeIn 0.5s ease; }
.result-neutral  { background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(180,83,9,0.08)); border: 1px solid rgba(245,158,11,0.3); border-radius: 16px; padding: 28px; text-align: center; animation: fadeIn 0.5s ease; }
.big-emoji    { font-size: 64px; display: block; margin-bottom: 8px; }
.result-label { font-size: 28px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; }
.result-conf  { font-size: 14px; opacity: 0.7; margin-top: 6px; font-family: 'JetBrains Mono', monospace; }
.conf-bar-wrap { background: rgba(255,255,255,0.08); border-radius: 99px; height: 10px; margin: 8px 0; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 99px; transition: width 0.8s ease; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.badge-ok   { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-err  { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-warn { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.model-card        { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 10px 14px; margin: 5px 0; font-size: 13px; }
.model-card-active { background: rgba(124,58,237,0.2); border: 1px solid rgba(124,58,237,0.5); border-radius: 12px; padding: 10px 14px; margin: 5px 0; font-size: 13px; }
.model-name { font-weight: 700; font-size: 14px; margin-bottom: 2px; }
.model-type { opacity: 0.5; font-size: 11px; font-family: 'JetBrains Mono', monospace; }
code { font-family: 'JetBrains Mono', monospace; background: rgba(255,255,255,0.07); padding: 2px 6px; border-radius: 4px; font-size: 13px; color: #a78bfa; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: rgba(255,255,255,0.5); font-weight: 600; }
.stTabs [aria-selected="true"] { background: rgba(139,92,246,0.3); color: #c4b5fd; }
.stTextArea textarea { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 12px !important; color: #e8e8f0 !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stButton > button { background: linear-gradient(135deg, #7c3aed, #6d28d9) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 10px 24px !important; font-weight: 700 !important; font-size: 14px !important; width: 100%; transition: all 0.2s !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 24px rgba(124,58,237,0.4) !important; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
.app-header { text-align: center; padding: 28px 0 20px; }
.app-title  { font-size: 40px; font-weight: 800; background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; }
.app-sub { color: rgba(255,255,255,0.4); font-size: 14px; margin-top: 4px; }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  LOAD FUNCTIONS (cached per unique args)
# ═══════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_sklearn_model(model_path, vectorizer_path=""):
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    vectorizer = None
    if vectorizer_path and os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
    return model, vectorizer

@st.cache_resource(show_spinner=False)
def load_huggingface_model(model_path, tokenizer_path=""):
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    tok_path  = tokenizer_path if (tokenizer_path and tokenizer_path.strip()) else model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

@st.cache_resource(show_spinner=False)
def load_keras_model(model_path, tokenizer_path="", label_path=""):
    from tensorflow import keras
    import pickle
    model = keras.models.load_model(model_path)
    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    labels = ["MENOLAK","MENDUKUNG"]
    if label_path and os.path.exists(label_path):
        with open(label_path) as f:
            labels = json.load(f)
    return model, tokenizer, labels


# ═══════════════════════════════════════════════
#  PREDICT FUNCTIONS
# ═══════════════════════════════════════════════

def predict_sklearn(text, model, vectorizer):
    X = vectorizer.transform([text]) if vectorizer else [text]
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return str(pred), proba

def predict_huggingface(text, clf):
    raw = clf(text)
    print("TEXT:", text)
    print("RAW OUTPUT:", raw)
    if isinstance(raw, list) and len(raw) > 0:
        results = raw[0] if isinstance(raw[0], list) else raw
    else:
        raise ValueError(f"Output pipeline tidak valid: {raw}")
    best   = max(results, key=lambda x: x["score"])
    scores = {r["label"]: r["score"] for r in results}
    print("RESULTS:", results)
    print("BEST:", best)
    print("SCORES:", scores)
    return best["label"], scores

def predict_keras(text, model, tokenizer, labels, max_len=128):
    import tensorflow as tf
    if not tokenizer:
        return "Unknown", None
    seq   = tokenizer.texts_to_sequences([text])
    X     = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    proba = model.predict(X)[0]
    return labels[int(np.argmax(proba))], proba.tolist()

def run_predict(text, entry):
    mtype = entry["type"]
    obj   = entry["obj"]
    if "sklearn" in mtype:
        model, vec   = obj
        label, proba = predict_sklearn(text, model, vec)
        conf         = float(max(proba)) if proba is not None else None
        all_scores   = dict(zip([str(c) for c in model.classes_], proba.tolist())) if proba is not None else None
    elif "HuggingFace" in mtype or "IndoBERT" in mtype:
        label, all_scores = predict_huggingface(text, obj)
        conf              = all_scores.get(label)
        print("FINAL LABEL:", label)
        print("FINAL CONF:", conf)
    else:
        mdl, tok, lbl  = obj
        label, proba   = predict_keras(text, mdl, tok, lbl)
        conf           = float(max(proba)) if proba else None
        all_scores     = dict(zip(lbl, proba)) if proba else None
    return label, conf, all_scores


# ═══════════════════════════════════════════════
#  RENDER HELPERS
# ═══════════════════════════════════════════════

SENTIMENT_MAP = {
    "label_1": ("👍", "support", "MENDUKUNG"),
    "label_0": ("👎", "against", "MENOLAK"),
}

def map_sentiment(raw_label):
    return SENTIMENT_MAP.get(raw_label.lower().strip(), ("🤔","neutral",raw_label.upper()))

def render_result(label, confidence=None, all_scores=None):
    emoji, css_class, display = map_sentiment(str(label))
    conf_str = f"{confidence*100:.1f}%" if confidence is not None else "—"

# tentukan warna DI LUAR markdown
    color = "#ef4444" if display == "MENOLAK" else "#10b981"

    st.markdown(f"""
    <div class="result-{css_class}">
        <span class="big-emoji">{emoji}</span>
        <div class="result-label" style="color:{color};">{display}</div>
        <div class="result-conf">Confidence: {conf_str}</div>
    </div>
    """, unsafe_allow_html=True)
    if all_scores is not None:
        st.markdown("---")
        st.markdown("**Distribusi Skor**")
        items = all_scores.items() if isinstance(all_scores, dict) else [(f"Kelas {i}", float(s)) for i,s in enumerate(all_scores)]
        bar_colors = {"positive":"#10b981","negative":"#ef4444","neutral":"#f59e0b"}
        for lbl, score in items:
            e2, cc2, _ = map_sentiment(str(lbl))
            pct = float(score)*100
            color = bar_colors.get(cc2,"#8b5cf6")
            st.markdown(f"""
            <div style="margin:6px 0;">
                <div style="display:flex;justify-content:space-between;font-size:13px;opacity:.75;margin-bottom:3px;">
                    <span>{e2} {lbl}</span>
                    <span style="font-family:'JetBrains Mono',monospace">{pct:.1f}%</span>
                </div>
                <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div></div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════
if "model_registry" not in st.session_state: st.session_state.model_registry = {}
if "active_model"   not in st.session_state: st.session_state.active_model   = None
if "history"        not in st.session_state: st.session_state.history        = []


# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🗂️ Model Manager")
    st.markdown("---")


    # Daftar model yang sudah dimuat
    if st.session_state.model_registry:
        st.markdown("### ✅ Model Tersimpan")
        for nama, entry in list(st.session_state.model_registry.items()):
            is_active  = (nama == st.session_state.active_model)
            card_class = "model-card-active" if is_active else "model-card"
            active_tag = " 🟢" if is_active else ""
            st.markdown(f"""
            <div class="{card_class}">
                <div class="model-name">{nama}{active_tag}</div>
                <div class="model-type">{entry['type']}</div>
            </div>""", unsafe_allow_html=True)
            col_use, col_del = st.columns([2,1])
            with col_use:
                if not is_active:
                    if st.button(f"▶ Gunakan", key=f"use_{nama}"):
                        st.session_state.active_model = nama
                        st.rerun()
                else:
                    st.markdown('<span class="badge badge-ok">● AKTIF</span>', unsafe_allow_html=True)
            with col_del:
                if st.button("🗑", key=f"del_{nama}", help=f"Hapus {nama}"):
                    del st.session_state.model_registry[nama]
                    remaining = list(st.session_state.model_registry.keys())
                    st.session_state.active_model = remaining[0] if remaining else None
                    st.rerun()
        st.markdown("---")

# ===== AUTO LOAD SEMUA MODEL =====
st.markdown("### 🤖 Daftar Model (Auto Loaded)")

MODEL_LIST = {
    "IndoBERTweet-BackTranslate": {
        "model_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERTweet BT\drive-download-20260527T154539Z-3-001\model",
        "tokenizer_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERTweet BT\drive-download-20260527T154539Z-3-001\tokenizer"
    },
    "IndoBERTweet-Parafrase": {
        "model_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERTTWEET parafrase\drive-download-20260527T163452Z-3-001\model",
        "tokenizer_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERTTWEET parafrase\drive-download-20260527T163452Z-3-001\tokenizer"
    },
    "IndoBERTweet-Undersampling": {
        "model_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERTWEET Undersampling\drive-download-20260507T163711Z-3-001\model",
        "tokenizer_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERTWEET Undersampling\drive-download-20260507T163711Z-3-001\tokenizer"
    },
     "IndoBERTweet-Class Weight": {
        "model_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERTWEET Class Weight\drive-download-20260619T145047Z-3-001\model",
        "tokenizer_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERTWEET Class Weight\drive-download-20260619T145047Z-3-001\tokenizer"
    },
    "IndoBERT-BackTranslate": {
        "model_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERT Oversampling BT 0.65\drive-download-20260515T081333Z-3-001\model",
        "tokenizer_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERT Oversampling BT 0.65\drive-download-20260515T081333Z-3-001\tokenizer"
    },
    "IndoBERT-Parafrase": {
        "model_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERT Oversampling Parafrase 0.65\drive-download-20260515T092606Z-3-001\model",
        "tokenizer_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERT Oversampling Parafrase 0.65\drive-download-20260515T092606Z-3-001\tokenizer"
    },
    "IndoBERT-Undersampling": {
        "model_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERT Undersampling\drive-download-20260515T061539Z-3-001\model",
        "tokenizer_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERT Undersampling\drive-download-20260515T061539Z-3-001\tokenizer"
    },
     "IndoBERT-Class Weight": {
        "model_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERT Class Weight\drive-download-20260619T154216Z-3-001\model",
        "tokenizer_path": r"C:\Users\bima\OneDrive\Documents\Kuliah\Tugas Akhir\Streamlit TA\INDOBERT Class Weight\drive-download-20260619T154216Z-3-001\tokenizer"
    }
}

# init session state
if "model_registry" not in st.session_state:
    st.session_state.model_registry = {}

if "active_model" not in st.session_state:
    st.session_state.active_model = None

# load semua model (sekali saja)
if len(st.session_state.model_registry) == 0:
    with st.spinner("🔄 Memuat semua model..."):
        for name, paths in MODEL_LIST.items():
            try:
                obj = load_huggingface_model(paths["model_path"], paths["tokenizer_path"])
                st.session_state.model_registry[name] = {
                    "type": "HuggingFace / IndoBERT / IndoBERTweet",
                    "obj": obj,
                    "path": paths["model_path"]
                }
            except Exception as e:
                st.error(f"❌ Gagal load {name}: {e}")

    # set default model pertama
    if st.session_state.model_registry:
        st.session_state.active_model = list(st.session_state.model_registry.keys())[0]

# tampilkan model
st.write("### 📌 Model tersedia:")
for m in st.session_state.model_registry.keys():
    st.write(f"- {m}")


# ═══════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="app-title">🎯 Stance Classification </div>
    <div class="app-sub">Multi-Model · Ganti model kapan saja tanpa reload</div>
</div>
""", unsafe_allow_html=True)

# ── SWITCHER MODEL AKTIF ──
if st.session_state.model_registry:
    model_names = list(st.session_state.model_registry.keys())
    if st.session_state.active_model not in model_names:
        st.session_state.active_model = model_names[0]

    col_sw, col_info = st.columns([2,3])
    with col_sw:
        chosen = st.selectbox("🔄 Model yang digunakan:", model_names,
                              index=model_names.index(st.session_state.active_model),
                              key="model_switcher")
        if chosen != st.session_state.active_model:
            st.session_state.active_model = chosen
            st.rerun()
    with col_info:
        entry = st.session_state.model_registry[st.session_state.active_model]
        short_path = entry['path'][:65] + ("..." if len(entry['path']) > 65 else "")
        st.markdown(f"""
            <div style="padding:10px 0 0 8px;font-size:13px;opacity:.7;">
                <span class="badge badge-ok">● AKTIF</span>&nbsp;
                <code>{st.session_state.active_model}</code>
            </div>""", unsafe_allow_html=True)
        st.markdown("---")
else:
    st.info("⬅️ Belum ada model. Tambahkan model di sidebar kiri terlebih dahulu.")

active_entry = st.session_state.model_registry.get(st.session_state.active_model)

tab_single, tab_batch, tab_history= st.tabs([
    "✍️  Teks Tunggal", "📋  Batch Teks", "🕐  Riwayat"
])


# ── TAB 1 ──
with tab_single:
    col_in, col_out = st.columns([1,1], gap="large")
    with col_in:
        st.markdown("#### Input Teks")
        user_text = st.text_area("", placeholder="Ketik atau tempel teks di sini...",
                                  height=180, label_visibility="collapsed")
        analyze_btn = st.button("🔍 Klasifikasi Teks", key="btn_single")
    with col_out:
        st.markdown("#### Hasil")
        if analyze_btn:
            if not active_entry:
                st.warning("⚠️ Muat model terlebih dahulu di sidebar.")
            elif not user_text.strip():
                st.warning("⚠️ Teks tidak boleh kosong.")
            else:
                with st.spinner("Menganalisis..."):
                    time.sleep(0.2)
                    try:
                        label, conf, all_scores = run_predict(user_text, active_entry)
                        render_result(label, conf, all_scores)
                        st.session_state.history.append({
                            "model": st.session_state.active_model,
                            "teks" : user_text[:80] + ("..." if len(user_text)>80 else ""),
                            "label": label,
                            "conf" : f"{conf*100:.1f}%" if conf else "—",
                        })
                    except Exception as e:
                        st.error(f"❌ Prediksi gagal: {e}")
        else:
            st.markdown("""<div style="height:220px;display:flex;align-items:center;
                justify-content:center;opacity:.3;font-size:48px;">🎯</div>""", unsafe_allow_html=True)

# ── TAB 2 ──
with tab_batch:
    st.markdown("#### Input Batch (satu kalimat per baris)")
    batch_text = st.text_area("", height=200, placeholder="Teks pertama\nTeks kedua\nTeks ketiga...",
                               label_visibility="collapsed")
    if st.button("🔍 Klasifikasi Semua Teks", key="btn_batch"):
        if not active_entry:
            st.warning("⚠️ Muat model terlebih dahulu.")
        else:
            lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
            if not lines:
                st.warning("Tidak ada teks yang dimasukkan.")
            else:
                results, prog = [], st.progress(0)
                for i, line in enumerate(lines):
                    try:
                        label, conf, _ = run_predict(line, active_entry)
                        e, _, disp = map_sentiment(str(label))
                        results.append({"Model":st.session_state.active_model,"Teks":line,
                                        "Sikap":f"{e} {disp}","Confidence":f"{conf*100:.1f}%" if conf else "—"})
                    except Exception as ex:
                        results.append({"Model":st.session_state.active_model,"Teks":line,"Sentimen":"ERROR","Confidence":str(ex)})
                    prog.progress((i+1)/len(lines))
                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "hasil_sentimen.csv","text/csv")

# ── TAB 3 ──
with tab_history:
    st.markdown("#### Riwayat Analisis (sesi ini)")
    if not st.session_state.history:
        st.info("Belum ada analisis yang dilakukan.")
    else:
        import pandas as pd
        st.dataframe(pd.DataFrame(st.session_state.history[::-1]), use_container_width=True)
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()