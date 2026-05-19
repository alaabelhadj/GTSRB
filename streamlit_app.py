"""
Application Streamlit v2 — Reconnaissance de Panneaux GTSRB
Fonctionnalités : Prédiction, Dessin, Batch, Comparaison, Grad-CAM, Conditions réelles, Historique
"""

import streamlit as st
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import io
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GTSRB — Reconnaissance de Panneaux",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:700; color:#1a1a2e; text-align:center; padding:1rem 0 0.5rem; }
    .subtitle   { text-align:center; color:#555; font-size:1.05rem; margin-bottom:2rem; }
    .metric-card { background:linear-gradient(135deg,#667eea,#764ba2); border-radius:12px;
                   padding:1.2rem; color:white; text-align:center; margin-bottom:1rem; }
    .metric-value { font-size:2rem; font-weight:700; }
    .metric-label { font-size:.85rem; opacity:.85; }
    .info-box   { background:#ebf8ff; border:2px solid #4299e1; border-radius:10px;
                  padding:1rem; margin:.5rem 0; color:#1a1a2e; }
    .warning-box { background:#fffaf0; border:2px solid #ed8936; border-radius:10px;
                   padding:1rem; color:#1a1a2e; }
    .pred-box   { border-radius:12px; padding:1.2rem; text-align:center; margin:.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0:'Limite de vitesse (20 km/h)', 1:'Limite de vitesse (30 km/h)',
    2:'Limite de vitesse (50 km/h)', 3:'Limite de vitesse (60 km/h)',
    4:'Limite de vitesse (70 km/h)', 5:'Limite de vitesse (80 km/h)',
    6:'Fin de la limite 80 km/h',   7:'Limite de vitesse (100 km/h)',
    8:'Limite de vitesse (120 km/h)',9:'Dépassement interdit',
    10:'Dépassement interdit (camions)', 11:'Priorité au prochain croisement',
    12:'Route prioritaire', 13:'Cédez le passage', 14:'STOP',
    15:'Circulation interdite', 16:'Accès interdit aux camions',
    17:'Sens interdit', 18:'Danger général', 19:'Virage à gauche',
    20:'Virage à droite', 21:'Double virage', 22:'Route bosselée',
    23:'Route glissante', 24:'Rétrécissement de chaussée', 25:'Travaux',
    26:'Signaux lumineux', 27:'Piétons', 28:'Attention enfants',
    29:'Cyclistes', 30:'Verglas / neige', 31:'Animaux sauvages',
    32:'Fin de toutes restrictions', 33:'Tourner à droite obligatoire',
    34:'Tourner à gauche obligatoire', 35:'Tout droit obligatoire',
    36:'Tout droit ou tourner à droite', 37:'Tout droit ou tourner à gauche',
    38:'Serrez à droite', 39:'Serrez à gauche', 40:'Rond-point obligatoire',
    41:'Fin du dépassement interdit', 42:'Fin dépassement interdit (camions)'
}

CLASS_EMOJIS = {
    **{i:'🔵' for i in range(9)}, 9:'⛔', 10:'⛔',
    11:'⚠️', 12:'⚠️', 13:'🔺', 14:'🛑',
    15:'🚫', 16:'🚫', 17:'🚫',
    **{i:'⚠️' for i in range(18, 32)}, 32:'✅',
    **{i:'💙' for i in range(33, 43)}
}

IMG_SIZE = (64, 64)

# ─────────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        import tensorflow as tf
        tl  = tf.keras.models.load_model("gtsrb_efficientnet_final.keras")
        cnn = tf.keras.models.load_model("gtsrb_cnn_final.keras")
        return tl, cnn, None
    except Exception as e:
        return None, None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# Prétraitement
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_tl(img: Image.Image) -> np.ndarray:
    """EfficientNetB0 : pixels bruts [0-255] — normalisation interne au modèle."""
    return np.array(img.convert('RGB').resize(IMG_SIZE), dtype=np.float32).reshape(1, 64, 64, 3)

def preprocess_cnn(img: Image.Image) -> np.ndarray:
    """CNN scratch : normalisation [0-1]."""
    return (np.array(img.convert('RGB').resize(IMG_SIZE), dtype=np.float32) / 255.0).reshape(1, 64, 64, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Prédiction
# ─────────────────────────────────────────────────────────────────────────────
def predict_top5(model, arr: np.ndarray):
    probs    = model.predict(arr, verbose=0)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    return probs, top5_idx, probs[top5_idx]

# ─────────────────────────────────────────────────────────────────────────────
# Affichage résultat
# ─────────────────────────────────────────────────────────────────────────────
def render_result_box(pred_class, pred_prob):
    emoji = CLASS_EMOJIS.get(pred_class, '🚦')
    color = "#48bb78" if pred_prob >= 0.7 else "#ed8936" if pred_prob >= 0.4 else "#e53e3e"
    st.markdown(f"""
    <div class='pred-box' style='background:{color}20;border:2px solid {color};'>
        <div style='font-size:2.5rem'>{emoji}</div>
        <div style='font-size:1.2rem;font-weight:700;color:#1a1a2e;margin:.4rem 0'>{CLASS_NAMES[pred_class]}</div>
        <div style='font-size:1.8rem;font-weight:700;color:{color}'>{pred_prob*100:.1f}%</div>
        <div style='color:#666;font-size:.85rem'>Confiance</div>
    </div>""", unsafe_allow_html=True)
    if pred_prob >= 0.9:
        st.success(f"✅ Très haute confiance ({pred_prob*100:.1f}%)")
    elif pred_prob >= 0.7:
        st.info(f"🔵 Haute confiance ({pred_prob*100:.1f}%)")
    elif pred_prob >= 0.4:
        st.warning(f"⚠️ Confiance modérée ({pred_prob*100:.1f}%)")
    else:
        st.error(f"❌ Faible confiance ({pred_prob*100:.1f}%) — Image difficile")

def plot_top5(top5_idx, top5_prob, title=""):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    labels = [CLASS_NAMES[i][:24] for i in top5_idx]
    colors = ['#2ecc71' if i == 0 else '#95a5a6' for i in range(5)]
    ax.barh(range(5), top5_prob * 100, color=colors, edgecolor='white')
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Probabilité (%)')
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    for i, v in enumerate(top5_prob):
        ax.text(v * 100 + 1, i, f'{v*100:.1f}%', va='center', fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Historique
# ─────────────────────────────────────────────────────────────────────────────
def add_history(source: str, file_name: str, class_name: str, confidence: float):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Horodatage":     datetime.now().strftime("%H:%M:%S"),
        "Source":         source,
        "Fichier":        file_name,
        "Classe prédite": class_name,
        "Confiance (%)":  round(confidence * 100, 1)
    })

# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM (gradient de saillance w.r.t. image d'entrée)
# ─────────────────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_array_raw: np.ndarray, class_idx: int):
    """
    Calcule un gradient de saillance par rapport à l'image d'entrée.
    Fonctionne avec les modèles imbriqués (EfficientNetB0 nested).
    """
    try:
        import tensorflow as tf
        img_var = tf.Variable(tf.cast(img_array_raw, tf.float32))
        with tf.GradientTape() as tape:
            preds = model(img_var, training=False)
            score = preds[:, class_idx]
        grads    = tape.gradient(score, img_var)[0]          # (64,64,3)
        heatmap  = tf.reduce_max(tf.abs(grads), axis=-1).numpy()  # (64,64)
        heatmap  = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap
    except Exception:
        return None

def overlay_heatmap(img_pil: Image.Image, heatmap: np.ndarray) -> Image.Image:
    h_small = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img_pil.size, Image.BILINEAR)
    colormap = mcm.get_cmap('jet')
    h_rgb = (colormap(np.array(h_small) / 255.0)[:, :, :3] * 255).astype(np.uint8)
    blended = Image.blend(img_pil.convert('RGB'), Image.fromarray(h_rgb), alpha=0.4)
    return blended

# ─────────────────────────────────────────────────────────────────────────────
# Init session_state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

tl_model, cnn_model, model_error = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("### 📊 Métriques des modèles")

    st.markdown("""
    <div class='metric-card'>
        <div class='metric-value'>~97%</div>
        <div class='metric-label'>EfficientNetB0 (Transfer Learning)</div>
    </div>
    <div class='metric-card' style='background:linear-gradient(135deg,#f093fb,#f5576c)'>
        <div class='metric-value'>~85%</div>
        <div class='metric-label'>CNN from scratch</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ Dataset")
    st.markdown("**GTSRB** | 43 classes | ~39 000 images")
    if st.session_state.history:
        st.markdown(f"---\n📜 **{len(st.session_state.history)}** prédiction(s) effectuée(s)")
    st.markdown("---\n*Paris School of Technology & Business*")

# ─────────────────────────────────────────────────────────────────────────────
# Titre
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-title'>🚦 Reconnaissance de Panneaux GTSRB</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>CNN from scratch & Transfer Learning EfficientNetB0 — 43 classes</p>", unsafe_allow_html=True)

if model_error:
    st.error(f"⚠️ Impossible de charger les modèles : {model_error}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Onglets
# ─────────────────────────────────────────────────────────────────────────────
(tab_predict, tab_draw, tab_batch,
 tab_compare, tab_gcam, tab_conditions, tab_history) = st.tabs([
    "🔮 Prédiction", "✏️ Dessiner", "📁 Batch",
    "⚔️ Comparaison", "🔥 Grad-CAM", "🌦️ Conditions réelles", "📜 Historique"
])

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("### 📤 Charger une image")
        uploaded = st.file_uploader("Déposez une image de panneau",
                                    type=["jpg","jpeg","png","ppm","bmp"],
                                    key="predict_upload")
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Image chargée", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Largeur", f"{image.size[0]} px")
            c2.metric("Hauteur", f"{image.size[1]} px")
            c3.metric("Mode", image.mode)

    with col_res:
        st.markdown("### 🎯 Résultat (EfficientNetB0)")
        if uploaded:
            with st.spinner("Analyse..."):
                arr = preprocess_tl(image)
                probs, top5_idx, top5_prob = predict_top5(tl_model, arr)
            pred_class, pred_prob = int(top5_idx[0]), float(top5_prob[0])
            render_result_box(pred_class, pred_prob)
            st.markdown("#### 🏆 Top 5")
            fig = plot_top5(top5_idx, top5_prob)
            st.pyplot(fig); plt.close()
            add_history("🔮 Prédiction", uploaded.name, CLASS_NAMES[pred_class], pred_prob)
        else:
            st.markdown("""
            <div class='info-box'>
                <b>Instructions :</b><br>
                1. Chargez une image de panneau<br>
                2. La prédiction s'affiche automatiquement<br>
                3. Le top-5 montre les 5 classes les plus probables
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — DESSIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_draw:
    st.markdown("### ✏️ Dessinez votre panneau à la main")
    try:
        from streamlit_drawable_canvas import st_canvas

        col_can, col_draw_res = st.columns([1, 1], gap="large")

        with col_can:
            st.markdown("**Dessinez sur le canvas ci-dessous :**")
            brush = st.slider("Taille du pinceau", 5, 30, 12, key="brush")
            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,0)",
                stroke_width=brush,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas_main"
            )
            predict_drawing = st.button("🔮 Prédire mon dessin", use_container_width=True)

        with col_draw_res:
            st.markdown("### 🎯 Résultat")
            if predict_drawing:
                if canvas_result.image_data is not None and canvas_result.image_data.sum() > 0:
                    img_data = canvas_result.image_data[:, :, :3].astype(np.uint8)
                    img_pil  = Image.fromarray(img_data)
                    arr = preprocess_tl(img_pil)
                    with st.spinner("Analyse..."):
                        probs, top5_idx, top5_prob = predict_top5(tl_model, arr)
                    pred_class, pred_prob = int(top5_idx[0]), float(top5_prob[0])
                    render_result_box(pred_class, pred_prob)
                    fig = plot_top5(top5_idx, top5_prob)
                    st.pyplot(fig); plt.close()
                    add_history("✏️ Dessin", "dessin-main", CLASS_NAMES[pred_class], pred_prob)
                else:
                    st.warning("⚠️ Canvas vide — dessinez d'abord un panneau.")
            else:
                st.markdown("""
                <div class='info-box'>
                    Dessinez un panneau sur le canvas à gauche,
                    puis cliquez sur <b>Prédire mon dessin</b>.
                </div>""", unsafe_allow_html=True)

    except ImportError:
        st.error("❌ Module `streamlit-drawable-canvas` non installé.")
        st.code("pip install streamlit-drawable-canvas")

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📁 Prédiction par lot")
    uploaded_files = st.file_uploader(
        "Chargez plusieurs images",
        type=["jpg","jpeg","png","ppm","bmp"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) chargée(s)**")
        rows = []
        for i in range(0, len(uploaded_files), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(uploaded_files):
                    break
                f = uploaded_files[idx]
                img = Image.open(f)
                arr = preprocess_tl(img)
                probs, top5_idx, top5_prob = predict_top5(tl_model, arr)
                pred_class = int(top5_idx[0])
                pred_prob  = float(top5_prob[0])
                color = "#48bb78" if pred_prob >= 0.7 else "#ed8936" if pred_prob >= 0.4 else "#e53e3e"
                with col:
                    st.image(img, use_container_width=True)
                    emoji = CLASS_EMOJIS.get(pred_class, '🚦')
                    st.markdown(
                        f"<div style='text-align:center;font-size:.85rem;'>"
                        f"<b>{emoji} {CLASS_NAMES[pred_class]}</b><br>"
                        f"<span style='color:{color};font-weight:700'>{pred_prob*100:.1f}%</span>"
                        f"</div>", unsafe_allow_html=True
                    )
                rows.append({
                    "Fichier": f.name,
                    "Classe prédite": CLASS_NAMES[pred_class],
                    "Confiance (%)": round(pred_prob * 100, 1)
                })
                add_history("📁 Batch", f.name, CLASS_NAMES[pred_class], pred_prob)

        st.markdown("---")
        st.markdown("#### 📋 Tableau récapitulatif")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger en CSV", csv, "batch_predictions.csv", "text/csv")
    else:
        st.markdown("""
        <div class='info-box'>
            Chargez plusieurs images pour obtenir leurs prédictions en grille.<br>
            Un tableau récapitulatif CSV est disponible au téléchargement.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — COMPARAISON CNN vs TL
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### ⚔️ CNN from scratch vs Transfer Learning (EfficientNetB0)")
    uploaded_cmp = st.file_uploader("Chargez une image pour comparer les deux modèles",
                                     type=["jpg","jpeg","png","ppm","bmp"],
                                     key="compare_upload")
    if uploaded_cmp:
        img_cmp = Image.open(uploaded_cmp)
        st.image(img_cmp, caption="Image analysée", width=200)

        arr_tl  = preprocess_tl(img_cmp)
        arr_cnn = preprocess_cnn(img_cmp)

        with st.spinner("Analyse des deux modèles..."):
            probs_cnn, top5_cnn, top5p_cnn = predict_top5(cnn_model, arr_cnn)
            probs_tl,  top5_tl,  top5p_tl  = predict_top5(tl_model,  arr_tl)

        pred_cnn,  conf_cnn  = int(top5_cnn[0]),  float(top5p_cnn[0])
        pred_tl,   conf_tl   = int(top5_tl[0]),   float(top5p_tl[0])

        col_cnn, col_tl = st.columns(2, gap="large")

        with col_cnn:
            st.markdown("#### 🔴 CNN from scratch")
            render_result_box(pred_cnn, conf_cnn)
            fig = plot_top5(top5_cnn, top5p_cnn, "CNN — Top 5")
            st.pyplot(fig); plt.close()
            add_history("⚔️ Comparaison (CNN)", uploaded_cmp.name, CLASS_NAMES[pred_cnn], conf_cnn)

        with col_tl:
            st.markdown("#### 🟢 Transfer Learning (EfficientNetB0)")
            render_result_box(pred_tl, conf_tl)
            fig = plot_top5(top5_tl, top5p_tl, "EfficientNetB0 — Top 5")
            st.pyplot(fig); plt.close()
            add_history("⚔️ Comparaison (TL)", uploaded_cmp.name, CLASS_NAMES[pred_tl], conf_tl)

        st.markdown("---")
        winner  = "EfficientNetB0" if conf_tl >= conf_cnn else "CNN from scratch"
        w_color = "#48bb78" if conf_tl >= conf_cnn else "#4299e1"
        delta   = abs(conf_tl - conf_cnn) * 100
        st.markdown(f"""
        <div style='background:{w_color}20;border:2px solid {w_color};border-radius:12px;padding:1rem;'>
            <b>🏆 Modèle le plus confiant : {winner}</b>
            ({delta:.1f} points d'écart)<br><br>
            <b>Pourquoi EfficientNetB0 est généralement meilleur :</b><br>
            • Pré-entraîné sur <b>ImageNet</b> (1.2M images, 1000 classes) — features visuelles riches dès le départ<br>
            • Architecture <b>Compound Scaling</b> : largeur, profondeur et résolution optimisées ensemble<br>
            • <b>Fine-tuning</b> sur GTSRB : réutilise les détecteurs de formes/couleurs/textures d'ImageNet<br>
            • Le CNN scratch apprend tout depuis zéro avec seulement 39k images → moins de généralisation
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box'>
            Chargez une image pour voir côte à côte les prédictions du CNN scratch (~85%)
            et de l'EfficientNetB0 (~97%).
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 5 — GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_gcam:
    st.markdown("### 🔥 Grad-CAM — Zones d'attention du modèle")
    st.markdown("""
    <div class='info-box'>
        <b>Qu'est-ce que le Grad-CAM ?</b><br>
        Le gradient de saillance montre <b>quels pixels de l'image influencent le plus la décision</b>
        du modèle. Les zones <span style='color:red'><b>rouges/chaudes</b></span> sont les régions
        que le modèle regarde pour identifier le panneau.
    </div>""", unsafe_allow_html=True)

    uploaded_gc = st.file_uploader("Chargez une image pour visualiser la carte de gradient",
                                    type=["jpg","jpeg","png","ppm","bmp"],
                                    key="gradcam_upload")
    if uploaded_gc:
        img_gc  = Image.open(uploaded_gc).convert('RGB')
        arr_gc  = preprocess_tl(img_gc)

        with st.spinner("Calcul du Grad-CAM..."):
            probs, top5_idx, top5_prob = predict_top5(tl_model, arr_gc)
            pred_class = int(top5_idx[0])
            heatmap    = compute_gradcam(tl_model, arr_gc, pred_class)

        if heatmap is not None:
            # Préparer les 3 images
            img_display = img_gc.resize((224, 224), Image.BILINEAR)

            hm_pil = Image.fromarray(
                (mcm.get_cmap('jet')(heatmap)[:, :, :3] * 255).astype(np.uint8)
            ).resize((224, 224), Image.BILINEAR)

            overlaid = overlay_heatmap(img_display, heatmap)

            col_orig, col_hm, col_ov = st.columns(3)
            with col_orig:
                st.image(img_display, caption="Image originale", use_container_width=True)
            with col_hm:
                st.image(hm_pil, caption="Carte de gradient (jet)", use_container_width=True)
            with col_ov:
                st.image(overlaid, caption="Superposition (α=0.4)", use_container_width=True)

            st.markdown("---")
            pred_prob = float(top5_prob[0])
            emoji     = CLASS_EMOJIS.get(pred_class, '🚦')
            render_result_box(pred_class, pred_prob)
            add_history("🔥 Grad-CAM", uploaded_gc.name, CLASS_NAMES[pred_class], pred_prob)
        else:
            st.error("Impossible de calculer le Grad-CAM pour cette image.")
    else:
        st.markdown("""
        <div class='info-box'>
            Chargez une image pour visualiser les zones d'attention du modèle EfficientNetB0.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 6 — CONDITIONS RÉELLES
# ══════════════════════════════════════════════════════════════════════════════
with tab_conditions:
    st.markdown("### 🌦️ Simulation de conditions réelles")
    st.markdown("""
    <div class='info-box'>
        Modifiez les conditions de l'image et observez comment la confiance du modèle évolue en temps réel.
    </div>""", unsafe_allow_html=True)

    uploaded_cond = st.file_uploader("Chargez une image de panneau",
                                      type=["jpg","jpeg","png","ppm","bmp"],
                                      key="conditions_upload")
    if uploaded_cond:
        img_base = Image.open(uploaded_cond).convert('RGB')

        st.markdown("#### ⚙️ Paramètres")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            brightness = st.slider("☀️ Luminosité",  0.2, 2.0, 1.0, 0.05, key="bright")
        with c2:
            blur       = st.slider("🌫️ Flou (px)",   0,   10,  0,        key="blur")
        with c3:
            noise      = st.slider("📡 Bruit",        0,   50,  0,        key="noise")
        with c4:
            rotation   = st.slider("🔄 Rotation (°)",-45, 45,  0,        key="rot")

        # Appliquer les transformations
        img_mod = img_base.copy()
        if brightness != 1.0:
            img_mod = ImageEnhance.Brightness(img_mod).enhance(brightness)
        if blur > 0:
            img_mod = img_mod.filter(ImageFilter.GaussianBlur(radius=blur))
        if noise > 0:
            arr_n = np.array(img_mod, dtype=np.float32)
            arr_n += np.random.normal(0, noise, arr_n.shape)
            arr_n  = np.clip(arr_n, 0, 255).astype(np.uint8)
            img_mod = Image.fromarray(arr_n)
        if rotation != 0:
            img_mod = img_mod.rotate(rotation, expand=False, fillcolor=(128, 128, 128))

        col_img, col_pred = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("#### 🖼️ Image modifiée")
            st.image(img_mod, use_container_width=True)

        with col_pred:
            st.markdown("#### 🎯 Prédiction en temps réel")
            arr_mod = preprocess_tl(img_mod)
            probs, top5_idx, top5_prob = predict_top5(tl_model, arr_mod)
            pred_class = int(top5_idx[0])
            pred_prob  = float(top5_prob[0])

            render_result_box(pred_class, pred_prob)

            # Jauge de confiance
            st.markdown("#### 📏 Jauge de confiance")
            st.progress(pred_prob)

            # Comparaison avec image originale
            arr_orig = preprocess_tl(img_base)
            _, _, top5p_orig = predict_top5(tl_model, arr_orig)
            conf_orig = float(top5p_orig[0])
            delta = (pred_prob - conf_orig) * 100
            st.metric("Confiance originale", f"{conf_orig*100:.1f}%",
                      delta=f"{delta:+.1f}%")

            add_history("🌦️ Conditions", uploaded_cond.name, CLASS_NAMES[pred_class], pred_prob)
    else:
        st.markdown("""
        <div class='info-box'>
            Chargez une image puis ajustez les sliders pour simuler :
            luminosité variable, flou de mouvement, bruit de capteur, rotation.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 7 — HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📜 Historique des prédictions")

    if not st.session_state.history:
        st.info("Aucune prédiction effectuée dans cette session.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📊 Classes les plus prédites")
        class_counts = df_hist["Classe prédite"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(class_counts.index, class_counts.values, color='#667eea')
        ax.invert_yaxis()
        ax.set_xlabel("Nombre de prédictions")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("---")
        col_dl, col_clr = st.columns([3, 1])
        with col_dl:
            csv_hist = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger l'historique en CSV",
                               csv_hist, "historique_predictions.csv", "text/csv")
        with col_clr:
            if st.button("🗑️ Effacer l'historique", type="secondary"):
                st.session_state.history = []
                st.rerun()
