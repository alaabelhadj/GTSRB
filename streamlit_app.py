"""
Application Streamlit — Reconnaissance de Panneaux de Signalisation (GTSRB)
Partie 2 : Interface utilisateur avec prédiction en temps réel
"""

import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# ─────────────────────────────────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GTSRB — Reconnaissance de Panneaux",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS personnalisé
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .prediction-box {
        background: #f0fff4;
        border: 2px solid #48bb78;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .warning-box {
        background: #fffaf0;
        border: 2px solid #ed8936;
        border-radius: 10px;
        padding: 1rem;
    }
    .info-box {
        background: #ebf8ff;
        border: 2px solid #4299e1;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div { background-color: #48bb78; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Noms des 43 classes GTSRB
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: 'Limite de vitesse (20 km/h)',
    1: 'Limite de vitesse (30 km/h)',
    2: 'Limite de vitesse (50 km/h)',
    3: 'Limite de vitesse (60 km/h)',
    4: 'Limite de vitesse (70 km/h)',
    5: 'Limite de vitesse (80 km/h)',
    6: 'Fin de la limite 80 km/h',
    7: 'Limite de vitesse (100 km/h)',
    8: 'Limite de vitesse (120 km/h)',
    9: 'Dépassement interdit',
    10: 'Dépassement interdit (camions)',
    11: 'Priorité au prochain croisement',
    12: 'Route prioritaire',
    13: 'Cédez le passage',
    14: 'STOP',
    15: 'Circulation interdite',
    16: 'Accès interdit aux camions',
    17: 'Sens interdit',
    18: 'Danger général',
    19: 'Virage à gauche',
    20: 'Virage à droite',
    21: 'Double virage',
    22: 'Route bosselée',
    23: 'Route glissante',
    24: 'Rétrécissement de chaussée',
    25: 'Travaux',
    26: 'Signaux lumineux',
    27: 'Piétons',
    28: 'Attention enfants',
    29: 'Cyclistes',
    30: 'Verglas / neige',
    31: 'Animaux sauvages',
    32: 'Fin de toutes restrictions',
    33: 'Tourner à droite obligatoire',
    34: 'Tourner à gauche obligatoire',
    35: 'Tout droit obligatoire',
    36: 'Tout droit ou tourner à droite',
    37: 'Tout droit ou tourner à gauche',
    38: 'Serrez à droite',
    39: 'Serrez à gauche',
    40: 'Rond-point obligatoire',
    41: 'Fin du dépassement interdit',
    42: 'Fin dépassement interdit (camions)'
}

# Émojis par type de panneau
CLASS_EMOJIS = {
    **{i: '🔵' for i in range(9)},   # Limitations vitesse
    9: '⛔', 10: '⛔',               # Interdictions dépassement
    11: '⚠️', 12: '⚠️',              # Priorité
    13: '🔺', 14: '🛑',              # Céder / Stop
    15: '🚫', 16: '🚫', 17: '🚫',   # Interdictions
    **{i: '⚠️' for i in range(18, 32)},  # Dangers
    32: '✅',                          # Fin restriction
    **{i: '💙' for i in range(33, 43)}   # Obligations
}

IMG_SIZE = (64, 64)

# ─────────────────────────────────────────────────────────────────────────────
# Chargement du modèle (avec cache)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    """Charge le modèle TensorFlow/Keras avec mise en cache."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Redimensionne et normalise l'image pour la prédiction."""
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return arr.reshape(1, *IMG_SIZE, 3)

def predict(model, img_array: np.ndarray):
    """Retourne les probabilités de prédiction."""
    probs = model.predict(img_array, verbose=0)[0]
    top5_idx  = np.argsort(probs)[::-1][:5]
    top5_prob = probs[top5_idx]
    return probs, top5_idx, top5_prob

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Stop_sign_MUTCD.svg/200px-Stop_sign_MUTCD.svg.png", width=80)
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 📁 Modèle")
    model_choice = st.selectbox(
        "Choisir le modèle",
        ["Transfer Learning (EfficientNetB0) ⭐", "CNN from scratch"],
        help="EfficientNetB0 est plus précis grâce au pré-entraînement ImageNet"
    )
    
    if "EfficientNet" in model_choice:
        model_path = "gtsrb_efficientnet_final.keras"
    else:
        model_path = "gtsrb_cnn_final.keras"

    st.markdown("### 📊 Métriques du modèle")
    if "EfficientNet" in model_choice:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>~97%</div>
            <div class='metric-label'>Accuracy sur le test set</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'>
            <div class='metric-value'>~85%</div>
            <div class='metric-label'>Accuracy sur le test set</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    **Dataset :** GTSRB  
    **Classes :** 43 panneaux  
    **Images train :** ~39 000  
    **Plateforme :** Kaggle Notebooks
    """)
    st.markdown("---")
    st.markdown("*Projet Deep Learning — Paris School of Technology & Business*")

# ─────────────────────────────────────────────────────────────────────────────
# En-tête principal
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-title'>🚦 Reconnaissance de Panneaux de Signalisation</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Dataset GTSRB — Deep Learning avec CNN & Transfer Learning (EfficientNetB0)</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Onglets
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "📊 Résultats du modèle", "📋 Classes du dataset"])

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 Charger une image")
        
        uploaded_file = st.file_uploader(
            "Déposez une image de panneau ici",
            type=["jpg", "jpeg", "png", "ppm", "bmp"],
            help="Formats acceptés : JPG, PNG, PPM, BMP"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image chargée", use_column_width=True)
            
            st.markdown("#### 📐 Informations sur l'image")
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Largeur", f"{image.size[0]} px")
            with info_col2:
                st.metric("Hauteur", f"{image.size[1]} px")
            with info_col3:
                st.metric("Mode", image.mode)

        else:
            st.markdown("""
            <div class='info-box'>
                <b>Instructions :</b><br>
                1. Chargez une image de panneau de signalisation<br>
                2. Le modèle prédit la classe automatiquement<br>
                3. Les 5 meilleures prédictions sont affichées avec leur probabilité
            </div>
            """, unsafe_allow_html=True)
            
            # Image exemple
            st.markdown("**Exemples de panneaux compatibles :**")
            st.markdown("Stop 🛑 | Cédez le passage 🔺 | Limitations de vitesse 🔵 | Dangers ⚠️")

    with col_result:
        st.markdown("### 🎯 Résultat de la prédiction")

        if uploaded_file is not None:
            # Chargement modèle
            with st.spinner("Chargement du modèle..."):
                model, error = load_model(model_path)

            if error:
                st.error(f"Erreur lors du chargement du modèle : {error}")
                st.markdown("""
                <div class='warning-box'>
                    <b>⚠️ Modèle non trouvé</b><br>
                    Assurez-vous d'avoir exécuté le notebook Kaggle et d'avoir téléchargé le fichier 
                    <code>gtsrb_efficientnet_final.keras</code> dans le même dossier que cette app.
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Analyse en cours..."):
                    img_array = preprocess_image(image)
                    probs, top5_idx, top5_prob = predict(model, img_array)

                pred_class = top5_idx[0]
                pred_prob  = top5_prob[0]
                emoji      = CLASS_EMOJIS.get(pred_class, '🚦')

                # Résultat principal
                confidence_color = "#48bb78" if pred_prob >= 0.7 else "#ed8936" if pred_prob >= 0.4 else "#e53e3e"
                st.markdown(f"""
                <div style='background:{confidence_color}20; border:2px solid {confidence_color}; 
                             border-radius:12px; padding:1.2rem; text-align:center;'>
                    <div style='font-size:3rem'>{emoji}</div>
                    <div style='font-size:1.3rem; font-weight:700; color:#1a1a2e; margin:0.5rem 0'>
                        {CLASS_NAMES[pred_class]}
                    </div>
                    <div style='font-size:2rem; font-weight:700; color:{confidence_color}'>
                        {pred_prob*100:.1f}%
                    </div>
                    <div style='color:#666; font-size:0.9rem'>Confiance</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 🏆 Top 5 prédictions")
                fig, ax = plt.subplots(figsize=(6, 3))
                y_pos   = range(5)
                colors  = ['#2ecc71' if i == 0 else '#95a5a6' for i in range(5)]
                labels  = [f"{CLASS_NAMES[idx][:28]}" for idx in top5_idx]
                ax.barh(y_pos, top5_prob * 100, color=colors, edgecolor='white')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel('Probabilité (%)')
                ax.set_xlim(0, 105)
                ax.invert_yaxis()
                for i, (v, idx) in enumerate(zip(top5_prob, top5_idx)):
                    ax.text(v * 100 + 1, i, f'{v*100:.1f}%', va='center', fontsize=9, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Jauge de confiance
                st.markdown("#### 📏 Niveau de confiance")
                if pred_prob >= 0.9:
                    st.success(f"✅ Très haute confiance ({pred_prob*100:.1f}%) — Prédiction fiable")
                elif pred_prob >= 0.7:
                    st.info(f"🔵 Haute confiance ({pred_prob*100:.1f}%) — Bonne prédiction")
                elif pred_prob >= 0.4:
                    st.warning(f"⚠️ Confiance modérée ({pred_prob*100:.1f}%) — À vérifier")
                else:
                    st.error(f"❌ Faible confiance ({pred_prob*100:.1f}%) — Image difficile à classifier")

        else:
            st.markdown("""
            <div style='text-align:center; padding:3rem; color:#aaa;'>
                <div style='font-size:4rem'>📸</div>
                <div style='font-size:1.1rem; margin-top:1rem'>
                    Chargez une image pour obtenir une prédiction
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — RÉSULTATS DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Performance des modèles sur GTSRB")

    # Métriques globales
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CNN from scratch", "~85%", help="Accuracy sur le test set")
    with c2:
        st.metric("Transfer Learning", "~97%", "+12%", help="Accuracy sur le test set")
    with c3:
        st.metric("Nombre de classes", "43")
    with c4:
        st.metric("Images d'entraînement", "~39 000")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Comparaison CNN vs Transfer Learning")
        fig, ax = plt.subplots(figsize=(6, 4))
        models  = ['CNN\nfrom scratch', 'Transfer Learning\n(EfficientNetB0)']
        accs    = [85, 97]
        bars    = ax.bar(models, accs, color=['#4e9af1', '#2ecc71'], width=0.45)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc}%', ha='center', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 110)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy sur le test set', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("#### Architecture des modèles")
        st.markdown("""
        **CNN from scratch**
        - 3 blocs Conv+BN+MaxPool
        - Filtres : 32 → 64 → 128
        - Dense 512 + 256
        - Dropout + BatchNorm
        - ~2M paramètres

        ---

        **Transfer Learning (EfficientNetB0)**
        - Base : EfficientNetB0 (ImageNet)
        - GlobalAveragePooling2D
        - Dense 256 + 128
        - Fine-tuning des 30 dernières couches
        - ~5.3M paramètres totaux
        """)

    st.markdown("---")
    st.markdown("#### Stratégie d'entraînement")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class='info-box'>
            <b>🔧 Prétraitement</b><br>
            • Resize → 64×64 px<br>
            • Normalisation [0, 1]<br>
            • Pas de flip horizontal<br>
            (panneaux asymétriques)
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='info-box'>
            <b>📈 Augmentation</b><br>
            • Rotation ±15°<br>
            • Zoom ±15%<br>
            • Décalage ±10%<br>
            • Luminosité [0.8, 1.2]
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class='info-box'>
            <b>🛡️ Régularisation</b><br>
            • EarlyStopping<br>
            • ReduceLROnPlateau<br>
            • BatchNormalization<br>
            • Dropout (0.25–0.5)
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — CLASSES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Les 43 classes du dataset GTSRB")

    # Filtres
    search = st.text_input("🔍 Rechercher une classe...", placeholder="Ex: stop, vitesse, priorité...")

    categories = {
        "🔵 Limitations de vitesse (0–8)": list(range(9)),
        "⛔ Interdictions de dépassement (9–10)": [9, 10],
        "⚠️ Priorité & dangers (11–31)": list(range(11, 32)),
        "✅ Fins de restrictions (32)": [32],
        "💙 Obligations (33–42)": list(range(33, 43))
    }

    if search:
        filtered = {k: v for k, v in CLASS_NAMES.items()
                    if search.lower() in v.lower() or search.lower() in str(k)}
        if filtered:
            st.markdown(f"**{len(filtered)} résultat(s) :**")
            for cls_id, name in filtered.items():
                emoji = CLASS_EMOJIS.get(cls_id, '🚦')
                st.markdown(f"- {emoji} **Classe {cls_id}** : {name}")
        else:
            st.warning("Aucune classe trouvée.")
    else:
        for cat_name, cls_ids in categories.items():
            with st.expander(cat_name, expanded=False):
                cols = st.columns(2)
                for i, cls_id in enumerate(cls_ids):
                    emoji = CLASS_EMOJIS.get(cls_id, '🚦')
                    cols[i % 2].markdown(f"**{emoji} Classe {cls_id}** — {CLASS_NAMES[cls_id]}")

    st.markdown("---")
    st.markdown("#### Distribution du dataset (estimation)")
    
    # Distribution approximative GTSRB
    approx_dist = {
        'Vitesse': 9, 'Interdictions': 6, 'Priorité': 2,
        'Cédez/Stop': 2, 'Dangers': 13, 'Obligations': 8, 'Fins': 3
    }
    fig, ax = plt.subplots(figsize=(6, 6))
    colors_pie = ['#4e9af1', '#e74c3c', '#f39c12', '#e91e63', '#ff9800', '#2ecc71', '#9b59b6']
    wedges, texts, autotexts = ax.pie(
        approx_dist.values(), labels=approx_dist.keys(),
        autopct='%1.0f%%', colors=colors_pie,
        startangle=140, pctdistance=0.8
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax.set_title('Répartition des types de panneaux', fontweight='bold')
    st.pyplot(fig)
    plt.close()
