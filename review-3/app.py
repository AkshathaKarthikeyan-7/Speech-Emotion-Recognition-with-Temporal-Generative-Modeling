"""
Streamlit Deployment App — Speech Emotion Recognition
======================================================
Upload a .wav file → preprocess → VAE encode → MLP classify → display results.
"""

import os
import sys
import tempfile
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Emotion styling ─────────────────────────────────────────────────────────
EMOTION_META = {
    "neutral":   {"emoji": "😐", "color": "#6c757d", "bg": "#f8f9fa"},
    "calm":      {"emoji": "😌", "color": "#17a2b8", "bg": "#e3f7fc"},
    "happy":     {"emoji": "😄", "color": "#ffc107", "bg": "#fff8e1"},
    "sad":       {"emoji": "😢", "color": "#007bff", "bg": "#e8f0fe"},
    "angry":     {"emoji": "😡", "color": "#dc3545", "bg": "#fdecea"},
    "fearful":   {"emoji": "😨", "color": "#6f42c1", "bg": "#f3eefb"},
    "disgust":   {"emoji": "🤢", "color": "#28a745", "bg": "#e8f5e9"},
    "surprised": {"emoji": "😲", "color": "#fd7e14", "bg": "#fff3e0"},
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("**Model:** Residual VAE + MLP Classifier")
st.sidebar.markdown("**Dataset:** RAVDESS (8 emotions)")
st.sidebar.markdown("**Features:** 3-channel Mel+Δ+Δ²")
st.sidebar.markdown("---")
st.sidebar.info(
    "📁 **Model files required:**\n"
    "- `outputs/encoder.keras`\n"
    "- `outputs/mlp_best.keras`\n"
    "- `outputs/label_encoder.npy`\n\n"
    "Run `main.ipynb` first to train the models."
)

# ─── Main Title ──────────────────────────────────────────────────────────────
st.title("🎤 Speech Emotion Recognition")
st.markdown(
    "Upload a `.wav` audio file. The system will extract 3-channel spectrogram "
    "features, encode them with the VAE, and classify the emotion."
)

# ─── Load Models (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load encoder + MLP classifier from disk."""
    try:
        import tensorflow as tf
        import numpy as np

        encoder = tf.keras.models.load_model("outputs/encoder.keras")
        mlp     = tf.keras.models.load_model("outputs/mlp_best.keras")
        label_names = list(np.load("outputs/label_encoder.npy", allow_pickle=True))

        return encoder, mlp, label_names

    except Exception as e:
        print("Model loading error:", e)
        return None, None, None


@st.cache_resource
def get_preprocessing():
    sys.path.insert(0, os.path.dirname(__file__))
    from audio_processing import extract_3channel_spectrogram
    return extract_3channel_spectrogram


# ─── File Upload ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📂 Upload a .wav file", type=["wav"])

if uploaded is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded, format="audio/wav")

    with st.spinner("🔄 Preprocessing audio..."):
        try:
            extract_fn = get_preprocessing()
            spec = extract_fn(tmp_path)   # (128, 128, 3)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

    # ─── Spectrogram Visualisation ────────────────────────────────────────────
    st.subheader("🎼 Feature Visualisation")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    ch_titles = [
    "Log-Mel Spectrogram",
    "Delta (Velocity)",
    "Delta-Delta (Accel.)"
    ]
    cmaps = ["magma", "coolwarm", "RdYlBu"]
    for ax, ch, title, cmap in zip(axes, range(3), ch_titles, cmaps):
        im = ax.imshow(spec[:, :, ch], aspect="auto", origin="lower",
                       cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time frames", fontsize=9)
        ax.set_ylabel("Mel bins", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ─── Model Inference ─────────────────────────────────────────────────────
    encoder, mlp, label_names = load_models()

    if encoder is None:
        st.warning(
            "⚠️ Trained models not found. Please run `main.ipynb` first to train "
            "and save the VAE + MLP models."
        )
        # Demo mode: random predictions
        label_names = list(EMOTION_META.keys())
        probs = np.random.dirichlet(np.ones(len(label_names)))
        predicted_idx = int(np.argmax(probs))
        st.info("🔬 **Demo Mode** (random predictions — train models for real results)")
    else:
        with st.spinner("🧠 Running VAE encoder + MLP classifier..."):
            import tensorflow as tf
            spec_batch = spec[np.newaxis, ...]   # (1, 128, 128, 3)
            z_mean = encoder(spec_batch, training=False)
            probs_raw = mlp(z_mean, training=False).numpy()[0]
            probs = probs_raw
            predicted_idx = int(np.argmax(probs))

    predicted_emotion = label_names[predicted_idx]
    confidence = float(probs[predicted_idx])
    meta = EMOTION_META.get(predicted_emotion, {"emoji": "❓", "color": "#333", "bg": "#eee"})

    # ─── Result Card ─────────────────────────────────────────────────────────
    st.subheader("🎯 Prediction Result")

    card_style = (
        f"background-color: {meta['bg']}; "
        f"border-left: 8px solid {meta['color']}; "
        f"border-radius: 12px; padding: 20px 30px; margin: 10px 0;"
    )
    st.markdown(
        f"""
        <div style="{card_style}">
            <h2 style="margin:0; color:{meta['color']};">
                {meta['emoji']} {predicted_emotion.upper()}
            </h2>
            <p style="font-size:18px; margin-top:8px;">
                Confidence: <strong>{confidence*100:.1f}%</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Top-3 Predictions ────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    top3_idx = np.argsort(probs)[::-1][:3]

    with col1:
        st.subheader("🏆 Top-3 Predictions")
        for rank, idx in enumerate(top3_idx, 1):
            emo   = label_names[idx]
            prob  = probs[idx]
            m     = EMOTION_META.get(emo, {"emoji": "❓", "color": "#333"})
            bar_w = int(prob * 100)
            st.markdown(
                f"**{rank}. {m['emoji']} {emo}** — {prob*100:.1f}%  "
                f"`{'█' * bar_w}{'░' * (100 - bar_w)}`[:30]"
            )

    with col2:
        st.subheader("📊 Probability Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        colors_bar = [
            EMOTION_META.get(label_names[i], {"color": "#999"})["color"]
            for i in range(len(label_names))
        ]
        ax2.barh(label_names, probs, color=colors_bar, edgecolor="white")
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Probability")
        ax2.set_title("Emotion Probability Distribution", fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)
        for i, (p, n) in enumerate(zip(probs, label_names)):
            ax2.text(p + 0.01, i, f"{p*100:.1f}%", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ─── About Panel ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ About This System"):
        st.markdown("""
### Architecture
- **Feature Extraction**: 3-channel spectrograms (Log-Mel + Δ + Δ²)
- **VAE Encoder**: 4 Residual Convolutional blocks → 128-dim latent space (z_mean)
- **MLP Classifier**: Dense(512→256→128→8) with BatchNorm + Dropout

### Why z_mean for inference?
The VAE reparameterization trick samples z = z_mean + ε·σ during training.
At inference we use z_mean directly — it's deterministic and lower-variance,
producing more reliable emotion embeddings.

### Dataset
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
~1440 audio clips, 24 actors (12 male, 12 female)

### Applications
- 🤖 AI Assistants — adaptive tone response
- 🧠 Mental Health Monitoring — distress detection
- 📞 Call Centre Analytics — customer frustration flagging
        """)

    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

else:
    # Landing state
    st.info("👆 Upload a `.wav` file above to get started.")

    st.subheader("🎭 Supported Emotions")
    cols = st.columns(4)
    for i, (emo, meta) in enumerate(EMOTION_META.items()):
        col = cols[i % 4]
        col.markdown(
            f"""
            <div style="background:{meta['bg']}; border-left:4px solid {meta['color']};
                        border-radius:8px; padding:10px; margin:4px; text-align:center;">
                <div style="font-size:28px;">{meta['emoji']}</div>
                <div style="font-weight:bold; color:{meta['color']};">{emo}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
