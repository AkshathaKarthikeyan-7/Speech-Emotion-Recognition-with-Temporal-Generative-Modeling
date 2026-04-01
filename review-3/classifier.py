"""
Speech Emotion Recognition Classifiers
=======================================

Application Context: Speech Emotion Recognition for Human-Computer Interaction
  - AI Assistants (Alexa, Siri): adapt response tone to user's emotional state
  - Mental Health Monitoring: detect distress from voice calls in real-time
  - Call Centre Analytics: flag frustrated/angry customers for priority routing
  - Autism Therapy Tools: help individuals interpret emotions from speech
  - E-learning Platforms: detect student frustration and adapt content delivery

Two classification models are built:

Model A — MLP on VAE z_mean embeddings
  Input: 128-dim z_mean from the trained VAE encoder.
  Why: The VAE has already learned a structured, emotion-relevant latent space.
  The MLP only needs to learn a decision boundary in this compact space.
  Dense(512→256→128→N_classes) with BN, ReLU, Dropout.

Model B — BiLSTM + Bahdanau Attention on sequential CNN features
  Input: 3-channel spectrogram → sequential CNN preserving time axis.
  Why: Emotions unfold over time (rising anger, fading calm) — temporal
  modelling captures these dynamics that a flat embedding cannot.
  BiLSTM: forward + backward LSTM → captures context from both directions.
  Bahdanau Attention: learns which time frames are most diagnostic for each emotion.
  Attention weights are visualisable → interpretability bonus.

Why Bahdanau (additive) attention over dot-product attention?
  Bahdanau computes score = V · tanh(W1·h_t + W2·query), where W1,W2 are
  learned projections. This is more expressive for small datasets because it
  introduces additional parameters that capture non-linear alignment.
  Dot-product attention (score = q·k) is faster but less expressive at small scale.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

tf.random.set_seed(42)
np.random.seed(42)

LABEL_SMOOTH   = 0.1   # Prevents overconfident predictions; improves calibration
DROPOUT_RATE   = 0.4


# ─── Model A: MLP on VAE Latent Embeddings ───────────────────────────────────

def build_mlp_classifier(input_dim: int, n_classes: int,
                          dropout_rate: float = DROPOUT_RATE) -> keras.Model:
    """
    MLP classifier on z_mean embeddings from the VAE encoder.
    Architecture: Dense(512)→BN→ReLU→Dropout → Dense(256)→BN→ReLU→Dropout
                → Dense(128)→BN→ReLU→Dropout(0.3) → Dense(N, softmax)
    Label smoothing applied via loss function, not here.
    """
    inp = keras.Input(shape=(input_dim,), name="z_mean_input")

    x = layers.Dense(512, use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(n_classes, activation="softmax", name="emotion_probs")(x)

    model = keras.Model(inp, out, name="MLP_Classifier")
    return model


# ─── Bahdanau Attention Layer ─────────────────────────────────────────────────

class BahdanauAttention(layers.Layer):
    """
    Additive (Bahdanau) attention mechanism.
    score(h_t, query) = V · tanh(W1·h_t + W2·query)
    α_t = softmax(score_t)
    context = Σ α_t · h_t

    Why additive over dot-product for speech emotion?
      - The alignment is non-linear (emotional peaks are not simply correlated
        with query magnitude) — tanh captures this.
      - More parameters (W1, W2, V) learn richer alignment for small datasets.
      - Attention weights α_t per time step are visualisable: we can see which
        frames the model focuses on for each emotion class.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units, use_bias=False)   # Project encoder states
        self.W2 = layers.Dense(units, use_bias=False)   # Project query
        self.V  = layers.Dense(1, use_bias=False)        # Score projection

    def call(self, encoder_outputs, query=None):
        """
        encoder_outputs: (batch, T, hidden_dim)
        query: (batch, hidden_dim) — typically last LSTM state
        Returns: context (batch, hidden_dim), attention_weights (batch, T, 1)
        """
        if query is None:
            query = encoder_outputs[:, -1, :]   # Use last timestep as default query

        # Expand query to (batch, 1, units) for broadcasting
        query_exp = tf.expand_dims(query, axis=1)

        # Score: (batch, T, 1)
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(query_exp)))

        # Attention weights: softmax over time axis
        alpha = tf.nn.softmax(score, axis=1)   # (batch, T, 1)

        # Context vector: weighted sum of encoder outputs
        context = tf.reduce_sum(alpha * encoder_outputs, axis=1)  # (batch, hidden_dim)
        return context, alpha

    def get_config(self):
        config = super().get_config()
        config["units"] = self.units
        return config


# ─── Model B: BiLSTM + Bahdanau Attention ────────────────────────────────────

def build_bilstm_attention_classifier(input_shape: tuple, n_classes: int) -> keras.Model:
    """
    BiLSTM + Bahdanau Attention Classifier.
    Pipeline:
      1. Sequential CNN encoder: extracts local features while PRESERVING time axis
         (no GlobalAveragePooling — we pool only the freq/channel dims)
      2. Reshape to (batch, T, features)
      3. 2-layer Bidirectional LSTM with recurrent dropout
      4. Bahdanau attention using final LSTM hidden state as query
      5. Dense classifier

    Why preserve time axis?
      Emotions are inherently temporal: anger builds, sadness lingers.
      Collapsing time at the CNN stage (GlobalAvgPool) would throw away this
      sequential structure. We preserve T so the LSTM can model temporal dynamics.

    Why BiLSTM?
      Forward LSTM sees past frames; backward LSTM sees future frames.
      Bidirectional concatenation gives context from both directions, which is
      important for detecting emotion cues that appear mid-utterance.
    """
    inp = keras.Input(shape=input_shape, name="spectrogram_input")   # (H, W, 3)

    # ── Sequential CNN: extract features preserving time axis ──
    # Conv along freq axis only, pool along freq (axis 1 = height)
    x = layers.Conv2D(32, (3, 1), padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, (3, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, (3, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Average pool over frequency axis → (batch, W, 128) = (batch, T, features)
    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
    # x shape: (batch, W=128, 128)

    # ── 2-layer Bidirectional LSTM ──
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, recurrent_dropout=0.2,
                    dropout=0.2),
        name="bilstm_1"
    )(x)
    # x: (batch, T, 256)

    x, forward_h, _, backward_h, _ = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, return_state=True,
                    recurrent_dropout=0.2, dropout=0.2),
        name="bilstm_2"
    )(x)
    # x: (batch, T, 128), forward_h: (batch, 64), backward_h: (batch, 64)

    # Concatenate final hidden states as query for attention
    query = layers.Concatenate()([forward_h, backward_h])   # (batch, 128)

    # ── Bahdanau Attention ──
    attn = BahdanauAttention(units=64, name="bahdanau_attention")
    context, attention_weights = attn(x, query)
    # context: (batch, 128)

    # ── Classifier head ──
    x = layers.Dense(128, activation="relu")(context)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="emotion_probs")(x)

    model = keras.Model(inp, out, name="BiLSTM_Attention_Classifier")
    return model


def build_attention_extractor(bilstm_model: keras.Model) -> keras.Model:
    """
    Sub-model that returns attention weights alongside predictions.
    Used for visualising which time frames the model focuses on per emotion.
    """
    inp      = bilstm_model.input
    attn_out = bilstm_model.get_layer("bahdanau_attention").output  # [context, alpha]
    probs    = bilstm_model.output
    return keras.Model(inp, [probs, attn_out[1]], name="attention_extractor")


# ─── Baseline and Ablation Models ────────────────────────────────────────────

def build_baseline_cnn(input_shape: tuple, n_classes: int) -> keras.Model:
    """
    Baseline: simple CNN on raw spectrograms, NO BatchNorm, NO Dropout.
    Used to demonstrate the benefit of regularisation in ablation study.
    """
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="Baseline_CNN")


def build_improved_cnn(input_shape: tuple, n_classes: int) -> keras.Model:
    """
    Improved CNN: adds BatchNorm + Dropout over the baseline.
    BatchNorm: accelerates convergence and acts as a regulariser.
    Dropout: prevents co-adaptation of neurons → better generalisation.
    """
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="Improved_CNN")


def build_ablation_no_bn(input_shape: tuple, n_classes: int) -> keras.Model:
    """Ablation: Improved CNN without BatchNormalization."""
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="Ablation_NoBN")


def build_ablation_no_dropout(input_shape: tuple, n_classes: int) -> keras.Model:
    """Ablation: Improved CNN without Dropout."""
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="Ablation_NoDropout")


# ─── Training Utilities ───────────────────────────────────────────────────────

def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute balanced class weights to handle RAVDESS class imbalance.
    weight_c = N / (n_classes × N_c)
    Up-weights minority classes so their gradients contribute equally.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def get_classifier_callbacks(monitor="val_accuracy",
                              checkpoint_path="outputs/best_classifier.keras"):
    """
    mode='max' for accuracy-based monitors; Keras cannot auto-infer direction
    for custom or ambiguously-named metrics, so we always set it explicitly.
    """
    mode = "min" if "loss" in monitor else "max"
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=15, mode=mode,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=7,
            mode=mode, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor=monitor,
            mode=mode, save_best_only=True, verbose=0
        ),
    ]


def compile_classifier(model: keras.Model, n_classes: int,
                        learning_rate: float = 1e-3,
                        label_smooth: float = LABEL_SMOOTH):
    """
    Compile with Adam + label smoothing cross-entropy.
    Label smoothing (0.1): replaces hard targets [0,0,1,0] with soft ones
    [0.0125, 0.0125, 0.8875, 0.0125]. This prevents the model from becoming
    overconfident, improving calibration and generalisation.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth),
        metrics=["accuracy"]
    )
    return model