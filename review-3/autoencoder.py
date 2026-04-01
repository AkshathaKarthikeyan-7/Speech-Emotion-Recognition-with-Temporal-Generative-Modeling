"""
Variational Autoencoder (VAE) — Residual Convolutional Architecture
====================================================================
Why VAE over plain Autoencoder?
  A plain AE learns a deterministic encoding → latent space is unstructured
  (gaps, discontinuities). VAE adds KL regularization that forces the posterior
  q(z|x) to stay close to the standard Normal prior p(z)=N(0,I). This produces:
    1. A SMOOTH, CONTINUOUS latent space — interpolating between two emotions
       in latent space gives semantically meaningful intermediate states.
    2. Better generalization on small datasets (RAVDESS ~1440 samples) because
       the model cannot overfit by memorising specific latent codes.
    3. Cleaner emotion clusters — emotions that are perceptually similar
       (e.g. calm ↔ neutral, angry ↔ surprised) cluster closer together,
       reflecting the geometry of emotional space.

Why z_mean for inference (not sampled z)?
  During training z = z_mean + ε·exp(0.5·z_log_var) introduces stochasticity
  which is necessary for the KL term. At inference time we use z_mean directly:
    - Deterministic: same audio always maps to same point → reliable classifier input
    - Minimum-variance estimate of the latent code → lower noise → better accuracy
    - The KL term has already ensured z_mean is well-regularised

Architecture: 4 Residual Blocks (Encoder) ↔ 4 ConvTranspose Blocks (Decoder)
Each residual block: Conv→BN→ReLU→Conv→BN→Add(skip with 1×1 proj)→ReLU
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ─── Reproducibility ────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ─── Config ─────────────────────────────────────────────────────────────────
LATENT_DIM   = 128
KL_WEIGHT    = 0.001    # β: small so reconstruction quality is preserved
INPUT_SHAPE  = (128, 128, 3)
BATCH_SIZE   = 16       # Small batch → better gradient noise → better generalisation
LR_AE        = 5e-4
DROPOUT_RATE = 0.4


# ─── Building Blocks ─────────────────────────────────────────────────────────

def residual_block(x, filters: int, stride: int = 1, dropout_rate: float = DROPOUT_RATE):
    """
    Residual block: Conv→BN→ReLU→Conv→BN→Add(skip)→ReLU
    Skip connection uses 1×1 conv projection when channel/spatial dims change.
    BN before activation: standard practice for stable deep network training.
    Dropout after block: regularises activations, reduces co-adaptation.
    """
    shortcut = x

    # Main path
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Project shortcut if shape changes
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                 use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def conv_transpose_block(x, filters: int):
    """
    Decoder upsampling block: ConvTranspose→BN→ReLU + UpSampling2D.
    UpSampling is paired after ConvTranspose to avoid checkerboard artefacts.
    """
    x = layers.Conv2DTranspose(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    return x


# ─── Sampling Layer ──────────────────────────────────────────────────────────

class Sampling(layers.Layer):
    """
    Reparameterization trick:
      z = z_mean + exp(0.5 * z_log_var) * ε,  ε ~ N(0, I)
    This keeps the sampling step differentiable w.r.t. z_mean and z_log_var,
    allowing gradients to flow through the stochastic node.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch  = tf.shape(z_mean)[0]
        dim    = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), seed=42)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ─── Encoder ─────────────────────────────────────────────────────────────────

def build_encoder(input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM):
    """
    4 Residual blocks with increasing filters (32→64→128→256), each followed
    by 2×2 MaxPool to progressively downsample the spatial dimensions.
    After flattening: Dense(512) → two parallel heads for z_mean and z_log_var.
    """
    inp = keras.Input(shape=input_shape, name="encoder_input")

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 4 residual blocks
    x = residual_block(x, 32)
    x = layers.MaxPooling2D(2)(x)          # 128→64

    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2)(x)          # 64→32

    x = residual_block(x, 128)
    x = layers.MaxPooling2D(2)(x)          # 32→16

    x = residual_block(x, 256)
    x = layers.MaxPooling2D(2)(x)          # 16→8

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling(name="z")([z_mean, z_log_var])

    encoder = keras.Model(inp, [z_mean, z_log_var, z], name="encoder")
    return encoder


# ─── Decoder ─────────────────────────────────────────────────────────────────

def build_decoder(latent_dim=LATENT_DIM, output_shape=INPUT_SHAPE):
    """
    Symmetric to encoder. Dense → Reshape → 4× ConvTranspose+UpSample blocks.
    Final tf.image.resize ensures output matches input_shape exactly regardless
    of intermediate spatial rounding errors — robust to any input size.
    We avoid Cropping2D because it silently fails on some input sizes.
    Sigmoid activation maps output to [0, 1] matching normalised spectrogram range.
    """
    latent_inp = keras.Input(shape=(latent_dim,), name="decoder_input")

    # Compute starting spatial size: INPUT_SHAPE / 2^4 = 128/16 = 8
    start_h = output_shape[0] // 16
    start_w = output_shape[1] // 16

    x = layers.Dense(start_h * start_w * 256, activation="relu")(latent_inp)
    x = layers.Reshape((start_h, start_w, 256))(x)   # (8, 8, 256)

    x = conv_transpose_block(x, 128)    # → (16, 16, 128)
    x = conv_transpose_block(x, 64)     # → (32, 32, 64)
    x = conv_transpose_block(x, 32)     # → (64, 64, 32)
    x = conv_transpose_block(x, 16)     # → (128, 128, 16)

    x = layers.Conv2DTranspose(output_shape[-1], 3, padding="same")(x)

    # tf.image.resize: robust output sizing — works for any input resolution
    x = layers.Lambda(
        lambda t: tf.image.resize(t, (output_shape[0], output_shape[1])),
        name="resize_output"
    )(x)

    x = layers.Activation("sigmoid", name="decoder_output")(x)

    decoder = keras.Model(latent_inp, x, name="decoder")
    return decoder


# ─── VAE Model ───────────────────────────────────────────────────────────────

class VAE(keras.Model):
    """
    Custom Keras Model encapsulating encoder + decoder.
    Overrides train_step and test_step to compute:
      total_loss = reconstruction_loss + β × KL_loss

    Reconstruction loss = 0.8×MSE + 0.2×L1
      MSE captures large errors strongly (squared penalty).
      L1 handles boundary sharpness and is more robust to outliers.
      Combined loss balances global structure (MSE) and detail (L1).

    KL loss = -0.5 × mean(1 + log_var - mean² - exp(log_var))
      Forces posterior q(z|x) towards N(0,I).
      β=0.001 keeps reconstruction quality high while still regularising.
    """

    def __init__(self, encoder, decoder, kl_weight=KL_WEIGHT, **kwargs):
        super().__init__(**kwargs)
        self.encoder    = encoder
        self.decoder    = decoder
        self.kl_weight  = kl_weight

        # Metric trackers
        self.total_loss_tracker        = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker           = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=False):
        """Forward pass: encode → sample z → decode."""
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction

    def encode(self, inputs, training=False):
        """Return z_mean, z_log_var, z for a batch."""
        return self.encoder(inputs, training=training)

    def decode(self, z, training=False):
        return self.decoder(z, training=training)

    def _compute_losses(self, data):
        """Shared loss computation for train and test steps."""
        z_mean, z_log_var, z = self.encoder(data, training=True)
        reconstruction = self.decoder(z, training=True)

        # Reconstruction: 0.8×MSE + 0.2×L1
        mse = tf.reduce_mean(tf.square(data - reconstruction))
        l1  = tf.reduce_mean(tf.abs(data - reconstruction))
        recon_loss = 0.8 * mse + 0.2 * l1

        # KL divergence (analytical form for Gaussian)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self._compute_losses(data)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        total_loss, recon_loss, kl_loss = self._compute_losses(data)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def get_latent_embeddings(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Extract z_mean for all samples in X.
        We use z_mean (NOT sampled z) for downstream classification because:
          - z_mean is deterministic: given the same audio, always same embedding
          - z_mean is the MAP estimate of z given x
          - KL regularisation ensures z_mean is already well-distributed
          - Sampled z adds unnecessary variance that would hurt classifier accuracy
        """
        z_means = []
        for i in range(0, len(X), batch_size):
            batch = X[i: i + batch_size]
            z_mean, _, _ = self.encoder(batch, training=False)
            z_means.append(z_mean.numpy())
        return np.concatenate(z_means, axis=0)


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_vae(input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM, kl_weight=KL_WEIGHT,
              learning_rate=LR_AE):
    """Build and compile the full VAE model."""
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)
    vae     = VAE(encoder, decoder, kl_weight=kl_weight, name="ResidualVAE")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    vae.compile(optimizer=optimizer)

    print(encoder.summary())
    print(decoder.summary())
    return vae


def get_vae_callbacks(checkpoint_path: str = "outputs/vae_best.keras"):
    """
    Standard training callbacks for VAE stability and early stopping.
    mode='min' is mandatory for custom metric names (total_loss, reconstruction_loss,
    kl_loss) because Keras cannot auto-infer direction from non-standard names.
    Without it, Keras raises ValueError on monitor='val_total_loss'.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_total_loss", patience=12, mode="min",
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_total_loss", factor=0.5, patience=6,
            mode="min", min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_total_loss",
            mode="min", save_best_only=True, verbose=1
        ),
    ]


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    PSNR > 30dB is generally considered good reconstruction quality.
    Formula: 10 × log10(MAX²/MSE), MAX=1.0 for [0,1] normalised data.
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_reconstruction_metrics(vae: VAE, X: np.ndarray, batch_size: int = 32):
    """Compute MSE and PSNR over the dataset."""
    recons = []
    for i in range(0, len(X), batch_size):
        batch = X[i: i + batch_size]
        recon = vae(batch, training=False).numpy()
        recons.append(recon)
    recons = np.concatenate(recons, axis=0)
    mse    = np.mean((X - recons) ** 2)
    psnr   = compute_psnr(X, recons)
    return {"mse": float(mse), "psnr": float(psnr), "reconstructions": recons}