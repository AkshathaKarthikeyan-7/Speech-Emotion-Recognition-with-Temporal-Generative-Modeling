"""
Generative Adversarial Network (GAN) for Spectrogram Synthesis
===============================================================
Objective: Train a GAN to generate realistic 3-channel mel spectrograms.
Use case: GAN-generated spectrograms can augment minority emotion classes
          in RAVDESS to address class imbalance and improve classifier robustness.

GAN min-max game:
  D tries to maximise log D(x) + log(1-D(G(z)))  [distinguish real vs fake]
  G tries to minimise log(1-D(G(z)))              [fool discriminator]

Stability techniques used:
  1. Label smoothing (real labels = 0.9, not 1.0) — prevents D from becoming
     too confident, which would kill G's gradients early in training.
  2. BatchNormalization in both G and D — normalises internal activations,
     reducing internal covariate shift and stabilising training dynamics.
  3. LeakyReLU in D (not ReLU) — allows small negative gradients, preventing
     dead neurons in the discriminator.
  4. Separate optimisers with balanced update frequency (1:1 D:G steps by default).
  5. tf.image.resize in generator output — robust to any latent → spatial mapping.

Mode collapse:
  Mode collapse occurs when G maps multiple latent vectors to the same output.
  We mitigate via: diverse noise sampling, label smoothing, monitoring D/G loss ratio.
  Healthy D/G ratio: D_loss ≈ G_loss; if D_loss ≪ G_loss, D is winning too fast.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

# ─── Config ──────────────────────────────────────────────────────────────────
NOISE_DIM      = 128
IMAGE_SHAPE    = (128, 128, 3)
LR_GAN         = 2e-4
LABEL_SMOOTH   = 0.1   # Real labels = 1 - LABEL_SMOOTH = 0.9


# ─── Generator ───────────────────────────────────────────────────────────────

def build_generator(noise_dim=NOISE_DIM, output_shape=IMAGE_SHAPE):
    """
    Generator: noise vector z → fake spectrogram.
    Architecture: Dense → Reshape → 4× ConvTranspose+UpSample → output.
    Uses BatchNorm + ReLU in all hidden layers.
    Final tf.image.resize ensures exact output spatial dimensions.
    Sigmoid output → [0, 1] to match normalised spectrogram range.
    """
    z_inp = keras.Input(shape=(noise_dim,), name="noise_input")

    # Project and reshape to spatial start
    x = layers.Dense(8 * 8 * 256, use_bias=False)(z_inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)

    # Upsample blocks
    for filters in [128, 64, 32, 16]:
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same",
                                   use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Final conv → 3 channels
    x = layers.Conv2DTranspose(output_shape[-1], 4, strides=1, padding="same",
                               use_bias=False)(x)

    # tf.image.resize: robust output shape (avoids Cropping2D issues)
    x = layers.Lambda(
        lambda t: tf.image.resize(t, (output_shape[0], output_shape[1])),
        name="resize_output"
    )(x)
    x = layers.Activation("sigmoid", name="fake_spectrogram")(x)

    return keras.Model(z_inp, x, name="Generator")


# ─── Discriminator ───────────────────────────────────────────────────────────

def build_discriminator(input_shape=IMAGE_SHAPE):
    """
    Discriminator: spectrogram image → probability of being real.
    Architecture: 4× Conv+BN+LeakyReLU+Dropout → Flatten → Dense(1, sigmoid).
    LeakyReLU (alpha=0.2): allows gradient flow for negative activations,
    preventing dead neurons that would stop D from learning.
    Dropout: prevents D from memorising specific real samples (overfitting).
    """
    img_inp = keras.Input(shape=input_shape, name="discriminator_input")

    x = img_inp
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="sigmoid", name="real_or_fake")(x)

    return keras.Model(img_inp, x, name="Discriminator")


# ─── Loss Functions ───────────────────────────────────────────────────────────

bce = keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output, label_smooth=LABEL_SMOOTH):
    """
    D loss = BCE(real, 0.9) + BCE(fake, 0.0)
    Label smoothing on real labels (0.9 instead of 1.0) prevents D from
    becoming overconfident, preserving gradient signal for G.
    """
    real_labels = tf.ones_like(real_output) * (1.0 - label_smooth)
    fake_labels = tf.zeros_like(fake_output)
    return bce(real_labels, real_output) + bce(fake_labels, fake_output)


def generator_loss(fake_output):
    """
    G loss = BCE(fake, 1.0)  [G wants D to classify fakes as real]
    Using real labels (1.0) for fakes because G tries to maximise D's error.
    """
    return bce(tf.ones_like(fake_output), fake_output)


# ─── GAN Trainer ─────────────────────────────────────────────────────────────

class GAN:
    """
    Encapsulates G, D, optimisers, and the training step.
    Training loop:
      1. Sample random noise z ~ N(0, I)
      2. Generate fake = G(z)
      3. Compute D loss on [real, fake] → update D
      4. Sample fresh noise z' ~ N(0, I)
      5. Compute G loss via D(G(z')) → update G
    Separate noise for D and G steps avoids gradient interference.
    """

    def __init__(self, noise_dim=NOISE_DIM, image_shape=IMAGE_SHAPE,
                 lr_g=LR_GAN, lr_d=LR_GAN, label_smooth=LABEL_SMOOTH):
        self.noise_dim    = noise_dim
        self.image_shape  = image_shape
        self.label_smooth = label_smooth

        self.generator     = build_generator(noise_dim, image_shape)
        self.discriminator = build_discriminator(image_shape)

        self.g_optimizer = keras.optimizers.Adam(lr_g, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(lr_d, beta_1=0.5)

        self.g_loss_history = []
        self.d_loss_history = []

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise      = tf.random.normal((batch_size, self.noise_dim))

        # ── Discriminator step ──
        with tf.GradientTape() as d_tape:
            fake_images  = self.generator(noise, training=True)
            real_output  = self.discriminator(real_images,  training=True)
            fake_output  = self.discriminator(fake_images,  training=True)
            d_loss       = discriminator_loss(real_output, fake_output,
                                              self.label_smooth)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )

        # ── Generator step (fresh noise) ──
        noise2 = tf.random.normal((batch_size, self.noise_dim))
        with tf.GradientTape() as g_tape:
            fake_images2 = self.generator(noise2, training=True)
            fake_output2 = self.discriminator(fake_images2, training=True)
            g_loss       = generator_loss(fake_output2)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        return d_loss, g_loss

    def train(self, dataset, epochs=50, verbose_every=5):
        """Train GAN and record loss history."""
        for epoch in range(1, epochs + 1):
            d_losses, g_losses = [], []
            for batch in dataset:
                d_l, g_l = self.train_step(batch)
                d_losses.append(float(d_l))
                g_losses.append(float(g_l))

            epoch_d = np.mean(d_losses)
            epoch_g = np.mean(g_losses)
            self.d_loss_history.append(epoch_d)
            self.g_loss_history.append(epoch_g)

            if epoch % verbose_every == 0 or epoch == 1:
                ratio = epoch_d / (epoch_g + 1e-8)
                print(f"Epoch {epoch:3d}/{epochs} | D loss: {epoch_d:.4f} | "
                      f"G loss: {epoch_g:.4f} | D/G ratio: {ratio:.3f}")

    def generate(self, n_samples: int = 16) -> np.ndarray:
        """Generate n fake spectrograms."""
        noise = tf.random.normal((n_samples, self.noise_dim))
        return self.generator(noise, training=False).numpy()

    def generate_for_class(self, n_samples: int = 20) -> np.ndarray:
        """
        Generate samples for augmentation.
        (Class conditioning can be added by concatenating class one-hot to noise.)
        """
        return self.generate(n_samples)

    def assess_mode_collapse(self) -> dict:
        """
        Mode collapse heuristic:
          - Generate 64 samples and measure pairwise L2 diversity.
          - Pairs sampled WITH replacement (replace=True) — avoids the
            ValueError that occurs when requesting more pairs than the
            population allows without replacement.
          - Self-pairs (a == b) are skipped so distance is never trivially 0.
          - Low mean diversity (< 10.0) → HIGH collapse risk.
          - D/G loss ratio: healthy ≈ 1.0.
              ratio << 1 → D losing, G may be producing noise.
              ratio >> 1 → D winning too fast, G gradients vanishing.
        """
        # ── Generate samples ──────────────────────────────────────────────
        n_samples = 64                              # pool size for diversity test
        samples   = self.generate(n_samples)        # (64, H, W, C)
        flat      = samples.reshape(n_samples, -1)  # (64, H*W*C)

        # ── Pairwise diversity (replace=True avoids population size issue) ─
        rng   = np.random.default_rng(42)
        pairs   = rng.choice(100, size=(200, 2), replace=True)

        dists = []
        for a, b in pairs:
            if a != b:                              # skip zero-distance self-pairs
                dists.append(np.linalg.norm(flat[a] - flat[b]))

        mean_div = float(np.mean(dists)) if dists else 0.0

        # ── D/G ratio from last 5 epochs (smoothed) ──────────────────────
        if self.d_loss_history and self.g_loss_history:
            d_fin = float(np.mean(self.d_loss_history[-5:]))
            g_fin = float(np.mean(self.g_loss_history[-5:]))
            final_ratio = d_fin / (g_fin + 1e-8)
        else:
            final_ratio = 1.0

        # ── Risk assessment ───────────────────────────────────────────────
        if mean_div < 5.0:
            risk = "HIGH — very low diversity, likely mode collapse"
        elif mean_div < 10.0:
            risk = "MEDIUM — moderate diversity, monitor training"
        else:
            risk = "LOW — good sample diversity"

        return {
            "mean_pairwise_diversity": mean_div,
            "final_d_g_ratio":        final_ratio,
            "mode_collapse_risk":     risk,
        }

    def spectrogram_to_audio(self, spectrogram: np.ndarray,
                              sr: int = 22050, n_iter: int = 60) -> np.ndarray:
        """
        Griffin-Lim algorithm to convert log-Mel spectrogram (channel 0) back
        to a waveform. This is an iterative phase reconstruction method.
        Steps:
          1. Take channel 0 (log-Mel) and convert from [0,1] back to dB scale
          2. Convert dB → power spectrum
          3. Apply Griffin-Lim to estimate phase and reconstruct waveform
        Note: Griffin-Lim is approximate; generated audio quality is limited
        by the invertibility of the Mel filterbank (many-to-one mapping).
        """
        import librosa
        from scipy.ndimage import zoom

        # Channel 0: normalised log-Mel → rescale to [-80, 0] dB range
        log_mel_norm  = spectrogram[..., 0]
        log_mel_db    = log_mel_norm * 80.0 - 80.0
        power         = librosa.db_to_power(log_mel_db)

        n_mels  = 128
        hop_len = 512
        n_fft   = 2048

        h, w          = power.shape
        power_resized = zoom(power, (n_mels / h, 1.0), order=1)

        mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_inv     = np.linalg.pinv(mel_filters)
        linear_S    = np.maximum(1e-8, mel_inv @ power_resized)

        audio = librosa.griffinlim(linear_S, n_iter=n_iter, hop_length=hop_len,
                                   win_length=n_fft)
        return audio


def build_gan_dataset(X_spec: np.ndarray, batch_size: int = 32):
    """Create tf.data.Dataset from spectrogram array for GAN training."""
    dataset = tf.data.Dataset.from_tensor_slices(X_spec.astype(np.float32))
    dataset = dataset.shuffle(buffer_size=1024, seed=42)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset