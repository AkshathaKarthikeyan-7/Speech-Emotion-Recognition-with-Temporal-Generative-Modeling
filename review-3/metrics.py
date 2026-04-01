"""
Evaluation Metrics and Visualisation Utilities
===============================================
All plots saved to outputs/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.makedirs("outputs", exist_ok=True)

# Emotion colour palette — unique colour per emotion for visual consistency
EMOTION_COLORS = {
    "neutral":   "#6c757d",
    "calm":      "#17a2b8",
    "happy":     "#ffc107",
    "sad":       "#007bff",
    "angry":     "#dc3545",
    "fearful":   "#6f42c1",
    "disgust":   "#28a745",
    "surprised": "#fd7e14",
}


def get_color_list(label_names):
    return [EMOTION_COLORS.get(n, "#333333") for n in label_names]


# ─── Classification Report ───────────────────────────────────────────────────

def compute_classification_metrics(y_true, y_pred, label_names):
    """Return accuracy, per-class precision/recall/F1 and macro averages."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(label_names)))
    )
    macro_f1 = f1.mean()
    report = {
        "accuracy":   acc,
        "macro_f1":   macro_f1,
        "per_class":  {
            label_names[i]: {"precision": precision[i], "recall": recall[i],
                             "f1": f1[i], "support": support[i]}
            for i in range(len(label_names))
        }
    }
    print(f"\nAccuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_names))
    return report


# ─── Confusion Matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, label_names,
                           title="Confusion Matrix", save_path=None):
    """Plot raw counts and normalised confusion matrix side by side."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, fmt, t in zip(axes, [cm, cm_norm], ["d", ".2f"],
                                 [f"{title} (Counts)", f"{title} (Normalised)"]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax,
                    linewidths=0.5, annot_kws={"size": 9})
        ax.set_title(t, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return cm, cm_norm


# ─── Class-wise F1 Bar Chart ─────────────────────────────────────────────────

def plot_f1_bar_chart(f1_scores: dict, label_names, title="Class-wise F1 Score",
                      save_path=None):
    """
    Colour-coded F1 bar chart:
      green  (F1 ≥ 0.7) — good performance
      orange (0.5 ≤ F1 < 0.7) — moderate
      red    (F1 < 0.5) — poor, needs attention
    """
    f1_vals = [f1_scores[n]["f1"] for n in label_names]
    colors  = ["#28a745" if v >= 0.7 else "#ffc107" if v >= 0.5 else "#dc3545"
               for v in f1_vals]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(label_names, f1_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Emotion Class", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.axhline(0.7, color="green",  linestyle="--", alpha=0.5, label="Good (0.7)")
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="Moderate (0.5)")
    ax.legend()
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Training Curves ─────────────────────────────────────────────────────────

def plot_vae_training_curves(history, save_path=None):
    """
    3-subplot figure: total loss, reconstruction loss, KL divergence loss.
    Plotting separately makes it easy to diagnose:
      - Rising KL with stable recon → KL weight too high
      - Rising recon with stable KL → model is underfitting
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    keys = [("total_loss", "val_total_loss", "Total Loss"),
            ("reconstruction_loss", "val_reconstruction_loss", "Reconstruction Loss"),
            ("kl_loss", "val_kl_loss", "KL Divergence Loss")]
    for ax, (train_k, val_k, title) in zip(axes, keys):
        if train_k in history:
            ax.plot(history[train_k], label="Train", color="#007bff")
        if val_k in history:
            ax.plot(history[val_k], label="Val", color="#dc3545", linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("VAE Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_classifier_curves(history, model_name="Classifier", save_path=None):
    """Plot loss and accuracy curves for classification models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.get("loss", []), label="Train", color="#007bff")
    ax1.plot(history.get("val_loss", []), label="Val", color="#dc3545", linestyle="--")
    ax1.set_title(f"{model_name} — Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.get("accuracy", []), label="Train", color="#007bff")
    ax2.plot(history.get("val_accuracy", []), label="Val", color="#dc3545", linestyle="--")
    ax2.set_title(f"{model_name} — Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle(f"{model_name} Training Dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Spectrogram Visualisation ───────────────────────────────────────────────

def plot_3channel_spectrogram(spec_3ch: np.ndarray, title="3-Channel Spectrogram",
                               save_path=None):
    """
    Side-by-side plot of Mel, Delta, Delta-Delta channels.
    Visualising all 3 channels reveals:
      - Mel: which frequencies are active (spectral envelope)
      - Delta: where spectral content changes rapidly (onset/offset)
      - Delta-Delta: where pitch/energy transitions accelerate (prosodic events)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Channel 0: Log-Mel\n(Spectral Energy)",
              "Channel 1: Delta\n(Spectral Velocity)",
              "Channel 2: Delta-Delta\n(Spectral Acceleration)"]
    cmaps  = ["magma", "coolwarm", "RdYlBu"]
    for ax, ch, t, cm in zip(axes, range(3), titles, cmaps):
        im = ax.imshow(spec_3ch[:, :, ch], aspect="auto", origin="lower",
                       cmap=cm, interpolation="nearest")
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reconstruction_comparison(original: np.ndarray, reconstructed: np.ndarray,
                                    n_samples: int = 3, save_path=None):
    """
    Side-by-side original vs reconstructed spectrograms (all 3 channels).
    Visual inspection reveals whether VAE has learned the spectral structure.
    """
    n_samples = min(n_samples, len(original))
    fig, axes = plt.subplots(n_samples * 2, 3, figsize=(15, n_samples * 4))
    ch_names  = ["Log-Mel", "Delta", "Delta²"]
    cmaps     = ["magma", "coolwarm", "RdYlBu"]

    for i in range(n_samples):
        for ch in range(3):
            row_o = i * 2
            row_r = i * 2 + 1
            axes[row_o, ch].imshow(original[i, :, :, ch], aspect="auto",
                                    origin="lower", cmap=cmaps[ch])
            axes[row_o, ch].set_title(f"Sample {i+1} | Original | {ch_names[ch]}",
                                       fontsize=9)
            axes[row_r, ch].imshow(reconstructed[i, :, :, ch], aspect="auto",
                                    origin="lower", cmap=cmaps[ch])
            axes[row_r, ch].set_title(f"Sample {i+1} | Reconstructed | {ch_names[ch]}",
                                       fontsize=9)
            for row in [row_o, row_r]:
                axes[row, ch].axis("off")

    plt.suptitle("Original vs Reconstructed Spectrograms (VAE)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Latent Space Visualisation ──────────────────────────────────────────────

def plot_pca_latent_space(z_mean: np.ndarray, y: np.ndarray, label_names: list,
                           save_path=None):
    """
    PCA reduces z_mean to 2D. Plots scatter with centroids.
    PCA shows global structure: if emotions are linearly separable in latent space,
    a simple linear classifier would suffice.
    Centroids help visualise inter-class distances.
    """
    pca  = PCA(n_components=2, random_state=42)
    z2d  = pca.fit_transform(z_mean)
    colors = get_color_list(label_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (name, col) in enumerate(zip(label_names, colors)):
        mask = y == i
        ax.scatter(z2d[mask, 0], z2d[mask, 1], c=col, alpha=0.6, s=30, label=name)
        cx, cy = z2d[mask, 0].mean(), z2d[mask, 1].mean()
        ax.scatter(cx, cy, c=col, s=200, marker="*", edgecolors="black", linewidths=0.8,
                   zorder=5)
        ax.annotate(name, (cx, cy), fontsize=8, fontweight="bold",
                    textcoords="offset points", xytext=(5, 5))

    ax.set_title("PCA of VAE Latent Space (z_mean) with Centroids",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return z2d, pca


def plot_tsne_latent_space(z_mean: np.ndarray, y: np.ndarray, label_names: list,
                            perplexity: int = 30, save_path=None):
    """
    t-SNE reveals local cluster structure not visible in PCA.
    Density contours (KDE) highlight where each emotion's samples concentrate.
    Overlapping contours → emotions that share perceptual features.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
            learning_rate="auto", init="pca", max_iter=1000)
    z2d  = tsne.fit_transform(z_mean)
    colors = get_color_list(label_names)

    fig, ax = plt.subplots(figsize=(11, 9))
    for i, (name, col) in enumerate(zip(label_names, colors)):
        mask = y == i
        pts  = z2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=col, alpha=0.5, s=25, label=name,
                   zorder=2)
        # KDE density contour
        if pts.shape[0] > 10:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(pts.T, bw_method=0.35)
                gx  = np.linspace(z2d[:, 0].min(), z2d[:, 0].max(), 80)
                gy  = np.linspace(z2d[:, 1].min(), z2d[:, 1].max(), 80)
                GX, GY = np.meshgrid(gx, gy)
                Z  = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
                ax.contour(GX, GY, Z, levels=4, colors=[col], alpha=0.4,
                           linewidths=0.8, zorder=3)
            except Exception:
                pass

    ax.set_title("t-SNE of VAE Latent Space with Density Contours",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return z2d


# ─── Attention Weights ────────────────────────────────────────────────────────

def plot_attention_weights(attention_weights: np.ndarray, y_labels: list,
                            label_names: list, n_per_class: int = 2, save_path=None):
    """
    Bar charts showing attention weight distribution over time frames per emotion.
    High attention at specific time positions → model focuses on those phonetic/
    prosodic features. E.g. angry speech peaks at onset (loud burst);
    sad speech has more uniform or late attention (trailing off).
    """
    n_classes = len(label_names)
    fig, axes = plt.subplots(n_classes, 1, figsize=(14, n_classes * 2.5), sharex=False)

    for i, (name, ax) in enumerate(zip(label_names, axes)):
        mask = [j for j, l in enumerate(y_labels) if l == i][:n_per_class]
        if not mask:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center")
            ax.set_ylabel(name, fontsize=9)
            continue
        mean_attn = np.mean([attention_weights[j].squeeze() for j in mask], axis=0)
        color = EMOTION_COLORS.get(name, "#333333")
        ax.bar(range(len(mean_attn)), mean_attn, color=color, alpha=0.8, width=0.9)
        ax.set_ylabel(f"{name}\n(attn)", fontsize=9)
        ax.set_xlim(-0.5, len(mean_attn) - 0.5)
        ax.axhline(1.0 / len(mean_attn), color="gray", linestyle="--", alpha=0.5,
                   label="Uniform")
        ax.set_ylim(0, mean_attn.max() * 1.3)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Time Frame Index")
    plt.suptitle("Bahdanau Attention Weights per Emotion Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── GAN Plots ────────────────────────────────────────────────────────────────

def plot_gan_generated_spectrograms(generated: np.ndarray, n: int = 8,
                                     save_path=None):
    """Display a grid of GAN-generated 3-channel spectrograms (showing ch0 = Mel)."""
    n = min(n, len(generated))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i in range(n):
        axes[i].imshow(generated[i, :, :, 0], aspect="auto", origin="lower",
                       cmap="magma", interpolation="nearest")
        axes[i].set_title(f"Generated #{i+1}", fontsize=9)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle("GAN-Generated Spectrograms (Channel 0: Log-Mel)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gan_training_curves(d_losses: list, g_losses: list, save_path=None):
    """
    Plot D loss vs G loss over epochs.
    Healthy training: both losses converge to ~0.5–0.7.
    D_loss ≪ G_loss: discriminator too strong → generator getting no gradient.
    D_loss ≫ G_loss: generator collapsed or discriminator not learning.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d_losses, label="Discriminator Loss", color="#dc3545", linewidth=2)
    ax.plot(g_losses, label="Generator Loss",     color="#007bff", linewidth=2)
    ax.axhline(0.693, color="gray", linestyle="--", alpha=0.6,
               label="Ideal ≈ 0.693 (log 2)")
    ax.set_title("GAN Training Curves — D Loss vs G Loss",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Model Comparison ────────────────────────────────────────────────────────

def plot_model_comparison(results: dict, metric="accuracy", save_path=None):
    """
    Bar chart comparing multiple models on a given metric.
    results: {"ModelA": 0.82, "ModelB": 0.79, ...}
    """
    names  = list(results.keys())
    values = [results[n] for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Model Comparison — {metric.capitalize()}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel(metric.capitalize())
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hyperparameter_comparison(results: list, save_path=None):
    """
    Structured bar chart for hyperparameter sweep results.
    results: [{"name": "lr=1e-3, bs=16", "val_accuracy": 0.80}, ...]
    """
    names  = [r["name"] for r in results]
    accs   = [r["val_accuracy"] for r in results]
    best   = max(accs)
    colors = ["#28a745" if a == best else "#6c757d" for a in accs]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_title("Hyperparameter Tuning Results", fontsize=13, fontweight="bold")
    ax.set_ylabel("Validation Accuracy")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
