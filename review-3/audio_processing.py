"""
Audio Preprocessing Pipeline for RAVDESS Emotional Speech Dataset
=================================================================
Converts raw .wav files into 3-channel spectrogram representations:
  Channel 0: Log-Mel Spectrogram  — captures timbral energy distribution
  Channel 1: Delta (first-order diff) — captures rate of spectral change
  Channel 2: Delta-Delta (second-order diff) — captures acceleration of change

Why 3 channels?
  Emotions are not only in WHAT frequencies are active (Mel), but HOW they
  change over time (Delta) and the dynamics of that change (Delta-Delta).
  Stacking these mirrors how CNNs use RGB channels, giving richer spatial
  structure for the network to learn from.
"""

import os
import glob
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── Global config ───────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050   # Standard SR; RAVDESS files are 24kHz but we resample
DURATION      = 3.0     # Fixed duration in seconds (most RAVDESS clips ≤ 3s)
N_MELS        = 128     # Mel filter banks → spatial resolution along freq axis
N_FFT         = 2048    # FFT window; larger = better freq resolution
HOP_LENGTH    = 512     # Hop size; controls time resolution
TARGET_SIZE   = (128, 128)  # Final H×W of spectrogram image

# RAVDESS emotion code → label mapping (from filename field 3)
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def load_audio(filepath: str) -> np.ndarray:
    """Load .wav and resample to SAMPLE_RATE; pad/trim to fixed DURATION."""
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    target_len = int(SAMPLE_RATE * DURATION)
    # Pad with zeros if shorter; trim if longer
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return y


def extract_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Compute log-Mel spectrogram.
    We use log (power_to_db) because human perception of loudness is logarithmic,
    and neural nets converge faster on log-compressed features.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_S = librosa.power_to_db(S, ref=np.max)   # shape: (N_MELS, T)
    return log_S


def extract_deltas(log_mel: np.ndarray):
    """
    Compute 1st and 2nd order temporal derivatives of the Mel spectrogram.
    Delta captures spectral velocity; Delta-Delta captures spectral acceleration.
    Together they encode prosodic dynamics essential for emotion recognition.
    """
    delta   = librosa.feature.delta(log_mel, order=1)
    delta2  = librosa.feature.delta(log_mel, order=2)
    return delta, delta2


def resize_to_target(arr: np.ndarray, target=(128, 128)) -> np.ndarray:
    """
    Resize 2D spectrogram to TARGET_SIZE via bilinear interpolation.
    We use scipy/numpy interpolation here to avoid TF dependency in preprocessing.
    """
    from scipy.ndimage import zoom
    h, w = arr.shape
    zh = target[0] / h
    zw = target[1] / w
    return zoom(arr, (zh, zw), order=1)


def normalize_channel(arr: np.ndarray) -> np.ndarray:
    """
    Normalize each channel independently to [0, 1].
    Independent normalization ensures each channel's range is fully utilised
    by the network, preventing one channel from dominating gradients.
    """
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def extract_3channel_spectrogram(filepath: str) -> np.ndarray:
    """
    Full pipeline: wav → 3-channel spectrogram image of shape (H, W, 3).
    Channel 0: normalized log-Mel
    Channel 1: normalized Delta
    Channel 2: normalized Delta-Delta
    """
    y       = load_audio(filepath)
    log_mel = extract_mel_spectrogram(y)
    delta, delta2 = extract_deltas(log_mel)

    # Resize each to TARGET_SIZE
    ch0 = resize_to_target(log_mel, TARGET_SIZE)
    ch1 = resize_to_target(delta,   TARGET_SIZE)
    ch2 = resize_to_target(delta2,  TARGET_SIZE)

    # Normalize independently
    ch0 = normalize_channel(ch0)
    ch1 = normalize_channel(ch1)
    ch2 = normalize_channel(ch2)

    # Stack → (H, W, 3)
    return np.stack([ch0, ch1, ch2], axis=-1)


def extract_mfcc(filepath: str, n_mfcc: int = 40) -> np.ndarray:
    """
    Extract MFCC features for secondary comparison experiments.
    Returns mean + std over time → flat 1D vector of length 2*n_mfcc.
    MFCCs are compact and fast but lose spatial structure compared to spectrograms.
    """
    y = load_audio(filepath)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=n_mfcc,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def parse_emotion_from_filename(filepath: str) -> str:
    """
    RAVDESS filename format:
      Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Emotion code is field index 2 (0-based).
    Example: 03-01-04-01-01-01-12.wav → emotion code "04" → "sad"
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts    = basename.split("-")
    code     = parts[2]
    return EMOTION_MAP.get(code, "unknown")


def build_dataset(data_dir: str, verbose: bool = True):
    """
    Traverse RAVDESS directory, extract features + labels for all .wav files.
    Returns:
        X_spec : np.ndarray shape (N, H, W, 3) — 3-channel spectrograms
        X_mfcc : np.ndarray shape (N, 80)      — MFCC features
        y      : np.ndarray shape (N,)          — integer class labels
        le     : LabelEncoder
        label_names : list of str
    """
    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {data_dir}. "
                                "Please download RAVDESS and place .wav files there.")
    if verbose:
        print(f"Found {len(wav_files)} .wav files")

    spectrograms, mfccs, labels = [], [], []
    from tqdm import tqdm
    for fp in tqdm(wav_files, desc="Extracting features", disable=not verbose):
        try:
            emotion = parse_emotion_from_filename(fp)
            if emotion == "unknown":
                continue
            spec = extract_3channel_spectrogram(fp)
            mfcc = extract_mfcc(fp)
            spectrograms.append(spec)
            mfccs.append(mfcc)
            labels.append(emotion)
        except Exception as e:
            if verbose:
                print(f"  Skipping {fp}: {e}")
            continue

    X_spec = np.array(spectrograms, dtype=np.float32)
    X_mfcc = np.array(mfccs,       dtype=np.float32)
    y_raw  = np.array(labels)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    if verbose:
        print(f"Dataset shape: {X_spec.shape}, Classes: {le.classes_}")

    return X_spec, X_mfcc, y, le, list(le.classes_)


def split_dataset(X_spec, X_mfcc, y, random_state=42):
    """
    Stratified 70/15/15 split to maintain class balance across splits.
    Stratification is critical for RAVDESS (8 emotion classes, ~180 samples each).
    """
    # Train 70% vs temp 30%
    X_s_tr, X_s_tmp, X_m_tr, X_m_tmp, y_tr, y_tmp = train_test_split(
        X_spec, X_mfcc, y, test_size=0.30, stratify=y, random_state=random_state
    )
    # Val 15% / Test 15%  (50/50 of temp)
    X_s_val, X_s_te, X_m_val, X_m_te, y_val, y_te = train_test_split(
        X_s_tmp, X_m_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=random_state
    )
    splits = {
        "X_spec_train": X_s_tr,  "X_mfcc_train": X_m_tr,  "y_train": y_tr,
        "X_spec_val":   X_s_val, "X_mfcc_val":   X_m_val, "y_val":   y_val,
        "X_spec_test":  X_s_te,  "X_mfcc_test":  X_m_te,  "y_test":  y_te,
    }
    return splits


def compute_dataset_stats(X: np.ndarray):
    """Return per-channel mean and std for standardization (optional path)."""
    stats = {}
    for c in range(X.shape[-1]):
        ch = X[..., c]
        stats[f"ch{c}_mean"] = ch.mean()
        stats[f"ch{c}_std"]  = ch.std()
    return stats
