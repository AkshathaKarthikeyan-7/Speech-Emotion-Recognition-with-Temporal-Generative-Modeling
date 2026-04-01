# 🎤 Speech Emotion Recognition — RAVDESS Dataset
## End-to-End Deep Learning System

A complete deep learning pipeline for speech emotion recognition featuring:
- **Residual Convolutional VAE** for unsupervised feature learning
- **DCGAN** for spectrogram synthesis and data augmentation
- **MLP Classifier** on VAE latent embeddings
- **BiLSTM + Bahdanau Attention** for temporal emotion modelling
- **Streamlit app** for real-time inference

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 📁 Dataset Setup

1. Download the **RAVDESS Emotional Speech Audio** dataset:
   - URL: https://zenodo.org/record/1188976
   - File: `Audio_Speech_Actors_01-24.zip`

2. Extract into the `data/` directory:
   ```
   project/
   └── data/
       ├── Actor_01/
       │   ├── 03-01-01-01-01-01-01.wav
       │   ├── 03-01-02-01-01-01-01.wav
       │   └── ...
       ├── Actor_02/
       └── ...
   ```

3. The code recursively finds all `.wav` files, so any nested structure works.

### RAVDESS Filename Format
```
Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
```
| Field | Values |
|-------|--------|
| Modality | 01=AV, 02=Video, 03=Audio |
| Vocal Channel | 01=Speech, 02=Song |
| **Emotion** | **01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised** |
| Intensity | 01=normal, 02=strong |
| Statement | 01="Kids are talking", 02="Dogs are sitting" |
| Repetition | 01, 02 |
| Actor | 01–24 |

---

## 🚀 Running the Project

### Step 1: Generate and run the notebook
```bash
python generate.py        # Creates main.ipynb
jupyter notebook main.ipynb
```

### Step 2: Launch the Streamlit app
```bash
streamlit run app.py
```
> The app requires trained models in `outputs/`. Run the notebook first.

---

## 📁 Project Structure

```
project/
├── data/                           ← RAVDESS .wav files go here
├── preprocessing/
│   └── audio_processing.py         ← Full audio pipeline (load→features→split)
├── models/
│   ├── autoencoder.py              ← Residual VAE with reparameterization
│   ├── gan.py                      ← DCGAN with Griffin-Lim audio synthesis
│   └── classifier.py              ← MLP + BiLSTM+Bahdanau + baselines
├── utils/
│   └── metrics.py                  ← All evaluation metrics and visualisations
├── outputs/                        ← Saved models + plots (auto-created)
├── app.py                          ← Streamlit deployment
├── generate.py                     ← Programmatically creates main.ipynb
├── main.ipynb                      ← Generated experiment notebook
├── requirements.txt
└── README.md
```

---

## 🧠 Architecture Details

### 3-Channel Spectrogram Features
| Channel | Feature | Why |
|---------|---------|-----|
| 0 | Log-Mel Spectrogram | Timbral texture — which frequencies are active |
| 1 | Delta (1st order diff) | Spectral velocity — rate of change |
| 2 | Delta-Delta (2nd order diff) | Spectral acceleration — prosodic dynamics |

### Residual VAE
```
Input (128×128×3)
  → Conv(32) + 4×[ResBlock + MaxPool]
  → Flatten → Dense(512)
  → z_mean(128) ‖ z_log_var(128)
  → z = z_mean + ε·exp(0.5·z_log_var)   [reparameterization]
  → Dense → Reshape → 4×[ConvTranspose + UpSample]
  → tf.image.resize → Sigmoid
Output (128×128×3)
```

**Loss:** `L = 0.8·MSE + 0.2·L1 + 0.001·KL`

### GAN
```
Generator:  z(128) → Dense → Reshape(8×8×256) → 4×ConvTranspose(×2) → sigmoid
Discriminator: (128×128×3) → 4×Conv(stride=2) → Flatten → Dense(1, sigmoid)
```

### MLP Classifier (on z_mean)
```
z_mean(128) → Dense(512)→BN→ReLU→DO(0.4)
            → Dense(256)→BN→ReLU→DO(0.4)
            → Dense(128)→BN→ReLU→DO(0.3)
            → Dense(8, softmax)
```

### BiLSTM + Bahdanau Attention
```
Spectrogram(128×128×3)
  → Sequential CNN (freq pooling only) → shape (batch, T=128, 128)
  → BiLSTM(128) → BiLSTM(64, return_state=True)
  → Bahdanau Attention (query = concatenated final hidden states)
  → Context vector → Dense(128) → Dense(8, softmax)
```

---

## 📊 Output Files

After running `main.ipynb`:

| File | Description |
|------|-------------|
| `outputs/01_3channel_features.png` | 3-channel spectrogram visualisation |
| `outputs/02_vae_training_curves.png` | VAE total/recon/KL loss curves |
| `outputs/03_vae_reconstruction.png` | Original vs reconstructed spectrograms |
| `outputs/04_pca_latent_space.png` | PCA of z_mean with centroids |
| `outputs/05_tsne_latent_space.png` | t-SNE with density contours |
| `outputs/06_gan_training_curves.png` | D loss vs G loss |
| `outputs/07_gan_generated_spectrograms.png` | Generated spectrogram grid |
| `outputs/08_mlp_training_curves.png` | MLP loss + accuracy |
| `outputs/09_mlp_confusion_matrix.png` | Raw + normalised confusion matrix |
| `outputs/10_mlp_f1_barchart.png` | Class-wise F1 (green/orange/red) |
| `outputs/11_bilstm_training_curves.png` | BiLSTM training dynamics |
| `outputs/12_bilstm_confusion_matrix.png` | BiLSTM confusion matrix |
| `outputs/13_bilstm_f1_barchart.png` | BiLSTM class-wise F1 |
| `outputs/14_attention_weights.png` | Bahdanau attention per emotion |
| `outputs/15_hyperparameter_comparison.png` | HP tuning bar chart |
| `outputs/16_model_comparison.png` | All models accuracy comparison |
| `outputs/vae_best.keras` | Saved VAE model |
| `outputs/mlp_best.keras` | Saved MLP classifier |
| `outputs/label_encoder.npy` | Label names array |
| `outputs/gan_generated_audio.wav` | Griffin-Lim reconstructed audio |

---

## 🔑 Key Design Decisions

| Decision | Reason |
|----------|--------|
| `tf.image.resize` instead of `Cropping2D` | Robust to any input size; Cropping2D fails silently on mismatched dimensions |
| `z_mean` for inference (not sampled `z`) | Deterministic + lower variance; KL ensures it's well-distributed |
| KL weight β=0.001 | Preserves reconstruction quality while regularising latent space |
| BATCH_SIZE=16 | More gradient noise → better regularisation on small dataset (~1440 samples) |
| Label smoothing=0.1 | Prevents overconfident predictions; improves generalisation on 8-class problem |
| Balanced class weights | RAVDESS has slight imbalance; ensures minority classes contribute equally to loss |
| Bahdanau (additive) attention | Non-linear alignment: more expressive than dot-product for small speech datasets |

---

## 📈 Expected Results (with full RAVDESS dataset)

| Model | Expected Test Accuracy |
|-------|----------------------|
| Baseline CNN | 55–65% |
| Improved CNN (BN+DO) | 65–72% |
| MLP on VAE z_mean | 68–75% |
| BiLSTM + Attention | 70–78% |
| MLP + GAN Augmentation | 70–77% |

*Results vary based on training duration and random seed.*

---

## 📚 References

- **RAVDESS**: Livingstone & Russo (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0196391
- **VAE**: Kingma & Welling (2014). Auto-Encoding Variational Bayes. *ICLR 2014*.
- **GAN**: Goodfellow et al. (2014). Generative Adversarial Networks. *NeurIPS 2014*.
- **Bahdanau Attention**: Bahdanau et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*.
- **Griffin-Lim**: Griffin & Lim (1984). Signal estimation from modified short-time Fourier transform. *IEEE ASSP*.
