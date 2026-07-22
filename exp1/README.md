# ♻️ Eco-Sorter — AI Garbage Classifier

A deep-learning web application that classifies a photo of a waste item into **10 material categories** and tells you which of **3 municipal bins** it belongs to. Built with a fine-tuned **MobileNetV2** model, a **Flask** backend, and a polished drag-and-drop web UI.

![Python](https://img.shields.io/badge/python-3.12-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange) ![Flask](https://img.shields.io/badge/Flask-3.1-green)

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Model Details](#-model-details)
- [Bin Mapping](#-bin-mapping)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running the App](#-running-the-app)
- [Retraining the Model](#-retraining-the-model-optional)
- [API Reference](#-api-reference)
- [Pushing to a Git Repository](#-pushing-to-a-git-repository)
- [Troubleshooting](#-troubleshooting)

---

## 🌍 Overview

Upload (or drag & drop) an image of a waste item. The app:
1. Resizes it to **256×256** and runs it through the trained CNN.
2. Predicts the exact material (e.g. *plastic*, *battery*, *cardboard*).
3. Maps that material to the correct **municipal bin** and animates the matching bin in the UI, with a confidence score.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Architecture** | MobileNetV2 (transfer learning, ImageNet weights) |
| **Base** | `include_top=False`, frozen convolutional base |
| **Head** | `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(10, softmax)` |
| **Input shape** | `(256, 256, 3)` |
| **Preprocessing** | `mobilenet_v2.preprocess_input` is **baked into the model graph** — feed raw `0–255` RGB values, no manual scaling |
| **Data augmentation** | RandomFlip (H/V), RandomRotation(0.2), RandomZoom(0.1) — inside the model, active only during training |
| **Optimizer** | AdamW (`lr=1e-3`, `weight_decay=1e-4`) |
| **Loss** | Categorical cross-entropy |
| **Epochs** | 10 (with EarlyStopping & ReduceLROnPlateau) |
| **Dataset** | [Garbage Classification v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) (`standardized_256`) via `kagglehub` |
| **Split** | 70% train / 15% val / 15% test |
| **Test accuracy** | **≈ 90.1 %** (test loss ≈ 0.337) |

### Classes (10, alphabetical — this order is fixed)
```
['battery', 'biological', 'cardboard', 'clothes', 'glass',
 'metal', 'paper', 'plastic', 'shoes', 'trash']
```

> ⚠️ The class order **must** match the training order. The model outputs a softmax vector indexed exactly as above — reordering this list will silently corrupt predictions.

---

## 🗑️ Bin Mapping

The 10 classes collapse into **3 municipal bins**:

| Bin | Color | Classes |
|---|---|---|
| **Biodegradable** | 🟢 Green | `biological` |
| **Recyclable** | 🔵 Blue | `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes` |
| **Non-recyclable / Hazardous** | ⚫ Black | `trash`, `battery` |

---

## 📁 Project Structure

```
exp1/
├── app.py                     # Flask server — loads model once, serves UI + /predict
├── garbage_classifier.py      # Training pipeline (MobileNetV2)
├── predict.py                 # CLI inference tool (terminal)
├── requirements.txt           # Python dependencies
├── .gitignore
├── README.md
├── templates/
│   └── index.html             # Web UI
├── static/
│   ├── style.css              # Styling (eco palette, animated bins)
│   └── script.js              # Drag-drop + result animation
└── outputs/
    ├── models/
    │   └── best_model.keras   # ✅ Trained model (committed, ~9.3 MB)
    └── metrics/
        └── *.json             # Config + test metrics (committed)
```

> `outputs/logs/` and `outputs/images/` are git-ignored (large / regenerable). See [.gitignore](.gitignore).

---

## 🚀 Setup & Installation

**Prerequisites:** Python **3.12** (3.10+ works), `git`.

### 1. Clone the repository
```bash
git clone <YOUR_REPO_URL>
cd <repo-folder>/exp1
```

### 2. Create & activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
> If activation is blocked by execution policy, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes: `tensorflow`, `numpy`, `matplotlib`, `kagglehub`, `flask`, `pillow`.

---

## ▶️ Running the App

With the venv activated:

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

- The model loads **once** at startup (first boot takes a few seconds while TensorFlow initializes).
- The server listens on `0.0.0.0:5000`, so it's also reachable from other devices on your network at `http://<your-machine-ip>:5000`.

To stop the server: **Ctrl + C**.

---

## 🔁 Retraining the Model (optional)

Only needed if you want to rebuild `best_model.keras` from scratch. The dataset auto-downloads via `kagglehub` (Kaggle credentials may be required):

```bash
python garbage_classifier.py
```

This regenerates `outputs/models/best_model.keras` plus logs, metrics, and a sample inference image.

**CLI inference** (no web UI):
```bash
python predict.py
# then paste an image path when prompted
```

---

## 🔌 API Reference

### `GET /`
Serves the web UI.

### `POST /predict`
Multipart form upload.

| Field | Type | Required |
|---|---|---|
| `image` | file (PNG/JPG/JPEG/WEBP/BMP/GIF) | ✅ |

**Success `200`:**
```json
{
  "predicted_class": "battery",
  "confidence": 99.49,
  "bin_id": "non-recyclable",
  "bin_label": "Non-recyclable / Hazardous",
  "breakdown": [
    { "class": "battery", "confidence": 99.49 },
    { "class": "cardboard", "confidence": 0.23 }
  ]
}
```

**Error responses:**
| Status | Meaning |
|---|---|
| `400` | No image / empty filename / unsupported type / unreadable image |
| `500` | Prediction failed |
| `503` | Model not loaded (missing `best_model.keras`) |

Max upload size: **16 MB**.

---

## 📤 Pushing to a Git Repository

The trained model (`best_model.keras`, ~9.3 MB) is **committed on purpose** so the app runs immediately after cloning — no retraining needed. At ~9 MB it is well under GitHub's 100 MB per-file limit, so **Git LFS is not required**.

From the **`exp1/`** directory:

```bash
# 1. Initialize (skip if the repo already exists)
git init

# 2. Stage everything (respecting .gitignore — venv/ and logs are excluded)
git add .

# 3. Verify the model is included and venv is NOT
git status
#   ✅ outputs/models/best_model.keras  → should be staged
#   🚫 venv/                            → should NOT appear

# 4. Commit
git commit -m "Eco-Sorter: garbage classifier web app + trained model"

# 5. Link your remote and push
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

> **If the model file ever grows past ~50 MB**, switch to Git LFS:
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes outputs/models/best_model.keras
> git commit -m "Track model with Git LFS"
> ```

### What gets committed vs. ignored
| Committed ✅ | Ignored 🚫 |
|---|---|
| `app.py`, `predict.py`, `garbage_classifier.py` | `venv/` |
| `templates/`, `static/` | `outputs/logs/`, `outputs/images/` |
| `outputs/models/best_model.keras` | `outputs/models/final_model_*.keras` |
| `outputs/metrics/*.json` | `__pycache__/`, `*.log` |
| `requirements.txt`, `.gitignore`, `README.md` | OS/editor cruft |

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `503 Model is not available` | `outputs/models/best_model.keras` is missing — retrain with `python garbage_classifier.py` or ensure the file was cloned. |
| PowerShell won't activate venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then re-run the activate command. |
| `TensorFlow GPU support is not available on native Windows` | Harmless warning — inference runs on CPU. For GPU, use WSL2. |
| Predictions look random | The `CLASS_NAMES` order was changed — restore the exact alphabetical list above. |
| Port 5000 already in use | Edit the `port=5000` argument at the bottom of `app.py`. |

---

*Eco-Sorter · 10-class garbage classifier · MobileNetV2 · Flask*
