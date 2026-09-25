# DrishtiManas: Retinal OCT Image Classification with a From-Scratch Neural Network

DrishtiManas (Sanskrit: *drishti* "vision" + *manas* "mind") is a full-stack web application. It classifies **retinal
optical coherence tomography (OCT) images** into four classes, using a **multilayer perceptron written from scratch in
NumPy**:

| Class | Meaning | Triage |
| --- | --- | --- |
| **CNV** | Choroidal neovascularization (wet age-related macular degeneration) | Urgent referral |
| **DME** | Diabetic macular edema | Urgent referral |
| **DRUSEN** | Drusen deposits (early age-related macular degeneration) | Routine follow-up |
| **NORMAL** | Healthy retina | None |

```
Upload OCT image → preprocess → feature extraction (pixels + HOG) → MLP (forward pass) → softmax → CNV / DME / DRUSEN / NORMAL
```

This project was built for **LCA1 (M.Tech AI-ML): "Design and Deployment of an AI-Based Web Application for
classification using Neural Networks"**, as an **image-based** classifier.

## Results (held-out OCTMNIST test set: 1,000 images, 250 per class)

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Single-layer perceptron (baseline) | 54.7% | – | – | 53.5% |
| Softmax regression, 0 hidden layers (ours) | 52.9% | – | – | 51.4% |
| scikit-learn MLP 1×100 (baseline MLP) | 62.7% | – | – | 60.9% |
| **DrishtiManas** (3-model ensemble + calibration) | **69.1%** | **73.6%** | **69.1%** | **67.9%** |

- **Architecture:** 64×64 input → 32×32 pixels + HOG features (2,788 dimensions) → 256 ReLU (40% dropout) → 4 softmax,
  715,012 weights per model. Trained with SGD + momentum (learning rate 0.001, decay ×0.97/epoch), batch size 64, L2
  1e-4 and a class-weighted cross-entropy on all 97,477 training images.
- **Served model:** 3 identically configured networks (different random seeds), with their softmax outputs averaged.
  A per-class log-bias, tuned on a class-balanced validation subsample, corrects the network's tendency to
  over-predict CNV. Measured latency for the whole path is about 1 ms per image.
- **Validation:** 87.5% accuracy, 82.3% macro-F1. The test split is deliberately balanced and drawn from different
  patients, so test scores are lower. Published CNN results on OCTMNIST are 74–78% (MedMNIST v2, ResNet-18/50).
- **Per-class test F1:** Normal 83%, CNV 69%, DME 67%, Drusen 52%.
- **Backpropagation gradient check:** max relative error 1.4e-9. The dropout backward pass is also checked separately.

### What each step contributed (test set)

| Step | Val. macro-F1 | Test accuracy | Test macro-F1 | Test Drusen F1 |
| --- | ---: | ---: | ---: | ---: |
| Previous version: 128 ReLU, no dropout, single model | 81.5% | 67.4% | 65.2% | 38.5% |
| Dropout 40% + 256 units, single model | 81.8% | 68.1% | 66.4% | 45.9% |
| + 3-model ensemble | 83.9% | 68.1% | 66.1% | 45.6% |
| **+ class-prior calibration (served)** | 82.3% | **69.1%** | **67.9%** | **52.5%** |

- **Dropout and calibration helped; the ensemble did not.** Averaging 3 models raised validation macro-F1 but left
  the test score unchanged. It stays in the served model because it was chosen on validation, and removing it now
  because of the test result would be choosing on the test set.
- **Only the Drusen gain is statistically significant.** A paired McNemar test on the same 1,000 test images puts
  the overall accuracy gain over the previous version (67.4% → 69.1%) at p = 0.16, which is within noise (the 95%
  confidence interval on accuracy is about ±2.9 points). Drusen recall rose from 26.4% to 40.4%: 45 scans newly
  correct against 10 newly wrong, p < 0.0001.
- **A calibration pitfall.** Tuning the class bias on the full, imbalanced validation split raised validation macro-F1
  (83.9% → 85.1%) but lowered test macro-F1 (66.1% → 60.6%). The correction had learned validation's class mix rather
  than the balanced mix of the test set and of a real deployment. Tuning on a class-balanced subsample fixed it. That
  fix was prompted by seeing the test result, so the test score for this step is slightly optimistic. See §8.2 of the
  report.
- The complete pipeline (feature comparison, baselines, greedy tuning of 9 hyperparameters, ensemble training,
  calibration, ablation) runs in about 40 minutes on a 4-core CPU: `python -m ml.train`.

All numbers, curves and tuning runs are browsable on the app's **Model report** page, and the full write-up — including
the calibration pitfall above — is in [`docs/LCA1_Report_DrishtiManas.docx`](docs/LCA1_Report_DrishtiManas.docx).


---

## How the assignment requirements map to the code

| Requirement | Where |
| --- | --- |
| Public dataset, preprocessing, feature extraction | `ml/data.py` (OCTMNIST loader), `ml/features.py` (grayscale, crop, resize, raw-pixel + HOG features, z-score) |
| Multilayer perceptron: working and limitations | `ml/nn/mlp.py`; the **How it works** page in the web app; the report in `docs/` |
| Neural network with ≥1 hidden layer, forward propagation | `MLP.forward` in `ml/nn/mlp.py` |
| Activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax) | `ml/nn/activations.py` |
| Cost function, plus a plot of loss vs epochs | `ml/nn/losses.py` (cross-entropy and MSE); loss curves on the **Model report** page and in `artifacts/figures/` |
| Backpropagation and convergence | `MLP.loss_and_gradients`; `gradient_check` (numerical verification); loss, accuracy and gradient-norm curves |
| Hyperparameter tuning (≥3 hyperparameters) | `ml/train.py` (greedy search): learning rate, hidden neurons, hidden layers, batch size, activation, optimizer, cost function, L2 penalty, epochs |
| Accuracy, precision, recall, F1, confusion matrix | `ml/metrics.py` (implemented from scratch; cross-checked against scikit-learn in `tests/`) |
| Web application | `frontend/` (React + TypeScript + Tailwind) with `backend/` (FastAPI) |
| Google Cloud deployment | `Dockerfile` and `cloudbuild.yaml` for **Cloud Run**; see `docs/DEPLOYMENT.md` |

The network does **not** use PyTorch, TensorFlow or scikit-learn's MLP. Forward propagation, backpropagation,
the optimizers (SGD, momentum, Adam), L2 regularisation, early stopping and weight initialisation are written by
hand in about 300 lines of NumPy. scikit-learn is used only for the *baseline* models and to cross-check the metrics.

## Architecture

```
            DATASET  OCTMNIST (MedMNIST v2, 64x64): 109,309 retinal OCT B-scans, 4 classes
               ↓
      Data pre-processing   grayscale → centre crop → resize → [0, 1]
               ↓
       Feature extraction   raw pixels + HOG (histogram of oriented gradients) → z-score
               ↓
    Baseline accuracy       single-layer perceptron · softmax regression · scikit-learn MLP
               ↓
  Artificial neural network NumPy MLP: forward propagation → activation → softmax
               ↓
         Cost function      categorical cross-entropy + L2
               ↓
        Backpropagation     chain rule, verified by numerical gradient check
               ↓
   Hyperparameter tuning    9 hyperparameters, greedy one-at-a-time search on the validation split
               ↓
   Best model selected      retrained on the full training set, evaluated once on the test set
               ↓
        Web application     React UI  ⇄  FastAPI  /api/predict
               ↓
          Google Cloud      Cloud Run (Docker)          also: Railway · Vercel
               ↓
      User → Prediction
```

```
├── ml/                     machine learning core (NumPy + Pillow at inference time)
│   ├── nn/                 activations, losses, optimizers, MLP + gradient check
│   ├── data.py             OCTMNIST download / loading / balanced sampling
│   ├── features.py         preprocessing + HOG feature extraction
│   ├── metrics.py          accuracy / precision / recall / F1 / confusion matrix
│   ├── train.py            full experiment pipeline (python -m ml.train)
│   ├── figures.py          report figures (matplotlib)
│   └── inference.py        model bundle loading + single-image prediction
├── backend/app/main.py     FastAPI service (+ serves the built frontend)
├── frontend/               React + Vite + TypeScript + Tailwind SPA
├── artifacts/              trained model, report.json, figures, sample images
├── tests/                  pytest: gradient checks, metrics, features, API
├── api/index.py            Vercel serverless entry point
├── Dockerfile              single image for Cloud Run / Railway
├── cloudbuild.yaml         Google Cloud Build → Cloud Run pipeline
├── railway.json · vercel.json
└── docs/                   DEPLOYMENT.md, LCA1_Report_DrishtiManas.docx
```

## Quick start (local)

Prerequisites: Python 3.11+ and Node.js 20+.

```bash
# 1. Python environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt

# 2. (Optional) re-train: downloads OCTMNIST into data/ and rewrites artifacts/
python -m ml.train                   # full run at 64x64 (~35-45 min on a laptop CPU, downloads ~310 MB)
python -m ml.train --size 28         # the classic 28x28 OCTMNIST (smaller download, lower accuracy)
python -m ml.train --quick          # 1-minute smoke test (writes to artifacts-quick/, not artifacts/)
python -m ml.figures                 # re-render report figures from artifacts/report.json

# 3. API
uvicorn backend.app.main:app --reload --port 8000     # http://localhost:8000/api/docs

# 4. Frontend (second terminal)
cd frontend
npm install
npm run dev                          # http://localhost:5173 (proxies /api to :8000)
```

Or run the production container:

```bash
docker build -t drishti-manas .
docker run --rm -p 8080:8080 drishti-manas     # http://localhost:8080
```

Tests: `pytest -q` (Python) and `cd frontend && npm run build` (type-check + build).

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness + loaded model metadata |
| `POST` | `/api/predict` | `multipart/form-data` field `file` (PNG/JPEG/BMP/TIFF/WebP, ≤ 8 MB) → prediction, class probabilities, the preprocessed input, warnings |
| `GET` | `/api/model` | Full training report: dataset, baselines, tuning runs, curves, metrics |
| `GET` | `/api/samples` | Held-out test images for the UI's example gallery |
| `GET` | `/api/docs` | Interactive OpenAPI docs |

```bash
curl -F "file=@artifacts/samples/cnv_1.png" http://localhost:8000/api/predict
```

## Deployment

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for step-by-step guides:

- **Google Cloud Run** (assignment requirement): `gcloud run deploy drishti-manas --source . --region asia-south1 --allow-unauthenticated`
- **Railway:** deploy from the GitHub repository; it picks up `railway.json` and the `Dockerfile`
- **Vercel:** static frontend plus a Python serverless API via `vercel.json`

## Dataset and credits

- **OCTMNIST**, part of MedMNIST v2: J. Yang et al., *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D
  biomedical image classification*, Scientific Data 10, 41 (2023). CC BY 4.0. https://medmnist.com/
- Source images: D. S. Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep
  Learning*, Cell 172(5), 2018.

## Disclaimer

DrishtiManas is an academic project for research and education. It is **not** a medical device and must not be
used for diagnosis.
