# Drishti Manas – Smart Screening and Diagnostics for Ocular and Neurological Diseases

> **Elevator pitch**: Drishti Manas is an AI-powered web platform that analyzes ocular images (fundus, retinal, OCT) and surfaces explainable predictions plus clinician-friendly reports so technicians can triage earlier, doctors can act faster, and admins can govern deployments with confidence.

## 🎯 Theme & Focus Areas

This solution advances child-friendly ocular screening while extending to accessible neurological diagnostics. Core focus areas:

- Screening and diagnosis of common and rare ocular diseases (refractive errors, amblyopia, cataract, glaucoma, retinal disorders).
- Early detection of congenital and developmental eye conditions with emphasis on pediatric usability.
- Neurological disorder screening pipelines that support both pediatric and adult populations.
- Integrated workflows that merge ophthalmic and neurological data for longitudinal monitoring and timely interventions.

## 👥 Target Personas

| Role | Responsibilities | Key UX needs |
| --- | --- | --- |
| Technician / Operator | Uploads fundus images, collects vitals, validates metadata | Guided forms, automated quality checks |
| Doctor / Specialist | Reviews AI probabilities, heatmaps, and writes final notes | Explainable visuals, structured reports, override controls |
| Admin | Manages users, audit history, model releases | Audit logs, RBAC, deployment and model drift dashboards |

## 🧱 Reference Architecture

```
┌─────────────┐      ┌────────────────────┐      ┌────────────────────┐
│ React + Vite│─────▶│ FastAPI Gateway     │─────▶│ PyTorch Inference   │
│ Tailwind UI │◀─────│ Auth, REST, Webhooks│◀─────│ Ocular + Neuro cores│
└──────┬──────┘      └──────────┬─────────┘      └───────┬────────────┘
       │                        │                        │
       │                        │                        ▼
       │                        │                Explainability (Grad-CAM)
       │                        │
       ▼                        ▼
PostgreSQL ◀────── SQLAlchemy ORM + Pydantic Schemas ─────▶ Object Storage (images)
```

- **ML Core**: Modular PyTorch pipelines for ocular and neurological models, optional TensorFlow parity.
- **Backend API**: FastAPI, async SQLAlchemy, background workers for batched inference, OpenAPI docs.
- **Frontend**: React + Vite SPA with Tailwind CSS and role-based layouts ready for future dashboards.
- **Infra**: Dockerized micro-stacks, `docker-compose` orchestration, and `infra/` for CI/CD blueprints.

## 🧰 Tech Stack

- **Languages**: Python 3.11+, TypeScript (Vite React)
- **AI / ML**: PyTorch, MONAI, timm, Grad-CAM, Captum
- **Backend**: FastAPI, SQLAlchemy, Pydantic v2, celery-ready structure
- **Storage**: PostgreSQL (metadata), object storage (images, not included)
- **Frontend**: React 18, Vite, TailwindCSS, Zustand (state) placeholder
- **DevOps**: Docker, Docker Compose, GitHub Actions (infra roadmap)

## 🖥️ Frontend experience

- **Technician workspace** – guided upload wizard with modality presets, automatic multi-disease screening panels, and a mock-mode toggle so clinics can demo without the API online.
- **Doctor dashboard** – AI-prioritized worklist, severity chips, Grad-CAM thumbnails, and acknowledgement controls based on learnings from the referenced Dry Eye / Ocular CNN repos.
- **Shared design language** – Tailwind-powered cards, responsive navigation, and ready hooks to attach live FastAPI endpoints once deployed.

## 📅 Roadmap & reference learnings

- We distilled actionable insights from [Dry-Eye-Disease](https://github.com/owais825/Dry-Eye-Disease), [Prediction-of-ocular-disease-using-fundus-images-using-CNN](https://github.com/Nithish-2002/Predection-of-ocular-disease-using-fundus-images-using-CNN), and [Ocular-Disease-Identifier](https://github.com/DSC-McMaster-U/Ocular-Disease-Identifier).
- See `docs/roadmap.md` for how those lessons shape upcoming milestones (dry-eye friendly intake, CNN ensemble tuning, explainable reporting, and governance dashboards).

## 🗂️ Project Structure (high level)

```
├── data/                      # Raw / processed datasets (tracked via DVC in future)
├── models/                    # Checkpoints (git-ignored)
├── notebooks/                 # Research / EDA notebooks
├── src/
│   ├── backend/               # FastAPI service
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── dependencies.py
│   │   │   ├── database.py
│   │   │   ├── routers/
│   │   │   ├── schemas/
│   │   │   └── services/
│   │   └── tests/
│   └── ml/                    # ML core (data, models, explainability)
├── frontend/                  # React + Vite SPA
├── infra/                     # CI/CD samples, IaC placeholders
├── docker-compose.yml         # Local orchestration
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt
├── environment.yml
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker Desktop (optional but recommended)

### 1. Python environment

```bash
conda env create -f environment.yml
conda activate drishti-manas
```

Or using pip:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Backend API (FastAPI)

**Important:** Make sure you've installed all Python dependencies first:

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies (including PyTorch)
pip install -r requirements.txt

# Navigate to backend directory
cd src/backend

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

OpenAPI docs: `http://localhost:8000/docs`.

Helpful endpoints:

- `GET /health/ping` – readiness signal for the frontend badge.
- `GET /metadata/diseases` – canonical ocular + neurological disease catalog for UI sync.
- `POST /predictions/` – run the screening pipeline; accepts optional inline images via `image_base64`.

### 3. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

**Camera Note:** Camera capture requires:
- HTTPS connection OR localhost access
- Browser camera permissions granted
- Physical camera device available

If camera is unavailable, use the file upload button instead - it works on all devices.

### 4. Full stack via Docker

```bash
docker compose up --build
```

Services exposed:

- `frontend` on http://localhost:5173
- `backend` on http://localhost:8000
- `postgres` on port 5432 (local-only)

Environment variables for the API live in `.env.backend`. Copy or edit that file to point `DATABASE_URL` at your preferred Postgres instance before composing the stack. The sample values use `drishti_app/drishti_secure_pw` and the database `drishti_ops`; if you change them, mirror the edits inside `docker-compose.yml` or provide your own secrets store.

### Env files overview

- `.env.backend` – FastAPI settings plus DSN template.
- `.env.postgres` – credentials loaded into the Postgres container via `env_file`.
- `.env.frontend` – default Vite variables; copy to `frontend/.env.local` when running `npm run dev` or export before building custom images.

## 📊 Data & Model Workflow

1. **Ingestion** – Technicians drop DICOM, fundus JPEG, or OCT stacks into `data/raw/`.
2. **Preprocessing** – `src/ml/pipeline.py` handles resizing, denoising, quality filtering.
3. **Training** – Vision Transformers and lightweight CNNs defined in `src/ml/models/` with task-specific heads.
4. **Inference** – FastAPI calls `PredictionService`, which loads latest checkpoint, runs Grad-CAM, and persists metadata via SQLAlchemy.
5. **Review** – Frontend surfaces predictions, certainty bands, and heatmaps for clinician validation.

## 🔐 Role-based Experience

- **Technician**: Guided upload wizard with QC alerts, bulk CSV import.
- **Doctor**: Case worklist, dual-panel image/heatmap viewer, ability to append notes and finalize assessment.
- **Admin**: User management, audit trail, model version selector (future dashboard stubs already in UI structure).

## 🧪 Testing & Quality Gates

- API unit tests live under `src/backend/tests/` (pytest + httpx).
- Frontend uses Vitest and React Testing Library (sample spec planned).
- Model evaluation harness outputs ROC, sensitivity, specificity, and Grad-CAM overlays.

Run backend tests:

```bash
cd src/backend
pytest
```

## 🛠️ Infra & Deployment

- `Dockerfile.backend` & `Dockerfile.frontend` for reproducible builds.
- `docker-compose.yml` wires API, Postgres, and frontend for local demos.
- `infra/ci/github-actions.yml` blueprint for lint + test + image publish workflow.
- Future IaC (Terraform, Azure/AWS) slots reserved inside `infra/`.

**Image tagging & push starter workflow**

```bash
docker build -f Dockerfile.backend -t ghcr.io/<org>/drishti-backend:$(git rev-parse --short HEAD) .
docker push ghcr.io/<org>/drishti-backend:$(git rev-parse --short HEAD)

docker build -f Dockerfile.frontend -t ghcr.io/<org>/drishti-frontend:$(git rev-parse --short HEAD) .
docker push ghcr.io/<org>/drishti-frontend:$(git rev-parse --short HEAD)
```

CI can reuse those commands after running tests, substituting tags like `main`, `release-<semver>`, or `${{ github.sha }}`. Compose deployments can then reference the published images via `image:` blocks instead of `build:` for parity with staging/production.

## 📈 Metrics & Explainability

- Clinical metrics: Accuracy, Sensitivity/Specificity, AUC-ROC per disease.
- Operational metrics: Throughput, queue latency, audit coverage.
- Explainability: Grad-CAM heatmaps streamed to UI, Captum attribution JSON.

## 🤝 Contributing & License

Contributions are welcome! Please open an issue to discuss major changes. This project ships under the MIT License.

## 🏥 Clinical Disclaimer

Drishti Manas is intended for screening support and research workflows only. Final diagnoses rest with qualified clinicians.

## 📧 Contact

Questions? Reach out at `contact@drishti-manas.health`.

---

**Built with care to advance accessible vision and neuro health**
