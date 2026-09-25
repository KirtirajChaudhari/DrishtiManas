# Deploying DrishtiManas

The whole application (React UI and FastAPI inference API) ships as **one Docker image**. Inference needs only
NumPy and Pillow, so the container is small (~200 MB), cold-starts in about a second and fits comfortably in the free
tiers below.

| Platform | Role | Config file |
| --- | --- | --- |
| **Google Cloud Run** (required by the assignment) | Full app: UI + API | `Dockerfile`, `cloudbuild.yaml`, `.gcloudignore` |
| Railway | Full app: UI + API (alternative) | `Dockerfile`, `railway.json` |
| Vercel | Static UI + Python serverless API | `vercel.json`, `api/index.py`, `.vercelignore` |

Every option serves the trained model from `artifacts/model.npz`, which is committed to the repository. Run
`python -m ml.train` before deploying if you want to retrain it.

---

## 1. Google Cloud Run (recommended)

### One-time setup

1. Redeem the Google Cloud education coupon, then create a project in the
   [Cloud Console](https://console.cloud.google.com/) (for example `drishti-manas`) and link the coupon's billing
   account to it.
2. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install), or use **Cloud Shell** from the console,
   which has everything preinstalled.
3. Log in and enable the services:

   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

### Option A: a single command (source deploy)

```bash
git clone https://github.com/KirtirajChaudhari/DrishtiManas.git && cd DrishtiManas
gcloud run deploy drishti-manas \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 512Mi --cpu 1 --max-instances 3
```

Cloud Build finds the `Dockerfile`, builds the image remotely, stores it in Artifact Registry and deploys it. When the
command finishes it prints the service URL, `https://drishti-manas-xxxxxxxx.asia-south1.run.app`.

### Option B: Cloud Build pipeline (`cloudbuild.yaml`)

```bash
gcloud artifacts repositories create drishti --repository-format=docker --location=asia-south1
gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION=asia-south1
```

To deploy automatically on every push, connect the GitHub repository in **Cloud Build → Triggers** and point the
trigger at `cloudbuild.yaml`.

If the deploy step fails with a permission error, grant the Cloud Build service account the **Cloud Run Admin** and
**Service Account User** roles.

### Verify

```bash
URL=$(gcloud run services describe drishti-manas --region asia-south1 --format 'value(status.url)')
curl $URL/api/health
curl -F "file=@artifacts/samples/cnv_1.png" $URL/api/predict
```

For the assignment evidence, screenshot the Cloud Run service page (URL, revision, metrics) and the running app.

**Cost:** Cloud Run bills only while requests are served. With `--min-instances 0` (the default), idle time costs
nothing, and a demo like this stays within the free tier.

---

## 2. Railway

1. Go to [railway.com](https://railway.com), then **New Project → Deploy from GitHub repo** and pick `DrishtiManas`.
2. Railway reads `railway.json`, builds the `Dockerfile` and injects `PORT`. The container already listens on
   `$PORT`.
3. Under **Settings → Networking**, click **Generate Domain**.

The health check path is `/api/health`.

---

## 3. Vercel

`vercel.json` builds the React app into `frontend/dist` (served from the CDN) and deploys `api/index.py` as a Python
serverless function that handles every `/api/*` route. The function bundles `ml/`, `backend/` and `artifacts/`.

```bash
npm i -g vercel
vercel          # first run links the project; accept the detected settings
vercel --prod
```

To deploy from the dashboard instead: **Add New → Project → Import** the GitHub repo. Keep the root directory as the
repository root; `vercel.json` supplies the build settings.

---

## 4. Split deployment (UI on Vercel, API on Cloud Run or Railway)

1. Deploy the API with the Docker image, as above.
2. Set `DRISHTI_ALLOWED_ORIGINS=https://your-ui.vercel.app` on the API service. This locks CORS to your UI's
   origin.
3. Build the UI with `VITE_API_BASE_URL=https://your-api-url` (in Vercel: **Project → Settings → Environment
   Variables**). Then remove the `functions` block and the `/api` rewrite from `vercel.json`.

---

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8080` | Port uvicorn listens on (set automatically by Cloud Run and Railway) |
| `DRISHTI_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `DRISHTI_MAX_UPLOAD_MB` | `8` | Upload size limit |
| `DRISHTI_ARTIFACTS_DIR` | `./artifacts` | Location of `model.npz`, `report.json` and `samples/` |
| `DRISHTI_FRONTEND_DIST` | `./frontend/dist` | Built UI served at `/` when present |
| `VITE_API_BASE_URL` | empty (same origin) | API URL used by the UI at build time |

## Run the production image locally

```bash
docker build -t drishti-manas .
docker run --rm -p 8080:8080 drishti-manas
# open http://localhost:8080
```
