# Deploying DrishtiManas

The whole application (React UI and FastAPI inference API) ships as **one Docker image**. Inference needs only
NumPy and Pillow, so the container is small (under 300 MB: a slim Python base plus about 130 MB of dependencies), cold-starts in about a second and fits comfortably in the free
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

**Yes, Vercel can host the whole app, model included.** The "ML model" is not a heavy framework. Inference is plain
NumPy matrix multiplication on a 7.7 MB weights file, so the backend is an ordinary FastAPI app as far as Vercel is
concerned.

How it deploys:

- The React app is built into `frontend/dist` and served from Vercel's CDN.
- `api/index.py` becomes one Python serverless function that handles every `/api/*` route (predict, report, samples).
  It bundles `ml/`, `backend/` and `artifacts/` (the model), per `includeFiles` in `vercel.json`.
- `"framework": null` tells Vercel to use the "Other" preset. Without it, Vercel may detect `requirements.txt` +
  FastAPI and treat the repo as a pure FastAPI project, skipping the frontend build.
- `.python-version` pins Python 3.12. `.vercelignore` keeps the 360 MB training data, docs, tests and figures out of
  the upload.
- Vercel's Python function build resolves dependencies with `uv`, which requires a `[project]` table in
  `pyproject.toml` (not just `requirements.txt`). `pyproject.toml` mirrors `requirements.txt`'s dependencies for this,
  and `uv.lock` is committed so the build doesn't need to re-resolve them.

Verified locally by building the same bundle and running it on Python 3.12:

| Check | Result |
| --- | --- |
| Function bundle size (dependencies + code + model) | 139 MB (NumPy and Pillow are most of it; the model is 7.7 MB) |
| Cold import of `api/index.py` | 0.4 s |
| `/api/health`, `/api/model`, `/api/samples/*`, `/api/predict` | all return 200 with correct results |
| Prediction latency (3-model ensemble) | about 1–5 ms |

Limits to be aware of (from Vercel's platform limits; check vercel.com/docs/functions/limitations if in doubt):

- **Function size:** Vercel caps the function bundle (250 MB for most runtimes). At 139 MB we are well inside it.
- **Request body:** Vercel caps function request bodies at about 4.5 MB. The web UI downscales any image over 4 MB in
  the browser before uploading. An 8 MB test image went out as 0.16 MB and was still classified correctly, and the
  model only uses a 64×64 crop, so nothing is lost. Direct API clients (e.g. `curl`) must keep uploads under 4.5 MB
  on Vercel.
- **Cold starts:** the first request after a period of inactivity loads NumPy and the model, which takes about a
  second. Later requests are warm.

Deploy from the dashboard (easiest):

1. Push the repo to GitHub.
2. In Vercel, choose **Add New → Project → Import** the `DrishtiManas` repository.
3. Leave **Root Directory** as the repository root. `vercel.json` supplies the build settings. Don't change the
   framework preset.
4. Click **Deploy**. When it finishes, open `https://<project>.vercel.app/api/health`, which should return
   `"loaded": true`.

Or with the CLI:

```bash
npm i -g vercel
vercel          # first run links the project; accept the settings from vercel.json
vercel --prod
```

**Vercel does not replace Google Cloud for the assignment.** The brief asks for deployment on Google Cloud, so keep
the Cloud Run deployment (section 1) as the one you report. A Vercel URL is a fine extra.

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
