# Public demo on Render (free, no card)

Render builds the app from the GitHub repository and gives it a public link. This project runs at
**<https://nyaya-ccnp.onrender.com/app>** (upload page: `/batch`, API: `/docs`).

Free plan: 512 MB memory (the app peaks at about 370 MB on a 3,000-applicant upload), no card.
After 15 minutes without visitors the app sleeps; the next visit wakes it in about a minute.
Hugging Face Spaces now needs a paid plan for apps that run Python, so this project uses Render.

## 1. Push the project to GitHub

`render.yaml` and `deploy/render/Dockerfile` must be in the repository:

```
git add -A
git commit -m "Render deployment"
git push
```

## 2. Create the service

1. <https://render.com> → **Get Started** → **Sign up with GitHub** (no card).
2. Allow Render to see your `nyaya` repository.
3. **New + → Blueprint** → choose the repository → Render reads `render.yaml` and shows the
   service **nyaya-credit-check (free)** → **Deploy Blueprint**.

   Or by hand: **New + → Web Service** → the repository → Language **Docker**, Dockerfile path
   `./deploy/render/Dockerfile`, Docker build context `.`, Instance type **Free**, Health check
   path `/health` → **Deploy**.

## 3. Wait for the build (about 5-10 minutes)

The **Logs** tab shows the packages installing, then `Test ROC AUC ...` from training, then
`Application startup complete`. The status turns **Live**. Open:

| Link | What it is |
|---|---|
| `https://<service>.onrender.com/app` | Credit check page |
| `https://<service>.onrender.com/batch` | Upload CSV / Excel / PDF (up to 3,000 applicants) |
| `https://<service>.onrender.com/docs` | API |
| `https://<service>.onrender.com/health` | Model name and training time |

Every `git push` deploys again automatically.

### Set up by hand instead of the Blueprint

**New + → Web Service** → the repository, then:

| Field | Value |
|---|---|
| Language | Docker |
| Branch | `main` |
| Region | Frankfurt (or any) |
| Instance type | Free |
| Dockerfile Path (Advanced) | `./deploy/render/Dockerfile` |
| Docker Build Context Directory (Advanced) | `.` |
| Health Check Path (Advanced) | `/health` (not the default `/healthz`) |

A first build took 5 min 37 s. The line `Failed to import Git` during training is a harmless MLflow
warning (git is not in the image); newer versions of the Dockerfile switch it off.

## The cloud model can differ from your laptop's

The image trains the model during the build with the newest library versions. On 17 October 2026 the
cloud build selected **gradient boosting (pass mark 83)**, while a laptop with older versions selected
**XGBoost (pass mark 81)**. The four models are within cross-validation noise of each other, so small
version differences can change the winner. `/health` and the upload page always show which model is
running. To make both identical, pin the laptop's versions (`pip freeze`) in
`deploy/hf_space/requirements.txt`.

## What is different from running it locally

- Only the credit check, batch upload and API. The regulation assistant needs Ollama on your
  computer (`/health` shows `rag_index: not installed`).
- Uploaded files are not stored. The audit log goes to `/tmp` and is lost when the app restarts.
- Anyone with the link can use it. It is a demo on public data: do not upload real customer data.
