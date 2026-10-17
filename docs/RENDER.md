# Public demo on Render (free, no card)

Render builds the app from your GitHub repository and gives it a public link like
`https://nyaya-credit-check.onrender.com/app`.

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
2. Allow Render to see your `compliance-ai` repository.
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

## What is different from running it locally

- Only the credit check, batch upload and API. The regulation assistant needs Ollama on your
  computer (`/health` shows `rag_index: not installed`).
- Uploaded files are not stored. The audit log goes to `/tmp` and is lost when the app restarts.
- Anyone with the link can use it. It is a demo on public data: do not upload real customer data.
