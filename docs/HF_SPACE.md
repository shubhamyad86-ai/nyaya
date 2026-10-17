# Public demo on Hugging Face Spaces

> **Note (October 2026):** Hugging Face now requires a paid plan (PRO) for Docker and Gradio
> Spaces; only static Spaces stay free. For a free public demo use Render instead: docs/RENDER.md.

A Space runs the same app as `start_api.bat`, with a public link:
`https://<your-hf-name>-nyaya-credit-check.hf.space/app`.
Free CPU hardware (2 vCPU, 16 GB RAM). It goes to sleep after 48 hours without visitors and wakes
up on the next visit (about a minute).

## 1. Build the folder (Command Prompt, in the project folder)

```
python deploy\hf_space\build.py
```

This makes `dist\hf_space\` with about 46 files: `Dockerfile`, `README.md` (the Space settings),
`requirements.txt`, `config.yaml`, `src\` and `data\raw\german_credit.csv`.

## 2. Create the Space

1. Sign up at <https://huggingface.co/join> (email only, no card) and confirm the email.
2. <https://huggingface.co/new-space>:
   - Space name: `nyaya-credit-check`
   - License: MIT
   - SDK: **Docker** → template **Blank**
   - Hardware: **CPU basic · Free**
   - Visibility: **Public**
   - **Create Space**

## 3. Upload the files

On the Space page: **Files → + Contribute → Upload files**. Open `dist\hf_space` in Explorer,
select **everything inside it** (Ctrl+A: the `src` and `data` folders and the four files) and drag
it into the browser. Commit message: `First version` → **Commit changes to main**.

Replace the README the Space created: yes. The files must sit at the top of the Space (the
`Dockerfile` next to `README.md`), not inside a `hf_space` folder.

## 4. Wait for the build

The **App** tab shows *Building* (about 5-8 minutes: installs the packages, then trains the model).
Then *Running*. Open:

| Link | What it is |
|---|---|
| `https://<name>-nyaya-credit-check.hf.space/app` | Credit check page |
| `https://<name>-nyaya-credit-check.hf.space/batch` | Upload CSV / Excel / PDF |
| `https://<name>-nyaya-credit-check.hf.space/docs` | API |
| `https://<name>-nyaya-credit-check.hf.space/health` | Model name and training time |

If the build fails, the **Logs** button at the top shows the error.

## Updating it later

Run `python deploy\hf_space\build.py` again and upload the changed files the same way.

## What is different from running it locally

- Only the credit check, batch upload and API are included. The regulation assistant needs Ollama
  on your computer, so it is not part of the Space (`/health` shows `rag_index: not installed`).
- Uploaded files are not stored. The audit log goes to `/tmp` and is lost when the Space restarts.
- Anyone with the link can use it. It is a demo on public data: do not upload real customer data.
