# Deploy the credit-scoring API to Azure Functions

The result is a public web page where anyone can fill in an applicant and see a credit score with
PASS or FAIL, plus HTTPS endpoints, protected by a function key, that score loan applicants and
check a batch for drift. It runs on the **Flex Consumption** plan: you pay per request, it scales to
zero when idle, and a small demo fits inside the monthly free grant. Delete it when you are done
(step 6).

What gets deployed: the trained credit model, the drift reference profile, and the `compliance_ai`
scoring code. The RAG assistant, the agent and Ollama stay on your PC (they need a local LLM).

| Route | Auth | Purpose |
|---|---|---|
| `GET /api/app` | none | **credit check page**: choose the applicant's values, get score, PASS/FAIL and reasons |
| `POST /api/app/score` | none | the page's scoring call; stores nothing, and the app is capped at 2 instances |
| `GET /api/health` | none | model name, threshold, training time |
| `POST /api/score` | function key | one applicant → P(default), credit score, PASS/FAIL, top reason codes |
| `POST /api/drift` | function key | `{"applicants": [...]}`, 30 or more → drift report |

## 1. Create the Azure account (once)

1. Go to <https://azure.microsoft.com/free> and sign up. A card is needed for identity checks; the
   free account does not charge it unless you upgrade.
2. Sign in to <https://portal.azure.com> and check that a subscription exists
   (search **Subscriptions** in the top bar).

## 2. Build the package on your PC

`setup.bat` must have run once (it trains the model). Then, in the project folder:

```
build_azure.bat
```

It creates two files in the `dist` folder and opens it:

- `compliance_ai_function.zip` - the app, the model, and a `requirements.txt` pinned to the exact
  package versions on your PC, so the model loads in Azure exactly as it was trained
- `deploy_azure.sh` - the script that creates the Azure resources and deploys

The build also records your Python version (for example 3.12), and Azure uses the same one.

## 3. Open Azure Cloud Shell and upload

1. Open <https://shell.azure.com> and choose **Bash**. The first time, pick
   **No storage account required** and your subscription.
2. Click **Manage files → Upload**, and upload both files from `dist`.

## 4. Deploy

```bash
bash deploy_azure.sh
```

The script:

1. registers the Azure resource providers it needs (the first time only, a few minutes)
2. checks that your region offers Flex Consumption (default `centralindia`)
3. creates the resource group `rg-compliance-ai`, a storage account and the Function App
   (with Application Insights for logs)
4. uploads the zip; Azure installs the Python packages on Linux (3-8 minutes)
5. calls `/api/health`, `/api/score`, `/api/app/score` and `/api/drift` and prints the results and URLs

The names are saved in `~/.compliance-ai-azure.env`, so running it again updates the same app. To use
another region:

```bash
LOCATION=southeastasia bash deploy_azure.sh
```

## 5. Open it and call it yourself

Open `https://<app>.azurewebsites.net/api/app` in any browser. That's the link to put on your CV,
LinkedIn and the GitHub README. The first request after a quiet period takes 5-20 seconds while
Azure starts the app (it scales to zero when idle); after that it answers in well under a second.

Cloud Shell, after deploying:

```bash
bash deploy_azure.sh test
```

From Windows Command Prompt (put in your app name and key; `bash deploy_azure.sh test` prints both):

```
curl -X POST "https://<app>.azurewebsites.net/api/score?code=<key>" -H "Content-Type: application/json" --data @sample_applicant.json
```

`sample_applicant.json` is inside the zip. In the Azure Portal, open the Function App →
**Monitoring → Logs / Application Insights** to see every request, its latency and any errors.

## 6. Delete everything (stops all costs)

```bash
bash deploy_azure.sh delete
```

This deletes the resource group and everything in it. Also check **Cost Management** in the portal
after a day.

## 7. Deploy from GitHub (optional, after the first deploy)

After steps 1-4 have created the app once, every later update can be one click in GitHub:
**Actions → Deploy to Azure → Run workflow**. It trains the model on the real data, runs the tests,
builds the zip and deploys to the same app. GitHub signs in to Azure with OpenID Connect, so no
password or key is stored in GitHub.

One-time setup, in Cloud Shell (replace nothing; it reads your names from step 4):

```bash
source ~/.compliance-ai-azure.env
SUB=$(az account show --query id -o tsv)
CLIENT=$(az ad app create --display-name nyaya-github-deploy --query appId -o tsv)
az ad sp create --id "$CLIENT" -o none
az role assignment create --assignee "$CLIENT" --role Contributor \
   --scope "/subscriptions/$SUB/resourceGroups/$RG" -o none
az ad app federated-credential create --id "$CLIENT" --parameters '{
  "name": "github-nyaya-azure",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:shubhamyad86-ai/nyaya:environment:azure",
  "audiences": ["api://AzureADTokenExchange"]}'
echo "AZURE_CLIENT_ID=$CLIENT"; echo "AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)"
echo "AZURE_SUBSCRIPTION_ID=$SUB"; echo "AZURE_FUNCTION_APP=$APP"; echo "AZURE_STORAGE_ACCOUNT=$STORAGE"
```

Then in GitHub: **Settings → Environments → New environment** named `azure`, and in **Settings →
Secrets and variables → Actions** add:

- Secrets: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
- Variables: `AZURE_FUNCTION_APP`, `AZURE_STORAGE_ACCOUNT` (and `AZURE_RESOURCE_GROUP` if you
  changed it)

The workflow's identity can change only the `rg-compliance-ai` resource group, nothing else in
your subscription. It trains with the same Python version as the running app, so the pinned
packages always install.

## Troubleshooting

| Message | Fix |
|---|---|
| `$'\r': command not found` | The script got Windows line endings. Run `sed -i 's/\r$//' deploy_azure.sh` and retry. |
| `Region 'centralindia' does not offer Flex Consumption` | Re-run with one of the listed regions: `LOCATION=<region> bash deploy_azure.sh` |
| Quota or `SubscriptionIsOverQuotaForSku` | Try another region, or request a quota increase in the portal. |
| The storage account name is taken | `rm ~/.compliance-ai-azure.env` and run again (new random names). |
| Health check fails after deploy | Portal → Function App → **Deployment Center → Logs** for the build; **Log stream** for start-up errors. |
| `ModuleNotFoundError` or an unpickling error | The model was trained with other package versions. Run `setup.bat` and `build_azure.bat` again, then redeploy. |
| `Python 3.x is not available in Azure Functions` | Azure supports 3.10-3.14. Install one of those, delete `.venv` and run `setup.bat` again. |
| `401 Unauthorized` | The `code=` key is missing or wrong. |

## Why it is built this way

- **Pinned versions from your own environment.** A pickled scikit-learn/XGBoost model should be
  loaded with the same library versions it was trained with, so `build.py` reads the installed
  versions instead of using a hand-written requirements file.
- **`xgboost-cpu` in Azure.** Same library without the 200 MB GPU dependency, so the build is faster.
- **Remote build.** Azure installs the packages on Linux; Windows wheels would not run there.
- **Function key on the API.** The scoring and drift API needs a key. Only `/health` and the demo
  page are anonymous; the page stores nothing and the app can scale to at most 2 instances, which
  caps the cost of abuse.
- **Drift check in production.** The reference profile from training ships with the model, so any
  batch of new applicants can be compared with the population the model learned from.

Not production-grade yet: there is no VNet, no Key Vault, no sign-in on the demo page and no
authentication beyond function keys. Those would be the next steps for a real bank deployment.
