#!/usr/bin/env bash
# Deploy the credit-scoring Azure Function from Azure Cloud Shell (Bash).
#
#   bash deploy_azure.sh            create the resources (first time) and deploy the zip
#   bash deploy_azure.sh test       call the deployed endpoints again
#   bash deploy_azure.sh delete     delete everything this script created (stops all costs)
#
# Expects compliance_ai_function.zip (made by build.py on your PC) in the same folder.
# Region: LOCATION=germanywestcentral bash deploy_azure.sh   (default centralindia)
# CI (GitHub Actions): set RG, APP and STORAGE in the environment so every run updates the same app.
set -euo pipefail
cd "$(dirname "$0")"

ZIP=compliance_ai_function.zip
STATE="$HOME/.compliance-ai-azure.env"
ACTION="${1:-deploy}"

# Names are saved after the first run so re-running updates the same app.
if [ -f "$STATE" ]; then
  # shellcheck disable=SC1090
  source "$STATE"
else
  SUFFIX=$(tr -dc 'a-z0-9' </dev/urandom | head -c 6 || true)
  RG="${RG:-rg-compliance-ai}"
  LOCATION="${LOCATION:-centralindia}"
  STORAGE="${STORAGE:-stcompai${SUFFIX}}"   # 3-24 lowercase letters/digits, globally unique
  APP="${APP:-compliance-ai-${SUFFIX}}"     # becomes https://<APP>.azurewebsites.net
fi

save_state() {
  printf 'RG=%s\nLOCATION=%s\nSTORAGE=%s\nAPP=%s\n' "$RG" "$LOCATION" "$STORAGE" "$APP" > "$STATE"
}

pretty() { python3 -m json.tool 2>/dev/null || cat; }

run_tests() {
  local host key tmp
  host=$(az functionapp show -g "$RG" -n "$APP" --query defaultHostName -o tsv)
  key=$(az functionapp keys list -g "$RG" -n "$APP" --query functionKeys.default -o tsv)
  tmp=$(mktemp -d)
  unzip -o -q "$ZIP" sample_applicant.json sample_batch.json -d "$tmp"

  echo "== GET /api/health (waits for the app to start, up to 5 minutes)"
  for _ in $(seq 1 30); do
    if curl -sf "https://$host/api/health" -o "$tmp/health.json"; then break; fi
    sleep 10
  done
  pretty < "$tmp/health.json" || { echo "Health check failed - see docs/AZURE_DEPLOY.md troubleshooting"; exit 1; }

  echo "== POST /api/score (one applicant)"
  curl -s -X POST "https://$host/api/score?code=$key" -H "Content-Type: application/json" \
       --data @"$tmp/sample_applicant.json" | pretty

  echo "== POST /api/app/score (public demo page's scoring call)"
  curl -s -X POST "https://$host/api/app/score" -H "Content-Type: application/json" \
       --data @"$tmp/sample_applicant.json" | python3 -c \
       'import json,sys; r=json.load(sys.stdin); print("credit score", r.get("credit_score"), "|", r.get("result", r))'

  echo "== POST /api/drift (60 applicants)"
  curl -s -X POST "https://$host/api/drift?code=$key" -H "Content-Type: application/json" \
       --data @"$tmp/sample_batch.json" | python3 -c \
       'import json,sys; r=json.load(sys.stdin); print(r.get("overall"), "-", r.get("summary", r))'

  cat <<EOF

Endpoints
  https://$host/api/app        <- open this in a browser: the credit check page
  https://$host/api/health
  https://$host/api/score?code=<function key>
  https://$host/api/drift?code=<function key>
Function key:  az functionapp keys list -g $RG -n $APP --query functionKeys.default -o tsv
Logs:          Azure Portal -> $APP -> Monitoring -> Logs / Application Insights
EOF
}

case "$ACTION" in
  test)
    [ -f "$STATE" ] || { echo "Nothing deployed yet - run: bash deploy_azure.sh"; exit 1; }
    run_tests; exit 0 ;;
  delete)
    [ -f "$STATE" ] || { echo "Nothing to delete."; exit 0; }
    echo "Deleting resource group $RG and everything in it..."
    az group delete -n "$RG" --yes
    rm -f "$STATE"; echo "Deleted."; exit 0 ;;
  deploy) ;;
  *) echo "Usage: bash deploy_azure.sh [deploy|test|delete]"; exit 1 ;;
esac

[ -f "$ZIP" ] || { echo "Upload $ZIP next to this script first (Cloud Shell: Manage files -> Upload)."; exit 1; }
PY=$(unzip -p "$ZIP" .python_version | tr -d '[:space:]')
echo "App: $APP | resource group: $RG | region: $LOCATION | Python $PY"

az account show -o none 2>/dev/null || az login
echo "Subscription: $(az account show --query name -o tsv)"

if az functionapp show -g "$RG" -n "$APP" -o none 2>/dev/null; then
  echo "== Function App $APP exists - updating it"
  save_state
else
  echo "== Registering resource providers (first time can take a few minutes)"
  for ns in Microsoft.Web Microsoft.Storage Microsoft.Insights Microsoft.OperationalInsights; do
    az provider register --namespace "$ns" --wait -o none
  done

  if [ -z "$(az functionapp list-flexconsumption-locations --query "[?name=='$LOCATION'].name" -o tsv)" ]; then
    echo "Region '$LOCATION' does not offer Flex Consumption. Available regions:"
    az functionapp list-flexconsumption-locations --query "sort_by(@, &name)[].name" -o tsv | paste -sd' '
    echo "Re-run with e.g.:  LOCATION=southeastasia bash deploy_azure.sh"
    exit 1
  fi
  save_state

  echo "== Resource group"
  az group show -n "$RG" -o none 2>/dev/null || az group create -n "$RG" -l "$LOCATION" -o none

  echo "== Storage account $STORAGE"
  if ! az storage account show -g "$RG" -n "$STORAGE" -o none 2>/dev/null; then
    az storage account create -g "$RG" -n "$STORAGE" -l "$LOCATION" --sku Standard_LRS \
       --allow-blob-public-access false --min-tls-version TLS1_2 -o none
  fi

  echo "== Function App $APP (Flex Consumption, Python $PY)"
  az functionapp create -g "$RG" -n "$APP" --storage-account "$STORAGE" \
     --flexconsumption-location "$LOCATION" --runtime python --runtime-version "$PY" -o none
fi

# The demo page is public, so cap how far the app can scale (and what it can cost).
az functionapp scale config set -g "$RG" -n "$APP" --maximum-instance-count 2 -o none \
  || echo "   (could not set the instance cap; set it in Portal -> $APP -> Scale and concurrency)"

echo "== Deploying the zip (Azure installs requirements.txt on Linux; takes 3-8 minutes)"
az functionapp deployment source config-zip -g "$RG" -n "$APP" --src "$ZIP" --build-remote true -o none

run_tests
echo
echo "Done. When finished with the demo, delete everything with:  bash deploy_azure.sh delete"
