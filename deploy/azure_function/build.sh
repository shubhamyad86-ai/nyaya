#!/usr/bin/env bash
# Package the Azure Function (macOS/Linux). Same as build.bat on Windows.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Activate the project virtual environment first (source .venv/bin/activate).
python deploy/azure_function/build.py
cp deploy/azure_function/deploy_azure.sh dist/
echo "Upload dist/compliance_ai_function.zip and dist/deploy_azure.sh to Azure Cloud Shell."
