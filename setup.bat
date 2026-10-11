@echo off
REM One-time setup for Windows: installs everything, trains the models and runs the tests.
REM Double-click this file, or run "setup.bat" from Command Prompt inside the project folder.
cd /d "%~dp0"

if not exist .venv (
    echo Creating the virtual environment...
    python -m venv .venv || goto :error
)
call .venv\Scripts\activate || goto :error

echo.
echo [1/8] Installing packages (PyTorch is large - this can take 10-20 minutes the first time)...
python -m pip install --upgrade pip
pip install -r requirements.txt pytest httpx || goto :error
pip install -r requirements-nlp.txt || goto :error
pip install -e . || goto :error

echo.
echo [2/8] Exploratory data analysis...
python scripts\eda.py || goto :error

echo.
echo [3/8] Training the credit-risk models (1-3 minutes)...
python -m compliance_ai.credit.train || goto :error

echo.
echo [4/8] Building the regulation search index (GDPR + EU AI Act)...
python -m compliance_ai.rag.build_index --backend tfidf || goto :error

echo.
echo [5/8] Evaluating regulation retrieval...
python -m compliance_ai.rag.evaluate || goto :error

echo.
echo [6/8] Training the financial-sentiment baseline...
python -m compliance_ai.nlp.train --method tfidf_baseline || goto :error

echo.
echo [7/8] Drift monitoring demo and updating README / model card with your results...
python scripts\drift_demo.py || goto :error
python scripts\update_docs.py || goto :error

echo.
echo [8/8] Running the tests...
pytest -q || goto :error

echo.
echo ============================================================
echo  Setup complete. Start the app with:  start_api.bat
echo ============================================================
pause
exit /b 0

:error
echo.
echo *** Setup stopped because the step above failed. Take a photo of the error and ask for help. ***
pause
exit /b 1
