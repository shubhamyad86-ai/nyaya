@echo off
REM Starts the API. Leave this window open while you use http://127.0.0.1:8000/app (credit check)
REM and http://127.0.0.1:8000/docs (all endpoints).
REM Do not click inside this window while it runs (Windows "Select" mode pauses the server).
cd /d "%~dp0"
call .venv\Scripts\activate || (echo Run setup.bat first. & pause & exit /b 1)

REM Local LLM used by the assistant and the agent. Change it if you pulled a different model.
if "%OLLAMA_MODEL%"=="" set OLLAMA_MODEL=llama3.2:1b

echo Starting Compliance AI with Ollama model %OLLAMA_MODEL% ...
echo The browser opens in a few seconds. Press Ctrl+C here to stop the server.
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:8000/app"
uvicorn compliance_ai.api:app --reload
