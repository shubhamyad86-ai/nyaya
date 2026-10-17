@echo off
REM Starts the API. Leave this window open while you use http://127.0.0.1:8000/app (credit check)
REM and http://127.0.0.1:8000/docs (all endpoints).
REM Do not click inside this window while it runs: the title then starts with "Select" and Windows
REM pauses the server. If that happens, press Esc in the window and it continues.
cd /d "%~dp0"
call .venv\Scripts\activate || (echo Run setup.bat first. & pause & exit /b 1)

REM Local LLM used by the assistant and the agent. Change it if you pulled a different model.
if "%OLLAMA_MODEL%"=="" set OLLAMA_MODEL=llama3.2:1b

echo Starting Compliance AI with Ollama model %OLLAMA_MODEL% ...
echo The browser opens in a few seconds. Press Ctrl+C here to stop the server.
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:8000/app"
REM No --reload: it restarts the server whenever any file in the folder changes (copying files,
REM pip installs, training) and can get stuck. Stop with Ctrl+C and start again after changes.
uvicorn compliance_ai.api:app --host 127.0.0.1 --port 8000
