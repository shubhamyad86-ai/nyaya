@echo off
REM Package the credit-scoring API for Azure Functions. Run setup.bat once before this.
REM Output: dist\compliance_ai_function.zip and dist\deploy_azure.sh - upload both to Azure Cloud Shell.
cd /d "%~dp0"
if not exist .venv (
    echo Run setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate || goto :error
python deploy\azure_function\build.py || goto :error
copy /y deploy\azure_function\deploy_azure.sh dist\ >nul || goto :error
echo.
echo ============================================================
echo  Ready. Next: open https://shell.azure.com (Bash), click Manage files - Upload,
echo  upload both files from the "dist" folder, then run:  bash deploy_azure.sh
echo  Full guide: docs\AZURE_DEPLOY.md
echo ============================================================
explorer dist
pause
exit /b 0

:error
echo.
echo *** Build failed - take a photo of the error above and ask for help. ***
pause
exit /b 1
