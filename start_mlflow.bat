@echo off
REM Start MLflow tracking server on Windows
REM UI available at: http://127.0.0.1:5000

set MLFLOW_BACKEND_STORE=mlruns
set MLFLOW_HOST=127.0.0.1
set MLFLOW_PORT=5000

echo ============================================================
echo  MLflow Tracking Server
echo  UI: http://%MLFLOW_HOST%:%MLFLOW_PORT%
echo  Backend store: %MLFLOW_BACKEND_STORE%
echo  Press Ctrl+C to stop
echo ============================================================
echo.

mlflow server --backend-store-uri %MLFLOW_BACKEND_STORE% --host %MLFLOW_HOST% --port %MLFLOW_PORT%
