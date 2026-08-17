@echo off
setlocal
set "REPO_ROOT=%~dp0"
set "PYTHON_EXE=%REPO_ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Python environment not found at: %PYTHON_EXE%
    exit /b 1
)

"%PYTHON_EXE%" "%REPO_ROOT%src\predict.py" %*
