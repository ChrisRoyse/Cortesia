@echo off
REM Setup script for downloading and preparing models on Windows

echo LLMKG Model Setup
echo =================

REM Create directories
echo Creating model directories...
mkdir model_weights\bert-base-uncased 2>nul
mkdir model_weights\minilm-l6-v2 2>nul
mkdir model_weights\bert-large-ner 2>nul

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is required
    exit /b 1
)

REM Install required Python packages
echo Installing Python dependencies...
pip install -q huggingface_hub

REM Run the download script
echo Downloading models...
python scripts\download_models.py

REM Create a marker file to indicate models are ready
if %errorlevel% equ 0 (
    echo Models downloaded successfully!
    echo. > model_weights\.models_ready
) else (
    echo Failed to download models
    exit /b 1
)

echo Setup complete!