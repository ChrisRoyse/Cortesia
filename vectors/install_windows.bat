@echo off
REM Installation script for MCP RAG Indexer on Windows

echo ================================================
echo    MCP RAG Indexer - Installation for Windows
echo ================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Installing dependencies...
pip install --upgrade pip
pip install mcp langchain langchain-community langchain-huggingface chromadb sentence-transformers gitpython psutil pyyaml tomli torch numpy

echo.
echo [2/5] Creating MCP executable wrapper...

REM Get Python Scripts directory
for /f "tokens=*" %%i in ('python -c "import site; print(site.USER_BASE)"') do set PYTHON_USER_BASE=%%i
set SCRIPTS_DIR=%PYTHON_USER_BASE%\Scripts

REM Create the executable wrapper
echo Creating %SCRIPTS_DIR%\mcp-rag-indexer.exe...
python -c "import sys; import os; from pathlib import Path; script_path = Path('%CD%') / 'mcp_server.py'; exec_path = Path('%SCRIPTS_DIR%') / 'mcp-rag-indexer.exe'; import shutil; shutil.copy(sys.executable, exec_path); print(f'Executable created at: {exec_path}')"

REM Create batch wrapper as backup
echo @echo off > "%SCRIPTS_DIR%\mcp-rag-indexer.bat"
echo python "%CD%\mcp_server.py" %%* >> "%SCRIPTS_DIR%\mcp-rag-indexer.bat"

echo.
echo [3/5] Setting up data directory...
if not exist "%USERPROFILE%\.mcp-rag-indexer" mkdir "%USERPROFILE%\.mcp-rag-indexer"
if not exist "%USERPROFILE%\.mcp-rag-indexer\databases" mkdir "%USERPROFILE%\.mcp-rag-indexer\databases"

echo.
echo [4/5] Configuring Claude Code...

REM Check if .claude.json exists
if exist "%USERPROFILE%\.claude.json" (
    echo Found existing Claude configuration.
    echo.
    echo Please add the following to your .claude.json file:
    echo.
    echo {
    echo   "mcpServers": {
    echo     "rag-indexer": {
    echo       "type": "stdio",
    echo       "command": "%SCRIPTS_DIR:\=\\%\\mcp-rag-indexer.exe",
    echo       "args": ["--log-level", "info"]
    echo     }
    echo   }
    echo }
    echo.
) else (
    echo Creating Claude configuration...
    (
        echo {
        echo   "mcpServers": {
        echo     "rag-indexer": {
        echo       "type": "stdio",
        echo       "command": "%SCRIPTS_DIR:\=\\%\\mcp-rag-indexer.exe",
        echo       "args": ["--log-level", "info"]
        echo     }
        echo   }
        echo }
    ) > "%USERPROFILE%\.claude.json"
    echo Configuration created at %USERPROFILE%\.claude.json
)

echo.
echo [5/5] Testing installation...
python mcp_server.py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Installation successful!
) else (
    echo × Installation test failed. Please check error messages above.
    pause
    exit /b 1
)

echo.
echo ================================================
echo    Installation Complete!
echo ================================================
echo.
echo Next steps:
echo 1. Restart Claude Code completely
echo 2. Check MCP connection: Type /mcp in Claude
echo 3. Index your first project: "Index C:\your\project\path"
echo.
echo Executable location: %SCRIPTS_DIR%\mcp-rag-indexer.exe
echo Log location: %USERPROFILE%\.mcp-rag-indexer\server.log
echo.
pause