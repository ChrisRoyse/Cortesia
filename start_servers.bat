@echo off
echo Starting LLMKG Servers...
echo =========================

echo [1/2] Building the API server...
cargo build --bin llmkg_api_server --release

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build API server
    pause
    exit /b 1
)

echo [2/2] Starting the API server (includes WebSocket on port 8081)...
start "LLMKG API Server" cargo run --bin llmkg_api_server --release

echo.
echo ✅ LLMKG API Server starting up...
echo.
echo Services will be available at:
echo   - API endpoints: http://localhost:3001/api/v1
echo   - Dashboard: http://localhost:8080
echo   - WebSocket: ws://localhost:8081
echo   - API Discovery: http://localhost:3001/api/v1/discovery
echo.
echo Press any key to close this window (server will continue running)...
pause >nul