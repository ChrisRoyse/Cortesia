@echo off
echo Checking LLMKG Servers Status...
echo ================================

echo.
echo Checking API Server (port 3001)...
curl -s -o nul -w "HTTP Status: %%{http_code}\n" http://localhost:3001/api/v1/discovery
if %ERRORLEVEL% EQU 0 (
    echo ✅ API Server is running
) else (
    echo ❌ API Server is NOT running
)

echo.
echo Checking Dashboard HTTP (port 8080)...
curl -s -o nul -w "HTTP Status: %%{http_code}\n" http://localhost:8080
if %ERRORLEVEL% EQU 0 (
    echo ✅ Dashboard HTTP is running
) else (
    echo ❌ Dashboard HTTP is NOT running
)

echo.
echo Checking WebSocket (port 8081)...
powershell -Command "try { $ws = New-Object System.Net.WebClient; $ws.DownloadString('http://localhost:8081') } catch { if ($_.Exception.InnerException.Response.StatusCode -eq 'BadRequest') { Write-Host '✅ WebSocket server is running (returns 400 for HTTP request as expected)' } else { Write-Host '❌ WebSocket server is NOT running' } }"

echo.
echo ================================
echo.
echo If servers are not running, start them with:
echo   start_servers.bat
echo.
pause