# PowerShell test runner with automatic cleanup for Windows
param(
    [string]$TestArgs = ""
)

Write-Host "LLMKG Test Runner - Windows Edition" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green

# Function to kill any lingering test processes
function Cleanup-TestProcesses {
    Write-Host "`nCleaning up test processes..." -ForegroundColor Yellow
    
    # Get all llmkg-related processes
    $processes = Get-Process | Where-Object { $_.ProcessName -like "*llmkg*" -and $_.ProcessName -ne "llmkg_api_server" }
    
    if ($processes.Count -gt 0) {
        Write-Host "Found $($processes.Count) test processes to clean up" -ForegroundColor Yellow
        $processes | ForEach-Object {
            try {
                Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
                Write-Host "  Killed process: $($_.ProcessName) (PID: $($_.Id))" -ForegroundColor Gray
            } catch {
                # Process might have already exited
            }
        }
    } else {
        Write-Host "No lingering test processes found" -ForegroundColor Gray
    }
    
    # Wait a moment for processes to fully terminate
    Start-Sleep -Milliseconds 500
    
    # Clean up any locked test executables
    $targetDir = Join-Path $PSScriptRoot "..\target\debug\deps"
    if (Test-Path $targetDir) {
        Get-ChildItem -Path $targetDir -Filter "llmkg-*.exe" | ForEach-Object {
            try {
                Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
                Write-Host "  Removed old test executable: $($_.Name)" -ForegroundColor Gray
            } catch {
                # File might be locked, that's okay
            }
        }
    }
}

# Function to run tests with proper error handling
function Run-Tests {
    param([string]$Args)
    
    Write-Host "`nRunning tests with args: $Args" -ForegroundColor Cyan
    
    try {
        # Set environment variables for this session
        $env:RUST_TEST_THREADS = "1"
        $env:RUST_BACKTRACE = "1"
        $env:CARGO_INCREMENTAL = "0"  # Disable incremental compilation for tests
        
        # Run the tests
        $process = Start-Process -FilePath "cargo" -ArgumentList "test $Args" -NoNewWindow -PassThru -Wait
        
        return $process.ExitCode
    } catch {
        Write-Host "Error running tests: $_" -ForegroundColor Red
        return 1
    }
}

# Main execution
try {
    # Initial cleanup
    Cleanup-TestProcesses
    
    # Register cleanup on script exit
    Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
        Cleanup-TestProcesses
    }
    
    # Run the tests
    $exitCode = Run-Tests -Args $TestArgs
    
    # Final cleanup
    Cleanup-TestProcesses
    
    if ($exitCode -eq 0) {
        Write-Host "`nTests completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "`nTests failed with exit code: $exitCode" -ForegroundColor Red
    }
    
    exit $exitCode
} catch {
    Write-Host "Unexpected error: $_" -ForegroundColor Red
    Cleanup-TestProcesses
    exit 1
}