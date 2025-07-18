# PowerShell script to run Phase 4 tests on Windows

Write-Host "🚀 Running Phase 4 Tests for LLMKG" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Set environment variables
$env:RUST_TEST_THREADS = "4"
$env:RUST_BACKTRACE = "1"
$env:RUST_LOG = "warn"

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ Error: .env file not found!" -ForegroundColor Red
    Write-Host "Please create a .env file with your DeepSeek API key" -ForegroundColor Yellow
    exit 1
}

# Load .env file
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $key = $matches[1]
        $value = $matches[2]
        Set-Item -Path "env:$key" -Value $value
    }
}

# Check if DeepSeek API key is set
if (-not $env:DEEPSEEK_API_KEY) {
    Write-Host "❌ Error: DEEPSEEK_API_KEY not set in .env!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Environment configured" -ForegroundColor Green
Write-Host ""

# Function to run a test suite
function Run-TestSuite {
    param(
        [string]$SuiteName,
        [string]$TestModule
    )
    
    Write-Host "📋 Running $SuiteName..." -ForegroundColor Yellow
    Write-Host "------------------------"
    
    $result = cargo test --test $TestModule -- --nocapture --test-threads=4 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host "✅ $SuiteName passed!" -ForegroundColor Green
    } else {
        Write-Host "❌ $SuiteName failed!" -ForegroundColor Red
        $script:FailedTests += $SuiteName
    }
    
    Write-Host ""
}

# Array to track failed tests
$FailedTests = @()

# Run basic unit tests first
Write-Host "1️⃣ Running Unit Tests" -ForegroundColor Cyan
Run-TestSuite "Realistic Tests" "phase4_realistic_tests"

# Run integration tests
Write-Host "2️⃣ Running Integration Tests" -ForegroundColor Cyan
Run-TestSuite "DeepSeek Integration" "phase4_deepseek_integration"

# Run stress tests (optional - can be slow)
if ($args -contains "--include-stress") {
    Write-Host "3️⃣ Running Stress Tests (this may take a while)" -ForegroundColor Cyan
    Run-TestSuite "Advanced Stress Tests" "phase4_advanced_stress_tests"
} else {
    Write-Host "3️⃣ Skipping stress tests (use --include-stress to run)" -ForegroundColor Gray
}

# Run scenario tests
Write-Host "4️⃣ Running Scenario Tests" -ForegroundColor Cyan
Run-TestSuite "Integration Scenarios" "phase4_integration_scenarios"

# Summary
Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "📊 Test Summary" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

if ($FailedTests.Count -eq 0) {
    Write-Host "✅ All tests passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ Failed tests:" -ForegroundColor Red
    foreach ($test in $FailedTests) {
        Write-Host "  - $test" -ForegroundColor Red
    }
    exit 1
}