@echo off
REM Windows test runner with automatic cleanup

echo LLMKG Test Runner - Windows
echo ==========================

REM Kill any existing test processes first
echo Cleaning up any existing test processes...
taskkill /F /IM "llmkg-*.exe" >nul 2>&1

REM Clean up old test executables
del /F /Q "target\debug\deps\llmkg-*.exe" >nul 2>&1

REM Set environment variables
set RUST_TEST_THREADS=1
set RUST_BACKTRACE=1
set CARGO_INCREMENTAL=0

REM Run tests with single thread to prevent conflicts
echo Running tests...
cargo test --test-threads=1 %*

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

REM Cleanup after tests
echo Cleaning up test processes...
timeout /t 1 /nobreak >nul
taskkill /F /IM "llmkg-*.exe" >nul 2>&1

REM Exit with the test exit code
exit /b %EXIT_CODE%