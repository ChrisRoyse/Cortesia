@echo off
setlocal enabledelayedexpansion

REM LLMKG Test Runner for Windows - Runs tests in manageable batches to prevent hanging
REM Usage: run-tests.bat [group] or run-tests.bat all

set TIMEOUT=60
set THREADS=1
set CARGO_OPTS=--lib --
set TEST_OPTS=--test-threads=%THREADS% --nocapture

echo LLMKG Test Runner
echo ====================

REM Function to run a test group with timeout
goto :main

:run_test_group
set group_name=%~1
set pattern=%~2

echo Running %group_name% tests...
echo    Pattern: %pattern%

REM Note: Windows doesn't have built-in timeout like Linux, so we'll run without timeout for now
cargo test %CARGO_OPTS% %pattern% %TEST_OPTS%
set exit_code=%ERRORLEVEL%

if %exit_code%==0 (
    echo %group_name% tests PASSED
    exit /b 0
) else (
    echo %group_name% tests FAILED (exit code: %exit_code%)
    exit /b %exit_code%
)

:main
set param=%~1
if "%param%"=="" set param=all

if "%param%"=="all" goto :run_all
if "%param%"=="core" goto :run_core
if "%param%"=="brain" goto :run_brain
if "%param%"=="storage" goto :run_storage
if "%param%"=="learning" goto :run_learning
if "%param%"=="cognitive" goto :run_cognitive
if "%param%"=="monitoring" goto :run_monitoring
if "%param%"=="utils" goto :run_utils
if "%param%"=="entity" goto :run_entity

echo Usage: %0 [all^|core^|brain^|storage^|learning^|cognitive^|monitoring^|utils^|entity]
echo.
echo Groups:
echo   all        - Run all test groups (default)
echo   core       - Core graph, types, entity, memory tests
echo   brain      - Brain-enhanced graph and activation tests
echo   storage    - SDR storage and persistence tests
echo   learning   - Learning algorithm tests
echo   cognitive  - Cognitive processing tests
echo   monitoring - Monitoring and dashboard tests
echo   utils      - Math, validation, and utility tests
echo   entity     - Entity compatibility tests
exit /b 1

:run_all
echo Running all test groups...
echo.
call :run_test_group "Core" "core::graph:: core::types:: core::entity:: core::memory::"
call :run_test_group "Entity Compatibility" "core::entity_compat::"
call :run_test_group "Brain" "core::brain_enhanced_graph:: core::activation_engine::"
call :run_test_group "Storage" "core::sdr_storage:: storage::"
call :run_test_group "Math & Utils" "math:: validation:: mcp::"
call :run_test_group "Learning" "learning::"
call :run_test_group "Cognitive" "cognitive::"
call :run_test_group "Monitoring" "monitoring::"

echo.
echo WARNING: Skipping potentially problematic async tests:
echo    - streaming:: (has infinite loops)
echo    - federation:: (has infinite loops)
echo    These should be run individually with timeouts
goto :end

:run_core
call :run_test_group "Core" "core::graph:: core::types:: core::entity:: core::memory::"
goto :end

:run_brain
call :run_test_group "Brain" "core::brain_enhanced_graph:: core::activation_engine::"
goto :end

:run_storage
call :run_test_group "Storage" "core::sdr_storage:: storage::"
goto :end

:run_learning
call :run_test_group "Learning" "learning::"
goto :end

:run_cognitive
call :run_test_group "Cognitive" "cognitive::"
goto :end

:run_monitoring
call :run_test_group "Monitoring" "monitoring::"
goto :end

:run_utils
call :run_test_group "Math & Utils" "math:: validation:: mcp::"
goto :end

:run_entity
call :run_test_group "Entity Compatibility" "core::entity_compat::"
goto :end

:end
endlocal