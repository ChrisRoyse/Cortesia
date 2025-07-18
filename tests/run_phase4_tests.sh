#!/bin/bash

# Script to run Phase 4 tests with proper configuration

echo "🚀 Running Phase 4 Tests for LLMKG"
echo "=================================="

# Set environment variables
export RUST_TEST_THREADS=4
export RUST_BACKTRACE=1
export RUST_LOG=warn

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your DeepSeek API key"
    exit 1
fi

# Source the .env file
source .env

# Check if DeepSeek API key is set
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "❌ Error: DEEPSEEK_API_KEY not set in .env!"
    exit 1
fi

echo "✓ Environment configured"
echo ""

# Function to run a test suite
run_test_suite() {
    local suite_name=$1
    local test_module=$2
    
    echo "📋 Running $suite_name..."
    echo "------------------------"
    
    if cargo test --test $test_module -- --nocapture --test-threads=4; then
        echo "✅ $suite_name passed!"
    else
        echo "❌ $suite_name failed!"
        FAILED_TESTS+=("$suite_name")
    fi
    echo ""
}

# Array to track failed tests
FAILED_TESTS=()

# Run basic unit tests first
echo "1️⃣ Running Unit Tests"
run_test_suite "Realistic Tests" "phase4_realistic_tests"

# Run integration tests
echo "2️⃣ Running Integration Tests"
run_test_suite "DeepSeek Integration" "phase4_deepseek_integration"

# Run stress tests (optional - can be slow)
if [ "$1" == "--include-stress" ]; then
    echo "3️⃣ Running Stress Tests (this may take a while)"
    run_test_suite "Advanced Stress Tests" "phase4_advanced_stress_tests"
else
    echo "3️⃣ Skipping stress tests (use --include-stress to run)"
fi

# Run scenario tests
echo "4️⃣ Running Scenario Tests"
run_test_suite "Integration Scenarios" "phase4_integration_scenarios"

# Summary
echo ""
echo "=================================="
echo "📊 Test Summary"
echo "=================================="

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    exit 1
fi