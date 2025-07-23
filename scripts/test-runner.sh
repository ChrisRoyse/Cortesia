#!/bin/bash
# Cross-platform test runner for Unix systems

echo "LLMKG Test Runner - Unix Edition"
echo "================================"

# Function to run tests
run_tests() {
    echo -e "\nRunning tests with args: $@"
    
    # Set environment variables
    export RUST_TEST_THREADS=1
    export RUST_BACKTRACE=1
    
    # Run the tests
    cargo test "$@"
    return $?
}

# Main execution
run_tests "$@"
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo -e "\nTests completed successfully!"
else
    echo -e "\nTests failed with exit code: $exit_code"
fi

exit $exit_code