#!/bin/bash
# Script to run knowledge engine integration tests

echo "Running Knowledge Engine Integration Tests..."
echo "=========================================="

# Run all knowledge engine tests
cargo test test_knowledge_engine --test '*' -- --nocapture

# Check exit code
if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
else
    echo "Some tests failed. Please check the output above."
    exit 1
fi