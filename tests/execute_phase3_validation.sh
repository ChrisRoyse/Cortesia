#!/bin/bash

# Phase 3 Validation Test Execution Script

echo "=================================================="
echo "PHASE 3 COMPREHENSIVE VALIDATION TEST SUITE"
echo "=================================================="
echo ""
echo "This script ensures all Phase 3 components are"
echo "working as intended through comprehensive testing."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run a test and check result
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "${BLUE}Running: ${test_name}${NC}"
    
    if eval $test_command; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}\n"
        return 0
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}\n"
        return 1
    fi
}

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0

echo "Starting Phase 3 validation tests..."
echo ""

# Test 1: Compile all test files
((TOTAL_TESTS++))
if run_test "Compilation Check" "cargo check --tests"; then
    ((PASSED_TESTS++))
fi

# Test 2: Run unit tests for working memory
((TOTAL_TESTS++))
if run_test "Working Memory Tests" "cargo test --test phase3_validation_suite test_working_memory -- --nocapture"; then
    ((PASSED_TESTS++))
fi

# Test 3: Run attention coordination tests
((TOTAL_TESTS++))
if run_test "Attention Coordination Tests" "cargo test --test phase3_validation_suite test_attention_memory -- --nocapture"; then
    ((PASSED_TESTS++))
fi

# Test 4: Run inhibition learning tests
((TOTAL_TESTS++))
if run_test "Inhibition Learning Tests" "cargo test --test phase3_validation_suite test_inhibition_learning -- --nocapture"; then
    ((PASSED_TESTS++))
fi

# Test 5: Run unified memory tests
((TOTAL_TESTS++))
if run_test "Unified Memory Tests" "cargo test --test phase3_validation_suite test_unified_memory -- --nocapture"; then
    ((PASSED_TESTS++))
fi

# Test 6: Run complex reasoning tests
((TOTAL_TESTS++))
if run_test "Complex Reasoning Tests" "cargo test --test phase3_validation_suite test_phase3_complex_reasoning -- --nocapture"; then
    ((PASSED_TESTS++))
fi

# Test 7: Run resilience tests
((TOTAL_TESTS++))
if run_test "System Resilience Tests" "cargo test --test phase3_validation_suite test_system_resilience -- --nocapture"; then
    ((PASSED_TESTS++))
fi

# Test 8: Run edge case tests
((TOTAL_TESTS++))
if run_test "Edge Case Tests" "cargo test --test phase3_validation_suite test_pathological_edge_cases -- --nocapture"; then
    ((PASSED_TESTS++))
fi

# Test 9: Run performance benchmarks
((TOTAL_TESTS++))
if run_test "Performance Benchmarks" "cargo test --test phase3_validation_suite test_performance_benchmarks -- --nocapture"; then
    ((PASSED_TESTS++))
fi

# Test 10: Run complete validation
((TOTAL_TESTS++))
if run_test "Complete Phase 3 Validation" "cargo test --test phase3_validation_suite test_phase3_complete_validation -- --nocapture"; then
    ((PASSED_TESTS++))
fi

# Summary
echo ""
echo "=================================================="
echo "VALIDATION SUMMARY"
echo "=================================================="
echo ""

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED! (${PASSED_TESTS}/${TOTAL_TESTS})${NC}"
    echo ""
    echo -e "${GREEN}Phase 3 is fully operational and working as intended!${NC}"
    echo ""
    echo "Validated Components:"
    echo "  ✓ Working Memory System (with capacity limits and decay)"
    echo "  ✓ Attention Management (with memory coordination)"
    echo "  ✓ Competitive Inhibition (with learning mechanisms)"
    echo "  ✓ Unified Memory Integration"
    echo "  ✓ Complex Reasoning Capabilities"
    echo "  ✓ System Resilience and Error Recovery"
    echo "  ✓ Performance Targets Met"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED (${PASSED_TESTS}/${TOTAL_TESTS})${NC}"
    echo ""
    echo "Please review the failed tests above and fix any issues."
    echo "All tests must pass for Phase 3 to be considered fully functional."
    exit 1
fi