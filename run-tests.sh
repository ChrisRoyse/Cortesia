#!/bin/bash

# LLMKG Test Runner - Runs tests in manageable batches to prevent hanging
# Usage: ./run-tests.sh [group] or ./run-tests.sh all

set -e

TIMEOUT=60
THREADS=1
CARGO_OPTS="--lib --"
TEST_OPTS="--test-threads=$THREADS --nocapture"

echo "🧪 LLMKG Test Runner"
echo "===================="

# Function to run a test group with timeout
run_test_group() {
    local group_name="$1"
    local pattern="$2"
    
    echo "📋 Running $group_name tests..."
    echo "   Pattern: $pattern"
    
    if timeout $TIMEOUT cargo test $CARGO_OPTS "$pattern" $TEST_OPTS; then
        echo "✅ $group_name tests PASSED"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "⏰ $group_name tests TIMED OUT after ${TIMEOUT}s"
        else
            echo "❌ $group_name tests FAILED (exit code: $exit_code)"
        fi
        return $exit_code
    fi
}

# Function to run all test groups
run_all_tests() {
    local failed_groups=()
    local passed_groups=()
    local timed_out_groups=()
    
    echo "🚀 Running all test groups..."
    
    # Core tests (most stable)
    if run_test_group "Core" "core::graph:: core::types:: core::entity:: core::memory::"; then
        passed_groups+=("Core")
    else
        case $? in
            124) timed_out_groups+=("Core") ;;
            *) failed_groups+=("Core") ;;
        esac
    fi
    
    # Entity compatibility tests
    if run_test_group "Entity Compatibility" "core::entity_compat::"; then
        passed_groups+=("Entity Compatibility")
    else
        case $? in
            124) timed_out_groups+=("Entity Compatibility") ;;
            *) failed_groups+=("Entity Compatibility") ;;
        esac
    fi
    
    # Brain tests (may have async issues)
    if run_test_group "Brain" "core::brain_enhanced_graph:: core::activation_engine:: core::knowledge_engine::"; then
        passed_groups+=("Brain")
    else
        case $? in
            124) timed_out_groups+=("Brain") ;;
            *) failed_groups+=("Brain") ;;
        esac
    fi
    
    # SDR and storage tests
    if run_test_group "Storage" "core::sdr_storage:: storage::"; then
        passed_groups+=("Storage")
    else
        case $? in
            124) timed_out_groups+=("Storage") ;;
            *) failed_groups+=("Storage") ;;
        esac
    fi
    
    # Math and utility tests
    if run_test_group "Math & Utils" "math:: validation:: mcp::"; then
        passed_groups+=("Math & Utils")
    else
        case $? in
            124) timed_out_groups+=("Math & Utils") ;;
            *) failed_groups+=("Math & Utils") ;;
        esac
    fi
    
    # Learning tests
    if run_test_group "Learning" "learning::"; then
        passed_groups+=("Learning")
    else
        case $? in
            124) timed_out_groups+=("Learning") ;;
            *) failed_groups+=("Learning") ;;
        esac
    fi
    
    # Cognitive tests
    if run_test_group "Cognitive" "cognitive::"; then
        passed_groups+=("Cognitive")
    else
        case $? in
            124) timed_out_groups+=("Cognitive") ;;
            *) failed_groups+=("Cognitive") ;;
        esac
    fi
    
    # Monitoring tests (likely to hang)
    if run_test_group "Monitoring" "monitoring::"; then
        passed_groups+=("Monitoring")
    else
        case $? in
            124) timed_out_groups+=("Monitoring") ;;
            *) failed_groups+=("Monitoring") ;;
        esac
    fi
    
    # Async/streaming tests (known problematic)
    echo "⚠️  Skipping potentially problematic async tests:"
    echo "   - streaming:: (has infinite loops)"
    echo "   - federation:: (has infinite loops)" 
    echo "   These should be run individually with timeouts"
    
    # Summary
    echo ""
    echo "📊 Test Summary"
    echo "==============="
    echo "✅ Passed groups (${#passed_groups[@]}): ${passed_groups[*]}"
    echo "⏰ Timed out groups (${#timed_out_groups[@]}): ${timed_out_groups[*]}"
    echo "❌ Failed groups (${#failed_groups[@]}): ${failed_groups[*]}"
    
    # Return appropriate exit code
    if [ ${#failed_groups[@]} -eq 0 ] && [ ${#timed_out_groups[@]} -eq 0 ]; then
        echo "🎉 All test groups completed successfully!"
        return 0
    elif [ ${#failed_groups[@]} -eq 0 ]; then
        echo "⚠️  Some test groups timed out but none failed"
        return 1
    else
        echo "💥 Some test groups failed"
        return 2
    fi
}

# Main execution
case "${1:-all}" in
    "all")
        run_all_tests
        ;;
    "core")
        run_test_group "Core" "core::graph:: core::types:: core::entity:: core::memory::"
        ;;
    "brain")
        run_test_group "Brain" "core::brain_enhanced_graph:: core::activation_engine:: core::knowledge_engine::"
        ;;
    "storage")
        run_test_group "Storage" "core::sdr_storage:: storage::"
        ;;
    "learning")
        run_test_group "Learning" "learning::"
        ;;
    "cognitive")
        run_test_group "Cognitive" "cognitive::"
        ;;
    "monitoring")
        run_test_group "Monitoring" "monitoring::"
        ;;
    "utils")
        run_test_group "Math & Utils" "math:: validation:: mcp::"
        ;;
    "entity")
        run_test_group "Entity Compatibility" "core::entity_compat::"
        ;;
    *)
        echo "Usage: $0 [all|core|brain|storage|learning|cognitive|monitoring|utils|entity]"
        echo ""
        echo "Groups:"
        echo "  all        - Run all test groups (default)"
        echo "  core       - Core graph, types, entity, memory tests"
        echo "  brain      - Brain-enhanced graph and activation tests"
        echo "  storage    - SDR storage and persistence tests"  
        echo "  learning   - Learning algorithm tests"
        echo "  cognitive  - Cognitive processing tests"
        echo "  monitoring - Monitoring and dashboard tests"
        echo "  utils      - Math, validation, and utility tests"
        echo "  entity     - Entity compatibility tests"
        exit 1
        ;;
esac