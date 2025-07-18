#!/bin/bash

# Integration Test Runner Script
# Orchestrates execution of integration tests in different scenarios

set -euo pipefail

# Configuration
SCENARIO=${SCENARIO:-"single-node"}
OUTPUT_DIR=${OUTPUT_DIR:-"/results"}
TEST_TIMEOUT=${TEST_TIMEOUT:-"3600"}
PARALLEL_TESTS=${PARALLEL_TESTS:-"false"}

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a test suite
run_test_suite() {
    local suite_name="$1"
    local test_args="$2"
    local timeout="$3"
    
    log "Starting test suite: $suite_name"
    
    local start_time=$(date +%s)
    local result_file="$OUTPUT_DIR/${suite_name}-results.xml"
    local log_file="$OUTPUT_DIR/${suite_name}.log"
    
    # Run tests with timeout
    if timeout "$timeout" cargo test $test_args \
        --message-format json \
        > "$result_file" 2> "$log_file"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "Test suite $suite_name completed successfully in ${duration}s"
        return 0
    else
        local exit_code=$?
        log_error "Test suite $suite_name failed with exit code $exit_code"
        return $exit_code
    fi
}

# Function to collect system metrics
collect_metrics() {
    local output_file="$1"
    
    {
        echo "=== System Information ==="
        uname -a
        echo ""
        
        echo "=== CPU Information ==="
        cat /proc/cpuinfo | grep "model name" | head -1
        nproc
        echo ""
        
        echo "=== Memory Information ==="
        free -h
        echo ""
        
        echo "=== Disk Information ==="
        df -h
        echo ""
        
        echo "=== Network Information ==="
        ip addr show || ifconfig
        echo ""
        
        echo "=== Environment Variables ==="
        env | grep -E "(LLMKG|RUST|CARGO)" | sort
        
    } > "$output_file"
}

# Function to monitor resources during test execution
monitor_resources() {
    local output_file="$1"
    local pid="$2"
    
    {
        echo "timestamp,cpu_percent,memory_mb,disk_io_read,disk_io_write"
        
        while kill -0 "$pid" 2>/dev/null; do
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            local cpu_percent=$(ps -p "$pid" -o %cpu --no-headers | tr -d ' ' || echo "0")
            local memory_kb=$(ps -p "$pid" -o rss --no-headers | tr -d ' ' || echo "0")
            local memory_mb=$((memory_kb / 1024))
            
            # Get disk I/O stats if available
            local disk_read="0"
            local disk_write="0"
            if [ -f "/proc/$pid/io" ]; then
                disk_read=$(grep "read_bytes" "/proc/$pid/io" | awk '{print $2}' || echo "0")
                disk_write=$(grep "write_bytes" "/proc/$pid/io" | awk '{print $2}' || echo "0")
            fi
            
            echo "$timestamp,$cpu_percent,$memory_mb,$disk_read,$disk_write"
            sleep 5
        done
    } > "$output_file"
}

# Main execution based on scenario
case "$SCENARIO" in
    "single-node")
        log "Running single-node integration tests"
        
        # Collect system information
        collect_metrics "$OUTPUT_DIR/system-info.txt"
        
        # Core integration tests
        run_test_suite "graph-storage" \
            "--test graph_storage_integration --release" \
            "600" &
        
        run_test_suite "embedding-graph" \
            "--test embedding_graph_integration --release" \
            "600" &
        
        run_test_suite "performance" \
            "--test performance_integration --release" \
            "1200" &
        
        # Wait for all tests if running in parallel
        if [ "$PARALLEL_TESTS" = "true" ]; then
            wait
        fi
        
        # WASM tests (sequential)
        if command -v wasm-pack >/dev/null 2>&1; then
            log "Running WASM integration tests"
            run_test_suite "wasm" \
                "--test wasm_integration --target wasm32-unknown-unknown" \
                "300"
        else
            log "Skipping WASM tests - wasm-pack not available"
        fi
        ;;
        
    "distributed")
        log "Running distributed integration tests"
        
        # Start multiple test nodes
        export LLMKG_NODE_ID=1
        export LLMKG_CLUSTER_SIZE=3
        
        # MCP federation tests
        run_test_suite "mcp-federation" \
            "--test mcp_integration --release -- federation" \
            "900"
        
        # Cross-node consistency tests
        run_test_suite "distributed-consistency" \
            "--test distributed_integration --release" \
            "1200"
        ;;
        
    "federation")
        log "Running federation integration tests"
        
        # Multi-database federation tests
        export LLMKG_FEDERATION_MODE=true
        export LLMKG_SHARD_COUNT=5
        
        run_test_suite "federation-core" \
            "--test mcp_integration --release -- federated" \
            "1200"
        
        run_test_suite "federation-performance" \
            "--test performance_integration --release -- federation" \
            "1800"
        ;;
        
    "stress")
        log "Running stress tests"
        
        # Extended stress testing
        export RUST_TEST_TIME_UNIT=60000
        export RUST_TEST_TIME_INTEGRATION=3600000
        export LLMKG_STRESS_TEST_SIZE=large
        
        # Monitor resources during stress test
        run_test_suite "stress" \
            "--test performance_integration --release -- stress --ignored" \
            "3600" &
        
        local test_pid=$!
        monitor_resources "$OUTPUT_DIR/stress-resource-usage.csv" "$test_pid" &
        local monitor_pid=$!
        
        wait "$test_pid"
        kill "$monitor_pid" 2>/dev/null || true
        ;;
        
    *)
        log_error "Unknown scenario: $SCENARIO"
        exit 1
        ;;
esac

# Generate test report
if command -v python3 >/dev/null 2>&1; then
    log "Generating test report"
    python3 /usr/local/bin/generate-test-report.py \
        --input-dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/integration-test-report.html" \
        --scenario "$SCENARIO"
fi

# Collect final metrics
{
    echo "=== Test Execution Summary ==="
    echo "Scenario: $SCENARIO"
    echo "Completion time: $(date)"
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    
    echo "=== Test Results ==="
    find "$OUTPUT_DIR" -name "*-results.xml" -exec basename {} \; | while read -r file; do
        echo "- $file"
    done
    echo ""
    
    echo "=== Log Files ==="
    find "$OUTPUT_DIR" -name "*.log" -exec basename {} \; | while read -r file; do
        local size=$(du -h "$OUTPUT_DIR/$file" | cut -f1)
        echo "- $file ($size)"
    done
    
} > "$OUTPUT_DIR/execution-summary.txt"

log "Integration test execution completed"

# Return appropriate exit code
if find "$OUTPUT_DIR" -name "*-results.xml" -exec grep -l "failures=\"0\"" {} \; | wc -l | grep -q "$(find "$OUTPUT_DIR" -name "*-results.xml" | wc -l)"; then
    log "All tests passed"
    exit 0
else
    log_error "Some tests failed"
    exit 1
fi