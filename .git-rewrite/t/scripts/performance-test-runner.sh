#!/bin/bash

# Performance Test Runner Script
# Orchestrates execution of performance integration tests with resource monitoring

set -euo pipefail

# Configuration
OUTPUT_DIR=${OUTPUT_DIR:-"/results"}
TEST_TIMEOUT=${TEST_TIMEOUT:-"3600"}  # 1 hour
SCENARIO=${SCENARIO:-"comprehensive"}
MONITORING_INTERVAL=${MONITORING_INTERVAL:-"5"}  # seconds
DEBUG_MODE=${DEBUG_MODE:-"false"}

# Performance test configurations
declare -A TEST_CONFIGS=(
    ["quick"]="1000,2000,5000"
    ["comprehensive"]="1000,5000,10000,25000,50000"
    ["stress"]="50000,100000,250000"
    ["memory"]="1000,5000,10000"
    ["concurrency"]="5000"
)

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_debug() {
    if [[ "$DEBUG_MODE" == "true" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: $1" >&2
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up performance test environment"
    
    # Kill monitoring processes
    pkill -f "monitor_resources" 2>/dev/null || true
    pkill -f "perf record" 2>/dev/null || true
    
    # Stop any stress test processes
    pkill -f "stress-ng" 2>/dev/null || true
    
    # Clean up temporary files
    rm -f /tmp/llmkg-perf-* /tmp/monitor-*.pid
}

trap cleanup EXIT

# Create output directory
mkdir -p "$OUTPUT_DIR"

# System resource monitoring
start_resource_monitoring() {
    local output_file="$1"
    local pid_file="/tmp/monitor-$$.pid"
    
    log "Starting resource monitoring (interval: ${MONITORING_INTERVAL}s)"
    
    {
        echo "timestamp,cpu_percent,memory_mb,memory_percent,disk_read_mb,disk_write_mb,network_rx_mb,network_tx_mb"
        
        while true; do
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            
            # CPU usage
            local cpu_percent=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
            
            # Memory usage
            local memory_info=$(free -m | grep "^Mem:")
            local memory_total=$(echo $memory_info | awk '{print $2}')
            local memory_used=$(echo $memory_info | awk '{print $3}')
            local memory_percent=$(echo "scale=2; $memory_used * 100 / $memory_total" | bc -l)
            
            # Disk I/O (if iostat is available)
            local disk_read=0
            local disk_write=0
            if command -v iostat &> /dev/null; then
                local disk_stats=$(iostat -d 1 1 | tail -n +4 | head -1)
                disk_read=$(echo $disk_stats | awk '{print $3}')
                disk_write=$(echo $disk_stats | awk '{print $4}')
            fi
            
            # Network I/O (if available)
            local network_rx=0
            local network_tx=0
            if [[ -f /proc/net/dev ]]; then
                local net_stats=$(grep "eth0\|enp" /proc/net/dev | head -1)
                if [[ -n "$net_stats" ]]; then
                    network_rx=$(echo $net_stats | awk '{print $2}')
                    network_tx=$(echo $net_stats | awk '{print $10}')
                    # Convert to MB
                    network_rx=$(echo "scale=2; $network_rx / 1048576" | bc -l)
                    network_tx=$(echo "scale=2; $network_tx / 1048576" | bc -l)
                fi
            fi
            
            echo "$timestamp,$cpu_percent,$memory_used,$memory_percent,$disk_read,$disk_write,$network_rx,$network_tx"
            
            sleep "$MONITORING_INTERVAL"
        done
    } > "$output_file" &
    
    local monitor_pid=$!
    echo "$monitor_pid" > "$pid_file"
    log_debug "Resource monitoring started with PID $monitor_pid"
}

stop_resource_monitoring() {
    local pid_file="/tmp/monitor-$$.pid"
    
    if [[ -f "$pid_file" ]]; then
        local monitor_pid=$(cat "$pid_file")
        kill "$monitor_pid" 2>/dev/null || true
        rm -f "$pid_file"
        log_debug "Resource monitoring stopped"
    fi
}

# Performance profiling
start_performance_profiling() {
    local output_file="$1"
    
    if command -v perf &> /dev/null; then
        log "Starting performance profiling"
        perf record -g -o "$output_file" -p $$ &
        echo $! > "/tmp/perf-$$.pid"
    else
        log_debug "perf not available, skipping performance profiling"
    fi
}

stop_performance_profiling() {
    local pid_file="/tmp/perf-$$.pid"
    
    if [[ -f "$pid_file" ]]; then
        local perf_pid=$(cat "$pid_file")
        kill -INT "$perf_pid" 2>/dev/null || true
        rm -f "$pid_file"
        log_debug "Performance profiling stopped"
    fi
}

# Memory profiling with Valgrind (if available)
run_with_memory_profiling() {
    local test_command="$1"
    local output_file="$2"
    
    if command -v valgrind &> /dev/null; then
        log "Running with Valgrind memory profiling"
        valgrind --tool=massif \
                 --massif-out-file="$output_file" \
                 --time-unit=ms \
                 --detailed-freq=1 \
                 --max-snapshots=100 \
                 $test_command
    else
        log_debug "Valgrind not available, running without memory profiling"
        eval "$test_command"
    fi
}

# Run specific performance test suite
run_performance_test_suite() {
    local suite_name="$1"
    local entity_counts="$2"
    
    log "Running performance test suite: $suite_name"
    
    IFS=',' read -ra COUNTS <<< "$entity_counts"
    
    for count in "${COUNTS[@]}"; do
        log "Running $suite_name tests with $count entities"
        
        local test_output="$OUTPUT_DIR/${suite_name}-${count}-entities.log"
        local resource_output="$OUTPUT_DIR/${suite_name}-${count}-entities-resources.csv"
        local memory_output="$OUTPUT_DIR/${suite_name}-${count}-entities-memory.out"
        
        # Start monitoring
        start_resource_monitoring "$resource_output"
        start_performance_profiling "$OUTPUT_DIR/${suite_name}-${count}-entities.perf"
        
        # Prepare test environment
        export LLMKG_PERF_TEST_SIZE="$count"
        export LLMKG_PERF_OUTPUT_DIR="$OUTPUT_DIR"
        
        local start_time=$(date +%s)
        
        # Run the test with appropriate configuration
        local test_command=""
        case "$suite_name" in
            "latency")
                test_command="cargo test --release --test performance_integration test_query_latency_integration -- --nocapture"
                ;;
            "memory")
                test_command="cargo test --release --test performance_integration test_memory_efficiency_integration -- --nocapture"
                run_with_memory_profiling "$test_command" "$memory_output"
                continue  # Skip regular execution since we used Valgrind
                ;;
            "compression")
                test_command="cargo test --release --test performance_integration test_compression_integration -- --nocapture"
                ;;
            "concurrency")
                test_command="cargo test --release --test performance_integration test_concurrent_access_integration -- --nocapture"
                ;;
            "stress")
                test_command="cargo test --release --test performance_integration test_stress_integration -- --nocapture --ignored"
                ;;
        esac
        
        # Execute the test
        if timeout "$TEST_TIMEOUT" bash -c "$test_command" > "$test_output" 2>&1; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log "✓ $suite_name test with $count entities completed in ${duration}s"
        else
            local exit_code=$?
            log_error "✗ $suite_name test with $count entities failed (exit code: $exit_code)"
        fi
        
        # Stop monitoring
        stop_performance_profiling
        stop_resource_monitoring
        
        # Generate test summary
        generate_test_summary "$suite_name" "$count" "$test_output" "$resource_output"
    done
}

# Generate individual test summary
generate_test_summary() {
    local suite_name="$1"
    local entity_count="$2"
    local test_log="$3"
    local resource_log="$4"
    
    local summary_file="$OUTPUT_DIR/${suite_name}-${entity_count}-summary.json"
    
    # Parse test results
    local test_status="UNKNOWN"
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    if [[ -f "$test_log" ]]; then
        total_tests=$(grep -c "test result:" "$test_log" 2>/dev/null || echo "0")
        passed_tests=$(grep "test result:" "$test_log" 2>/dev/null | grep -c "ok" || echo "0")
        failed_tests=$(grep "test result:" "$test_log" 2>/dev/null | grep -c "FAILED" || echo "0")
        
        if [[ $failed_tests -eq 0 && $total_tests -gt 0 ]]; then
            test_status="PASSED"
        elif [[ $total_tests -gt 0 ]]; then
            test_status="FAILED"
        fi
    fi
    
    # Parse resource usage
    local max_memory=0
    local avg_cpu=0
    local peak_disk_read=0
    local peak_disk_write=0
    
    if [[ -f "$resource_log" ]]; then
        max_memory=$(tail -n +2 "$resource_log" | cut -d',' -f3 | sort -n | tail -1)
        avg_cpu=$(tail -n +2 "$resource_log" | cut -d',' -f2 | awk '{sum+=$1; count++} END {print (count>0) ? sum/count : 0}')
        peak_disk_read=$(tail -n +2 "$resource_log" | cut -d',' -f5 | sort -n | tail -1)
        peak_disk_write=$(tail -n +2 "$resource_log" | cut -d',' -f6 | sort -n | tail -1)
    fi
    
    # Create summary JSON
    cat > "$summary_file" << EOF
{
    "suite": "$suite_name",
    "entity_count": $entity_count,
    "timestamp": "$(date -Iseconds)",
    "test_results": {
        "status": "$test_status",
        "total_tests": $total_tests,
        "passed_tests": $passed_tests,
        "failed_tests": $failed_tests
    },
    "resource_usage": {
        "max_memory_mb": ${max_memory:-0},
        "avg_cpu_percent": ${avg_cpu:-0},
        "peak_disk_read_mb": ${peak_disk_read:-0},
        "peak_disk_write_mb": ${peak_disk_write:-0}
    },
    "files": {
        "test_log": "$(basename "$test_log")",
        "resource_log": "$(basename "$resource_log")"
    }
}
EOF
}

# Generate comprehensive performance report
generate_performance_report() {
    local report_file="$OUTPUT_DIR/performance-test-report.json"
    
    log "Generating comprehensive performance report"
    
    # Collect all individual summaries
    local summaries=()
    for summary_file in "$OUTPUT_DIR"/*-summary.json; do
        if [[ -f "$summary_file" ]]; then
            summaries+=("$(cat "$summary_file")")
        fi
    done
    
    # Create comprehensive report
    cat > "$report_file" << EOF
{
    "scenario": "$SCENARIO",
    "timestamp": "$(date -Iseconds)",
    "system_info": {
        "hostname": "$(hostname)",
        "kernel": "$(uname -r)",
        "cpu_cores": $(nproc),
        "total_memory_mb": $(free -m | grep "^Mem:" | awk '{print $2}'),
        "disk_space_gb": $(df -BG / | tail -1 | awk '{print $2}' | sed 's/G//')
    },
    "test_configuration": {
        "timeout_seconds": $TEST_TIMEOUT,
        "monitoring_interval_seconds": $MONITORING_INTERVAL,
        "debug_mode": $DEBUG_MODE
    },
    "test_results": [
$(IFS=$'\n'; echo "${summaries[*]}" | sed 's/^/        /' | sed '$!s/$/,/')
    ]
}
EOF

    # Generate HTML report if Python is available
    if command -v python3 &> /dev/null; then
        python3 - << 'EOF'
import json
import sys
from datetime import datetime

try:
    with open('performance-test-report.json', 'r') as f:
        data = json.load(f)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLMKG Performance Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .test-result {{ margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ border-left: 5px solid #28a745; }}
        .failed {{ border-left: 5px solid #dc3545; }}
        .metric {{ display: inline-block; margin: 5px 10px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLMKG Performance Test Report</h1>
        <p><strong>Scenario:</strong> {data['scenario']}</p>
        <p><strong>Generated:</strong> {data['timestamp']}</p>
        <p><strong>System:</strong> {data['system_info']['hostname']} ({data['system_info']['cpu_cores']} cores, {data['system_info']['total_memory_mb']} MB RAM)</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <table>
            <tr><th>Suite</th><th>Entity Count</th><th>Status</th><th>Tests</th><th>Max Memory (MB)</th><th>Avg CPU (%)</th></tr>
"""
    
    for result in data['test_results']:
        status_class = 'passed' if result['test_results']['status'] == 'PASSED' else 'failed'
        html_content += f"""
            <tr class="{status_class}">
                <td>{result['suite']}</td>
                <td>{result['entity_count']}</td>
                <td>{result['test_results']['status']}</td>
                <td>{result['test_results']['passed_tests']}/{result['test_results']['total_tests']}</td>
                <td>{result['resource_usage']['max_memory_mb']:.1f}</td>
                <td>{result['resource_usage']['avg_cpu_percent']:.1f}</td>
            </tr>
"""
    
    html_content += """
        </table>
    </div>
</body>
</html>
"""
    
    with open('performance-test-report.html', 'w') as f:
        f.write(html_content)
    
    print("HTML report generated: performance-test-report.html")
    
except Exception as e:
    print(f"Error generating HTML report: {e}")
EOF
        
        if [[ -f "performance-test-report.html" ]]; then
            mv "performance-test-report.html" "$OUTPUT_DIR/"
            log "HTML performance report generated"
        fi
    fi
    
    log "Performance report saved to $report_file"
}

# Main execution
main() {
    log "Starting performance test runner"
    log "Scenario: $SCENARIO"
    log "Output directory: $OUTPUT_DIR"
    log "Test timeout: ${TEST_TIMEOUT}s"
    
    # Validate scenario
    if [[ ! "${TEST_CONFIGS[$SCENARIO]+isset}" ]]; then
        log_error "Unknown scenario: $SCENARIO"
        log "Available scenarios: ${!TEST_CONFIGS[*]}"
        exit 1
    fi
    
    local entity_counts="${TEST_CONFIGS[$SCENARIO]}"
    log "Entity counts for testing: $entity_counts"
    
    # Check system requirements
    local available_memory=$(free -m | grep "^Mem:" | awk '{print $7}')
    if [[ $available_memory -lt 1000 ]]; then
        log_error "Insufficient available memory: ${available_memory}MB (minimum: 1000MB)"
        exit 1
    fi
    
    # Run performance test suites based on scenario
    case "$SCENARIO" in
        "quick"|"comprehensive"|"stress")
            run_performance_test_suite "latency" "$entity_counts"
            run_performance_test_suite "memory" "$entity_counts"
            run_performance_test_suite "compression" "$entity_counts"
            
            if [[ "$SCENARIO" != "quick" ]]; then
                run_performance_test_suite "concurrency" "5000"
            fi
            
            if [[ "$SCENARIO" == "stress" ]]; then
                run_performance_test_suite "stress" "$entity_counts"
            fi
            ;;
        "memory")
            run_performance_test_suite "memory" "$entity_counts"
            ;;
        "concurrency")
            run_performance_test_suite "concurrency" "$entity_counts"
            ;;
        *)
            log_error "Unsupported scenario execution: $SCENARIO"
            exit 1
            ;;
    esac
    
    # Generate comprehensive report
    generate_performance_report
    
    # Check overall results
    local failed_tests=$(find "$OUTPUT_DIR" -name "*-summary.json" -exec grep -l '"status": "FAILED"' {} \; | wc -l)
    
    if [[ $failed_tests -eq 0 ]]; then
        log "✅ All performance tests passed"
        exit 0
    else
        log_error "❌ $failed_tests performance test suite(s) failed"
        exit 1
    fi
}

# Execute main function
main "$@"