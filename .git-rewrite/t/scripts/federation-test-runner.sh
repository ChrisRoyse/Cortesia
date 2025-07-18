#!/bin/bash

# Federation Test Runner Script
# Orchestrates execution of federated integration tests across multiple nodes

set -euo pipefail

# Configuration
FEDERATION_SIZE=${FEDERATION_SIZE:-3}
OUTPUT_DIR=${OUTPUT_DIR:-"/results"}
TEST_TIMEOUT=${TEST_TIMEOUT:-"1800"}  # 30 minutes
DEBUG_MODE=${DEBUG_MODE:-"false"}

# Test scenarios
SCENARIO=${SCENARIO:-"basic-federation"}

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
    log "Cleaning up federation test environment"
    
    # Stop all federation nodes
    if command -v docker &> /dev/null; then
        docker-compose -f federation-compose.yml down 2>/dev/null || true
        docker network rm llmkg-federation-net 2>/dev/null || true
    fi
    
    # Kill any remaining test processes
    pkill -f "federation-node" 2>/dev/null || true
    
    # Clean up temporary files
    rm -f federation-compose.yml node-*.toml
}

trap cleanup EXIT

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create federation configuration
create_federation_config() {
    local node_count=$1
    
    log "Creating federation configuration for $node_count nodes"
    
    # Create docker-compose file for federation
    cat > federation-compose.yml << EOF
version: '3.8'

networks:
  llmkg-federation:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
EOF

    # Add nodes to compose file
    for i in $(seq 0 $((node_count - 1))); do
        local node_ip="172.20.0.$((10 + i))"
        
        cat >> federation-compose.yml << EOF
  node-$i:
    image: llmkg-integration:federation-test
    container_name: federation-node-$i
    hostname: node-$i
    networks:
      llmkg-federation:
        ipv4_address: $node_ip
    environment:
      - NODE_ID=$i
      - FEDERATION_SIZE=$node_count
      - NODE_IP=$node_ip
      - RUST_LOG=info
      - LLMKG_TEST_ENV=federation
    volumes:
      - "$OUTPUT_DIR:/output"
      - "./node-$i.toml:/config/node.toml:ro"
    command: ["cargo", "test", "--test", "federation_integration", "--", "--test-threads=1"]
    depends_on:
      - coordinator
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3

EOF
    done
    
    # Add coordinator service
    cat >> federation-compose.yml << EOF
  coordinator:
    image: llmkg-integration:federation-test
    container_name: federation-coordinator
    hostname: coordinator
    networks:
      llmkg-federation:
        ipv4_address: 172.20.0.5
    environment:
      - ROLE=coordinator
      - FEDERATION_SIZE=$node_count
      - RUST_LOG=info
    ports:
      - "8080:8080"
    volumes:
      - "$OUTPUT_DIR:/output"
    command: ["cargo", "run", "--bin", "federation-coordinator"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 5
EOF

    # Create node configuration files
    for i in $(seq 0 $((node_count - 1))); do
        local node_ip="172.20.0.$((10 + i))"
        
        cat > "node-$i.toml" << EOF
[node]
id = $i
ip = "$node_ip"
port = 8080

[federation]
coordinator_ip = "172.20.0.5"
coordinator_port = 8080
size = $node_count

[storage]
data_dir = "/tmp/llmkg-node-$i"

[logging]
level = "info"
file = "/output/node-$i.log"
EOF
    done
}

# Wait for federation to be ready
wait_for_federation() {
    local max_wait=300  # 5 minutes
    local wait_time=0
    
    log "Waiting for federation to be ready..."
    
    while [[ $wait_time -lt $max_wait ]]; do
        if docker-compose -f federation-compose.yml ps | grep -q "healthy"; then
            local healthy_nodes=$(docker-compose -f federation-compose.yml ps | grep "healthy" | wc -l)
            if [[ $healthy_nodes -eq $((FEDERATION_SIZE + 1)) ]]; then
                log "Federation is ready with $healthy_nodes healthy services"
                return 0
            fi
        fi
        
        sleep 10
        wait_time=$((wait_time + 10))
        log_debug "Waiting for federation... (${wait_time}s elapsed)"
    done
    
    log_error "Federation failed to become ready within $max_wait seconds"
    return 1
}

# Run federation tests
run_federation_tests() {
    local scenario=$1
    
    log "Running federation tests for scenario: $scenario"
    
    case "$scenario" in
        "basic-federation")
            run_basic_federation_tests
            ;;
        "cross-shard-queries")
            run_cross_shard_tests
            ;;
        "fault-tolerance")
            run_fault_tolerance_tests
            ;;
        "load-balancing")
            run_load_balancing_tests
            ;;
        "all")
            run_basic_federation_tests
            run_cross_shard_tests
            run_fault_tolerance_tests
            run_load_balancing_tests
            ;;
        *)
            log_error "Unknown scenario: $scenario"
            return 1
            ;;
    esac
}

run_basic_federation_tests() {
    log "Running basic federation tests"
    
    # Test data distribution
    log "Testing data distribution across nodes"
    docker-compose -f federation-compose.yml exec -T coordinator \
        cargo test test_data_distribution -- --nocapture > "$OUTPUT_DIR/basic-federation.log" 2>&1
    
    # Test basic queries
    log "Testing basic federated queries"
    docker-compose -f federation-compose.yml exec -T coordinator \
        cargo test test_federated_search -- --nocapture >> "$OUTPUT_DIR/basic-federation.log" 2>&1
    
    # Collect results from all nodes
    for i in $(seq 0 $((FEDERATION_SIZE - 1))); do
        docker-compose -f federation-compose.yml exec -T "node-$i" \
            cp /output/node-results.json "/output/node-$i-results.json" 2>/dev/null || true
    done
}

run_cross_shard_tests() {
    log "Running cross-shard query tests"
    
    # Test cross-shard entity lookups
    docker-compose -f federation-compose.yml exec -T coordinator \
        cargo test test_cross_shard_lookup -- --nocapture > "$OUTPUT_DIR/cross-shard.log" 2>&1
    
    # Test cross-shard relationship traversal
    docker-compose -f federation-compose.yml exec -T coordinator \
        cargo test test_cross_shard_paths -- --nocapture >> "$OUTPUT_DIR/cross-shard.log" 2>&1
    
    # Test aggregated queries
    docker-compose -f federation-compose.yml exec -T coordinator \
        cargo test test_federated_aggregation -- --nocapture >> "$OUTPUT_DIR/cross-shard.log" 2>&1
}

run_fault_tolerance_tests() {
    log "Running fault tolerance tests"
    
    # Test with one node down
    log "Testing with node-0 down"
    docker-compose -f federation-compose.yml stop "node-0"
    
    sleep 10
    
    docker-compose -f federation-compose.yml exec -T coordinator \
        cargo test test_node_failure_resilience -- --nocapture > "$OUTPUT_DIR/fault-tolerance.log" 2>&1
    
    # Restart the node
    docker-compose -f federation-compose.yml start "node-0"
    
    # Wait for node to rejoin
    sleep 30
    
    # Test node rejoin
    docker-compose -f federation-compose.yml exec -T coordinator \
        cargo test test_node_rejoin -- --nocapture >> "$OUTPUT_DIR/fault-tolerance.log" 2>&1
}

run_load_balancing_tests() {
    log "Running load balancing tests"
    
    # Test concurrent requests
    docker-compose -f federation-compose.yml exec -T coordinator \
        cargo test test_load_balancing -- --nocapture > "$OUTPUT_DIR/load-balancing.log" 2>&1
    
    # Test hot spot handling
    docker-compose -f federation-compose.yml exec -T coordinator \
        cargo test test_hot_spot_distribution -- --nocapture >> "$OUTPUT_DIR/load-balancing.log" 2>&1
}

# Collect and analyze results
collect_results() {
    log "Collecting federation test results"
    
    # Collect logs from all containers
    for i in $(seq 0 $((FEDERATION_SIZE - 1))); do
        docker-compose -f federation-compose.yml logs "node-$i" > "$OUTPUT_DIR/node-$i.log" 2>&1 || true
    done
    
    docker-compose -f federation-compose.yml logs coordinator > "$OUTPUT_DIR/coordinator.log" 2>&1 || true
    
    # Generate federation test report
    cat > "$OUTPUT_DIR/federation-summary.json" << EOF
{
    "scenario": "$SCENARIO",
    "federation_size": $FEDERATION_SIZE,
    "timestamp": "$(date -Iseconds)",
    "test_files": [
        "basic-federation.log",
        "cross-shard.log",
        "fault-tolerance.log",
        "load-balancing.log"
    ],
    "node_logs": [
$(for i in $(seq 0 $((FEDERATION_SIZE - 1))); do
    echo "        \"node-$i.log\"$([ $i -lt $((FEDERATION_SIZE - 1)) ] && echo ",")"
done)
    ],
    "coordinator_log": "coordinator.log"
}
EOF

    # Analyze test results
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    for log_file in "$OUTPUT_DIR"/*.log; do
        if [[ -f "$log_file" ]]; then
            local tests=$(grep -c "test result:" "$log_file" 2>/dev/null || echo "0")
            local passed=$(grep "test result:" "$log_file" 2>/dev/null | grep -c "ok" || echo "0")
            local failed=$(grep "test result:" "$log_file" 2>/dev/null | grep -c "FAILED" || echo "0")
            
            total_tests=$((total_tests + tests))
            passed_tests=$((passed_tests + passed))
            failed_tests=$((failed_tests + failed))
        fi
    done
    
    # Update summary with results
    cat > "$OUTPUT_DIR/federation-results.json" << EOF
{
    "scenario": "$SCENARIO",
    "federation_size": $FEDERATION_SIZE,
    "timestamp": "$(date -Iseconds)",
    "results": {
        "total_tests": $total_tests,
        "passed_tests": $passed_tests,
        "failed_tests": $failed_tests,
        "success_rate": $(echo "scale=4; $passed_tests / $total_tests * 100" | bc -l 2>/dev/null || echo "0")
    },
    "status": "$([ $failed_tests -eq 0 ] && echo "PASSED" || echo "FAILED")"
}
EOF
    
    log "Federation test results summary:"
    log "  Total tests: $total_tests"
    log "  Passed: $passed_tests"
    log "  Failed: $failed_tests"
    log "  Success rate: $(echo "scale=1; $passed_tests / $total_tests * 100" | bc -l 2>/dev/null || echo "0")%"
}

# Main execution
main() {
    log "Starting federation test runner"
    log "Scenario: $SCENARIO"
    log "Federation size: $FEDERATION_SIZE"
    log "Output directory: $OUTPUT_DIR"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    # Create federation configuration
    create_federation_config "$FEDERATION_SIZE"
    
    # Start federation
    log "Starting federation with $FEDERATION_SIZE nodes"
    docker-compose -f federation-compose.yml up -d
    
    # Wait for federation to be ready
    if ! wait_for_federation; then
        log_error "Federation failed to start properly"
        docker-compose -f federation-compose.yml logs
        exit 1
    fi
    
    # Run tests
    if timeout "$TEST_TIMEOUT" bash -c "run_federation_tests '$SCENARIO'"; then
        log "Federation tests completed successfully"
        exit_code=0
    else
        log_error "Federation tests failed or timed out"
        exit_code=1
    fi
    
    # Collect results
    collect_results
    
    log "Federation test runner completed"
    exit $exit_code
}

# Execute main function
main "$@"