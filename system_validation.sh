#!/bin/bash

# LLMKG System Validation Script
# Version: 1.0.0
# Description: Complete validation of LLMKG dashboard system for production readiness

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKEND_URL="http://localhost:8082"
WEBSOCKET_URL="ws://localhost:8083"
FRONTEND_URL="http://localhost:3001"
TEST_LOG_FILE="logs/system_validation_$(date +%Y%m%d_%H%M%S).log"
VALIDATION_DURATION=300  # 5 minutes in seconds for initial validation

mkdir -p logs

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  LLMKG System Validation Suite     ${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Function to log with timestamp
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$message"
    echo "$message" >> "$TEST_LOG_FILE"
}

# Function to test WebSocket connectivity
test_websocket() {
    log "${YELLOW}Testing WebSocket connectivity...${NC}"
    
    # Check if ws module is available
    if ! node -e "require('ws')" 2>/dev/null; then
        log "${YELLOW}⚠️  WebSocket module not available, testing port connectivity instead${NC}"
        
        # Test if the WebSocket port is listening
        if netstat -ano | grep ":8083" | grep "LISTENING" >/dev/null; then
            log "${GREEN}✅ WebSocket port is listening${NC}"
            return 0
        else
            log "${RED}❌ WebSocket port not listening${NC}"
            return 1
        fi
    fi
    
    node -e "
        const WebSocket = require('ws');
        const ws = new WebSocket('$WEBSOCKET_URL');
        let messageReceived = false;
        
        ws.on('open', () => {
            console.log('✅ WebSocket connection established');
        });
        
        ws.on('message', (data) => {
            messageReceived = true;
            console.log('✅ Real-time data received');
            ws.close();
            process.exit(0);
        });
        
        ws.on('error', (error) => {
            console.log('❌ WebSocket error:', error.message);
            process.exit(1);
        });
        
        setTimeout(() => {
            if (!messageReceived) {
                console.log('❌ No data received within timeout');
                process.exit(1);
            }
        }, 10000);
    " 2>/dev/null
    
    return $?
}

# Function to test API endpoints
test_api_endpoints() {
    log "${YELLOW}Testing API endpoints...${NC}"
    
    local core_endpoints=("/" "/api/metrics" "/api/history")
    local optional_endpoints=("/mcp/health")
    local core_success=0
    local optional_success=0
    
    # Test core endpoints (must work)
    for endpoint in "${core_endpoints[@]}"; do
        if curl -s -f "$BACKEND_URL$endpoint" >/dev/null 2>&1; then
            log "${GREEN}✅ $endpoint endpoint working${NC}"
            ((core_success++))
        else
            log "${RED}❌ $endpoint endpoint failed${NC}"
        fi
    done
    
    # Test optional endpoints (nice to have)
    for endpoint in "${optional_endpoints[@]}"; do
        if curl -s -f "$BACKEND_URL$endpoint" >/dev/null 2>&1; then
            log "${GREEN}✅ $endpoint endpoint working${NC}"
            ((optional_success++))
        else
            log "${YELLOW}⚠️  $endpoint endpoint unavailable (optional)${NC}"
        fi
    done
    
    if [ $core_success -eq ${#core_endpoints[@]} ]; then
        log "${GREEN}✅ All core API endpoints working (${optional_success}/${#optional_endpoints[@]} optional endpoints)${NC}"
        return 0
    else
        log "${RED}❌ Core API endpoints failed ($core_success/${#core_endpoints[@]} working)${NC}"
        return 1
    fi
}

# Function to run stability test
run_stability_test() {
    log "${YELLOW}Running $((VALIDATION_DURATION/60))-minute stability test...${NC}"
    
    local end_time=$(($(date +%s) + VALIDATION_DURATION))
    local check_count=0
    local failure_count=0
    
    while [ $(date +%s) -lt $end_time ]; do
        ((check_count++))
        
        # Test HTTP endpoint
        if ! curl -s -f "$BACKEND_URL/" >/dev/null 2>&1; then
            ((failure_count++))
            log "${RED}❌ HTTP health check failed (check $check_count)${NC}"
        fi
        
        # Test WebSocket every 5th check
        if [ $((check_count % 5)) -eq 0 ]; then
            if ! test_websocket >/dev/null 2>&1; then
                ((failure_count++))
                log "${RED}❌ WebSocket test failed (check $check_count)${NC}"
            fi
        fi
        
        # Progress indicator
        local remaining=$((end_time - $(date +%s)))
        local minutes=$((remaining / 60))
        local seconds=$((remaining % 60))
        log "Stability test progress: ${minutes}m ${seconds}s remaining (checks: $check_count, failures: $failure_count)"
        
        sleep 30
    done
    
    local success_rate=$(( (check_count - failure_count) * 100 / check_count ))
    log "${GREEN}Stability test completed: $success_rate% success rate ($check_count checks, $failure_count failures)${NC}"
    
    if [ $success_rate -ge 95 ]; then
        return 0
    else
        return 1
    fi
}

# Function to monitor resource usage
monitor_resources() {
    log "${YELLOW}Monitoring resource usage...${NC}"
    
    local resource_log="logs/resource_monitor_$(date +%Y%m%d_%H%M%S).log"
    echo "timestamp,cpu_percent,memory_mb" > "$resource_log"
    
    # Monitor for 5 minutes
    for i in {1..10}; do
        # Get CPU and memory usage of LLMKG process
        local pid=$(pgrep -f "llmkg-brain-server" || echo "0")
        
        if [ "$pid" != "0" ]; then
            # Use wmic on Windows to get process info
            local cpu_mem=$(wmic process where "ProcessId=$pid" get "PageFileUsage,WorkingSetSize" /format:csv 2>/dev/null | grep -E "^[^,]*,[0-9]" | head -1)
            if [ -n "$cpu_mem" ]; then
                local memory_kb=$(echo "$cpu_mem" | cut -d',' -f3)
                local memory_mb=$((memory_kb / 1024))
                echo "$(date '+%Y-%m-%d %H:%M:%S'),0,$memory_mb" >> "$resource_log"
                log "Resource usage: Memory: ${memory_mb}MB"
            fi
        fi
        
        sleep 30
    done
    
    # Analyze results
    local avg_memory=$(awk -F',' 'NR>1 && $3>0 {sum+=$3; count++} END {if(count>0) print int(sum/count); else print 0}' "$resource_log")
    
    log "${GREEN}Average memory usage: ${avg_memory}MB${NC}"
    
    if [ "$avg_memory" -lt 500 ]; then
        log "${GREEN}✅ Memory usage within acceptable limits${NC}"
        return 0
    else
        log "${YELLOW}⚠️  High memory usage detected${NC}"
        return 1
    fi
}

# Main validation sequence
main() {
    log "${BLUE}Starting LLMKG System Validation...${NC}"
    
    # Step 1: Check if backend is running
    if ! curl -s -f "$BACKEND_URL/" >/dev/null 2>&1; then
        log "${RED}❌ Backend is not running. Please start with ./production_startup.sh first${NC}"
        exit 1
    fi
    
    log "${GREEN}✅ Backend is running${NC}"
    
    # Step 2: Test all endpoints
    if ! test_api_endpoints; then
        log "${RED}❌ API endpoint tests failed${NC}"
        exit 1
    fi
    
    # Step 3: Test WebSocket connectivity
    if ! test_websocket; then
        log "${RED}❌ WebSocket connectivity test failed${NC}"
        exit 1
    fi
    
    # Step 4: Monitor resources in background
    monitor_resources &
    MONITOR_PID=$!
    
    # Step 5: Run stability test
    if ! run_stability_test; then
        log "${RED}❌ Stability test failed${NC}"
        kill $MONITOR_PID 2>/dev/null || true
        exit 1
    fi
    
    # Wait for resource monitoring to complete
    wait $MONITOR_PID
    
    # Final validation report
    echo ""
    log "${BLUE}======================================${NC}"
    log "${BLUE}  VALIDATION RESULTS                 ${NC}"
    log "${BLUE}======================================${NC}"
    log "${GREEN}✅ API Endpoints: PASSED${NC}"
    log "${GREEN}✅ WebSocket Connectivity: PASSED${NC}"
    log "${GREEN}✅ Extended Stability Test: PASSED${NC}"
    log "${GREEN}✅ Resource Usage Monitoring: PASSED${NC}"
    echo ""
    log "${GREEN}🎉 SYSTEM IS PRODUCTION READY${NC}"
    log "${GREEN}All validation tests passed successfully${NC}"
    echo ""
    log "📊 Dashboard URLs:"
    log "   Backend API:        $BACKEND_URL"
    log "   WebSocket Stream:   $WEBSOCKET_URL"
    log "   Frontend Dashboard: $FRONTEND_URL"
    echo ""
    log "📁 Validation Log: $TEST_LOG_FILE"
    
    return 0
}

# Run the validation
main "$@"