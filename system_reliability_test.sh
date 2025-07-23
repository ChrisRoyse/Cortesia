#!/bin/bash

# LLMKG System Reliability Testing Suite
# Version: 1.0.0
# Description: Comprehensive testing for production reliability and performance

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
TEST_DURATION_MINUTES=10
LOAD_TEST_CONNECTIONS=10
RESOURCE_MONITOR_INTERVAL=30
WEBSOCKET_RECONNECT_TESTS=5
BACKEND_URL="http://localhost:8082"
WEBSOCKET_URL="ws://localhost:8083"
FRONTEND_URL="http://localhost:3001"
TEST_LOG_FILE="logs/reliability_test_$(date +%Y%m%d_%H%M%S).log"

mkdir -p logs

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  LLMKG System Reliability Testing Suite      ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Function to log with timestamp
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$message"
    echo "$message" >> "$TEST_LOG_FILE"
}

# Function to run background resource monitoring
start_resource_monitoring() {
    log "${YELLOW}Starting resource monitoring...${NC}"
    
    # Create resource monitoring script
    cat > logs/resource_monitor.sh << 'EOF'
#!/bin/bash
MONITOR_LOG="logs/resource_usage_$(date +%Y%m%d_%H%M%S).log"
echo "timestamp,cpu_percent,memory_mb,network_rx_mb,network_tx_mb,disk_io_read,disk_io_write" > "$MONITOR_LOG"

while true; do
    # Get CPU usage
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    
    # Get memory usage in MB
    MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $3}')
    
    # Get network stats (simplified)
    NETWORK_RX=$(cat /proc/net/dev | grep -E "(eth0|wlan0|enp)" | head -1 | awk '{print $2}' | awk '{printf "%.2f", $1/1048576}')
    NETWORK_TX=$(cat /proc/net/dev | grep -E "(eth0|wlan0|enp)" | head -1 | awk '{print $10}' | awk '{printf "%.2f", $1/1048576}')
    
    # Get disk I/O (simplified)
    DISK_READ=$(iostat -d 1 1 2>/dev/null | tail -1 | awk '{print $3}' || echo "0")
    DISK_WRITE=$(iostat -d 1 1 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")
    
    # Fallback if tools aren't available
    [ -z "$CPU" ] && CPU="0"
    [ -z "$MEMORY" ] && MEMORY="0"
    [ -z "$NETWORK_RX" ] && NETWORK_RX="0"
    [ -z "$NETWORK_TX" ] && NETWORK_TX="0"
    [ -z "$DISK_READ" ] && DISK_READ="0"
    [ -z "$DISK_WRITE" ] && DISK_WRITE="0"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$CPU,$MEMORY,$NETWORK_RX,$NETWORK_TX,$DISK_READ,$DISK_WRITE" >> "$MONITOR_LOG"
    sleep 30
done
EOF
    
    chmod +x logs/resource_monitor.sh
    nohup ./logs/resource_monitor.sh > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo "$MONITOR_PID" > logs/monitor_pid.txt
    
    log "${GREEN}✅ Resource monitoring started (PID: $MONITOR_PID)${NC}"
}

# Function to stop resource monitoring
stop_resource_monitoring() {
    if [ -f logs/monitor_pid.txt ]; then
        MONITOR_PID=$(cat logs/monitor_pid.txt)
        kill "$MONITOR_PID" 2>/dev/null || true
        rm -f logs/monitor_pid.txt
        log "${GREEN}Resource monitoring stopped${NC}"
    fi
}

# Function to test WebSocket reconnection
test_websocket_reconnection() {
    log "${YELLOW}Testing WebSocket reconnection capability...${NC}"
    
    for i in $(seq 1 $WEBSOCKET_RECONNECT_TESTS); do
        log "WebSocket reconnection test $i/$WEBSOCKET_RECONNECT_TESTS"
        
        # Test script for WebSocket reconnection
        node -e "
            const WebSocket = require('ws');
            let ws = new WebSocket('$WEBSOCKET_URL');
            let reconnectCount = 0;
            let messageReceived = false;
            
            function connect() {
                ws = new WebSocket('$WEBSOCKET_URL');
                
                ws.on('open', () => {
                    console.log('Connected (attempt ' + (reconnectCount + 1) + ')');
                });
                
                ws.on('message', (data) => {
                    messageReceived = true;
                    console.log('Data received after reconnection');
                });
                
                ws.on('close', () => {
                    console.log('Connection closed');
                    if (reconnectCount < 2) {
                        reconnectCount++;
                        setTimeout(connect, 1000);
                    }
                });
                
                ws.on('error', (error) => {
                    console.error('WebSocket error:', error.message);
                });
            }
            
            connect();
            
            // Force disconnect after 3 seconds
            setTimeout(() => {
                ws.close();
            }, 3000);
            
            // Check if reconnection worked
            setTimeout(() => {
                if (messageReceived) {
                    console.log('✅ WebSocket reconnection test passed');
                    process.exit(0);
                } else {
                    console.log('❌ WebSocket reconnection test failed');
                    process.exit(1);
                }
            }, 10000);
        " 2>/dev/null
        
        if [ $? -eq 0 ]; then
            log "${GREEN}✅ WebSocket reconnection test $i passed${NC}"
        else
            log "${RED}❌ WebSocket reconnection test $i failed${NC}"
            return 1
        fi
        
        sleep 2
    done
    
    log "${GREEN}✅ All WebSocket reconnection tests passed${NC}"
    return 0
}

# Function to run load testing
run_load_test() {
    log "${YELLOW}Running load test with $LOAD_TEST_CONNECTIONS concurrent connections...${NC}"
    
    # Create load test script
    cat > logs/load_test.js << EOF
const WebSocket = require('ws');
const http = require('http');

const BACKEND_URL = '$BACKEND_URL';
const WEBSOCKET_URL = '$WEBSOCKET_URL'; 
const CONNECTIONS = $LOAD_TEST_CONNECTIONS;
const TEST_DURATION = 120000; // 2 minutes

let connections = [];
let totalMessages = 0;
let totalErrors = 0;
let httpRequests = 0;
let httpErrors = 0;

// Function to make HTTP requests
function makeHttpRequest() {
    const endpoints = ['/api/system/metrics', '/api/brain/metrics', '/health'];
    const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
    
    const options = {
        hostname: 'localhost',
        port: 8080,
        path: endpoint,
        method: 'GET'
    };
    
    const req = http.request(options, (res) => {
        httpRequests++;
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
            if (res.statusCode !== 200) httpErrors++;
        });
    });
    
    req.on('error', () => httpErrors++);
    req.end();
}

// Create WebSocket connections
for (let i = 0; i < CONNECTIONS; i++) {
    setTimeout(() => {
        const ws = new WebSocket(WEBSOCKET_URL);
        
        ws.on('open', () => {
            console.log(\`Connection \${i + 1} established\`);
        });
        
        ws.on('message', (data) => {
            totalMessages++;
        });
        
        ws.on('error', (error) => {
            totalErrors++;
            console.error(\`Connection \${i + 1} error:\`, error.message);
        });
        
        ws.on('close', () => {
            console.log(\`Connection \${i + 1} closed\`);
        });
        
        connections.push(ws);
    }, i * 100); // Stagger connections
}

// Make periodic HTTP requests
const httpInterval = setInterval(() => {
    for (let i = 0; i < 5; i++) {
        makeHttpRequest();
    }
}, 1000);

// Report results after test duration
setTimeout(() => {
    clearInterval(httpInterval);
    
    console.log('=== Load Test Results ===');
    console.log(\`Total WebSocket messages received: \${totalMessages}\`);
    console.log(\`WebSocket errors: \${totalErrors}\`);
    console.log(\`HTTP requests made: \${httpRequests}\`);
    console.log(\`HTTP errors: \${httpErrors}\`);
    console.log(\`Success rate: \${((totalMessages > 0 && totalErrors < CONNECTIONS/2) ? 'PASS' : 'FAIL')}\`);
    
    // Close all connections
    connections.forEach(ws => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    });
    
    process.exit(totalErrors < CONNECTIONS/2 ? 0 : 1);
}, TEST_DURATION);
EOF
    
    if node logs/load_test.js 2>&1 | tee -a "$TEST_LOG_FILE"; then
        log "${GREEN}✅ Load test completed successfully${NC}"
        return 0
    else
        log "${RED}❌ Load test failed${NC}"
        return 1
    fi
}

# Function to test error recovery scenarios
test_error_recovery() {
    log "${YELLOW}Testing error recovery scenarios...${NC}"
    
    # Test 1: Invalid API endpoint
    log "Testing invalid API endpoint handling..."
    if curl -s -f "$BACKEND_URL/api/invalid/endpoint" >/dev/null 2>&1; then
        log "${RED}❌ Invalid endpoint should return error${NC}"
        return 1
    else
        log "${GREEN}✅ Invalid endpoint properly rejected${NC}"
    fi
    
    # Test 2: Malformed WebSocket connection
    log "Testing malformed WebSocket connection..."
    if ! curl -s "$WEBSOCKET_URL" >/dev/null 2>&1; then
        log "${GREEN}✅ WebSocket properly rejects HTTP connections${NC}"
    else
        log "${RED}❌ WebSocket should reject HTTP connections${NC}"
        return 1
    fi
    
    # Test 3: High frequency requests
    log "Testing rate limiting/high frequency request handling..."
    for i in {1..50}; do
        curl -s "$BACKEND_URL/health" >/dev/null &
    done
    wait
    
    # Check if system is still responsive
    if curl -s -f "$BACKEND_URL/health" >/dev/null; then
        log "${GREEN}✅ System remains responsive under high request load${NC}"
    else
        log "${RED}❌ System became unresponsive under high request load${NC}"
        return 1
    fi
    
    log "${GREEN}✅ All error recovery tests passed${NC}"
    return 0
}

# Function to analyze system performance
analyze_performance() {
    log "${YELLOW}Analyzing system performance...${NC}"
    
    # Find the most recent resource log
    RESOURCE_LOG=$(ls -t logs/resource_usage_*.log 2>/dev/null | head -1)
    
    if [ -z "$RESOURCE_LOG" ]; then
        log "${RED}❌ No resource usage data found${NC}"
        return 1
    fi
    
    log "Analyzing resource usage data from: $RESOURCE_LOG"
    
    # Use awk to analyze the CSV data
    awk -F',' '
    NR==1 {next}  # Skip header
    {
        if ($2 != "" && $2 != "0") cpu_sum += $2; cpu_count++
        if ($3 != "" && $3 != "0") mem_sum += $3; mem_count++
    }
    END {
        if (cpu_count > 0) avg_cpu = cpu_sum / cpu_count
        if (mem_count > 0) avg_mem = mem_sum / mem_count
        
        printf "Average CPU Usage: %.2f%%\n", avg_cpu
        printf "Average Memory Usage: %.0f MB\n", avg_mem
        
        if (avg_cpu > 80) {
            print "⚠️  High CPU usage detected"
            exit 1
        }
        if (avg_mem > 1024) {
            print "⚠️  High memory usage detected (>1GB)"
            exit 1
        }
        
        print "✅ Resource usage within acceptable limits"
    }' "$RESOURCE_LOG"
    
    return $?
}

# Main testing sequence
main() {
    log "${BLUE}Starting comprehensive reliability testing...${NC}"
    log "Test duration: $TEST_DURATION_MINUTES minutes"
    log "Load test connections: $LOAD_TEST_CONNECTIONS"
    log "WebSocket reconnection tests: $WEBSOCKET_RECONNECT_TESTS"
    echo ""
    
    # Check if services are running
    if ! curl -s -f "$BACKEND_URL/health" >/dev/null; then
        log "${RED}❌ Backend service is not running. Start with ./production_startup.sh first${NC}"
        exit 1
    fi
    
    if ! curl -s -f "$FRONTEND_URL" >/dev/null; then
        log "${RED}❌ Frontend service is not running. Start with ./production_startup.sh first${NC}"
        exit 1
    fi
    
    log "${GREEN}✅ All services are running${NC}"
    
    # Start background monitoring
    start_resource_monitoring
    
    # Set up cleanup on exit
    trap 'stop_resource_monitoring; log "Test interrupted"; exit 1' INT TERM
    
    # Run tests
    local test_failures=0
    
    log "${YELLOW}=== Phase 1: WebSocket Reconnection Testing ===${NC}"
    if ! test_websocket_reconnection; then
        ((test_failures++))
    fi
    
    log "${YELLOW}=== Phase 2: Error Recovery Testing ===${NC}"
    if ! test_error_recovery; then
        ((test_failures++))
    fi
    
    log "${YELLOW}=== Phase 3: Load Testing ===${NC}"
    if ! run_load_test; then
        ((test_failures++))
    fi
    
    log "${YELLOW}=== Phase 4: Extended Stability Test ===${NC}"
    log "Running extended stability test for $TEST_DURATION_MINUTES minutes..."
    
    local end_time=$(($(date +%s) + TEST_DURATION_MINUTES * 60))
    local health_check_failures=0
    
    while [ $(date +%s) -lt $end_time ]; do
        # Check service health
        if ! curl -s -f "$BACKEND_URL/health" >/dev/null; then
            ((health_check_failures++))
            log "${RED}❌ Backend health check failed${NC}"
        fi
        
        if ! curl -s -f "$FRONTEND_URL" >/dev/null; then
            ((health_check_failures++))
            log "${RED}❌ Frontend health check failed${NC}"
        fi
        
        if [ $health_check_failures -gt 3 ]; then
            log "${RED}❌ Too many health check failures during stability test${NC}"
            ((test_failures++))
            break
        fi
        
        # Test WebSocket connectivity
        if ! node -e "
            const WebSocket = require('ws');
            const ws = new WebSocket('$WEBSOCKET_URL');
            ws.on('open', () => { ws.close(); process.exit(0); });
            ws.on('error', () => process.exit(1));
            setTimeout(() => process.exit(1), 5000);
        " 2>/dev/null; then
            log "${RED}❌ WebSocket connectivity test failed${NC}"
            ((test_failures++))
            break
        fi
        
        # Progress indicator
        local remaining=$((end_time - $(date +%s)))
        log "Stability test in progress... ${remaining}s remaining"
        
        sleep 60
    done
    
    if [ $health_check_failures -le 3 ]; then
        log "${GREEN}✅ Extended stability test completed successfully${NC}"
    fi
    
    # Stop monitoring and analyze results
    stop_resource_monitoring
    sleep 2  # Allow monitor to finish writing
    
    log "${YELLOW}=== Phase 5: Performance Analysis ===${NC}"
    if ! analyze_performance; then
        ((test_failures++))
    fi
    
    # Final report
    echo ""
    log "${BLUE}=================================================${NC}"
    log "${BLUE}  RELIABILITY TEST RESULTS                     ${NC}"
    log "${BLUE}=================================================${NC}"
    
    if [ $test_failures -eq 0 ]; then
        log "${GREEN}🎉 ALL TESTS PASSED - SYSTEM IS PRODUCTION READY${NC}"
        log "${GREEN}✅ WebSocket reconnection: PASSED${NC}"
        log "${GREEN}✅ Error recovery: PASSED${NC}"  
        log "${GREEN}✅ Load testing: PASSED${NC}"
        log "${GREEN}✅ Extended stability: PASSED${NC}"
        log "${GREEN}✅ Performance analysis: PASSED${NC}"
        echo ""
        log "${GREEN}System demonstrated reliable operation for $TEST_DURATION_MINUTES minutes${NC}"
        log "${GREEN}Ready for production deployment${NC}"
        return 0
    else
        log "${RED}❌ $test_failures TEST(S) FAILED - SYSTEM NOT READY FOR PRODUCTION${NC}"
        log "${RED}Please review the test logs and address the issues before deployment${NC}"
        return 1
    fi
}

# Run the main testing sequence
main "$@"