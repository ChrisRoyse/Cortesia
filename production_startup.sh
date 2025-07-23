#!/bin/bash

# LLMKG Production Startup Script
# Version: 1.0.0
# Description: Complete production deployment startup for LLMKG Dashboard System

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_LOG_FILE="logs/backend_$(date +%Y%m%d_%H%M%S).log"
FRONTEND_LOG_FILE="logs/frontend_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/llmkg_pids.txt"
HEALTH_CHECK_URL="http://localhost:8082/"
WEBSOCKET_URL="ws://localhost:8083"
FRONTEND_URL="http://localhost:3001"

# Create logs directory
mkdir -p logs

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}  LLMKG Production Startup Procedure     ${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Function to log with timestamp
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if port is available
check_port() {
    local port=$1
    if netstat -tuln | grep -q ":$port "; then
        echo -e "${RED}❌ Port $port is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $port is available${NC}"
        return 0
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    log "${YELLOW}Waiting for $name to be ready at $url...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ $name is ready${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ $name failed to start within timeout${NC}"
    return 1
}

# Function to check WebSocket connectivity
check_websocket() {
    local ws_url=$1
    log "${YELLOW}Testing WebSocket connection to $ws_url...${NC}"
    
    # Use Node.js to test WebSocket connection
    node -e "
        const WebSocket = require('ws');
        const ws = new WebSocket('$ws_url');
        
        ws.on('open', () => {
            console.log('✅ WebSocket connection successful');
            ws.close();
            process.exit(0);
        });
        
        ws.on('error', (error) => {
            console.log('❌ WebSocket connection failed:', error.message);
            process.exit(1);
        });
        
        setTimeout(() => {
            console.log('❌ WebSocket connection timeout');
            process.exit(1);
        }, 10000);
    " 2>/dev/null
}

# Function to cleanup on exit
cleanup() {
    log "${YELLOW}Cleaning up processes...${NC}"
    if [ -f "$PID_FILE" ]; then
        while read -r pid name; do
            if kill -0 "$pid" 2>/dev/null; then
                log "Stopping $name (PID: $pid)"
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
}

# Set up signal handlers
trap cleanup EXIT INT TERM

log "${BLUE}Starting LLMKG Production Deployment...${NC}"

# Step 1: Environment Validation
log "${YELLOW}Step 1: Validating environment...${NC}"

# Check required tools
for tool in cargo node npm curl netstat; do
    if ! command -v $tool &> /dev/null; then
        echo -e "${RED}❌ Required tool '$tool' is not installed${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ $tool is available${NC}"
    fi
done

# Check ports
log "Checking port availability..."
check_port 8082 || exit 1  # Backend HTTP
check_port 8083 || exit 1  # Backend WebSocket
check_port 3001 || exit 1  # Frontend

# Step 2: Backend Compilation and Startup
log "${YELLOW}Step 2: Building and starting backend server...${NC}"

# Build the backend
log "Building Rust backend..."
if ! cargo build --release --bin llmkg-brain-server > "$BACKEND_LOG_FILE" 2>&1; then
    echo -e "${RED}❌ Backend build failed. Check $BACKEND_LOG_FILE for details${NC}"
    tail -20 "$BACKEND_LOG_FILE"
    exit 1
fi
echo -e "${GREEN}✅ Backend build completed${NC}"

# Start the backend server
log "Starting LLMKG Brain Server..."
nohup ./target/release/llmkg-brain-server.exe >> "$BACKEND_LOG_FILE" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID backend_server" >> "$PID_FILE"

log "Backend server started with PID: $BACKEND_PID"
log "Backend logs: $BACKEND_LOG_FILE"

# Wait for backend to be ready
if ! wait_for_service "$HEALTH_CHECK_URL" "Backend HTTP Server"; then
    log "${RED}Backend failed to start properly${NC}"
    tail -20 "$BACKEND_LOG_FILE"
    exit 1
fi

# Test WebSocket connection
if ! check_websocket "$WEBSOCKET_URL"; then
    log "${RED}WebSocket server failed to start properly${NC}"
    exit 1
fi
echo -e "${GREEN}✅ WebSocket server is operational${NC}"

# Step 3: Frontend Setup and Startup
log "${YELLOW}Step 3: Setting up and starting frontend dashboard...${NC}"

cd visualization/dashboard

# Install dependencies if needed
if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules/.package-lock.json" ]; then
    log "Installing frontend dependencies..."
    if ! npm install >> "../../$FRONTEND_LOG_FILE" 2>&1; then
        echo -e "${RED}❌ Frontend dependency installation failed${NC}"
        cd ../..
        exit 1
    fi
    echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
fi

# Build the frontend
log "Building frontend application..."
if ! npm run build >> "../../$FRONTEND_LOG_FILE" 2>&1; then
    echo -e "${RED}❌ Frontend build failed. Check $FRONTEND_LOG_FILE for details${NC}"
    cd ../..
    tail -20 "$FRONTEND_LOG_FILE"
    exit 1
fi
echo -e "${GREEN}✅ Frontend build completed${NC}"

# Start the frontend server
log "Starting frontend server..."
nohup npm run preview -- --port 3001 >> "../../$FRONTEND_LOG_FILE" 2>&1 &
FRONTEND_PID=$!
cd ../..
echo "$FRONTEND_PID frontend_server" >> "$PID_FILE"

log "Frontend server started with PID: $FRONTEND_PID"
log "Frontend logs: $FRONTEND_LOG_FILE"

# Wait for frontend to be ready
if ! wait_for_service "$FRONTEND_URL" "Frontend Dashboard"; then
    log "${RED}Frontend failed to start properly${NC}"
    tail -20 "$FRONTEND_LOG_FILE"
    exit 1
fi

# Step 4: System Integration Validation
log "${YELLOW}Step 4: Validating system integration...${NC}"

log "Testing backend API endpoints..."
# Test system metrics endpoint
if curl -s -f "http://localhost:8082/api/system/metrics" >/dev/null; then
    echo -e "${GREEN}✅ System metrics endpoint is working${NC}"
else
    echo -e "${RED}❌ System metrics endpoint failed${NC}"
fi

# Test brain metrics endpoint
if curl -s -f "http://localhost:8082/api/brain/metrics" >/dev/null; then
    echo -e "${GREEN}✅ Brain metrics endpoint is working${NC}"
else
    echo -e "${RED}❌ Brain metrics endpoint failed${NC}"
fi

# Test real-time WebSocket data flow
log "Testing real-time data flow..."
node -e "
    const WebSocket = require('ws');
    const ws = new WebSocket('ws://localhost:8083');
    let messageCount = 0;
    
    ws.on('open', () => {
        console.log('Connected to WebSocket server');
    });
    
    ws.on('message', (data) => {
        messageCount++;
        if (messageCount >= 3) {
            console.log('✅ Real-time data flow is working');
            ws.close();
            process.exit(0);
        }
    });
    
    ws.on('error', (error) => {
        console.log('❌ WebSocket data flow test failed:', error.message);
        process.exit(1);
    });
    
    setTimeout(() => {
        console.log('❌ Real-time data flow test timeout');
        process.exit(1);
    }, 15000);
" 2>/dev/null

# Step 5: Production Readiness Summary
log "${GREEN}========================================${NC}"
log "${GREEN}  PRODUCTION STARTUP COMPLETED         ${NC}"
log "${GREEN}========================================${NC}"
echo ""
log "${GREEN}✅ All systems operational${NC}"
echo ""
log "📊 Dashboard URLs:"
log "   Frontend Dashboard: ${FRONTEND_URL}"
log "   Backend API:        http://localhost:8082"
log "   WebSocket Stream:   ws://localhost:8083"
echo ""
log "📁 Log Files:"
log "   Backend:  $BACKEND_LOG_FILE"
log "   Frontend: $FRONTEND_LOG_FILE"
echo ""
log "🔧 Process Information:"
while read -r pid name; do
    log "   $name: PID $pid"
done < "$PID_FILE"
echo ""
log "${BLUE}System is ready for production use!${NC}"
log "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Keep the script running to monitor services
while true; do
    sleep 30
    
    # Check if processes are still running
    while read -r pid name; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "${RED}❌ $name (PID: $pid) has stopped unexpectedly${NC}"
            exit 1
        fi
    done < "$PID_FILE"
    
    # Health check endpoints
    if ! curl -s -f "$HEALTH_CHECK_URL" >/dev/null; then
        log "${RED}❌ Backend health check failed${NC}"
        exit 1
    fi
    
    if ! curl -s -f "$FRONTEND_URL" >/dev/null; then
        log "${RED}❌ Frontend health check failed${NC}"
        exit 1
    fi
    
    log "${GREEN}✅ All services healthy${NC}"
done