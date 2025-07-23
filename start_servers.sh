#!/bin/bash

echo "Starting LLMKG Servers..."
echo "========================="

echo "[1/2] Building the API server..."
cargo build --bin llmkg_api_server --release

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build API server"
    exit 1
fi

echo "[2/2] Starting the API server (includes WebSocket on port 8081)..."
cargo run --bin llmkg_api_server --release &

API_PID=$!
echo "API Server PID: $API_PID"

echo ""
echo "✅ LLMKG API Server starting up..."
echo ""
echo "Services will be available at:"
echo "  - API endpoints: http://localhost:3001/api/v1"
echo "  - Dashboard: http://localhost:8080"
echo "  - WebSocket: ws://localhost:8081"
echo "  - API Discovery: http://localhost:3001/api/v1/discovery"
echo ""
echo "Press Ctrl+C to stop the server..."

# Wait for interrupt
trap "kill $API_PID; exit" INT
wait