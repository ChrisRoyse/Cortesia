#!/bin/bash

echo "🧠 Testing LLMKG Brain Metrics Integration"
echo "=========================================="
echo ""

# Check if the server is running
echo "1. Checking if LLMKG brain server is running..."
if lsof -i:8083 > /dev/null 2>&1; then
    echo "✅ WebSocket server is running on port 8083"
else
    echo "❌ WebSocket server is not running on port 8083"
    echo "   Please start the server with: cargo run --bin llmkg_brain_server"
    exit 1
fi

echo ""
echo "2. Starting brain metrics debug monitor..."
echo "   This will show real-time brain metrics from the WebSocket"
echo ""

# Run the debug script
node debug_brain_metrics.js