#!/bin/bash

echo "🧪 Verifying Test Backend Integration"
echo "======================================"
echo ""

# Check if the dashboard server is running
echo "1. Checking dashboard server on port 8082..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8082 | grep -q "200"; then
    echo "✅ Dashboard server is running"
else
    echo "❌ Dashboard server is not running on port 8082"
    echo "   Please start the server with: cargo run --bin llmkg_brain_server"
    exit 1
fi

# Check WebSocket connectivity
echo ""
echo "2. Checking WebSocket server on port 8083..."
if nc -z localhost 8083 2>/dev/null; then
    echo "✅ WebSocket server is listening"
else
    echo "❌ WebSocket server is not listening on port 8083"
    exit 1
fi

# Test the discover endpoint
echo ""
echo "3. Testing test discovery endpoint..."
DISCOVER_RESPONSE=$(curl -s http://localhost:8082/api/tests/discover)
if [ $? -eq 0 ]; then
    echo "✅ Test discovery endpoint is working"
    echo "   Response: $(echo $DISCOVER_RESPONSE | jq -r '.total_suites') test suites found"
else
    echo "❌ Test discovery endpoint failed"
    exit 1
fi

# Test the execute endpoint
echo ""
echo "4. Testing test execution endpoint..."
EXECUTE_RESPONSE=$(curl -s -X POST http://localhost:8082/api/tests/execute \
    -H "Content-Type: application/json" \
    -d '{"suite_name": "core::graph", "filter": null, "nocapture": false, "parallel": true}')
if [ $? -eq 0 ]; then
    EXECUTION_ID=$(echo $EXECUTE_RESPONSE | jq -r '.execution_id')
    echo "✅ Test execution endpoint is working"
    echo "   Execution ID: $EXECUTION_ID"
else
    echo "❌ Test execution endpoint failed"
    exit 1
fi

# Test the status endpoint
echo ""
echo "5. Testing test status endpoint..."
sleep 2
STATUS_RESPONSE=$(curl -s http://localhost:8082/api/tests/status/$EXECUTION_ID)
if [ $? -eq 0 ]; then
    echo "✅ Test status endpoint is working"
    echo "   Status: $(echo $STATUS_RESPONSE | jq -r '.status')"
else
    echo "❌ Test status endpoint failed"
fi

echo ""
echo "======================================"
echo "Backend integration verification complete!"
echo ""
echo "Next steps:"
echo "1. Open the React dashboard at http://localhost:3001"
echo "2. Navigate to the API Testing page"
echo "3. Click on the 'Test Suites' tab"
echo "4. Click 'Run Tests' on any suite"
echo "5. Watch the real-time test output in the console"