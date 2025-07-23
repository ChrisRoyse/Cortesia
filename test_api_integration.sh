#!/bin/bash

echo "🧪 Testing LLMKG API Integration"
echo "================================"

# Start API server in background
echo "🚀 Starting API server..."
cargo run --bin llmkg_api_server &
API_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 5

# Test API discovery endpoint
echo "📡 Testing API discovery..."
curl -s http://localhost:3001/api/v1/discovery | jq .

# Test store triple endpoint
echo "📝 Testing store triple..."
curl -s -X POST http://localhost:3001/api/v1/triple \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Einstein",
    "predicate": "invented",
    "object": "Theory of Relativity",
    "confidence": 1.0
  }' | jq .

# Test query endpoint
echo "🔍 Testing query..."
curl -s -X POST http://localhost:3001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Einstein"
  }' | jq .

# Test metrics endpoint
echo "📊 Testing metrics..."
curl -s http://localhost:3001/api/v1/metrics | jq .

# Test monitoring metrics
echo "📈 Testing monitoring metrics..."
curl -s http://localhost:3001/api/v1/monitoring/metrics | jq .

# Kill the server
echo "🛑 Stopping server..."
kill $API_PID

echo "✅ Integration test complete!"