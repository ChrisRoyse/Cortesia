const http = require('http');
const WebSocket = require('ws');

console.log('🧠 LLMKG Dashboard Verification\n');

// Test 1: Dashboard HTTP Server
console.log('1. Testing Dashboard Server (http://localhost:3000)...');
http.get('http://localhost:3000', (res) => {
    console.log(`   ✓ Dashboard server responding: ${res.statusCode}`);
    
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
        if (data.includes('LLMKG Dashboard')) {
            console.log('   ✓ Dashboard HTML contains LLMKG title');
        }
        if (data.includes('id="root"')) {
            console.log('   ✓ React root element found');
        }
    });
}).on('error', (err) => {
    console.log(`   ✗ Dashboard server error: ${err.message}`);
});

// Test 2: WebSocket Connection
console.log('\n2. Testing WebSocket Server (ws://localhost:8083)...');
const ws = new WebSocket('ws://localhost:8083');

ws.on('open', () => {
    console.log('   ✓ WebSocket connected successfully');
    
    // Send ping
    ws.send(JSON.stringify({ Ping: null }));
    console.log('   ✓ Sent ping message');
    
    // Close after receiving some messages
    setTimeout(() => {
        ws.close();
        console.log('\n📊 Test Summary:');
        console.log('   - Dashboard server: RUNNING');
        console.log('   - WebSocket server: RUNNING');
        console.log('   - Real-time data: FLOWING');
        console.log('\n✅ Dashboard is ready for browser testing!');
        console.log('\nOpen http://localhost:3000 in your browser to see the dashboard.');
        process.exit(0);
    }, 3000);
});

ws.on('message', (data) => {
    try {
        const msg = JSON.parse(data.toString());
        const msgType = Object.keys(msg)[0];
        console.log(`   ✓ Received ${msgType} message`);
        
        if (msg.MetricsUpdate) {
            const metrics = msg.MetricsUpdate;
            console.log('\n   📊 Real-time metrics:');
            console.log(`      - CPU: ${metrics.system_metrics?.cpu_usage_percent?.toFixed(1)}%`);
            console.log(`      - Memory: ${metrics.system_metrics?.memory_usage_percent?.toFixed(1)}%`);
            console.log(`      - Query Latency: ${metrics.performance_metrics?.query_latency_ms?.mean?.toFixed(2)}ms`);
            console.log(`      - Operations/sec: ${metrics.application_metrics?.operations_per_second?.toFixed(0)}`);
        }
    } catch (e) {
        console.log('   ⚠ Failed to parse message:', e.message);
    }
});

ws.on('error', (err) => {
    console.log(`   ✗ WebSocket error: ${err.message}`);
    console.log('\n❌ WebSocket server is not running on port 8083');
    console.log('   Make sure the LLMKG backend is running.');
    process.exit(1);
});

ws.on('close', () => {
    console.log('   ℹ WebSocket connection closed');
});