const http = require('http');
const WebSocket = require('ws');

const BACKEND_URL = 'http://localhost:8082';
const WEBSOCKET_URL = 'ws://localhost:8083';
const CONNECTIONS = 5;
const TEST_DURATION = 60000; // 1 minute

let connections = [];
let totalMessages = 0;
let totalErrors = 0;
let httpRequests = 0;
let httpErrors = 0;

console.log(`🚀 Starting load test with ${CONNECTIONS} connections for ${TEST_DURATION/1000} seconds...`);

// Function to make HTTP requests
function makeHttpRequest() {
    const endpoints = ['/api/metrics', '/api/history', '/'];
    const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
    
    const options = {
        hostname: 'localhost',
        port: 8082,
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
            console.log(`✅ Connection ${i + 1} established`);
        });
        
        ws.on('message', (data) => {
            totalMessages++;
        });
        
        ws.on('error', (error) => {
            totalErrors++;
            console.error(`❌ Connection ${i + 1} error:`, error.message);
        });
        
        ws.on('close', () => {
            console.log(`Connection ${i + 1} closed`);
        });
        
        connections.push(ws);
    }, i * 200); // Stagger connections
}

// Make periodic HTTP requests
const httpInterval = setInterval(() => {
    for (let i = 0; i < 3; i++) {
        makeHttpRequest();
    }
}, 2000);

// Report results after test duration
setTimeout(() => {
    clearInterval(httpInterval);
    
    console.log('\n=== Load Test Results ===');
    console.log(`Total WebSocket messages received: ${totalMessages}`);
    console.log(`WebSocket errors: ${totalErrors}`);
    console.log(`HTTP requests made: ${httpRequests}`);
    console.log(`HTTP errors: ${httpErrors}`);
    
    const wsSuccessRate = totalMessages > 0 && totalErrors < CONNECTIONS/2 ? 'PASS' : 'FAIL';
    const httpSuccessRate = httpRequests > 0 && httpErrors < httpRequests/2 ? 'PASS' : 'FAIL';
    
    console.log(`WebSocket test: ${wsSuccessRate}`);
    console.log(`HTTP test: ${httpSuccessRate}`);
    console.log(`Overall result: ${wsSuccessRate === 'PASS' && httpSuccessRate === 'PASS' ? '✅ PASS' : '❌ FAIL'}`);
    
    // Close all connections
    connections.forEach(ws => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    });
    
    process.exit(wsSuccessRate === 'PASS' && httpSuccessRate === 'PASS' ? 0 : 1);
}, TEST_DURATION);