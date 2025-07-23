#!/usr/bin/env node

/**
 * Test Dashboard Integration Script
 * Verifies that the test execution backend is properly integrated
 */

const http = require('http');
const WebSocket = require('ws');

const API_BASE = 'http://localhost:8082';
const WS_URL = 'ws://localhost:8083';

console.log('🧪 Testing Dashboard Integration...\n');

// Test 1: Discover available test suites
async function testDiscoverSuites() {
    console.log('1️⃣ Testing test discovery endpoint...');
    try {
        const response = await fetch(`${API_BASE}/api/tests/discover`);
        const data = await response.json();
        console.log('✅ Test discovery successful!');
        console.log(`   Found ${data.total_suites} test suites with ${data.total_tests} total tests`);
        console.log('   Suites:', data.suites.map(s => s.name).join(', '));
        return data.suites;
    } catch (error) {
        console.error('❌ Test discovery failed:', error.message);
        return [];
    }
}

// Test 2: Connect to WebSocket and listen for test events
async function testWebSocketConnection() {
    console.log('\n2️⃣ Testing WebSocket connection...');
    return new Promise((resolve) => {
        const ws = new WebSocket(WS_URL);
        
        ws.on('open', () => {
            console.log('✅ WebSocket connected successfully!');
            
            // Subscribe to all test executions
            ws.send(JSON.stringify({
                type: 'subscribe_all',
                timestamp: new Date().toISOString()
            }));
            
            resolve(ws);
        });
        
        ws.on('error', (error) => {
            console.error('❌ WebSocket connection failed:', error.message);
            resolve(null);
        });
        
        ws.on('message', (data) => {
            try {
                const message = JSON.parse(data.toString());
                console.log('📨 WebSocket message:', message);
            } catch (e) {
                console.log('📨 WebSocket raw message:', data.toString());
            }
        });
    });
}

// Test 3: Execute a test suite
async function testExecuteTests(suiteName) {
    console.log(`\n3️⃣ Testing test execution for suite: ${suiteName}...`);
    try {
        const response = await fetch(`${API_BASE}/api/tests/execute`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                suite_name: suiteName,
                filter: null,
                nocapture: false,
                parallel: true
            })
        });
        
        const data = await response.json();
        console.log('✅ Test execution started!');
        console.log('   Execution ID:', data.execution_id);
        console.log('   Status:', data.status);
        console.log('   Message:', data.message);
        return data.execution_id;
    } catch (error) {
        console.error('❌ Test execution failed:', error.message);
        return null;
    }
}

// Test 4: Check test status
async function testGetStatus(executionId) {
    console.log(`\n4️⃣ Testing status endpoint for execution: ${executionId}...`);
    try {
        const response = await fetch(`${API_BASE}/api/tests/status/${executionId}`);
        const data = await response.json();
        console.log('✅ Status retrieved successfully!');
        console.log('   Status:', data.status);
        console.log('   Progress:', `${data.progress.current}/${data.progress.total}`);
        console.log('   Current test:', data.current_test);
        return data;
    } catch (error) {
        console.error('❌ Status retrieval failed:', error.message);
        return null;
    }
}

// Main test flow
async function runTests() {
    console.log('Starting integration tests...\n');
    
    // Test 1: Discover suites
    const suites = await testDiscoverSuites();
    
    // Test 2: Connect WebSocket
    const ws = await testWebSocketConnection();
    
    if (suites.length > 0 && ws) {
        // Test 3: Execute first available suite
        const executionId = await testExecuteTests(suites[0].name);
        
        if (executionId) {
            // Test 4: Check status
            await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
            await testGetStatus(executionId);
            
            // Listen for test completion
            console.log('\n📡 Listening for test completion events...');
            console.log('   (Press Ctrl+C to exit)\n');
        }
    }
    
    // Keep the WebSocket connection open
    if (ws) {
        ws.on('close', () => {
            console.log('\n🔌 WebSocket connection closed');
            process.exit(0);
        });
    }
}

// Run the tests
runTests().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});