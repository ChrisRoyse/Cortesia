#!/usr/bin/env node

/**
 * Demo: Real-time Test Execution with WebSocket Streaming
 * 
 * This demonstrates the complete backend integration for test execution
 */

const WebSocket = require('ws');

console.log('🚀 LLMKG Test Execution Demo\n');

const API_BASE = 'http://localhost:8082';
const WS_URL = 'ws://localhost:8083';

// Step 1: Connect to WebSocket for real-time updates
console.log('1️⃣ Connecting to WebSocket...');
const ws = new WebSocket(WS_URL);

ws.on('open', async () => {
    console.log('✅ WebSocket connected!\n');
    
    // Step 2: Discover available test suites
    console.log('2️⃣ Discovering test suites...');
    const discoverResponse = await fetch(`${API_BASE}/api/tests/discover`);
    const testData = await discoverResponse.json();
    
    console.log(`✅ Found ${testData.total_suites} test suites:`);
    testData.suites.forEach(suite => {
        console.log(`   - ${suite.name} (${suite.test_count} tests)`);
    });
    console.log('');
    
    // Step 3: Execute the first test suite
    const firstSuite = testData.suites[0];
    console.log(`3️⃣ Executing test suite: ${firstSuite.name}...`);
    
    const executeResponse = await fetch(`${API_BASE}/api/tests/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            suite_name: firstSuite.name,
            filter: null,
            nocapture: false,
            parallel: true
        })
    });
    
    const execution = await executeResponse.json();
    console.log(`✅ Test execution started!`);
    console.log(`   Execution ID: ${execution.execution_id}\n`);
    
    console.log('4️⃣ Streaming test output...\n');
    console.log('─'.repeat(60));
});

// Handle WebSocket messages
ws.on('message', (data) => {
    try {
        const message = JSON.parse(data.toString());
        
        switch (message.TestStarted ? 'TestStarted' : 
                message.TestProgress ? 'TestProgress' :
                message.TestCompleted ? 'TestCompleted' :
                message.TestFailed ? 'TestFailed' :
                message.TestLog ? 'TestLog' : null) {
            
            case 'TestStarted':
                console.log(`\n🏁 TEST SUITE STARTED: ${message.TestStarted.suite_name}`);
                console.log(`   Total tests: ${message.TestStarted.total_tests || 'calculating...'}\n`);
                break;
                
            case 'TestProgress':
                const progress = message.TestProgress;
                const percent = Math.round((progress.current / progress.total) * 100);
                console.log(`📊 Progress: [${progress.current}/${progress.total}] ${percent}%`);
                console.log(`   Current test: ${progress.test_name} - ${progress.status.toUpperCase()}`);
                break;
                
            case 'TestCompleted':
                const completed = message.TestCompleted;
                console.log('\n─'.repeat(60));
                console.log(`\n✅ TEST SUITE COMPLETED!`);
                console.log(`   Passed:  ${completed.passed} ✓`);
                console.log(`   Failed:  ${completed.failed} ✗`);
                console.log(`   Ignored: ${completed.ignored} -`);
                console.log(`   Duration: ${(completed.duration_ms / 1000).toFixed(2)}s\n`);
                break;
                
            case 'TestFailed':
                console.log(`\n❌ TEST EXECUTION FAILED: ${message.TestFailed.error}\n`);
                break;
                
            case 'TestLog':
                const log = message.TestLog;
                if (log.message.includes('test') && !log.message.includes('running')) {
                    // Format test output nicely
                    if (log.message.includes('ok')) {
                        console.log(`   ✓ ${log.message}`);
                    } else if (log.message.includes('FAILED')) {
                        console.log(`   ✗ ${log.message}`);
                    }
                }
                break;
        }
    } catch (e) {
        // Raw message, just display it
        console.log(`📨 ${data.toString()}`);
    }
});

ws.on('error', (error) => {
    console.error('❌ WebSocket error:', error.message);
});

ws.on('close', () => {
    console.log('\n🔌 WebSocket connection closed');
    process.exit(0);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n\nShutting down...');
    ws.close();
    process.exit(0);
});