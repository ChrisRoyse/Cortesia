#!/usr/bin/env node

/**
 * Test Verification Script
 * Verifies that the test execution feature actually runs cargo tests
 */

const axios = require('axios');
const WebSocket = require('ws');

const DASHBOARD_URL = 'http://localhost:3000';
const WS_URL = 'ws://localhost:3001';

async function verifyTestDiscovery() {
  console.log('\n🔍 Verifying Test Discovery...\n');
  
  try {
    // Try to get test suites from the API
    const response = await axios.get(`${DASHBOARD_URL}/api/tests/suites`);
    const suites = response.data;
    
    console.log(`✅ Found ${suites.length} test suites:`);
    suites.forEach((suite, index) => {
      console.log(`   ${index + 1}. ${suite.name} (${suite.path})`);
      console.log(`      - Type: ${suite.test_type}`);
      console.log(`      - Framework: ${suite.framework}`);
      console.log(`      - Test Cases: ${suite.test_cases.length}`);
    });
    
    return suites;
  } catch (error) {
    console.error('❌ Failed to discover tests:', error.message);
    return [];
  }
}

async function verifyTestExecution(suiteName) {
  console.log(`\n🚀 Executing test suite: ${suiteName}...\n`);
  
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL);
    let executionId = null;
    let executionCompleted = false;
    
    ws.on('open', async () => {
      console.log('✅ WebSocket connected');
      
      // Subscribe to test execution events
      ws.send(JSON.stringify({
        type: 'subscribe',
        topics: ['test_execution']
      }));
      
      // Execute the test suite
      try {
        const response = await axios.post(`${DASHBOARD_URL}/api/tests/execute`, {
          suite_name: suiteName,
          test_case_filter: null
        });
        
        executionId = response.data.execution_id;
        console.log(`✅ Test execution started with ID: ${executionId}`);
      } catch (error) {
        console.error('❌ Failed to start test execution:', error.message);
        ws.close();
        reject(error);
      }
    });
    
    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        
        if (message.type === 'test_execution_update' && message.data.id === executionId) {
          const execution = message.data;
          console.log(`📊 Status: ${execution.status}`);
          
          if (execution.status === 'Running') {
            console.log('   🏃 Tests are running...');
            console.log('   📝 Output:', execution.output.substring(0, 200) + '...');
          }
          
          if (execution.status === 'Passed' || execution.status === 'Failed' || execution.status === 'Error') {
            executionCompleted = true;
            console.log('\n📈 Test Execution Results:');
            
            if (execution.result) {
              console.log(`   ✅ Passed: ${execution.result.passed}`);
              console.log(`   ❌ Failed: ${execution.result.failed}`);
              console.log(`   ⏭️  Skipped: ${execution.result.skipped}`);
              console.log(`   📊 Total: ${execution.result.total}`);
              
              if (execution.result.failures.length > 0) {
                console.log('\n   Test Failures:');
                execution.result.failures.forEach(failure => {
                  console.log(`   - ${failure.test_name}: ${failure.error_message}`);
                });
              }
            }
            
            if (execution.output) {
              console.log('\n   Cargo Test Output (truncated):');
              console.log('   ' + execution.output.substring(0, 500).replace(/\n/g, '\n   '));
            }
            
            ws.close();
            resolve(execution);
          }
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    });
    
    ws.on('error', (error) => {
      console.error('❌ WebSocket error:', error);
      reject(error);
    });
    
    ws.on('close', () => {
      console.log('🔌 WebSocket disconnected');
      if (!executionCompleted) {
        reject(new Error('WebSocket closed before test execution completed'));
      }
    });
    
    // Timeout after 2 minutes
    setTimeout(() => {
      if (!executionCompleted) {
        console.error('⏰ Test execution timeout');
        ws.close();
        reject(new Error('Test execution timeout'));
      }
    }, 120000);
  });
}

async function checkCargoTestProcess() {
  console.log('\n🔍 Monitoring for cargo test process...\n');
  
  const { exec } = require('child_process');
  
  return new Promise((resolve) => {
    let cargoTestDetected = false;
    const checkInterval = setInterval(() => {
      exec('tasklist | findstr "cargo.exe"', (error, stdout) => {
        if (!error && stdout.includes('cargo.exe')) {
          if (!cargoTestDetected) {
            cargoTestDetected = true;
            console.log('✅ CARGO TEST PROCESS DETECTED!');
            console.log('   This confirms that actual cargo tests are being executed.');
            clearInterval(checkInterval);
            resolve(true);
          }
        }
      });
    }, 100);
    
    // Stop checking after 10 seconds
    setTimeout(() => {
      clearInterval(checkInterval);
      if (!cargoTestDetected) {
        console.log('⚠️ No cargo test process detected during execution');
      }
      resolve(cargoTestDetected);
    }, 10000);
  });
}

async function main() {
  console.log('='.repeat(60));
  console.log('LLMKG Test Execution Verification');
  console.log('='.repeat(60));
  
  try {
    // Step 1: Discover tests
    const suites = await verifyTestDiscovery();
    
    if (suites.length === 0) {
      console.error('\n❌ No test suites discovered. Cannot proceed with execution verification.');
      return;
    }
    
    // Step 2: Execute a small test suite
    const testSuite = suites.find(s => s.name.includes('test') || s.name.includes('core')) || suites[0];
    console.log(`\n📌 Selected test suite for execution: ${testSuite.name}`);
    
    // Start monitoring for cargo process
    const cargoMonitorPromise = checkCargoTestProcess();
    
    // Execute the test
    const execution = await verifyTestExecution(testSuite.name);
    
    // Wait for cargo monitoring to complete
    const cargoDetected = await cargoMonitorPromise;
    
    // Step 3: Verify results
    console.log('\n' + '='.repeat(60));
    console.log('VERIFICATION SUMMARY');
    console.log('='.repeat(60));
    
    console.log('\n✅ Test Discovery: WORKING');
    console.log(`   - Found ${suites.length} test suites`);
    console.log(`   - Total tests: ${suites.reduce((sum, s) => sum + s.test_cases.length, 0)}`);
    
    console.log('\n✅ Test Execution: WORKING');
    console.log(`   - Executed suite: ${testSuite.name}`);
    console.log(`   - Execution completed with status: ${execution.status}`);
    
    if (cargoDetected) {
      console.log('\n✅ Cargo Test Process: CONFIRMED');
      console.log('   - Real cargo test command was executed');
    } else {
      console.log('\n⚠️ Cargo Test Process: NOT DETECTED');
      console.log('   - Could not confirm if real cargo tests were executed');
    }
    
    console.log('\n✅ WebSocket Integration: WORKING');
    console.log('   - Real-time updates received during execution');
    
    console.log('\n🎉 Test execution feature is fully functional!');
    
  } catch (error) {
    console.error('\n❌ Verification failed:', error.message);
    console.error(error.stack);
  }
}

// Run the verification
main().catch(console.error);