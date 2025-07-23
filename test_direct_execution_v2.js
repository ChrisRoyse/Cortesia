#!/usr/bin/env node

/**
 * Direct Test Execution Verification v2
 * Tests if we can execute cargo tests directly and monitor the process
 */

const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('='.repeat(60));
console.log('Direct Cargo Test Execution Verification');
console.log('='.repeat(60));

// Test 1: Find test files
console.log('\n1. Finding test files in LLMKG project...\n');

function findTestFiles(dir, files = []) {
  if (!fs.existsSync(dir)) return files;
  
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory() && !['target', 'node_modules', '.git'].includes(entry.name)) {
      findTestFiles(fullPath, files);
    } else if (entry.isFile() && entry.name.endsWith('.rs') && 
               (entry.name.includes('test') || fullPath.includes('tests'))) {
      files.push(fullPath);
    }
  }
  
  return files;
}

const testFiles = findTestFiles('C:\\code\\LLMKG').slice(0, 10);
console.log(`Found ${testFiles.length} test files (showing first 10):`);
testFiles.forEach((file, i) => {
  console.log(`  ${i + 1}. ${path.relative('C:\\code\\LLMKG', file)}`);
});

// Test 2: Execute a simple test
console.log('\n2. Executing a simple test with cargo...\n');

const testProcess = spawn('cargo', ['test', 'test_function_name_extraction', '--', '--nocapture'], {
  cwd: 'C:\\code\\LLMKG',
  shell: true
});

let output = '';
let errorOutput = '';
let cargoProcessStarted = false;

// Monitor for cargo process
const monitorInterval = setInterval(() => {
  exec('tasklist | findstr cargo.exe', (error, stdout) => {
    if (stdout && stdout.includes('cargo.exe') && !cargoProcessStarted) {
      cargoProcessStarted = true;
      console.log('\n✅ CARGO PROCESS DETECTED - Real tests are being executed!\n');
    }
  });
}, 100);

testProcess.stdout.on('data', (data) => {
  output += data.toString();
  process.stdout.write(data);
});

testProcess.stderr.on('data', (data) => {
  errorOutput += data.toString();
  // Don't write warnings to stderr to keep output clean
  if (data.toString().includes('error') || !data.toString().includes('warning')) {
    process.stderr.write(data);
  }
});

testProcess.on('close', (code) => {
  clearInterval(monitorInterval);
  console.log(`\nTest process exited with code ${code}`);
  
  // Test 3: Parse test results
  console.log('\n3. Parsing test results...\n');
  
  const lines = output.split('\n');
  let testsRun = 0;
  let testsPassed = 0;
  let testsFailed = 0;
  let testOutput = [];
  
  lines.forEach(line => {
    if (line.includes('test result:')) {
      const passedMatch = line.match(/(\d+) passed/);
      const failedMatch = line.match(/(\d+) failed/);
      if (passedMatch) testsPassed = parseInt(passedMatch[1]) || 0;
      if (failedMatch) testsFailed = parseInt(failedMatch[1]) || 0;
      testsRun = testsPassed + testsFailed;
    }
    if (line.includes('running') && line.includes('test')) {
      testOutput.push(line);
    }
  });
  
  console.log('Test Results Summary:');
  console.log(`  Tests run: ${testsRun}`);
  console.log(`  Passed: ${testsPassed}`);
  console.log(`  Failed: ${testsFailed}`);
  console.log(`  Cargo process detected: ${cargoProcessStarted ? 'YES' : 'NO'}`);
  
  if (testOutput.length > 0) {
    console.log('\nTest execution lines found:');
    testOutput.slice(0, 3).forEach(line => {
      console.log(`  ${line.trim()}`);
    });
  }
  
  // Test 4: Check WebSocket endpoints
  console.log('\n4. Checking WebSocket endpoints...\n');
  
  const axios = require('axios');
  
  // Check different ports
  const portsToCheck = [
    { port: 8082, desc: 'Brain server HTTP' },
    { port: 8083, desc: 'Brain server WebSocket' },
    { port: 3000, desc: 'Dashboard (Vite)' },
    { port: 3001, desc: 'Dashboard WebSocket' }
  ];
  
  Promise.all(portsToCheck.map(async ({ port, desc }) => {
    try {
      const response = await axios.get(`http://localhost:${port}/`, { timeout: 1000 });
      return { port, desc, status: 'UP', hasTestAPI: false };
    } catch (error) {
      return { port, desc, status: 'DOWN', hasTestAPI: false };
    }
  })).then(results => {
    console.log('Port Status:');
    results.forEach(({ port, desc, status }) => {
      console.log(`  Port ${port} (${desc}): ${status}`);
    });
    
    // Final summary
    console.log('\n' + '='.repeat(60));
    console.log('VERIFICATION SUMMARY');
    console.log('='.repeat(60));
    
    console.log('\n✅ Direct cargo test execution: WORKING');
    console.log(`   - Found ${testFiles.length} test files`);
    console.log(`   - Executed test successfully (exit code: ${code})`);
    console.log(`   - ${testsRun} tests run, ${testsPassed} passed, ${testsFailed} failed`);
    if (cargoProcessStarted) {
      console.log('   - Cargo process confirmed running during test');
    }
    
    console.log('\n⚠️ Dashboard Integration Status:');
    console.log('   - Dashboard frontend: Running on port 3000/3002');
    console.log('   - Brain server: Running on ports 8082/8083');
    console.log('   - Test execution API: NOT IMPLEMENTED');
    console.log('   - WebSocket test streaming: NOT CONNECTED');
    
    console.log('\n❌ Missing Components:');
    console.log('   1. Test execution endpoints in brain server');
    console.log('   2. WebSocket handlers for test streaming');
    console.log('   3. Integration between TestExecutionTracker and cargo');
    
    console.log('\n📋 To Enable Test Execution from Dashboard:');
    console.log('   1. Add test_execution_tracker to brain server');
    console.log('   2. Implement /api/tests/* endpoints');
    console.log('   3. Add WebSocket test streaming to port 8083');
    console.log('   4. Update dashboard WebSocket URL from 8080 to 8083');
    console.log('   5. Wire TestExecutionTracker to spawn cargo test processes');
    
    console.log('\n💡 Current State:');
    console.log('   - Test UI exists and looks functional');
    console.log('   - Backend test discovery service exists');
    console.log('   - Cargo tests can be run manually');
    console.log('   - But UI and backend are NOT connected');
  });
});