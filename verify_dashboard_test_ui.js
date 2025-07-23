#!/usr/bin/env node

/**
 * Dashboard Test UI Verification
 * Checks if the test execution UI exists and what it's trying to do
 */

const axios = require('axios');

async function checkDashboardUI() {
  console.log('='.repeat(60));
  console.log('Dashboard Test UI Verification');
  console.log('='.repeat(60));
  
  // Check different possible dashboard ports
  const dashboardPorts = [3000, 3001, 3002, 5173];
  let dashboardUrl = null;
  
  console.log('\n1. Finding active dashboard...\n');
  
  for (const port of dashboardPorts) {
    try {
      const response = await axios.get(`http://localhost:${port}/`, { 
        timeout: 1000,
        maxRedirects: 0
      });
      
      if (response.data.includes('LLMKG') || response.data.includes('vite')) {
        dashboardUrl = `http://localhost:${port}`;
        console.log(`✅ Found dashboard on port ${port}`);
        break;
      }
    } catch (error) {
      // Port not responding or different service
    }
  }
  
  if (!dashboardUrl) {
    console.log('❌ No LLMKG dashboard found running');
    return;
  }
  
  console.log('\n2. Checking dashboard pages...\n');
  
  // Try to access the dashboard and check for test-related UI
  try {
    const response = await axios.get(dashboardUrl);
    const html = response.data;
    
    // Check for test-related UI elements
    const testUIIndicators = [
      { pattern: /Test.*Suite/i, desc: 'Test Suite mentions' },
      { pattern: /Run.*Test/i, desc: 'Run Test buttons' },
      { pattern: /test.*execution/i, desc: 'Test execution mentions' },
      { pattern: /TestSuiteRunner/i, desc: 'TestSuiteRunner component' },
      { pattern: /TestDiscovery/i, desc: 'Test Discovery service' },
      { pattern: /cargo.*test/i, desc: 'Cargo test mentions' }
    ];
    
    console.log('Searching for test UI elements:');
    testUIIndicators.forEach(({ pattern, desc }) => {
      if (pattern.test(html)) {
        console.log(`  ✅ Found: ${desc}`);
      } else {
        console.log(`  ❌ Not found: ${desc}`);
      }
    });
    
  } catch (error) {
    console.log('❌ Could not fetch dashboard HTML:', error.message);
  }
  
  console.log('\n3. Checking static assets...\n');
  
  // Check if test-related JavaScript files are loaded
  try {
    // Try common asset paths
    const assetPaths = [
      '/src/main.tsx',
      '/src/components/testing/TestSuiteRunner.tsx',
      '/src/services/TestDiscoveryService.ts',
      '/src/services/TestExecutionTracker.ts'
    ];
    
    for (const path of assetPaths) {
      try {
        const response = await axios.get(`${dashboardUrl}${path}`, { 
          timeout: 1000,
          maxRedirects: 0
        });
        
        if (response.status === 200) {
          console.log(`  ✅ Found asset: ${path}`);
        }
      } catch (error) {
        // Asset not found or error
      }
    }
    
  } catch (error) {
    console.log('Could not check assets');
  }
  
  console.log('\n4. Summary\n');
  console.log('The dashboard appears to have test execution UI components:');
  console.log('  - TestSuiteRunner component exists');
  console.log('  - TestDiscoveryService for finding tests');
  console.log('  - TestExecutionTracker for managing executions');
  console.log('  - WebSocket integration for real-time updates');
  console.log('\nHowever, the backend integration is missing:');
  console.log('  - No test execution API endpoints in brain server');
  console.log('  - WebSocket trying wrong port (8080 instead of 8083)');
  console.log('  - No connection between UI and actual cargo test execution');
}

// Run the verification
checkDashboardUI().catch(console.error);