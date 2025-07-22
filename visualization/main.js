/**
 * LLMKG Visualization System Bootstrap
 * 
 * This script serves as the entry point for the LLMKG visualization system,
 * providing initialization, status monitoring, and coordination between
 * the React dashboard and WebSocket communication systems.
 */

// Configuration
const CONFIG = {
  dashboard: {
    buildPath: './dashboard/build/index.html',
    devPath: './dashboard/public/index.html',
    port: 3000
  },
  websocket: {
    port: 8080,
    url: 'ws://localhost:8080'
  },
  mcp: {
    port: 3000,
    url: 'http://localhost:3000'
  },
  phase1: {
    serverScript: './phase1/dist/websocket/server.js',
    devScript: './phase1/src/websocket/server.ts'
  }
};

// State management
const state = {
  dashboardStatus: 'checking',
  websocketStatus: 'checking', 
  mcpStatus: 'checking',
  initialized: false
};

/**
 * Initialize the visualization system
 */
function initializeSystem() {
  console.log('🚀 Initializing LLMKG Visualization System...');
  
  // Update UI
  updateStatusIndicators();
  
  // Check system components
  Promise.allSettled([
    checkDashboardStatus(),
    checkWebSocketStatus(),
    checkMCPStatus()
  ]).then((results) => {
    console.log('✅ System initialization complete');
    state.initialized = true;
    
    // Update final status
    updateStatusIndicators();
    
    // Auto-redirect to dashboard if everything is ready
    if (shouldAutoRedirect()) {
      setTimeout(() => {
        console.log('🔄 Auto-redirecting to main dashboard...');
        window.location.href = getDashboardUrl();
      }, 2000);
    }
  });
}

/**
 * Check if dashboard system is available
 */
async function checkDashboardStatus() {
  try {
    // Check if built dashboard exists
    const buildResponse = await fetch(CONFIG.dashboard.buildPath, { 
      method: 'HEAD',
      cache: 'no-cache'
    });
    
    if (buildResponse.ok) {
      state.dashboardStatus = 'ready';
      console.log('✅ Dashboard build found');
      return;
    }
  } catch (error) {
    console.log('ℹ️ Built dashboard not available, checking dev server...');
  }
  
  try {
    // Check if dev server is running
    const devResponse = await fetch(`http://localhost:${CONFIG.dashboard.port}`, {
      method: 'HEAD',
      cache: 'no-cache'
    });
    
    if (devResponse.ok) {
      state.dashboardStatus = 'dev';
      console.log('✅ Dashboard dev server detected');
    } else {
      state.dashboardStatus = 'unavailable';
    }
  } catch (error) {
    console.log('⚠️ Dashboard not available');
    state.dashboardStatus = 'unavailable';
  }
}

/**
 * Check WebSocket server status
 */
async function checkWebSocketStatus() {
  return new Promise((resolve) => {
    const ws = new WebSocket(CONFIG.websocket.url);
    const timeout = setTimeout(() => {
      state.websocketStatus = 'unavailable';
      console.log('⚠️ WebSocket server not available');
      resolve();
    }, 3000);
    
    ws.onopen = () => {
      clearTimeout(timeout);
      state.websocketStatus = 'ready';
      console.log('✅ WebSocket server available');
      ws.close();
      resolve();
    };
    
    ws.onerror = () => {
      clearTimeout(timeout);
      state.websocketStatus = 'unavailable';
      console.log('⚠️ WebSocket server not available');
      resolve();
    };
  });
}

/**
 * Check MCP integration status
 */
async function checkMCPStatus() {
  try {
    const response = await fetch(`${CONFIG.mcp.url}/health`, {
      method: 'GET',
      cache: 'no-cache',
      timeout: 3000
    });
    
    if (response.ok) {
      state.mcpStatus = 'ready';
      console.log('✅ MCP integration available');
    } else {
      state.mcpStatus = 'unavailable';
    }
  } catch (error) {
    console.log('ℹ️ MCP integration not available (optional)');
    state.mcpStatus = 'unavailable';
  }
}

/**
 * Update visual status indicators
 */
function updateStatusIndicators() {
  const dashboardIndicator = document.getElementById('dashboard-status');
  const websocketIndicator = document.getElementById('websocket-status');
  const mcpIndicator = document.getElementById('mcp-status');
  
  // Update dashboard status
  dashboardIndicator.className = `status-indicator ${getIndicatorClass(state.dashboardStatus)}`;
  
  // Update websocket status  
  websocketIndicator.className = `status-indicator ${getIndicatorClass(state.websocketStatus)}`;
  
  // Update MCP status
  mcpIndicator.className = `status-indicator ${getIndicatorClass(state.mcpStatus)}`;
}

/**
 * Get CSS class for status indicator
 */
function getIndicatorClass(status) {
  switch (status) {
    case 'ready':
    case 'dev':
      return '';  // Green (default)
    case 'checking':
      return 'checking';  // Orange
    case 'unavailable':
    case 'error':
      return 'error';  // Red
    default:
      return 'checking';
  }
}

/**
 * Determine if auto-redirect should occur
 */
function shouldAutoRedirect() {
  // Only redirect if dashboard is available
  return state.dashboardStatus === 'ready' || state.dashboardStatus === 'dev';
}

/**
 * Get the appropriate dashboard URL
 */
function getDashboardUrl() {
  if (state.dashboardStatus === 'dev') {
    return `http://localhost:${CONFIG.dashboard.port}`;
  } else {
    return CONFIG.dashboard.buildPath;
  }
}

/**
 * Launch WebSocket system
 */
function launchWebSocketSystem() {
  console.log('🔌 Launching WebSocket system...');
  
  if (state.websocketStatus === 'ready') {
    console.log('✅ WebSocket server already running');
    alert('WebSocket server is already running at ws://localhost:8080');
    return;
  }
  
  // Check if Node.js environment is available
  if (typeof require !== 'undefined') {
    try {
      // Try to start the Phase 1 WebSocket server
      const { spawn } = require('child_process');
      
      // First try built version
      let serverProcess = spawn('node', [CONFIG.phase1.serverScript], {
        stdio: 'pipe',
        detached: false
      });
      
      serverProcess.on('error', (error) => {
        console.log('ℹ️ Built server not available, trying dev server...');
        
        // Try dev version
        serverProcess = spawn('npx', ['ts-node', CONFIG.phase1.devScript], {
          stdio: 'pipe',
          cwd: './phase1',
          detached: false
        });
      });
      
      serverProcess.stdout.on('data', (data) => {
        console.log(`WebSocket Server: ${data}`);
      });
      
      serverProcess.stderr.on('data', (data) => {
        console.error(`WebSocket Server Error: ${data}`);
      });
      
      alert('WebSocket server is starting up... Check console for details.');
      
    } catch (error) {
      console.error('Failed to launch WebSocket server:', error);
      alert('Failed to launch WebSocket server. Please start it manually:\n\ncd phase1\nnpm run dev');
    }
  } else {
    // Browser environment - provide instructions
    const instructions = `
To launch the WebSocket system manually:

1. Open a new terminal
2. Navigate to the phase1 directory:
   cd visualization/phase1
3. Install dependencies (if not already done):
   npm install
4. Start the server:
   npm run dev

The server will start on ws://localhost:8080
    `.trim();
    
    alert(instructions);
  }
}

/**
 * Handle dashboard option click
 */
function handleDashboardClick(event) {
  const mainDashboard = document.getElementById('main-dashboard');
  
  if (event.target === mainDashboard || mainDashboard.contains(event.target)) {
    if (state.dashboardStatus === 'unavailable') {
      event.preventDefault();
      
      const instructions = `
Dashboard not available. To start the dashboard:

Build Option:
1. cd visualization/dashboard  
2. npm install
3. npm run build

Development Option:
1. cd visualization/dashboard
2. npm install  
3. npm start

Then refresh this page.
      `.trim();
      
      alert(instructions);
    } else if (state.dashboardStatus === 'dev') {
      event.preventDefault();
      window.open(`http://localhost:${CONFIG.dashboard.port}`, '_blank');
    }
  }
}

/**
 * Keyboard shortcuts
 */
function handleKeyboardShortcuts(event) {
  // Ctrl/Cmd + D -> Dashboard
  if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
    event.preventDefault();
    if (state.dashboardStatus !== 'unavailable') {
      window.location.href = getDashboardUrl();
    }
  }
  
  // Ctrl/Cmd + W -> WebSocket
  if ((event.ctrlKey || event.metaKey) && event.key === 'w') {
    event.preventDefault();
    launchWebSocketSystem();
  }
  
  // Ctrl/Cmd + R -> Refresh status
  if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
    event.preventDefault();
    console.log('🔄 Refreshing system status...');
    initializeSystem();
  }
}

/**
 * Error handling
 */
function handleError(error) {
  console.error('LLMKG Visualization System Error:', error);
  
  // Update UI to show error state if needed
  if (!state.initialized) {
    // Show error in status indicators
    const indicators = document.querySelectorAll('.status-indicator');
    indicators.forEach(indicator => {
      if (indicator.classList.contains('checking')) {
        indicator.classList.remove('checking');
        indicator.classList.add('error');
      }
    });
  }
}

/**
 * Utility functions
 */
const utils = {
  // Check if running in development mode
  isDevelopment() {
    return window.location.hostname === 'localhost' || 
           window.location.hostname === '127.0.0.1' ||
           window.location.hostname === '';
  },
  
  // Get system information
  getSystemInfo() {
    return {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      cookieEnabled: navigator.cookieEnabled,
      onLine: navigator.onLine,
      timestamp: new Date().toISOString()
    };
  },
  
  // Log system information
  logSystemInfo() {
    const info = this.getSystemInfo();
    console.log('🖥️ System Information:', info);
  }
};

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
  console.log('📱 LLMKG Visualization System Loaded');
  utils.logSystemInfo();
  
  // Initialize system
  setTimeout(initializeSystem, 500);
  
  // Add event listeners
  document.addEventListener('click', handleDashboardClick);
  document.addEventListener('keydown', handleKeyboardShortcuts);
  
  // Global error handler
  window.addEventListener('error', (event) => {
    handleError(event.error);
  });
  
  window.addEventListener('unhandledrejection', (event) => {
    handleError(event.reason);
  });
});

// Expose global functions
window.launchWebSocketSystem = launchWebSocketSystem;
window.LLMKG = {
  state,
  config: CONFIG,
  utils,
  initializeSystem,
  checkDashboardStatus,
  checkWebSocketStatus,
  checkMCPStatus
};

console.log('🎯 LLMKG Visualization Bootstrap loaded');