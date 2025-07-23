// Browser Dashboard Test Script
// Run this in browser console to verify dashboard functionality

console.log('=== LLMKG Dashboard Verification ===');

// 1. Check React is loaded
if (window.React) {
    console.log('✓ React is loaded');
} else {
    console.error('✗ React is NOT loaded');
}

// 2. Check Redux store
const checkReduxStore = () => {
    const rootElement = document.getElementById('root');
    if (rootElement && rootElement._reactRootContainer) {
        console.log('✓ React root container found');
        // Try to find Redux store in React DevTools
        if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
            console.log('✓ React DevTools detected');
        }
    } else {
        console.error('✗ React root container NOT found');
    }
};

// 3. Check WebSocket connection
const checkWebSocket = () => {
    console.log('\n--- WebSocket Status ---');
    
    // Check if WebSocket exists in window
    const wsConnections = [];
    
    // Try to create a test connection
    try {
        const testWs = new WebSocket('ws://localhost:8083');
        
        testWs.onopen = () => {
            console.log('✓ WebSocket connection successful to ws://localhost:8083');
            testWs.close();
        };
        
        testWs.onerror = (error) => {
            console.error('✗ WebSocket connection failed:', error);
        };
        
        testWs.onclose = () => {
            console.log('WebSocket test connection closed');
        };
    } catch (e) {
        console.error('✗ Failed to create WebSocket:', e);
    }
};

// 4. Check dashboard tabs
const checkDashboardTabs = () => {
    console.log('\n--- Dashboard Tabs ---');
    
    // Look for tab elements
    const tabs = document.querySelectorAll('[role="tab"], .ant-tabs-tab, a[href*="/"]');
    if (tabs.length > 0) {
        console.log(`✓ Found ${tabs.length} navigation elements:`);
        tabs.forEach((tab, index) => {
            const text = tab.textContent || tab.innerText || 'Unknown';
            const href = tab.getAttribute('href') || 'No href';
            console.log(`  ${index + 1}. ${text.trim()} (${href})`);
        });
    } else {
        console.error('✗ No tabs found');
    }
};

// 5. Check for loading states
const checkLoadingStates = () => {
    console.log('\n--- Loading States ---');
    
    const loadingElements = document.querySelectorAll(
        '.loading, .spinner, [class*="loading"], [class*="spinner"], .ant-spin'
    );
    
    if (loadingElements.length > 0) {
        console.warn(`⚠ Found ${loadingElements.length} loading elements - data might still be loading`);
    } else {
        console.log('✓ No loading spinners detected');
    }
};

// 6. Check for error messages
const checkErrors = () => {
    console.log('\n--- Error Check ---');
    
    const errorElements = document.querySelectorAll(
        '.error, .alert, [class*="error"], [class*="alert"], .ant-alert-error'
    );
    
    if (errorElements.length > 0) {
        console.error(`✗ Found ${errorElements.length} error elements:`);
        errorElements.forEach((el, index) => {
            console.error(`  ${index + 1}. ${el.textContent || 'No text'}`);
        });
    } else {
        console.log('✓ No error messages found');
    }
};

// 7. Check for data display
const checkDataDisplay = () => {
    console.log('\n--- Data Display ---');
    
    // Check for metric cards
    const metricCards = document.querySelectorAll(
        '[class*="metric"], [class*="card"], .ant-card'
    );
    console.log(`Found ${metricCards.length} metric/card elements`);
    
    // Check for charts
    const charts = document.querySelectorAll(
        'canvas, svg[class*="chart"], [class*="chart"], .recharts-wrapper'
    );
    console.log(`Found ${charts.length} chart elements`);
    
    // Check for tables
    const tables = document.querySelectorAll(
        'table, .ant-table, [class*="table"]'
    );
    console.log(`Found ${tables.length} table elements`);
};

// 8. Check console for errors
const checkConsoleErrors = () => {
    console.log('\n--- Console Error Summary ---');
    console.log('Check browser console for any red error messages above this test output');
};

// Run all checks
setTimeout(() => {
    checkReduxStore();
    checkWebSocket();
    checkDashboardTabs();
    checkLoadingStates();
    checkErrors();
    checkDataDisplay();
    checkConsoleErrors();
    
    console.log('\n=== End of Dashboard Verification ===');
    console.log('To get more details, inspect the Elements tab and Network tab in DevTools');
}, 1000);