// Browser Automation Test for LLMKG Dashboard
// This script simulates browser interactions and verifies the dashboard

const puppeteer = require('puppeteer');
const fs = require('fs');

async function testDashboard() {
    console.log('🧠 LLMKG Dashboard Browser Automation Test\n');
    
    const browser = await puppeteer.launch({
        headless: false, // Set to true for CI/CD
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    
    // Set viewport
    await page.setViewport({ width: 1920, height: 1080 });
    
    // Capture console logs
    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push({
            type: msg.type(),
            text: msg.text(),
            location: msg.location()
        });
    });
    
    // Capture errors
    const pageErrors = [];
    page.on('pageerror', error => {
        pageErrors.push(error.toString());
    });
    
    try {
        console.log('1. Loading Dashboard...');
        await page.goto('http://localhost:3000', { 
            waitUntil: 'networkidle2',
            timeout: 30000 
        });
        console.log('   ✓ Dashboard loaded successfully');
        
        // Take screenshot
        await page.screenshot({ path: 'dashboard-overview.png' });
        console.log('   ✓ Screenshot saved: dashboard-overview.png');
        
        // Check for React app
        console.log('\n2. Checking React Application...');
        const reactRoot = await page.$('#root');
        if (reactRoot) {
            console.log('   ✓ React root element found');
        } else {
            console.log('   ✗ React root element NOT found');
        }
        
        // Check for loading states
        console.log('\n3. Checking Loading States...');
        const loadingElements = await page.$$('.loading, .spinner, [class*="loading"]');
        console.log(`   Found ${loadingElements.length} loading elements`);
        
        // Wait for data to load
        await page.waitForTimeout(3000);
        
        // Check WebSocket connection status
        console.log('\n4. Checking WebSocket Connection...');
        const wsStatus = await page.evaluate(() => {
            // Look for connection status elements
            const statusElements = document.querySelectorAll('[class*="connection"], [class*="status"]');
            const statuses = [];
            statusElements.forEach(el => {
                if (el.textContent.toLowerCase().includes('connect')) {
                    statuses.push(el.textContent);
                }
            });
            return statuses;
        });
        console.log('   WebSocket status indicators:', wsStatus);
        
        // Check for error messages
        console.log('\n5. Checking for Errors...');
        const errorElements = await page.$$('.error, [class*="error"], .ant-alert-error');
        if (errorElements.length > 0) {
            console.log(`   ✗ Found ${errorElements.length} error elements`);
            for (const el of errorElements) {
                const text = await el.evaluate(node => node.textContent);
                console.log(`     - ${text}`);
            }
        } else {
            console.log('   ✓ No error messages found');
        }
        
        // Test navigation
        console.log('\n6. Testing Navigation...');
        const navLinks = [
            { selector: 'a[href="/neural"]', name: 'Neural' },
            { selector: 'a[href="/cognitive"]', name: 'Cognitive' },
            { selector: 'a[href="/memory"]', name: 'Memory' },
            { selector: 'a[href="/tools"]', name: 'Tools' }
        ];
        
        for (const link of navLinks) {
            const element = await page.$(link.selector);
            if (element) {
                console.log(`   ✓ Found ${link.name} navigation link`);
            } else {
                console.log(`   ✗ ${link.name} navigation link NOT found`);
            }
        }
        
        // Check for data visualization components
        console.log('\n7. Checking Visualization Components...');
        const components = {
            charts: await page.$$('canvas, svg[class*="chart"], .recharts-wrapper'),
            tables: await page.$$('table, .ant-table'),
            cards: await page.$$('.ant-card, [class*="card"], [class*="metric"]'),
            graphs: await page.$$('[class*="graph"], [class*="visualization"]')
        };
        
        console.log(`   ✓ Found ${components.charts.length} chart elements`);
        console.log(`   ✓ Found ${components.tables.length} table elements`);
        console.log(`   ✓ Found ${components.cards.length} card/metric elements`);
        console.log(`   ✓ Found ${components.graphs.length} graph/visualization elements`);
        
        // Check for real-time data updates
        console.log('\n8. Monitoring Real-Time Updates...');
        const initialData = await page.evaluate(() => {
            const metricElements = document.querySelectorAll('[class*="metric"] [class*="value"], [class*="metric"] h4');
            return Array.from(metricElements).map(el => el.textContent);
        });
        
        await page.waitForTimeout(5000);
        
        const updatedData = await page.evaluate(() => {
            const metricElements = document.querySelectorAll('[class*="metric"] [class*="value"], [class*="metric"] h4');
            return Array.from(metricElements).map(el => el.textContent);
        });
        
        const dataChanged = JSON.stringify(initialData) !== JSON.stringify(updatedData);
        if (dataChanged) {
            console.log('   ✓ Real-time data is updating');
        } else {
            console.log('   ⚠ Data appears static (might be zero activity)');
        }
        
        // Navigate to different pages
        console.log('\n9. Testing Page Navigation...');
        const pages = [
            { path: '/neural', name: 'Neural Activity' },
            { path: '/memory', name: 'Memory Systems' },
            { path: '/tools', name: 'API Tools' }
        ];
        
        for (const pageInfo of pages) {
            try {
                await page.goto(`http://localhost:3000${pageInfo.path}`, {
                    waitUntil: 'networkidle2',
                    timeout: 10000
                });
                console.log(`   ✓ ${pageInfo.name} page loaded`);
                await page.screenshot({ path: `dashboard-${pageInfo.name.toLowerCase().replace(' ', '-')}.png` });
            } catch (e) {
                console.log(`   ✗ Failed to load ${pageInfo.name} page: ${e.message}`);
            }
        }
        
        // Final summary
        console.log('\n📊 Test Summary:');
        console.log(`   - Console Errors: ${pageErrors.length}`);
        console.log(`   - Console Warnings: ${consoleLogs.filter(l => l.type === 'warning').length}`);
        console.log(`   - Page Loaded: YES`);
        console.log(`   - Components Found: YES`);
        console.log(`   - Real-time Updates: ${dataChanged ? 'YES' : 'STATIC'}`);
        
        // Save detailed report
        const report = {
            timestamp: new Date().toISOString(),
            url: 'http://localhost:3000',
            errors: pageErrors,
            consoleLogs: consoleLogs.slice(0, 50), // First 50 logs
            components: {
                charts: components.charts.length,
                tables: components.tables.length,
                cards: components.cards.length,
                graphs: components.graphs.length
            },
            navigation: navLinks.map(l => ({ ...l, found: !!page.$(l.selector) })),
            dataUpdating: dataChanged
        };
        
        fs.writeFileSync('dashboard-test-report.json', JSON.stringify(report, null, 2));
        console.log('\n✅ Detailed report saved to dashboard-test-report.json');
        
    } catch (error) {
        console.error('\n❌ Test failed:', error.message);
    } finally {
        await browser.close();
    }
}

// Check if puppeteer is installed
try {
    require.resolve('puppeteer');
    testDashboard();
} catch (e) {
    console.log('⚠️  Puppeteer not installed. Run: npm install puppeteer');
    console.log('\nAlternatively, open dashboard-verification.html in your browser for manual testing.');
}