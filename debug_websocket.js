const WebSocket = require('ws');

console.log('🔍 LLMKG WebSocket Debugger');
console.log('==========================\n');

const wsUrl = 'ws://localhost:8083';
let messageCount = 0;
let lastMetricsUpdate = null;

console.log(`📡 Connecting to ${wsUrl}...`);
const ws = new WebSocket(wsUrl);

ws.on('open', () => {
    console.log('✅ Connected successfully!\n');
    console.log('📤 Sending Ping message...');
    ws.send(JSON.stringify({ Ping: null }));
});

ws.on('message', (data) => {
    messageCount++;
    
    try {
        const message = JSON.parse(data.toString());
        console.log(`\n📨 Message #${messageCount} received at ${new Date().toLocaleTimeString()}`);
        console.log('Message type:', Object.keys(message)[0]);
        
        if (message.MetricsUpdate) {
            lastMetricsUpdate = message.MetricsUpdate;
            console.log('\n🧠 MetricsUpdate Details:');
            console.log('- Timestamp:', new Date(message.MetricsUpdate.timestamp * 1000).toLocaleString());
            
            // Check system metrics
            if (message.MetricsUpdate.system_metrics) {
                console.log('\n📊 System Metrics:');
                console.log('  - CPU Usage:', message.MetricsUpdate.system_metrics.cpu_usage_percent + '%');
                console.log('  - Memory Usage:', message.MetricsUpdate.system_metrics.memory_usage_percent + '%');
            }
            
            // Check for brain-specific metrics
            if (message.MetricsUpdate.metrics) {
                console.log('\n🧠 Brain-Specific Metrics Found:');
                const brainMetrics = {};
                for (const [key, value] of Object.entries(message.MetricsUpdate.metrics)) {
                    if (key.startsWith('brain_')) {
                        brainMetrics[key] = value;
                        console.log(`  - ${key}: ${value}`);
                    }
                }
                
                if (Object.keys(brainMetrics).length === 0) {
                    console.log('  ⚠️  No brain_ prefixed metrics found');
                    console.log('  Available metrics:', Object.keys(message.MetricsUpdate.metrics).slice(0, 10).join(', '));
                }
            } else {
                console.log('\n⚠️  No metrics field found in MetricsUpdate');
                console.log('Available fields:', Object.keys(message.MetricsUpdate));
            }
            
            // Save sample for analysis
            require('fs').writeFileSync(
                'sample_metrics_update.json', 
                JSON.stringify(message.MetricsUpdate, null, 2)
            );
            console.log('\n💾 Sample saved to sample_metrics_update.json');
        } else if (message.Pong) {
            console.log('🏓 Pong received');
        } else {
            console.log('Full message:', JSON.stringify(message, null, 2));
        }
    } catch (error) {
        console.error('❌ Error parsing message:', error);
        console.error('Raw data:', data.toString());
    }
});

ws.on('error', (error) => {
    console.error('\n❌ WebSocket error:', error.message);
    if (error.code === 'ECONNREFUSED') {
        console.error('🚫 Connection refused. Is the LLMKG server running on port 8083?');
        console.error('   Check with: netstat -ano | findstr :8083');
    }
});

ws.on('close', (code, reason) => {
    console.log(`\n🔌 Connection closed: ${code} - ${reason}`);
    console.log(`📊 Total messages received: ${messageCount}`);
    
    if (lastMetricsUpdate) {
        console.log('\n📋 Summary of last MetricsUpdate:');
        console.log('- Has system_metrics:', !!lastMetricsUpdate.system_metrics);
        console.log('- Has application_metrics:', !!lastMetricsUpdate.application_metrics);
        console.log('- Has performance_metrics:', !!lastMetricsUpdate.performance_metrics);
        console.log('- Has metrics field:', !!lastMetricsUpdate.metrics);
        console.log('- Has alerts:', !!lastMetricsUpdate.alerts);
    }
});

// Keep the script running for 30 seconds to collect data
console.log('\n⏱️  Running for 30 seconds to collect data...');
setTimeout(() => {
    console.log('\n👋 Closing connection...');
    ws.close();
    process.exit(0);
}, 30000);