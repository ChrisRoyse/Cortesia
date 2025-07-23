// Debug script to verify brain metrics are being collected and sent
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8083');

let messageCount = 0;
let lastBrainMetrics = null;

ws.on('open', () => {
    console.log('🔌 Connected to LLMKG WebSocket server');
    console.log('📊 Monitoring for brain metrics...\n');
});

ws.on('message', (data) => {
    try {
        const message = JSON.parse(data);
        messageCount++;
        
        if (message.MetricsUpdate) {
            const metrics = message.MetricsUpdate.metrics || {};
            const brainMetrics = {};
            let hasBrainMetrics = false;
            
            // Extract all brain_ prefixed metrics
            for (const [key, value] of Object.entries(metrics)) {
                if (key.startsWith('brain_')) {
                    brainMetrics[key] = value;
                    hasBrainMetrics = true;
                }
            }
            
            if (hasBrainMetrics) {
                console.log(`\n🧠 Brain Metrics Update #${messageCount}:`);
                console.log('─'.repeat(50));
                
                // Display metrics in a formatted way
                if (brainMetrics.brain_entity_count !== undefined) {
                    console.log(`📦 Entities: ${Math.floor(brainMetrics.brain_entity_count)}`);
                }
                if (brainMetrics.brain_relationship_count !== undefined) {
                    console.log(`🔗 Relationships: ${Math.floor(brainMetrics.brain_relationship_count)}`);
                }
                if (brainMetrics.brain_active_entities !== undefined) {
                    console.log(`✨ Active Entities: ${Math.floor(brainMetrics.brain_active_entities)}`);
                }
                if (brainMetrics.brain_avg_activation !== undefined) {
                    console.log(`📊 Avg Activation: ${brainMetrics.brain_avg_activation.toFixed(3)}`);
                }
                if (brainMetrics.brain_max_activation !== undefined) {
                    console.log(`🔥 Max Activation: ${brainMetrics.brain_max_activation.toFixed(3)}`);
                }
                if (brainMetrics.brain_total_activation !== undefined) {
                    console.log(`💫 Total Activation: ${brainMetrics.brain_total_activation.toFixed(3)}`);
                }
                if (brainMetrics.brain_graph_density !== undefined) {
                    console.log(`🌐 Graph Density: ${brainMetrics.brain_graph_density.toFixed(3)}`);
                }
                if (brainMetrics.brain_clustering_coefficient !== undefined) {
                    console.log(`🔀 Clustering: ${brainMetrics.brain_clustering_coefficient.toFixed(3)}`);
                }
                if (brainMetrics.brain_concept_coherence !== undefined) {
                    console.log(`🎯 Coherence: ${brainMetrics.brain_concept_coherence.toFixed(3)}`);
                }
                if (brainMetrics.brain_learning_efficiency !== undefined) {
                    console.log(`📈 Learning: ${brainMetrics.brain_learning_efficiency.toFixed(3)}`);
                }
                
                // Check for changes
                if (lastBrainMetrics) {
                    const changes = [];
                    for (const [key, value] of Object.entries(brainMetrics)) {
                        if (lastBrainMetrics[key] !== value) {
                            changes.push(`${key}: ${lastBrainMetrics[key]?.toFixed(3) || 'N/A'} → ${value.toFixed(3)}`);
                        }
                    }
                    if (changes.length > 0) {
                        console.log('\n🔄 Changes detected:');
                        changes.forEach(change => console.log(`   ${change}`));
                    } else {
                        console.log('\n⚠️  No changes in brain metrics (data might be static)');
                    }
                }
                
                lastBrainMetrics = {...brainMetrics};
                
                // Check if metrics look synthetic (all zeros or static values)
                const allZero = Object.values(brainMetrics).every(v => v === 0);
                const allStatic = lastBrainMetrics && 
                    Object.entries(brainMetrics).every(([k, v]) => lastBrainMetrics[k] === v);
                
                if (allZero) {
                    console.log('\n❌ WARNING: All brain metrics are zero!');
                    console.log('   The BrainMetricsCollector might not be running properly.');
                } else if (allStatic && messageCount > 3) {
                    console.log('\n⚠️  WARNING: Brain metrics are static!');
                    console.log('   The brain simulation might not be updating entities.');
                }
                
            } else {
                console.log(`\n❌ No brain metrics found in update #${messageCount}`);
                console.log('Available metrics:', Object.keys(metrics).join(', '));
            }
            
            // Also check system metrics to ensure connection is working
            const systemMetrics = message.MetricsUpdate.system_metrics;
            if (systemMetrics) {
                console.log(`\n💻 System: CPU ${systemMetrics.cpu_usage_percent.toFixed(1)}% | Memory ${systemMetrics.memory_usage_percent.toFixed(1)}%`);
            }
        }
    } catch (error) {
        console.error('Error parsing message:', error);
    }
});

ws.on('error', (error) => {
    console.error('❌ WebSocket error:', error);
});

ws.on('close', () => {
    console.log('\n🔌 WebSocket connection closed');
    console.log(`📊 Total messages received: ${messageCount}`);
    process.exit(0);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Shutting down...');
    ws.close();
});

console.log('Press Ctrl+C to exit\n');