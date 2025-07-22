/**
 * @fileoverview Data Collectors Demo for LLMKG Visualization
 * 
 * This example demonstrates how to use the LLMKG data collectors for high-frequency
 * data collection and real-time monitoring of LLMKG systems. It shows setup,
 * configuration, and usage patterns for all collector types.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

import { 
  CollectorManager,
  CollectorFactory,
  CollectorType,
  CollectorUtils,
  HIGH_PERFORMANCE_CONFIGS,
  type CollectorConfigs,
  type ManagerStats
} from '../collectors/index.js';
import { MCPClient } from '../mcp/client.js';

/**
 * Comprehensive demo of LLMKG data collectors
 */
async function demonstrateCollectors() {
  console.log('🚀 Starting LLMKG Data Collectors Demo');
  console.log('=' * 50);

  // Step 1: Initialize MCP Client
  console.log('\n📡 Step 1: Initializing MCP Client...');
  const mcpClient = new MCPClient({
    clientInfo: {
      name: 'LLMKG-Visualization-Demo',
      version: '1.0.0'
    },
    enableTelemetry: true,
    autoDiscoverTools: true
  });

  try {
    // Connect to LLMKG MCP servers
    console.log('Connecting to LLMKG MCP servers...');
    await mcpClient.connectMultiple([
      'ws://localhost:8001', // BrainInspiredMCPServer
      'ws://localhost:8002'  // FederatedMCPServer
    ]);
    console.log('✅ Connected to MCP servers successfully');

  } catch (error) {
    console.warn('⚠️  MCP connection failed, using demo mode:', error.message);
  }

  // Step 2: Demonstrate Individual Collectors
  console.log('\n🔬 Step 2: Individual Collector Demonstrations');
  await demonstrateIndividualCollectors(mcpClient);

  // Step 3: Demonstrate Collector Manager
  console.log('\n🎛️  Step 3: Collector Manager Demonstration');
  await demonstrateCollectorManager(mcpClient);

  // Step 4: Demonstrate High-Performance Setup
  console.log('\n⚡ Step 4: High-Performance Collection Demo');
  await demonstrateHighPerformanceCollection(mcpClient);

  // Step 5: Demonstrate Health Monitoring
  console.log('\n🏥 Step 5: Health Monitoring and Analytics');
  await demonstrateHealthMonitoring(mcpClient);

  // Step 6: Cleanup
  console.log('\n🧹 Step 6: Cleanup and Resource Management');
  await mcpClient.disconnectAll();
  console.log('✅ Demo completed successfully');
}

/**
 * Demonstrates individual collector usage
 */
async function demonstrateIndividualCollectors(mcpClient: MCPClient) {
  console.log('\n--- Individual Collector Demonstration ---');

  // Knowledge Graph Collector Demo
  console.log('\n🕸️  Knowledge Graph Collector:');
  const kgCollector = CollectorFactory.createDefault(CollectorType.KNOWLEDGE_GRAPH, mcpClient);
  
  // Setup event listeners
  kgCollector.on('data:collected', (event) => {
    console.log(`  📊 KG Data: ${event.data.type} - ${Object.keys(event.data.data).join(', ')}`);
  });

  kgCollector.on('collection:error', (error) => {
    console.log(`  ❌ KG Error: ${error.message}`);
  });

  try {
    await kgCollector.start();
    console.log('  ✅ Knowledge Graph Collector started');
    
    // Let it collect for a few seconds
    await sleep(3000);
    
    const kgStats = kgCollector.getStats();
    console.log(`  📈 Stats: ${kgStats.totalCollected} total, ${kgStats.eventsPerSecond.toFixed(1)} eps`);
    
    await kgCollector.stop();
    console.log('  🛑 Knowledge Graph Collector stopped');
    
  } catch (error) {
    console.log(`  ⚠️  KG Collector demo failed: ${error.message}`);
  }

  // Neural Activity Collector Demo
  console.log('\n🧠 Neural Activity Collector:');
  const neuralCollector = CollectorFactory.createHighPerformance(CollectorType.NEURAL_ACTIVITY, mcpClient);
  
  neuralCollector.on('data:collected', (event) => {
    console.log(`  🔬 Neural Data: ${event.data.type}`);
  });

  neuralCollector.on('sdr:realtime:analysis', (analysis) => {
    console.log(`  🧮 Real-time SDR Analysis: ${analysis.patternCount} patterns, ${analysis.avgSparsity.toFixed(3)} sparsity`);
  });

  try {
    await neuralCollector.start();
    console.log('  ✅ Neural Activity Collector started (high-performance mode)');
    
    await sleep(2000);
    
    const neuralStats = neuralCollector.getStats();
    console.log(`  📈 Stats: ${neuralStats.totalCollected} total, ${neuralStats.eventsPerSecond.toFixed(1)} eps`);
    
    await neuralCollector.stop();
    console.log('  🛑 Neural Activity Collector stopped');
    
  } catch (error) {
    console.log(`  ⚠️  Neural Collector demo failed: ${error.message}`);
  }

  // Cognitive Patterns Collector Demo
  console.log('\n🤔 Cognitive Patterns Collector:');
  const cognitiveCollector = CollectorFactory.createLowLatency(CollectorType.COGNITIVE_PATTERNS, mcpClient);
  
  cognitiveCollector.on('data:collected', (event) => {
    console.log(`  🧩 Cognitive Data: ${event.data.type}`);
  });

  cognitiveCollector.on('attention:switch', (event) => {
    console.log(`  👁️  Attention Switch: ${event.fromArea} → ${event.toArea}`);
  });

  try {
    await cognitiveCollector.start();
    console.log('  ✅ Cognitive Patterns Collector started (low-latency mode)');
    
    await sleep(2000);
    
    const cognitiveStats = cognitiveCollector.getStats();
    console.log(`  📈 Stats: ${cognitiveStats.totalCollected} total, ${cognitiveStats.eventsPerSecond.toFixed(1)} eps`);
    
    await cognitiveCollector.stop();
    console.log('  🛑 Cognitive Patterns Collector stopped');
    
  } catch (error) {
    console.log(`  ⚠️  Cognitive Collector demo failed: ${error.message}`);
  }

  // Memory Systems Collector Demo
  console.log('\n💾 Memory Systems Collector:');
  const memoryCollector = CollectorFactory.createMemoryOptimized(CollectorType.MEMORY_SYSTEMS, mcpClient);
  
  memoryCollector.on('data:collected', (event) => {
    console.log(`  🗃️  Memory Data: ${event.data.type}`);
  });

  memoryCollector.on('memory:patterns', (patterns) => {
    console.log(`  📊 Memory Patterns: ${patterns.length} patterns detected`);
  });

  try {
    await memoryCollector.start();
    console.log('  ✅ Memory Systems Collector started (memory-optimized mode)');
    
    await sleep(2000);
    
    const memoryStats = memoryCollector.getStats();
    console.log(`  📈 Stats: ${memoryStats.totalCollected} total, ${memoryStats.eventsPerSecond.toFixed(1)} eps`);
    
    await memoryCollector.stop();
    console.log('  🛑 Memory Systems Collector stopped');
    
  } catch (error) {
    console.log(`  ⚠️  Memory Collector demo failed: ${error.message}`);
  }
}

/**
 * Demonstrates centralized collector manager
 */
async function demonstrateCollectorManager(mcpClient: MCPClient) {
  console.log('\n--- Collector Manager Demonstration ---');

  // Create collector manager with custom configuration
  const collectorConfigs: Partial<CollectorConfigs> = {
    knowledgeGraph: {
      ...HIGH_PERFORMANCE_CONFIGS.knowledgeGraph,
      monitorTopology: true,
      topologyInterval: 5000 // 5 second topology updates
    },
    cognitivePatterns: {
      name: 'cognitive-patterns-managed',
      collectionInterval: 100,
      monitorAttention: true,
      monitorReasoning: true,
      attentionSamplingRate: 50
    },
    neuralActivity: {
      name: 'neural-activity-managed',
      collectionInterval: 50,
      neuralSamplingRate: 100,
      realTimeAnalysis: true
    },
    memorySystems: {
      name: 'memory-systems-managed',
      collectionInterval: 200,
      monitorWorkingMemory: true,
      monitorZeroCopy: true
    }
  };

  const manager = new CollectorManager(mcpClient, {
    autoStart: false,
    loadBalancingStrategy: 'adaptive',
    performanceMonitoring: true,
    healthCheckInterval: 10000, // 10 seconds
    aggregationWindow: 5000,    // 5 seconds
    collectorPriorities: {
      'neural-activity': 1,     // Highest priority
      'cognitive-patterns': 2,
      'knowledge-graph': 3,
      'memory-systems': 4       // Lowest priority
    }
  });

  // Setup manager event listeners
  manager.on('manager:initialized', (event) => {
    console.log(`  🎯 Manager initialized with ${event.collectorCount} collectors`);
  });

  manager.on('data:processed', (event) => {
    console.log(`  📦 Processed data from ${event.collector}: ${event.data.type}`);
  });

  manager.on('health:check:complete', (results) => {
    const healthStatus = results.overallHealth;
    const activeCollectors = Object.keys(results.collectorHealth).length;
    console.log(`  🏥 Health Check: ${healthStatus} (${activeCollectors} collectors)`);
    
    if (results.alerts.length > 0) {
      console.log(`  ⚠️  ${results.alerts.length} alerts detected`);
    }
  });

  manager.on('load:balanced', (event) => {
    console.log(`  ⚖️  Load balanced using ${event.strategy} strategy`);
  });

  try {
    // Initialize and start the manager
    await manager.initialize(collectorConfigs);
    await manager.startAllCollectors();
    
    console.log('  ✅ All collectors started via manager');
    
    // Monitor for 10 seconds
    console.log('  ⏱️  Monitoring for 10 seconds...');
    for (let i = 0; i < 10; i++) {
      await sleep(1000);
      const stats = manager.getStats();
      console.log(`  📊 Manager Stats [${i+1}s]: ${stats.activeCollectors} active, ${stats.totalDataPoints} points, ${stats.overallCollectionRate.toFixed(1)} rate`);
    }

    // Get final statistics
    console.log('\n  📈 Final Manager Statistics:');
    const finalStats = manager.getStats();
    displayManagerStats(finalStats);

    // Get health status
    const healthStatus = await manager.getHealthStatus();
    console.log('\n  🏥 Health Status Summary:');
    displayHealthStatus(healthStatus);

    // Aggregate data
    const aggregatedData = await manager.aggregateData();
    console.log('\n  📊 Data Aggregation Results:');
    console.log(`    Total Points: ${aggregatedData.totalPoints}`);
    console.log(`    Processing Time: ${aggregatedData.aggregationMetrics.processingTime}ms`);
    console.log(`    Compression Ratio: ${aggregatedData.aggregationMetrics.compressionRatio.toFixed(2)}`);
    console.log(`    Duplicates Removed: ${aggregatedData.aggregationMetrics.duplicatesRemoved}`);

    // Stop all collectors
    await manager.stopAllCollectors();
    console.log('  🛑 All collectors stopped via manager');

  } catch (error) {
    console.log(`  ❌ Manager demo failed: ${error.message}`);
  }
}

/**
 * Demonstrates high-performance data collection setup
 */
async function demonstrateHighPerformanceCollection(mcpClient: MCPClient) {
  console.log('\n--- High-Performance Collection Demonstration ---');
  console.log('  🎯 Target: >1000 events/second aggregate rate');

  const manager = CollectorFactory.createManager(mcpClient, 'high-performance');

  // Configure for maximum performance
  const performanceConfigs: Partial<CollectorConfigs> = {
    knowledgeGraph: {
      collectionInterval: 25,  // 40 Hz
      bufferSize: 25000,
      sampleRate: 0.9,        // 90% sampling
      autoFlush: true,
      flushInterval: 2000     // 2 second flushes
    },
    cognitivePatterns: {
      collectionInterval: 20,  // 50 Hz
      attentionSamplingRate: 200, // 200 Hz attention sampling
      bufferSize: 20000
    },
    neuralActivity: {
      collectionInterval: 10,  // 100 Hz
      neuralSamplingRate: 500, // 500 Hz neural sampling
      bufferSize: 30000,
      realTimeAnalysis: true
    },
    memorySystems: {
      collectionInterval: 50,  // 20 Hz
      memorySamplingRate: 100, // 100 Hz memory sampling
      bufferSize: 15000
    }
  };

  // Performance tracking
  let totalEvents = 0;
  let maxRate = 0;
  const rateHistory: number[] = [];

  manager.on('data:processed', () => {
    totalEvents++;
  });

  try {
    await manager.initialize(performanceConfigs);
    await manager.startAllCollectors();
    
    console.log('  🚀 High-performance collection started');
    
    // Monitor performance for 15 seconds
    console.log('  ⏱️  Performance monitoring (15 seconds)...');
    const startTime = Date.now();
    
    for (let i = 0; i < 15; i++) {
      const beforeCount = totalEvents;
      await sleep(1000);
      const afterCount = totalEvents;
      const currentRate = afterCount - beforeCount;
      
      rateHistory.push(currentRate);
      maxRate = Math.max(maxRate, currentRate);
      
      console.log(`  📊 [${String(i+1).padStart(2)}s] Rate: ${String(currentRate).padStart(4)} eps, Total: ${afterCount}, Max: ${maxRate}`);
    }

    const totalTime = (Date.now() - startTime) / 1000;
    const avgRate = totalEvents / totalTime;
    
    console.log('\n  🏁 High-Performance Results:');
    console.log(`    Total Events: ${totalEvents}`);
    console.log(`    Average Rate: ${avgRate.toFixed(1)} eps`);
    console.log(`    Peak Rate: ${maxRate} eps`);
    console.log(`    Target Met: ${avgRate > 1000 ? '✅ YES' : '❌ NO'}`);

    // Analyze rate stability
    const rateVariance = rateHistory.reduce((sum, rate) => sum + Math.pow(rate - avgRate, 2), 0) / rateHistory.length;
    const rateStdDev = Math.sqrt(rateVariance);
    console.log(`    Rate Stability: ±${rateStdDev.toFixed(1)} eps`);

    await manager.stopAllCollectors();
    
  } catch (error) {
    console.log(`  ❌ High-performance demo failed: ${error.message}`);
  }
}

/**
 * Demonstrates health monitoring and analytics
 */
async function demonstrateHealthMonitoring(mcpClient: MCPClient) {
  console.log('\n--- Health Monitoring and Analytics Demonstration ---');

  const manager = new CollectorManager(mcpClient, CollectorUtils.createHealthMonitoringConfig());
  
  // Track health events
  const healthEvents: any[] = [];
  const alerts: any[] = [];

  manager.on('health:check:complete', (results) => {
    healthEvents.push({
      timestamp: Date.now(),
      overallHealth: results.overallHealth,
      alertCount: results.alerts.length,
      collectorCount: Object.keys(results.collectorHealth).length
    });
  });

  manager.on('system:alert', (alert) => {
    alerts.push(alert);
    console.log(`  🚨 ALERT [${alert.severity}]: ${alert.message}`);
  });

  manager.on('error:recovered', (event) => {
    console.log(`  🔧 Recovered from error in ${event.collector}`);
  });

  try {
    await manager.initialize();
    await manager.startAllCollectors();
    
    console.log('  🏥 Health monitoring started');
    
    // Simulate some load and monitor health
    console.log('  ⏱️  Health monitoring (20 seconds)...');
    
    for (let i = 0; i < 20; i++) {
      await sleep(1000);
      
      // Get current health status every 5 seconds
      if (i % 5 === 4) {
        const health = await manager.getHealthStatus();
        console.log(`  🏥 [${String(i+1).padStart(2)}s] Health: ${health.overallHealth}, Alerts: ${health.alerts.length}`);
        
        // Display collector health
        for (const [name, status] of Object.entries(health.collectorHealth)) {
          const emoji = status.status === 'healthy' ? '💚' : 
                       status.status === 'warning' ? '💛' : 
                       status.status === 'critical' ? '❤️' : '🖤';
          console.log(`    ${emoji} ${name}: ${status.status} (${status.performanceScore.toFixed(2)} perf)`);
        }
      }
    }

    // Final health analysis
    console.log('\n  📊 Health Monitoring Results:');
    console.log(`    Health Checks: ${healthEvents.length}`);
    console.log(`    Total Alerts: ${alerts.length}`);
    
    if (healthEvents.length > 0) {
      const healthyChecks = healthEvents.filter(e => e.overallHealth === 'healthy').length;
      const healthRate = (healthyChecks / healthEvents.length) * 100;
      console.log(`    Health Rate: ${healthRate.toFixed(1)}%`);
    }

    // Alert analysis
    if (alerts.length > 0) {
      const alertTypes = alerts.reduce((acc, alert) => {
        acc[alert.severity] = (acc[alert.severity] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      
      console.log('    Alert Breakdown:');
      for (const [severity, count] of Object.entries(alertTypes)) {
        console.log(`      ${severity}: ${count}`);
      }
    }

    await manager.stopAllCollectors();
    
  } catch (error) {
    console.log(`  ❌ Health monitoring demo failed: ${error.message}`);
  }
}

/**
 * Helper function to display manager statistics
 */
function displayManagerStats(stats: ManagerStats) {
  console.log(`    Active Collectors: ${stats.activeCollectors}`);
  console.log(`    Total Data Points: ${stats.totalDataPoints}`);
  console.log(`    Collection Rate: ${stats.overallCollectionRate.toFixed(2)} eps`);
  console.log(`    Memory Usage: ${stats.resourceUsage.memoryUsage.toFixed(1)} MB`);
  console.log(`    CPU Usage: ${stats.resourceUsage.cpuUsage.toFixed(1)}%`);
  console.log(`    Total Errors: ${stats.errorStats.totalErrors}`);
  console.log(`    Recovery Rate: ${stats.errorStats.recoverySuccesses}/${stats.errorStats.recoveryFailures + stats.errorStats.recoverySuccesses}`);
  console.log(`    Avg Response Time: ${stats.performanceMetrics.avgResponseTime.toFixed(1)}ms`);
}

/**
 * Helper function to display health status
 */
function displayHealthStatus(health: any) {
  const statusEmoji = health.overallHealth === 'healthy' ? '💚' : 
                     health.overallHealth === 'warning' ? '💛' : '❤️';
  console.log(`    Overall Health: ${statusEmoji} ${health.overallHealth}`);
  console.log(`    Active Alerts: ${health.alerts.length}`);
  console.log(`    Recommendations: ${health.recommendations.length}`);
  
  if (health.recommendations.length > 0) {
    console.log('    📝 Recommendations:');
    health.recommendations.forEach((rec: string, i: number) => {
      console.log(`      ${i+1}. ${rec}`);
    });
  }
}

/**
 * Helper function for async delays
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Configuration validation demonstration
 */
function demonstrateConfigurationValidation() {
  console.log('\n🔧 Configuration Validation Examples:');
  
  // Test valid configuration
  const validConfig = {
    collectionInterval: 100,
    bufferSize: 5000,
    sampleRate: 0.8
  };
  
  console.log(`  Valid config: ${CollectorUtils.validateConfig(validConfig) ? '✅' : '❌'}`);
  
  // Test invalid configuration
  const invalidConfig = {
    collectionInterval: 5,  // Too low
    bufferSize: 100000,     // Too high
    sampleRate: 1.5         // Out of range
  };
  
  console.log(`  Invalid config: ${CollectorUtils.validateConfig(invalidConfig) ? '✅' : '❌'}`);
  
  // Calculate optimal intervals
  console.log(`  Optimal interval for 100 eps: ${CollectorUtils.calculateOptimalInterval(100)}ms`);
  console.log(`  Optimal interval for 1000 eps: ${CollectorUtils.calculateOptimalInterval(1000)}ms`);
  
  // Memory usage estimation
  console.log(`  Memory estimate (default): ${CollectorUtils.estimateMemoryUsage({}).toFixed(1)} MB`);
  console.log(`  Memory estimate (large): ${CollectorUtils.estimateMemoryUsage({bufferSize: 20000}).toFixed(1)} MB`);
}

/**
 * Main demo execution
 */
async function main() {
  try {
    // Configuration validation demo
    demonstrateConfigurationValidation();
    
    // Main collectors demo
    await demonstrateCollectors();
    
  } catch (error) {
    console.error('❌ Demo failed:', error);
    process.exit(1);
  }
}

// Run the demo
if (require.main === module) {
  main();
}

export { demonstrateCollectors, main as runCollectorsDemo };