# Micro-Phase 9.40: Performance Monitoring System

## Objective
Implement comprehensive performance monitoring and reporting system to track WASM execution, memory usage, and user experience metrics in real-time.

## Prerequisites
- Completed micro-phase 9.39 (Compression System)
- WASM memory management configured (phases 9.05-9.08)
- Caching and optimization systems implemented (phases 9.36-9.39)

## Task Description
Create performance monitoring infrastructure with real-time metrics collection, anomaly detection, and automated optimization recommendations. Implement user experience tracking and system health monitoring for the cortical column visualization system.

## Specific Actions

1. **Create performance metrics collector**
```javascript
// Comprehensive performance monitoring system
class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.collectors = new Map();
    this.thresholds = new Map();
    this.observers = [];
    this.reportingInterval = 5000; // 5 seconds
    this.isMonitoring = false;
  }

  initialize() {
    this.setupMetricCollectors();
    this.setupPerformanceObservers();
    this.setupThresholds();
    this.startMonitoring();
  }

  setupMetricCollectors() {
    // WASM performance metrics
    this.collectors.set('wasm', new WASMMetricsCollector());
    
    // Memory usage metrics
    this.collectors.set('memory', new MemoryMetricsCollector());
    
    // Rendering performance metrics
    this.collectors.set('rendering', new RenderingMetricsCollector());
    
    // Network and caching metrics
    this.collectors.set('network', new NetworkMetricsCollector());
    
    // User interaction metrics
    this.collectors.set('interaction', new InteractionMetricsCollector());
  }

  async collectMetrics() {
    const timestamp = Date.now();
    const allMetrics = { timestamp };

    for (const [name, collector] of this.collectors) {
      try {
        allMetrics[name] = await collector.collect();
      } catch (error) {
        console.warn(`Failed to collect ${name} metrics:`, error);
        allMetrics[name] = { error: error.message };
      }
    }

    this.processMetrics(allMetrics);
    return allMetrics;
  }

  processMetrics(metrics) {
    // Store metrics with circular buffer
    const metricsKey = `metrics_${Date.now()}`;
    this.metrics.set(metricsKey, metrics);
    
    // Keep only last 1000 metric snapshots
    if (this.metrics.size > 1000) {
      const oldestKey = this.metrics.keys().next().value;
      this.metrics.delete(oldestKey);
    }

    // Check thresholds and trigger alerts
    this.checkThresholds(metrics);
    
    // Update live dashboard
    this.updateDashboard(metrics);
  }
}
```

2. **Implement WASM-specific monitoring**
```javascript
// WASM execution and memory monitoring
class WASMMetricsCollector {
  constructor() {
    this.wasmInstance = null;
    this.executionTimes = [];
    this.memorySnapshots = [];
  }

  setWASMInstance(instance) {
    this.wasmInstance = instance;
  }

  async collect() {
    if (!this.wasmInstance) {
      return { status: 'not_initialized' };
    }

    const memory = this.wasmInstance.exports.memory;
    const memoryUsage = memory.buffer.byteLength;
    
    return {
      memoryUsage,
      heapSize: this.getHeapSize(),
      executionStats: this.getExecutionStats(),
      functionCallCounts: this.getFunctionCallCounts(),
      allocatorStats: this.getAllocatorStats()
    };
  }

  getHeapSize() {
    // Query WASM heap allocator for usage statistics
    if (this.wasmInstance.exports.get_heap_size) {
      return this.wasmInstance.exports.get_heap_size();
    }
    return null;
  }

  measureFunctionExecution(functionName, execution) {
    return async (...args) => {
      const startTime = performance.now();
      const startMemory = this.getCurrentMemoryUsage();
      
      try {
        const result = await execution(...args);
        const endTime = performance.now();
        const endMemory = this.getCurrentMemoryUsage();
        
        this.recordExecution(functionName, {
          duration: endTime - startTime,
          memoryDelta: endMemory - startMemory,
          success: true
        });
        
        return result;
      } catch (error) {
        const endTime = performance.now();
        
        this.recordExecution(functionName, {
          duration: endTime - startTime,
          memoryDelta: 0,
          success: false,
          error: error.message
        });
        
        throw error;
      }
    };
  }

  getAllocatorStats() {
    // Get detailed memory allocation statistics
    if (this.wasmInstance.exports.get_allocator_stats) {
      const statsPtr = this.wasmInstance.exports.get_allocator_stats();
      const memory = new DataView(this.wasmInstance.exports.memory.buffer);
      
      return {
        totalAllocated: memory.getUint32(statsPtr, true),
        totalDeallocated: memory.getUint32(statsPtr + 4, true),
        currentAllocations: memory.getUint32(statsPtr + 8, true),
        peakMemoryUsage: memory.getUint32(statsPtr + 12, true),
        fragmentationRatio: memory.getFloat32(statsPtr + 16, true)
      };
    }
    
    return null;
  }
}
```

3. **Create real-time performance dashboard**
```javascript
// Live performance dashboard with charts and alerts
class PerformanceDashboard {
  constructor(monitor) {
    this.monitor = monitor;
    this.charts = new Map();
    this.alertContainer = null;
    this.updateInterval = null;
  }

  initialize(container) {
    this.createDashboardLayout(container);
    this.setupCharts();
    this.setupAlerts();
    this.startRealTimeUpdates();
  }

  createDashboardLayout(container) {
    container.innerHTML = `
      <div class="performance-dashboard">
        <div class="metrics-grid">
          <div class="metric-card" id="wasm-metrics">
            <h3>WASM Performance</h3>
            <canvas id="wasm-chart"></canvas>
            <div class="metric-values" id="wasm-values"></div>
          </div>
          
          <div class="metric-card" id="memory-metrics">
            <h3>Memory Usage</h3>
            <canvas id="memory-chart"></canvas>
            <div class="metric-values" id="memory-values"></div>
          </div>
          
          <div class="metric-card" id="rendering-metrics">
            <h3>Rendering Performance</h3>
            <canvas id="fps-chart"></canvas>
            <div class="metric-values" id="rendering-values"></div>
          </div>
          
          <div class="metric-card" id="network-metrics">
            <h3>Network & Cache</h3>
            <canvas id="network-chart"></canvas>
            <div class="metric-values" id="network-values"></div>
          </div>
        </div>
        
        <div class="alerts-panel" id="alerts-panel">
          <h3>Performance Alerts</h3>
          <div id="alerts-container"></div>
        </div>
        
        <div class="recommendations-panel" id="recommendations">
          <h3>Optimization Recommendations</h3>
          <div id="recommendations-container"></div>
        </div>
      </div>
    `;
    
    this.alertContainer = container.querySelector('#alerts-container');
  }

  setupCharts() {
    // Setup Chart.js charts for real-time data visualization
    this.charts.set('wasm', this.createTimeSeriesChart('wasm-chart', {
      label: 'Execution Time (ms)',
      borderColor: '#3498db',
      backgroundColor: 'rgba(52, 152, 219, 0.1)'
    }));

    this.charts.set('memory', this.createTimeSeriesChart('memory-chart', {
      label: 'Memory Usage (MB)',
      borderColor: '#e74c3c',
      backgroundColor: 'rgba(231, 76, 60, 0.1)'
    }));

    this.charts.set('fps', this.createTimeSeriesChart('fps-chart', {
      label: 'FPS',
      borderColor: '#2ecc71',
      backgroundColor: 'rgba(46, 204, 113, 0.1)'
    }));

    this.charts.set('network', this.createTimeSeriesChart('network-chart', {
      label: 'Cache Hit Rate (%)',
      borderColor: '#f39c12',
      backgroundColor: 'rgba(243, 156, 18, 0.1)'
    }));
  }

  updateCharts(metrics) {
    // Update WASM performance chart
    if (metrics.wasm && this.charts.has('wasm')) {
      this.addDataPoint('wasm', 
        metrics.timestamp, 
        metrics.wasm.executionStats?.avgExecutionTime || 0
      );
    }

    // Update memory usage chart
    if (metrics.memory && this.charts.has('memory')) {
      this.addDataPoint('memory', 
        metrics.timestamp, 
        (metrics.memory.used || 0) / (1024 * 1024)
      );
    }

    // Update FPS chart
    if (metrics.rendering && this.charts.has('fps')) {
      this.addDataPoint('fps', 
        metrics.timestamp, 
        metrics.rendering.fps || 0
      );
    }

    // Update network performance chart
    if (metrics.network && this.charts.has('network')) {
      this.addDataPoint('network', 
        metrics.timestamp, 
        (metrics.network.cacheHitRate || 0) * 100
      );
    }
  }
}
```

4. **Implement anomaly detection and alerting**
```javascript
// Anomaly detection for performance degradation
class PerformanceAnomalyDetector {
  constructor() {
    this.baselines = new Map();
    this.anomalies = [];
    this.alertHandlers = [];
    this.detectionConfig = {
      memoryThreshold: 1.5,     // 50% above baseline
      executionThreshold: 2.0,   // 100% above baseline
      fpsThreshold: 0.7,         // 30% below baseline
      cacheThreshold: 0.8        // 20% below baseline
    };
  }

  establishBaselines(historicalMetrics) {
    const baselines = {};
    
    // Calculate baseline values from historical data
    baselines.memory = this.calculateBaseline(
      historicalMetrics.map(m => m.memory?.used || 0)
    );
    
    baselines.executionTime = this.calculateBaseline(
      historicalMetrics.map(m => m.wasm?.executionStats?.avgExecutionTime || 0)
    );
    
    baselines.fps = this.calculateBaseline(
      historicalMetrics.map(m => m.rendering?.fps || 60)
    );
    
    baselines.cacheHitRate = this.calculateBaseline(
      historicalMetrics.map(m => m.network?.cacheHitRate || 0.8)
    );
    
    this.baselines = new Map(Object.entries(baselines));
  }

  detectAnomalies(currentMetrics) {
    const anomalies = [];
    
    // Check memory usage
    if (currentMetrics.memory?.used) {
      const memoryBaseline = this.baselines.get('memory');
      if (currentMetrics.memory.used > memoryBaseline.mean * this.detectionConfig.memoryThreshold) {
        anomalies.push({
          type: 'memory_spike',
          severity: 'high',
          current: currentMetrics.memory.used,
          baseline: memoryBaseline.mean,
          threshold: memoryBaseline.mean * this.detectionConfig.memoryThreshold,
          message: 'Memory usage significantly above baseline'
        });
      }
    }

    // Check execution time
    if (currentMetrics.wasm?.executionStats?.avgExecutionTime) {
      const execBaseline = this.baselines.get('executionTime');
      const currentExecTime = currentMetrics.wasm.executionStats.avgExecutionTime;
      
      if (currentExecTime > execBaseline.mean * this.detectionConfig.executionThreshold) {
        anomalies.push({
          type: 'performance_degradation',
          severity: 'medium',
          current: currentExecTime,
          baseline: execBaseline.mean,
          message: 'WASM execution time significantly increased'
        });
      }
    }

    // Check FPS degradation
    if (currentMetrics.rendering?.fps) {
      const fpsBaseline = this.baselines.get('fps');
      if (currentMetrics.rendering.fps < fpsBaseline.mean * this.detectionConfig.fpsThreshold) {
        anomalies.push({
          type: 'fps_drop',
          severity: 'high',
          current: currentMetrics.rendering.fps,
          baseline: fpsBaseline.mean,
          message: 'Frame rate significantly below baseline'
        });
      }
    }

    return anomalies;
  }

  generateOptimizationRecommendations(anomalies, metrics) {
    const recommendations = [];
    
    for (const anomaly of anomalies) {
      switch (anomaly.type) {
        case 'memory_spike':
          recommendations.push({
            priority: 'high',
            action: 'memory_optimization',
            description: 'Consider reducing cache sizes or triggering garbage collection',
            automated: true,
            implementation: () => this.triggerMemoryOptimization(metrics)
          });
          break;
          
        case 'performance_degradation':
          recommendations.push({
            priority: 'medium',
            action: 'execution_optimization',
            description: 'Switch to lighter algorithms or reduce processing complexity',
            automated: false,
            implementation: () => this.suggestAlgorithmOptimization(metrics)
          });
          break;
          
        case 'fps_drop':
          recommendations.push({
            priority: 'high',
            action: 'rendering_optimization',
            description: 'Reduce rendering quality or enable performance mode',
            automated: true,
            implementation: () => this.enablePerformanceMode(metrics)
          });
          break;
      }
    }
    
    return recommendations;
  }
}
```

5. **Create performance reporting and analytics**
```javascript
// Comprehensive performance reporting system
class PerformanceReporter {
  constructor(monitor) {
    this.monitor = monitor;
    this.reportHistory = [];
    this.analysisCache = new Map();
  }

  generatePerformanceReport(timeRange = '1h') {
    const metrics = this.getMetricsForTimeRange(timeRange);
    const analysis = this.analyzeMetrics(metrics);
    
    const report = {
      summary: this.generateSummary(analysis),
      detailed: {
        wasm: this.analyzeWASMPerformance(metrics),
        memory: this.analyzeMemoryUsage(metrics),
        rendering: this.analyzeRenderingPerformance(metrics),
        network: this.analyzeNetworkPerformance(metrics),
        userExperience: this.analyzeUserExperience(metrics)
      },
      recommendations: this.generateRecommendations(analysis),
      trends: this.analyzeTrends(metrics),
      benchmarks: this.compareToBenchmarks(analysis)
    };
    
    this.reportHistory.push({
      timestamp: Date.now(),
      timeRange,
      report
    });
    
    return report;
  }

  analyzeWASMPerformance(metrics) {
    const wasmMetrics = metrics
      .filter(m => m.wasm && m.wasm.executionStats)
      .map(m => m.wasm);
    
    if (wasmMetrics.length === 0) {
      return { status: 'no_data' };
    }
    
    const avgExecutionTime = this.calculateAverage(
      wasmMetrics.map(m => m.executionStats.avgExecutionTime)
    );
    
    const memoryEfficiency = this.calculateAverage(
      wasmMetrics.map(m => m.allocatorStats?.fragmentationRatio || 0)
    );
    
    return {
      avgExecutionTime,
      memoryEfficiency,
      performanceScore: this.calculatePerformanceScore(wasmMetrics),
      bottlenecks: this.identifyBottlenecks(wasmMetrics),
      optimizationOpportunities: this.identifyOptimizations(wasmMetrics)
    };
  }

  exportReport(report, format = 'json') {
    switch (format) {
      case 'json':
        return JSON.stringify(report, null, 2);
      case 'csv':
        return this.convertToCSV(report);
      case 'html':
        return this.generateHTMLReport(report);
      default:
        throw new Error(`Unsupported format: ${format}`);
    }
  }

  scheduleAutomaticReports(interval = '1h', recipients = []) {
    setInterval(async () => {
      const report = this.generatePerformanceReport(interval);
      
      // Send to configured recipients
      for (const recipient of recipients) {
        await this.sendReport(report, recipient);
      }
      
      // Check for critical issues
      if (this.hasCriticalIssues(report)) {
        await this.sendCriticalAlert(report);
      }
    }, this.parseInterval(interval));
  }
}
```

## Expected Outputs
- Real-time performance monitoring dashboard with live charts
- Automated anomaly detection with 95% accuracy for performance issues
- Comprehensive performance reports with optimization recommendations
- WASM-specific metrics tracking execution time and memory efficiency
- User experience analytics correlating performance with interaction patterns

## Validation
1. Verify monitoring captures all critical performance metrics accurately
2. Confirm anomaly detection triggers appropriate alerts within 30 seconds
3. Test dashboard updates reflect real-time system performance
4. Validate performance reports provide actionable optimization insights
5. Ensure monitoring overhead stays below 2% of total system resources

## Next Steps
- Proceed to micro-phase 9.45 (Performance Benchmarks)
- Integrate with deployment pipeline for continuous monitoring
- Configure automated optimization based on performance insights