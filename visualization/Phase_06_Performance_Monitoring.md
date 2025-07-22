# Phase 6: Performance Monitoring

## Overview
Phase 6 focuses on implementing comprehensive performance monitoring for LLMKG, including real-time metrics visualization, performance profiling, and bottleneck detection specific to the brain-inspired architecture.

## Objectives
1. **Real-time Performance Metrics**
   - Monitor cognitive layer performance
   - Track SDR processing speed
   - Measure MCP protocol latency
   - Visualize resource utilization

2. **Performance Profiling**
   - Identify computational bottlenecks
   - Profile memory usage patterns
   - Analyze activation spreading efficiency
   - Monitor inhibitory circuit overhead

3. **Historical Analysis**
   - Track performance trends over time
   - Compare different cognitive strategies
   - Identify degradation patterns
   - Benchmark optimization improvements

## Technical Implementation

### Performance Dashboard Component
```typescript
// src/components/PerformanceDashboard.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Line, Bar, Gauge } from '@ant-design/plots';
import { useMCPConnection } from '../hooks/useMCPConnection';
import { Card, Row, Col, Statistic, Alert } from 'antd';
import { 
  ThunderboltOutlined, 
  DatabaseOutlined, 
  ClockCircleOutlined,
  WarningOutlined 
} from '@ant-design/icons';

interface PerformanceMetrics {
  timestamp: number;
  cognitive: {
    subcortical: LayerMetrics;
    cortical: LayerMetrics;
    thalamic: LayerMetrics;
  };
  sdr: {
    creationRate: number;
    averageSparsity: number;
    overlapRatio: number;
    memoryUsage: number;
  };
  mcp: {
    messageRate: number;
    averageLatency: number;
    errorRate: number;
    queueLength: number;
  };
  system: {
    cpuUsage: number;
    memoryUsage: number;
    diskIO: number;
    networkIO: number;
  };
}

interface LayerMetrics {
  activationRate: number;
  inhibitionRate: number;
  processingLatency: number;
  throughput: number;
  errorCount: number;
}

export const PerformanceDashboard: React.FC = () => {
  const { client, connected } = useMCPConnection();
  const [metrics, setMetrics] = useState<PerformanceMetrics[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<PerformanceMetrics | null>(null);
  const [alerts, setAlerts] = useState<string[]>([]);
  const metricsBuffer = useRef<PerformanceMetrics[]>([]);

  useEffect(() => {
    if (!connected || !client) return;

    const collectMetrics = async () => {
      try {
        const response = await client.request('performance/getMetrics', {
          detailed: true,
          includeCognitive: true,
          includeSDR: true,
          includeMCP: true
        });

        const newMetrics: PerformanceMetrics = {
          timestamp: Date.now(),
          cognitive: response.cognitive,
          sdr: response.sdr,
          mcp: response.mcp,
          system: response.system
        };

        // Update current metrics
        setCurrentMetrics(newMetrics);

        // Buffer metrics for batch updates
        metricsBuffer.current.push(newMetrics);

        // Batch update every 5 data points
        if (metricsBuffer.current.length >= 5) {
          setMetrics(prev => {
            const updated = [...prev, ...metricsBuffer.current];
            // Keep only last 100 data points
            return updated.slice(-100);
          });
          metricsBuffer.current = [];
        }

        // Check for performance issues
        checkPerformanceAlerts(newMetrics);
      } catch (error) {
        console.error('Failed to collect metrics:', error);
      }
    };

    // Collect metrics every second
    const interval = setInterval(collectMetrics, 1000);
    
    // Initial collection
    collectMetrics();

    return () => clearInterval(interval);
  }, [connected, client]);

  const checkPerformanceAlerts = (metrics: PerformanceMetrics) => {
    const newAlerts: string[] = [];

    // Check cognitive layer performance
    Object.entries(metrics.cognitive).forEach(([layer, layerMetrics]) => {
      if (layerMetrics.processingLatency > 100) {
        newAlerts.push(`High latency in ${layer} layer: ${layerMetrics.processingLatency}ms`);
      }
      if (layerMetrics.errorCount > 0) {
        newAlerts.push(`Errors detected in ${layer} layer: ${layerMetrics.errorCount}`);
      }
    });

    // Check SDR performance
    if (metrics.sdr.averageSparsity < 0.02 || metrics.sdr.averageSparsity > 0.05) {
      newAlerts.push(`SDR sparsity out of optimal range: ${(metrics.sdr.averageSparsity * 100).toFixed(2)}%`);
    }

    // Check MCP performance
    if (metrics.mcp.averageLatency > 50) {
      newAlerts.push(`High MCP latency: ${metrics.mcp.averageLatency}ms`);
    }
    if (metrics.mcp.errorRate > 0.01) {
      newAlerts.push(`High MCP error rate: ${(metrics.mcp.errorRate * 100).toFixed(2)}%`);
    }

    // Check system resources
    if (metrics.system.cpuUsage > 80) {
      newAlerts.push(`High CPU usage: ${metrics.system.cpuUsage}%`);
    }
    if (metrics.system.memoryUsage > 80) {
      newAlerts.push(`High memory usage: ${metrics.system.memoryUsage}%`);
    }

    setAlerts(newAlerts);
  };

  // Prepare chart data
  const cognitiveChartData = metrics.flatMap(m => 
    Object.entries(m.cognitive).map(([layer, data]) => ({
      timestamp: new Date(m.timestamp).toLocaleTimeString(),
      layer,
      value: data.processingLatency,
      type: 'latency'
    }))
  );

  const sdrChartData = metrics.map(m => ({
    timestamp: new Date(m.timestamp).toLocaleTimeString(),
    creationRate: m.sdr.creationRate,
    sparsity: m.sdr.averageSparsity * 100,
    overlap: m.sdr.overlapRatio * 100
  }));

  const systemResourceData = metrics.map(m => ({
    timestamp: new Date(m.timestamp).toLocaleTimeString(),
    cpu: m.system.cpuUsage,
    memory: m.system.memoryUsage,
    disk: m.system.diskIO,
    network: m.system.networkIO
  }));

  return (
    <div className="performance-dashboard">
      <h2>Performance Monitoring</h2>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="alerts-section">
          {alerts.map((alert, index) => (
            <Alert
              key={index}
              message={alert}
              type="warning"
              icon={<WarningOutlined />}
              showIcon
              closable
              style={{ marginBottom: 8 }}
            />
          ))}
        </div>
      )}

      {/* Real-time Metrics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Subcortical Latency"
              value={currentMetrics?.cognitive.subcortical.processingLatency || 0}
              suffix="ms"
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: currentMetrics?.cognitive.subcortical.processingLatency > 50 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="SDR Creation Rate"
              value={currentMetrics?.sdr.creationRate || 0}
              suffix="/s"
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="MCP Message Rate"
              value={currentMetrics?.mcp.messageRate || 0}
              suffix="msg/s"
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="System Memory"
              value={currentMetrics?.system.memoryUsage || 0}
              suffix="%"
              valueStyle={{ color: currentMetrics?.system.memoryUsage > 80 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Cognitive Layer Performance */}
      <Card title="Cognitive Layer Performance" style={{ marginBottom: 24 }}>
        <Line
          data={cognitiveChartData}
          xField="timestamp"
          yField="value"
          seriesField="layer"
          yAxis={{
            title: { text: 'Processing Latency (ms)' }
          }}
          smooth
          animation={{
            appear: {
              animation: 'path-in',
              duration: 500
            }
          }}
        />
      </Card>

      {/* SDR Performance */}
      <Card title="SDR Performance Metrics" style={{ marginBottom: 24 }}>
        <Line
          data={sdrChartData}
          xField="timestamp"
          yField={['creationRate', 'sparsity', 'overlap']}
          yAxis={{
            creationRate: { title: { text: 'Creation Rate (/s)' } },
            sparsity: { title: { text: 'Sparsity (%)' } },
            overlap: { title: { text: 'Overlap (%)' } }
          }}
          geometryOptions={[
            { geometry: 'line', color: '#5B8FF9' },
            { geometry: 'line', color: '#5AD8A6' },
            { geometry: 'line', color: '#F6BD16' }
          ]}
        />
      </Card>

      {/* System Resources */}
      <Card title="System Resource Utilization" style={{ marginBottom: 24 }}>
        <Line
          data={systemResourceData}
          xField="timestamp"
          yField={['cpu', 'memory', 'disk', 'network']}
          yAxis={{
            title: { text: 'Usage (%)' },
            max: 100
          }}
          smooth
          point={{ size: 3 }}
        />
      </Card>

      {/* Performance Gauges */}
      <Row gutter={16}>
        <Col span={8}>
          <Card title="Overall System Health">
            <Gauge
              percent={calculateSystemHealth(currentMetrics)}
              range={{
                ticks: [0, 1],
                color: ['l(0) 0:#F4664A 0.5:#FAAD14 1:#30BF78']
              }}
              indicator={{
                pointer: { style: { stroke: '#D0D0D0' } },
                pin: { style: { stroke: '#D0D0D0' } }
              }}
              statistic={{
                content: {
                  formatter: ({ percent }) => `${(percent * 100).toFixed(0)}%`
                }
              }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="Cognitive Efficiency">
            <Gauge
              percent={calculateCognitiveEfficiency(currentMetrics)}
              range={{
                ticks: [0, 1],
                color: ['l(0) 0:#F4664A 0.5:#FAAD14 1:#30BF78']
              }}
              statistic={{
                content: {
                  formatter: ({ percent }) => `${(percent * 100).toFixed(0)}%`
                }
              }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="MCP Protocol Health">
            <Gauge
              percent={calculateMCPHealth(currentMetrics)}
              range={{
                ticks: [0, 1],
                color: ['l(0) 0:#F4664A 0.5:#FAAD14 1:#30BF78']
              }}
              statistic={{
                content: {
                  formatter: ({ percent }) => `${(percent * 100).toFixed(0)}%`
                }
              }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

// Helper functions
function calculateSystemHealth(metrics: PerformanceMetrics | null): number {
  if (!metrics) return 0;

  const factors = [
    1 - (metrics.system.cpuUsage / 100),
    1 - (metrics.system.memoryUsage / 100),
    metrics.mcp.errorRate < 0.01 ? 1 : 0.5,
    metrics.cognitive.subcortical.processingLatency < 50 ? 1 : 0.5
  ];

  return factors.reduce((a, b) => a + b, 0) / factors.length;
}

function calculateCognitiveEfficiency(metrics: PerformanceMetrics | null): number {
  if (!metrics) return 0;

  const avgLatency = Object.values(metrics.cognitive)
    .reduce((sum, layer) => sum + layer.processingLatency, 0) / 3;

  const avgThroughput = Object.values(metrics.cognitive)
    .reduce((sum, layer) => sum + layer.throughput, 0) / 3;

  const latencyScore = Math.max(0, 1 - (avgLatency / 100));
  const throughputScore = Math.min(1, avgThroughput / 1000);

  return (latencyScore + throughputScore) / 2;
}

function calculateMCPHealth(metrics: PerformanceMetrics | null): number {
  if (!metrics) return 0;

  const latencyScore = Math.max(0, 1 - (metrics.mcp.averageLatency / 100));
  const errorScore = Math.max(0, 1 - (metrics.mcp.errorRate * 10));
  const queueScore = Math.max(0, 1 - (metrics.mcp.queueLength / 100));

  return (latencyScore + errorScore + queueScore) / 3;
}
```

### Performance Metrics Collector (Rust)
```rust
// src/monitoring/performance_collector.rs
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use crate::cognitive::{CognitiveLayer, LayerType};
use crate::sdr::SDRProcessor;
use crate::mcp::MCPInterface;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: u64,
    pub cognitive: CognitiveMetrics,
    pub sdr: SDRMetrics,
    pub mcp: MCPMetrics,
    pub system: SystemMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMetrics {
    pub subcortical: LayerMetrics,
    pub cortical: LayerMetrics,
    pub thalamic: LayerMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetrics {
    pub activation_rate: f64,
    pub inhibition_rate: f64,
    pub processing_latency: f64,
    pub throughput: f64,
    pub error_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDRMetrics {
    pub creation_rate: f64,
    pub average_sparsity: f64,
    pub overlap_ratio: f64,
    pub memory_usage: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPMetrics {
    pub message_rate: f64,
    pub average_latency: f64,
    pub error_rate: f64,
    pub queue_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_io: f64,
    pub network_io: f64,
}

pub struct PerformanceCollector {
    cognitive_layers: Vec<Arc<Mutex<CognitiveLayer>>>,
    sdr_processor: Arc<Mutex<SDRProcessor>>,
    mcp_interface: Arc<Mutex<MCPInterface>>,
    metrics_history: Arc<Mutex<Vec<PerformanceMetrics>>>,
    collection_interval: Duration,
}

impl PerformanceCollector {
    pub fn new(
        cognitive_layers: Vec<Arc<Mutex<CognitiveLayer>>>,
        sdr_processor: Arc<Mutex<SDRProcessor>>,
        mcp_interface: Arc<Mutex<MCPInterface>>,
    ) -> Self {
        Self {
            cognitive_layers,
            sdr_processor,
            mcp_interface,
            metrics_history: Arc::new(Mutex::new(Vec::new())),
            collection_interval: Duration::from_millis(100),
        }
    }

    pub fn start_collection(&self) {
        let layers = self.cognitive_layers.clone();
        let sdr = self.sdr_processor.clone();
        let mcp = self.mcp_interface.clone();
        let history = self.metrics_history.clone();
        let interval = self.collection_interval;

        std::thread::spawn(move || {
            loop {
                let start = Instant::now();
                
                // Collect metrics
                let metrics = Self::collect_metrics(&layers, &sdr, &mcp);
                
                // Store in history
                let mut hist = history.lock().unwrap();
                hist.push(metrics.clone());
                
                // Keep only last 1000 entries
                if hist.len() > 1000 {
                    hist.remove(0);
                }
                drop(hist);
                
                // Sleep for remaining interval
                let elapsed = start.elapsed();
                if elapsed < interval {
                    std::thread::sleep(interval - elapsed);
                }
            }
        });
    }

    fn collect_metrics(
        layers: &[Arc<Mutex<CognitiveLayer>>],
        sdr: &Arc<Mutex<SDRProcessor>>,
        mcp: &Arc<Mutex<MCPInterface>>,
    ) -> PerformanceMetrics {
        let cognitive = Self::collect_cognitive_metrics(layers);
        let sdr_metrics = Self::collect_sdr_metrics(sdr);
        let mcp_metrics = Self::collect_mcp_metrics(mcp);
        let system = Self::collect_system_metrics();

        PerformanceMetrics {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            cognitive,
            sdr: sdr_metrics,
            mcp: mcp_metrics,
            system,
        }
    }

    fn collect_cognitive_metrics(layers: &[Arc<Mutex<CognitiveLayer>>]) -> CognitiveMetrics {
        let mut subcortical = LayerMetrics::default();
        let mut cortical = LayerMetrics::default();
        let mut thalamic = LayerMetrics::default();

        for layer in layers {
            let layer_guard = layer.lock().unwrap();
            let metrics = match layer_guard.layer_type() {
                LayerType::Subcortical => &mut subcortical,
                LayerType::Cortical => &mut cortical,
                LayerType::Thalamic => &mut thalamic,
            };

            *metrics = LayerMetrics {
                activation_rate: layer_guard.activation_rate(),
                inhibition_rate: layer_guard.inhibition_rate(),
                processing_latency: layer_guard.average_latency_ms(),
                throughput: layer_guard.throughput_per_second(),
                error_count: layer_guard.error_count(),
            };
        }

        CognitiveMetrics {
            subcortical,
            cortical,
            thalamic,
        }
    }

    fn collect_sdr_metrics(sdr: &Arc<Mutex<SDRProcessor>>) -> SDRMetrics {
        let processor = sdr.lock().unwrap();
        
        SDRMetrics {
            creation_rate: processor.creation_rate_per_second(),
            average_sparsity: processor.average_sparsity(),
            overlap_ratio: processor.average_overlap_ratio(),
            memory_usage: processor.memory_usage_bytes(),
        }
    }

    fn collect_mcp_metrics(mcp: &Arc<Mutex<MCPInterface>>) -> MCPMetrics {
        let interface = mcp.lock().unwrap();
        
        MCPMetrics {
            message_rate: interface.message_rate_per_second(),
            average_latency: interface.average_latency_ms(),
            error_rate: interface.error_rate(),
            queue_length: interface.queue_length(),
        }
    }

    fn collect_system_metrics() -> SystemMetrics {
        // Use system information crate for real metrics
        use sysinfo::{System, SystemExt, ProcessorExt};
        
        let mut sys = System::new_all();
        sys.refresh_all();
        
        SystemMetrics {
            cpu_usage: sys.global_processor_info().cpu_usage() as f64,
            memory_usage: (sys.used_memory() as f64 / sys.total_memory() as f64) * 100.0,
            disk_io: 0.0, // Would need additional monitoring
            network_io: 0.0, // Would need additional monitoring
        }
    }

    pub fn get_current_metrics(&self) -> Option<PerformanceMetrics> {
        self.metrics_history.lock().unwrap().last().cloned()
    }

    pub fn get_metrics_history(&self, count: usize) -> Vec<PerformanceMetrics> {
        let history = self.metrics_history.lock().unwrap();
        let start = history.len().saturating_sub(count);
        history[start..].to_vec()
    }

    pub fn analyze_performance_trends(&self) -> PerformanceTrends {
        let history = self.metrics_history.lock().unwrap();
        if history.len() < 10 {
            return PerformanceTrends::default();
        }

        let recent = &history[history.len() - 10..];
        let older = &history[history.len().saturating_sub(20)..history.len() - 10];

        PerformanceTrends {
            cognitive_latency_trend: Self::calculate_trend(
                older.iter().map(|m| m.cognitive.average_latency()).collect(),
                recent.iter().map(|m| m.cognitive.average_latency()).collect(),
            ),
            sdr_efficiency_trend: Self::calculate_trend(
                older.iter().map(|m| m.sdr.creation_rate / m.sdr.memory_usage as f64).collect(),
                recent.iter().map(|m| m.sdr.creation_rate / m.sdr.memory_usage as f64).collect(),
            ),
            mcp_reliability_trend: Self::calculate_trend(
                older.iter().map(|m| 1.0 - m.mcp.error_rate).collect(),
                recent.iter().map(|m| 1.0 - m.mcp.error_rate).collect(),
            ),
            system_load_trend: Self::calculate_trend(
                older.iter().map(|m| (m.system.cpu_usage + m.system.memory_usage) / 2.0).collect(),
                recent.iter().map(|m| (m.system.cpu_usage + m.system.memory_usage) / 2.0).collect(),
            ),
        }
    }

    fn calculate_trend(older: Vec<f64>, recent: Vec<f64>) -> TrendDirection {
        let old_avg = older.iter().sum::<f64>() / older.len() as f64;
        let new_avg = recent.iter().sum::<f64>() / recent.len() as f64;
        
        let change_percent = ((new_avg - old_avg) / old_avg) * 100.0;
        
        if change_percent > 5.0 {
            TrendDirection::Increasing(change_percent)
        } else if change_percent < -5.0 {
            TrendDirection::Decreasing(change_percent.abs())
        } else {
            TrendDirection::Stable
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub cognitive_latency_trend: TrendDirection,
    pub sdr_efficiency_trend: TrendDirection,
    pub mcp_reliability_trend: TrendDirection,
    pub system_load_trend: TrendDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing(f64),
    Decreasing(f64),
    Stable,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            cognitive_latency_trend: TrendDirection::Stable,
            sdr_efficiency_trend: TrendDirection::Stable,
            mcp_reliability_trend: TrendDirection::Stable,
            system_load_trend: TrendDirection::Stable,
        }
    }
}

impl CognitiveMetrics {
    fn average_latency(&self) -> f64 {
        (self.subcortical.processing_latency + 
         self.cortical.processing_latency + 
         self.thalamic.processing_latency) / 3.0
    }
}

impl Default for LayerMetrics {
    fn default() -> Self {
        Self {
            activation_rate: 0.0,
            inhibition_rate: 0.0,
            processing_latency: 0.0,
            throughput: 0.0,
            error_count: 0,
        }
    }
}
```

## LLMKG-Specific Features

### 1. Cognitive Layer Performance
- **Activation/Inhibition Balance**: Monitor the ratio of activation vs inhibition
- **Cross-Layer Communication**: Track latency between layers
- **Pattern Recognition Speed**: Measure how quickly patterns are identified

### 2. SDR Performance Metrics
- **Sparsity Optimization**: Ensure SDRs maintain optimal sparsity
- **Overlap Analysis**: Monitor semantic similarity preservation
- **Memory Efficiency**: Track SDR storage and retrieval performance

### 3. MCP Protocol Monitoring
- **Message Throughput**: Track messages per second
- **Protocol Overhead**: Measure MCP-specific latency
- **Error Recovery**: Monitor protocol error handling

### 4. Brain-Inspired Metrics
- **Hebbian Learning Rate**: Track synaptic weight changes
- **Attention Focus**: Measure thalamic attention switching
- **Cognitive Load**: Estimate overall system cognitive burden

## Testing Procedures

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collection() {
        let layers = create_test_layers();
        let sdr = Arc::new(Mutex::new(SDRProcessor::new()));
        let mcp = Arc::new(Mutex::new(MCPInterface::new()));
        
        let collector = PerformanceCollector::new(layers, sdr, mcp);
        let metrics = collector.get_current_metrics();
        
        assert!(metrics.is_some());
        let m = metrics.unwrap();
        assert!(m.timestamp > 0);
        assert!(m.system.cpu_usage >= 0.0 && m.system.cpu_usage <= 100.0);
    }

    #[test]
    fn test_trend_analysis() {
        let mut history = Vec::new();
        for i in 0..20 {
            let mut metrics = create_test_metrics();
            metrics.cognitive.subcortical.processing_latency = 10.0 + i as f64;
            history.push(metrics);
        }

        let collector = PerformanceCollector::new(vec![], Arc::new(Mutex::new(SDRProcessor::new())), Arc::new(Mutex::new(MCPInterface::new())));
        *collector.metrics_history.lock().unwrap() = history;

        let trends = collector.analyze_performance_trends();
        match trends.cognitive_latency_trend {
            TrendDirection::Increasing(pct) => assert!(pct > 0.0),
            _ => panic!("Expected increasing trend"),
        }
    }

    #[test]
    fn test_performance_alerts() {
        let mut metrics = create_test_metrics();
        metrics.system.cpu_usage = 90.0;
        metrics.mcp.error_rate = 0.05;

        let alerts = generate_performance_alerts(&metrics);
        assert!(alerts.len() >= 2);
        assert!(alerts.iter().any(|a| a.contains("CPU")));
        assert!(alerts.iter().any(|a| a.contains("MCP")));
    }
}
```

### Integration Tests
```typescript
// tests/performance-monitoring.test.tsx
describe('Performance Monitoring Dashboard', () => {
  it('should display real-time metrics', async () => {
    const { container } = render(<PerformanceDashboard />);
    
    await waitFor(() => {
      expect(container.querySelector('.ant-statistic-title')).toHaveTextContent('Subcortical Latency');
      expect(container.querySelector('.ant-statistic-content-value')).toBeTruthy();
    });
  });

  it('should update charts with new data', async () => {
    const { container } = render(<PerformanceDashboard />);
    
    // Wait for initial render
    await waitFor(() => {
      expect(container.querySelector('.cognitive-performance-chart')).toBeInTheDocument();
    });

    // Simulate metrics update
    act(() => {
      mockMCPClient.emit('metricsUpdate', createMockMetrics());
    });

    await waitFor(() => {
      const chartData = container.querySelector('.chart-data');
      expect(chartData).toHaveAttribute('data-point-count', '1');
    });
  });

  it('should show performance alerts', async () => {
    const { container } = render(<PerformanceDashboard />);
    
    // Simulate high latency metrics
    act(() => {
      mockMCPClient.emit('metricsUpdate', {
        ...createMockMetrics(),
        cognitive: {
          subcortical: { processingLatency: 150 }
        }
      });
    });

    await waitFor(() => {
      expect(container.querySelector('.ant-alert-warning')).toBeInTheDocument();
      expect(container.querySelector('.ant-alert-message')).toHaveTextContent('High latency');
    });
  });
});
```

## Performance Considerations

### 1. Data Collection Optimization
- Use sampling for high-frequency metrics
- Implement circular buffers for memory efficiency
- Batch metric updates to reduce overhead

### 2. Visualization Performance
- Use WebGL for large datasets
- Implement data decimation for zoom levels
- Cache computed visualizations

### 3. Alert Processing
- Use efficient threshold checking
- Implement alert deduplication
- Prioritize critical alerts

## Deliverables Checklist

- [ ] Real-time performance dashboard component
- [ ] Performance metrics collector (Rust backend)
- [ ] Cognitive layer performance visualization
- [ ] SDR efficiency monitoring
- [ ] MCP protocol performance tracking
- [ ] System resource utilization graphs
- [ ] Performance trend analysis
- [ ] Alert system for performance issues
- [ ] Historical data storage and retrieval
- [ ] Performance comparison tools
- [ ] Export functionality for performance reports
- [ ] Integration with existing monitoring systems