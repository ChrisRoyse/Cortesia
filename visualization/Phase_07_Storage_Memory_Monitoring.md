# Phase 7: Storage & Memory Monitoring

## Overview
Phase 7 focuses on comprehensive monitoring of LLMKG's storage and memory systems, including SDR storage patterns, knowledge graph memory usage, cognitive layer memory allocation, and zero-copy optimization tracking.

## Objectives
1. **SDR Storage Monitoring**
   - Track SDR creation and storage patterns
   - Monitor sparsity and compression efficiency
   - Visualize SDR overlap and similarity metrics
   - Analyze memory footprint per SDR

2. **Knowledge Graph Memory**
   - Monitor entity and relation storage
   - Track graph expansion over time
   - Visualize memory hotspots
   - Analyze query performance impact

3. **Cognitive Layer Memory**
   - Track per-layer memory allocation
   - Monitor activation pattern storage
   - Visualize inhibitory circuit memory usage
   - Analyze memory fragmentation

4. **Zero-Copy Performance**
   - Track zero-copy operation efficiency
   - Monitor memory page usage
   - Visualize copy-on-write patterns
   - Analyze performance gains

## Technical Implementation

### Memory Monitoring Dashboard
```typescript
// src/components/MemoryMonitoring.tsx
import React, { useState, useEffect, useRef } from 'react';
import { TreeMap, Sankey, Waterfall, DualAxes } from '@ant-design/plots';
import { Card, Row, Col, Statistic, Progress, Table, Tag } from 'antd';
import { DatabaseOutlined, CloudServerOutlined, ThunderboltOutlined } from '@ant-design/icons';
import * as d3 from 'd3';
import { useMCPConnection } from '../hooks/useMCPConnection';

interface MemoryMetrics {
  timestamp: number;
  sdr: {
    totalSDRs: number;
    activeSDRs: number;
    totalMemory: number;
    avgSparsity: number;
    compressionRatio: number;
    fragmentationLevel: number;
  };
  knowledgeGraph: {
    entities: number;
    relations: number;
    totalMemory: number;
    indexMemory: number;
    cacheMemory: number;
    queryCache: {
      size: number;
      hitRate: number;
    };
  };
  cognitive: {
    subcortical: LayerMemory;
    cortical: LayerMemory;
    thalamic: LayerMemory;
  };
  zeroCopy: {
    activeRegions: number;
    sharedPages: number;
    copyOnWriteEvents: number;
    memorySaved: number;
    efficiency: number;
  };
  system: {
    totalAllocated: number;
    totalAvailable: number;
    swapUsage: number;
    pageCache: number;
  };
}

interface LayerMemory {
  allocated: number;
  active: number;
  cached: number;
  patterns: number;
  connections: number;
}

export const MemoryMonitoring: React.FC = () => {
  const { client, connected } = useMCPConnection();
  const [metrics, setMetrics] = useState<MemoryMetrics | null>(null);
  const [history, setHistory] = useState<MemoryMetrics[]>([]);
  const [memoryMap, setMemoryMap] = useState<any[]>([]);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!connected || !client) return;

    const fetchMemoryMetrics = async () => {
      try {
        const response = await client.request('memory/getMetrics', {
          detailed: true,
          includeHistory: true
        });

        setMetrics(response.current);
        setHistory(response.history || []);
        setMemoryMap(response.memoryMap || []);
      } catch (error) {
        console.error('Failed to fetch memory metrics:', error);
      }
    };

    // Initial fetch
    fetchMemoryMetrics();

    // Set up real-time updates
    const interval = setInterval(fetchMemoryMetrics, 2000);

    return () => clearInterval(interval);
  }, [connected, client]);

  // Prepare TreeMap data for memory allocation
  const memoryTreeMapData = {
    name: 'LLMKG Memory',
    children: [
      {
        name: 'SDR Storage',
        value: metrics?.sdr.totalMemory || 0,
        children: [
          { name: 'Active SDRs', value: (metrics?.sdr.totalMemory || 0) * 0.6 },
          { name: 'Archived SDRs', value: (metrics?.sdr.totalMemory || 0) * 0.3 },
          { name: 'Metadata', value: (metrics?.sdr.totalMemory || 0) * 0.1 }
        ]
      },
      {
        name: 'Knowledge Graph',
        value: metrics?.knowledgeGraph.totalMemory || 0,
        children: [
          { name: 'Entities', value: (metrics?.knowledgeGraph.totalMemory || 0) * 0.4 },
          { name: 'Relations', value: (metrics?.knowledgeGraph.totalMemory || 0) * 0.3 },
          { name: 'Indexes', value: metrics?.knowledgeGraph.indexMemory || 0 },
          { name: 'Cache', value: metrics?.knowledgeGraph.cacheMemory || 0 }
        ]
      },
      {
        name: 'Cognitive Layers',
        value: Object.values(metrics?.cognitive || {}).reduce((sum, layer) => sum + layer.allocated, 0),
        children: [
          {
            name: 'Subcortical',
            value: metrics?.cognitive.subcortical.allocated || 0,
            children: [
              { name: 'Patterns', value: metrics?.cognitive.subcortical.patterns || 0 },
              { name: 'Connections', value: metrics?.cognitive.subcortical.connections || 0 }
            ]
          },
          {
            name: 'Cortical',
            value: metrics?.cognitive.cortical.allocated || 0,
            children: [
              { name: 'Patterns', value: metrics?.cognitive.cortical.patterns || 0 },
              { name: 'Connections', value: metrics?.cognitive.cortical.connections || 0 }
            ]
          },
          {
            name: 'Thalamic',
            value: metrics?.cognitive.thalamic.allocated || 0,
            children: [
              { name: 'Patterns', value: metrics?.cognitive.thalamic.patterns || 0 },
              { name: 'Connections', value: metrics?.cognitive.thalamic.connections || 0 }
            ]
          }
        ]
      }
    ]
  };

  // Prepare Sankey diagram for memory flow
  const memorySankeyData = {
    nodes: [
      { name: 'Total Memory' },
      { name: 'SDR Storage' },
      { name: 'Knowledge Graph' },
      { name: 'Cognitive Layers' },
      { name: 'Zero-Copy Shared' },
      { name: 'Active SDRs' },
      { name: 'Cached Patterns' },
      { name: 'Entity Storage' },
      { name: 'Relation Storage' },
      { name: 'Query Cache' }
    ],
    links: [
      { source: 0, target: 1, value: metrics?.sdr.totalMemory || 0 },
      { source: 0, target: 2, value: metrics?.knowledgeGraph.totalMemory || 0 },
      { source: 0, target: 3, value: Object.values(metrics?.cognitive || {}).reduce((sum, layer) => sum + layer.allocated, 0) },
      { source: 0, target: 4, value: metrics?.zeroCopy.memorySaved || 0 },
      { source: 1, target: 5, value: (metrics?.sdr.totalMemory || 0) * 0.6 },
      { source: 1, target: 6, value: (metrics?.sdr.totalMemory || 0) * 0.4 },
      { source: 2, target: 7, value: (metrics?.knowledgeGraph.totalMemory || 0) * 0.4 },
      { source: 2, target: 8, value: (metrics?.knowledgeGraph.totalMemory || 0) * 0.3 },
      { source: 2, target: 9, value: (metrics?.knowledgeGraph.totalMemory || 0) * 0.3 }
    ]
  };

  // SDR Memory Visualization
  useEffect(() => {
    if (!svgRef.current || !metrics) return;

    const svg = d3.select(svgRef.current);
    const width = 600;
    const height = 400;

    svg.selectAll('*').remove();

    // Create SDR memory visualization
    const sdrData = Array.from({ length: 100 }, (_, i) => ({
      id: i,
      active: i < (metrics.sdr.activeSDRs / metrics.sdr.totalSDRs) * 100,
      memory: Math.random() * 1000,
      sparsity: metrics.sdr.avgSparsity + (Math.random() - 0.5) * 0.01
    }));

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 1000]);

    const cellSize = 20;
    const cells = svg.selectAll('rect')
      .data(sdrData)
      .enter()
      .append('rect')
      .attr('x', d => (d.id % 10) * (cellSize + 2))
      .attr('y', d => Math.floor(d.id / 10) * (cellSize + 2))
      .attr('width', cellSize)
      .attr('height', cellSize)
      .attr('fill', d => d.active ? colorScale(d.memory) : '#333')
      .attr('stroke', d => d.active ? '#fff' : '#666')
      .attr('stroke-width', 1)
      .attr('opacity', d => d.active ? 1 : 0.3);

    // Add tooltip
    cells.append('title')
      .text(d => `SDR ${d.id}\nMemory: ${d.memory.toFixed(0)} bytes\nSparsity: ${(d.sparsity * 100).toFixed(2)}%`);

    // Add legend
    const legendWidth = 200;
    const legendHeight = 20;
    
    const legendScale = d3.scaleLinear()
      .domain([0, 1000])
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => `${d}B`);

    const legend = svg.append('g')
      .attr('transform', `translate(${width - legendWidth - 20}, ${height - 40})`);

    legend.append('g')
      .attr('transform', `translate(0, ${legendHeight})`)
      .call(legendAxis);

    const gradientId = 'memory-gradient';
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', gradientId)
      .attr('x1', '0%')
      .attr('x2', '100%');

    gradient.selectAll('stop')
      .data(d3.range(0, 1.1, 0.1))
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => colorScale(d * 1000));

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('fill', `url(#${gradientId})`);

  }, [metrics]);

  // Time series data for memory trends
  const memoryTrendData = history.map(m => ({
    timestamp: new Date(m.timestamp).toLocaleTimeString(),
    sdr: m.sdr.totalMemory / 1024 / 1024, // Convert to MB
    knowledgeGraph: m.knowledgeGraph.totalMemory / 1024 / 1024,
    cognitive: Object.values(m.cognitive).reduce((sum, layer) => sum + layer.allocated, 0) / 1024 / 1024,
    zeroCopy: m.zeroCopy.memorySaved / 1024 / 1024
  }));

  return (
    <div className="memory-monitoring">
      <h2>Storage & Memory Monitoring</h2>

      {/* Summary Statistics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Memory Usage"
              value={((metrics?.system.totalAllocated || 0) / 1024 / 1024).toFixed(2)}
              suffix="MB"
              prefix={<DatabaseOutlined />}
            />
            <Progress 
              percent={(metrics?.system.totalAllocated || 0) / (metrics?.system.totalAvailable || 1) * 100} 
              strokeColor="#1890ff"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Active SDRs"
              value={metrics?.sdr.activeSDRs || 0}
              suffix={`/ ${metrics?.sdr.totalSDRs || 0}`}
              prefix={<CloudServerOutlined />}
            />
            <div style={{ fontSize: '12px', color: '#666' }}>
              Compression: {((metrics?.sdr.compressionRatio || 0) * 100).toFixed(1)}%
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Knowledge Graph Size"
              value={metrics?.knowledgeGraph.entities || 0}
              suffix="entities"
            />
            <div style={{ fontSize: '12px', color: '#666' }}>
              {metrics?.knowledgeGraph.relations || 0} relations
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Zero-Copy Efficiency"
              value={((metrics?.zeroCopy.efficiency || 0) * 100).toFixed(1)}
              suffix="%"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: metrics?.zeroCopy.efficiency > 0.8 ? '#3f8600' : '#cf1322' }}
            />
            <div style={{ fontSize: '12px', color: '#666' }}>
              Saved: {((metrics?.zeroCopy.memorySaved || 0) / 1024 / 1024).toFixed(1)} MB
            </div>
          </Card>
        </Col>
      </Row>

      {/* Memory Allocation TreeMap */}
      <Card title="Memory Allocation Breakdown" style={{ marginBottom: 24 }}>
        <TreeMap
          data={memoryTreeMapData}
          colorField="name"
          animation={{
            appear: {
              animation: 'zoom-in',
              duration: 500
            }
          }}
          label={{
            style: {
              fontSize: 12,
              fill: 'white'
            }
          }}
          tooltip={{
            formatter: (datum) => ({
              name: datum.name,
              value: `${(datum.value / 1024 / 1024).toFixed(2)} MB`
            })
          }}
        />
      </Card>

      <Row gutter={16}>
        <Col span={12}>
          {/* SDR Memory Visualization */}
          <Card title="SDR Memory Map" style={{ marginBottom: 24 }}>
            <svg ref={svgRef} width={600} height={400} />
            <div style={{ marginTop: 16 }}>
              <Tag color="green">Active SDRs: {metrics?.sdr.activeSDRs || 0}</Tag>
              <Tag color="blue">Avg Sparsity: {((metrics?.sdr.avgSparsity || 0) * 100).toFixed(2)}%</Tag>
              <Tag color="orange">Fragmentation: {((metrics?.sdr.fragmentationLevel || 0) * 100).toFixed(1)}%</Tag>
            </div>
          </Card>
        </Col>

        <Col span={12}>
          {/* Memory Flow Sankey */}
          <Card title="Memory Flow Visualization" style={{ marginBottom: 24 }}>
            <Sankey
              sourceField="source"
              targetField="target"
              weightField="value"
              nodeAlign="justify"
              nodePaddingRatio={0.03}
              data={memorySankeyData}
              label={{
                formatter: (datum) => datum.name,
                style: {
                  fill: '#666',
                  fontSize: 10
                }
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Memory Trends */}
      <Card title="Memory Usage Trends" style={{ marginBottom: 24 }}>
        <DualAxes
          data={[memoryTrendData, memoryTrendData]}
          xField="timestamp"
          yField={["sdr", "knowledgeGraph"]}
          geometryOptions={[
            {
              geometry: 'line',
              color: '#5B8FF9',
            },
            {
              geometry: 'line',
              color: '#5AD8A6',
              yAxisIndex: 1
            }
          ]}
          yAxis={{
            sdr: {
              title: { text: 'SDR Memory (MB)' }
            },
            knowledgeGraph: {
              title: { text: 'Knowledge Graph (MB)' }
            }
          }}
          annotations={[
            {
              type: 'regionFilter',
              start: ['min', 'median'],
              end: ['max', '0'],
              color: '#F4664A',
              opacity: 0.1
            }
          ]}
        />
      </Card>

      {/* Detailed Memory Table */}
      <Card title="Detailed Memory Breakdown">
        <Table
          dataSource={[
            {
              key: 'sdr',
              component: 'SDR Storage',
              allocated: metrics?.sdr.totalMemory || 0,
              active: (metrics?.sdr.totalMemory || 0) * 0.7,
              cached: (metrics?.sdr.totalMemory || 0) * 0.3,
              efficiency: metrics?.sdr.compressionRatio || 0
            },
            {
              key: 'kg',
              component: 'Knowledge Graph',
              allocated: metrics?.knowledgeGraph.totalMemory || 0,
              active: (metrics?.knowledgeGraph.totalMemory || 0) * 0.8,
              cached: metrics?.knowledgeGraph.cacheMemory || 0,
              efficiency: metrics?.knowledgeGraph.queryCache.hitRate || 0
            },
            ...Object.entries(metrics?.cognitive || {}).map(([layer, data]) => ({
              key: layer,
              component: `${layer.charAt(0).toUpperCase() + layer.slice(1)} Layer`,
              allocated: data.allocated,
              active: data.active,
              cached: data.cached,
              efficiency: data.active / data.allocated
            }))
          ]}
          columns={[
            {
              title: 'Component',
              dataIndex: 'component',
              key: 'component',
            },
            {
              title: 'Allocated (MB)',
              dataIndex: 'allocated',
              key: 'allocated',
              render: (value) => (value / 1024 / 1024).toFixed(2),
              sorter: (a, b) => a.allocated - b.allocated
            },
            {
              title: 'Active (MB)',
              dataIndex: 'active',
              key: 'active',
              render: (value) => (value / 1024 / 1024).toFixed(2)
            },
            {
              title: 'Cached (MB)',
              dataIndex: 'cached',
              key: 'cached',
              render: (value) => (value / 1024 / 1024).toFixed(2)
            },
            {
              title: 'Efficiency',
              dataIndex: 'efficiency',
              key: 'efficiency',
              render: (value) => (
                <Progress 
                  percent={value * 100} 
                  size="small" 
                  format={percent => `${percent.toFixed(1)}%`}
                />
              )
            }
          ]}
          pagination={false}
        />
      </Card>
    </div>
  );
};
```

### Memory Monitoring Backend
```rust
// src/monitoring/memory_monitor.rs
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};
use crate::sdr::{SDR, SDRStorage};
use crate::knowledge_graph::KnowledgeGraph;
use crate::cognitive::CognitiveLayer;
use crate::zero_copy::ZeroCopyManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub timestamp: u64,
    pub sdr: SDRMemoryMetrics,
    pub knowledge_graph: KnowledgeGraphMemoryMetrics,
    pub cognitive: CognitiveMemoryMetrics,
    pub zero_copy: ZeroCopyMetrics,
    pub system: SystemMemoryMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDRMemoryMetrics {
    pub total_sdrs: usize,
    pub active_sdrs: usize,
    pub total_memory: usize,
    pub avg_sparsity: f64,
    pub compression_ratio: f64,
    pub fragmentation_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphMemoryMetrics {
    pub entities: usize,
    pub relations: usize,
    pub total_memory: usize,
    pub index_memory: usize,
    pub cache_memory: usize,
    pub query_cache: QueryCacheMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheMetrics {
    pub size: usize,
    pub hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMemoryMetrics {
    pub subcortical: LayerMemoryMetrics,
    pub cortical: LayerMemoryMetrics,
    pub thalamic: LayerMemoryMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMemoryMetrics {
    pub allocated: usize,
    pub active: usize,
    pub cached: usize,
    pub patterns: usize,
    pub connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCopyMetrics {
    pub active_regions: usize,
    pub shared_pages: usize,
    pub copy_on_write_events: usize,
    pub memory_saved: usize,
    pub efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMemoryMetrics {
    pub total_allocated: usize,
    pub total_available: usize,
    pub swap_usage: usize,
    pub page_cache: usize,
}

pub struct MemoryMonitor {
    sdr_storage: Arc<Mutex<SDRStorage>>,
    knowledge_graph: Arc<Mutex<KnowledgeGraph>>,
    cognitive_layers: Vec<Arc<Mutex<CognitiveLayer>>>,
    zero_copy_manager: Arc<Mutex<ZeroCopyManager>>,
    metrics_history: Arc<Mutex<Vec<MemoryMetrics>>>,
}

impl MemoryMonitor {
    pub fn new(
        sdr_storage: Arc<Mutex<SDRStorage>>,
        knowledge_graph: Arc<Mutex<KnowledgeGraph>>,
        cognitive_layers: Vec<Arc<Mutex<CognitiveLayer>>>,
        zero_copy_manager: Arc<Mutex<ZeroCopyManager>>,
    ) -> Self {
        Self {
            sdr_storage,
            knowledge_graph,
            cognitive_layers,
            zero_copy_manager,
            metrics_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn collect_metrics(&self) -> MemoryMetrics {
        let sdr = self.collect_sdr_metrics();
        let knowledge_graph = self.collect_knowledge_graph_metrics();
        let cognitive = self.collect_cognitive_metrics();
        let zero_copy = self.collect_zero_copy_metrics();
        let system = self.collect_system_metrics();

        let metrics = MemoryMetrics {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            sdr,
            knowledge_graph,
            cognitive,
            zero_copy,
            system,
        };

        // Store in history
        let mut history = self.metrics_history.lock().unwrap();
        history.push(metrics.clone());
        if history.len() > 1000 {
            history.remove(0);
        }

        metrics
    }

    fn collect_sdr_metrics(&self) -> SDRMemoryMetrics {
        let storage = self.sdr_storage.lock().unwrap();
        
        let total_sdrs = storage.total_count();
        let active_sdrs = storage.active_count();
        let total_memory = storage.memory_usage();
        let avg_sparsity = storage.average_sparsity();
        let compression_ratio = storage.compression_ratio();
        let fragmentation_level = storage.fragmentation_level();

        SDRMemoryMetrics {
            total_sdrs,
            active_sdrs,
            total_memory,
            avg_sparsity,
            compression_ratio,
            fragmentation_level,
        }
    }

    fn collect_knowledge_graph_metrics(&self) -> KnowledgeGraphMemoryMetrics {
        let graph = self.knowledge_graph.lock().unwrap();
        
        KnowledgeGraphMemoryMetrics {
            entities: graph.entity_count(),
            relations: graph.relation_count(),
            total_memory: graph.total_memory_usage(),
            index_memory: graph.index_memory_usage(),
            cache_memory: graph.cache_memory_usage(),
            query_cache: QueryCacheMetrics {
                size: graph.query_cache_size(),
                hit_rate: graph.query_cache_hit_rate(),
            },
        }
    }

    fn collect_cognitive_metrics(&self) -> CognitiveMemoryMetrics {
        use crate::cognitive::LayerType;
        
        let mut subcortical = LayerMemoryMetrics::default();
        let mut cortical = LayerMemoryMetrics::default();
        let mut thalamic = LayerMemoryMetrics::default();

        for layer in &self.cognitive_layers {
            let layer_guard = layer.lock().unwrap();
            let metrics = match layer_guard.layer_type() {
                LayerType::Subcortical => &mut subcortical,
                LayerType::Cortical => &mut cortical,
                LayerType::Thalamic => &mut thalamic,
            };

            *metrics = LayerMemoryMetrics {
                allocated: layer_guard.allocated_memory(),
                active: layer_guard.active_memory(),
                cached: layer_guard.cached_memory(),
                patterns: layer_guard.pattern_memory(),
                connections: layer_guard.connection_memory(),
            };
        }

        CognitiveMemoryMetrics {
            subcortical,
            cortical,
            thalamic,
        }
    }

    fn collect_zero_copy_metrics(&self) -> ZeroCopyMetrics {
        let manager = self.zero_copy_manager.lock().unwrap();
        
        ZeroCopyMetrics {
            active_regions: manager.active_regions(),
            shared_pages: manager.shared_pages(),
            copy_on_write_events: manager.cow_events(),
            memory_saved: manager.memory_saved(),
            efficiency: manager.efficiency(),
        }
    }

    fn collect_system_metrics(&self) -> SystemMemoryMetrics {
        use sysinfo::{System, SystemExt};
        
        let mut sys = System::new_all();
        sys.refresh_memory();
        
        SystemMemoryMetrics {
            total_allocated: sys.used_memory() as usize,
            total_available: sys.total_memory() as usize,
            swap_usage: sys.used_swap() as usize,
            page_cache: 0, // Would need platform-specific implementation
        }
    }

    pub fn analyze_memory_patterns(&self) -> MemoryAnalysis {
        let history = self.metrics_history.lock().unwrap();
        if history.len() < 10 {
            return MemoryAnalysis::default();
        }

        let recent = &history[history.len() - 10..];
        
        // Analyze SDR growth pattern
        let sdr_growth_rate = Self::calculate_growth_rate(
            recent.iter().map(|m| m.sdr.total_sdrs).collect()
        );

        // Analyze memory fragmentation trend
        let fragmentation_trend = Self::calculate_trend(
            recent.iter().map(|m| m.sdr.fragmentation_level).collect()
        );

        // Analyze cache efficiency
        let cache_efficiency = recent.iter()
            .map(|m| m.knowledge_graph.query_cache.hit_rate)
            .sum::<f64>() / recent.len() as f64;

        // Identify memory hotspots
        let hotspots = self.identify_memory_hotspots(&recent[recent.len() - 1]);

        MemoryAnalysis {
            sdr_growth_rate,
            fragmentation_trend,
            cache_efficiency,
            hotspots,
            recommendations: self.generate_recommendations(&recent[recent.len() - 1]),
        }
    }

    fn calculate_growth_rate(values: Vec<usize>) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let first = values[0] as f64;
        let last = values[values.len() - 1] as f64;
        
        (last - first) / first
    }

    fn calculate_trend(values: Vec<f64>) -> Trend {
        if values.len() < 2 {
            return Trend::Stable;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let n = values.len() as f64;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        if slope > 0.01 {
            Trend::Increasing
        } else if slope < -0.01 {
            Trend::Decreasing
        } else {
            Trend::Stable
        }
    }

    fn identify_memory_hotspots(&self, metrics: &MemoryMetrics) -> Vec<MemoryHotspot> {
        let mut hotspots = Vec::new();

        // Check SDR memory usage
        if metrics.sdr.fragmentation_level > 0.3 {
            hotspots.push(MemoryHotspot {
                component: "SDR Storage".to_string(),
                issue: "High fragmentation".to_string(),
                severity: Severity::High,
                memory_impact: (metrics.sdr.total_memory as f64 * metrics.sdr.fragmentation_level) as usize,
            });
        }

        // Check knowledge graph cache
        if metrics.knowledge_graph.query_cache.hit_rate < 0.5 {
            hotspots.push(MemoryHotspot {
                component: "Knowledge Graph".to_string(),
                issue: "Low cache hit rate".to_string(),
                severity: Severity::Medium,
                memory_impact: metrics.knowledge_graph.cache_memory,
            });
        }

        // Check cognitive layer efficiency
        let total_cognitive = metrics.cognitive.subcortical.allocated +
                            metrics.cognitive.cortical.allocated +
                            metrics.cognitive.thalamic.allocated;
        let active_cognitive = metrics.cognitive.subcortical.active +
                              metrics.cognitive.cortical.active +
                              metrics.cognitive.thalamic.active;
        
        if active_cognitive < total_cognitive / 2 {
            hotspots.push(MemoryHotspot {
                component: "Cognitive Layers".to_string(),
                issue: "Low memory utilization".to_string(),
                severity: Severity::Low,
                memory_impact: total_cognitive - active_cognitive,
            });
        }

        hotspots
    }

    fn generate_recommendations(&self, metrics: &MemoryMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();

        if metrics.sdr.fragmentation_level > 0.3 {
            recommendations.push("Consider running SDR defragmentation to improve memory efficiency".to_string());
        }

        if metrics.sdr.compression_ratio < 0.5 {
            recommendations.push("SDR compression is suboptimal, consider adjusting sparsity parameters".to_string());
        }

        if metrics.knowledge_graph.query_cache.hit_rate < 0.7 {
            recommendations.push("Increase query cache size to improve performance".to_string());
        }

        if metrics.zero_copy.efficiency < 0.8 {
            recommendations.push("Zero-copy efficiency is low, review memory access patterns".to_string());
        }

        let memory_pressure = metrics.system.total_allocated as f64 / metrics.system.total_available as f64;
        if memory_pressure > 0.8 {
            recommendations.push("System memory pressure is high, consider memory optimization".to_string());
        }

        recommendations
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    pub sdr_growth_rate: f64,
    pub fragmentation_trend: Trend,
    pub cache_efficiency: f64,
    pub hotspots: Vec<MemoryHotspot>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHotspot {
    pub component: String,
    pub issue: String,
    pub severity: Severity,
    pub memory_impact: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
}

impl Default for MemoryAnalysis {
    fn default() -> Self {
        Self {
            sdr_growth_rate: 0.0,
            fragmentation_trend: Trend::Stable,
            cache_efficiency: 0.0,
            hotspots: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}

impl Default for LayerMemoryMetrics {
    fn default() -> Self {
        Self {
            allocated: 0,
            active: 0,
            cached: 0,
            patterns: 0,
            connections: 0,
        }
    }
}
```

## LLMKG-Specific Features

### 1. SDR Memory Optimization
- **Sparsity-Aware Storage**: Monitor how sparsity affects memory usage
- **Overlap-Based Compression**: Track memory savings from SDR similarities
- **Activation History**: Visualize SDR activation patterns over time

### 2. Knowledge Graph Memory
- **Entity Clustering**: Show memory hotspots in the graph
- **Relation Indexing**: Monitor index memory overhead
- **Query Pattern Analysis**: Optimize cache based on access patterns

### 3. Cognitive Layer Memory
- **Pattern Storage**: Track memory used for learned patterns
- **Connection Weights**: Monitor synaptic weight storage
- **Inhibitory Circuits**: Visualize inhibition memory overhead

### 4. Zero-Copy Optimization
- **Shared Memory Regions**: Track zero-copy efficiency
- **Copy-on-Write Events**: Monitor when copies are triggered
- **Memory Savings**: Calculate actual memory saved

## Testing Procedures

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_metrics_collection() {
        let monitor = create_test_monitor();
        let metrics = monitor.collect_metrics();
        
        assert!(metrics.sdr.total_memory > 0);
        assert!(metrics.knowledge_graph.entities > 0);
        assert!(metrics.system.total_allocated > 0);
    }

    #[test]
    fn test_fragmentation_detection() {
        let monitor = create_test_monitor();
        
        // Simulate fragmentation
        let mut storage = monitor.sdr_storage.lock().unwrap();
        storage.simulate_fragmentation(0.5);
        drop(storage);
        
        let metrics = monitor.collect_metrics();
        assert!(metrics.sdr.fragmentation_level > 0.4);
        
        let analysis = monitor.analyze_memory_patterns();
        assert!(!analysis.hotspots.is_empty());
    }

    #[test]
    fn test_memory_trend_analysis() {
        let monitor = create_test_monitor();
        
        // Collect multiple metrics
        for i in 0..20 {
            let mut storage = monitor.sdr_storage.lock().unwrap();
            storage.add_sdrs(i * 10);
            drop(storage);
            
            monitor.collect_metrics();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        
        let analysis = monitor.analyze_memory_patterns();
        assert!(analysis.sdr_growth_rate > 0.0);
    }
}
```

### Integration Tests
```typescript
// tests/memory-monitoring.test.tsx
describe('Memory Monitoring Dashboard', () => {
  it('should display memory allocation treemap', async () => {
    const { container } = render(<MemoryMonitoring />);
    
    await waitFor(() => {
      expect(container.querySelector('.memory-treemap')).toBeInTheDocument();
      expect(container.querySelector('[data-node-name="SDR Storage"]')).toBeInTheDocument();
    });
  });

  it('should update memory metrics in real-time', async () => {
    const { container } = render(<MemoryMonitoring />);
    
    // Initial render
    await waitFor(() => {
      expect(container.querySelector('.ant-statistic-content-value')).toBeTruthy();
    });

    const initialValue = container.querySelector('.ant-statistic-content-value').textContent;

    // Simulate metric update
    mockMCPClient.emit('memoryUpdate', {
      current: {
        system: { totalAllocated: 2048 * 1024 * 1024 }
      }
    });

    await waitFor(() => {
      const newValue = container.querySelector('.ant-statistic-content-value').textContent;
      expect(newValue).not.toBe(initialValue);
    });
  });

  it('should show memory hotspots and recommendations', async () => {
    const { container } = render(<MemoryMonitoring />);
    
    // Simulate high fragmentation
    mockMCPClient.emit('memoryUpdate', {
      current: {
        sdr: { fragmentationLevel: 0.5 }
      },
      analysis: {
        recommendations: ['Consider running SDR defragmentation']
      }
    });

    await waitFor(() => {
      expect(container.querySelector('.memory-recommendation')).toHaveTextContent('defragmentation');
    });
  });
});
```

## Performance Considerations

### 1. Memory Sampling
- Use statistical sampling for large memory regions
- Implement incremental metric collection
- Cache frequently accessed metrics

### 2. Visualization Optimization
- Use WebGL for large memory maps
- Implement virtual scrolling for detailed views
- Throttle real-time updates

### 3. Analysis Efficiency
- Run heavy analysis in background threads
- Use incremental statistics updates
- Implement result caching

## Deliverables Checklist

- [ ] Memory monitoring dashboard component
- [ ] SDR memory visualization with fragmentation analysis
- [ ] Knowledge graph memory treemap
- [ ] Cognitive layer memory breakdown
- [ ] Zero-copy efficiency tracking
- [ ] Memory flow Sankey diagram
- [ ] Real-time memory metrics collection
- [ ] Memory trend analysis and predictions
- [ ] Hotspot detection and recommendations
- [ ] Memory optimization suggestions
- [ ] Export functionality for memory reports
- [ ] Integration with performance monitoring