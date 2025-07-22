import React, { useMemo } from 'react';
import { Heatmap } from '@ant-design/charts';
import { format } from 'date-fns';
import type { PerformanceMetrics } from '../types';

export interface PerformanceHeatmapProps {
  metrics: PerformanceMetrics[];
  height?: number;
}

export const PerformanceHeatmap: React.FC<PerformanceHeatmapProps> = ({
  metrics,
  height = 400
}) => {
  const heatmapData = useMemo(() => {
    // Group metrics by time buckets (5-minute intervals)
    const bucketSize = 5 * 60 * 1000; // 5 minutes
    const buckets = new Map<number, PerformanceMetrics[]>();
    
    metrics.forEach(m => {
      const bucket = Math.floor(m.timestamp / bucketSize) * bucketSize;
      if (!buckets.has(bucket)) {
        buckets.set(bucket, []);
      }
      buckets.get(bucket)!.push(m);
    });

    const data: any[] = [];
    
    // Components to track
    const components = [
      { key: 'cognitive.subcortical', label: 'Subcortical' },
      { key: 'cognitive.cortical', label: 'Cortical' },
      { key: 'cognitive.thalamic', label: 'Thalamic' },
      { key: 'sdr', label: 'SDR' },
      { key: 'mcp', label: 'MCP' },
      { key: 'system.cpu', label: 'CPU' },
      { key: 'system.memory', label: 'Memory' }
    ];

    Array.from(buckets.entries()).forEach(([timestamp, metricsGroup]) => {
      const timeLabel = format(new Date(timestamp), 'HH:mm');
      
      components.forEach(component => {
        let value = 0;
        
        if (component.key.startsWith('cognitive.')) {
          const layer = component.key.split('.')[1] as keyof typeof metricsGroup[0]['cognitive'];
          const avgLatency = metricsGroup.reduce((sum, m) => 
            sum + m.cognitive[layer].processingLatency, 0
          ) / metricsGroup.length;
          value = Math.min(100, avgLatency); // Normalize to 0-100
        } else if (component.key === 'sdr') {
          const avgEfficiency = metricsGroup.reduce((sum, m) => 
            sum + (m.sdr.creationRate / (m.sdr.memoryUsage / 1000000)), 0
          ) / metricsGroup.length;
          value = Math.min(100, avgEfficiency * 10); // Normalize
        } else if (component.key === 'mcp') {
          const avgHealth = metricsGroup.reduce((sum, m) => 
            sum + (100 - m.mcp.averageLatency), 0
          ) / metricsGroup.length;
          value = Math.max(0, avgHealth);
        } else if (component.key.startsWith('system.')) {
          const metric = component.key.split('.')[1] as keyof typeof metricsGroup[0]['system'];
          value = metricsGroup.reduce((sum, m) => 
            sum + m.system[metric], 0
          ) / metricsGroup.length;
        }
        
        data.push({
          time: timeLabel,
          component: component.label,
          value: Math.round(value)
        });
      });
    });

    return data;
  }, [metrics]);

  const config = {
    data: heatmapData,
    xField: 'time',
    yField: 'component',
    colorField: 'value',
    height,
    color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'].reverse(),
    shape: 'square' as const,
    label: {
      visible: true,
      style: {
        fill: '#fff',
        shadowBlur: 2,
        shadowColor: 'rgba(0, 0, 0, .45)',
      },
    },
    tooltip: {
      formatter: (datum: any) => {
        return {
          name: datum.component,
          value: `${datum.value}%`,
        };
      },
    },
    xAxis: {
      label: {
        rotate: -45,
        style: {
          fontSize: 10,
        },
      },
    },
    yAxis: {
      label: {
        style: {
          fontSize: 12,
        },
      },
    },
    legend: {
      position: 'right' as const,
      title: {
        text: 'Performance',
        style: {
          fontSize: 12,
        },
      },
    },
  };

  return (
    <div className="performance-heatmap">
      <Heatmap {...config} />
      
      {/* Heatmap Summary */}
      <div className="heatmap-summary" style={{ marginTop: 16, fontSize: 12, color: '#666' }}>
        <p>
          The heatmap shows performance metrics across different system components over time. 
          Warmer colors (red) indicate higher values, which may represent:
        </p>
        <ul style={{ marginLeft: 20, marginTop: 8 }}>
          <li>For cognitive layers: Higher processing latency (worse performance)</li>
          <li>For SDR: Higher efficiency (better performance)</li>
          <li>For MCP: Better protocol health</li>
          <li>For system resources: Higher utilization</li>
        </ul>
      </div>
    </div>
  );
};