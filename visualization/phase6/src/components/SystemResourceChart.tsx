import React, { useMemo } from 'react';
import { Line } from '@ant-design/charts';
import { format } from 'date-fns';
import type { PerformanceMetrics } from '../types';

export interface SystemResourceChartProps {
  metrics: PerformanceMetrics[];
  timeWindow: number;
  height?: number;
}

export const SystemResourceChart: React.FC<SystemResourceChartProps> = ({
  metrics,
  timeWindow,
  height = 300
}) => {
  const chartData = useMemo(() => {
    const now = Date.now();
    const windowStart = now - (timeWindow * 1000);
    
    return metrics
      .filter(m => m.timestamp >= windowStart)
      .flatMap(m => [
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          resource: 'CPU',
          value: m.system.cpuUsage,
          category: 'compute'
        },
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          resource: 'Memory',
          value: m.system.memoryUsage,
          category: 'compute'
        },
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          resource: 'Disk I/O',
          value: m.system.diskIO,
          category: 'io'
        },
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          resource: 'Network I/O',
          value: m.system.networkIO,
          category: 'io'
        }
      ]);
  }, [metrics, timeWindow]);

  const config = {
    data: chartData,
    xField: 'timestamp',
    yField: 'value',
    seriesField: 'resource',
    height,
    smooth: true,
    yAxis: {
      title: {
        text: 'Usage (%)',
        style: {
          fontSize: 12,
        },
      },
      max: 100,
      min: 0,
    },
    xAxis: {
      label: {
        rotate: -45,
        style: {
          fontSize: 10,
        },
      },
    },
    color: ['#5B8FF9', '#5AD8A6', '#F6BD16', '#E86452'],
    legend: {
      position: 'top-right' as const,
    },
    tooltip: {
      formatter: (datum: any) => {
        return {
          name: datum.resource,
          value: `${datum.value.toFixed(1)}%`,
        };
      },
    },
    annotations: [
      {
        type: 'line',
        start: ['min', 80],
        end: ['max', 80],
        style: {
          stroke: '#ff0000',
          lineWidth: 1,
          lineDash: [4, 4],
        },
      },
      {
        type: 'text',
        position: ['95%', 80],
        content: 'Warning',
        style: {
          fontSize: 10,
          fill: '#ff0000',
        },
      },
    ],
  };

  // Calculate resource utilization summary
  const resourceSummary = useMemo(() => {
    if (metrics.length === 0) return null;
    
    const latest = metrics[metrics.length - 1];
    const avgCPU = metrics.slice(-10).reduce((sum, m) => sum + m.system.cpuUsage, 0) / Math.min(metrics.length, 10);
    const avgMemory = metrics.slice(-10).reduce((sum, m) => sum + m.system.memoryUsage, 0) / Math.min(metrics.length, 10);
    
    return {
      current: {
        cpu: latest.system.cpuUsage,
        memory: latest.system.memoryUsage,
        disk: latest.system.diskIO,
        network: latest.system.networkIO
      },
      average: {
        cpu: avgCPU,
        memory: avgMemory
      },
      status: latest.system.cpuUsage > 80 || latest.system.memoryUsage > 80 ? 'warning' : 'normal'
    };
  }, [metrics]);

  return (
    <div className="system-resource-chart">
      <Line {...config} />
      
      {/* Resource Summary Cards */}
      {resourceSummary && (
        <div className="resource-summary" style={{ marginTop: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
            <div style={{ 
              padding: 12, 
              backgroundColor: resourceSummary.current.cpu > 80 ? '#fff2e8' : '#f6ffed',
              borderRadius: 8,
              border: `1px solid ${resourceSummary.current.cpu > 80 ? '#ffbb96' : '#b7eb8f'}`
            }}>
              <div style={{ fontSize: 12, color: '#666' }}>CPU Usage</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: resourceSummary.current.cpu > 80 ? '#ff7875' : '#52c41a' }}>
                  {resourceSummary.current.cpu.toFixed(1)}%
                </div>
                <div style={{ fontSize: 12, color: '#999' }}>
                  avg: {resourceSummary.average.cpu.toFixed(1)}%
                </div>
              </div>
            </div>
            
            <div style={{ 
              padding: 12, 
              backgroundColor: resourceSummary.current.memory > 80 ? '#fff2e8' : '#f6ffed',
              borderRadius: 8,
              border: `1px solid ${resourceSummary.current.memory > 80 ? '#ffbb96' : '#b7eb8f'}`
            }}>
              <div style={{ fontSize: 12, color: '#666' }}>Memory Usage</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: resourceSummary.current.memory > 80 ? '#ff7875' : '#52c41a' }}>
                  {resourceSummary.current.memory.toFixed(1)}%
                </div>
                <div style={{ fontSize: 12, color: '#999' }}>
                  avg: {resourceSummary.average.memory.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* I/O Metrics */}
          <div style={{ marginTop: 16, padding: 12, backgroundColor: '#fafafa', borderRadius: 8 }}>
            <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>I/O Performance</div>
            <div style={{ display: 'flex', justifyContent: 'space-around' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 10, color: '#999' }}>Disk I/O</div>
                <div style={{ fontSize: 16, fontWeight: 'bold', color: '#1890ff' }}>
                  {resourceSummary.current.disk.toFixed(1)}%
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 10, color: '#999' }}>Network I/O</div>
                <div style={{ fontSize: 16, fontWeight: 'bold', color: '#1890ff' }}>
                  {resourceSummary.current.network.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};