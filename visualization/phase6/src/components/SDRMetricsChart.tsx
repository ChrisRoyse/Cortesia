import React, { useMemo } from 'react';
import { Area } from '@ant-design/charts';
import { format } from 'date-fns';
import type { PerformanceMetrics } from '../types';

export interface SDRMetricsChartProps {
  metrics: PerformanceMetrics[];
  timeWindow: number;
  height?: number;
}

export const SDRMetricsChart: React.FC<SDRMetricsChartProps> = ({
  metrics,
  timeWindow,
  height = 300
}) => {
  const chartData = useMemo(() => {
    const now = Date.now();
    const windowStart = now - (timeWindow * 1000);
    
    return metrics
      .filter(m => m.timestamp >= windowStart)
      .map(m => ({
        timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
        creationRate: m.sdr.creationRate,
        sparsity: m.sdr.averageSparsity * 100,
        overlap: m.sdr.overlapRatio * 100,
        memory: m.sdr.memoryUsage / 1024 / 1024 // Convert to MB
      }));
  }, [metrics, timeWindow]);

  const config = {
    data: chartData,
    xField: 'timestamp',
    yField: 'sparsity',
    height,
    smooth: true,
    areaStyle: {
      fillOpacity: 0.3,
    },
    color: '#5AD8A6',
    yAxis: {
      title: {
        text: 'Sparsity (%)',
        style: {
          fontSize: 12,
        },
      },
      min: 0,
      max: 10,
    },
    xAxis: {
      label: {
        rotate: -45,
        style: {
          fontSize: 10,
        },
      },
    },
    annotations: [
      {
        type: 'region',
        start: ['min', 2],
        end: ['max', 5],
        style: {
          fill: '#5AD8A6',
          fillOpacity: 0.1,
        },
      },
      {
        type: 'text',
        position: ['50%', 3.5],
        content: 'Optimal Range',
        style: {
          fontSize: 10,
          fill: '#5AD8A6',
          textAlign: 'center',
        },
      },
    ],
    tooltip: {
      formatter: (datum: any) => {
        return {
          name: 'Sparsity',
          value: `${datum.sparsity.toFixed(2)}%`,
        };
      },
    },
  };

  // Calculate efficiency score
  const efficiencyScore = useMemo(() => {
    if (metrics.length === 0) return 0;
    const latest = metrics[metrics.length - 1];
    const efficiency = (latest.sdr.creationRate / (latest.sdr.memoryUsage / 1024 / 1024)) * 100;
    return Math.min(100, efficiency);
  }, [metrics]);

  return (
    <div className="sdr-metrics-chart">
      <Area {...config} />
      
      {/* SDR Statistics */}
      <div className="sdr-stats" style={{ marginTop: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, fontSize: 12 }}>
          <div>
            <div style={{ color: '#666' }}>Creation Rate</div>
            <div style={{ fontSize: 20, fontWeight: 'bold', color: '#1890ff' }}>
              {metrics.length > 0 ? metrics[metrics.length - 1].sdr.creationRate.toFixed(0) : 0}/s
            </div>
          </div>
          <div>
            <div style={{ color: '#666' }}>Memory Usage</div>
            <div style={{ fontSize: 20, fontWeight: 'bold', color: '#ff7875' }}>
              {metrics.length > 0 
                ? (metrics[metrics.length - 1].sdr.memoryUsage / 1024 / 1024).toFixed(1) 
                : 0} MB
            </div>
          </div>
          <div>
            <div style={{ color: '#666' }}>Overlap Ratio</div>
            <div style={{ fontSize: 20, fontWeight: 'bold', color: '#faad14' }}>
              {metrics.length > 0 
                ? (metrics[metrics.length - 1].sdr.overlapRatio * 100).toFixed(1) 
                : 0}%
            </div>
          </div>
          <div>
            <div style={{ color: '#666' }}>Efficiency Score</div>
            <div style={{ fontSize: 20, fontWeight: 'bold', color: '#52c41a' }}>
              {efficiencyScore.toFixed(0)}/100
            </div>
          </div>
        </div>
      </div>

      {/* Compression Ratio Gauge */}
      <div className="compression-gauge" style={{ marginTop: 16 }}>
        <div style={{ color: '#666', fontSize: 12, marginBottom: 8 }}>Compression Ratio</div>
        <div style={{ position: 'relative', height: 20, backgroundColor: '#f0f0f0', borderRadius: 10 }}>
          <div 
            style={{ 
              position: 'absolute',
              left: 0,
              top: 0,
              height: '100%',
              width: `${metrics.length > 0 ? Math.min(100, (metrics[metrics.length - 1].sdr.compressionRatio || 10) * 10) : 0}%`,
              backgroundColor: '#52c41a',
              borderRadius: 10,
              transition: 'width 0.3s ease'
            }}
          />
          <div style={{ 
            position: 'absolute', 
            left: '50%', 
            top: '50%', 
            transform: 'translate(-50%, -50%)',
            fontSize: 12,
            fontWeight: 'bold'
          }}>
            {metrics.length > 0 ? `${(metrics[metrics.length - 1].sdr.compressionRatio || 10).toFixed(1)}:1` : '0:1'}
          </div>
        </div>
      </div>
    </div>
  );
};