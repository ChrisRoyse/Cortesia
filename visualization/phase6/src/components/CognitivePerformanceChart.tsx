import React, { useMemo } from 'react';
import { Line } from '@ant-design/charts';
import { format } from 'date-fns';
import type { PerformanceMetrics } from '../types';

export interface CognitivePerformanceChartProps {
  metrics: PerformanceMetrics[];
  timeWindow: number;
  height?: number;
  showLegend?: boolean;
}

export const CognitivePerformanceChart: React.FC<CognitivePerformanceChartProps> = ({
  metrics,
  timeWindow,
  height = 300,
  showLegend = true
}) => {
  const chartData = useMemo(() => {
    const now = Date.now();
    const windowStart = now - (timeWindow * 1000);
    
    return metrics
      .filter(m => m.timestamp >= windowStart)
      .flatMap(m => [
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          layer: 'Subcortical',
          value: m.cognitive.subcortical.processingLatency,
          type: 'latency'
        },
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          layer: 'Cortical',
          value: m.cognitive.cortical.processingLatency,
          type: 'latency'
        },
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          layer: 'Thalamic',
          value: m.cognitive.thalamic.processingLatency,
          type: 'latency'
        }
      ]);
  }, [metrics, timeWindow]);

  const config = {
    data: chartData,
    xField: 'timestamp',
    yField: 'value',
    seriesField: 'layer',
    smooth: true,
    height,
    legend: showLegend ? {
      position: 'top-right' as const,
    } : false,
    yAxis: {
      title: {
        text: 'Processing Latency (ms)',
        style: {
          fontSize: 12,
        },
      },
      grid: {
        line: {
          style: {
            stroke: '#e8e8e8',
            lineWidth: 0.5,
            lineDash: [4, 2],
          },
        },
      },
    },
    xAxis: {
      title: {
        text: 'Time',
        style: {
          fontSize: 12,
        },
      },
      label: {
        rotate: -45,
        style: {
          fontSize: 10,
        },
      },
    },
    color: ['#ff6b6b', '#4ecdc4', '#45b7d1'],
    tooltip: {
      formatter: (datum: any) => {
        return {
          name: datum.layer,
          value: `${datum.value.toFixed(2)}ms`,
        };
      },
    },
    annotations: [
      {
        type: 'line',
        start: ['min', 50],
        end: ['max', 50],
        style: {
          stroke: '#ff0000',
          lineWidth: 1,
          lineDash: [4, 4],
        },
      },
      {
        type: 'text',
        position: ['95%', 50],
        content: 'Threshold',
        style: {
          fontSize: 10,
          fill: '#ff0000',
        },
      },
    ],
    animation: {
      appear: {
        animation: 'path-in',
        duration: 500,
      },
    },
  };

  // Add throughput data as a secondary metric
  const throughputData = useMemo(() => {
    const now = Date.now();
    const windowStart = now - (timeWindow * 1000);
    
    return metrics
      .filter(m => m.timestamp >= windowStart)
      .flatMap(m => [
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          layer: 'Subcortical',
          value: m.cognitive.subcortical.throughput,
          type: 'throughput'
        },
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          layer: 'Cortical',
          value: m.cognitive.cortical.throughput,
          type: 'throughput'
        },
        {
          timestamp: format(new Date(m.timestamp), 'HH:mm:ss'),
          layer: 'Thalamic',
          value: m.cognitive.thalamic.throughput,
          type: 'throughput'
        }
      ]);
  }, [metrics, timeWindow]);

  return (
    <div className="cognitive-performance-chart">
      <Line {...config} />
      
      {/* Additional metrics display */}
      <div className="cognitive-metrics-summary" style={{ marginTop: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-around', fontSize: 12 }}>
          <div>
            <span style={{ color: '#666' }}>Avg Activation Rate: </span>
            <span style={{ fontWeight: 'bold' }}>
              {metrics.length > 0 
                ? ((metrics[metrics.length - 1].cognitive.subcortical.activationRate +
                    metrics[metrics.length - 1].cognitive.cortical.activationRate +
                    metrics[metrics.length - 1].cognitive.thalamic.activationRate) / 3).toFixed(2)
                : '0.00'}
            </span>
          </div>
          <div>
            <span style={{ color: '#666' }}>Inhibition Balance: </span>
            <span style={{ fontWeight: 'bold' }}>
              {metrics.length > 0 
                ? ((metrics[metrics.length - 1].cognitive.subcortical.inhibitionRate +
                    metrics[metrics.length - 1].cognitive.cortical.inhibitionRate +
                    metrics[metrics.length - 1].cognitive.thalamic.inhibitionRate) / 3).toFixed(2)
                : '0.00'}
            </span>
          </div>
          <div>
            <span style={{ color: '#666' }}>Error Count: </span>
            <span style={{ 
              fontWeight: 'bold',
              color: metrics.length > 0 && 
                     (metrics[metrics.length - 1].cognitive.subcortical.errorCount +
                      metrics[metrics.length - 1].cognitive.cortical.errorCount +
                      metrics[metrics.length - 1].cognitive.thalamic.errorCount) > 0 
                     ? '#ff0000' : '#52c41a'
            }}>
              {metrics.length > 0 
                ? (metrics[metrics.length - 1].cognitive.subcortical.errorCount +
                   metrics[metrics.length - 1].cognitive.cortical.errorCount +
                   metrics[metrics.length - 1].cognitive.thalamic.errorCount)
                : 0}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};