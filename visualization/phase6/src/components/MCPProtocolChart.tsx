import React, { useMemo } from 'react';
import { Column } from '@ant-design/charts';
import { format } from 'date-fns';
import type { PerformanceMetrics } from '../types';

export interface MCPProtocolChartProps {
  metrics: PerformanceMetrics[];
  timeWindow: number;
  height?: number;
}

export const MCPProtocolChart: React.FC<MCPProtocolChartProps> = ({
  metrics,
  timeWindow,
  height = 300
}) => {
  const chartData = useMemo(() => {
    const now = Date.now();
    const windowStart = now - (timeWindow * 1000);
    
    // Group metrics by time intervals
    const interval = timeWindow > 300 ? 60000 : 10000; // 1 minute or 10 seconds
    const grouped = new Map<number, PerformanceMetrics[]>();
    
    metrics
      .filter(m => m.timestamp >= windowStart)
      .forEach(m => {
        const bucket = Math.floor(m.timestamp / interval) * interval;
        if (!grouped.has(bucket)) {
          grouped.set(bucket, []);
        }
        grouped.get(bucket)!.push(m);
      });

    return Array.from(grouped.entries()).map(([timestamp, metricsGroup]) => {
      const avgLatency = metricsGroup.reduce((sum, m) => sum + m.mcp.averageLatency, 0) / metricsGroup.length;
      const totalMessages = metricsGroup.reduce((sum, m) => sum + m.mcp.messageRate, 0);
      const avgErrorRate = metricsGroup.reduce((sum, m) => sum + m.mcp.errorRate, 0) / metricsGroup.length;
      
      return {
        timestamp: format(new Date(timestamp), 'HH:mm:ss'),
        latency: avgLatency,
        messages: totalMessages,
        errors: avgErrorRate * 100,
        queueLength: metricsGroup[metricsGroup.length - 1].mcp.queueLength
      };
    });
  }, [metrics, timeWindow]);

  const config = {
    data: chartData,
    xField: 'timestamp',
    yField: 'messages',
    height,
    color: '#1890ff',
    columnStyle: {
      radius: [4, 4, 0, 0],
    },
    yAxis: {
      title: {
        text: 'Messages per Second',
        style: {
          fontSize: 12,
        },
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
    tooltip: {
      formatter: (datum: any) => {
        return {
          name: 'Message Rate',
          value: `${datum.messages.toFixed(0)} msg/s`,
        };
      },
    },
    annotations: chartData.length > 0 ? [
      {
        type: 'line',
        start: ['min', 1000],
        end: ['max', 1000],
        style: {
          stroke: '#52c41a',
          lineWidth: 1,
          lineDash: [4, 4],
        },
      },
      {
        type: 'text',
        position: ['95%', 1000],
        content: 'Target',
        style: {
          fontSize: 10,
          fill: '#52c41a',
        },
      },
    ] : [],
  };

  // Calculate protocol health
  const protocolHealth = useMemo(() => {
    if (metrics.length === 0) return { score: 0, status: 'unknown' };
    
    const latest = metrics[metrics.length - 1];
    const latencyScore = Math.max(0, 100 - latest.mcp.averageLatency);
    const errorScore = Math.max(0, 100 - (latest.mcp.errorRate * 1000));
    const queueScore = Math.max(0, 100 - (latest.mcp.queueLength / 10));
    
    const score = (latencyScore + errorScore + queueScore) / 3;
    const status = score >= 80 ? 'healthy' : score >= 60 ? 'warning' : 'critical';
    
    return { score, status };
  }, [metrics]);

  return (
    <div className="mcp-protocol-chart">
      <Column {...config} />
      
      {/* Protocol Statistics */}
      <div className="protocol-stats" style={{ marginTop: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, fontSize: 12 }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#666' }}>Avg Latency</div>
            <div style={{ 
              fontSize: 24, 
              fontWeight: 'bold', 
              color: metrics.length > 0 && metrics[metrics.length - 1].mcp.averageLatency > 50 ? '#ff4d4f' : '#52c41a' 
            }}>
              {metrics.length > 0 ? metrics[metrics.length - 1].mcp.averageLatency.toFixed(1) : 0}ms
            </div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#666' }}>Error Rate</div>
            <div style={{ 
              fontSize: 24, 
              fontWeight: 'bold', 
              color: metrics.length > 0 && metrics[metrics.length - 1].mcp.errorRate > 0.01 ? '#ff4d4f' : '#52c41a' 
            }}>
              {metrics.length > 0 ? (metrics[metrics.length - 1].mcp.errorRate * 100).toFixed(2) : 0}%
            </div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#666' }}>Queue Length</div>
            <div style={{ 
              fontSize: 24, 
              fontWeight: 'bold', 
              color: metrics.length > 0 && metrics[metrics.length - 1].mcp.queueLength > 100 ? '#faad14' : '#52c41a' 
            }}>
              {metrics.length > 0 ? metrics[metrics.length - 1].mcp.queueLength : 0}
            </div>
          </div>
        </div>
      </div>

      {/* Protocol Health Indicator */}
      <div className="protocol-health" style={{ marginTop: 16, padding: 16, backgroundColor: '#f0f2f5', borderRadius: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 'bold' }}>Protocol Health</div>
            <div style={{ fontSize: 12, color: '#666' }}>
              {protocolHealth.status === 'healthy' ? 'All systems operational' :
               protocolHealth.status === 'warning' ? 'Performance degradation detected' :
               'Critical issues detected'}
            </div>
          </div>
          <div style={{ 
            fontSize: 32, 
            fontWeight: 'bold',
            color: protocolHealth.status === 'healthy' ? '#52c41a' :
                   protocolHealth.status === 'warning' ? '#faad14' : '#ff4d4f'
          }}>
            {protocolHealth.score.toFixed(0)}%
          </div>
        </div>
      </div>
    </div>
  );
};