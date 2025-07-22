import React from 'react';
import { useRealtimeData } from '../providers/WebSocketProvider';

const MemoryPage: React.FC = () => {
  const memoryData = useRealtimeData(data => data?.memory);

  return (
    <div className="memory-page">
      <h1>Memory System Status</h1>
      <p>Memory usage, performance metrics, and storage optimization.</p>
      
      {memoryData && (
        <div className="memory-metrics">
          <div className="metric-card">
            <h3>Memory Usage</h3>
            <div className="metric-value">{memoryData.usage?.percentage?.toFixed(1) || 'N/A'}%</div>
          </div>
          
          <div className="metric-card">
            <h3>Latency</h3>
            <div className="metric-value">{memoryData.performance?.latency?.toFixed(2) || 'N/A'}ms</div>
          </div>
          
          <div className="metric-card">
            <h3>Throughput</h3>
            <div className="metric-value">{memoryData.performance?.throughput?.toFixed(1) || 'N/A'}/s</div>
          </div>
          
          <div className="metric-card">
            <h3>Stores</h3>
            <div className="metric-value">{memoryData.stores?.length || 0}</div>
          </div>
        </div>
      )}

      <style jsx>{`
        .memory-page {
          padding: 2rem;
        }
        
        .memory-metrics {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem;
          margin-top: 2rem;
        }
        
        .metric-card {
          background: var(--bg-secondary, #2d2d2d);
          border: 1px solid var(--border-color, #404040);
          border-radius: 8px;
          padding: 1.5rem;
          text-align: center;
        }
        
        .metric-card h3 {
          margin: 0 0 1rem 0;
          color: var(--text-primary, #ffffff);
        }
        
        .metric-value {
          font-size: 2rem;
          font-weight: 600;
          color: var(--accent-color, #007acc);
        }
      `}</style>
    </div>
  );
};

export default MemoryPage;