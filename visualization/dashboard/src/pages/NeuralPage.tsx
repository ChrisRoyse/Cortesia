import React from 'react';
import { useRealtimeData } from '../providers/WebSocketProvider';

const NeuralPage: React.FC = () => {
  const neuralData = useRealtimeData(data => data?.neural);

  return (
    <div className="neural-page">
      <h1>Neural Activity</h1>
      <p>Real-time neural network activity visualization and analysis.</p>
      
      {neuralData && (
        <div className="neural-metrics">
          <div className="metric-card">
            <h3>Overall Activity</h3>
            <div className="metric-value">{neuralData.overallActivity?.toFixed(2) || 'N/A'}</div>
          </div>
          
          <div className="metric-card">
            <h3>Active Nodes</h3>
            <div className="metric-value">{neuralData.activity?.length || 0}</div>
          </div>
          
          <div className="metric-card">
            <h3>Layers</h3>
            <div className="metric-value">{neuralData.layers?.length || 0}</div>
          </div>
          
          <div className="metric-card">
            <h3>Connections</h3>
            <div className="metric-value">{neuralData.connections?.length || 0}</div>
          </div>
        </div>
      )}

      <style jsx>{`
        .neural-page {
          padding: 2rem;
        }
        
        .neural-metrics {
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

export default NeuralPage;