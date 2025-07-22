import React from 'react';
import { useRealtimeData } from '../providers/WebSocketProvider';

const CognitivePage: React.FC = () => {
  const cognitiveData = useRealtimeData(data => data?.cognitive);

  return (
    <div className="cognitive-page">
      <h1>Cognitive Pattern Analysis</h1>
      <p>Real-time visualization of cognitive patterns and inhibitory mechanisms.</p>
      
      {cognitiveData && (
        <div className="cognitive-metrics">
          <div className="metric-card">
            <h3>Inhibitory Level</h3>
            <div className="metric-value">{cognitiveData.inhibitoryLevel?.toFixed(2) || 'N/A'}</div>
          </div>
          
          <div className="metric-card">
            <h3>Active Patterns</h3>
            <div className="metric-value">{cognitiveData.patterns?.length || 0}</div>
          </div>
        </div>
      )}

      <style jsx>{`
        .cognitive-page {
          padding: 2rem;
        }
        
        .cognitive-metrics {
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

export default CognitivePage;