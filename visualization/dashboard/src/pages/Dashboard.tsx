import React from 'react';
import { useWebSocket } from '../providers/WebSocketProvider';
import { useAppSelector } from '../stores';

const Dashboard: React.FC = () => {
  const { isConnected, connectionState } = useWebSocket();
  const currentData = useAppSelector(state => state.data.current);

  return (
    <div className="dashboard-overview">
      <h1>System Overview</h1>
      
      <div className="status-grid">
        <div className="status-card">
          <h3>Connection Status</h3>
          <div className={`status-indicator ${connectionState}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
        
        <div className="status-card">
          <h3>Data Stream</h3>
          <div className="data-info">
            {currentData ? 'Receiving data' : 'No data'}
          </div>
        </div>
      </div>

      <style jsx>{`
        .dashboard-overview {
          padding: 2rem;
        }
        
        .status-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1rem;
          margin-top: 2rem;
        }
        
        .status-card {
          background: var(--bg-secondary, #2d2d2d);
          border: 1px solid var(--border-color, #404040);
          border-radius: 8px;
          padding: 1.5rem;
        }
        
        .status-card h3 {
          margin: 0 0 1rem 0;
          color: var(--text-primary, #ffffff);
        }
        
        .status-indicator {
          font-weight: 500;
          padding: 0.5rem 1rem;
          border-radius: 4px;
        }
        
        .status-indicator.connected {
          background: #00d26a;
          color: white;
        }
        
        .status-indicator.disconnected {
          background: #ff6b6b;
          color: white;
        }
        
        .data-info {
          color: var(--text-secondary, #b3b3b3);
        }
      `}</style>
    </div>
  );
};

export default Dashboard;