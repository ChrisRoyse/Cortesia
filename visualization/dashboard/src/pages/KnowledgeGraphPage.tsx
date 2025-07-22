import React from 'react';
import { useRealtimeData } from '../providers/WebSocketProvider';

const KnowledgeGraphPage: React.FC = () => {
  const knowledgeData = useRealtimeData(data => data?.knowledgeGraph);

  return (
    <div className="knowledge-page">
      <h1>Knowledge Graph Visualization</h1>
      <p>Interactive knowledge graph visualization and exploration.</p>
      
      {knowledgeData && (
        <div className="knowledge-metrics">
          <div className="metric-card">
            <h3>Nodes</h3>
            <div className="metric-value">{knowledgeData.nodes?.length || 0}</div>
          </div>
          
          <div className="metric-card">
            <h3>Edges</h3>
            <div className="metric-value">{knowledgeData.edges?.length || 0}</div>
          </div>
          
          <div className="metric-card">
            <h3>Clusters</h3>
            <div className="metric-value">{knowledgeData.clusters?.length || 0}</div>
          </div>
          
          <div className="metric-card">
            <h3>Density</h3>
            <div className="metric-value">{knowledgeData.metrics?.density?.toFixed(3) || 'N/A'}</div>
          </div>
        </div>
      )}

      <style jsx>{`
        .knowledge-page {
          padding: 2rem;
        }
        
        .knowledge-metrics {
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

export default KnowledgeGraphPage;