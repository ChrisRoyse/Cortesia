import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { ToolsMainPage } from '../features/tools/pages/ToolsMainPage';

// Legacy MCP tools interface for backward compatibility
import { useMCP, useMCPExecution } from '../providers/MCPProvider';

const LegacyToolsView: React.FC = () => {
  const { tools, loading, error } = useMCP();
  const { execute, history } = useMCPExecution();

  const handleExecuteTool = async (toolName: string) => {
    try {
      const result = await execute(toolName, {});
      console.log('Tool execution result:', result);
    } catch (error) {
      console.error('Tool execution failed:', error);
    }
  };

  return (
    <div className="legacy-tools-view">
      <div className="legacy-header">
        <h2>Legacy MCP Tools</h2>
        <p>Direct MCP tool execution interface for development and testing.</p>
      </div>
      
      {loading && <div className="loading">Loading tools...</div>}
      {error && <div className="error">Error: {error}</div>}
      
      <div className="tools-grid">
        {tools.map(tool => (
          <div key={tool.name} className="tool-card">
            <h3>{tool.name}</h3>
            <p>{tool.description}</p>
            <div className="tool-category">{tool.category}</div>
            <button 
              className="execute-button"
              onClick={() => handleExecuteTool(tool.name)}
            >
              Execute
            </button>
          </div>
        ))}
      </div>
      
      {history.length > 0 && (
        <div className="execution-history">
          <h3>Recent Executions</h3>
          <div className="history-list">
            {history.slice(-5).map((execution, index) => (
              <div key={index} className="history-item">
                <div className="history-tool">{execution.toolName}</div>
                <div className="history-duration">{execution.duration}ms</div>
                <div className="history-time">
                  {new Date(execution.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <style jsx>{`
        .legacy-tools-view {
          padding: 1.5rem;
          background: var(--bg-primary);
          color: var(--text-primary);
        }

        .legacy-header {
          margin-bottom: 2rem;
        }

        .legacy-header h2 {
          margin: 0 0 0.5rem 0;
          color: var(--text-primary);
        }

        .legacy-header p {
          margin: 0;
          color: var(--text-secondary);
          font-size: 0.9rem;
        }
        
        .loading {
          padding: 1rem;
          text-align: center;
          color: var(--text-secondary);
        }
        
        .error {
          color: var(--error-color);
          margin: 1rem 0;
          padding: 1rem;
          background: rgba(var(--error-color-rgb), 0.1);
          border: 1px solid var(--error-color);
          border-radius: 4px;
        }
        
        .tools-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 1rem;
          margin-bottom: 2rem;
        }
        
        .tool-card {
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 1.5rem;
        }
        
        .tool-card h3 {
          margin: 0 0 1rem 0;
          color: var(--text-primary);
        }
        
        .tool-card p {
          color: var(--text-secondary);
          margin-bottom: 1rem;
        }
        
        .tool-category {
          display: inline-block;
          background: var(--accent-color);
          color: white;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
          margin-bottom: 1rem;
        }
        
        .execute-button {
          background: var(--accent-color);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          cursor: pointer;
          font-weight: 500;
          transition: opacity 0.2s;
        }
        
        .execute-button:hover {
          opacity: 0.9;
        }
        
        .execution-history {
          margin-top: 2rem;
        }
        
        .execution-history h3 {
          color: var(--text-primary);
          margin-bottom: 1rem;
        }
        
        .history-list {
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 1rem;
        }
        
        .history-item {
          display: grid;
          grid-template-columns: 1fr auto auto;
          gap: 1rem;
          padding: 0.5rem 0;
          border-bottom: 1px solid var(--border-color);
        }
        
        .history-item:last-child {
          border-bottom: none;
        }
        
        .history-tool {
          font-weight: 500;
          color: var(--text-primary);
        }
        
        .history-duration {
          color: var(--text-secondary);
        }
        
        .history-time {
          color: var(--text-secondary);
          font-size: 0.875rem;
        }
      `}</style>
    </div>
  );
};

const ToolsPage: React.FC = () => {
  return (
    <Routes>
      {/* Main Phase 3 Tool Catalog */}
      <Route path="/*" element={<ToolsMainPage />} />
      
      {/* Legacy MCP tools interface */}
      <Route path="/legacy" element={<LegacyToolsView />} />
    </Routes>
  );
};

export default ToolsPage;