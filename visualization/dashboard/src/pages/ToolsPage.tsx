import React from 'react';
import { useMCP, useMCPExecution } from '../providers/MCPProvider';

const ToolsPage: React.FC = () => {
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
    <div className="tools-page">
      <h1>MCP Tools</h1>
      <p>Available MCP tools for LLMKG system interaction and testing.</p>
      
      {loading && <div>Loading tools...</div>}
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
          <h2>Recent Executions</h2>
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
        .tools-page {
          padding: 2rem;
        }
        
        .error {
          color: var(--error-color, #ff6b6b);
          margin: 1rem 0;
          padding: 1rem;
          background: rgba(255, 107, 107, 0.1);
          border-radius: 4px;
        }
        
        .tools-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 1rem;
          margin-top: 2rem;
        }
        
        .tool-card {
          background: var(--bg-secondary, #2d2d2d);
          border: 1px solid var(--border-color, #404040);
          border-radius: 8px;
          padding: 1.5rem;
        }
        
        .tool-card h3 {
          margin: 0 0 1rem 0;
          color: var(--text-primary, #ffffff);
        }
        
        .tool-card p {
          color: var(--text-secondary, #b3b3b3);
          margin-bottom: 1rem;
        }
        
        .tool-category {
          display: inline-block;
          background: var(--accent-color, #007acc);
          color: white;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
          margin-bottom: 1rem;
        }
        
        .execute-button {
          background: var(--accent-color, #007acc);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          cursor: pointer;
          font-weight: 500;
        }
        
        .execute-button:hover {
          background: #005a99;
        }
        
        .execution-history {
          margin-top: 3rem;
        }
        
        .execution-history h2 {
          color: var(--text-primary, #ffffff);
          margin-bottom: 1rem;
        }
        
        .history-list {
          background: var(--bg-secondary, #2d2d2d);
          border: 1px solid var(--border-color, #404040);
          border-radius: 8px;
          padding: 1rem;
        }
        
        .history-item {
          display: grid;
          grid-template-columns: 1fr auto auto;
          gap: 1rem;
          padding: 0.5rem 0;
          border-bottom: 1px solid var(--border-color, #404040);
        }
        
        .history-item:last-child {
          border-bottom: none;
        }
        
        .history-tool {
          font-weight: 500;
          color: var(--text-primary, #ffffff);
        }
        
        .history-duration {
          color: var(--text-secondary, #b3b3b3);
        }
        
        .history-time {
          color: var(--text-secondary, #b3b3b3);
          font-size: 0.875rem;
        }
      `}</style>
    </div>
  );
};

export default ToolsPage;