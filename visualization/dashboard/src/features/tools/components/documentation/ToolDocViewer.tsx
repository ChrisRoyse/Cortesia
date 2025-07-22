import React, { useState, useEffect } from 'react';
import { MCPTool, ToolDocumentation } from '../../types';
import { documentationGenerator } from '../../services/DocumentationGenerator';
import { CodeExamples } from './CodeExamples';
import { ParameterTable } from './ParameterTable';
import { ApiReference } from './ApiReference';
import './ToolDocViewer.css';

interface ToolDocViewerProps {
  tool: MCPTool;
  showExamples?: boolean;
  showApiReference?: boolean;
  language?: 'javascript' | 'python' | 'curl' | 'rust';
  onRelatedToolClick?: (toolId: string) => void;
}

export const ToolDocViewer: React.FC<ToolDocViewerProps> = ({
  tool,
  showExamples = true,
  showApiReference = true,
  language = 'javascript',
  onRelatedToolClick
}) => {
  const [documentation, setDocumentation] = useState<ToolDocumentation | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'examples' | 'api'>('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      setIsLoading(true);
      const doc = documentationGenerator.generateDocumentation(tool);
      setDocumentation(doc);
      setError(null);
    } catch (err) {
      setError('Failed to generate documentation');
      console.error('Documentation generation error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [tool]);

  if (isLoading) {
    return (
      <div className="tool-doc-viewer loading">
        <div className="spinner">Generating documentation...</div>
      </div>
    );
  }

  if (error || !documentation) {
    return (
      <div className="tool-doc-viewer error">
        <div className="error-message">{error || 'No documentation available'}</div>
      </div>
    );
  }

  const renderBreadcrumb = () => (
    <div className="doc-breadcrumb">
      <span className="breadcrumb-item">Tools</span>
      <span className="breadcrumb-separator">/</span>
      <span className="breadcrumb-item">{tool.category}</span>
      <span className="breadcrumb-separator">/</span>
      <span className="breadcrumb-item active">{tool.name}</span>
    </div>
  );

  const renderHeader = () => (
    <div className="doc-header">
      <div className="doc-title-section">
        <h1 className="doc-title">{tool.name}</h1>
        <span className="doc-version">v{tool.version}</span>
        <span className={`doc-status status-${tool.status}`}>{tool.status}</span>
      </div>
      <p className="doc-summary">{documentation.summary}</p>
      {tool.tags && tool.tags.length > 0 && (
        <div className="doc-tags">
          {tool.tags.map(tag => (
            <span key={tag} className="doc-tag">{tag}</span>
          ))}
        </div>
      )}
    </div>
  );

  const renderTabs = () => (
    <div className="doc-tabs">
      <button
        className={`doc-tab ${activeTab === 'overview' ? 'active' : ''}`}
        onClick={() => setActiveTab('overview')}
      >
        Overview
      </button>
      {showExamples && (
        <button
          className={`doc-tab ${activeTab === 'examples' ? 'active' : ''}`}
          onClick={() => setActiveTab('examples')}
        >
          Examples
        </button>
      )}
      {showApiReference && (
        <button
          className={`doc-tab ${activeTab === 'api' ? 'active' : ''}`}
          onClick={() => setActiveTab('api')}
        >
          API Reference
        </button>
      )}
    </div>
  );

  const renderOverview = () => (
    <div className="doc-overview">
      <section className="doc-section">
        <h2>Description</h2>
        <div className="doc-description">
          {documentation.description.split('\n\n').map((paragraph, index) => (
            <p key={index}>{paragraph}</p>
          ))}
        </div>
      </section>

      <section className="doc-section">
        <h2>Parameters</h2>
        <ParameterTable parameters={documentation.parameters} />
      </section>

      <section className="doc-section">
        <h2>Returns</h2>
        <div className="doc-returns">
          <div className="return-type">
            <strong>Type:</strong> <code>{documentation.returns.type}</code>
          </div>
          <div className="return-description">
            {documentation.returns.description}
          </div>
        </div>
      </section>

      {documentation.relatedTools && documentation.relatedTools.length > 0 && (
        <section className="doc-section">
          <h2>Related Tools</h2>
          <div className="related-tools">
            {documentation.relatedTools.map(relatedId => (
              <button
                key={relatedId}
                className="related-tool-link"
                onClick={() => onRelatedToolClick?.(relatedId)}
              >
                {relatedId}
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="doc-section">
        <h2>Performance Characteristics</h2>
        <div className="performance-info">
          {tool.responseTime && (
            <div className="performance-metric">
              <span className="metric-label">Average Response Time:</span>
              <span className="metric-value">{tool.responseTime}ms</span>
            </div>
          )}
          <div className="performance-metric">
            <span className="metric-label">Rate Limit:</span>
            <span className="metric-value">100 requests/minute</span>
          </div>
          <div className="performance-metric">
            <span className="metric-label">Timeout:</span>
            <span className="metric-value">30 seconds</span>
          </div>
        </div>
      </section>

      <section className="doc-section">
        <h2>Best Practices</h2>
        <div className="best-practices">
          <ul>
            <li>Always validate input parameters before sending requests</li>
            <li>Implement proper error handling for network and tool-specific errors</li>
            <li>Use pagination for large result sets when available</li>
            <li>Cache results when appropriate to reduce API calls</li>
            {tool.category === 'knowledge-graph' && (
              <>
                <li>Use specific entity IDs when possible instead of broad queries</li>
                <li>Leverage relationship filters to narrow down results</li>
              </>
            )}
            {tool.category === 'cognitive' && (
              <>
                <li>Provide sufficient context for pattern recognition</li>
                <li>Use confidence thresholds to filter results</li>
              </>
            )}
            {tool.category === 'neural' && (
              <>
                <li>Monitor neural activity in real-time for dynamic systems</li>
                <li>Aggregate data over time windows for analysis</li>
              </>
            )}
          </ul>
        </div>
      </section>
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'examples':
        return showExamples ? (
          <CodeExamples
            examples={documentation.examples}
            tool={tool}
            defaultLanguage={language}
          />
        ) : null;
      case 'api':
        return showApiReference ? (
          <ApiReference tool={tool} />
        ) : null;
      default:
        return renderOverview();
    }
  };

  return (
    <div className="tool-doc-viewer">
      {renderBreadcrumb()}
      {renderHeader()}
      {renderTabs()}
      <div className="doc-content">
        {renderContent()}
      </div>
    </div>
  );
};