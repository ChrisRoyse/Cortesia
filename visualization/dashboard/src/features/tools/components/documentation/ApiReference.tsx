import React, { useState } from 'react';
import { MCPTool } from '../../types';
import { documentationGenerator } from '../../services/DocumentationGenerator';
import './ApiReference.css';

interface ApiReferenceProps {
  tool: MCPTool;
  baseUrl?: string;
  showAuthentication?: boolean;
}

export const ApiReference: React.FC<ApiReferenceProps> = ({
  tool,
  baseUrl = 'https://api.llmkg.com',
  showAuthentication = true
}) => {
  const [activeSection, setActiveSection] = useState<'request' | 'response' | 'errors'>('request');
  const apiRef = documentationGenerator.generateApiReference(tool);

  const renderEndpointInfo = () => (
    <div className="api-endpoint-section">
      <h3>Endpoint</h3>
      <div className="endpoint-box">
        <span className="http-method">{apiRef.method}</span>
        <code className="endpoint-url">{baseUrl}{apiRef.endpoint}</code>
      </div>
      
      {tool.version && (
        <div className="version-info">
          <strong>API Version:</strong> {tool.version}
        </div>
      )}
    </div>
  );

  const renderAuthentication = () => {
    if (!showAuthentication) return null;

    return (
      <div className="api-auth-section">
        <h3>Authentication</h3>
        <div className="auth-info">
          <p>{apiRef.authentication}</p>
          <div className="auth-example">
            <div className="auth-header">
              <strong>Header Example:</strong>
            </div>
            <pre className="auth-code">
              <code>{`Authorization: Bearer YOUR_API_KEY`}</code>
            </pre>
          </div>
          <div className="auth-note">
            <strong>Note:</strong> API keys can be obtained from your LLMKG dashboard.
          </div>
        </div>
      </div>
    );
  };

  const renderHeaders = () => (
    <div className="api-headers-section">
      <h3>Request Headers</h3>
      <table className="headers-table">
        <thead>
          <tr>
            <th>Header</th>
            <th>Value</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(apiRef.headers || {}).map(([header, value]) => (
            <tr key={header}>
              <td><code>{header}</code></td>
              <td><code>{value}</code></td>
              <td>{getHeaderDescription(header)}</td>
            </tr>
          ))}
          {showAuthentication && (
            <tr>
              <td><code>Authorization</code></td>
              <td><code>Bearer YOUR_API_KEY</code></td>
              <td>Your LLMKG API key for authentication</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );

  const getHeaderDescription = (header: string): string => {
    const descriptions: Record<string, string> = {
      'Content-Type': 'The media type of the request body',
      'X-Tool-Version': 'Specific version of the tool to use',
      'X-Request-ID': 'Optional unique identifier for request tracking',
      'X-Idempotency-Key': 'Optional key for idempotent requests'
    };
    return descriptions[header] || 'Custom header';
  };

  const renderRequestBody = () => {
    if (!apiRef.requestBody) return null;

    return (
      <div className="api-body-section">
        <h3>Request Body</h3>
        <div className="body-schema">
          <pre className="schema-code">
            <code>{JSON.stringify(apiRef.requestBody, null, 2)}</code>
          </pre>
        </div>
        
        <div className="request-example">
          <h4>Example Request</h4>
          <pre className="example-code">
            <code>{JSON.stringify({
              toolId: tool.id,
              params: tool.examples?.[0]?.input || generateExampleInput(apiRef.requestBody)
            }, null, 2)}</code>
          </pre>
        </div>
      </div>
    );
  };

  const generateExampleInput = (schema: any): Record<string, any> => {
    const example: Record<string, any> = {};
    
    if (schema.properties) {
      Object.entries(schema.properties).forEach(([key, prop]: [string, any]) => {
        if (prop.examples && prop.examples.length > 0) {
          example[key] = prop.examples[0];
        } else if (prop.enum) {
          example[key] = prop.enum[0];
        } else {
          switch (prop.type) {
            case 'string':
              example[key] = 'example';
              break;
            case 'number':
              example[key] = 42;
              break;
            case 'boolean':
              example[key] = true;
              break;
            case 'array':
              example[key] = [];
              break;
            case 'object':
              example[key] = {};
              break;
            default:
              example[key] = null;
          }
        }
      });
    }
    
    return example;
  };

  const renderResponseBody = () => {
    if (!apiRef.responseBody) return null;

    return (
      <div className="api-response-section">
        <h3>Response Body</h3>
        <div className="response-schema">
          <pre className="schema-code">
            <code>{JSON.stringify(apiRef.responseBody, null, 2)}</code>
          </pre>
        </div>
        
        <div className="response-example">
          <h4>Example Response</h4>
          <pre className="example-code">
            <code>{JSON.stringify({
              success: true,
              data: tool.examples?.[0]?.output || generateExampleOutput(apiRef.responseBody),
              metadata: {
                executionTime: tool.responseTime || 150,
                timestamp: new Date().toISOString(),
                version: tool.version
              }
            }, null, 2)}</code>
          </pre>
        </div>
      </div>
    );
  };

  const generateExampleOutput = (schema: any): any => {
    switch (schema.type) {
      case 'object':
        return { result: 'Success', details: {} };
      case 'array':
        return [{ id: '1', value: 'Example' }];
      case 'string':
        return 'Example output';
      case 'number':
        return 42;
      case 'boolean':
        return true;
      default:
        return null;
    }
  };

  const renderErrorResponses = () => (
    <div className="api-errors-section">
      <h3>Error Responses</h3>
      <div className="error-codes">
        <div className="error-code-item">
          <div className="error-header">
            <span className="error-status">400</span>
            <span className="error-name">Bad Request</span>
          </div>
          <p className="error-description">Invalid parameters or request format</p>
          <pre className="error-example">
            <code>{JSON.stringify({
              error: {
                code: 'VALIDATION_ERROR',
                message: 'Invalid parameter: threshold must be between 0 and 1',
                details: {
                  parameter: 'threshold',
                  value: 2.5,
                  constraints: { min: 0, max: 1 }
                }
              }
            }, null, 2)}</code>
          </pre>
        </div>

        <div className="error-code-item">
          <div className="error-header">
            <span className="error-status">401</span>
            <span className="error-name">Unauthorized</span>
          </div>
          <p className="error-description">Missing or invalid API key</p>
          <pre className="error-example">
            <code>{JSON.stringify({
              error: {
                code: 'UNAUTHORIZED',
                message: 'Invalid API key provided'
              }
            }, null, 2)}</code>
          </pre>
        </div>

        <div className="error-code-item">
          <div className="error-header">
            <span className="error-status">429</span>
            <span className="error-name">Too Many Requests</span>
          </div>
          <p className="error-description">Rate limit exceeded</p>
          <pre className="error-example">
            <code>{JSON.stringify({
              error: {
                code: 'RATE_LIMIT',
                message: 'Rate limit exceeded',
                retryAfter: 60
              }
            }, null, 2)}</code>
          </pre>
        </div>

        <div className="error-code-item">
          <div className="error-header">
            <span className="error-status">500</span>
            <span className="error-name">Internal Server Error</span>
          </div>
          <p className="error-description">Server error while processing request</p>
          <pre className="error-example">
            <code>{JSON.stringify({
              error: {
                code: 'INTERNAL_ERROR',
                message: 'An unexpected error occurred',
                requestId: 'req_123456'
              }
            }, null, 2)}</code>
          </pre>
        </div>
      </div>
    </div>
  );

  const renderRateLimits = () => (
    <div className="api-ratelimit-section">
      <h3>Rate Limits</h3>
      <div className="ratelimit-info">
        <p>{apiRef.rateLimit}</p>
        
        <div className="ratelimit-headers">
          <h4>Rate Limit Headers</h4>
          <table className="ratelimit-table">
            <thead>
              <tr>
                <th>Header</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>X-RateLimit-Limit</code></td>
                <td>The rate limit ceiling for that request</td>
              </tr>
              <tr>
                <td><code>X-RateLimit-Remaining</code></td>
                <td>The number of requests left for the time window</td>
              </tr>
              <tr>
                <td><code>X-RateLimit-Reset</code></td>
                <td>The time at which the rate limit window resets (Unix timestamp)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="ratelimit-tips">
          <h4>Tips for Handling Rate Limits</h4>
          <ul>
            <li>Implement exponential backoff when receiving 429 responses</li>
            <li>Use the <code>X-RateLimit-Remaining</code> header to proactively slow down requests</li>
            <li>Cache responses when possible to reduce API calls</li>
            <li>Consider using batch operations for multiple items</li>
          </ul>
        </div>
      </div>
    </div>
  );

  const renderTabs = () => (
    <div className="api-tabs">
      <button
        className={`api-tab ${activeSection === 'request' ? 'active' : ''}`}
        onClick={() => setActiveSection('request')}
      >
        Request
      </button>
      <button
        className={`api-tab ${activeSection === 'response' ? 'active' : ''}`}
        onClick={() => setActiveSection('response')}
      >
        Response
      </button>
      <button
        className={`api-tab ${activeSection === 'errors' ? 'active' : ''}`}
        onClick={() => setActiveSection('errors')}
      >
        Errors
      </button>
    </div>
  );

  return (
    <div className="api-reference-container">
      {renderEndpointInfo()}
      {renderAuthentication()}
      {renderRateLimits()}
      
      {renderTabs()}
      
      <div className="api-tab-content">
        {activeSection === 'request' && (
          <>
            {renderHeaders()}
            {renderRequestBody()}
          </>
        )}
        {activeSection === 'response' && renderResponseBody()}
        {activeSection === 'errors' && renderErrorResponses()}
      </div>

      <div className="api-testing-section">
        <h3>Testing the API</h3>
        <p>You can test this endpoint using the built-in tool tester or with your preferred API client:</p>
        <ul>
          <li>Use the "Try it out" button in the tool catalog</li>
          <li>Test with Postman, Insomnia, or similar tools</li>
          <li>Use the provided code examples in your application</li>
        </ul>
      </div>
    </div>
  );
};