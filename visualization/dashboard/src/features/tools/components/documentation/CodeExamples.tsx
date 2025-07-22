import React, { useState } from 'react';
import { CodeExample, MCPTool, ToolExample } from '../../types';
import './CodeExamples.css';

interface CodeExamplesProps {
  examples: CodeExample[];
  tool: MCPTool;
  defaultLanguage?: 'javascript' | 'python' | 'curl' | 'rust';
  onTryExample?: (example: ToolExample) => void;
}

export const CodeExamples: React.FC<CodeExamplesProps> = ({
  examples,
  tool,
  defaultLanguage = 'javascript',
  onTryExample
}) => {
  const [selectedLanguage, setSelectedLanguage] = useState(defaultLanguage);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const handleCopyCode = async (code: string, index: number) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const currentExample = examples.find(ex => ex.language === selectedLanguage) || examples[0];

  const renderLanguageSelector = () => (
    <div className="language-selector">
      {examples.map(example => (
        <button
          key={example.language}
          className={`language-option ${selectedLanguage === example.language ? 'active' : ''}`}
          onClick={() => setSelectedLanguage(example.language)}
        >
          {getLanguageLabel(example.language)}
        </button>
      ))}
    </div>
  );

  const getLanguageLabel = (language: string): string => {
    const labels: Record<string, string> = {
      javascript: 'JavaScript',
      python: 'Python',
      curl: 'cURL',
      rust: 'Rust'
    };
    return labels[language] || language;
  };

  const renderToolExamples = () => {
    if (!tool.examples || tool.examples.length === 0) return null;

    return (
      <div className="tool-examples-section">
        <h3>Example Inputs & Outputs</h3>
        <div className="example-cards">
          {tool.examples.map((example, index) => (
            <div key={index} className="example-card">
              <div className="example-header">
                <h4>{example.name}</h4>
                {onTryExample && (
                  <button
                    className="try-example-btn"
                    onClick={() => onTryExample(example)}
                  >
                    Try This Example
                  </button>
                )}
              </div>
              <p className="example-description">{example.description}</p>
              
              <div className="example-code-block">
                <div className="code-label">Input:</div>
                <pre className="code-content">
                  <code>{JSON.stringify(example.input, null, 2)}</code>
                </pre>
              </div>

              {example.output && (
                <div className="example-code-block">
                  <div className="code-label">Expected Output:</div>
                  <pre className="code-content">
                    <code>{JSON.stringify(example.output, null, 2)}</code>
                  </pre>
                </div>
              )}

              {example.tags && example.tags.length > 0 && (
                <div className="example-tags">
                  {example.tags.map(tag => (
                    <span key={tag} className="example-tag">{tag}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderCodeExample = () => {
    if (!currentExample) return null;

    const index = examples.indexOf(currentExample);

    return (
      <div className="code-example">
        <div className="code-header">
          <span className="code-description">
            {currentExample.description || `${getLanguageLabel(currentExample.language)} Example`}
          </span>
          <button
            className="copy-button"
            onClick={() => handleCopyCode(currentExample.code, index)}
          >
            {copiedIndex === index ? 'Copied!' : 'Copy'}
          </button>
        </div>
        <pre className="code-block">
          <code className={`language-${currentExample.language}`}>
            {currentExample.code}
          </code>
        </pre>
      </div>
    );
  };

  const renderQuickStart = () => (
    <div className="quick-start-section">
      <h3>Quick Start</h3>
      <div className="quick-start-steps">
        <div className="step">
          <div className="step-number">1</div>
          <div className="step-content">
            <h4>Install the SDK</h4>
            <pre className="inline-code">
              {selectedLanguage === 'javascript' && 'npm install @llmkg/client'}
              {selectedLanguage === 'python' && 'pip install llmkg-client'}
              {selectedLanguage === 'rust' && 'cargo add llmkg-client'}
              {selectedLanguage === 'curl' && 'No installation required'}
            </pre>
          </div>
        </div>

        <div className="step">
          <div className="step-number">2</div>
          <div className="step-content">
            <h4>Set up authentication</h4>
            <pre className="inline-code">
              {selectedLanguage === 'curl' 
                ? 'export LLMKG_API_KEY="your-api-key"'
                : 'Set LLMKG_API_KEY environment variable'}
            </pre>
          </div>
        </div>

        <div className="step">
          <div className="step-number">3</div>
          <div className="step-content">
            <h4>Run the example</h4>
            <p>Copy the code above and run it in your environment</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderErrorHandling = () => (
    <div className="error-handling-section">
      <h3>Error Handling</h3>
      <p>Common error scenarios and how to handle them:</p>
      <div className="error-examples">
        <div className="error-example">
          <h4>Rate Limiting</h4>
          <p>When you exceed the rate limit, you'll receive a 429 status code.</p>
          <pre className="error-code">
            {selectedLanguage === 'javascript' && `catch (error) {
  if (error.code === 'RATE_LIMIT') {
    console.log('Rate limited, retry after:', error.retryAfter);
    await sleep(error.retryAfter * 1000);
    // Retry the request
  }
}`}
            {selectedLanguage === 'python' && `except RateLimitError as e:
    print(f"Rate limited, retry after: {e.retry_after}")
    time.sleep(e.retry_after)
    # Retry the request`}
            {selectedLanguage === 'rust' && `Err(Error::RateLimit { retry_after }) => {
    println!("Rate limited, retry after: {} seconds", retry_after);
    tokio::time::sleep(Duration::from_secs(retry_after)).await;
    // Retry the request
}`}
            {selectedLanguage === 'curl' && `# Check for 429 status code
if [ $http_code -eq 429 ]; then
  retry_after=$(curl -s -I ... | grep -i "retry-after" | cut -d' ' -f2)
  sleep $retry_after
  # Retry the request
fi`}
          </pre>
        </div>

        <div className="error-example">
          <h4>Invalid Parameters</h4>
          <p>Validation errors return a 400 status code with details.</p>
          <pre className="error-code">
            {selectedLanguage === 'javascript' && `catch (error) {
  if (error.code === 'VALIDATION_ERROR') {
    console.error('Invalid parameters:', error.details);
    // Fix the parameters and retry
  }
}`}
            {selectedLanguage === 'python' && `except ValidationError as e:
    print(f"Invalid parameters: {e.details}")
    # Fix the parameters and retry`}
            {selectedLanguage === 'rust' && `Err(Error::Validation { details }) => {
    eprintln!("Invalid parameters: {:?}", details);
    // Fix the parameters and retry
}`}
            {selectedLanguage === 'curl' && `# Parse error response
error_msg=$(echo $response | jq -r '.error.message')
echo "Validation error: $error_msg"`}
          </pre>
        </div>
      </div>
    </div>
  );

  return (
    <div className="code-examples-container">
      {renderLanguageSelector()}
      {renderCodeExample()}
      {renderQuickStart()}
      {renderToolExamples()}
      {renderErrorHandling()}
    </div>
  );
};