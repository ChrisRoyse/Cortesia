import React, { useState } from 'react';
import { ParameterDoc } from '../../types';
import './ParameterTable.css';

interface ParameterTableProps {
  parameters: ParameterDoc[];
  showExamples?: boolean;
  onParameterClick?: (param: ParameterDoc) => void;
}

export const ParameterTable: React.FC<ParameterTableProps> = ({
  parameters,
  showExamples = true,
  onParameterClick
}) => {
  const [expandedParams, setExpandedParams] = useState<Set<string>>(new Set());

  const toggleExpanded = (paramName: string) => {
    const newExpanded = new Set(expandedParams);
    if (newExpanded.has(paramName)) {
      newExpanded.delete(paramName);
    } else {
      newExpanded.add(paramName);
    }
    setExpandedParams(newExpanded);
  };

  const renderParameterType = (type: string) => {
    // Parse and format complex types
    if (type.includes('|')) {
      const types = type.split('|').map(t => t.trim());
      return (
        <span className="param-type union-type">
          {types.map((t, index) => (
            <React.Fragment key={index}>
              {index > 0 && <span className="type-separator">|</span>}
              <span className="type-part">{t}</span>
            </React.Fragment>
          ))}
        </span>
      );
    }

    if (type.endsWith('[]')) {
      return (
        <span className="param-type array-type">
          <span className="type-part">{type.slice(0, -2)}</span>
          <span className="array-brackets">[]</span>
        </span>
      );
    }

    return <span className="param-type">{type}</span>;
  };

  const renderExamples = (examples: any[]) => {
    if (!examples || examples.length === 0) return null;

    return (
      <div className="param-examples">
        <div className="examples-label">Examples:</div>
        <div className="example-values">
          {examples.map((example, index) => (
            <code key={index} className="example-value">
              {JSON.stringify(example)}
            </code>
          ))}
        </div>
      </div>
    );
  };

  if (parameters.length === 0) {
    return (
      <div className="parameters-empty">
        <p>This tool does not require any parameters.</p>
      </div>
    );
  }

  return (
    <div className="parameter-table-container">
      <table className="parameter-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Required</th>
            <th>Description</th>
            {showExamples && <th>Details</th>}
          </tr>
        </thead>
        <tbody>
          {parameters.map((param) => {
            const isExpanded = expandedParams.has(param.name);
            const hasDetails = param.default !== undefined || (param.examples && param.examples.length > 0);

            return (
              <React.Fragment key={param.name}>
                <tr 
                  className={`parameter-row ${onParameterClick ? 'clickable' : ''}`}
                  onClick={() => onParameterClick?.(param)}
                >
                  <td className="param-name">
                    <code>{param.name}</code>
                  </td>
                  <td className="param-type-cell">
                    {renderParameterType(param.type)}
                  </td>
                  <td className="param-required">
                    <span className={`required-badge ${param.required ? 'required' : 'optional'}`}>
                      {param.required ? 'Required' : 'Optional'}
                    </span>
                  </td>
                  <td className="param-description">
                    {param.description}
                    {param.default !== undefined && (
                      <div className="param-default">
                        Default: <code>{JSON.stringify(param.default)}</code>
                      </div>
                    )}
                  </td>
                  {showExamples && (
                    <td className="param-actions">
                      {hasDetails && (
                        <button
                          className="expand-button"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleExpanded(param.name);
                          }}
                        >
                          {isExpanded ? '−' : '+'}
                        </button>
                      )}
                    </td>
                  )}
                </tr>
                {isExpanded && hasDetails && (
                  <tr className="parameter-details-row">
                    <td colSpan={showExamples ? 5 : 4}>
                      <div className="parameter-details">
                        {param.examples && showExamples && renderExamples(param.examples)}
                        
                        {/* Additional details based on parameter type */}
                        {param.type.includes('object') && (
                          <div className="param-schema-info">
                            <div className="schema-label">Object Schema:</div>
                            <pre className="schema-content">
                              {`{
  // Add specific object properties here
  // This would be parsed from the schema
}`}
                            </pre>
                          </div>
                        )}

                        {param.type.includes('array') && (
                          <div className="param-constraints">
                            <div className="constraint">
                              <strong>Array constraints:</strong> Can contain multiple items of type {param.type.replace('[]', '')}
                            </div>
                          </div>
                        )}

                        {(param.name.includes('query') || param.name.includes('pattern')) && (
                          <div className="param-syntax-help">
                            <div className="syntax-label">Query Syntax:</div>
                            <ul className="syntax-examples">
                              <li><code>entity:name</code> - Find entities by name</li>
                              <li><code>relation:type</code> - Filter by relationship type</li>
                              <li><code>property:value</code> - Match property values</li>
                              <li><code>*</code> - Wildcard matching</li>
                            </ul>
                          </div>
                        )}

                        {param.name.includes('threshold') && (
                          <div className="param-range-info">
                            <div className="range-label">Valid Range:</div>
                            <div className="range-values">
                              Typically between 0.0 and 1.0 (confidence/probability values)
                            </div>
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>

      <div className="parameter-notes">
        <h4>Notes:</h4>
        <ul>
          <li>Parameters marked as required must be provided for the tool to execute successfully.</li>
          <li>Optional parameters will use their default values if not specified.</li>
          <li>All string parameters support UTF-8 encoding.</li>
          <li>Array parameters can be empty unless otherwise specified.</li>
        </ul>
      </div>
    </div>
  );
};