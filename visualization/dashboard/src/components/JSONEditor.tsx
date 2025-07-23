import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Typography, Alert, Tooltip, Tag } from 'antd';
import { 
  FormatPainterOutlined, CheckOutlined, CopyOutlined, 
  ExpandOutlined, CompressOutlined, UndoOutlined, RedoOutlined,
  BugOutlined, InfoCircleOutlined
} from '@ant-design/icons';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

const { Text } = Typography;

interface JSONEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  readOnly?: boolean;
  height?: number;
  showToolbar?: boolean;
  validateOnChange?: boolean;
  onValidationChange?: (isValid: boolean, errors: string[]) => void;
}

interface ValidationError {
  line: number;
  column: number;
  message: string;
}

export const JSONEditor: React.FC<JSONEditorProps> = ({
  value,
  onChange,
  placeholder = '{\n  \n}',
  readOnly = false,
  height = 300,
  showToolbar = true,
  validateOnChange = true,
  onValidationChange
}) => {
  const [isValid, setIsValid] = useState(true);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);
  const [history, setHistory] = useState<string[]>([value]);
  const [historyIndex, setHistoryIndex] = useState(0);

  useEffect(() => {
    if (validateOnChange) {
      validateJSON(value);
    }
  }, [value, validateOnChange]);

  const validateJSON = (jsonString: string) => {
    if (!jsonString.trim()) {
      setIsValid(true);
      setValidationErrors([]);
      onValidationChange?.(true, []);
      return;
    }

    try {
      JSON.parse(jsonString);
      setIsValid(true);
      setValidationErrors([]);
      onValidationChange?.(true, []);
    } catch (error) {
      setIsValid(false);
      const errorMsg = error instanceof Error ? error.message : 'Invalid JSON';
      
      // Extract line and column from error message if possible
      const lineMatch = errorMsg.match(/line (\d+)/);
      const columnMatch = errorMsg.match(/column (\d+)/);
      const line = lineMatch ? parseInt(lineMatch[1]) : 1;
      const column = columnMatch ? parseInt(columnMatch[1]) : 1;
      
      const validationError: ValidationError = {
        line,
        column,
        message: errorMsg
      };
      
      setValidationErrors([validationError]);
      onValidationChange?.(false, [errorMsg]);
    }
  };

  const formatJSON = () => {
    try {
      const parsed = JSON.parse(value);
      const formatted = JSON.stringify(parsed, null, 2);
      updateValue(formatted);
    } catch (error) {
      // If parsing fails, just clean up whitespace
      const cleaned = value.replace(/\s+/g, ' ').trim();
      updateValue(cleaned);
    }
  };

  const minifyJSON = () => {
    try {
      const parsed = JSON.parse(value);
      const minified = JSON.stringify(parsed);
      updateValue(minified);
    } catch (error) {
      // If parsing fails, just remove extra whitespace
      const minified = value.replace(/\s+/g, ' ').replace(/\s*([{}[\]:,])\s*/g, '$1');
      updateValue(minified);
    }
  };

  const updateValue = (newValue: string) => {
    // Add to history if different from current value
    if (newValue !== value) {
      const newHistory = history.slice(0, historyIndex + 1);
      newHistory.push(newValue);
      setHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
    }
    onChange(newValue);
  };

  const handleUndo = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      onChange(history[newIndex]);
    }
  };

  const handleRedo = () => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      onChange(history[newIndex]);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(value).then(() => {
      // Could show a notification here
    });
  };

  const handleTextAreaChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = event.target.value;
    updateValue(newValue);
  };

  const getValidationStatus = () => {
    if (!value.trim()) return 'empty';
    return isValid ? 'valid' : 'invalid';
  };

  const renderToolbar = () => {
    if (!showToolbar) return null;

    const status = getValidationStatus();

    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        padding: '8px 12px',
        borderBottom: '1px solid #f0f0f0',
        background: '#fafafa'
      }}>
        <Space>
          <Tooltip title="Format JSON">
            <Button 
              size="small" 
              icon={<FormatPainterOutlined />} 
              onClick={formatJSON}
              disabled={readOnly || status === 'invalid'}
            >
              Format
            </Button>
          </Tooltip>
          
          <Tooltip title="Minify JSON">
            <Button 
              size="small" 
              icon={<CompressOutlined />} 
              onClick={minifyJSON}
              disabled={readOnly || status === 'invalid'}
            >
              Minify
            </Button>
          </Tooltip>
          
          <Tooltip title="Copy to clipboard">
            <Button 
              size="small" 
              icon={<CopyOutlined />} 
              onClick={copyToClipboard}
            >
              Copy
            </Button>
          </Tooltip>
          
          <div style={{ width: 1, height: 16, background: '#d9d9d9', margin: '0 4px' }} />
          
          <Tooltip title="Undo">
            <Button 
              size="small" 
              icon={<UndoOutlined />} 
              onClick={handleUndo}
              disabled={readOnly || historyIndex <= 0}
            />
          </Tooltip>
          
          <Tooltip title="Redo">
            <Button 
              size="small" 
              icon={<RedoOutlined />} 
              onClick={handleRedo}
              disabled={readOnly || historyIndex >= history.length - 1}
            />
          </Tooltip>
        </Space>
        
        <Space>
          {status === 'valid' && value.trim() && (
            <Tag color="success" icon={<CheckOutlined />}>
              Valid JSON
            </Tag>
          )}
          {status === 'invalid' && (
            <Tag color="error" icon={<BugOutlined />}>
              Invalid JSON
            </Tag>
          )}
          {status === 'empty' && (
            <Tag icon={<InfoCircleOutlined />}>
              Empty
            </Tag>
          )}
          
          <Tooltip title={isExpanded ? "Collapse" : "Expand"}>
            <Button 
              size="small" 
              icon={isExpanded ? <CompressOutlined /> : <ExpandOutlined />} 
              onClick={() => setIsExpanded(!isExpanded)}
            />
          </Tooltip>
        </Space>
      </div>
    );
  };

  const renderEditor = () => {
    const editorHeight = isExpanded ? height * 2 : height;

    if (readOnly && value.trim()) {
      // Read-only mode with syntax highlighting
      return (
        <SyntaxHighlighter
          language="json"
          style={tomorrow}
          customStyle={{
            margin: 0,
            border: 'none',
            borderRadius: 0,
            background: 'transparent',
            height: editorHeight,
            overflow: 'auto'
          }}
          showLineNumbers
          wrapLines
        >
          {value || '{}'}
        </SyntaxHighlighter>
      );
    }

    return (
      <textarea
        value={value}
        onChange={handleTextAreaChange}
        placeholder={placeholder}
        readOnly={readOnly}
        style={{
          width: '100%',
          height: editorHeight,
          border: 'none',
          outline: 'none',
          resize: 'none',
          fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
          fontSize: '13px',
          lineHeight: '1.5',
          padding: '12px',
          background: readOnly ? '#f5f5f5' : 'transparent',
          color: readOnly ? '#666' : '#000'
        }}
      />
    );
  };

  const renderValidationErrors = () => {
    if (isValid || validationErrors.length === 0) return null;

    return (
      <div style={{ padding: '8px 12px', borderTop: '1px solid #f0f0f0' }}>
        <Alert
          type="error"
          showIcon
          message="JSON Validation Error"
          description={
            <div style={{ fontSize: '12px', marginTop: 4 }}>
              {validationErrors.map((error, index) => (
                <div key={index}>
                  Line {error.line}, Column {error.column}: {error.message}
                </div>
              ))}
            </div>
          }
        />
      </div>
    );
  };

  return (
    <div style={{ 
      border: '1px solid #d9d9d9', 
      borderRadius: 6,
      background: '#fff',
      position: 'relative'
    }}>
      {renderToolbar()}
      
      <div style={{ 
        position: 'relative',
        background: readOnly ? '#fafafa' : '#fff'
      }}>
        {renderEditor()}
      </div>
      
      {renderValidationErrors()}
      
      {/* Character count */}
      <div style={{
        position: 'absolute',
        bottom: '4px',
        right: '8px',
        fontSize: '11px',
        color: '#999',
        background: 'rgba(255, 255, 255, 0.8)',
        padding: '2px 6px',
        borderRadius: 3
      }}>
        {value.length} chars
      </div>
    </div>
  );
};

export default JSONEditor;