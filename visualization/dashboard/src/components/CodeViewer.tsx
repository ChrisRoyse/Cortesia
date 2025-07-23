import React, { useState, useEffect, useRef } from 'react';
import { Card, Tooltip, Badge, Button, Space, Select, Spin, Alert } from 'antd';
import { FunctionOutlined, BugOutlined, ThunderboltOutlined, EyeOutlined, LinkOutlined } from '@ant-design/icons';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CodeViewerProps {
  code: string;
  language: string;
  selectedFile: string;
  functionMap?: Record<string, FunctionInfo>;
  onFunctionClick?: (functionName: string) => void;
  showLineNumbers?: boolean;
  highlightLines?: number[];
}

interface FunctionInfo {
  name: string;
  file_path: string;
  line_number: number;
  parameters: string[];
  return_type?: string;
  complexity: number;
  is_public: boolean;
  is_async: boolean;
  calls: string[];
  called_by: string[];
}

interface CodeAnnotation {
  line: number;
  type: 'function' | 'struct' | 'enum' | 'import' | 'comment' | 'complexity' | 'call';
  content: string;
  severity?: 'low' | 'medium' | 'high';
  data?: any;
}

export const CodeViewer: React.FC<CodeViewerProps> = ({
  code,
  language,
  selectedFile,
  functionMap = {},
  onFunctionClick,
  showLineNumbers = true,
  highlightLines = []
}) => {
  const [annotations, setAnnotations] = useState<CodeAnnotation[]>([]);
  const [selectedFunction, setSelectedFunction] = useState<string | null>(null);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [annotationType, setAnnotationType] = useState<'all' | 'functions' | 'complexity' | 'calls'>('all');
  const codeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    analyzeCode();
  }, [code, functionMap, selectedFile]);

  const analyzeCode = () => {
    const lines = code.split('\n');
    const newAnnotations: CodeAnnotation[] = [];

    lines.forEach((line, index) => {
      const lineNumber = index + 1;
      const trimmedLine = line.trim();

      // Function declarations
      if (trimmedLine.includes('fn ') && !trimmedLine.startsWith('//')) {
        const functionMatch = trimmedLine.match(/fn\s+(\w+)/);
        if (functionMatch) {
          const funcName = functionMatch[1];
          const funcInfo = functionMap[funcName];
          
          newAnnotations.push({
            line: lineNumber,
            type: 'function',
            content: `Function: ${funcName}`,
            severity: funcInfo?.complexity > 10 ? 'high' : funcInfo?.complexity > 5 ? 'medium' : 'low',
            data: funcInfo
          });
        }
      }

      // Struct declarations
      if (trimmedLine.includes('struct ') && !trimmedLine.startsWith('//')) {
        const structMatch = trimmedLine.match(/struct\s+(\w+)/);
        if (structMatch) {
          newAnnotations.push({
            line: lineNumber,
            type: 'struct',
            content: `Struct: ${structMatch[1]}`,
            data: { name: structMatch[1] }
          });
        }
      }

      // Enum declarations
      if (trimmedLine.includes('enum ') && !trimmedLine.startsWith('//')) {
        const enumMatch = trimmedLine.match(/enum\s+(\w+)/);
        if (enumMatch) {
          newAnnotations.push({
            line: lineNumber,
            type: 'enum',
            content: `Enum: ${enumMatch[1]}`,
            data: { name: enumMatch[1] }
          });
        }
      }

      // Import/use statements
      if (trimmedLine.startsWith('use ') || trimmedLine.startsWith('import ')) {
        newAnnotations.push({
          line: lineNumber,
          type: 'import',
          content: 'Import/Use statement',
          data: { statement: trimmedLine }
        });
      }

      // Function calls
      const callMatches = trimmedLine.match(/(\w+)\s*\(/g);
      if (callMatches) {
        callMatches.forEach(match => {
          const funcName = match.replace(/\s*\($/, '');
          if (functionMap[funcName]) {
            newAnnotations.push({
              line: lineNumber,
              type: 'call',
              content: `Calls: ${funcName}`,
              data: { calledFunction: funcName, functionInfo: functionMap[funcName] }
            });
          }
        });
      }

      // Complexity indicators
      if (trimmedLine.includes('if ') || trimmedLine.includes('for ') || 
          trimmedLine.includes('while ') || trimmedLine.includes('match ')) {
        newAnnotations.push({
          line: lineNumber,
          type: 'complexity',
          content: 'Complexity contributor',
          severity: 'low',
          data: { type: 'control_flow' }
        });
      }
    });

    setAnnotations(newAnnotations);
  };

  const getFilteredAnnotations = () => {
    switch (annotationType) {
      case 'functions':
        return annotations.filter(a => a.type === 'function');
      case 'complexity':
        return annotations.filter(a => a.type === 'complexity' || (a.type === 'function' && a.severity === 'high'));
      case 'calls':
        return annotations.filter(a => a.type === 'call');
      default:
        return annotations;
    }
  };

  const getLineProps = (lineNumber: number) => {
    const lineAnnotations = annotations.filter(a => a.line === lineNumber);
    const isHighlighted = highlightLines.includes(lineNumber);
    const hasHighComplexity = lineAnnotations.some(a => a.severity === 'high');
    
    const style: React.CSSProperties = {};
    
    if (isHighlighted) {
      style.backgroundColor = '#fff3cd';
    } else if (hasHighComplexity) {
      style.backgroundColor = '#f8d7da';
    }
    
    return { style };
  };

  const renderAnnotationTooltip = (lineNumber: number) => {
    const lineAnnotations = getFilteredAnnotations().filter(a => a.line === lineNumber);
    if (lineAnnotations.length === 0) return null;

    return (
      <div style={{ maxWidth: 300 }}>
        {lineAnnotations.map((annotation, index) => (
          <div key={index} style={{ marginBottom: 8 }}>
            <div style={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 4 }}>
              {annotation.type === 'function' && <FunctionOutlined />}
              {annotation.type === 'complexity' && <ThunderboltOutlined />}
              {annotation.type === 'call' && <LinkOutlined />}
              {annotation.content}
            </div>
            {annotation.data && (
              <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                {annotation.type === 'function' && annotation.data.complexity && (
                  <div>Complexity: {annotation.data.complexity}</div>
                )}
                {annotation.type === 'function' && annotation.data.parameters && (
                  <div>Parameters: {annotation.data.parameters.join(', ')}</div>
                )}
                {annotation.type === 'call' && annotation.data.calledFunction && (
                  <div>
                    Calls: {annotation.data.calledFunction}
                    <br />
                    <Button 
                      size="small" 
                      type="link" 
                      onClick={() => onFunctionClick?.(annotation.data.calledFunction)}
                    >
                      Go to definition
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderLineNumbers = () => {
    if (!showLineNumbers) return null;
    
    const lines = code.split('\n');
    return (
      <div style={{ 
        borderRight: '1px solid #e1e1e1', 
        paddingRight: 8, 
        marginRight: 8,
        minWidth: 40,
        textAlign: 'right',
        color: '#666',
        fontSize: '12px',
        lineHeight: '1.5em',
        userSelect: 'none'
      }}>
        {lines.map((_, index) => {
          const lineNumber = index + 1;
          const lineAnnotations = getFilteredAnnotations().filter(a => a.line === lineNumber);
          const hasAnnotations = lineAnnotations.length > 0;
          
          return (
            <div 
              key={lineNumber} 
              style={{ 
                minHeight: '1.5em',
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end'
              }}
            >
              <span>{lineNumber}</span>
              {hasAnnotations && showAnnotations && (
                <Tooltip title={renderAnnotationTooltip(lineNumber)} placement="left">
                  <div style={{
                    position: 'absolute',
                    right: -20,
                    width: 6,
                    height: 6,
                    borderRadius: '50%',
                    backgroundColor: lineAnnotations.some(a => a.severity === 'high') ? '#ff4d4f' : 
                                     lineAnnotations.some(a => a.severity === 'medium') ? '#faad14' : '#52c41a'
                  }} />
                </Tooltip>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const customStyle = {
    ...tomorrow,
    'pre[class*="language-"]': {
      ...tomorrow['pre[class*="language-"]'],
      margin: 0,
      padding: 0,
      background: 'transparent',
    },
    'code[class*="language-"]': {
      ...tomorrow['code[class*="language-"]'],
      background: 'transparent',
    }
  };

  return (
    <Card
      title={
        <Space>
          <EyeOutlined />
          Code Viewer
          {selectedFile && (
            <span style={{ fontSize: '14px', color: '#666' }}>
              - {selectedFile.split('/').pop() || selectedFile.split('\\').pop()}
            </span>
          )}
        </Space>
      }
      extra={
        <Space>
          <Select
            value={annotationType}
            onChange={setAnnotationType}
            size="small"
            style={{ minWidth: 120 }}
          >
            <Select.Option value="all">All Annotations</Select.Option>
            <Select.Option value="functions">Functions</Select.Option>
            <Select.Option value="complexity">Complexity</Select.Option>
            <Select.Option value="calls">Function Calls</Select.Option>
          </Select>
          <Button
            size="small"
            type={showAnnotations ? 'primary' : 'default'}
            onClick={() => setShowAnnotations(!showAnnotations)}
          >
            Annotations
          </Button>
        </Space>
      }
      style={{ height: '100%' }}
    >
      <div style={{ 
        height: 'calc(100% - 60px)', 
        overflow: 'auto', 
        border: '1px solid #f0f0f0',
        borderRadius: 6,
        backgroundColor: '#fafafa'
      }}>
        {code ? (
          <div style={{ display: 'flex', padding: 12 }}>
            {renderLineNumbers()}
            <div style={{ flex: 1, overflow: 'auto' }}>
              <SyntaxHighlighter
                language={language}
                style={customStyle}
                showLineNumbers={false}
                wrapLines
                lineProps={getLineProps}
                customStyle={{
                  background: 'transparent',
                  padding: 0,
                  margin: 0,
                  fontSize: '13px',
                  lineHeight: '1.5em'
                }}
              >
                {code}
              </SyntaxHighlighter>
            </div>
          </div>
        ) : (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            height: '100%',
            color: '#666'
          }}>
            <div style={{ textAlign: 'center' }}>
              <EyeOutlined style={{ fontSize: 48, marginBottom: 16 }} />
              <div>No file selected</div>
              <div style={{ fontSize: '12px', marginTop: 8 }}>
                Select a file from the explorer to view its content
              </div>
            </div>
          </div>
        )}
      </div>
      
      {showAnnotations && getFilteredAnnotations().length > 0 && (
        <div style={{ 
          position: 'absolute', 
          bottom: 16, 
          right: 16, 
          background: 'rgba(255, 255, 255, 0.9)',
          padding: 8,
          borderRadius: 4,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          fontSize: '12px'
        }}>
          <Space>
            <Badge count={getFilteredAnnotations().filter(a => a.type === 'function').length} style={{ backgroundColor: '#52c41a' }}>
              <FunctionOutlined title="Functions" />
            </Badge>
            <Badge count={getFilteredAnnotations().filter(a => a.severity === 'high').length} style={{ backgroundColor: '#ff4d4f' }}>
              <BugOutlined title="High Complexity" />
            </Badge>
            <Badge count={getFilteredAnnotations().filter(a => a.type === 'call').length} style={{ backgroundColor: '#1890ff' }}>
              <LinkOutlined title="Function Calls" />
            </Badge>
          </Space>
        </div>
      )}
    </Card>
  );
};

export default CodeViewer;