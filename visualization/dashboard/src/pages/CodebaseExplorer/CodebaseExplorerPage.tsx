import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Tree, Typography, Tabs, Spin, Alert, Badge, Progress, Tooltip, Button } from 'antd';
import { FolderOutlined, FileOutlined, FunctionOutlined, ApiOutlined, BugOutlined, ThunderboltOutlined, EyeOutlined, SettingOutlined } from '@ant-design/icons';
import { useRealTimeData } from '../../hooks/useRealTimeData';
import { CodeViewer } from '../../components/CodeViewer';
import { FunctionGraph } from '../../components/FunctionGraph';
import { DependencyGraph } from '../../components/DependencyGraph';
import { ComplexityHeatmap } from '../../components/ComplexityHeatmap';
import type { TreeDataNode } from 'antd/es/tree';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface CodebaseMetrics {
  total_files: number;
  total_lines: number;
  total_functions: number;
  total_structs: number;
  total_enums: number;
  total_modules: number;
  file_structure: FileStructure;
  function_map: Record<string, FunctionInfo>;
  dependency_graph: DependencyGraph;
  complexity_metrics: ComplexityMetrics;
}

interface FileStructure {
  path: string;
  file_type: 'Directory' | 'RustFile' | 'TypeScriptFile' | 'JsonFile' | 'TomlFile' | 'MarkdownFile' | { Other: string };
  size_bytes: number;
  line_count: number;
  children: FileStructure[];
  functions: string[];
  structs: string[];
  enums: string[];
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

interface DependencyGraph {
  modules: Record<string, ModuleInfo>;
  edges: DependencyEdge[];
}

interface ModuleInfo {
  name: string;
  path: string;
  exports: string[];
  imports: string[];
  internal_calls: number;
  external_calls: number;
}

interface DependencyEdge {
  from: string;
  to: string;
  dependency_type: 'Import' | 'FunctionCall' | 'StructUsage' | 'TraitImplementation';
  strength: number;
}

interface ComplexityMetrics {
  cyclomatic_complexity: Record<string, number>;
  cognitive_complexity: Record<string, number>;
  coupling_metrics: Record<string, number>;
  cohesion_metrics: Record<string, number>;
}

export const CodebaseExplorerPage: React.FC = () => {
  const { data: codebaseData, loading, error } = useRealTimeData<CodebaseMetrics>('codebase_metrics');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [selectedFunction, setSelectedFunction] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [expandedKeys, setExpandedKeys] = useState<React.Key[]>([]);
  const [fileContent, setFileContent] = useState<string>('');
  const [loadingFile, setLoadingFile] = useState(false);

  const convertToTreeData = (structure: FileStructure): TreeDataNode[] => {
    const getFileTypeIcon = (fileType: FileStructure['file_type']) => {
      if (fileType === 'Directory') return <FolderOutlined />;
      if (fileType === 'RustFile') return <FileOutlined style={{ color: '#CE422B' }} />;
      if (fileType === 'TypeScriptFile') return <FileOutlined style={{ color: '#3178C6' }} />;
      if (fileType === 'JsonFile') return <FileOutlined style={{ color: '#FCA326' }} />;
      return <FileOutlined />;
    };

    const mapStructure = (item: FileStructure): TreeDataNode => {
      const isDirectory = item.file_type === 'Directory';
      const functionCount = item.functions.length;
      const structCount = item.structs.length;
      const enumCount = item.enums.length;
      
      return {
        title: (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {getFileTypeIcon(item.file_type)}
            <span>{item.path.split('/').pop() || item.path.split('\\').pop()}</span>
            {!isDirectory && (
              <div style={{ display: 'flex', gap: 4 }}>
                {functionCount > 0 && (
                  <Badge count={functionCount} style={{ backgroundColor: '#52c41a' }} size="small" title="Functions" />
                )}
                {structCount > 0 && (
                  <Badge count={structCount} style={{ backgroundColor: '#1890ff' }} size="small" title="Structs" />
                )}
                {enumCount > 0 && (
                  <Badge count={enumCount} style={{ backgroundColor: '#722ed1' }} size="small" title="Enums" />
                )}
              </div>
            )}
            {!isDirectory && (
              <Text type="secondary" style={{ fontSize: '12px' }}>
                ({item.line_count} lines)
              </Text>
            )}
          </div>
        ),
        key: item.path,
        icon: getFileTypeIcon(item.file_type),
        children: item.children.length > 0 ? item.children.map(mapStructure) : undefined,
        isLeaf: item.children.length === 0,
      };
    };

    return [mapStructure(structure)];
  };

  const loadFileContent = async (filePath: string) => {
    setLoadingFile(true);
    try {
      // In a real implementation, this would fetch the file content
      // For now, we'll simulate it
      await new Promise(resolve => setTimeout(resolve, 500));
      const mockContent = `// File: ${filePath}\n// This is mock content for demonstration\n// In the real implementation, this would be the actual file content\n\nfunction example() {\n  console.log("Hello from ${filePath}");\n}`;
      setFileContent(mockContent);
    } catch (err) {
      console.error('Failed to load file content:', err);
      setFileContent('// Failed to load file content');
    } finally {
      setLoadingFile(false);
    }
  };

  const handleFileSelect = (selectedKeys: React.Key[]) => {
    if (selectedKeys.length > 0) {
      const filePath = selectedKeys[0] as string;
      setSelectedFile(filePath);
      
      // Only load content for actual files, not directories
      if (codebaseData?.file_structure) {
        const findFile = (structure: FileStructure, path: string): FileStructure | null => {
          if (structure.path === path) return structure;
          for (const child of structure.children) {
            const found = findFile(child, path);
            if (found) return found;
          }
          return null;
        };
        
        const file = findFile(codebaseData.file_structure, filePath);
        if (file && file.file_type !== 'Directory') {
          loadFileContent(filePath);
        }
      }
    }
  };

  const getComplexityColor = (complexity: number) => {
    if (complexity <= 5) return '#52c41a';
    if (complexity <= 10) return '#faad14';
    return '#ff4d4f';
  };

  const renderOverviewTab = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Row gutter={16}>
          <Col span={6}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <FileOutlined style={{ fontSize: 24, color: '#1890ff' }} />
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 24, fontWeight: 'bold' }}>{codebaseData?.total_files || 0}</div>
                  <div style={{ color: '#666' }}>Total Files</div>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <FunctionOutlined style={{ fontSize: 24, color: '#52c41a' }} />
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 24, fontWeight: 'bold' }}>{codebaseData?.total_functions || 0}</div>
                  <div style={{ color: '#666' }}>Functions</div>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <ApiOutlined style={{ fontSize: 24, color: '#722ed1' }} />
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 24, fontWeight: 'bold' }}>{codebaseData?.total_structs || 0}</div>
                  <div style={{ color: '#666' }}>Structs</div>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <SettingOutlined style={{ fontSize: 24, color: '#fa541c' }} />
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 24, fontWeight: 'bold' }}>{codebaseData?.total_lines || 0}</div>
                  <div style={{ color: '#666' }}>Lines of Code</div>
                </div>
              </div>
            </Card>
          </Col>
        </Row>
      </Col>
      
      <Col span={12}>
        <Card title="File Structure" style={{ height: 600 }}>
          {codebaseData?.file_structure ? (
            <Tree
              treeData={convertToTreeData(codebaseData.file_structure)}
              onSelect={handleFileSelect}
              expandedKeys={expandedKeys}
              onExpand={setExpandedKeys}
              showIcon
              style={{ height: 520, overflow: 'auto' }}
            />
          ) : (
            <div style={{ textAlign: 'center', padding: 40 }}>
              <Spin size="large" />
              <div style={{ marginTop: 16 }}>Loading file structure...</div>
            </div>
          )}
        </Card>
      </Col>
      
      <Col span={12}>
        <Card title="Complexity Overview" style={{ height: 600 }}>
          {codebaseData?.complexity_metrics ? (
            <div style={{ height: 520, overflow: 'auto' }}>
              <Title level={5}>Function Complexity</Title>
              {Object.entries(codebaseData.complexity_metrics.cyclomatic_complexity)
                .slice(0, 20)
                .map(([funcName, complexity]) => (
                  <div key={funcName} style={{ marginBottom: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text ellipsis style={{ maxWidth: 200 }}>{funcName}</Text>
                      <Badge 
                        count={complexity} 
                        style={{ backgroundColor: getComplexityColor(complexity) }}
                      />
                    </div>
                    <Progress 
                      percent={(complexity / 20) * 100} 
                      strokeColor={getComplexityColor(complexity)}
                      size="small"
                      showInfo={false}
                    />
                  </div>
                ))}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 40 }}>
              <Spin size="large" />
              <div style={{ marginTop: 16 }}>Analyzing complexity...</div>
            </div>
          )}
        </Card>
      </Col>
    </Row>
  );

  const renderCodeViewerTab = () => (
    <Row gutter={[16, 16]} style={{ height: 'calc(100vh - 200px)' }}>
      <Col span={8}>
        <Card title="File Explorer" style={{ height: '100%' }}>
          {codebaseData?.file_structure ? (
            <Tree
              treeData={convertToTreeData(codebaseData.file_structure)}
              onSelect={handleFileSelect}
              expandedKeys={expandedKeys}
              onExpand={setExpandedKeys}
              showIcon
              style={{ height: 'calc(100% - 60px)', overflow: 'auto' }}
            />
          ) : (
            <Spin size="large" />
          )}
        </Card>
      </Col>
      <Col span={16}>
        <Card 
          title={selectedFile ? `Code: ${selectedFile.split('/').pop() || selectedFile.split('\\').pop()}` : 'Select a file'} 
          style={{ height: '100%' }}
          extra={
            selectedFile && (
              <Button icon={<EyeOutlined />} size="small">
                View Details
              </Button>
            )
          }
        >
          {selectedFile ? (
            loadingFile ? (
              <div style={{ textAlign: 'center', padding: 40 }}>
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>Loading file content...</div>
              </div>
            ) : (
              <CodeViewer 
                code={fileContent} 
                language="rust" 
                selectedFile={selectedFile}
                functionMap={codebaseData?.function_map}
              />
            )
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: '#666' }}>
              <FileOutlined style={{ fontSize: 48, marginBottom: 16 }} />
              <div>Select a file from the explorer to view its content</div>
            </div>
          )}
        </Card>
      </Col>
    </Row>
  );

  const renderDependencyTab = () => (
    <Card title="Dependency Analysis" style={{ height: 'calc(100vh - 200px)' }}>
      {codebaseData?.dependency_graph ? (
        <DependencyGraph 
          dependencyGraph={codebaseData.dependency_graph}
          onNodeSelect={(nodeId: string) => {
            console.log('Selected node:', nodeId);
          }}
        />
      ) : (
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>Analyzing dependencies...</div>
        </div>
      )}
    </Card>
  );

  const renderFunctionTab = () => (
    <Card title="Function Analysis" style={{ height: 'calc(100vh - 200px)' }}>
      {codebaseData?.function_map ? (
        <FunctionGraph 
          functionMap={codebaseData.function_map}
          onFunctionSelect={(functionName: string) => {
            setSelectedFunction(functionName);
          }}
          selectedFunction={selectedFunction}
        />
      ) : (
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>Analyzing functions...</div>
        </div>
      )}
    </Card>
  );

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 50 }}>
        <Spin size="large" />
        <div style={{ marginTop: 20 }}>Loading codebase analysis...</div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message="Error Loading Codebase Data"
        description={error.message}
        type="error"
        showIcon
      />
    );
  }

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <BugOutlined style={{ marginRight: 8 }} />
          Codebase Explorer
        </Title>
        <Text type="secondary">
          Interactive exploration and analysis of the LLMKG codebase
        </Text>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab} type="card">
        <TabPane 
          tab={
            <span>
              <EyeOutlined />
              Overview
            </span>
          } 
          key="overview"
        >
          {renderOverviewTab()}
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <FileOutlined />
              Code Viewer
            </span>
          } 
          key="codeviewer"
        >
          {renderCodeViewerTab()}
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <ApiOutlined />
              Dependencies
            </span>
          } 
          key="dependencies"
        >
          {renderDependencyTab()}
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <FunctionOutlined />
              Functions
            </span>
          } 
          key="functions"
        >
          {renderFunctionTab()}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default CodebaseExplorerPage;