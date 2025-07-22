# Phase 3: MCP Tool Catalog & Testing

## Overview

Phase 3 implements a comprehensive MCP Tool Catalog and Testing system for LLMKG. This system provides an intuitive interface for discovering, exploring, testing, and monitoring all MCP tools available in the LLMKG ecosystem. The catalog features live status monitoring, dynamic UI generation based on tool schemas, and comprehensive visualization of tool interactions.

## Architecture

### Core Components

```typescript
// Tool Catalog System Architecture
interface MCPToolCatalog {
  discovery: ToolDiscoveryService;
  registry: ToolRegistry;
  monitor: ToolStatusMonitor;
  tester: ToolTester;
  visualizer: ToolVisualizer;
  analytics: ToolAnalytics;
  documentation: ToolDocumentationGenerator;
}

// Tool Model
interface MCPTool {
  id: string;
  name: string;
  version: string;
  description: string;
  category: ToolCategory;
  schema: ToolSchema;
  status: ToolStatus;
  metrics: ToolMetrics;
  documentation: ToolDocumentation;
}

// Tool Schema for Dynamic UI Generation
interface ToolSchema {
  inputSchema: JSONSchema;
  outputSchema: JSONSchema;
  errorSchema?: JSONSchema;
  examples?: ToolExample[];
}

// Tool Status
interface ToolStatus {
  available: boolean;
  health: 'healthy' | 'degraded' | 'unavailable';
  lastChecked: Date;
  responseTime: number;
  errorRate: number;
}
```

## 1. MCP Tool Discovery and Catalog System

### Discovery Service Implementation

```typescript
class ToolDiscoveryService {
  private discoveryInterval: number = 30000; // 30 seconds
  private toolEndpoints: string[] = [
    'http://localhost:8080/mcp/tools',
    'http://localhost:8081/mcp/tools',
    // Additional MCP endpoints
  ];

  async discoverTools(): Promise<MCPTool[]> {
    const discoveredTools: MCPTool[] = [];
    
    for (const endpoint of this.toolEndpoints) {
      try {
        const response = await fetch(endpoint);
        const tools = await response.json();
        
        for (const tool of tools) {
          const mappedTool = await this.mapToMCPTool(tool, endpoint);
          discoveredTools.push(mappedTool);
        }
      } catch (error) {
        console.error(`Failed to discover tools from ${endpoint}:`, error);
      }
    }
    
    return discoveredTools;
  }

  private async mapToMCPTool(rawTool: any, endpoint: string): Promise<MCPTool> {
    return {
      id: `${endpoint}_${rawTool.name}`,
      name: rawTool.name,
      version: rawTool.version || '1.0.0',
      description: rawTool.description,
      category: this.categorizeTools(rawTool),
      schema: await this.fetchToolSchema(rawTool, endpoint),
      status: await this.checkToolStatus(rawTool, endpoint),
      metrics: await this.fetchToolMetrics(rawTool, endpoint),
      documentation: await this.generateDocumentation(rawTool)
    };
  }

  private categorizeTools(tool: any): ToolCategory {
    // Categorize based on tool name and functionality
    if (tool.name.includes('knowledge')) return 'Knowledge Management';
    if (tool.name.includes('sdr')) return 'SDR Processing';
    if (tool.name.includes('biological')) return 'Biological Learning';
    if (tool.name.includes('cognitive')) return 'Cognitive Functions';
    return 'Utility';
  }
}
```

### Tool Registry

```typescript
class ToolRegistry {
  private tools: Map<string, MCPTool> = new Map();
  private subscribers: Set<(tools: MCPTool[]) => void> = new Set();

  registerTool(tool: MCPTool): void {
    this.tools.set(tool.id, tool);
    this.notifySubscribers();
  }

  unregisterTool(toolId: string): void {
    this.tools.delete(toolId);
    this.notifySubscribers();
  }

  getTool(toolId: string): MCPTool | undefined {
    return this.tools.get(toolId);
  }

  getAllTools(): MCPTool[] {
    return Array.from(this.tools.values());
  }

  getToolsByCategory(category: ToolCategory): MCPTool[] {
    return this.getAllTools().filter(tool => tool.category === category);
  }

  subscribe(callback: (tools: MCPTool[]) => void): () => void {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  private notifySubscribers(): void {
    const tools = this.getAllTools();
    this.subscribers.forEach(callback => callback(tools));
  }
}
```

## 2. Live Status Monitoring

### Status Monitor Implementation

```typescript
class ToolStatusMonitor {
  private statusCheckInterval: number = 10000; // 10 seconds
  private statusHistory: Map<string, ToolStatusHistory[]> = new Map();

  async monitorTool(tool: MCPTool): Promise<ToolStatus> {
    const startTime = performance.now();
    
    try {
      const response = await this.performHealthCheck(tool);
      const responseTime = performance.now() - startTime;
      
      const status: ToolStatus = {
        available: response.ok,
        health: this.determineHealth(response, responseTime),
        lastChecked: new Date(),
        responseTime,
        errorRate: this.calculateErrorRate(tool.id)
      };
      
      this.recordStatusHistory(tool.id, status);
      return status;
    } catch (error) {
      return {
        available: false,
        health: 'unavailable',
        lastChecked: new Date(),
        responseTime: performance.now() - startTime,
        errorRate: 1.0
      };
    }
  }

  private async performHealthCheck(tool: MCPTool): Promise<Response> {
    const healthEndpoint = `${tool.endpoint}/health`;
    return fetch(healthEndpoint, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });
  }

  private determineHealth(response: Response, responseTime: number): 'healthy' | 'degraded' | 'unavailable' {
    if (!response.ok) return 'unavailable';
    if (responseTime > 5000) return 'degraded';
    if (responseTime > 1000) return 'degraded';
    return 'healthy';
  }

  private calculateErrorRate(toolId: string): number {
    const history = this.statusHistory.get(toolId) || [];
    const recentHistory = history.slice(-100); // Last 100 checks
    
    if (recentHistory.length === 0) return 0;
    
    const errors = recentHistory.filter(h => !h.status.available).length;
    return errors / recentHistory.length;
  }

  private recordStatusHistory(toolId: string, status: ToolStatus): void {
    const history = this.statusHistory.get(toolId) || [];
    history.push({ timestamp: new Date(), status });
    
    // Keep only last 1000 entries
    if (history.length > 1000) {
      history.splice(0, history.length - 1000);
    }
    
    this.statusHistory.set(toolId, history);
  }
}
```

### Status Visualization Component

```tsx
const ToolStatusIndicator: React.FC<{ tool: MCPTool }> = ({ tool }) => {
  const [status, setStatus] = useState<ToolStatus>(tool.status);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    const monitor = new ToolStatusMonitor();
    
    const checkStatus = async () => {
      const newStatus = await monitor.monitorTool(tool);
      setStatus(newStatus);
    };

    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, [tool]);

  const getStatusColor = (health: string) => {
    switch (health) {
      case 'healthy': return '#00ff00';
      case 'degraded': return '#ffaa00';
      case 'unavailable': return '#ff0000';
      default: return '#888888';
    }
  };

  return (
    <div className="tool-status-indicator">
      <div className="status-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div 
          className="status-dot"
          style={{ backgroundColor: getStatusColor(status.health) }}
        />
        <span className="tool-name">{tool.name}</span>
        <span className="status-text">{status.health}</span>
        <span className="response-time">{status.responseTime.toFixed(0)}ms</span>
      </div>
      
      {isExpanded && (
        <div className="status-details">
          <div className="metric">
            <span>Last Checked:</span>
            <span>{status.lastChecked.toLocaleTimeString()}</span>
          </div>
          <div className="metric">
            <span>Error Rate:</span>
            <span>{(status.errorRate * 100).toFixed(1)}%</span>
          </div>
          <div className="metric">
            <span>Availability:</span>
            <span>{status.available ? 'Available' : 'Unavailable'}</span>
          </div>
        </div>
      )}
    </div>
  );
};
```

## 3. Interactive MCP Tool Tester

### Dynamic Form Generator

```typescript
class DynamicFormGenerator {
  generateForm(schema: JSONSchema): FormDefinition {
    const fields: FormField[] = this.parseSchema(schema);
    
    return {
      fields,
      validation: this.generateValidation(schema),
      layout: this.generateLayout(fields)
    };
  }

  private parseSchema(schema: JSONSchema, path: string = ''): FormField[] {
    const fields: FormField[] = [];

    if (schema.type === 'object' && schema.properties) {
      Object.entries(schema.properties).forEach(([key, prop]) => {
        const fieldPath = path ? `${path}.${key}` : key;
        
        if (prop.type === 'object') {
          fields.push(...this.parseSchema(prop, fieldPath));
        } else {
          fields.push(this.createField(key, prop, fieldPath, schema.required?.includes(key)));
        }
      });
    }

    return fields;
  }

  private createField(name: string, schema: any, path: string, required: boolean): FormField {
    return {
      name,
      path,
      type: this.mapSchemaTypeToFieldType(schema),
      label: schema.title || this.humanizeFieldName(name),
      description: schema.description,
      required,
      validation: this.createFieldValidation(schema),
      options: schema.enum,
      defaultValue: schema.default,
      placeholder: schema.examples?.[0]
    };
  }

  private mapSchemaTypeToFieldType(schema: any): FieldType {
    if (schema.enum) return 'select';
    if (schema.type === 'boolean') return 'checkbox';
    if (schema.type === 'integer' || schema.type === 'number') return 'number';
    if (schema.type === 'string') {
      if (schema.format === 'date-time') return 'datetime';
      if (schema.format === 'date') return 'date';
      if (schema.maxLength > 100) return 'textarea';
    }
    return 'text';
  }

  private humanizeFieldName(name: string): string {
    return name
      .replace(/([A-Z])/g, ' $1')
      .replace(/[_-]/g, ' ')
      .trim()
      .replace(/^\w/, c => c.toUpperCase());
  }
}
```

### Tool Tester UI Component

```tsx
const ToolTester: React.FC<{ tool: MCPTool }> = ({ tool }) => {
  const [formData, setFormData] = useState<any>({});
  const [response, setResponse] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [executionTime, setExecutionTime] = useState<number>(0);

  const formGenerator = new DynamicFormGenerator();
  const formDefinition = formGenerator.generateForm(tool.schema.inputSchema);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResponse(null);

    const startTime = performance.now();

    try {
      const result = await executeTool(tool, formData);
      setResponse(result);
      setExecutionTime(performance.now() - startTime);
    } catch (err) {
      setError(err.message || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFieldChange = (path: string, value: any) => {
    setFormData(prev => {
      const newData = { ...prev };
      const pathParts = path.split('.');
      let current = newData;

      for (let i = 0; i < pathParts.length - 1; i++) {
        if (!current[pathParts[i]]) {
          current[pathParts[i]] = {};
        }
        current = current[pathParts[i]];
      }

      current[pathParts[pathParts.length - 1]] = value;
      return newData;
    });
  };

  return (
    <div className="tool-tester">
      <h3>{tool.name} Tester</h3>
      <p className="tool-description">{tool.description}</p>

      <form onSubmit={handleSubmit} className="tool-form">
        {formDefinition.fields.map(field => (
          <FormField
            key={field.path}
            field={field}
            value={getValueByPath(formData, field.path)}
            onChange={(value) => handleFieldChange(field.path, value)}
          />
        ))}

        <button 
          type="submit" 
          disabled={isLoading}
          className="submit-button"
        >
          {isLoading ? 'Executing...' : 'Execute Tool'}
        </button>
      </form>

      {response && (
        <div className="response-section">
          <h4>Response ({executionTime.toFixed(2)}ms)</h4>
          <ResponseVisualizer data={response} schema={tool.schema.outputSchema} />
        </div>
      )}

      {error && (
        <div className="error-section">
          <h4>Error</h4>
          <pre className="error-message">{error}</pre>
        </div>
      )}
    </div>
  );
};
```

## 4. Request/Response Visualization

### Syntax Highlighting and Visualization

```typescript
class PayloadVisualizer {
  private syntaxHighlighter: SyntaxHighlighter;
  private schemaValidator: SchemaValidator;

  constructor() {
    this.syntaxHighlighter = new SyntaxHighlighter();
    this.schemaValidator = new SchemaValidator();
  }

  visualizeRequest(request: any, schema: JSONSchema): VisualizationResult {
    const validation = this.schemaValidator.validate(request, schema);
    const highlighted = this.syntaxHighlighter.highlight(request, 'json');
    const structure = this.analyzeStructure(request);

    return {
      raw: JSON.stringify(request, null, 2),
      highlighted,
      validation,
      structure,
      summary: this.generateSummary(request, structure)
    };
  }

  visualizeResponse(response: any, schema: JSONSchema): VisualizationResult {
    const validation = this.schemaValidator.validate(response, schema);
    const highlighted = this.syntaxHighlighter.highlight(response, 'json');
    const structure = this.analyzeStructure(response);

    return {
      raw: JSON.stringify(response, null, 2),
      highlighted,
      validation,
      structure,
      summary: this.generateSummary(response, structure),
      visualization: this.generateVisualization(response, schema)
    };
  }

  private analyzeStructure(data: any): DataStructure {
    const structure: DataStructure = {
      type: Array.isArray(data) ? 'array' : typeof data,
      depth: this.calculateDepth(data),
      size: this.calculateSize(data),
      keys: typeof data === 'object' ? Object.keys(data) : []
    };

    if (Array.isArray(data)) {
      structure.length = data.length;
    }

    return structure;
  }

  private calculateDepth(obj: any, currentDepth = 0): number {
    if (typeof obj !== 'object' || obj === null) return currentDepth;

    let maxDepth = currentDepth;
    for (const value of Object.values(obj)) {
      const depth = this.calculateDepth(value, currentDepth + 1);
      maxDepth = Math.max(maxDepth, depth);
    }

    return maxDepth;
  }

  private calculateSize(obj: any): number {
    return JSON.stringify(obj).length;
  }

  private generateSummary(data: any, structure: DataStructure): string {
    const summaryParts = [
      `Type: ${structure.type}`,
      `Depth: ${structure.depth}`,
      `Size: ${this.formatBytes(structure.size)}`
    ];

    if (structure.length !== undefined) {
      summaryParts.push(`Length: ${structure.length}`);
    }

    if (structure.keys.length > 0) {
      summaryParts.push(`Keys: ${structure.keys.slice(0, 5).join(', ')}${structure.keys.length > 5 ? '...' : ''}`);
    }

    return summaryParts.join(' | ');
  }

  private formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  private generateVisualization(data: any, schema: JSONSchema): any {
    // Generate appropriate visualization based on data type
    if (this.isGraphData(data)) {
      return this.generateGraphVisualization(data);
    }
    if (this.isTimeSeriesData(data)) {
      return this.generateTimeSeriesVisualization(data);
    }
    if (this.isTableData(data)) {
      return this.generateTableVisualization(data);
    }
    return null;
  }

  private isGraphData(data: any): boolean {
    return data.nodes && data.edges;
  }

  private isTimeSeriesData(data: any): boolean {
    return Array.isArray(data) && data.length > 0 && 
           data[0].timestamp !== undefined;
  }

  private isTableData(data: any): boolean {
    return Array.isArray(data) && data.length > 0 && 
           typeof data[0] === 'object';
  }
}
```

### Response Visualizer Component

```tsx
const ResponseVisualizer: React.FC<{ data: any; schema: JSONSchema }> = ({ data, schema }) => {
  const [viewMode, setViewMode] = useState<'raw' | 'formatted' | 'visual'>('formatted');
  const visualizer = new PayloadVisualizer();
  const visualization = visualizer.visualizeResponse(data, schema);

  return (
    <div className="response-visualizer">
      <div className="view-mode-selector">
        <button 
          className={viewMode === 'raw' ? 'active' : ''}
          onClick={() => setViewMode('raw')}
        >
          Raw
        </button>
        <button 
          className={viewMode === 'formatted' ? 'active' : ''}
          onClick={() => setViewMode('formatted')}
        >
          Formatted
        </button>
        {visualization.visualization && (
          <button 
            className={viewMode === 'visual' ? 'active' : ''}
            onClick={() => setViewMode('visual')}
          >
            Visual
          </button>
        )}
      </div>

      <div className="visualization-content">
        {viewMode === 'raw' && (
          <pre className="raw-view">{visualization.raw}</pre>
        )}

        {viewMode === 'formatted' && (
          <div className="formatted-view">
            <div className="summary">{visualization.summary}</div>
            <div 
              className="highlighted-code"
              dangerouslySetInnerHTML={{ __html: visualization.highlighted }}
            />
            {!visualization.validation.valid && (
              <div className="validation-errors">
                <h5>Validation Errors:</h5>
                <ul>
                  {visualization.validation.errors.map((error, idx) => (
                    <li key={idx}>{error}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {viewMode === 'visual' && visualization.visualization && (
          <div className="visual-view">
            <DataVisualization 
              data={visualization.visualization} 
              type={visualization.visualization.type}
            />
          </div>
        )}
      </div>
    </div>
  );
};
```

## 5. Tool Documentation Viewer

### Documentation Generator

```typescript
class ToolDocumentationGenerator {
  generateDocumentation(tool: MCPTool): ToolDocumentation {
    return {
      overview: this.generateOverview(tool),
      apiReference: this.generateAPIReference(tool),
      examples: this.generateExamples(tool),
      changelog: this.generateChangelog(tool),
      troubleshooting: this.generateTroubleshooting(tool)
    };
  }

  private generateOverview(tool: MCPTool): string {
    return `
# ${tool.name}

${tool.description}

## Version
${tool.version}

## Category
${tool.category}

## Status
- Health: ${tool.status.health}
- Response Time: ${tool.status.responseTime}ms
- Error Rate: ${(tool.status.errorRate * 100).toFixed(1)}%
    `;
  }

  private generateAPIReference(tool: MCPTool): APIReference {
    return {
      endpoint: tool.endpoint,
      method: tool.method || 'POST',
      headers: tool.headers || { 'Content-Type': 'application/json' },
      inputSchema: this.documentSchema(tool.schema.inputSchema),
      outputSchema: this.documentSchema(tool.schema.outputSchema),
      errorSchema: tool.schema.errorSchema ? this.documentSchema(tool.schema.errorSchema) : null
    };
  }

  private documentSchema(schema: JSONSchema): SchemaDocumentation {
    const doc: SchemaDocumentation = {
      description: schema.description || '',
      properties: {},
      required: schema.required || []
    };

    if (schema.properties) {
      Object.entries(schema.properties).forEach(([key, prop]) => {
        doc.properties[key] = {
          type: prop.type,
          description: prop.description || '',
          format: prop.format,
          examples: prop.examples,
          constraints: this.extractConstraints(prop)
        };
      });
    }

    return doc;
  }

  private extractConstraints(prop: any): any {
    const constraints: any = {};
    
    if (prop.minimum !== undefined) constraints.minimum = prop.minimum;
    if (prop.maximum !== undefined) constraints.maximum = prop.maximum;
    if (prop.minLength !== undefined) constraints.minLength = prop.minLength;
    if (prop.maxLength !== undefined) constraints.maxLength = prop.maxLength;
    if (prop.pattern !== undefined) constraints.pattern = prop.pattern;
    if (prop.enum !== undefined) constraints.enum = prop.enum;
    
    return constraints;
  }

  private generateExamples(tool: MCPTool): Example[] {
    const examples: Example[] = [];

    // Generate examples from schema
    if (tool.schema.examples) {
      examples.push(...tool.schema.examples.map(ex => ({
        title: ex.title || 'Example',
        description: ex.description || '',
        request: ex.input,
        response: ex.output,
        code: this.generateCodeExample(tool, ex)
      })));
    }

    // Generate default example
    if (examples.length === 0) {
      examples.push(this.generateDefaultExample(tool));
    }

    return examples;
  }

  private generateCodeExample(tool: MCPTool, example: any): CodeExample[] {
    return [
      {
        language: 'javascript',
        code: this.generateJavaScriptExample(tool, example)
      },
      {
        language: 'python',
        code: this.generatePythonExample(tool, example)
      },
      {
        language: 'curl',
        code: this.generateCurlExample(tool, example)
      }
    ];
  }

  private generateJavaScriptExample(tool: MCPTool, example: any): string {
    return `
// ${tool.name} - ${example.title || 'Example'}
const response = await fetch('${tool.endpoint}', {
  method: '${tool.method || 'POST'}',
  headers: ${JSON.stringify(tool.headers || { 'Content-Type': 'application/json' }, null, 2)},
  body: JSON.stringify(${JSON.stringify(example.input, null, 2)})
});

const result = await response.json();
console.log(result);
    `.trim();
  }

  private generatePythonExample(tool: MCPTool, example: any): string {
    return `
# ${tool.name} - ${example.title || 'Example'}
import requests

response = requests.post(
    '${tool.endpoint}',
    headers=${JSON.stringify(tool.headers || { 'Content-Type': 'application/json' })},
    json=${JSON.stringify(example.input, null, 4).replace(/"/g, "'")}
)

result = response.json()
print(result)
    `.trim();
  }

  private generateCurlExample(tool: MCPTool, example: any): string {
    return `
# ${tool.name} - ${example.title || 'Example'}
curl -X ${tool.method || 'POST'} '${tool.endpoint}' \\
  ${Object.entries(tool.headers || {}).map(([k, v]) => `-H '${k}: ${v}'`).join(' \\\n  ')} \\
  -d '${JSON.stringify(example.input)}'
    `.trim();
  }
}
```

### Documentation Viewer Component

```tsx
const ToolDocumentationViewer: React.FC<{ tool: MCPTool }> = ({ tool }) => {
  const [activeSection, setActiveSection] = useState<'overview' | 'api' | 'examples' | 'changelog' | 'troubleshooting'>('overview');
  const documentation = tool.documentation;

  return (
    <div className="documentation-viewer">
      <nav className="doc-nav">
        <button 
          className={activeSection === 'overview' ? 'active' : ''}
          onClick={() => setActiveSection('overview')}
        >
          Overview
        </button>
        <button 
          className={activeSection === 'api' ? 'active' : ''}
          onClick={() => setActiveSection('api')}
        >
          API Reference
        </button>
        <button 
          className={activeSection === 'examples' ? 'active' : ''}
          onClick={() => setActiveSection('examples')}
        >
          Examples
        </button>
        <button 
          className={activeSection === 'changelog' ? 'active' : ''}
          onClick={() => setActiveSection('changelog')}
        >
          Changelog
        </button>
        <button 
          className={activeSection === 'troubleshooting' ? 'active' : ''}
          onClick={() => setActiveSection('troubleshooting')}
        >
          Troubleshooting
        </button>
      </nav>

      <div className="doc-content">
        {activeSection === 'overview' && (
          <div className="overview-section">
            <ReactMarkdown>{documentation.overview}</ReactMarkdown>
          </div>
        )}

        {activeSection === 'api' && (
          <APIReferenceView reference={documentation.apiReference} />
        )}

        {activeSection === 'examples' && (
          <ExamplesView examples={documentation.examples} />
        )}

        {activeSection === 'changelog' && (
          <ChangelogView changelog={documentation.changelog} />
        )}

        {activeSection === 'troubleshooting' && (
          <TroubleshootingView items={documentation.troubleshooting} />
        )}
      </div>
    </div>
  );
};
```

## 6. Security Information Display

### Security Status Components

```typescript
interface SecurityInformation {
  authentication: AuthenticationStatus;
  authorization: AuthorizationDetails;
  encryption: EncryptionInfo;
  configuration: SecurityConfiguration;
  tokens: TokenManagement;
}

interface AuthenticationStatus {
  enabled: boolean;
  type: 'apiKey' | 'oauth2' | 'jwt' | 'basic' | 'custom';
  status: 'authenticated' | 'unauthenticated' | 'expired' | 'invalid';
  expiration?: Date;
  lastAuthenticated?: Date;
  provider?: string;
}

interface AuthorizationDetails {
  roles: string[];
  permissions: Permission[];
  restrictions: AccessRestriction[];
  scope?: string[];
  quotas?: QuotaInfo[];
}

interface Permission {
  resource: string;
  actions: string[];
  conditions?: string[];
}

interface AccessRestriction {
  type: 'ip' | 'time' | 'rate' | 'geographic';
  description: string;
  status: 'active' | 'inactive';
  details: any;
}

interface EncryptionInfo {
  inTransit: EncryptionDetails;
  atRest: EncryptionDetails;
  keyManagement: KeyManagementInfo;
}

interface EncryptionDetails {
  enabled: boolean;
  algorithm: string;
  strength: '128-bit' | '256-bit' | '512-bit';
  protocol?: string;
  certificate?: CertificateInfo;
}

interface SecurityConfiguration {
  tls: TLSConfiguration;
  cors: CORSConfiguration;
  rateLimit: RateLimitConfiguration;
  audit: AuditConfiguration;
}
```

### Security Monitor Implementation

```typescript
class SecurityMonitor {
  private securityStore: SecurityStore;
  private validator: SecurityValidator;
  private auditor: SecurityAuditor;

  async getSecurityStatus(tool: MCPTool): Promise<SecurityInformation> {
    const auth = await this.checkAuthentication(tool);
    const authz = await this.getAuthorizationDetails(tool);
    const encryption = await this.getEncryptionInfo(tool);
    const config = await this.getSecurityConfiguration(tool);
    const tokens = await this.getTokenManagement(tool);

    return {
      authentication: auth,
      authorization: authz,
      encryption,
      configuration: config,
      tokens
    };
  }

  private async checkAuthentication(tool: MCPTool): Promise<AuthenticationStatus> {
    try {
      const authConfig = await this.fetchAuthConfig(tool);
      const currentAuth = await this.validateCurrentAuth(tool);
      
      return {
        enabled: authConfig.enabled,
        type: authConfig.type,
        status: currentAuth.status,
        expiration: currentAuth.expiration,
        lastAuthenticated: currentAuth.lastAuthenticated,
        provider: authConfig.provider
      };
    } catch (error) {
      return {
        enabled: false,
        type: 'custom',
        status: 'unauthenticated'
      };
    }
  }

  private async getAuthorizationDetails(tool: MCPTool): Promise<AuthorizationDetails> {
    const authzInfo = await this.fetchAuthorizationInfo(tool);
    
    return {
      roles: authzInfo.roles || [],
      permissions: this.parsePermissions(authzInfo.permissions),
      restrictions: this.parseRestrictions(authzInfo.restrictions),
      scope: authzInfo.scope,
      quotas: this.parseQuotas(authzInfo.quotas)
    };
  }

  private parsePermissions(rawPermissions: any[]): Permission[] {
    return rawPermissions.map(perm => ({
      resource: perm.resource,
      actions: Array.isArray(perm.actions) ? perm.actions : [perm.actions],
      conditions: perm.conditions
    }));
  }

  private parseRestrictions(rawRestrictions: any[]): AccessRestriction[] {
    return rawRestrictions.map(restriction => ({
      type: restriction.type,
      description: restriction.description,
      status: this.isRestrictionActive(restriction) ? 'active' : 'inactive',
      details: restriction.details
    }));
  }

  private async getEncryptionInfo(tool: MCPTool): Promise<EncryptionInfo> {
    const tlsInfo = await this.checkTLSStatus(tool);
    const storageEncryption = await this.checkStorageEncryption(tool);
    const keyInfo = await this.getKeyManagementInfo(tool);

    return {
      inTransit: {
        enabled: tlsInfo.enabled,
        algorithm: tlsInfo.cipher,
        strength: this.getCipherStrength(tlsInfo.cipher),
        protocol: tlsInfo.protocol,
        certificate: tlsInfo.certificate
      },
      atRest: {
        enabled: storageEncryption.enabled,
        algorithm: storageEncryption.algorithm,
        strength: storageEncryption.keySize
      },
      keyManagement: keyInfo
    };
  }

  private getCipherStrength(cipher: string): '128-bit' | '256-bit' | '512-bit' {
    if (cipher.includes('256')) return '256-bit';
    if (cipher.includes('512')) return '512-bit';
    return '128-bit';
  }

  async validateSecurity(tool: MCPTool): Promise<SecurityValidation> {
    const status = await this.getSecurityStatus(tool);
    const issues: SecurityIssue[] = [];

    // Check authentication
    if (!status.authentication.enabled) {
      issues.push({
        severity: 'high',
        category: 'authentication',
        message: 'Authentication is disabled',
        recommendation: 'Enable authentication to secure API access'
      });
    }

    // Check encryption
    if (!status.encryption.inTransit.enabled) {
      issues.push({
        severity: 'critical',
        category: 'encryption',
        message: 'TLS/SSL is not enabled',
        recommendation: 'Enable HTTPS to encrypt data in transit'
      });
    }

    // Check authorization
    if (status.authorization.permissions.length === 0) {
      issues.push({
        severity: 'medium',
        category: 'authorization',
        message: 'No permissions configured',
        recommendation: 'Define granular permissions for better access control'
      });
    }

    return {
      score: this.calculateSecurityScore(status, issues),
      issues,
      recommendations: this.generateRecommendations(issues),
      lastValidated: new Date()
    };
  }

  private calculateSecurityScore(status: SecurityInformation, issues: SecurityIssue[]): number {
    let score = 100;
    
    issues.forEach(issue => {
      switch (issue.severity) {
        case 'critical': score -= 25; break;
        case 'high': score -= 15; break;
        case 'medium': score -= 10; break;
        case 'low': score -= 5; break;
      }
    });

    return Math.max(0, score);
  }
}
```

### Security Status Indicator Component

```tsx
const SecurityStatusIndicator: React.FC<{ tool: MCPTool }> = ({ tool }) => {
  const [security, setSecurity] = useState<SecurityInformation | null>(null);
  const [validation, setValidation] = useState<SecurityValidation | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    const monitor = new SecurityMonitor();
    
    const checkSecurity = async () => {
      const status = await monitor.getSecurityStatus(tool);
      const val = await monitor.validateSecurity(tool);
      setSecurity(status);
      setValidation(val);
    };

    checkSecurity();
    const interval = setInterval(checkSecurity, 60000); // Check every minute
    return () => clearInterval(interval);
  }, [tool]);

  const getScoreColor = (score: number): string => {
    if (score >= 90) return '#00ff00';
    if (score >= 70) return '#ffaa00';
    if (score >= 50) return '#ff6600';
    return '#ff0000';
  };

  const getStatusIcon = (status: string): string => {
    switch (status) {
      case 'authenticated': return '🔓';
      case 'unauthenticated': return '🔒';
      case 'expired': return '⏰';
      case 'invalid': return '❌';
      default: return '❓';
    }
  };

  if (!security || !validation) return <div>Loading security info...</div>;

  return (
    <div className="security-status-indicator">
      <div className="security-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="security-score" style={{ color: getScoreColor(validation.score) }}>
          <span className="score-value">{validation.score}</span>
          <span className="score-label">Security Score</span>
        </div>
        <div className="auth-status">
          <span className="status-icon">{getStatusIcon(security.authentication.status)}</span>
          <span className="status-text">{security.authentication.status}</span>
        </div>
        <button className="expand-button">{isExpanded ? '▼' : '▶'}</button>
      </div>

      {isExpanded && (
        <div className="security-details">
          <AuthenticationPanel authentication={security.authentication} />
          <AuthorizationPanel authorization={security.authorization} />
          <EncryptionPanel encryption={security.encryption} />
          <SecurityConfigPanel configuration={security.configuration} />
          <TokenManagementPanel tokens={security.tokens} />
          {validation.issues.length > 0 && (
            <SecurityIssuesPanel issues={validation.issues} />
          )}
        </div>
      )}
    </div>
  );
};
```

### Authentication Panel Component

```tsx
const AuthenticationPanel: React.FC<{ authentication: AuthenticationStatus }> = ({ authentication }) => {
  const getExpirationStatus = () => {
    if (!authentication.expiration) return null;
    
    const now = new Date();
    const exp = new Date(authentication.expiration);
    const hoursLeft = (exp.getTime() - now.getTime()) / (1000 * 60 * 60);
    
    if (hoursLeft < 0) return { text: 'Expired', color: '#ff0000' };
    if (hoursLeft < 24) return { text: `Expires in ${Math.floor(hoursLeft)}h`, color: '#ffaa00' };
    return { text: `Expires ${exp.toLocaleDateString()}`, color: '#00ff00' };
  };

  const expirationStatus = getExpirationStatus();

  return (
    <div className="security-panel authentication-panel">
      <h4>Authentication</h4>
      <div className="panel-content">
        <div className="info-row">
          <span className="label">Status:</span>
          <span className={`value status-${authentication.status}`}>
            {authentication.status}
          </span>
        </div>
        <div className="info-row">
          <span className="label">Type:</span>
          <span className="value">{authentication.type.toUpperCase()}</span>
        </div>
        {authentication.provider && (
          <div className="info-row">
            <span className="label">Provider:</span>
            <span className="value">{authentication.provider}</span>
          </div>
        )}
        {expirationStatus && (
          <div className="info-row">
            <span className="label">Expiration:</span>
            <span className="value" style={{ color: expirationStatus.color }}>
              {expirationStatus.text}
            </span>
          </div>
        )}
        {authentication.lastAuthenticated && (
          <div className="info-row">
            <span className="label">Last Auth:</span>
            <span className="value">
              {new Date(authentication.lastAuthenticated).toLocaleString()}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
```

### Authorization Panel Component

```tsx
const AuthorizationPanel: React.FC<{ authorization: AuthorizationDetails }> = ({ authorization }) => {
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  return (
    <div className="security-panel authorization-panel">
      <h4>Authorization</h4>
      <div className="panel-content">
        {authorization.roles.length > 0 && (
          <div className="auth-section">
            <h5 
              className="section-header"
              onClick={() => setExpandedSection(expandedSection === 'roles' ? null : 'roles')}
            >
              Roles ({authorization.roles.length})
            </h5>
            {expandedSection === 'roles' && (
              <ul className="role-list">
                {authorization.roles.map((role, idx) => (
                  <li key={idx} className="role-item">{role}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {authorization.permissions.length > 0 && (
          <div className="auth-section">
            <h5 
              className="section-header"
              onClick={() => setExpandedSection(expandedSection === 'permissions' ? null : 'permissions')}
            >
              Permissions ({authorization.permissions.length})
            </h5>
            {expandedSection === 'permissions' && (
              <div className="permissions-list">
                {authorization.permissions.map((perm, idx) => (
                  <div key={idx} className="permission-item">
                    <span className="resource">{perm.resource}</span>
                    <span className="actions">{perm.actions.join(', ')}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {authorization.restrictions.length > 0 && (
          <div className="auth-section">
            <h5 
              className="section-header"
              onClick={() => setExpandedSection(expandedSection === 'restrictions' ? null : 'restrictions')}
            >
              Restrictions ({authorization.restrictions.filter(r => r.status === 'active').length} active)
            </h5>
            {expandedSection === 'restrictions' && (
              <div className="restrictions-list">
                {authorization.restrictions.map((restriction, idx) => (
                  <div key={idx} className={`restriction-item ${restriction.status}`}>
                    <span className="type">{restriction.type}</span>
                    <span className="description">{restriction.description}</span>
                    <span className={`status status-${restriction.status}`}>
                      {restriction.status}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
```

### Encryption Panel Component

```tsx
const EncryptionPanel: React.FC<{ encryption: EncryptionInfo }> = ({ encryption }) => {
  const getCertificateStatus = (cert?: CertificateInfo) => {
    if (!cert) return null;
    
    const now = new Date();
    const exp = new Date(cert.expiration);
    const daysLeft = (exp.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);
    
    if (daysLeft < 0) return { text: 'Expired', color: '#ff0000', icon: '❌' };
    if (daysLeft < 30) return { text: `Expires in ${Math.floor(daysLeft)} days`, color: '#ffaa00', icon: '⚠️' };
    return { text: 'Valid', color: '#00ff00', icon: '✅' };
  };

  return (
    <div className="security-panel encryption-panel">
      <h4>Encryption</h4>
      <div className="panel-content">
        <div className="encryption-section">
          <h5>In Transit</h5>
          <div className="encryption-details">
            <div className="info-row">
              <span className="label">Status:</span>
              <span className={`value ${encryption.inTransit.enabled ? 'enabled' : 'disabled'}`}>
                {encryption.inTransit.enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            {encryption.inTransit.enabled && (
              <>
                <div className="info-row">
                  <span className="label">Protocol:</span>
                  <span className="value">{encryption.inTransit.protocol || 'TLS'}</span>
                </div>
                <div className="info-row">
                  <span className="label">Algorithm:</span>
                  <span className="value">{encryption.inTransit.algorithm}</span>
                </div>
                <div className="info-row">
                  <span className="label">Strength:</span>
                  <span className="value">{encryption.inTransit.strength}</span>
                </div>
                {encryption.inTransit.certificate && (
                  <div className="certificate-info">
                    <span className="cert-status">
                      {getCertificateStatus(encryption.inTransit.certificate)?.icon}
                    </span>
                    <span className="cert-text">
                      Certificate: {getCertificateStatus(encryption.inTransit.certificate)?.text}
                    </span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <div className="encryption-section">
          <h5>At Rest</h5>
          <div className="encryption-details">
            <div className="info-row">
              <span className="label">Status:</span>
              <span className={`value ${encryption.atRest.enabled ? 'enabled' : 'disabled'}`}>
                {encryption.atRest.enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            {encryption.atRest.enabled && (
              <>
                <div className="info-row">
                  <span className="label">Algorithm:</span>
                  <span className="value">{encryption.atRest.algorithm}</span>
                </div>
                <div className="info-row">
                  <span className="label">Strength:</span>
                  <span className="value">{encryption.atRest.strength}</span>
                </div>
              </>
            )}
          </div>
        </div>

        {encryption.keyManagement && (
          <div className="encryption-section">
            <h5>Key Management</h5>
            <div className="key-management-info">
              <div className="info-row">
                <span className="label">Provider:</span>
                <span className="value">{encryption.keyManagement.provider}</span>
              </div>
              <div className="info-row">
                <span className="label">Rotation:</span>
                <span className="value">
                  {encryption.keyManagement.rotationEnabled ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
```

### Security Configuration Viewer Component

```tsx
const SecurityConfigPanel: React.FC<{ configuration: SecurityConfiguration }> = ({ configuration }) => {
  return (
    <div className="security-panel config-panel">
      <h4>Security Configuration</h4>
      <div className="panel-content">
        <div className="config-section">
          <h5>TLS Configuration</h5>
          <div className="config-details">
            <div className="info-row">
              <span className="label">Min Version:</span>
              <span className="value">{configuration.tls.minVersion}</span>
            </div>
            <div className="info-row">
              <span className="label">Ciphers:</span>
              <span className="value cipher-list">
                {configuration.tls.ciphers.slice(0, 3).join(', ')}
                {configuration.tls.ciphers.length > 3 && ` +${configuration.tls.ciphers.length - 3} more`}
              </span>
            </div>
          </div>
        </div>

        <div className="config-section">
          <h5>CORS Configuration</h5>
          <div className="config-details">
            <div className="info-row">
              <span className="label">Enabled:</span>
              <span className={`value ${configuration.cors.enabled ? 'enabled' : 'disabled'}`}>
                {configuration.cors.enabled ? 'Yes' : 'No'}
              </span>
            </div>
            {configuration.cors.enabled && (
              <div className="info-row">
                <span className="label">Origins:</span>
                <span className="value">
                  {configuration.cors.allowedOrigins.join(', ') || '*'}
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="config-section">
          <h5>Rate Limiting</h5>
          <div className="config-details">
            <div className="info-row">
              <span className="label">Enabled:</span>
              <span className={`value ${configuration.rateLimit.enabled ? 'enabled' : 'disabled'}`}>
                {configuration.rateLimit.enabled ? 'Yes' : 'No'}
              </span>
            </div>
            {configuration.rateLimit.enabled && (
              <>
                <div className="info-row">
                  <span className="label">Limit:</span>
                  <span className="value">
                    {configuration.rateLimit.requestsPerMinute} req/min
                  </span>
                </div>
                <div className="info-row">
                  <span className="label">Burst:</span>
                  <span className="value">{configuration.rateLimit.burst}</span>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="config-section">
          <h5>Audit Logging</h5>
          <div className="config-details">
            <div className="info-row">
              <span className="label">Enabled:</span>
              <span className={`value ${configuration.audit.enabled ? 'enabled' : 'disabled'}`}>
                {configuration.audit.enabled ? 'Yes' : 'No'}
              </span>
            </div>
            {configuration.audit.enabled && (
              <div className="info-row">
                <span className="label">Level:</span>
                <span className="value">{configuration.audit.level}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
```

### Token Management Interface Component

```tsx
const TokenManagementPanel: React.FC<{ tokens: TokenManagement }> = ({ tokens }) => {
  const [showTokenModal, setShowTokenModal] = useState(false);
  const [selectedToken, setSelectedToken] = useState<Token | null>(null);

  const handleRevokeToken = async (tokenId: string) => {
    if (confirm('Are you sure you want to revoke this token?')) {
      await tokens.revokeToken(tokenId);
    }
  };

  const handleRefreshToken = async (tokenId: string) => {
    await tokens.refreshToken(tokenId);
  };

  const getTokenStatus = (token: Token) => {
    const now = new Date();
    const exp = new Date(token.expiration);
    
    if (token.revoked) return { text: 'Revoked', color: '#ff0000' };
    if (exp < now) return { text: 'Expired', color: '#ff6600' };
    if (exp.getTime() - now.getTime() < 24 * 60 * 60 * 1000) {
      return { text: 'Expiring Soon', color: '#ffaa00' };
    }
    return { text: 'Active', color: '#00ff00' };
  };

  return (
    <div className="security-panel token-panel">
      <h4>Token Management</h4>
      <div className="panel-content">
        <div className="token-stats">
          <div className="stat">
            <span className="stat-value">{tokens.activeTokens.length}</span>
            <span className="stat-label">Active Tokens</span>
          </div>
          <div className="stat">
            <span className="stat-value">{tokens.expiringTokens.length}</span>
            <span className="stat-label">Expiring Soon</span>
          </div>
          <div className="stat">
            <span className="stat-value">{tokens.revokedTokens.length}</span>
            <span className="stat-label">Revoked</span>
          </div>
        </div>

        <div className="token-list">
          {tokens.activeTokens.map(token => {
            const status = getTokenStatus(token);
            return (
              <div key={token.id} className="token-item">
                <div className="token-info">
                  <span className="token-name">{token.name}</span>
                  <span className="token-type">{token.type}</span>
                  <span 
                    className="token-status" 
                    style={{ color: status.color }}
                  >
                    {status.text}
                  </span>
                </div>
                <div className="token-actions">
                  <button 
                    onClick={() => {
                      setSelectedToken(token);
                      setShowTokenModal(true);
                    }}
                    className="action-button view"
                  >
                    View
                  </button>
                  {token.refreshable && (
                    <button 
                      onClick={() => handleRefreshToken(token.id)}
                      className="action-button refresh"
                    >
                      Refresh
                    </button>
                  )}
                  <button 
                    onClick={() => handleRevokeToken(token.id)}
                    className="action-button revoke"
                  >
                    Revoke
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        <button 
          className="create-token-button"
          onClick={() => setShowTokenModal(true)}
        >
          Create New Token
        </button>
      </div>

      {showTokenModal && (
        <TokenModal 
          token={selectedToken}
          onClose={() => {
            setShowTokenModal(false);
            setSelectedToken(null);
          }}
          onSave={tokens.createToken}
        />
      )}
    </div>
  );
};
```

### Security Issues Panel Component

```tsx
const SecurityIssuesPanel: React.FC<{ issues: SecurityIssue[] }> = ({ issues }) => {
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return '🔴';
      case 'high': return '🟠';
      case 'medium': return '🟡';
      case 'low': return '🟢';
      default: return '⚪';
    }
  };

  const groupedIssues = issues.reduce((acc, issue) => {
    if (!acc[issue.category]) acc[issue.category] = [];
    acc[issue.category].push(issue);
    return acc;
  }, {} as Record<string, SecurityIssue[]>);

  return (
    <div className="security-panel issues-panel">
      <h4>Security Issues ({issues.length})</h4>
      <div className="panel-content">
        {Object.entries(groupedIssues).map(([category, categoryIssues]) => (
          <div key={category} className="issue-category">
            <h5>{category.charAt(0).toUpperCase() + category.slice(1)}</h5>
            {categoryIssues.map((issue, idx) => (
              <div key={idx} className={`issue-item severity-${issue.severity}`}>
                <div className="issue-header">
                  <span className="severity-icon">{getSeverityIcon(issue.severity)}</span>
                  <span className="issue-message">{issue.message}</span>
                </div>
                {issue.recommendation && (
                  <div className="issue-recommendation">
                    <span className="recommendation-label">Recommendation:</span>
                    <span className="recommendation-text">{issue.recommendation}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};
```

## 7. Performance Metrics

### Metrics Collection

```typescript
class ToolMetricsCollector {
  private metricsStore: MetricsStore;
  private aggregator: MetricsAggregator;

  async collectMetrics(tool: MCPTool, execution: ToolExecution): Promise<void> {
    const metrics: ToolMetrics = {
      toolId: tool.id,
      timestamp: new Date(),
      executionTime: execution.duration,
      success: execution.success,
      errorType: execution.error?.type,
      requestSize: this.calculateSize(execution.request),
      responseSize: this.calculateSize(execution.response),
      memoryUsage: execution.memoryUsage,
      cpuUsage: execution.cpuUsage
    };

    await this.metricsStore.store(metrics);
    this.aggregator.update(tool.id, metrics);
  }

  getAggregatedMetrics(toolId: string, timeRange: TimeRange): AggregatedMetrics {
    const rawMetrics = this.metricsStore.query(toolId, timeRange);
    
    return {
      totalExecutions: rawMetrics.length,
      successRate: this.calculateSuccessRate(rawMetrics),
      averageExecutionTime: this.calculateAverage(rawMetrics, 'executionTime'),
      p50ExecutionTime: this.calculatePercentile(rawMetrics, 'executionTime', 50),
      p95ExecutionTime: this.calculatePercentile(rawMetrics, 'executionTime', 95),
      p99ExecutionTime: this.calculatePercentile(rawMetrics, 'executionTime', 99),
      errorDistribution: this.calculateErrorDistribution(rawMetrics),
      throughput: this.calculateThroughput(rawMetrics, timeRange),
      averageRequestSize: this.calculateAverage(rawMetrics, 'requestSize'),
      averageResponseSize: this.calculateAverage(rawMetrics, 'responseSize')
    };
  }

  private calculateSuccessRate(metrics: ToolMetrics[]): number {
    if (metrics.length === 0) return 0;
    const successes = metrics.filter(m => m.success).length;
    return (successes / metrics.length) * 100;
  }

  private calculateAverage(metrics: ToolMetrics[], field: keyof ToolMetrics): number {
    if (metrics.length === 0) return 0;
    const sum = metrics.reduce((acc, m) => acc + (m[field] as number || 0), 0);
    return sum / metrics.length;
  }

  private calculatePercentile(metrics: ToolMetrics[], field: keyof ToolMetrics, percentile: number): number {
    if (metrics.length === 0) return 0;
    
    const values = metrics
      .map(m => m[field] as number)
      .filter(v => v !== undefined)
      .sort((a, b) => a - b);
    
    const index = Math.ceil((percentile / 100) * values.length) - 1;
    return values[index] || 0;
  }

  private calculateThroughput(metrics: ToolMetrics[], timeRange: TimeRange): number {
    const durationHours = (timeRange.end.getTime() - timeRange.start.getTime()) / (1000 * 60 * 60);
    return metrics.length / durationHours;
  }
}
```

### Performance Dashboard Component

```tsx
const ToolPerformanceDashboard: React.FC<{ tool: MCPTool }> = ({ tool }) => {
  const [timeRange, setTimeRange] = useState<TimeRange>({
    start: new Date(Date.now() - 24 * 60 * 60 * 1000),
    end: new Date()
  });
  const [metrics, setMetrics] = useState<AggregatedMetrics | null>(null);

  useEffect(() => {
    const collector = new ToolMetricsCollector();
    const aggregated = collector.getAggregatedMetrics(tool.id, timeRange);
    setMetrics(aggregated);
  }, [tool.id, timeRange]);

  if (!metrics) return <div>Loading metrics...</div>;

  return (
    <div className="performance-dashboard">
      <div className="metrics-grid">
        <MetricCard
          title="Total Executions"
          value={metrics.totalExecutions}
          format="number"
        />
        <MetricCard
          title="Success Rate"
          value={metrics.successRate}
          format="percentage"
          trend={calculateTrend(metrics.successRate, 95)}
        />
        <MetricCard
          title="Avg Response Time"
          value={metrics.averageExecutionTime}
          format="duration"
          trend={calculateTrend(metrics.averageExecutionTime, 100, true)}
        />
        <MetricCard
          title="Throughput"
          value={metrics.throughput}
          format="throughput"
        />
      </div>

      <div className="charts-section">
        <ResponseTimeChart
          p50={metrics.p50ExecutionTime}
          p95={metrics.p95ExecutionTime}
          p99={metrics.p99ExecutionTime}
        />
        
        <ErrorDistributionChart
          distribution={metrics.errorDistribution}
        />
        
        <ThroughputTimeSeriesChart
          toolId={tool.id}
          timeRange={timeRange}
        />
      </div>

      <div className="detailed-metrics">
        <h4>Detailed Metrics</h4>
        <table className="metrics-table">
          <tbody>
            <tr>
              <td>P50 Response Time</td>
              <td>{metrics.p50ExecutionTime.toFixed(2)}ms</td>
            </tr>
            <tr>
              <td>P95 Response Time</td>
              <td>{metrics.p95ExecutionTime.toFixed(2)}ms</td>
            </tr>
            <tr>
              <td>P99 Response Time</td>
              <td>{metrics.p99ExecutionTime.toFixed(2)}ms</td>
            </tr>
            <tr>
              <td>Avg Request Size</td>
              <td>{formatBytes(metrics.averageRequestSize)}</td>
            </tr>
            <tr>
              <td>Avg Response Size</td>
              <td>{formatBytes(metrics.averageResponseSize)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};
```

## 7. Tool Usage History and Analytics

### Usage Analytics System

```typescript
class ToolUsageAnalytics {
  private usageStore: UsageStore;
  private analyzer: UsageAnalyzer;

  async recordUsage(tool: MCPTool, usage: ToolUsage): Promise<void> {
    const record: UsageRecord = {
      id: generateId(),
      toolId: tool.id,
      userId: usage.userId,
      timestamp: new Date(),
      action: usage.action,
      parameters: usage.parameters,
      result: usage.result,
      context: usage.context
    };

    await this.usageStore.store(record);
    await this.analyzer.analyze(record);
  }

  getUsageAnalytics(toolId: string, options: AnalyticsOptions): UsageAnalytics {
    const records = this.usageStore.query({
      toolId,
      timeRange: options.timeRange,
      limit: options.limit
    });

    return {
      totalUsage: records.length,
      uniqueUsers: this.countUniqueUsers(records),
      usageByHour: this.groupByHour(records),
      usageByDay: this.groupByDay(records),
      topUsers: this.getTopUsers(records),
      commonParameters: this.analyzeParameters(records),
      usagePatterns: this.detectPatterns(records),
      trends: this.calculateTrends(records, options.timeRange)
    };
  }

  private countUniqueUsers(records: UsageRecord[]): number {
    return new Set(records.map(r => r.userId)).size;
  }

  private groupByHour(records: UsageRecord[]): HourlyUsage[] {
    const grouped = new Map<number, number>();
    
    records.forEach(record => {
      const hour = record.timestamp.getHours();
      grouped.set(hour, (grouped.get(hour) || 0) + 1);
    });

    return Array.from(grouped.entries()).map(([hour, count]) => ({
      hour,
      count,
      percentage: (count / records.length) * 100
    }));
  }

  private analyzeParameters(records: UsageRecord[]): ParameterAnalysis[] {
    const parameterCounts = new Map<string, Map<any, number>>();

    records.forEach(record => {
      Object.entries(record.parameters).forEach(([key, value]) => {
        if (!parameterCounts.has(key)) {
          parameterCounts.set(key, new Map());
        }
        const valueCounts = parameterCounts.get(key)!;
        const valueStr = JSON.stringify(value);
        valueCounts.set(valueStr, (valueCounts.get(valueStr) || 0) + 1);
      });
    });

    return Array.from(parameterCounts.entries()).map(([parameter, values]) => ({
      parameter,
      topValues: Array.from(values.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([value, count]) => ({
          value: JSON.parse(value),
          count,
          percentage: (count / records.length) * 100
        }))
    }));
  }

  private detectPatterns(records: UsageRecord[]): UsagePattern[] {
    const patterns: UsagePattern[] = [];

    // Detect sequential usage patterns
    const sequences = this.findSequences(records);
    patterns.push(...sequences.map(seq => ({
      type: 'sequence',
      description: `Common sequence: ${seq.tools.join(' → ')}`,
      frequency: seq.count,
      confidence: seq.confidence
    })));

    // Detect temporal patterns
    const temporalPatterns = this.findTemporalPatterns(records);
    patterns.push(...temporalPatterns);

    // Detect parameter correlation patterns
    const correlations = this.findParameterCorrelations(records);
    patterns.push(...correlations);

    return patterns;
  }
}
```

### Usage History Component

```tsx
const ToolUsageHistory: React.FC<{ tool: MCPTool }> = ({ tool }) => {
  const [analytics, setAnalytics] = useState<UsageAnalytics | null>(null);
  const [timeRange, setTimeRange] = useState<TimeRange>({
    start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    end: new Date()
  });
  const [selectedView, setSelectedView] = useState<'timeline' | 'analytics' | 'patterns'>('timeline');

  useEffect(() => {
    const analyzer = new ToolUsageAnalytics();
    const data = analyzer.getUsageAnalytics(tool.id, {
      timeRange,
      limit: 1000
    });
    setAnalytics(data);
  }, [tool.id, timeRange]);

  if (!analytics) return <div>Loading analytics...</div>;

  return (
    <div className="usage-history">
      <div className="view-selector">
        <button 
          className={selectedView === 'timeline' ? 'active' : ''}
          onClick={() => setSelectedView('timeline')}
        >
          Timeline
        </button>
        <button 
          className={selectedView === 'analytics' ? 'active' : ''}
          onClick={() => setSelectedView('analytics')}
        >
          Analytics
        </button>
        <button 
          className={selectedView === 'patterns' ? 'active' : ''}
          onClick={() => setSelectedView('patterns')}
        >
          Patterns
        </button>
      </div>

      {selectedView === 'timeline' && (
        <UsageTimeline
          toolId={tool.id}
          timeRange={timeRange}
        />
      )}

      {selectedView === 'analytics' && (
        <div className="analytics-view">
          <div className="stats-row">
            <StatCard title="Total Usage" value={analytics.totalUsage} />
            <StatCard title="Unique Users" value={analytics.uniqueUsers} />
            <StatCard title="Avg Daily Usage" value={analytics.totalUsage / 7} />
          </div>

          <div className="charts-row">
            <HourlyUsageChart data={analytics.usageByHour} />
            <DailyUsageChart data={analytics.usageByDay} />
          </div>

          <div className="parameter-analysis">
            <h4>Common Parameters</h4>
            {analytics.commonParameters.map(param => (
              <ParameterAnalysisCard key={param.parameter} analysis={param} />
            ))}
          </div>
        </div>
      )}

      {selectedView === 'patterns' && (
        <div className="patterns-view">
          <h4>Detected Usage Patterns</h4>
          {analytics.usagePatterns.map((pattern, idx) => (
            <PatternCard key={idx} pattern={pattern} />
          ))}

          <h4>Usage Trends</h4>
          <TrendsChart trends={analytics.trends} />
        </div>
      )}
    </div>
  );
};
```

## 8. Testing Procedures

### Catalog System Testing

```typescript
describe('MCP Tool Catalog System', () => {
  let catalog: MCPToolCatalog;
  let mockTools: MCPTool[];

  beforeEach(() => {
    catalog = new MCPToolCatalog();
    mockTools = generateMockTools();
  });

  describe('Tool Discovery', () => {
    it('should discover tools from all configured endpoints', async () => {
      const discoveredTools = await catalog.discovery.discoverTools();
      expect(discoveredTools.length).toBeGreaterThan(0);
      expect(discoveredTools).toContainEqual(
        expect.objectContaining({
          id: expect.any(String),
          name: expect.any(String),
          schema: expect.any(Object)
        })
      );
    });

    it('should handle endpoint failures gracefully', async () => {
      // Mock one endpoint failure
      jest.spyOn(global, 'fetch').mockImplementationOnce(() => 
        Promise.reject(new Error('Network error'))
      );

      const tools = await catalog.discovery.discoverTools();
      expect(tools.length).toBeGreaterThan(0); // Should still discover from other endpoints
    });

    it('should categorize tools correctly', async () => {
      const tools = await catalog.discovery.discoverTools();
      const categories = new Set(tools.map(t => t.category));
      
      expect(categories).toContain('Knowledge Management');
      expect(categories).toContain('SDR Processing');
      expect(categories).toContain('Biological Learning');
    });
  });

  describe('Status Monitoring', () => {
    it('should monitor tool health status', async () => {
      const tool = mockTools[0];
      const status = await catalog.monitor.monitorTool(tool);
      
      expect(status).toMatchObject({
        available: expect.any(Boolean),
        health: expect.stringMatching(/healthy|degraded|unavailable/),
        lastChecked: expect.any(Date),
        responseTime: expect.any(Number),
        errorRate: expect.any(Number)
      });
    });

    it('should track status history', async () => {
      const tool = mockTools[0];
      
      // Monitor multiple times
      for (let i = 0; i < 5; i++) {
        await catalog.monitor.monitorTool(tool);
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      const history = catalog.monitor.getStatusHistory(tool.id);
      expect(history.length).toBe(5);
    });

    it('should calculate error rates correctly', async () => {
      const tool = mockTools[0];
      
      // Simulate some failures
      jest.spyOn(catalog.monitor, 'performHealthCheck')
        .mockResolvedValueOnce({ ok: false })
        .mockResolvedValueOnce({ ok: true })
        .mockResolvedValueOnce({ ok: false })
        .mockResolvedValueOnce({ ok: true });

      const statuses = [];
      for (let i = 0; i < 4; i++) {
        const status = await catalog.monitor.monitorTool(tool);
        statuses.push(status);
      }

      const lastStatus = statuses[statuses.length - 1];
      expect(lastStatus.errorRate).toBeCloseTo(0.5, 2);
    });
  });

  describe('Tool Tester', () => {
    it('should validate input against schema', async () => {
      const tool = mockTools[0];
      const invalidInput = { wrongField: 'value' };

      await expect(catalog.tester.execute(tool, invalidInput))
        .rejects.toThrow('Validation failed');
    });

    it('should execute tool with valid input', async () => {
      const tool = mockTools[0];
      const validInput = { requiredField: 'value' };

      const result = await catalog.tester.execute(tool, validInput);
      expect(result).toBeDefined();
      expect(result.success).toBe(true);
    });

    it('should measure execution time', async () => {
      const tool = mockTools[0];
      const input = { field: 'value' };

      const execution = await catalog.tester.executeWithMetrics(tool, input);
      expect(execution.duration).toBeGreaterThan(0);
      expect(execution.duration).toBeLessThan(10000); // Less than 10 seconds
    });
  });

  describe('Dynamic Form Generation', () => {
    it('should generate form from JSON schema', () => {
      const schema: JSONSchema = {
        type: 'object',
        properties: {
          name: { type: 'string', title: 'Name' },
          age: { type: 'integer', minimum: 0 },
          active: { type: 'boolean' }
        },
        required: ['name']
      };

      const form = new DynamicFormGenerator().generateForm(schema);
      
      expect(form.fields).toHaveLength(3);
      expect(form.fields[0]).toMatchObject({
        name: 'name',
        type: 'text',
        required: true
      });
      expect(form.fields[1]).toMatchObject({
        name: 'age',
        type: 'number',
        required: false
      });
      expect(form.fields[2]).toMatchObject({
        name: 'active',
        type: 'checkbox',
        required: false
      });
    });

    it('should handle nested schemas', () => {
      const schema: JSONSchema = {
        type: 'object',
        properties: {
          user: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              email: { type: 'string', format: 'email' }
            }
          }
        }
      };

      const form = new DynamicFormGenerator().generateForm(schema);
      
      expect(form.fields).toHaveLength(2);
      expect(form.fields[0].path).toBe('user.name');
      expect(form.fields[1].path).toBe('user.email');
    });
  });

  describe('Visualization', () => {
    it('should highlight JSON syntax correctly', () => {
      const data = { key: 'value', number: 42 };
      const visualizer = new PayloadVisualizer();
      const result = visualizer.visualizeRequest(data, {});

      expect(result.highlighted).toContain('class="json-key"');
      expect(result.highlighted).toContain('class="json-string"');
      expect(result.highlighted).toContain('class="json-number"');
    });

    it('should detect visualization type', () => {
      const graphData = { nodes: [], edges: [] };
      const timeSeriesData = [{ timestamp: new Date(), value: 1 }];
      const tableData = [{ col1: 'a', col2: 'b' }];

      const visualizer = new PayloadVisualizer();
      
      expect(visualizer.getVisualizationType(graphData)).toBe('graph');
      expect(visualizer.getVisualizationType(timeSeriesData)).toBe('timeseries');
      expect(visualizer.getVisualizationType(tableData)).toBe('table');
    });
  });

  describe('Performance Metrics', () => {
    it('should collect execution metrics', async () => {
      const tool = mockTools[0];
      const execution = {
        duration: 150,
        success: true,
        request: { data: 'test' },
        response: { result: 'success' }
      };

      const collector = new ToolMetricsCollector();
      await collector.collectMetrics(tool, execution);

      const metrics = collector.getAggregatedMetrics(tool.id, {
        start: new Date(Date.now() - 3600000),
        end: new Date()
      });

      expect(metrics.totalExecutions).toBe(1);
      expect(metrics.successRate).toBe(100);
      expect(metrics.averageExecutionTime).toBe(150);
    });

    it('should calculate percentiles correctly', () => {
      const metrics = Array.from({ length: 100 }, (_, i) => ({
        executionTime: i + 1
      }));

      const collector = new ToolMetricsCollector();
      const p50 = collector.calculatePercentile(metrics, 'executionTime', 50);
      const p95 = collector.calculatePercentile(metrics, 'executionTime', 95);
      const p99 = collector.calculatePercentile(metrics, 'executionTime', 99);

      expect(p50).toBe(50);
      expect(p95).toBe(95);
      expect(p99).toBe(99);
    });
  });

  describe('Usage Analytics', () => {
    it('should track tool usage', async () => {
      const tool = mockTools[0];
      const usage = {
        userId: 'user1',
        action: 'execute',
        parameters: { input: 'test' },
        result: { success: true }
      };

      const analytics = new ToolUsageAnalytics();
      await analytics.recordUsage(tool, usage);

      const analysis = analytics.getUsageAnalytics(tool.id, {
        timeRange: {
          start: new Date(Date.now() - 3600000),
          end: new Date()
        }
      });

      expect(analysis.totalUsage).toBe(1);
      expect(analysis.uniqueUsers).toBe(1);
    });

    it('should detect usage patterns', async () => {
      const analytics = new ToolUsageAnalytics();
      
      // Simulate pattern: tool1 → tool2 → tool3
      const sequence = ['tool1', 'tool2', 'tool3'];
      
      for (let i = 0; i < 10; i++) {
        for (const toolId of sequence) {
          await analytics.recordUsage(
            { id: toolId },
            { userId: 'user1', action: 'execute' }
          );
        }
      }

      const patterns = analytics.detectPatterns('user1');
      expect(patterns).toContainEqual(
        expect.objectContaining({
          type: 'sequence',
          description: expect.stringContaining('tool1 → tool2 → tool3')
        })
      );
    });
  });

  describe('Documentation Generation', () => {
    it('should generate comprehensive documentation', () => {
      const tool = mockTools[0];
      const generator = new ToolDocumentationGenerator();
      const docs = generator.generateDocumentation(tool);

      expect(docs.overview).toContain(tool.name);
      expect(docs.overview).toContain(tool.description);
      expect(docs.apiReference).toBeDefined();
      expect(docs.examples.length).toBeGreaterThan(0);
    });

    it('should generate code examples in multiple languages', () => {
      const tool = mockTools[0];
      const generator = new ToolDocumentationGenerator();
      const docs = generator.generateDocumentation(tool);

      const example = docs.examples[0];
      expect(example.code).toHaveLength(3); // JavaScript, Python, cURL
      expect(example.code[0].language).toBe('javascript');
      expect(example.code[1].language).toBe('python');
      expect(example.code[2].language).toBe('curl');
    });
  });
});
```

### Integration Testing

```typescript
describe('MCP Tool Catalog Integration', () => {
  let catalog: MCPToolCatalog;
  let testServer: TestMCPServer;

  beforeAll(async () => {
    testServer = new TestMCPServer();
    await testServer.start();
    catalog = new MCPToolCatalog();
  });

  afterAll(async () => {
    await testServer.stop();
  });

  it('should provide end-to-end tool discovery and testing', async () => {
    // Discover tools
    const tools = await catalog.discovery.discoverTools();
    expect(tools.length).toBeGreaterThan(0);

    // Select a tool
    const tool = tools[0];

    // Monitor its status
    const status = await catalog.monitor.monitorTool(tool);
    expect(status.available).toBe(true);

    // Generate form from schema
    const form = new DynamicFormGenerator().generateForm(tool.schema.inputSchema);
    expect(form.fields.length).toBeGreaterThan(0);

    // Execute the tool
    const input = generateValidInput(tool.schema.inputSchema);
    const result = await catalog.tester.execute(tool, input);
    expect(result.success).toBe(true);

    // Collect metrics
    await catalog.metrics.collectMetrics(tool, {
      duration: 100,
      success: true,
      request: input,
      response: result
    });

    // Verify analytics
    const analytics = catalog.analytics.getUsageAnalytics(tool.id, {
      timeRange: {
        start: new Date(Date.now() - 3600000),
        end: new Date()
      }
    });
    expect(analytics.totalUsage).toBeGreaterThan(0);
  });

  describe('Security Information Display', () => {
    it('should validate security status', async () => {
      const tool = mockTools[0];
      const monitor = new SecurityMonitor();
      const validation = await monitor.validateSecurity(tool);

      expect(validation).toMatchObject({
        score: expect.any(Number),
        issues: expect.any(Array),
        recommendations: expect.any(Array),
        lastValidated: expect.any(Date)
      });
    });

    it('should check authentication status', async () => {
      const tool = mockTools[0];
      const monitor = new SecurityMonitor();
      const security = await monitor.getSecurityStatus(tool);

      expect(security.authentication).toMatchObject({
        enabled: expect.any(Boolean),
        type: expect.stringMatching(/apiKey|oauth2|jwt|basic|custom/),
        status: expect.stringMatching(/authenticated|unauthenticated|expired|invalid/)
      });
    });

    it('should evaluate encryption configuration', async () => {
      const tool = mockTools[0];
      const monitor = new SecurityMonitor();
      const security = await monitor.getSecurityStatus(tool);

      expect(security.encryption).toHaveProperty('inTransit');
      expect(security.encryption).toHaveProperty('atRest');
      expect(security.encryption.inTransit).toMatchObject({
        enabled: expect.any(Boolean),
        algorithm: expect.any(String),
        strength: expect.stringMatching(/128-bit|256-bit|512-bit/)
      });
    });

    it('should calculate security score correctly', async () => {
      const tool = mockTools[0];
      const monitor = new SecurityMonitor();
      const validation = await monitor.validateSecurity(tool);

      expect(validation.score).toBeGreaterThanOrEqual(0);
      expect(validation.score).toBeLessThanOrEqual(100);
    });

    it('should identify security issues', async () => {
      const tool = { ...mockTools[0], security: { authentication: { enabled: false }, encryption: { inTransit: { enabled: false } } } };
      const monitor = new SecurityMonitor();
      const validation = await monitor.validateSecurity(tool);

      expect(validation.issues.length).toBeGreaterThan(0);
      expect(validation.issues).toContainEqual(
        expect.objectContaining({
          severity: expect.stringMatching(/critical|high|medium|low/),
          category: expect.any(String),
          message: expect.any(String)
        })
      );
    });
  });
});
```

## Implementation Guidelines

### 1. Start with Core Infrastructure
- Implement the discovery service first
- Set up the tool registry and status monitoring
- Create the basic catalog UI structure

### 2. Build Interactive Components
- Implement dynamic form generation
- Create the tool tester interface
- Add request/response visualization

### 3. Add Analytics and Monitoring
- Implement metrics collection
- Build performance dashboards
- Create usage analytics

### 4. Polish and Optimize
- Add comprehensive error handling
- Implement caching for better performance
- Create thorough documentation

### 5. Testing Strategy
- Unit test each component
- Integration test the full system
- Performance test with realistic load
- User acceptance testing for UI/UX

## Conclusion

Phase 3 provides a comprehensive MCP Tool Catalog and Testing system that enables users to:
- Discover and explore all available MCP tools
- Monitor tool health and performance in real-time
- Test tools interactively with dynamically generated forms
- Visualize complex request/response data
- Track usage patterns and analytics
- Access auto-generated documentation
- Monitor security status and compliance
- Manage authentication and authorization
- Review encryption configurations and certificates
- Handle token lifecycle management

The system is designed to be intuitive, performant, and extensible, providing a solid foundation for managing and testing LLMKG's MCP tool ecosystem with comprehensive security oversight.