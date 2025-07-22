import { MCPTool, ToolCategory, ToolSchema, ToolStatusInfo, ToolStatus, JSONSchema } from '../types';

interface MCPEndpointInfo {
  url: string;
  type: 'knowledge' | 'cognitive' | 'neural' | 'memory' | 'federation' | 'analysis';
  priority: number;
}

interface RawMCPTool {
  name: string;
  description: string;
  inputSchema?: JSONSchema;
  outputSchema?: JSONSchema;
  version?: string;
  category?: string;
  tags?: string[];
  examples?: any[];
}

interface DiscoveryOptions {
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  parallel?: boolean;
}

export class ToolDiscoveryService {
  private static instance: ToolDiscoveryService;
  private discoveryCache: Map<string, { tools: MCPTool[], timestamp: number }> = new Map();
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes
  private defaultTimeout = 5000;
  private maxRetries = 3;
  private baseRetryDelay = 1000;

  private constructor() {}

  static getInstance(): ToolDiscoveryService {
    if (!ToolDiscoveryService.instance) {
      ToolDiscoveryService.instance = new ToolDiscoveryService();
    }
    return ToolDiscoveryService.instance;
  }

  /**
   * Discover tools from multiple MCP endpoints
   */
  async discoverTools(
    endpoints: string[],
    options: DiscoveryOptions = {}
  ): Promise<MCPTool[]> {
    const {
      timeout = this.defaultTimeout,
      retries = this.maxRetries,
      retryDelay = this.baseRetryDelay,
      parallel = true
    } = options;

    if (parallel) {
      // Discover from all endpoints in parallel
      const discoveryPromises = endpoints.map(endpoint =>
        this.discoverFromEndpoint(endpoint, { timeout, retries, retryDelay })
          .catch(error => {
            console.error(`Failed to discover tools from ${endpoint}:`, error);
            return [];
          })
      );

      const results = await Promise.all(discoveryPromises);
      return results.flat();
    } else {
      // Discover sequentially
      const allTools: MCPTool[] = [];
      for (const endpoint of endpoints) {
        try {
          const tools = await this.discoverFromEndpoint(endpoint, { timeout, retries, retryDelay });
          allTools.push(...tools);
        } catch (error) {
          console.error(`Failed to discover tools from ${endpoint}:`, error);
        }
      }
      return allTools;
    }
  }

  /**
   * Discover tools from a single endpoint with caching
   */
  private async discoverFromEndpoint(
    endpoint: string,
    options: DiscoveryOptions
  ): Promise<MCPTool[]> {
    // Check cache first
    const cached = this.discoveryCache.get(endpoint);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.tools;
    }

    const tools = await this.fetchToolsWithRetry(endpoint, options);
    
    // Cache the results
    this.discoveryCache.set(endpoint, { tools, timestamp: Date.now() });
    
    return tools;
  }

  /**
   * Fetch tools from endpoint with exponential backoff retry
   */
  private async fetchToolsWithRetry(
    endpoint: string,
    options: DiscoveryOptions
  ): Promise<MCPTool[]> {
    const { timeout = this.defaultTimeout, retries = this.maxRetries, retryDelay = this.baseRetryDelay } = options;
    
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const response = await fetch(`${endpoint}/mcp/tools`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        const rawTools = Array.isArray(data) ? data : data.tools || [];
        
        // Process and validate each tool
        const processedTools = await Promise.all(
          rawTools.map(async (rawTool: RawMCPTool) => {
            try {
              return await this.processRawTool(rawTool, endpoint);
            } catch (error) {
              console.warn(`Failed to process tool ${rawTool.name}:`, error);
              return null;
            }
          })
        );

        return processedTools.filter((tool): tool is MCPTool => tool !== null);
      } catch (error) {
        lastError = error as Error;
        
        if (attempt < retries) {
          // Exponential backoff
          const delay = retryDelay * Math.pow(2, attempt);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError || new Error('Failed to discover tools');
  }

  /**
   * Process raw tool data into MCPTool format
   */
  private async processRawTool(rawTool: RawMCPTool, endpoint: string): Promise<MCPTool> {
    const id = this.generateToolId(rawTool.name, endpoint);
    const category = this.categorizeTools(rawTool);
    const schema = await this.fetchToolSchema(rawTool, endpoint);
    const status = await this.checkToolStatus(rawTool.name, endpoint);

    return {
      id,
      name: rawTool.name,
      version: rawTool.version || '1.0.0',
      description: rawTool.description || 'No description available',
      category,
      schema,
      status,
      metrics: {
        totalExecutions: 0,
        successRate: 100,
        averageResponseTime: 0,
        p95ResponseTime: 0,
        p99ResponseTime: 0,
        errorCount: 0,
        errorTypes: {},
      },
      documentation: {
        summary: rawTool.description || '',
        description: rawTool.description || '',
        parameters: this.extractParameters(schema.inputSchema),
        returns: {
          type: schema.outputSchema.type || 'object',
          description: 'Tool execution result',
          schema: schema.outputSchema,
        },
        examples: this.formatExamples(rawTool.examples),
        tags: rawTool.tags,
      },
      endpoint,
      tags: rawTool.tags || [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
  }

  /**
   * Fetch detailed schema for a tool
   */
  async fetchToolSchema(tool: RawMCPTool, endpoint: string): Promise<ToolSchema> {
    // If schema is already provided, use it
    if (tool.inputSchema && tool.outputSchema) {
      return {
        inputSchema: tool.inputSchema,
        outputSchema: tool.outputSchema,
        examples: tool.examples?.map(ex => ({
          name: ex.name || 'Example',
          description: ex.description || '',
          input: ex.input || {},
          output: ex.output,
          tags: ex.tags,
        })),
      };
    }

    // Otherwise, try to fetch detailed schema
    try {
      const response = await fetch(`${endpoint}/mcp/tools/${tool.name}/schema`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn(`Failed to fetch schema for ${tool.name}:`, error);
    }

    // Return default schema
    return {
      inputSchema: { type: 'object', properties: {}, additionalProperties: true },
      outputSchema: { type: 'object', properties: {}, additionalProperties: true },
    };
  }

  /**
   * Categorize tools based on name and tags
   */
  categorizeTools(tool: RawMCPTool): ToolCategory {
    const name = tool.name.toLowerCase();
    const tags = (tool.tags || []).map(t => t.toLowerCase());
    const category = tool.category?.toLowerCase();

    // Check explicit category
    if (category) {
      if (category.includes('knowledge') || category.includes('graph')) return 'knowledge-graph';
      if (category.includes('cognitive') || category.includes('pattern')) return 'cognitive';
      if (category.includes('neural') || category.includes('spike')) return 'neural';
      if (category.includes('memory') || category.includes('storage')) return 'memory';
      if (category.includes('federation') || category.includes('distributed')) return 'federation';
      if (category.includes('analysis') || category.includes('debug')) return 'analysis';
    }

    // Check tool name
    if (name.includes('knowledge') || name.includes('graph') || name.includes('triple')) {
      return 'knowledge-graph';
    }
    if (name.includes('cognitive') || name.includes('pattern') || name.includes('attention')) {
      return 'cognitive';
    }
    if (name.includes('neural') || name.includes('spike') || name.includes('neuron')) {
      return 'neural';
    }
    if (name.includes('memory') || name.includes('store') || name.includes('retrieve')) {
      return 'memory';
    }
    if (name.includes('federation') || name.includes('cluster') || name.includes('distributed')) {
      return 'federation';
    }
    if (name.includes('analyze') || name.includes('debug') || name.includes('monitor')) {
      return 'analysis';
    }

    // Check tags
    for (const tag of tags) {
      if (tag.includes('knowledge') || tag.includes('graph')) return 'knowledge-graph';
      if (tag.includes('cognitive') || tag.includes('pattern')) return 'cognitive';
      if (tag.includes('neural') || tag.includes('spike')) return 'neural';
      if (tag.includes('memory') || tag.includes('storage')) return 'memory';
      if (tag.includes('federation') || tag.includes('distributed')) return 'federation';
      if (tag.includes('analysis') || tag.includes('debug')) return 'analysis';
    }

    return 'utility';
  }

  /**
   * Validate tool to ensure it has required properties
   */
  async validateTool(tool: MCPTool): Promise<boolean> {
    try {
      // Check required fields
      if (!tool.id || !tool.name || !tool.description) {
        return false;
      }

      // Validate schema
      if (!tool.schema || !tool.schema.inputSchema || !tool.schema.outputSchema) {
        return false;
      }

      // Validate category
      const validCategories: ToolCategory[] = [
        'knowledge-graph', 'cognitive', 'neural', 'memory',
        'analysis', 'federation', 'utility'
      ];
      if (!validCategories.includes(tool.category)) {
        return false;
      }

      // Check endpoint connectivity if provided
      if (tool.endpoint) {
        const status = await this.checkToolStatus(tool.name, tool.endpoint);
        return status.available;
      }

      return true;
    } catch (error) {
      console.error(`Validation failed for tool ${tool.name}:`, error);
      return false;
    }
  }

  /**
   * Check the current status of a tool
   */
  private async checkToolStatus(toolName: string, endpoint: string): Promise<ToolStatusInfo> {
    const startTime = Date.now();
    
    try {
      const response = await fetch(`${endpoint}/mcp/tools/${toolName}/status`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      const responseTime = Date.now() - startTime;

      if (response.ok) {
        const data = await response.json();
        return {
          available: true,
          health: 'healthy' as ToolStatus,
          lastChecked: new Date(),
          responseTime,
          errorRate: data.errorRate || 0,
          message: data.message || 'Tool is healthy',
          details: data.details || {},
        };
      } else {
        return {
          available: response.status < 500,
          health: response.status >= 500 ? 'unavailable' : 'degraded' as ToolStatus,
          lastChecked: new Date(),
          responseTime,
          errorRate: 0,
          message: `HTTP ${response.status}: ${response.statusText}`,
        };
      }
    } catch (error) {
      return {
        available: false,
        health: 'unavailable' as ToolStatus,
        lastChecked: new Date(),
        responseTime: Date.now() - startTime,
        errorRate: 100,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Generate unique tool ID
   */
  private generateToolId(toolName: string, endpoint: string): string {
    const endpointHash = this.hashCode(endpoint);
    return `${toolName}_${endpointHash}`;
  }

  /**
   * Simple hash function for strings
   */
  private hashCode(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Extract parameter documentation from schema
   */
  private extractParameters(schema: JSONSchema): any[] {
    if (!schema.properties) return [];

    return Object.entries(schema.properties).map(([name, prop]) => ({
      name,
      type: prop.type || 'any',
      description: prop.description || '',
      required: schema.required?.includes(name) || false,
      default: prop.default,
      examples: prop.examples,
    }));
  }

  /**
   * Format examples for documentation
   */
  private formatExamples(examples?: any[]): any[] {
    if (!examples || !Array.isArray(examples)) return [];

    return examples.map(ex => ({
      language: 'javascript',
      code: JSON.stringify(ex, null, 2),
      description: ex.description || '',
    }));
  }

  /**
   * Clear discovery cache
   */
  clearCache(): void {
    this.discoveryCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; endpoints: string[] } {
    return {
      size: this.discoveryCache.size,
      endpoints: Array.from(this.discoveryCache.keys()),
    };
  }
}

export default ToolDiscoveryService.getInstance();