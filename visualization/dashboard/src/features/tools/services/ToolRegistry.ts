import { MCPTool, ToolCategory, ToolStatusInfo, ToolFilter, ToolStatus } from '../types';

interface RegistryOptions {
  maxTools?: number;
  enableVersioning?: boolean;
  enablePersistence?: boolean;
}

interface ToolVersion {
  version: string;
  tool: MCPTool;
  registeredAt: Date;
}

export class ToolRegistry {
  private static instance: ToolRegistry;
  private tools: Map<string, MCPTool> = new Map();
  private toolVersions: Map<string, ToolVersion[]> = new Map();
  private toolsByCategory: Map<ToolCategory, Set<string>> = new Map();
  private toolTags: Map<string, Set<string>> = new Map();
  private searchIndex: Map<string, Set<string>> = new Map();
  private options: Required<RegistryOptions>;
  private listeners: Set<(event: RegistryEvent) => void> = new Set();

  private constructor(options: RegistryOptions = {}) {
    this.options = {
      maxTools: options.maxTools || 1000,
      enableVersioning: options.enableVersioning ?? true,
      enablePersistence: options.enablePersistence ?? true,
    };

    // Initialize category maps
    const categories: ToolCategory[] = [
      'knowledge-graph', 'cognitive', 'neural', 'memory',
      'analysis', 'federation', 'utility'
    ];
    categories.forEach(category => {
      this.toolsByCategory.set(category, new Set());
    });

    // Load from persistence if enabled
    if (this.options.enablePersistence) {
      this.loadFromStorage();
    }
  }

  static getInstance(options?: RegistryOptions): ToolRegistry {
    if (!ToolRegistry.instance) {
      ToolRegistry.instance = new ToolRegistry(options);
    }
    return ToolRegistry.instance;
  }

  /**
   * Register a new tool or update existing one
   */
  registerTool(tool: MCPTool): void {
    const existingTool = this.tools.get(tool.id);
    
    // Check capacity
    if (!existingTool && this.tools.size >= this.options.maxTools) {
      throw new Error(`Registry is full. Maximum ${this.options.maxTools} tools allowed.`);
    }

    // Store version history if enabled
    if (this.options.enableVersioning) {
      this.addToolVersion(tool);
    }

    // Update main registry
    this.tools.set(tool.id, tool);

    // Update category index
    if (existingTool && existingTool.category !== tool.category) {
      this.toolsByCategory.get(existingTool.category)?.delete(tool.id);
    }
    this.toolsByCategory.get(tool.category)?.add(tool.id);

    // Update tag index
    this.updateTagIndex(tool);

    // Update search index
    this.updateSearchIndex(tool);

    // Persist if enabled
    if (this.options.enablePersistence) {
      this.saveToStorage();
    }

    // Notify listeners
    this.emit({
      type: existingTool ? 'tool-updated' : 'tool-registered',
      toolId: tool.id,
      tool,
    });
  }

  /**
   * Register multiple tools at once
   */
  registerTools(tools: MCPTool[]): void {
    const results: { success: string[]; failed: Array<{ id: string; error: string }> } = {
      success: [],
      failed: [],
    };

    for (const tool of tools) {
      try {
        this.registerTool(tool);
        results.success.push(tool.id);
      } catch (error) {
        results.failed.push({
          id: tool.id,
          error: error instanceof Error ? error.message : 'Unknown error',
        });
      }
    }

    // Emit batch registration event
    this.emit({
      type: 'batch-registered',
      results,
    });
  }

  /**
   * Get a tool by ID
   */
  getTool(id: string): MCPTool | undefined {
    return this.tools.get(id);
  }

  /**
   * Get tools by category
   */
  getToolsByCategory(category: ToolCategory): MCPTool[] {
    const toolIds = this.toolsByCategory.get(category) || new Set();
    return Array.from(toolIds)
      .map(id => this.tools.get(id))
      .filter((tool): tool is MCPTool => tool !== undefined);
  }

  /**
   * Search tools by query
   */
  searchTools(query: string): MCPTool[] {
    const normalizedQuery = query.toLowerCase().trim();
    if (!normalizedQuery) return Array.from(this.tools.values());

    const terms = normalizedQuery.split(/\s+/);
    const matchingIds = new Set<string>();

    // Search across all terms
    for (const term of terms) {
      const termMatches = this.searchIndex.get(term) || new Set();
      if (matchingIds.size === 0) {
        termMatches.forEach(id => matchingIds.add(id));
      } else {
        // Intersection with previous results
        const intersection = new Set<string>();
        termMatches.forEach(id => {
          if (matchingIds.has(id)) intersection.add(id);
        });
        matchingIds.clear();
        intersection.forEach(id => matchingIds.add(id));
      }
    }

    // Also do partial matching on name and description
    this.tools.forEach((tool, id) => {
      const searchText = `${tool.name} ${tool.description}`.toLowerCase();
      if (terms.every(term => searchText.includes(term))) {
        matchingIds.add(id);
      }
    });

    return Array.from(matchingIds)
      .map(id => this.tools.get(id))
      .filter((tool): tool is MCPTool => tool !== undefined);
  }

  /**
   * Filter tools with advanced criteria
   */
  filterTools(filter: ToolFilter): MCPTool[] {
    let results = Array.from(this.tools.values());

    // Filter by categories
    if (filter.categories && filter.categories.length > 0) {
      results = results.filter(tool => filter.categories!.includes(tool.category));
    }

    // Filter by status
    if (filter.status && filter.status.length > 0) {
      results = results.filter(tool => filter.status!.includes(tool.status.health));
    }

    // Filter by search term
    if (filter.searchTerm) {
      const searchResults = this.searchTools(filter.searchTerm);
      const searchIds = new Set(searchResults.map(t => t.id));
      results = results.filter(tool => searchIds.has(tool.id));
    }

    // Filter by tags
    if (filter.tags && filter.tags.length > 0) {
      results = results.filter(tool => 
        filter.tags!.some(tag => tool.tags?.includes(tag))
      );
    }

    // Sort results
    if (filter.sortBy) {
      results.sort((a, b) => {
        let comparison = 0;
        
        switch (filter.sortBy) {
          case 'name':
            comparison = a.name.localeCompare(b.name);
            break;
          case 'category':
            comparison = a.category.localeCompare(b.category);
            break;
          case 'status':
            comparison = this.getStatusPriority(a.status.health) - this.getStatusPriority(b.status.health);
            break;
          case 'performance':
            comparison = a.metrics.averageResponseTime - b.metrics.averageResponseTime;
            break;
        }

        return filter.sortOrder === 'desc' ? -comparison : comparison;
      });
    }

    return results;
  }

  /**
   * Update tool status
   */
  updateToolStatus(id: string, status: ToolStatusInfo): void {
    const tool = this.tools.get(id);
    if (!tool) {
      throw new Error(`Tool ${id} not found`);
    }

    const updatedTool = {
      ...tool,
      status,
      updatedAt: new Date(),
    };

    this.registerTool(updatedTool);

    this.emit({
      type: 'tool-status-updated',
      toolId: id,
      status,
    });
  }

  /**
   * Update tool metrics
   */
  updateToolMetrics(id: string, metrics: Partial<MCPTool['metrics']>): void {
    const tool = this.tools.get(id);
    if (!tool) {
      throw new Error(`Tool ${id} not found`);
    }

    const updatedTool = {
      ...tool,
      metrics: { ...tool.metrics, ...metrics },
      updatedAt: new Date(),
    };

    this.registerTool(updatedTool);
  }

  /**
   * Remove a tool from registry
   */
  removeTool(id: string): boolean {
    const tool = this.tools.get(id);
    if (!tool) return false;

    // Remove from main registry
    this.tools.delete(id);

    // Remove from category index
    this.toolsByCategory.get(tool.category)?.delete(id);

    // Remove from tag index
    tool.tags?.forEach(tag => {
      this.toolTags.get(tag)?.delete(id);
    });

    // Remove from search index
    this.removeFromSearchIndex(id);

    // Persist if enabled
    if (this.options.enablePersistence) {
      this.saveToStorage();
    }

    // Notify listeners
    this.emit({
      type: 'tool-removed',
      toolId: id,
    });

    return true;
  }

  /**
   * Get all tools
   */
  getAllTools(): MCPTool[] {
    return Array.from(this.tools.values());
  }

  /**
   * Get tools grouped by category
   */
  getToolsGroupedByCategory(): Record<ToolCategory, MCPTool[]> {
    const grouped: Record<ToolCategory, MCPTool[]> = {} as any;
    
    this.toolsByCategory.forEach((toolIds, category) => {
      grouped[category] = Array.from(toolIds)
        .map(id => this.tools.get(id))
        .filter((tool): tool is MCPTool => tool !== undefined);
    });

    return grouped;
  }

  /**
   * Get registry statistics
   */
  getStats(): RegistryStats {
    const categoryCounts: Record<ToolCategory, number> = {} as any;
    this.toolsByCategory.forEach((ids, category) => {
      categoryCounts[category] = ids.size;
    });

    const statusCounts: Record<ToolStatus, number> = {
      healthy: 0,
      degraded: 0,
      unavailable: 0,
      unknown: 0,
    };

    this.tools.forEach(tool => {
      statusCounts[tool.status.health]++;
    });

    return {
      totalTools: this.tools.size,
      categoryCounts,
      statusCounts,
      totalTags: this.toolTags.size,
      averageResponseTime: this.calculateAverageResponseTime(),
      lastUpdated: new Date(),
    };
  }

  /**
   * Clear all tools from registry
   */
  clear(): void {
    this.tools.clear();
    this.toolVersions.clear();
    this.toolsByCategory.forEach(set => set.clear());
    this.toolTags.clear();
    this.searchIndex.clear();

    if (this.options.enablePersistence) {
      this.clearStorage();
    }

    this.emit({ type: 'registry-cleared' });
  }

  /**
   * Subscribe to registry events
   */
  subscribe(listener: (event: RegistryEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // Private helper methods

  private updateTagIndex(tool: MCPTool): void {
    // Remove old tags
    this.toolTags.forEach((toolIds, tag) => {
      toolIds.delete(tool.id);
    });

    // Add new tags
    tool.tags?.forEach(tag => {
      if (!this.toolTags.has(tag)) {
        this.toolTags.set(tag, new Set());
      }
      this.toolTags.get(tag)!.add(tool.id);
    });
  }

  private updateSearchIndex(tool: MCPTool): void {
    // Remove old entries
    this.removeFromSearchIndex(tool.id);

    // Index tool properties
    const searchableText = [
      tool.name,
      tool.description,
      tool.category,
      ...(tool.tags || []),
    ].join(' ').toLowerCase();

    // Create index entries for each word
    const words = searchableText.split(/\s+/).filter(word => word.length > 2);
    words.forEach(word => {
      if (!this.searchIndex.has(word)) {
        this.searchIndex.set(word, new Set());
      }
      this.searchIndex.get(word)!.add(tool.id);
    });
  }

  private removeFromSearchIndex(toolId: string): void {
    this.searchIndex.forEach(toolIds => {
      toolIds.delete(toolId);
    });
  }

  private addToolVersion(tool: MCPTool): void {
    if (!this.toolVersions.has(tool.id)) {
      this.toolVersions.set(tool.id, []);
    }

    const versions = this.toolVersions.get(tool.id)!;
    versions.push({
      version: tool.version,
      tool: { ...tool },
      registeredAt: new Date(),
    });

    // Keep only last 10 versions
    if (versions.length > 10) {
      this.toolVersions.set(tool.id, versions.slice(-10));
    }
  }

  private getStatusPriority(status: ToolStatus): number {
    const priorities: Record<ToolStatus, number> = {
      healthy: 0,
      degraded: 1,
      unknown: 2,
      unavailable: 3,
    };
    return priorities[status];
  }

  private calculateAverageResponseTime(): number {
    const responseTimes = Array.from(this.tools.values())
      .map(tool => tool.metrics.averageResponseTime)
      .filter(time => time > 0);

    if (responseTimes.length === 0) return 0;

    return responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length;
  }

  private emit(event: RegistryEvent): void {
    this.listeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        console.error('Registry event listener error:', error);
      }
    });
  }

  // Persistence methods

  private saveToStorage(): void {
    try {
      const data = {
        tools: Array.from(this.tools.entries()),
        toolVersions: Array.from(this.toolVersions.entries()),
        timestamp: Date.now(),
      };
      localStorage.setItem('llmkg-tool-registry', JSON.stringify(data));
    } catch (error) {
      console.error('Failed to save registry to storage:', error);
    }
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem('llmkg-tool-registry');
      if (!stored) return;

      const data = JSON.parse(stored);
      
      // Restore tools
      data.tools?.forEach(([id, tool]: [string, MCPTool]) => {
        // Convert date strings back to Date objects
        tool.createdAt = new Date(tool.createdAt);
        tool.updatedAt = new Date(tool.updatedAt);
        tool.status.lastChecked = new Date(tool.status.lastChecked);
        
        this.tools.set(id, tool);
        this.toolsByCategory.get(tool.category)?.add(id);
        this.updateTagIndex(tool);
        this.updateSearchIndex(tool);
      });

      // Restore version history
      data.toolVersions?.forEach(([id, versions]: [string, ToolVersion[]]) => {
        this.toolVersions.set(id, versions.map(v => ({
          ...v,
          registeredAt: new Date(v.registeredAt),
        })));
      });
    } catch (error) {
      console.error('Failed to load registry from storage:', error);
    }
  }

  private clearStorage(): void {
    try {
      localStorage.removeItem('llmkg-tool-registry');
    } catch (error) {
      console.error('Failed to clear registry storage:', error);
    }
  }
}

// Type definitions for registry events and stats

type RegistryEvent = 
  | { type: 'tool-registered'; toolId: string; tool: MCPTool }
  | { type: 'tool-updated'; toolId: string; tool: MCPTool }
  | { type: 'tool-removed'; toolId: string }
  | { type: 'tool-status-updated'; toolId: string; status: ToolStatusInfo }
  | { type: 'batch-registered'; results: { success: string[]; failed: Array<{ id: string; error: string }> } }
  | { type: 'registry-cleared' };

interface RegistryStats {
  totalTools: number;
  categoryCounts: Record<ToolCategory, number>;
  statusCounts: Record<ToolStatus, number>;
  totalTags: number;
  averageResponseTime: number;
  lastUpdated: Date;
}

export default ToolRegistry.getInstance();