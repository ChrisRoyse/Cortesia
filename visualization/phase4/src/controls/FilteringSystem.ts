/**
 * Advanced filtering system for LLMKG Phase 4 visualization
 * Provides real-time filtering with complex boolean combinations
 */

export interface FilterCondition {
  id: string;
  field: string;
  operator: 'equals' | 'contains' | 'startsWith' | 'endsWith' | 'regex' | 'range' | 'in' | 'not_in';
  value: any;
  type: 'string' | 'number' | 'boolean' | 'date' | 'array';
}

export interface FilterGroup {
  id: string;
  name: string;
  conditions: FilterCondition[];
  operator: 'AND' | 'OR';
  enabled: boolean;
  color?: string;
}

export interface FilterPreset {
  id: string;
  name: string;
  description: string;
  groups: FilterGroup[];
  category: 'cognitive' | 'performance' | 'memory' | 'requests' | 'custom';
  tags: string[];
  created: Date;
  lastUsed?: Date;
}

export interface TimeWindow {
  start: Date;
  end: Date;
  live: boolean;
  windowSize?: number; // for rolling window in ms
}

export interface FilteringState {
  groups: FilterGroup[];
  activePreset?: string;
  timeWindow: TimeWindow;
  globalOperator: 'AND' | 'OR';
  performance: {
    maxItems: number;
    updateFrequency: number;
    enableCaching: boolean;
  };
}

export class FilteringSystem {
  private state: FilteringState;
  private presets: Map<string, FilterPreset> = new Map();
  private cache: Map<string, any[]> = new Map();
  private listeners: Map<string, (data: any[]) => void> = new Map();
  private updateTimer?: NodeJS.Timeout;

  constructor() {
    this.state = {
      groups: [],
      timeWindow: {
        start: new Date(Date.now() - 3600000), // 1 hour ago
        end: new Date(),
        live: true,
        windowSize: 3600000
      },
      globalOperator: 'AND',
      performance: {
        maxItems: 10000,
        updateFrequency: 100, // ms
        enableCaching: true
      }
    };

    this.loadPresets();
    this.startPerformanceOptimization();
  }

  // Filter management
  addFilterGroup(name: string): FilterGroup {
    const group: FilterGroup = {
      id: crypto.randomUUID(),
      name,
      conditions: [],
      operator: 'AND',
      enabled: true
    };
    
    this.state.groups.push(group);
    this.invalidateCache();
    return group;
  }

  removeFilterGroup(groupId: string): boolean {
    const index = this.state.groups.findIndex(g => g.id === groupId);
    if (index >= 0) {
      this.state.groups.splice(index, 1);
      this.invalidateCache();
      return true;
    }
    return false;
  }

  addCondition(groupId: string, condition: Omit<FilterCondition, 'id'>): FilterCondition | null {
    const group = this.state.groups.find(g => g.id === groupId);
    if (!group) return null;

    const newCondition: FilterCondition = {
      ...condition,
      id: crypto.randomUUID()
    };

    group.conditions.push(newCondition);
    this.invalidateCache();
    return newCondition;
  }

  removeCondition(groupId: string, conditionId: string): boolean {
    const group = this.state.groups.find(g => g.id === groupId);
    if (!group) return false;

    const index = group.conditions.findIndex(c => c.id === conditionId);
    if (index >= 0) {
      group.conditions.splice(index, 1);
      this.invalidateCache();
      return true;
    }
    return false;
  }

  // Filter execution
  applyFilters(data: any[]): any[] {
    if (!data || data.length === 0) return [];
    
    const cacheKey = this.generateCacheKey();
    if (this.state.performance.enableCaching && this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    let filtered = this.applyTimeWindow(data);
    filtered = this.applyGroupFilters(filtered);
    filtered = this.applyPerformanceLimits(filtered);

    if (this.state.performance.enableCaching) {
      this.cache.set(cacheKey, filtered);
    }

    return filtered;
  }

  private applyTimeWindow(data: any[]): any[] {
    const { start, end, live, windowSize } = this.state.timeWindow;
    
    let actualStart = start;
    let actualEnd = end;

    if (live && windowSize) {
      const now = new Date();
      actualEnd = now;
      actualStart = new Date(now.getTime() - windowSize);
    }

    return data.filter(item => {
      const timestamp = this.extractTimestamp(item);
      return timestamp >= actualStart && timestamp <= actualEnd;
    });
  }

  private applyGroupFilters(data: any[]): any[] {
    if (this.state.groups.length === 0) return data;

    const enabledGroups = this.state.groups.filter(g => g.enabled);
    if (enabledGroups.length === 0) return data;

    return data.filter(item => {
      const groupResults = enabledGroups.map(group => 
        this.evaluateGroup(item, group)
      );

      return this.state.globalOperator === 'AND' 
        ? groupResults.every(r => r)
        : groupResults.some(r => r);
    });
  }

  private evaluateGroup(item: any, group: FilterGroup): boolean {
    if (group.conditions.length === 0) return true;

    const conditionResults = group.conditions.map(condition =>
      this.evaluateCondition(item, condition)
    );

    return group.operator === 'AND'
      ? conditionResults.every(r => r)
      : conditionResults.some(r => r);
  }

  private evaluateCondition(item: any, condition: FilterCondition): boolean {
    const fieldValue = this.getFieldValue(item, condition.field);
    
    switch (condition.operator) {
      case 'equals':
        return fieldValue === condition.value;
      
      case 'contains':
        return String(fieldValue).toLowerCase().includes(String(condition.value).toLowerCase());
      
      case 'startsWith':
        return String(fieldValue).toLowerCase().startsWith(String(condition.value).toLowerCase());
      
      case 'endsWith':
        return String(fieldValue).toLowerCase().endsWith(String(condition.value).toLowerCase());
      
      case 'regex':
        try {
          const regex = new RegExp(condition.value, 'i');
          return regex.test(String(fieldValue));
        } catch {
          return false;
        }
      
      case 'range':
        const [min, max] = condition.value;
        const numValue = Number(fieldValue);
        return numValue >= min && numValue <= max;
      
      case 'in':
        return Array.isArray(condition.value) && condition.value.includes(fieldValue);
      
      case 'not_in':
        return Array.isArray(condition.value) && !condition.value.includes(fieldValue);
      
      default:
        return true;
    }
  }

  private getFieldValue(item: any, field: string): any {
    // Support nested field access with dot notation
    return field.split('.').reduce((obj, key) => obj?.[key], item);
  }

  private extractTimestamp(item: any): Date {
    // Try multiple common timestamp fields
    const timestampFields = ['timestamp', 'created_at', 'time', 'date'];
    for (const field of timestampFields) {
      const value = item[field];
      if (value) {
        return new Date(value);
      }
    }
    return new Date(); // fallback to current time
  }

  private applyPerformanceLimits(data: any[]): any[] {
    if (data.length <= this.state.performance.maxItems) {
      return data;
    }
    
    // Keep most recent items when limiting
    return data
      .sort((a, b) => this.extractTimestamp(b).getTime() - this.extractTimestamp(a).getTime())
      .slice(0, this.state.performance.maxItems);
  }

  // Preset management
  savePreset(name: string, description: string, category: FilterPreset['category'], tags: string[] = []): FilterPreset {
    const preset: FilterPreset = {
      id: crypto.randomUUID(),
      name,
      description,
      groups: JSON.parse(JSON.stringify(this.state.groups)), // deep clone
      category,
      tags,
      created: new Date()
    };

    this.presets.set(preset.id, preset);
    this.savePresetsToStorage();
    return preset;
  }

  loadPreset(presetId: string): boolean {
    const preset = this.presets.get(presetId);
    if (!preset) return false;

    this.state.groups = JSON.parse(JSON.stringify(preset.groups)); // deep clone
    this.state.activePreset = presetId;
    preset.lastUsed = new Date();
    
    this.invalidateCache();
    this.savePresetsToStorage();
    return true;
  }

  deletePreset(presetId: string): boolean {
    if (this.presets.delete(presetId)) {
      if (this.state.activePreset === presetId) {
        this.state.activePreset = undefined;
      }
      this.savePresetsToStorage();
      return true;
    }
    return false;
  }

  getPresets(): FilterPreset[] {
    return Array.from(this.presets.values())
      .sort((a, b) => (b.lastUsed?.getTime() || 0) - (a.lastUsed?.getTime() || 0));
  }

  // Default presets
  private loadPresets(): void {
    this.createDefaultPresets();
    this.loadPresetsFromStorage();
  }

  private createDefaultPresets(): void {
    // Cognitive patterns preset
    const cognitivePreset: FilterPreset = {
      id: 'cognitive-patterns',
      name: 'Cognitive Patterns',
      description: 'Filter for cognitive pattern activations',
      category: 'cognitive',
      tags: ['patterns', 'thinking', 'cognitive'],
      created: new Date(),
      groups: [{
        id: 'cognitive-group',
        name: 'Pattern Types',
        operator: 'OR',
        enabled: true,
        conditions: [{
          id: 'convergent',
          field: 'pattern_type',
          operator: 'in',
          value: ['ConvergentThinking', 'DivergentThinking', 'AnalyticalReasoning'],
          type: 'array'
        }]
      }]
    };

    // Performance issues preset
    const performancePreset: FilterPreset = {
      id: 'performance-issues',
      name: 'Performance Issues',
      description: 'Show slow operations and performance problems',
      category: 'performance',
      tags: ['slow', 'performance', 'issues'],
      created: new Date(),
      groups: [{
        id: 'slow-ops',
        name: 'Slow Operations',
        operator: 'OR',
        enabled: true,
        conditions: [{
          id: 'slow-duration',
          field: 'duration',
          operator: 'range',
          value: [1000, Infinity], // > 1 second
          type: 'number'
        }]
      }]
    };

    // Memory operations preset
    const memoryPreset: FilterPreset = {
      id: 'memory-operations',
      name: 'Memory Operations',
      description: 'Filter memory-related operations',
      category: 'memory',
      tags: ['memory', 'storage', 'retrieval'],
      created: new Date(),
      groups: [{
        id: 'memory-ops',
        name: 'Operation Types',
        operator: 'OR',
        enabled: true,
        conditions: [{
          id: 'memory-types',
          field: 'operation_type',
          operator: 'in',
          value: ['store', 'retrieve', 'update', 'delete'],
          type: 'array'
        }]
      }]
    };

    this.presets.set(cognitivePreset.id, cognitivePreset);
    this.presets.set(performancePreset.id, performancePreset);
    this.presets.set(memoryPreset.id, memoryPreset);
  }

  // Storage
  private savePresetsToStorage(): void {
    try {
      const presetsData = Array.from(this.presets.entries());
      localStorage.setItem('llmkg-filter-presets', JSON.stringify(presetsData));
    } catch (error) {
      console.warn('Failed to save filter presets:', error);
    }
  }

  private loadPresetsFromStorage(): void {
    try {
      const stored = localStorage.getItem('llmkg-filter-presets');
      if (stored) {
        const presetsData: [string, FilterPreset][] = JSON.parse(stored);
        for (const [id, preset] of presetsData) {
          // Convert date strings back to Date objects
          preset.created = new Date(preset.created);
          if (preset.lastUsed) {
            preset.lastUsed = new Date(preset.lastUsed);
          }
          this.presets.set(id, preset);
        }
      }
    } catch (error) {
      console.warn('Failed to load filter presets:', error);
    }
  }

  // Performance optimization
  private startPerformanceOptimization(): void {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
    }

    this.updateTimer = setInterval(() => {
      this.cleanupCache();
      this.updateLiveTimeWindow();
    }, this.state.performance.updateFrequency);
  }

  private cleanupCache(): void {
    // Limit cache size
    if (this.cache.size > 100) {
      const keys = Array.from(this.cache.keys());
      for (let i = 0; i < keys.length - 50; i++) {
        this.cache.delete(keys[i]);
      }
    }
  }

  private updateLiveTimeWindow(): void {
    if (this.state.timeWindow.live && this.state.timeWindow.windowSize) {
      const now = new Date();
      this.state.timeWindow.end = now;
      this.state.timeWindow.start = new Date(now.getTime() - this.state.timeWindow.windowSize);
      this.invalidateCache();
    }
  }

  private invalidateCache(): void {
    this.cache.clear();
    this.notifyListeners();
  }

  private generateCacheKey(): string {
    return JSON.stringify({
      groups: this.state.groups,
      timeWindow: this.state.timeWindow,
      globalOperator: this.state.globalOperator
    });
  }

  // Event system
  addListener(id: string, callback: (data: any[]) => void): void {
    this.listeners.set(id, callback);
  }

  removeListener(id: string): void {
    this.listeners.delete(id);
  }

  private notifyListeners(): void {
    // Debounce notifications
    setTimeout(() => {
      for (const callback of this.listeners.values()) {
        callback([]);
      }
    }, 50);
  }

  // State management
  getState(): FilteringState {
    return { ...this.state };
  }

  updateTimeWindow(timeWindow: Partial<TimeWindow>): void {
    this.state.timeWindow = { ...this.state.timeWindow, ...timeWindow };
    this.invalidateCache();
  }

  updatePerformanceSettings(settings: Partial<FilteringState['performance']>): void {
    this.state.performance = { ...this.state.performance, ...settings };
    this.invalidateCache();
  }

  setGlobalOperator(operator: 'AND' | 'OR'): void {
    this.state.globalOperator = operator;
    this.invalidateCache();
  }

  toggleGroup(groupId: string): boolean {
    const group = this.state.groups.find(g => g.id === groupId);
    if (group) {
      group.enabled = !group.enabled;
      this.invalidateCache();
      return true;
    }
    return false;
  }

  // Cleanup
  destroy(): void {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
    }
    this.cache.clear();
    this.listeners.clear();
  }
}

// Singleton instance
export const filteringSystem = new FilteringSystem();