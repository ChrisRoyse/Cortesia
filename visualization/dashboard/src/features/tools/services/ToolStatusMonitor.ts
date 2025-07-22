import { MCPTool, ToolStatusInfo, ToolStatus } from '../types';
import { store } from '../../../app/store';
import { updateMultipleToolStatus } from '../stores/toolsSlice';

export interface StatusHistory {
  timestamp: Date;
  status: ToolStatus;
  responseTime: number;
  errorRate: number;
  available: boolean;
  details?: Record<string, any>;
}

export interface AlertThresholds {
  responseTime: {
    healthy: number;    // < 200ms
    degraded: number;   // 200-500ms
    unavailable: number; // > 500ms
  };
  errorRate: {
    healthy: number;    // < 1%
    degraded: number;   // 1-5%
    unavailable: number; // > 5%
  };
  availability: {
    healthy: number;    // 100%
    degraded: number;   // > 95%
    unavailable: number; // < 95%
  };
}

export interface MonitoringConfig {
  interval: number;
  timeout: number;
  retryAttempts: number;
  historyRetentionHours: number;
}

export type StatusChangeCallback = (toolId: string, oldStatus: ToolStatus, newStatus: ToolStatus) => void;
export type AlertCallback = (toolId: string, status: ToolStatus, message: string) => void;

class ToolStatusMonitor {
  private static instance: ToolStatusMonitor;
  private monitoring: Map<string, NodeJS.Timeout> = new Map();
  private statusHistory: Map<string, StatusHistory[]> = new Map();
  private config: MonitoringConfig = {
    interval: 30000, // 30 seconds
    timeout: 5000,   // 5 seconds
    retryAttempts: 3,
    historyRetentionHours: 24
  };
  private thresholds: AlertThresholds = {
    responseTime: {
      healthy: 200,
      degraded: 500,
      unavailable: 1000
    },
    errorRate: {
      healthy: 0.01,    // 1%
      degraded: 0.05,   // 5%
      unavailable: 0.10 // 10%
    },
    availability: {
      healthy: 1.0,     // 100%
      degraded: 0.95,   // 95%
      unavailable: 0.90 // 90%
    }
  };
  private statusChangeCallbacks: StatusChangeCallback[] = [];
  private alertCallbacks: AlertCallback[] = [];
  private webSocketConnection: WebSocket | null = null;

  private constructor() {
    this.initializeWebSocket();
    this.startHistoryCleanup();
  }

  static getInstance(): ToolStatusMonitor {
    if (!ToolStatusMonitor.instance) {
      ToolStatusMonitor.instance = new ToolStatusMonitor();
    }
    return ToolStatusMonitor.instance;
  }

  // Initialize WebSocket connection for real-time updates
  private initializeWebSocket(): void {
    try {
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:3001';
      this.webSocketConnection = new WebSocket(`${wsUrl}/tool-status`);

      this.webSocketConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'status-update') {
          this.handleWebSocketStatusUpdate(data.toolId, data.status);
        }
      };

      this.webSocketConnection.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      this.webSocketConnection.onclose = () => {
        // Attempt to reconnect after 5 seconds
        setTimeout(() => this.initializeWebSocket(), 5000);
      };
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  }

  // Start monitoring specific tools
  startMonitoring(toolIds: string[], interval?: number): void {
    const monitoringInterval = interval || this.config.interval;

    toolIds.forEach(toolId => {
      // Clear existing monitoring if any
      this.stopMonitoring([toolId]);

      // Immediate check
      this.checkToolHealthWithRetry(toolId);

      // Set up periodic monitoring
      const intervalId = setInterval(() => {
        this.checkToolHealthWithRetry(toolId);
      }, monitoringInterval);

      this.monitoring.set(toolId, intervalId);
    });
  }

  // Stop monitoring specific tools
  stopMonitoring(toolIds?: string[]): void {
    if (!toolIds) {
      // Stop all monitoring
      this.monitoring.forEach((intervalId) => clearInterval(intervalId));
      this.monitoring.clear();
    } else {
      toolIds.forEach(toolId => {
        const intervalId = this.monitoring.get(toolId);
        if (intervalId) {
          clearInterval(intervalId);
          this.monitoring.delete(toolId);
        }
      });
    }
  }

  // Check tool health with retry logic
  private async checkToolHealthWithRetry(toolId: string): Promise<void> {
    const state = store.getState();
    const tool = state.tools.tools.find(t => t.id === toolId);
    if (!tool) return;

    let lastError: Error | null = null;
    for (let attempt = 0; attempt < this.config.retryAttempts; attempt++) {
      try {
        const statusInfo = await this.checkToolHealth(tool);
        this.updateToolStatus(tool, statusInfo);
        return;
      } catch (error) {
        lastError = error as Error;
        if (attempt < this.config.retryAttempts - 1) {
          await this.delay(1000 * (attempt + 1)); // Exponential backoff
        }
      }
    }

    // All retries failed
    if (lastError) {
      const statusInfo: ToolStatusInfo = {
        available: false,
        health: 'unavailable',
        lastChecked: new Date(),
        responseTime: this.config.timeout,
        errorRate: 1.0,
        message: `Health check failed: ${lastError.message}`,
        details: { error: lastError.message }
      };
      this.updateToolStatus(tool, statusInfo);
    }
  }

  // Perform actual health check
  async checkToolHealth(tool: MCPTool): Promise<ToolStatusInfo> {
    const startTime = Date.now();
    
    try {
      // Simulate health check - in real implementation, this would call the tool's endpoint
      const response = await this.performHealthCheck(tool);
      const responseTime = Date.now() - startTime;

      // Calculate status based on response
      const status = this.calculateHealthStatus(responseTime, response.errorRate, response.availability);

      return {
        available: response.available,
        health: status,
        lastChecked: new Date(),
        responseTime,
        errorRate: response.errorRate,
        message: response.message,
        details: response.details
      };
    } catch (error) {
      const responseTime = Date.now() - startTime;
      return {
        available: false,
        health: 'unavailable',
        lastChecked: new Date(),
        responseTime,
        errorRate: 1.0,
        message: `Health check failed: ${(error as Error).message}`,
        details: { error: (error as Error).message }
      };
    }
  }

  // Simulate health check (replace with actual implementation)
  private async performHealthCheck(tool: MCPTool): Promise<{
    available: boolean;
    errorRate: number;
    availability: number;
    message?: string;
    details?: Record<string, any>;
  }> {
    // Simulate network delay
    await this.delay(Math.random() * 200);

    // LLMKG-specific monitoring logic
    const categorySpecificChecks = {
      'cognitive': () => this.checkCognitiveToolHealth(tool),
      'neural': () => this.checkNeuralToolHealth(tool),
      'knowledge-graph': () => this.checkKnowledgeGraphHealth(tool),
      'memory': () => this.checkMemoryToolHealth(tool),
      'federation': () => this.checkFederationToolHealth(tool),
      'analysis': () => this.checkAnalysisToolHealth(tool),
      'utility': () => this.checkUtilityToolHealth(tool)
    };

    const checkFunction = categorySpecificChecks[tool.category] || (() => this.defaultHealthCheck(tool));
    return checkFunction();
  }

  // Category-specific health checks
  private async checkCognitiveToolHealth(tool: MCPTool): Promise<any> {
    // Check pattern recognition performance
    const patternRecognitionLatency = Math.random() * 150 + 50;
    const errorRate = Math.random() * 0.02;
    
    return {
      available: true,
      errorRate,
      availability: 0.98 + Math.random() * 0.02,
      message: 'Cognitive systems operational',
      details: {
        patternRecognitionLatency,
        activePatterns: Math.floor(Math.random() * 100),
        inhibitoryBalance: 0.7 + Math.random() * 0.3
      }
    };
  }

  private async checkNeuralToolHealth(tool: MCPTool): Promise<any> {
    // Monitor neural activity response times
    const neuralResponseTime = Math.random() * 100 + 50;
    const spikeRate = Math.random() * 1000;
    
    return {
      available: true,
      errorRate: Math.random() * 0.01,
      availability: 0.99 + Math.random() * 0.01,
      message: 'Neural activity within normal parameters',
      details: {
        neuralResponseTime,
        spikeRate,
        synchronization: 0.8 + Math.random() * 0.2,
        plasticityIndex: 0.6 + Math.random() * 0.4
      }
    };
  }

  private async checkKnowledgeGraphHealth(tool: MCPTool): Promise<any> {
    // Check knowledge graph query performance
    const queryLatency = Math.random() * 200 + 100;
    const indexHealth = 0.95 + Math.random() * 0.05;
    
    return {
      available: true,
      errorRate: Math.random() * 0.03,
      availability: 0.97 + Math.random() * 0.03,
      message: 'Knowledge graph queries operational',
      details: {
        queryLatency,
        indexHealth,
        tripleCount: Math.floor(Math.random() * 1000000),
        cacheHitRate: 0.7 + Math.random() * 0.3
      }
    };
  }

  private async checkMemoryToolHealth(tool: MCPTool): Promise<any> {
    // Monitor memory consolidation rates
    const consolidationRate = 0.85 + Math.random() * 0.15;
    const memoryUtilization = Math.random() * 0.8;
    
    return {
      available: true,
      errorRate: Math.random() * 0.02,
      availability: 0.98 + Math.random() * 0.02,
      message: 'Memory systems functioning normally',
      details: {
        consolidationRate,
        memoryUtilization,
        compressionRatio: 2.5 + Math.random() * 1.5,
        retrievalLatency: Math.random() * 50 + 20
      }
    };
  }

  private async checkFederationToolHealth(tool: MCPTool): Promise<any> {
    // Track multi-instance connectivity
    const connectedInstances = Math.floor(Math.random() * 5) + 1;
    const syncLatency = Math.random() * 500 + 200;
    
    return {
      available: connectedInstances > 0,
      errorRate: Math.random() * 0.05,
      availability: 0.95 + Math.random() * 0.05,
      message: `Connected to ${connectedInstances} instances`,
      details: {
        connectedInstances,
        syncLatency,
        consensusHealth: 0.9 + Math.random() * 0.1,
        networkBandwidth: Math.random() * 1000 + 500
      }
    };
  }

  private async checkAnalysisToolHealth(tool: MCPTool): Promise<any> {
    return this.defaultHealthCheck(tool);
  }

  private async checkUtilityToolHealth(tool: MCPTool): Promise<any> {
    return this.defaultHealthCheck(tool);
  }

  private async defaultHealthCheck(tool: MCPTool): Promise<any> {
    return {
      available: Math.random() > 0.05,
      errorRate: Math.random() * 0.05,
      availability: 0.95 + Math.random() * 0.05,
      message: 'Service operational',
      details: {
        uptime: Math.floor(Math.random() * 86400),
        requestsPerSecond: Math.floor(Math.random() * 100)
      }
    };
  }

  // Calculate health status based on metrics
  private calculateHealthStatus(responseTime: number, errorRate: number, availability: number): ToolStatus {
    let score = 0;
    let factors = 0;

    // Response time scoring
    if (responseTime < this.thresholds.responseTime.healthy) {
      score += 3;
    } else if (responseTime < this.thresholds.responseTime.degraded) {
      score += 2;
    } else if (responseTime < this.thresholds.responseTime.unavailable) {
      score += 1;
    }
    factors++;

    // Error rate scoring
    if (errorRate < this.thresholds.errorRate.healthy) {
      score += 3;
    } else if (errorRate < this.thresholds.errorRate.degraded) {
      score += 2;
    } else if (errorRate < this.thresholds.errorRate.unavailable) {
      score += 1;
    }
    factors++;

    // Availability scoring
    if (availability >= this.thresholds.availability.healthy) {
      score += 3;
    } else if (availability >= this.thresholds.availability.degraded) {
      score += 2;
    } else if (availability >= this.thresholds.availability.unavailable) {
      score += 1;
    }
    factors++;

    const averageScore = score / factors;
    
    if (averageScore >= 2.5) return 'healthy';
    if (averageScore >= 1.5) return 'degraded';
    return 'unavailable';
  }

  // Update tool status and trigger callbacks
  private updateToolStatus(tool: MCPTool, statusInfo: ToolStatusInfo): void {
    const oldStatus = tool.status.health;
    const newStatus = statusInfo.health;

    // Update history
    this.addToHistory(tool.id, statusInfo);

    // Update store
    store.dispatch(updateMultipleToolStatus([{
      id: tool.id,
      status: statusInfo
    }]));

    // Trigger callbacks if status changed
    if (oldStatus !== newStatus) {
      this.statusChangeCallbacks.forEach(callback => {
        callback(tool.id, oldStatus, newStatus);
      });

      // Trigger alerts for degraded or unavailable status
      if (newStatus === 'degraded' || newStatus === 'unavailable') {
        const message = this.generateAlertMessage(tool, statusInfo);
        this.alertCallbacks.forEach(callback => {
          callback(tool.id, newStatus, message);
        });
      }
    }

    // Send WebSocket update
    if (this.webSocketConnection?.readyState === WebSocket.OPEN) {
      this.webSocketConnection.send(JSON.stringify({
        type: 'status-update',
        toolId: tool.id,
        status: statusInfo
      }));
    }
  }

  // Add status to history
  private addToHistory(toolId: string, statusInfo: ToolStatusInfo): void {
    const history = this.statusHistory.get(toolId) || [];
    history.push({
      timestamp: statusInfo.lastChecked,
      status: statusInfo.health,
      responseTime: statusInfo.responseTime,
      errorRate: statusInfo.errorRate,
      available: statusInfo.available,
      details: statusInfo.details
    });

    // Limit history size
    const maxEntries = (this.config.historyRetentionHours * 60 * 60) / (this.config.interval / 1000);
    if (history.length > maxEntries) {
      history.splice(0, history.length - maxEntries);
    }

    this.statusHistory.set(toolId, history);
  }

  // Get status history for a tool
  getStatusHistory(toolId: string, periodHours?: number): StatusHistory[] {
    const history = this.statusHistory.get(toolId) || [];
    if (!periodHours) return history;

    const cutoff = new Date(Date.now() - periodHours * 60 * 60 * 1000);
    return history.filter(entry => entry.timestamp >= cutoff);
  }

  // Set alert thresholds
  setAlertThresholds(thresholds: Partial<AlertThresholds>): void {
    this.thresholds = { ...this.thresholds, ...thresholds };
  }

  // Set monitoring configuration
  setConfig(config: Partial<MonitoringConfig>): void {
    this.config = { ...this.config, ...config };
  }

  // Register status change callback
  onStatusChange(callback: StatusChangeCallback): () => void {
    this.statusChangeCallbacks.push(callback);
    return () => {
      const index = this.statusChangeCallbacks.indexOf(callback);
      if (index >= 0) {
        this.statusChangeCallbacks.splice(index, 1);
      }
    };
  }

  // Register alert callback
  onAlert(callback: AlertCallback): () => void {
    this.alertCallbacks.push(callback);
    return () => {
      const index = this.alertCallbacks.indexOf(callback);
      if (index >= 0) {
        this.alertCallbacks.splice(index, 1);
      }
    };
  }

  // Generate alert message
  private generateAlertMessage(tool: MCPTool, statusInfo: ToolStatusInfo): string {
    const parts: string[] = [`Tool '${tool.name}' is ${statusInfo.health}`];

    if (statusInfo.responseTime > this.thresholds.responseTime.degraded) {
      parts.push(`High response time: ${statusInfo.responseTime}ms`);
    }

    if (statusInfo.errorRate > this.thresholds.errorRate.degraded) {
      parts.push(`High error rate: ${(statusInfo.errorRate * 100).toFixed(1)}%`);
    }

    if (!statusInfo.available) {
      parts.push('Tool is unavailable');
    }

    if (statusInfo.message) {
      parts.push(statusInfo.message);
    }

    return parts.join('. ');
  }

  // Handle WebSocket status update
  private handleWebSocketStatusUpdate(toolId: string, status: ToolStatusInfo): void {
    const state = store.getState();
    const tool = state.tools.tools.find(t => t.id === toolId);
    if (tool) {
      this.updateToolStatus(tool, status);
    }
  }

  // Clean up old history entries
  private startHistoryCleanup(): void {
    setInterval(() => {
      const cutoff = new Date(Date.now() - this.config.historyRetentionHours * 60 * 60 * 1000);
      
      this.statusHistory.forEach((history, toolId) => {
        const filtered = history.filter(entry => entry.timestamp >= cutoff);
        if (filtered.length !== history.length) {
          this.statusHistory.set(toolId, filtered);
        }
      });
    }, 60 * 60 * 1000); // Run every hour
  }

  // Utility function for delays
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Get monitoring statistics
  getMonitoringStats(): {
    activeMonitors: number;
    totalChecks: number;
    averageResponseTime: number;
    healthDistribution: Record<ToolStatus, number>;
  } {
    let totalChecks = 0;
    let totalResponseTime = 0;
    const healthDistribution: Record<ToolStatus, number> = {
      healthy: 0,
      degraded: 0,
      unavailable: 0,
      unknown: 0
    };

    this.statusHistory.forEach(history => {
      history.forEach(entry => {
        totalChecks++;
        totalResponseTime += entry.responseTime;
        healthDistribution[entry.status]++;
      });
    });

    return {
      activeMonitors: this.monitoring.size,
      totalChecks,
      averageResponseTime: totalChecks > 0 ? totalResponseTime / totalChecks : 0,
      healthDistribution
    };
  }
}

export default ToolStatusMonitor.getInstance();