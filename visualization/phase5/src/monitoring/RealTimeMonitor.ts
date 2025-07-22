/**
 * Phase 5 Real-Time Monitor
 * 
 * Core monitoring engine collecting real-time metrics from LLMKG systems.
 * Provides sub-100ms update latency for critical metrics with WebSocket integration,
 * automatic reconnection, and LLMKG-specific cognitive pattern monitoring.
 */

import {
  MonitoringConfig,
  ComponentHealth,
  PerformanceMetrics,
  SystemHealthSummary,
  SystemAlert,
  WebSocketMessage,
  HealthUpdateMessage,
  PerformanceMetricsMessage,
  CognitiveActivationMessage,
  AlertMessage,
  SystemStatusMessage,
  CognitivePatternActivation,
  MemorySystemMetrics,
  FederationNodeHealth,
  MCPToolHealth,
  BrainComponentHealth,
  LLMKGComponentType,
  MonitoringSystemInterface,
  MonitoringSubscription
} from '../types/MonitoringTypes';

export class RealTimeMonitor implements MonitoringSystemInterface {
  private config: MonitoringConfig | null = null;
  private websocket: WebSocket | null = null;
  private isMonitoring = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second
  private maxReconnectDelay = 30000; // Max 30 seconds

  // Data storage
  private componentHealthMap = new Map<string, ComponentHealth>();
  private performanceMetricsHistory = new Map<string, PerformanceMetrics[]>();
  private cognitivePatternActivations = new Map<string, CognitivePatternActivation>();
  private memorySystemMetrics: MemorySystemMetrics | null = null;
  private federationHealth = new Map<string, FederationNodeHealth>();
  private mcpToolHealth = new Map<string, MCPToolHealth>();
  private brainComponentHealth = new Map<string, BrainComponentHealth>();
  private activeAlerts = new Map<string, SystemAlert>();

  // Event subscriptions
  private subscriptions = new Map<string, MonitoringSubscription>();
  private eventListeners = new Map<string, Set<(data: any) => void>>();

  // Performance optimization
  private updateThrottles = new Map<string, number>();
  private batchedUpdates: any[] = [];
  private batchTimeout: NodeJS.Timeout | null = null;

  /**
   * Initialize and start real-time monitoring system
   */
  async startMonitoring(config: MonitoringConfig): Promise<void> {
    this.config = config;
    
    console.log('Starting LLMKG Real-Time Monitoring System...');
    
    try {
      await this.initializeWebSocketConnection();
      await this.initializeDataCollectors();
      this.startPerformanceOptimization();
      
      this.isMonitoring = true;
      console.log('Real-Time Monitor started successfully');
      
      // Initial system health check
      await this.performInitialHealthCheck();
      
    } catch (error) {
      console.error('Failed to start monitoring system:', error);
      throw new Error(`Monitoring initialization failed: ${error}`);
    }
  }

  /**
   * Stop monitoring system and cleanup resources
   */
  async stopMonitoring(): Promise<void> {
    console.log('Stopping Real-Time Monitoring System...');
    
    this.isMonitoring = false;
    
    // Cleanup WebSocket connection
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    // Clear timers
    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
      this.batchTimeout = null;
    }
    
    // Clear data structures
    this.clearAllData();
    
    console.log('Real-Time Monitor stopped');
  }

  /**
   * Subscribe to component monitoring updates
   */
  subscribeToComponent(subscription: MonitoringSubscription): () => void {
    const subscriptionId = `${subscription.componentId}_${Date.now()}`;
    this.subscriptions.set(subscriptionId, subscription);
    
    console.log(`Subscribed to component: ${subscription.componentId}`);
    
    // Return unsubscribe function
    return () => {
      this.subscriptions.delete(subscriptionId);
      console.log(`Unsubscribed from component: ${subscription.componentId}`);
    };
  }

  /**
   * Get current system health summary
   */
  async getSystemHealth(): Promise<SystemHealthSummary> {
    const components = Array.from(this.componentHealthMap.values());
    
    if (components.length === 0) {
      return {
        overall: 'offline',
        healthScore: 0,
        totalComponents: 0,
        activeComponents: 0,
        degradedComponents: 0,
        offlineComponents: 0,
        activeAlerts: this.activeAlerts.size,
        cognitivePatternActivity: 0,
        memorySystemHealth: 0,
        federationHealth: 0,
        lastUpdated: Date.now()
      };
    }

    const healthScores = components.map(c => c.healthScore);
    const avgHealthScore = healthScores.reduce((a, b) => a + b, 0) / healthScores.length;
    
    const activeComponents = components.filter(c => c.status === 'active').length;
    const degradedComponents = components.filter(c => 
      c.status === 'degraded' || c.status === 'error'
    ).length;
    const offlineComponents = components.filter(c => c.status === 'offline').length;
    
    // Calculate cognitive pattern activity level
    const cognitiveActivations = Array.from(this.cognitivePatternActivations.values());
    const cognitivePatternActivity = cognitiveActivations.length > 0 
      ? cognitiveActivations.reduce((sum, activation) => sum + activation.activationLevel, 0) / cognitiveActivations.length
      : 0;
    
    // Calculate memory system health
    const memorySystemHealth = this.memorySystemMetrics 
      ? (this.memorySystemMetrics.hitRate + (1 - this.memorySystemMetrics.fragmentationLevel)) / 2
      : 0;
    
    // Calculate federation health
    const federationNodes = Array.from(this.federationHealth.values());
    const federationHealthScore = federationNodes.length > 0
      ? federationNodes.reduce((sum, node) => sum + node.connectionQuality * node.trustScore, 0) / federationNodes.length
      : 0;
    
    // Determine overall status
    let overall: ComponentStatus = 'active';
    if (avgHealthScore < 30) overall = 'offline';
    else if (avgHealthScore < 60 || degradedComponents > totalComponents * 0.3) overall = 'degraded';
    else if (avgHealthScore < 80 || degradedComponents > 0) overall = 'processing';
    
    return {
      overall,
      healthScore: Math.round(avgHealthScore),
      totalComponents: components.length,
      activeComponents,
      degradedComponents,
      offlineComponents,
      activeAlerts: this.activeAlerts.size,
      cognitivePatternActivity: Math.round(cognitivePatternActivity * 100),
      memorySystemHealth: Math.round(memorySystemHealth * 100),
      federationHealth: Math.round(federationHealthScore * 100),
      lastUpdated: Date.now()
    };
  }

  /**
   * Get specific component health data
   */
  async getComponentHealth(componentId: string): Promise<ComponentHealth> {
    const health = this.componentHealthMap.get(componentId);
    
    if (!health) {
      throw new Error(`Component not found: ${componentId}`);
    }
    
    return { ...health };
  }

  /**
   * Get active cognitive pattern activations
   */
  async getCognitivePatternActivity(): Promise<CognitivePatternActivation[]> {
    return Array.from(this.cognitivePatternActivations.values());
  }

  /**
   * Get memory system metrics
   */
  async getMemorySystemMetrics(): Promise<MemorySystemMetrics> {
    if (!this.memorySystemMetrics) {
      throw new Error('Memory system metrics not available');
    }
    
    return { ...this.memorySystemMetrics };
  }

  /**
   * Get federation node health data
   */
  async getFederationHealth(): Promise<FederationNodeHealth[]> {
    return Array.from(this.federationHealth.values());
  }

  /**
   * Configure alert thresholds
   */
  configureAlerts(thresholds: any): void {
    if (this.config) {
      this.config.alertThresholds = thresholds;
      console.log('Alert thresholds updated');
    }
  }

  /**
   * Acknowledge a system alert
   */
  acknowledgeAlert(alertId: string): void {
    const alert = this.activeAlerts.get(alertId);
    if (alert) {
      alert.acknowledged = true;
      this.notifyEventListeners('alert_acknowledged', alert);
      console.log(`Alert acknowledged: ${alertId}`);
    }
  }

  /**
   * Export health data in specified format
   */
  async exportHealthData(format: 'json' | 'csv' | 'xml'): Promise<Blob> {
    const data = {
      systemHealth: await this.getSystemHealth(),
      componentHealth: Object.fromEntries(this.componentHealthMap),
      cognitivePatterns: Object.fromEntries(this.cognitivePatternActivations),
      memoryMetrics: this.memorySystemMetrics,
      federationHealth: Object.fromEntries(this.federationHealth),
      activeAlerts: Object.fromEntries(this.activeAlerts),
      exportTimestamp: Date.now()
    };

    switch (format) {
      case 'json':
        return new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      
      case 'csv':
        const csv = this.convertToCSV(data);
        return new Blob([csv], { type: 'text/csv' });
      
      case 'xml':
        const xml = this.convertToXML(data);
        return new Blob([xml], { type: 'application/xml' });
      
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }

  // Private methods

  private async initializeWebSocketConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.config) {
        reject(new Error('Configuration not available'));
        return;
      }

      this.websocket = new WebSocket(this.config.websocketEndpoint);
      
      this.websocket.onopen = () => {
        console.log('WebSocket connection established');
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        resolve();
      };

      this.websocket.onmessage = (event) => {
        this.handleWebSocketMessage(event);
      };

      this.websocket.onclose = (event) => {
        console.log('WebSocket connection closed:', event.reason);
        if (this.isMonitoring) {
          this.attemptReconnection();
        }
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (this.reconnectAttempts === 0) {
          reject(error);
        }
      };

      // Timeout for initial connection
      setTimeout(() => {
        if (this.websocket?.readyState !== WebSocket.OPEN) {
          reject(new Error('WebSocket connection timeout'));
        }
      }, 10000);
    });
  }

  private handleWebSocketMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      // Check if update should be throttled
      if (this.shouldThrottleUpdate(message)) {
        return;
      }

      switch (message.type) {
        case 'health_update':
          this.handleHealthUpdate(message as HealthUpdateMessage);
          break;
        case 'performance_metrics':
          this.handlePerformanceMetrics(message as PerformanceMetricsMessage);
          break;
        case 'cognitive_activation':
          this.handleCognitiveActivation(message as CognitiveActivationMessage);
          break;
        case 'alert':
          this.handleAlert(message as AlertMessage);
          break;
        case 'system_status':
          this.handleSystemStatus(message as SystemStatusMessage);
          break;
        default:
          console.warn('Unknown message type:', message.type);
      }
      
      // Add to batched updates for processing
      this.addToBatchedUpdates(message);
      
    } catch (error) {
      console.error('Error processing WebSocket message:', error);
    }
  }

  private handleHealthUpdate(message: HealthUpdateMessage): void {
    const health = message.data;
    this.componentHealthMap.set(health.componentId, health);
    
    // Check for alert conditions
    this.checkHealthAlerts(health);
    
    // Notify subscribers
    this.notifySubscribers(health.componentId, health);
    this.notifyEventListeners('health_update', health);
  }

  private handlePerformanceMetrics(message: PerformanceMetricsMessage): void {
    const metrics = message.data;
    
    // Store in history
    if (!this.performanceMetricsHistory.has(metrics.componentId)) {
      this.performanceMetricsHistory.set(metrics.componentId, []);
    }
    
    const history = this.performanceMetricsHistory.get(metrics.componentId)!;
    history.push(metrics);
    
    // Limit history size
    if (this.config && history.length > this.config.maxHistorySize) {
      history.shift();
    }
    
    // Check for performance alerts
    this.checkPerformanceAlerts(metrics);
    
    // Notify subscribers
    this.notifySubscribers(metrics.componentId, metrics);
    this.notifyEventListeners('performance_metrics', metrics);
  }

  private handleCognitiveActivation(message: CognitiveActivationMessage): void {
    const activation = message.data;
    this.cognitivePatternActivations.set(activation.patternId, activation);
    
    // Check cognitive pattern alerts
    this.checkCognitivePatternAlerts(activation);
    
    this.notifyEventListeners('cognitive_activation', activation);
  }

  private handleAlert(message: AlertMessage): void {
    const alert = message.data;
    this.activeAlerts.set(alert.id, alert);
    
    console.warn(`System Alert [${alert.severity}]: ${alert.title} - ${alert.message}`);
    this.notifyEventListeners('alert', alert);
  }

  private handleSystemStatus(message: SystemStatusMessage): void {
    const status = message.data;
    this.notifyEventListeners('system_status', status);
  }

  private shouldThrottleUpdate(message: WebSocketMessage): boolean {
    if (!this.config) return false;
    
    const key = `${message.type}_${Date.now()}`;
    const lastUpdate = this.updateThrottles.get(message.type) || 0;
    const now = Date.now();
    
    if (now - lastUpdate < this.config.updateInterval) {
      return true;
    }
    
    this.updateThrottles.set(message.type, now);
    return false;
  }

  private checkHealthAlerts(health: ComponentHealth): void {
    if (!this.config) return;
    
    const thresholds = this.config.alertThresholds;
    
    // CPU usage alert
    if (health.cpuUsage > thresholds.cpu.critical) {
      this.createAlert('critical', 'High CPU Usage', 
        `Component ${health.componentId} CPU usage: ${health.cpuUsage.toFixed(1)}%`, 
        health.componentId);
    } else if (health.cpuUsage > thresholds.cpu.warning) {
      this.createAlert('warning', 'Elevated CPU Usage', 
        `Component ${health.componentId} CPU usage: ${health.cpuUsage.toFixed(1)}%`, 
        health.componentId);
    }
    
    // Memory usage alert
    if (health.memoryUsage > thresholds.memory.critical) {
      this.createAlert('critical', 'High Memory Usage', 
        `Component ${health.componentId} memory usage: ${health.memoryUsage.toFixed(1)}%`, 
        health.componentId);
    } else if (health.memoryUsage > thresholds.memory.warning) {
      this.createAlert('warning', 'Elevated Memory Usage', 
        `Component ${health.componentId} memory usage: ${health.memoryUsage.toFixed(1)}%`, 
        health.componentId);
    }
    
    // Health score alert
    if (health.healthScore < thresholds.healthScore.critical) {
      this.createAlert('critical', 'Component Health Critical', 
        `Component ${health.componentId} health score: ${health.healthScore}`, 
        health.componentId);
    } else if (health.healthScore < thresholds.healthScore.warning) {
      this.createAlert('warning', 'Component Health Warning', 
        `Component ${health.componentId} health score: ${health.healthScore}`, 
        health.componentId);
    }
  }

  private createAlert(severity: 'info' | 'warning' | 'critical' | 'emergency', 
                     title: string, message: string, componentId: string): void {
    const alert: SystemAlert = {
      id: `${componentId}_${Date.now()}`,
      severity,
      title,
      message,
      componentId,
      componentType: this.getComponentType(componentId),
      timestamp: Date.now(),
      acknowledged: false,
      metadata: {}
    };
    
    this.activeAlerts.set(alert.id, alert);
    this.notifyEventListeners('alert', alert);
  }

  private getComponentType(componentId: string): LLMKGComponentType {
    // Determine component type from ID
    if (componentId.includes('activation')) return 'activation_engine';
    if (componentId.includes('inhibitory')) return 'inhibitory_circuit';
    if (componentId.includes('cognitive')) return 'cognitive_pattern';
    if (componentId.includes('memory')) return 'working_memory';
    if (componentId.includes('sdr')) return 'sdr_storage';
    if (componentId.includes('knowledge')) return 'knowledge_engine';
    if (componentId.includes('mcp')) return 'mcp_tool';
    if (componentId.includes('federation')) return 'federation_node';
    if (componentId.includes('neural')) return 'neural_bridge';
    if (componentId.includes('attention')) return 'attention_manager';
    if (componentId.includes('pattern')) return 'pattern_detector';
    
    return 'activation_engine'; // Default fallback
  }

  private notifySubscribers(componentId: string, data: any): void {
    for (const subscription of this.subscriptions.values()) {
      if (subscription.componentId === componentId) {
        // Apply throttling
        const now = Date.now();
        const lastCall = this.updateThrottles.get(`sub_${componentId}`) || 0;
        
        if (now - lastCall >= subscription.throttleMs) {
          subscription.callback(data);
          this.updateThrottles.set(`sub_${componentId}`, now);
        }
      }
    }
  }

  private notifyEventListeners(event: string, data: any): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(callback => callback(data));
    }
  }

  private addToBatchedUpdates(message: WebSocketMessage): void {
    this.batchedUpdates.push(message);
    
    if (!this.batchTimeout) {
      this.batchTimeout = setTimeout(() => {
        this.processBatchedUpdates();
        this.batchTimeout = null;
      }, 100); // Process batches every 100ms
    }
  }

  private processBatchedUpdates(): void {
    if (this.batchedUpdates.length > 0) {
      // Process batch updates for optimization
      console.log(`Processed ${this.batchedUpdates.length} batched updates`);
      this.batchedUpdates = [];
    }
  }

  private async attemptReconnection(): Promise<void> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }
    
    this.reconnectAttempts++;
    console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
    
    setTimeout(async () => {
      try {
        await this.initializeWebSocketConnection();
        console.log('Reconnection successful');
      } catch (error) {
        console.error('Reconnection failed:', error);
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
        this.attemptReconnection();
      }
    }, this.reconnectDelay);
  }

  private async initializeDataCollectors(): Promise<void> {
    // Initialize LLMKG-specific data collection
    console.log('Initializing LLMKG data collectors...');
    
    // Set up cognitive pattern monitoring
    if (this.config?.enableCognitivePatterns) {
      this.initializeCognitivePatternMonitoring();
    }
    
    // Set up brain component monitoring
    if (this.config?.enableBrainComponents) {
      this.initializeBrainComponentMonitoring();
    }
    
    // Set up MCP tool monitoring
    if (this.config?.enableMCPToolMonitoring) {
      this.initializeMCPToolMonitoring();
    }
    
    // Set up memory system monitoring
    if (this.config?.enableMemorySystemMonitoring) {
      this.initializeMemorySystemMonitoring();
    }
    
    // Set up federation monitoring
    if (this.config?.enableFederationMonitoring) {
      this.initializeFederationMonitoring();
    }
  }

  private initializeCognitivePatternMonitoring(): void {
    console.log('Cognitive pattern monitoring enabled');
  }

  private initializeBrainComponentMonitoring(): void {
    console.log('Brain component monitoring enabled');
  }

  private initializeMCPToolMonitoring(): void {
    console.log('MCP tool monitoring enabled');
  }

  private initializeMemorySystemMonitoring(): void {
    console.log('Memory system monitoring enabled');
  }

  private initializeFederationMonitoring(): void {
    console.log('Federation monitoring enabled');
  }

  private startPerformanceOptimization(): void {
    // Implement performance optimizations for large-scale monitoring
  }

  private async performInitialHealthCheck(): Promise<void> {
    console.log('Performing initial system health check...');
    const systemHealth = await this.getSystemHealth();
    console.log('System health:', systemHealth);
  }

  private clearAllData(): void {
    this.componentHealthMap.clear();
    this.performanceMetricsHistory.clear();
    this.cognitivePatternActivations.clear();
    this.federationHealth.clear();
    this.mcpToolHealth.clear();
    this.brainComponentHealth.clear();
    this.activeAlerts.clear();
    this.subscriptions.clear();
    this.eventListeners.clear();
    this.updateThrottles.clear();
  }

  // Utility methods for data conversion
  private convertToCSV(data: any): string {
    // Implement CSV conversion logic
    return 'CSV data representation';
  }

  private convertToXML(data: any): string {
    // Implement XML conversion logic
    return '<data>XML representation</data>';
  }

  private checkPerformanceAlerts(metrics: PerformanceMetrics): void {
    if (!this.config) return;
    
    const thresholds = this.config.alertThresholds;
    
    if (metrics.latency > thresholds.latency.critical) {
      this.createAlert('critical', 'High Latency', 
        `Component ${metrics.componentId} latency: ${metrics.latency.toFixed(2)}ms`, 
        metrics.componentId);
    }
  }

  private checkCognitivePatternAlerts(activation: CognitivePatternActivation): void {
    if (!this.config) return;
    
    const thresholds = this.config.alertThresholds.cognitivePatterns;
    
    if (activation.duration > thresholds.maxActivationTime) {
      this.createAlert('warning', 'Prolonged Pattern Activation', 
        `Pattern ${activation.patternId} active for ${activation.duration}ms`, 
        activation.patternId);
    }
  }

  // Event listener management
  addEventListener(event: string, callback: (data: any) => void): () => void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    
    this.eventListeners.get(event)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      this.eventListeners.get(event)?.delete(callback);
    };
  }
}

export default RealTimeMonitor;