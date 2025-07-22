/**
 * Phase 5 Alert System
 * 
 * Alert management with thresholds, notifications, and alert history.
 * Provides configurable alerting for LLMKG-specific components including
 * cognitive pattern anomalies, brain component degradation, and system health issues.
 */

import {
  SystemAlert,
  AlertThresholds,
  ComponentHealth,
  PerformanceMetrics,
  CognitivePatternActivation,
  BrainComponentHealth,
  MemorySystemMetrics,
  FederationNodeHealth,
  MCPToolHealth,
  LLMKGComponentType,
  ComponentStatus
} from '../types/MonitoringTypes';

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  componentType?: LLMKGComponentType;
  condition: AlertCondition;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  enabled: boolean;
  cooldownPeriod: number; // milliseconds
  actions: AlertAction[];
}

export interface AlertCondition {
  type: 'threshold' | 'pattern' | 'composite' | 'anomaly';
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'ne' | 'gte' | 'lte';
  value: number;
  duration?: number; // milliseconds - condition must persist for this duration
  aggregation?: 'avg' | 'max' | 'min' | 'sum' | 'count';
  timeWindow?: number; // milliseconds - time window for aggregation
}

export interface AlertAction {
  type: 'notification' | 'webhook' | 'email' | 'log' | 'auto_remediation';
  config: Record<string, any>;
}

export interface AlertHistory {
  alert: SystemAlert;
  createdAt: number;
  resolvedAt?: number;
  acknowledgedAt?: number;
  acknowledgedBy?: string;
  resolution?: string;
  actions: AlertActionExecution[];
}

export interface AlertActionExecution {
  action: AlertAction;
  executedAt: number;
  success: boolean;
  error?: string;
  result?: any;
}

export interface NotificationChannel {
  id: string;
  type: 'browser' | 'websocket' | 'webhook' | 'email';
  config: Record<string, any>;
  enabled: boolean;
}

export class AlertSystem {
  private rules: Map<string, AlertRule> = new Map();
  private activeAlerts: Map<string, AlertHistory> = new Map();
  private alertHistory: AlertHistory[] = [];
  private thresholds: AlertThresholds;
  private notificationChannels: Map<string, NotificationChannel> = new Map();
  private ruleEvaluationCache: Map<string, any> = new Map();
  private cooldownTimers: Map<string, number> = new Map();

  // Event callbacks
  private onAlertCallbacks: Set<(alert: SystemAlert) => void> = new Set();
  private onResolveCallbacks: Set<(alert: SystemAlert) => void> = new Set();
  private onAcknowledgeCallbacks: Set<(alert: SystemAlert) => void> = new Set();

  constructor(thresholds: AlertThresholds) {
    this.thresholds = thresholds;
    this.initializeDefaultRules();
    this.initializeDefaultNotificationChannels();
  }

  // Rule Management
  public addRule(rule: AlertRule): void {
    this.rules.set(rule.id, rule);
    console.log(`Alert rule added: ${rule.name}`);
  }

  public removeRule(ruleId: string): void {
    this.rules.delete(ruleId);
    console.log(`Alert rule removed: ${ruleId}`);
  }

  public updateRule(ruleId: string, updates: Partial<AlertRule>): void {
    const rule = this.rules.get(ruleId);
    if (rule) {
      this.rules.set(ruleId, { ...rule, ...updates });
      console.log(`Alert rule updated: ${ruleId}`);
    }
  }

  public enableRule(ruleId: string): void {
    this.updateRule(ruleId, { enabled: true });
  }

  public disableRule(ruleId: string): void {
    this.updateRule(ruleId, { enabled: false });
  }

  public getAllRules(): AlertRule[] {
    return Array.from(this.rules.values());
  }

  // Alert Evaluation
  public evaluateComponentHealth(health: ComponentHealth): SystemAlert[] {
    const alerts: SystemAlert[] = [];
    const applicableRules = Array.from(this.rules.values()).filter(
      rule => rule.enabled && this.ruleAppliesToComponent(rule, health.componentId)
    );

    for (const rule of applicableRules) {
      if (this.isInCooldown(rule.id, health.componentId)) {
        continue;
      }

      const violation = this.evaluateRule(rule, {
        componentId: health.componentId,
        health,
        timestamp: Date.now()
      });

      if (violation) {
        const alert = this.createAlert(rule, health.componentId, violation);
        alerts.push(alert);
        this.setCooldown(rule.id, health.componentId, rule.cooldownPeriod);
      }
    }

    return alerts;
  }

  public evaluatePerformanceMetrics(metrics: PerformanceMetrics): SystemAlert[] {
    const alerts: SystemAlert[] = [];
    const applicableRules = Array.from(this.rules.values()).filter(
      rule => rule.enabled && this.ruleAppliesToComponent(rule, metrics.componentId)
    );

    for (const rule of applicableRules) {
      if (this.isInCooldown(rule.id, metrics.componentId)) {
        continue;
      }

      const violation = this.evaluateRule(rule, {
        componentId: metrics.componentId,
        metrics,
        timestamp: Date.now()
      });

      if (violation) {
        const alert = this.createAlert(rule, metrics.componentId, violation);
        alerts.push(alert);
        this.setCooldown(rule.id, metrics.componentId, rule.cooldownPeriod);
      }
    }

    return alerts;
  }

  public evaluateCognitivePatterns(patterns: CognitivePatternActivation[]): SystemAlert[] {
    const alerts: SystemAlert[] = [];
    const cognitiveRules = Array.from(this.rules.values()).filter(
      rule => rule.enabled && rule.condition.type === 'pattern'
    );

    for (const rule of cognitiveRules) {
      const violation = this.evaluateCognitiveRule(rule, patterns);
      if (violation) {
        const alert = this.createAlert(rule, 'cognitive_system', violation);
        alerts.push(alert);
      }
    }

    return alerts;
  }

  public evaluateBrainComponents(brainHealth: BrainComponentHealth[]): SystemAlert[] {
    const alerts: SystemAlert[] = [];
    const brainRules = Array.from(this.rules.values()).filter(
      rule => rule.enabled && (rule.componentType?.includes('neural') || rule.componentType?.includes('brain'))
    );

    for (const component of brainHealth) {
      for (const rule of brainRules) {
        const violation = this.evaluateRule(rule, {
          componentId: component.componentType,
          brainHealth: component,
          timestamp: Date.now()
        });

        if (violation) {
          const alert = this.createAlert(rule, component.componentType, violation);
          alerts.push(alert);
        }
      }
    }

    return alerts;
  }

  public evaluateMemorySystem(memoryMetrics: MemorySystemMetrics): SystemAlert[] {
    const alerts: SystemAlert[] = [];
    const memoryRules = Array.from(this.rules.values()).filter(
      rule => rule.enabled && rule.condition.metric.includes('memory')
    );

    for (const rule of memoryRules) {
      const violation = this.evaluateRule(rule, {
        componentId: 'memory_system',
        memoryMetrics,
        timestamp: Date.now()
      });

      if (violation) {
        const alert = this.createAlert(rule, 'memory_system', violation);
        alerts.push(alert);
      }
    }

    return alerts;
  }

  // Alert Management
  public triggerAlert(alert: SystemAlert): void {
    const alertHistory: AlertHistory = {
      alert,
      createdAt: Date.now(),
      actions: []
    };

    this.activeAlerts.set(alert.id, alertHistory);
    this.alertHistory.push(alertHistory);

    // Execute alert actions
    this.executeAlertActions(alert);

    // Notify callbacks
    this.onAlertCallbacks.forEach(callback => callback(alert));

    console.log(`Alert triggered: [${alert.severity}] ${alert.title} - ${alert.componentId}`);
  }

  public acknowledgeAlert(alertId: string, acknowledgedBy?: string): void {
    const alertHistory = this.activeAlerts.get(alertId);
    if (alertHistory) {
      alertHistory.acknowledgedAt = Date.now();
      alertHistory.acknowledgedBy = acknowledgedBy;
      alertHistory.alert.acknowledged = true;

      this.onAcknowledgeCallbacks.forEach(callback => callback(alertHistory.alert));
      console.log(`Alert acknowledged: ${alertId}`);
    }
  }

  public resolveAlert(alertId: string, resolution?: string): void {
    const alertHistory = this.activeAlerts.get(alertId);
    if (alertHistory) {
      alertHistory.resolvedAt = Date.now();
      alertHistory.resolution = resolution;
      alertHistory.alert.resolvedAt = Date.now();

      this.activeAlerts.delete(alertId);

      this.onResolveCallbacks.forEach(callback => callback(alertHistory.alert));
      console.log(`Alert resolved: ${alertId} - ${resolution || 'No resolution provided'}`);
    }
  }

  public getActiveAlerts(): SystemAlert[] {
    return Array.from(this.activeAlerts.values()).map(history => history.alert);
  }

  public getAlertHistory(limit: number = 100): AlertHistory[] {
    return this.alertHistory.slice(-limit).reverse();
  }

  public getAlertsByComponent(componentId: string): SystemAlert[] {
    return this.getActiveAlerts().filter(alert => alert.componentId === componentId);
  }

  public getAlertsBySeverity(severity: SystemAlert['severity']): SystemAlert[] {
    return this.getActiveAlerts().filter(alert => alert.severity === severity);
  }

  // Notification Channels
  public addNotificationChannel(channel: NotificationChannel): void {
    this.notificationChannels.set(channel.id, channel);
    console.log(`Notification channel added: ${channel.id}`);
  }

  public removeNotificationChannel(channelId: string): void {
    this.notificationChannels.delete(channelId);
    console.log(`Notification channel removed: ${channelId}`);
  }

  // Event Listeners
  public onAlert(callback: (alert: SystemAlert) => void): () => void {
    this.onAlertCallbacks.add(callback);
    return () => this.onAlertCallbacks.delete(callback);
  }

  public onResolve(callback: (alert: SystemAlert) => void): () => void {
    this.onResolveCallbacks.add(callback);
    return () => this.onResolveCallbacks.delete(callback);
  }

  public onAcknowledge(callback: (alert: SystemAlert) => void): () => void {
    this.onAcknowledgeCallbacks.add(callback);
    return () => this.onAcknowledgeCallbacks.delete(callback);
  }

  // Private Methods
  private initializeDefaultRules(): void {
    // CPU Usage Rules
    this.addRule({
      id: 'cpu_high_usage_warning',
      name: 'High CPU Usage Warning',
      description: 'CPU usage exceeds warning threshold',
      condition: {
        type: 'threshold',
        metric: 'cpu',
        operator: 'gt',
        value: this.thresholds.cpu.warning,
        duration: 60000 // 1 minute
      },
      severity: 'warning',
      enabled: true,
      cooldownPeriod: 300000, // 5 minutes
      actions: [{ type: 'notification', config: {} }]
    });

    this.addRule({
      id: 'cpu_critical_usage',
      name: 'Critical CPU Usage',
      description: 'CPU usage exceeds critical threshold',
      condition: {
        type: 'threshold',
        metric: 'cpu',
        operator: 'gt',
        value: this.thresholds.cpu.critical,
        duration: 30000 // 30 seconds
      },
      severity: 'critical',
      enabled: true,
      cooldownPeriod: 300000,
      actions: [
        { type: 'notification', config: {} },
        { type: 'log', config: { level: 'error' } }
      ]
    });

    // Memory Usage Rules
    this.addRule({
      id: 'memory_high_usage_warning',
      name: 'High Memory Usage Warning',
      description: 'Memory usage exceeds warning threshold',
      condition: {
        type: 'threshold',
        metric: 'memory',
        operator: 'gt',
        value: this.thresholds.memory.warning,
        duration: 60000
      },
      severity: 'warning',
      enabled: true,
      cooldownPeriod: 300000,
      actions: [{ type: 'notification', config: {} }]
    });

    // Latency Rules
    this.addRule({
      id: 'latency_high_warning',
      name: 'High Latency Warning',
      description: 'Response latency exceeds acceptable limits',
      condition: {
        type: 'threshold',
        metric: 'latency',
        operator: 'gt',
        value: this.thresholds.latency.warning,
        duration: 120000 // 2 minutes
      },
      severity: 'warning',
      enabled: true,
      cooldownPeriod: 300000,
      actions: [{ type: 'notification', config: {} }]
    });

    // Cognitive Pattern Rules
    this.addRule({
      id: 'cognitive_pattern_prolonged_activation',
      name: 'Prolonged Cognitive Pattern Activation',
      description: 'Cognitive pattern activated for longer than expected',
      condition: {
        type: 'pattern',
        metric: 'activation_duration',
        operator: 'gt',
        value: this.thresholds.cognitivePatterns.maxActivationTime
      },
      severity: 'warning',
      enabled: true,
      cooldownPeriod: 600000, // 10 minutes
      actions: [{ type: 'notification', config: {} }]
    });

    // Memory System Rules
    this.addRule({
      id: 'memory_system_fragmentation_high',
      name: 'High Memory Fragmentation',
      description: 'Memory system fragmentation exceeds threshold',
      condition: {
        type: 'threshold',
        metric: 'fragmentation_level',
        operator: 'gt',
        value: this.thresholds.memorySystem.maxFragmentation
      },
      severity: 'warning',
      enabled: true,
      cooldownPeriod: 900000, // 15 minutes
      actions: [
        { type: 'notification', config: {} },
        { type: 'auto_remediation', config: { action: 'defragment_memory' } }
      ]
    });

    // Federation Rules
    this.addRule({
      id: 'federation_low_trust_score',
      name: 'Low Federation Trust Score',
      description: 'Federation node trust score below minimum threshold',
      condition: {
        type: 'threshold',
        metric: 'trust_score',
        operator: 'lt',
        value: this.thresholds.federation.minTrustScore
      },
      severity: 'critical',
      enabled: true,
      cooldownPeriod: 1800000, // 30 minutes
      actions: [
        { type: 'notification', config: {} },
        { type: 'log', config: { level: 'warn' } }
      ]
    });
  }

  private initializeDefaultNotificationChannels(): void {
    // Browser notification channel
    this.addNotificationChannel({
      id: 'browser',
      type: 'browser',
      config: {
        requestPermission: true,
        icon: '/favicon.ico',
        timeout: 10000
      },
      enabled: true
    });

    // WebSocket notification channel
    this.addNotificationChannel({
      id: 'websocket',
      type: 'websocket',
      config: {
        endpoint: 'ws://localhost:8080/notifications'
      },
      enabled: true
    });

    // Console logging channel
    this.addNotificationChannel({
      id: 'console',
      type: 'webhook',
      config: {
        url: 'console://log',
        method: 'POST'
      },
      enabled: true
    });
  }

  private ruleAppliesToComponent(rule: AlertRule, componentId: string): boolean {
    if (!rule.componentType) {
      return true; // Generic rule applies to all components
    }

    // Check if component ID matches the rule's component type
    const componentType = this.getComponentTypeFromId(componentId);
    return componentType === rule.componentType;
  }

  private getComponentTypeFromId(componentId: string): LLMKGComponentType {
    // Infer component type from ID (similar to RealTimeMonitor)
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
    
    return 'activation_engine';
  }

  private evaluateRule(rule: AlertRule, context: any): string | null {
    const { condition } = rule;
    
    let value: number;
    switch (condition.metric) {
      case 'cpu':
        value = context.health?.cpuUsage || context.metrics?.cpu || 0;
        break;
      case 'memory':
        value = context.health?.memoryUsage || context.metrics?.memory || 0;
        break;
      case 'latency':
        value = context.health?.responseTime || context.metrics?.latency || 0;
        break;
      case 'throughput':
        value = context.metrics?.throughput || 0;
        break;
      case 'health_score':
        value = context.health?.healthScore || 0;
        break;
      case 'neural_activity':
        value = context.brainHealth?.neuralActivityLevel || 0;
        break;
      case 'fragmentation_level':
        value = context.memoryMetrics?.fragmentationLevel || 0;
        break;
      case 'trust_score':
        value = context.federationHealth?.trustScore || 0;
        break;
      default:
        return null; // Unknown metric
    }

    const violation = this.checkCondition(condition, value);
    if (violation) {
      return `${condition.metric} ${condition.operator} ${condition.value} (actual: ${value.toFixed(2)})`;
    }

    return null;
  }

  private evaluateCognitiveRule(rule: AlertRule, patterns: CognitivePatternActivation[]): string | null {
    const { condition } = rule;
    
    if (condition.metric === 'activation_duration') {
      const prolongedPatterns = patterns.filter(p => p.duration > condition.value);
      if (prolongedPatterns.length > 0) {
        const pattern = prolongedPatterns[0];
        return `Pattern ${pattern.patternType} active for ${pattern.duration}ms (threshold: ${condition.value}ms)`;
      }
    }

    if (condition.metric === 'activation_level') {
      const highActivations = patterns.filter(p => p.activationLevel > condition.value);
      if (highActivations.length > 0) {
        const pattern = highActivations[0];
        return `Pattern ${pattern.patternType} activation level ${(pattern.activationLevel * 100).toFixed(1)}% (threshold: ${(condition.value * 100).toFixed(1)}%)`;
      }
    }

    return null;
  }

  private checkCondition(condition: AlertCondition, value: number): boolean {
    switch (condition.operator) {
      case 'gt': return value > condition.value;
      case 'gte': return value >= condition.value;
      case 'lt': return value < condition.value;
      case 'lte': return value <= condition.value;
      case 'eq': return value === condition.value;
      case 'ne': return value !== condition.value;
      default: return false;
    }
  }

  private createAlert(rule: AlertRule, componentId: string, violation: string): SystemAlert {
    return {
      id: `${rule.id}_${componentId}_${Date.now()}`,
      severity: rule.severity,
      title: rule.name,
      message: `${rule.description}: ${violation}`,
      componentId,
      componentType: this.getComponentTypeFromId(componentId),
      timestamp: Date.now(),
      acknowledged: false,
      metadata: {
        ruleId: rule.id,
        violation
      }
    };
  }

  private executeAlertActions(alert: SystemAlert): void {
    const rule = this.rules.get(alert.metadata.ruleId);
    if (!rule) return;

    for (const action of rule.actions) {
      setTimeout(() => this.executeAction(alert, action), 0);
    }
  }

  private async executeAction(alert: SystemAlert, action: AlertAction): Promise<void> {
    const execution: AlertActionExecution = {
      action,
      executedAt: Date.now(),
      success: false
    };

    try {
      switch (action.type) {
        case 'notification':
          await this.sendNotification(alert, action.config);
          break;
        case 'webhook':
          await this.sendWebhook(alert, action.config);
          break;
        case 'email':
          await this.sendEmail(alert, action.config);
          break;
        case 'log':
          this.logAlert(alert, action.config);
          break;
        case 'auto_remediation':
          await this.executeRemediation(alert, action.config);
          break;
      }

      execution.success = true;
    } catch (error) {
      execution.error = error instanceof Error ? error.message : 'Unknown error';
      console.error(`Alert action failed: ${action.type}`, error);
    }

    // Record action execution
    const alertHistory = this.activeAlerts.get(alert.id);
    if (alertHistory) {
      alertHistory.actions.push(execution);
    }
  }

  private async sendNotification(alert: SystemAlert, config: any): Promise<void> {
    // Send browser notification if permission granted
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(`LLMKG Alert: ${alert.title}`, {
        body: alert.message,
        icon: config.icon || '/favicon.ico',
        tag: alert.id
      });
    }

    // Send via enabled notification channels
    for (const channel of this.notificationChannels.values()) {
      if (channel.enabled && channel.type === 'websocket') {
        // Send via WebSocket if available
        // This would integrate with the WebSocket connection from RealTimeMonitor
        console.log(`Sending alert notification via ${channel.id}:`, alert);
      }
    }
  }

  private async sendWebhook(alert: SystemAlert, config: any): Promise<void> {
    const payload = {
      alert,
      timestamp: Date.now(),
      source: 'llmkg-monitoring'
    };

    await fetch(config.url, {
      method: config.method || 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...config.headers
      },
      body: JSON.stringify(payload)
    });
  }

  private async sendEmail(alert: SystemAlert, config: any): Promise<void> {
    // Email implementation would depend on email service integration
    console.log(`Email alert would be sent to ${config.recipients}:`, alert);
  }

  private logAlert(alert: SystemAlert, config: any): void {
    const level = config.level || 'info';
    const message = `[${alert.severity.toUpperCase()}] ${alert.title} - ${alert.componentId}: ${alert.message}`;
    
    switch (level) {
      case 'error':
        console.error(message);
        break;
      case 'warn':
        console.warn(message);
        break;
      case 'info':
      default:
        console.log(message);
        break;
    }
  }

  private async executeRemediation(alert: SystemAlert, config: any): Promise<void> {
    console.log(`Auto-remediation would be executed for ${alert.componentId}: ${config.action}`);
    // Auto-remediation implementation would depend on the specific actions available
    // This could include restarting services, clearing caches, rebalancing loads, etc.
  }

  private isInCooldown(ruleId: string, componentId: string): boolean {
    const key = `${ruleId}_${componentId}`;
    const cooldownEnd = this.cooldownTimers.get(key);
    return cooldownEnd ? Date.now() < cooldownEnd : false;
  }

  private setCooldown(ruleId: string, componentId: string, duration: number): void {
    const key = `${ruleId}_${componentId}`;
    this.cooldownTimers.set(key, Date.now() + duration);
    
    // Clean up expired cooldowns
    setTimeout(() => {
      this.cooldownTimers.delete(key);
    }, duration);
  }

  // Cleanup methods
  public cleanup(): void {
    this.activeAlerts.clear();
    this.cooldownTimers.clear();
    this.ruleEvaluationCache.clear();
    this.onAlertCallbacks.clear();
    this.onResolveCallbacks.clear();
    this.onAcknowledgeCallbacks.clear();
  }
}

export default AlertSystem;