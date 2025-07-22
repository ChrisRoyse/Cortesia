/**
 * SystemHealthEngine - Core health monitoring engine with multi-dimensional health scoring
 * Provides comprehensive health assessment for LLMKG system components
 */

export interface HealthDimension {
  name: string;
  weight: number;
  score: number;
  trend: number;
  status: HealthStatus;
  lastUpdated: Date;
  metadata?: Record<string, any>;
}

export interface ComponentHealth {
  componentId: string;
  componentName: string;
  overallHealth: number;
  dimensions: HealthDimension[];
  dependencies: string[];
  criticalPath: boolean;
  lastHealthCheck: Date;
  healthHistory: HealthSnapshot[];
}

export interface HealthSnapshot {
  timestamp: Date;
  overallHealth: number;
  dimensions: Record<string, number>;
  events?: HealthEvent[];
}

export interface HealthEvent {
  eventId: string;
  timestamp: Date;
  severity: 'critical' | 'warning' | 'info';
  component: string;
  dimension: string;
  message: string;
  impact: number;
  resolved?: boolean;
  resolvedAt?: Date;
}

export interface PredictiveHealthAnalysis {
  componentId: string;
  predictedHealth: number;
  timeframe: number; // hours ahead
  confidence: number;
  riskFactors: string[];
  recommendedActions: string[];
}

export interface HealthThresholds {
  excellent: number; // >= 0.9
  good: number; // >= 0.7
  warning: number; // >= 0.5
  critical: number; // >= 0.3
  // < 0.3 is failure
}

export enum HealthStatus {
  EXCELLENT = 'excellent',
  GOOD = 'good',
  WARNING = 'warning',
  CRITICAL = 'critical',
  FAILURE = 'failure',
  UNKNOWN = 'unknown'
}

export interface LLMKGHealthIndicators {
  // Cognitive pattern health
  cognitivePatternBalance: number;
  inhibitorySystemHealth: number;
  hierarchicalControlHealth: number;
  
  // Memory system health
  sdrUtilization: number;
  consolidationEfficiency: number;
  memoryFragmentation: number;
  
  // Learning system health
  adaptationRate: number;
  convergenceIndicator: number;
  learningStability: number;
  
  // MCP tool ecosystem health
  toolAvailability: number;
  toolResponseTime: number;
  toolErrorRate: number;
  
  // Federation health
  interGraphSyncHealth: number;
  loadBalanceEfficiency: number;
  federationLatency: number;
}

export class SystemHealthEngine {
  private healthData: Map<string, ComponentHealth> = new Map();
  private healthThresholds: HealthThresholds;
  private healthHistory: HealthSnapshot[] = [];
  private maxHistorySize: number = 1000;
  private predictionModel: HealthPredictionModel;

  constructor(thresholds?: HealthThresholds) {
    this.healthThresholds = thresholds || {
      excellent: 0.9,
      good: 0.7,
      warning: 0.5,
      critical: 0.3
    };
    this.predictionModel = new HealthPredictionModel();
  }

  /**
   * Register a component for health monitoring
   */
  registerComponent(componentId: string, componentName: string, dependencies: string[] = []): void {
    const componentHealth: ComponentHealth = {
      componentId,
      componentName,
      overallHealth: 1.0,
      dimensions: this.initializeHealthDimensions(componentId),
      dependencies,
      criticalPath: this.isOnCriticalPath(componentId, dependencies),
      lastHealthCheck: new Date(),
      healthHistory: []
    };

    this.healthData.set(componentId, componentHealth);
  }

  /**
   * Update health metrics for a component
   */
  updateComponentHealth(
    componentId: string, 
    dimensionUpdates: Partial<Record<string, number>>,
    metadata?: Record<string, any>
  ): void {
    const component = this.healthData.get(componentId);
    if (!component) {
      throw new Error(`Component ${componentId} not registered`);
    }

    const now = new Date();
    
    // Update dimensions
    for (const [dimensionName, score] of Object.entries(dimensionUpdates)) {
      const dimension = component.dimensions.find(d => d.name === dimensionName);
      if (dimension) {
        const previousScore = dimension.score;
        dimension.score = Math.max(0, Math.min(1, score));
        dimension.trend = dimension.score - previousScore;
        dimension.status = this.calculateHealthStatus(dimension.score);
        dimension.lastUpdated = now;
        dimension.metadata = { ...dimension.metadata, ...metadata };
      }
    }

    // Calculate overall health
    component.overallHealth = this.calculateOverallHealth(component.dimensions);
    component.lastHealthCheck = now;

    // Store health snapshot
    const snapshot: HealthSnapshot = {
      timestamp: now,
      overallHealth: component.overallHealth,
      dimensions: Object.fromEntries(
        component.dimensions.map(d => [d.name, d.score])
      )
    };
    
    component.healthHistory.push(snapshot);
    if (component.healthHistory.length > 100) {
      component.healthHistory.shift();
    }

    // Update global health history
    this.updateGlobalHealthHistory();
  }

  /**
   * Update LLMKG-specific health indicators
   */
  updateLLMKGHealth(componentId: string, indicators: Partial<LLMKGHealthIndicators>): void {
    const dimensionUpdates: Record<string, number> = {};

    if (indicators.cognitivePatternBalance !== undefined) {
      dimensionUpdates['cognitive_balance'] = indicators.cognitivePatternBalance;
    }
    if (indicators.inhibitorySystemHealth !== undefined) {
      dimensionUpdates['inhibitory_health'] = indicators.inhibitorySystemHealth;
    }
    if (indicators.sdrUtilization !== undefined) {
      dimensionUpdates['memory_utilization'] = indicators.sdrUtilization;
    }
    if (indicators.adaptationRate !== undefined) {
      dimensionUpdates['learning_health'] = indicators.adaptationRate;
    }
    if (indicators.toolAvailability !== undefined) {
      dimensionUpdates['tool_ecosystem'] = indicators.toolAvailability;
    }
    if (indicators.interGraphSyncHealth !== undefined) {
      dimensionUpdates['federation_health'] = indicators.interGraphSyncHealth;
    }

    this.updateComponentHealth(componentId, dimensionUpdates, { llmkgIndicators: indicators });
  }

  /**
   * Get health status for a component
   */
  getComponentHealth(componentId: string): ComponentHealth | undefined {
    return this.healthData.get(componentId);
  }

  /**
   * Get system-wide health overview
   */
  getSystemHealth(): {
    overallHealth: number;
    status: HealthStatus;
    componentCount: number;
    healthyComponents: number;
    warningComponents: number;
    criticalComponents: number;
    failedComponents: number;
    criticalPathHealth: number;
  } {
    const components = Array.from(this.healthData.values());
    const overallHealth = this.calculateSystemOverallHealth(components);

    const healthyComponents = components.filter(c => c.overallHealth >= this.healthThresholds.good).length;
    const warningComponents = components.filter(c => 
      c.overallHealth >= this.healthThresholds.warning && c.overallHealth < this.healthThresholds.good
    ).length;
    const criticalComponents = components.filter(c => 
      c.overallHealth >= this.healthThresholds.critical && c.overallHealth < this.healthThresholds.warning
    ).length;
    const failedComponents = components.filter(c => c.overallHealth < this.healthThresholds.critical).length;

    const criticalPathComponents = components.filter(c => c.criticalPath);
    const criticalPathHealth = criticalPathComponents.length > 0 
      ? criticalPathComponents.reduce((sum, c) => sum + c.overallHealth, 0) / criticalPathComponents.length
      : 1.0;

    return {
      overallHealth,
      status: this.calculateHealthStatus(overallHealth),
      componentCount: components.length,
      healthyComponents,
      warningComponents,
      criticalComponents,
      failedComponents,
      criticalPathHealth
    };
  }

  /**
   * Perform predictive health analysis
   */
  async predictHealth(componentId: string, hoursAhead: number = 24): Promise<PredictiveHealthAnalysis> {
    const component = this.healthData.get(componentId);
    if (!component) {
      throw new Error(`Component ${componentId} not found`);
    }

    return this.predictionModel.predict(component, hoursAhead);
  }

  /**
   * Get health dependency impact analysis
   */
  getDependencyImpact(componentId: string): {
    dependsOn: string[];
    dependents: string[];
    impactRadius: number;
    cascadeRisk: number;
  } {
    const component = this.healthData.get(componentId);
    if (!component) {
      throw new Error(`Component ${componentId} not found`);
    }

    const dependents = Array.from(this.healthData.values())
      .filter(c => c.dependencies.includes(componentId))
      .map(c => c.componentId);

    const impactRadius = this.calculateImpactRadius(componentId);
    const cascadeRisk = this.calculateCascadeRisk(componentId);

    return {
      dependsOn: component.dependencies,
      dependents,
      impactRadius,
      cascadeRisk
    };
  }

  /**
   * Get health events within a time range
   */
  getHealthEvents(
    startTime: Date, 
    endTime: Date, 
    componentId?: string,
    severity?: HealthEvent['severity']
  ): HealthEvent[] {
    const events: HealthEvent[] = [];
    
    for (const snapshot of this.healthHistory) {
      if (snapshot.timestamp >= startTime && snapshot.timestamp <= endTime && snapshot.events) {
        let filteredEvents = snapshot.events;
        
        if (componentId) {
          filteredEvents = filteredEvents.filter(e => e.component === componentId);
        }
        
        if (severity) {
          filteredEvents = filteredEvents.filter(e => e.severity === severity);
        }
        
        events.push(...filteredEvents);
      }
    }
    
    return events.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  private initializeHealthDimensions(componentId: string): HealthDimension[] {
    const baseDimensions = [
      { name: 'performance', weight: 0.3, description: 'Response time and throughput' },
      { name: 'availability', weight: 0.25, description: 'Service uptime and accessibility' },
      { name: 'resource_usage', weight: 0.2, description: 'CPU, memory, and storage utilization' },
      { name: 'error_rate', weight: 0.15, description: 'Error frequency and severity' },
      { name: 'dependency_health', weight: 0.1, description: 'Health of dependent services' }
    ];

    // Add LLMKG-specific dimensions based on component type
    if (this.isLLMKGComponent(componentId)) {
      baseDimensions.push(
        { name: 'cognitive_balance', weight: 0.15, description: 'Cognitive pattern activation balance' },
        { name: 'inhibitory_health', weight: 0.1, description: 'Inhibitory system effectiveness' },
        { name: 'memory_utilization', weight: 0.12, description: 'SDR and memory efficiency' },
        { name: 'learning_health', weight: 0.13, description: 'Learning system adaptation' },
        { name: 'tool_ecosystem', weight: 0.1, description: 'MCP tool ecosystem health' },
        { name: 'federation_health', weight: 0.05, description: 'Inter-graph synchronization' }
      );
    }

    return baseDimensions.map(dim => ({
      name: dim.name,
      weight: dim.weight,
      score: 1.0, // Start with perfect health
      trend: 0.0,
      status: HealthStatus.EXCELLENT,
      lastUpdated: new Date(),
      metadata: { description: dim.description }
    }));
  }

  private calculateOverallHealth(dimensions: HealthDimension[]): number {
    const totalWeight = dimensions.reduce((sum, d) => sum + d.weight, 0);
    const weightedSum = dimensions.reduce((sum, d) => sum + (d.score * d.weight), 0);
    return totalWeight > 0 ? weightedSum / totalWeight : 0;
  }

  private calculateSystemOverallHealth(components: ComponentHealth[]): number {
    if (components.length === 0) return 1.0;

    // Give higher weight to critical path components
    const totalWeight = components.reduce((sum, c) => sum + (c.criticalPath ? 2.0 : 1.0), 0);
    const weightedSum = components.reduce((sum, c) => 
      sum + (c.overallHealth * (c.criticalPath ? 2.0 : 1.0)), 0
    );

    return weightedSum / totalWeight;
  }

  private calculateHealthStatus(score: number): HealthStatus {
    if (score >= this.healthThresholds.excellent) return HealthStatus.EXCELLENT;
    if (score >= this.healthThresholds.good) return HealthStatus.GOOD;
    if (score >= this.healthThresholds.warning) return HealthStatus.WARNING;
    if (score >= this.healthThresholds.critical) return HealthStatus.CRITICAL;
    return HealthStatus.FAILURE;
  }

  private isOnCriticalPath(componentId: string, dependencies: string[]): boolean {
    // Define critical components for LLMKG
    const criticalComponents = [
      'knowledge_engine',
      'activation_engine',
      'cognitive_controller',
      'memory_manager',
      'learning_engine'
    ];
    return criticalComponents.includes(componentId);
  }

  private isLLMKGComponent(componentId: string): boolean {
    const llmkgComponents = [
      'knowledge_engine',
      'activation_engine',
      'cognitive_controller',
      'memory_manager',
      'learning_engine',
      'inhibitory_system',
      'federation_manager'
    ];
    return llmkgComponents.includes(componentId);
  }

  private calculateImpactRadius(componentId: string): number {
    // BFS to calculate impact radius
    const visited = new Set<string>();
    let radius = 0;
    let currentLevel = [componentId];

    while (currentLevel.length > 0) {
      const nextLevel: string[] = [];
      
      for (const id of currentLevel) {
        if (visited.has(id)) continue;
        visited.add(id);
        
        // Find all dependents
        const dependents = Array.from(this.healthData.values())
          .filter(c => c.dependencies.includes(id))
          .map(c => c.componentId);
        
        nextLevel.push(...dependents);
      }
      
      if (nextLevel.length > 0) {
        radius++;
        currentLevel = nextLevel;
      } else {
        break;
      }
    }

    return radius;
  }

  private calculateCascadeRisk(componentId: string): number {
    const component = this.healthData.get(componentId);
    if (!component) return 0;

    const dependents = Array.from(this.healthData.values())
      .filter(c => c.dependencies.includes(componentId));

    if (dependents.length === 0) return 0;

    // Risk is based on component health, number of dependents, and their criticality
    const healthRisk = 1 - component.overallHealth;
    const dependentRisk = dependents.length * 0.1;
    const criticalityRisk = component.criticalPath ? 0.3 : 0.1;

    return Math.min(1.0, healthRisk + dependentRisk + criticalityRisk);
  }

  private updateGlobalHealthHistory(): void {
    const systemHealth = this.getSystemHealth();
    const snapshot: HealthSnapshot = {
      timestamp: new Date(),
      overallHealth: systemHealth.overallHealth,
      dimensions: {
        'system_health': systemHealth.overallHealth,
        'critical_path_health': systemHealth.criticalPathHealth,
        'component_availability': systemHealth.healthyComponents / systemHealth.componentCount
      }
    };

    this.healthHistory.push(snapshot);
    if (this.healthHistory.length > this.maxHistorySize) {
      this.healthHistory.shift();
    }
  }
}

/**
 * Health prediction model using trend analysis and machine learning techniques
 */
class HealthPredictionModel {
  async predict(component: ComponentHealth, hoursAhead: number): Promise<PredictiveHealthAnalysis> {
    const history = component.healthHistory.slice(-24); // Last 24 data points
    
    if (history.length < 3) {
      // Not enough data for prediction
      return {
        componentId: component.componentId,
        predictedHealth: component.overallHealth,
        timeframe: hoursAhead,
        confidence: 0.1,
        riskFactors: ['Insufficient historical data'],
        recommendedActions: ['Collect more health data']
      };
    }

    // Simple linear regression for trend analysis
    const trend = this.calculateTrend(history);
    const volatility = this.calculateVolatility(history);
    const cyclicalPattern = this.detectCyclicalPattern(history);
    
    // Predict future health
    const currentHealth = component.overallHealth;
    const trendImpact = trend * hoursAhead * 0.1; // Trend impact over time
    const cyclicalImpact = cyclicalPattern * 0.05; // Cyclical adjustment
    
    let predictedHealth = currentHealth + trendImpact + cyclicalImpact;
    predictedHealth = Math.max(0, Math.min(1, predictedHealth));

    // Calculate confidence based on trend stability and data quality
    const confidence = Math.max(0.1, Math.min(0.9, 
      (1 - volatility) * (history.length / 24) * 0.8
    ));

    // Identify risk factors
    const riskFactors = this.identifyRiskFactors(component, trend, volatility);
    
    // Generate recommendations
    const recommendedActions = this.generateRecommendations(component, predictedHealth, riskFactors);

    return {
      componentId: component.componentId,
      predictedHealth,
      timeframe: hoursAhead,
      confidence,
      riskFactors,
      recommendedActions
    };
  }

  private calculateTrend(history: HealthSnapshot[]): number {
    if (history.length < 2) return 0;
    
    const n = history.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (let i = 0; i < n; i++) {
      const x = i;
      const y = history[i].overallHealth;
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return isNaN(slope) ? 0 : slope;
  }

  private calculateVolatility(history: HealthSnapshot[]): number {
    if (history.length < 2) return 0;
    
    const mean = history.reduce((sum, h) => sum + h.overallHealth, 0) / history.length;
    const variance = history.reduce((sum, h) => sum + Math.pow(h.overallHealth - mean, 2), 0) / history.length;
    
    return Math.sqrt(variance);
  }

  private detectCyclicalPattern(history: HealthSnapshot[]): number {
    // Simple cyclical pattern detection based on time of day
    const now = new Date();
    const hourOfDay = now.getHours();
    
    // Assume lower health during peak hours (9-17)
    if (hourOfDay >= 9 && hourOfDay <= 17) {
      return -0.05; // Slight decrease during business hours
    }
    
    return 0.02; // Slight increase during off-hours
  }

  private identifyRiskFactors(component: ComponentHealth, trend: number, volatility: number): string[] {
    const riskFactors: string[] = [];
    
    if (trend < -0.01) {
      riskFactors.push('Declining health trend detected');
    }
    
    if (volatility > 0.1) {
      riskFactors.push('High health volatility');
    }
    
    if (component.overallHealth < 0.7) {
      riskFactors.push('Current health below good threshold');
    }
    
    // Check dimension-specific risks
    for (const dimension of component.dimensions) {
      if (dimension.score < 0.5) {
        riskFactors.push(`${dimension.name} dimension in critical state`);
      }
      if (dimension.trend < -0.05) {
        riskFactors.push(`${dimension.name} showing rapid decline`);
      }
    }
    
    if (component.criticalPath) {
      riskFactors.push('Component is on critical path');
    }
    
    return riskFactors;
  }

  private generateRecommendations(
    component: ComponentHealth, 
    predictedHealth: number, 
    riskFactors: string[]
  ): string[] {
    const recommendations: string[] = [];
    
    if (predictedHealth < 0.7) {
      recommendations.push('Schedule preventive maintenance');
    }
    
    if (predictedHealth < 0.5) {
      recommendations.push('Immediate intervention required');
      recommendations.push('Activate backup systems if available');
    }
    
    if (riskFactors.some(rf => rf.includes('critical path'))) {
      recommendations.push('Monitor critical path dependencies closely');
    }
    
    if (riskFactors.some(rf => rf.includes('volatility'))) {
      recommendations.push('Investigate sources of health instability');
    }
    
    // Component-specific recommendations
    if (component.componentId.includes('memory')) {
      recommendations.push('Consider memory optimization or cleanup');
    }
    
    if (component.componentId.includes('learning')) {
      recommendations.push('Review learning parameters and convergence criteria');
    }
    
    return recommendations;
  }
}

export { SystemHealthEngine };