/**
 * MCP Request Tracing System
 * 
 * Complete visualization and analytics system for MCP request flow through
 * the LLMKG cognitive architecture.
 */

export { 
  MCPRequestTracer,
  type MCPRequest,
  type MCPResponse,
  type TraceEvent
} from './MCPRequestTracer';

export {
  RequestPathVisualizer,
  type PathNode,
  type PathSegment,
  type VisualizationConfig
} from './RequestPathVisualizer';

export {
  ParticleEffects,
  type Particle,
  type ParticleTrail,
  type ParticleSystemConfig
} from './ParticleEffects';

export {
  TraceAnalytics,
  type PerformanceMetric,
  type PathAnalysis,
  type SystemHealth,
  type Alert,
  type AnalyticsConfig
} from './TraceAnalytics';

/**
 * Main MCP Tracing System Integration Class
 * 
 * Coordinates all tracing components for complete request visualization.
 */
import * as THREE from 'three';
import { MCPRequestTracer } from './MCPRequestTracer';
import { RequestPathVisualizer } from './RequestPathVisualizer';
import { ParticleEffects } from './ParticleEffects';
import { TraceAnalytics } from './TraceAnalytics';

export interface MCPTracingSystemConfig {
  websocketUrl?: string;
  visualization?: any;
  particles?: any;
  analytics?: any;
}

export class MCPTracingSystem {
  private tracer: MCPRequestTracer;
  private pathVisualizer: RequestPathVisualizer;
  private particleEffects: ParticleEffects;
  private analytics: TraceAnalytics;

  constructor(scene: THREE.Scene, config?: MCPTracingSystemConfig) {
    // Initialize components
    this.tracer = new MCPRequestTracer(config?.websocketUrl);
    this.pathVisualizer = new RequestPathVisualizer(scene, config?.visualization);
    this.particleEffects = new ParticleEffects(scene, config?.particles);
    this.analytics = new TraceAnalytics(config?.analytics);

    // Wire up event handlers
    this.setupEventHandlers();
  }

  /**
   * Setup event handlers to coordinate all components
   */
  private setupEventHandlers(): void {
    this.tracer.addEventListener('request', (event) => {
      this.pathVisualizer.processTraceEvent(event);
      this.particleEffects.processTraceEvent(event);
      this.analytics.processTraceEvent(event);
    });

    this.tracer.addEventListener('cognitive_activation', (event) => {
      this.pathVisualizer.processTraceEvent(event);
      this.particleEffects.processTraceEvent(event);
      this.analytics.processTraceEvent(event);
    });

    this.tracer.addEventListener('response', (event) => {
      this.pathVisualizer.processTraceEvent(event);
      this.particleEffects.processTraceEvent(event);
      this.analytics.processTraceEvent(event);
    });

    this.tracer.addEventListener('error', (event) => {
      this.pathVisualizer.processTraceEvent(event);
      this.particleEffects.processTraceEvent(event);
      this.analytics.processTraceEvent(event);
    });

    this.tracer.addEventListener('performance', (event) => {
      this.particleEffects.processTraceEvent(event);
      this.analytics.processTraceEvent(event);
    });
  }

  /**
   * Get the tracer component
   */
  public getTracer(): MCPRequestTracer {
    return this.tracer;
  }

  /**
   * Get the path visualizer component
   */
  public getPathVisualizer(): RequestPathVisualizer {
    return this.pathVisualizer;
  }

  /**
   * Get the particle effects component
   */
  public getParticleEffects(): ParticleEffects {
    return this.particleEffects;
  }

  /**
   * Get the analytics component
   */
  public getAnalytics(): TraceAnalytics {
    return this.analytics;
  }

  /**
   * Check if connected to Phase 1
   */
  public isConnectedToPhase1(): boolean {
    return this.tracer.isConnectedToPhase1();
  }

  /**
   * Simulate a request for testing
   */
  public simulateRequest(request: any): void {
    this.tracer.simulateRequest(request);
  }

  /**
   * Clear all visualization data
   */
  public clearAll(): void {
    this.pathVisualizer.clearAllPaths();
    this.particleEffects.clearAll();
    this.tracer.clear();
  }

  /**
   * Get system status
   */
  public getSystemStatus(): {
    connected: boolean;
    activeRequests: number;
    activePaths: number;
    particles: number;
    systemHealth: number;
    alerts: number;
  } {
    const health = this.analytics.getCurrentHealth();
    
    return {
      connected: this.tracer.isConnectedToPhase1(),
      activeRequests: this.tracer.getRequests().filter(r => r.phase !== 'complete' && r.phase !== 'error').length,
      activePaths: this.pathVisualizer.getActivePaths().length,
      particles: this.particleEffects.getParticleCount(),
      systemHealth: health?.overallHealth || 0,
      alerts: this.analytics.getActiveAlerts().length
    };
  }

  /**
   * Dispose all resources
   */
  public dispose(): void {
    this.tracer.disconnect();
    this.pathVisualizer.dispose();
    this.particleEffects.dispose();
    this.analytics.dispose();
  }
}