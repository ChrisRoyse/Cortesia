/**
 * Environment-based Telemetry Injection System
 * 
 * Non-intrusive system for injecting telemetry capabilities into LLMKG
 * through environment variables and runtime proxy patterns without 
 * modifying any core Rust code.
 */

import { telemetryConfig, TelemetryConfig } from './config.js';
import { telemetryRecorder, TelemetryRecorder } from './recorder.js';
import { mcpTelemetryProxy, MCPTelemetryProxy } from './proxy.js';
import { MCPClient } from '../mcp/client.js';

export interface InjectionTarget {
  /** Target name/identifier */
  name: string;
  
  /** Target type */
  type: 'mcp_server' | 'tool' | 'client' | 'module';
  
  /** Target endpoint or module path */
  target: string;
  
  /** Injection configuration */
  config: {
    enabled: boolean;
    samplingRate?: number;
    categories?: string[];
    customMetadata?: Record<string, any>;
  };
}

export interface InjectionResult {
  /** Whether injection was successful */
  success: boolean;
  
  /** Injected proxy or wrapper */
  proxy?: any;
  
  /** Original object reference */
  original?: any;
  
  /** Error information if injection failed */
  error?: Error;
  
  /** Injection metadata */
  metadata: {
    timestamp: number;
    target: InjectionTarget;
    overhead: number;
  };
}

export interface RuntimeHook {
  /** Hook name */
  name: string;
  
  /** Target pattern to match */
  pattern: RegExp;
  
  /** Hook function */
  hook: (target: any, metadata: Record<string, any>) => any;
  
  /** Priority (higher = executed first) */
  priority: number;
  
  /** Whether hook is active */
  active: boolean;
}

/**
 * Environment-based telemetry injection system
 */
export class TelemetryInjector {
  private injectionTargets: Map<string, InjectionTarget> = new Map();
  private injectionResults: Map<string, InjectionResult> = new Map();
  private runtimeHooks: Map<string, RuntimeHook> = new Map();
  private isInitialized = false;
  private globalProxies: Map<string, any> = new Map();
  private environmentWatcher?: NodeJS.Timeout;

  constructor() {
    // Initialize from environment variables
    this.loadConfigurationFromEnvironment();
  }

  /**
   * Initialize the injection system
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Setup non-intrusive monitoring
      await this.setupNonIntrusiveMonitoring();
      
      // Inject environment configuration
      this.injectEnvironmentConfig();
      
      // Setup runtime hooks
      this.setupRuntimeHooks();
      
      // Start environment watching
      this.startEnvironmentWatching();
      
      this.isInitialized = true;
      
      telemetryRecorder.recordLog(
        'system',
        'telemetry.injector.initialized',
        'Telemetry injection system initialized successfully',
        'info'
      );
      
    } catch (error) {
      telemetryRecorder.recordLog(
        'system',
        'telemetry.injector.init_error',
        `Failed to initialize telemetry injector: ${error.message}`,
        'error',
        { error: error.stack }
      );
      throw error;
    }
  }

  /**
   * Setup non-intrusive monitoring without code modifications
   */
  async setupNonIntrusiveMonitoring(): Promise<void> {
    // Inject global MCP client proxy if available
    if (typeof global !== 'undefined') {
      this.injectGlobalMCPProxy();
    }

    // Setup process-level hooks
    this.setupProcessHooks();
    
    // Setup module loading hooks
    this.setupModuleLoadingHooks();
    
    // Setup network request interception
    this.setupNetworkInterception();
  }

  /**
   * Inject environment-based configuration
   */
  injectEnvironmentConfig(): void {
    const config = telemetryConfig.getConfig();
    
    // Set environment variables that LLMKG might read
    const envVars = telemetryConfig.exportToEnv();
    
    for (const [key, value] of Object.entries(envVars)) {
      if (!process.env[key]) {
        process.env[key] = value;
      }
    }

    // Inject telemetry hooks into common Node.js modules if available
    this.injectCommonModuleHooks();
    
    telemetryRecorder.recordLog(
      'system',
      'telemetry.config.injected',
      'Environment configuration injected',
      'info',
      { variableCount: Object.keys(envVars).length }
    );
  }

  /**
   * Register an injection target
   */
  registerInjectionTarget(target: InjectionTarget): void {
    this.injectionTargets.set(target.name, target);
    
    // Attempt immediate injection if system is initialized
    if (this.isInitialized) {
      this.attemptInjection(target);
    }
  }

  /**
   * Unregister an injection target
   */
  unregisterInjectionTarget(name: string): void {
    const result = this.injectionResults.get(name);
    if (result?.proxy && typeof result.proxy.cleanup === 'function') {
      result.proxy.cleanup();
    }
    
    this.injectionTargets.delete(name);
    this.injectionResults.delete(name);
  }

  /**
   * Inject telemetry into MCP client
   */
  injectMCPClient(client: MCPClient): MCPClient {
    if (!telemetryConfig.isEnabled()) {
      return client;
    }

    const proxy = mcpTelemetryProxy.wrapMCPClient(client);
    
    this.injectionResults.set('mcp_client', {
      success: true,
      proxy,
      original: client,
      metadata: {
        timestamp: Date.now(),
        target: {
          name: 'mcp_client',
          type: 'client',
          target: 'MCPClient',
          config: { enabled: true },
        },
        overhead: 0,
      },
    });

    telemetryRecorder.recordLog(
      'mcp',
      'telemetry.injection.mcp_client',
      'MCP client telemetry injection successful',
      'info'
    );

    return proxy;
  }

  /**
   * Inject telemetry into a tool function
   */
  injectTool(
    toolName: string, 
    originalFunction: Function,
    category?: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph'
  ): Function {
    if (!telemetryConfig.isInstrumentationEnabled(category || 'sdr')) {
      return originalFunction;
    }

    const wrappedFunction = mcpTelemetryProxy.wrapTool(toolName, originalFunction, category);
    
    this.injectionResults.set(`tool_${toolName}`, {
      success: true,
      proxy: wrappedFunction,
      original: originalFunction,
      metadata: {
        timestamp: Date.now(),
        target: {
          name: `tool_${toolName}`,
          type: 'tool',
          target: toolName,
          config: { enabled: true, categories: [category || 'sdr'] },
        },
        overhead: 0,
      },
    });

    return wrappedFunction;
  }

  /**
   * Setup runtime hooks for automatic injection
   */
  private setupRuntimeHooks(): void {
    // Hook for MCP server connections
    this.addRuntimeHook({
      name: 'mcp_server_hook',
      pattern: /mcp.*server/i,
      hook: (target: any, metadata: Record<string, any>) => {
        return this.injectMCPServerProxy(target, metadata);
      },
      priority: 100,
      active: true,
    });

    // Hook for tool executions
    this.addRuntimeHook({
      name: 'tool_execution_hook',
      pattern: /tool.*execute|execute.*tool/i,
      hook: (target: any, metadata: Record<string, any>) => {
        return this.injectToolProxy(target, metadata);
      },
      priority: 90,
      active: true,
    });

    // Hook for LLMKG-specific operations
    this.addRuntimeHook({
      name: 'llmkg_operations_hook',
      pattern: /sdr|cognitive|neural|memory|attention|graph/i,
      hook: (target: any, metadata: Record<string, any>) => {
        return this.injectLLMKGOperationProxy(target, metadata);
      },
      priority: 80,
      active: true,
    });
  }

  /**
   * Add a runtime hook
   */
  addRuntimeHook(hook: RuntimeHook): void {
    this.runtimeHooks.set(hook.name, hook);
  }

  /**
   * Remove a runtime hook
   */
  removeRuntimeHook(name: string): void {
    this.runtimeHooks.delete(name);
  }

  /**
   * Inject global MCP proxy
   */
  private injectGlobalMCPProxy(): void {
    if (typeof window !== 'undefined') {
      // Browser environment
      this.injectBrowserMCPProxy();
    } else if (typeof global !== 'undefined') {
      // Node.js environment
      this.injectNodeMCPProxy();
    }
  }

  /**
   * Inject Node.js MCP proxy
   */
  private injectNodeMCPProxy(): void {
    // Proxy global require to intercept MCP-related modules
    const originalRequire = global.require;
    
    if (originalRequire) {
      global.require = new Proxy(originalRequire, {
        apply: (target, thisArg, args) => {
          const moduleName = args[0];
          const result = Reflect.apply(target, thisArg, args);
          
          // Check if this looks like an MCP module
          if (typeof moduleName === 'string' && this.isMCPRelatedModule(moduleName)) {
            return this.wrapMCPModule(result, moduleName);
          }
          
          return result;
        },
      });
    }
  }

  /**
   * Inject browser MCP proxy
   */
  private injectBrowserMCPProxy(): void {
    // Proxy WebSocket for MCP connections
    if (typeof WebSocket !== 'undefined') {
      const OriginalWebSocket = WebSocket;
      
      (window as any).WebSocket = class extends OriginalWebSocket {
        constructor(url: string | URL, protocols?: string | string[]) {
          super(url, protocols);
          this.wrapWebSocketForTelemetry();
        }
        
        private wrapWebSocketForTelemetry(): void {
          const originalSend = this.send;
          this.send = (data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
            this.recordWebSocketMessage('send', data);
            return originalSend.call(this, data);
          };
          
          this.addEventListener('message', (event) => {
            this.recordWebSocketMessage('receive', event.data);
          });
        }
        
        private recordWebSocketMessage(direction: 'send' | 'receive', data: any): void {
          if (telemetryConfig.isEnabled()) {
            telemetryRecorder.recordTrace(
              'mcp',
              `websocket.${direction}`,
              { 
                size: typeof data === 'string' ? data.length : data.byteLength || 0,
                timestamp: Date.now(),
              }
            );
          }
        }
      };
    }
  }

  /**
   * Setup process-level hooks
   */
  private setupProcessHooks(): void {
    // Hook into process.nextTick for performance monitoring
    const originalNextTick = process.nextTick;
    process.nextTick = (...args) => {
      if (telemetryConfig.isEnabled() && Math.random() < 0.01) { // 1% sampling
        telemetryRecorder.recordMetric('system', 'process.next_tick', 1);
      }
      return originalNextTick.apply(process, args);
    };

    // Hook into process exit for cleanup
    process.on('exit', () => {
      this.cleanup();
    });

    process.on('SIGTERM', () => {
      this.cleanup();
      process.exit(0);
    });

    process.on('SIGINT', () => {
      this.cleanup();
      process.exit(0);
    });
  }

  /**
   * Setup module loading hooks
   */
  private setupModuleLoadingHooks(): void {
    // Hook into Module.prototype.require if available
    try {
      const Module = require('module');
      const originalLoad = Module.prototype.load;
      
      Module.prototype.load = function(filename: string) {
        const result = originalLoad.call(this, filename);
        
        // Check if this is an LLMKG-related module
        if (filename.includes('llmkg') || filename.includes('mcp')) {
          telemetryRecorder.recordLog(
            'system',
            'module.loaded',
            `Module loaded: ${filename}`,
            'debug',
            { filename }
          );
        }
        
        return result;
      };
    } catch (error) {
      // Silently continue if Module hooking fails
    }
  }

  /**
   * Setup network request interception
   */
  private setupNetworkInterception(): void {
    // Hook into http/https modules if available
    try {
      const http = require('http');
      const https = require('https');
      
      [http, https].forEach((module) => {
        const originalRequest = module.request;
        module.request = (...args: any[]) => {
          const req = originalRequest.apply(module, args);
          this.wrapHttpRequest(req);
          return req;
        };
      });
    } catch (error) {
      // Silently continue if HTTP hooking fails
    }
  }

  /**
   * Wrap HTTP request for telemetry
   */
  private wrapHttpRequest(req: any): void {
    if (!telemetryConfig.isEnabled()) return;

    const startTime = Date.now();
    
    req.on('response', (res: any) => {
      const duration = Date.now() - startTime;
      
      telemetryRecorder.recordMetric(
        'system',
        'http.request.duration',
        duration,
        {
          status_code: res.statusCode?.toString() || 'unknown',
          method: req.method || 'unknown',
        }
      );
    });

    req.on('error', (error: Error) => {
      telemetryRecorder.recordLog(
        'system',
        'http.request.error',
        `HTTP request error: ${error.message}`,
        'error',
        { error: error.stack }
      );
    });
  }

  /**
   * Start watching environment variables for changes
   */
  private startEnvironmentWatching(): void {
    this.environmentWatcher = setInterval(() => {
      // Check for configuration changes in environment variables
      const currentConfig = telemetryConfig.getConfig();
      telemetryConfig.onConfigChange((newConfig) => {
        if (JSON.stringify(currentConfig) !== JSON.stringify(newConfig)) {
          this.handleConfigurationChange(newConfig);
        }
      });
    }, 10000); // Check every 10 seconds
  }

  /**
   * Handle configuration changes
   */
  private handleConfigurationChange(newConfig: TelemetryConfig): void {
    telemetryRecorder.recordLog(
      'system',
      'telemetry.config.changed',
      'Telemetry configuration changed',
      'info',
      { config: newConfig }
    );

    // Re-evaluate injection targets
    for (const target of this.injectionTargets.values()) {
      this.attemptInjection(target);
    }
  }

  /**
   * Load configuration from environment variables
   */
  private loadConfigurationFromEnvironment(): void {
    // Parse injection targets from environment
    const targetsEnv = process.env.LLMKG_TELEMETRY_INJECTION_TARGETS;
    if (targetsEnv) {
      try {
        const targets = JSON.parse(targetsEnv) as InjectionTarget[];
        targets.forEach(target => this.registerInjectionTarget(target));
      } catch (error) {
        console.warn('Failed to parse injection targets from environment:', error);
      }
    }
  }

  /**
   * Inject common module hooks
   */
  private injectCommonModuleHooks(): void {
    // Common modules that might be used by LLMKG
    const commonModules = ['fs', 'path', 'os', 'child_process'];
    
    commonModules.forEach(moduleName => {
      try {
        const module = require(moduleName);
        this.wrapCommonModule(module, moduleName);
      } catch (error) {
        // Module not available, skip
      }
    });
  }

  /**
   * Wrap common module for telemetry
   */
  private wrapCommonModule(module: any, moduleName: string): any {
    if (!telemetryConfig.isEnabled()) return module;

    return new Proxy(module, {
      get: (target, prop) => {
        const originalValue = target[prop];
        
        if (typeof originalValue === 'function' && Math.random() < 0.001) { // 0.1% sampling
          return (...args: any[]) => {
            const startTime = performance.now();
            try {
              const result = originalValue.apply(target, args);
              const duration = performance.now() - startTime;
              
              telemetryRecorder.recordMetric(
                'system',
                `module.${moduleName}.${String(prop)}`,
                duration
              );
              
              return result;
            } catch (error) {
              const duration = performance.now() - startTime;
              
              telemetryRecorder.recordMetric(
                'system',
                `module.${moduleName}.${String(prop)}.error`,
                duration
              );
              
              throw error;
            }
          };
        }
        
        return originalValue;
      },
    });
  }

  /**
   * Check if a module is MCP-related
   */
  private isMCPRelatedModule(moduleName: string): boolean {
    const mcpPatterns = [
      /mcp/i,
      /model.*context.*protocol/i,
      /llmkg/i,
      /brain.*inspired/i,
      /federated.*learning/i,
    ];

    return mcpPatterns.some(pattern => pattern.test(moduleName));
  }

  /**
   * Wrap MCP-related module
   */
  private wrapMCPModule(module: any, moduleName: string): any {
    if (!telemetryConfig.isEnabled()) return module;

    telemetryRecorder.recordLog(
      'mcp',
      'module.mcp.loaded',
      `MCP-related module loaded: ${moduleName}`,
      'info',
      { moduleName }
    );

    return mcpTelemetryProxy.wrapMCPClient(module);
  }

  /**
   * Attempt injection for a target
   */
  private attemptInjection(target: InjectionTarget): void {
    const startTime = performance.now();
    
    try {
      let result: InjectionResult;
      
      switch (target.type) {
        case 'mcp_server':
          result = this.injectMCPServerTarget(target);
          break;
        case 'tool':
          result = this.injectToolTarget(target);
          break;
        case 'client':
          result = this.injectClientTarget(target);
          break;
        default:
          throw new Error(`Unknown injection target type: ${target.type}`);
      }
      
      result.metadata.overhead = performance.now() - startTime;
      this.injectionResults.set(target.name, result);
      
    } catch (error) {
      const result: InjectionResult = {
        success: false,
        error: error as Error,
        metadata: {
          timestamp: Date.now(),
          target,
          overhead: performance.now() - startTime,
        },
      };
      
      this.injectionResults.set(target.name, result);
    }
  }

  /**
   * Inject MCP server target
   */
  private injectMCPServerTarget(target: InjectionTarget): InjectionResult {
    // This would attempt to proxy MCP server connections
    // Implementation depends on how LLMKG exposes server connections
    return {
      success: true,
      metadata: {
        timestamp: Date.now(),
        target,
        overhead: 0,
      },
    };
  }

  /**
   * Inject tool target
   */
  private injectToolTarget(target: InjectionTarget): InjectionResult {
    // This would attempt to wrap tool functions
    return {
      success: true,
      metadata: {
        timestamp: Date.now(),
        target,
        overhead: 0,
      },
    };
  }

  /**
   * Inject client target
   */
  private injectClientTarget(target: InjectionTarget): InjectionResult {
    // This would attempt to wrap client instances
    return {
      success: true,
      metadata: {
        timestamp: Date.now(),
        target,
        overhead: 0,
      },
    };
  }

  /**
   * Inject MCP server proxy
   */
  private injectMCPServerProxy(target: any, metadata: Record<string, any>): any {
    return mcpTelemetryProxy.wrapMCPClient(target);
  }

  /**
   * Inject tool proxy
   */
  private injectToolProxy(target: any, metadata: Record<string, any>): any {
    return mcpTelemetryProxy.wrapTool(metadata.name || 'unknown', target);
  }

  /**
   * Inject LLMKG operation proxy
   */
  private injectLLMKGOperationProxy(target: any, metadata: Record<string, any>): any {
    const category = this.inferOperationCategory(metadata.name || '');
    return mcpTelemetryProxy.wrapTool(metadata.name || 'unknown', target, category);
  }

  /**
   * Infer operation category from name
   */
  private inferOperationCategory(name: string): 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph' {
    const categoryMap = {
      sdr: /sdr|sparse.*distributed/i,
      cognitive: /cognitive|brain|cortex/i,
      neural: /neural|neuron|synapse/i,
      memory: /memory|store|cache/i,
      attention: /attention|focus|salience/i,
      graph: /graph|node|edge|relationship/i,
    };

    for (const [category, pattern] of Object.entries(categoryMap)) {
      if (pattern.test(name)) {
        return category as any;
      }
    }

    return 'sdr';
  }

  /**
   * Get injection statistics
   */
  getStats(): {
    totalTargets: number;
    successfulInjections: number;
    failedInjections: number;
    activeHooks: number;
    averageOverhead: number;
  } {
    const results = Array.from(this.injectionResults.values());
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);
    const avgOverhead = results.length > 0 ?
      results.reduce((sum, r) => sum + r.metadata.overhead, 0) / results.length : 0;

    return {
      totalTargets: this.injectionTargets.size,
      successfulInjections: successful.length,
      failedInjections: failed.length,
      activeHooks: Array.from(this.runtimeHooks.values()).filter(h => h.active).length,
      averageOverhead: avgOverhead,
    };
  }

  /**
   * Clean up injection system
   */
  cleanup(): void {
    // Clear environment watcher
    if (this.environmentWatcher) {
      clearInterval(this.environmentWatcher);
      this.environmentWatcher = undefined;
    }

    // Cleanup all injection results
    for (const result of this.injectionResults.values()) {
      if (result.proxy && typeof result.proxy.cleanup === 'function') {
        result.proxy.cleanup();
      }
    }

    // Clear all maps
    this.injectionTargets.clear();
    this.injectionResults.clear();
    this.runtimeHooks.clear();
    this.globalProxies.clear();

    this.isInitialized = false;
  }
}

// Global injector instance
export const telemetryInjector = new TelemetryInjector();