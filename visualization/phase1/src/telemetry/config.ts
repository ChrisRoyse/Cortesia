/**
 * Telemetry Configuration Management System
 * 
 * Provides non-intrusive configuration management for LLMKG telemetry
 * through environment variables and runtime settings.
 */

export interface TelemetryConfig {
  /** Enable/disable telemetry collection */
  enabled: boolean;
  
  /** Telemetry collection level */
  level: 'minimal' | 'standard' | 'verbose';
  
  /** Maximum performance overhead allowed (percentage) */
  maxOverhead: number;
  
  /** Buffer size for telemetry data */
  bufferSize: number;
  
  /** Flush interval in milliseconds */
  flushInterval: number;
  
  /** Enable specific instrumentation types */
  instrumentation: {
    sdr: boolean;
    cognitive: boolean;
    neural: boolean;
    memory: boolean;
    attention: boolean;
    graph: boolean;
  };
  
  /** MCP server configuration */
  mcpServers: {
    brainInspired: boolean;
    federated: boolean;
  };
  
  /** Performance monitoring settings */
  performance: {
    enableImpactMonitoring: boolean;
    samplingRate: number;
    alertThreshold: number;
  };
  
  /** Data collection settings */
  collection: {
    enableMetrics: boolean;
    enableTraces: boolean;
    enableLogs: boolean;
    retentionPeriod: number;
  };
}

export const DEFAULT_CONFIG: TelemetryConfig = {
  enabled: true,
  level: 'standard',
  maxOverhead: 1.0, // 1% maximum overhead
  bufferSize: 1000,
  flushInterval: 5000, // 5 seconds
  instrumentation: {
    sdr: true,
    cognitive: true,
    neural: true,
    memory: true,
    attention: true,
    graph: true,
  },
  mcpServers: {
    brainInspired: true,
    federated: true,
  },
  performance: {
    enableImpactMonitoring: true,
    samplingRate: 0.1, // 10% sampling for performance monitoring
    alertThreshold: 0.8, // Alert at 80% of max overhead
  },
  collection: {
    enableMetrics: true,
    enableTraces: false, // Disabled by default for minimal overhead
    enableLogs: true,
    retentionPeriod: 24 * 60 * 60 * 1000, // 24 hours in milliseconds
  },
};

/**
 * Configuration manager for LLMKG telemetry system
 */
export class TelemetryConfigManager {
  private config: TelemetryConfig;
  private configChangeListeners: Array<(config: TelemetryConfig) => void> = [];

  constructor() {
    this.config = this.loadConfiguration();
  }

  /**
   * Load configuration from environment variables and defaults
   */
  private loadConfiguration(): TelemetryConfig {
    const envConfig: Partial<TelemetryConfig> = {};

    // Core settings
    if (process.env.LLMKG_TELEMETRY_ENABLED !== undefined) {
      envConfig.enabled = process.env.LLMKG_TELEMETRY_ENABLED === 'true';
    }

    if (process.env.LLMKG_TELEMETRY_LEVEL) {
      const level = process.env.LLMKG_TELEMETRY_LEVEL;
      if (['minimal', 'standard', 'verbose'].includes(level)) {
        envConfig.level = level as 'minimal' | 'standard' | 'verbose';
      }
    }

    if (process.env.LLMKG_TELEMETRY_MAX_OVERHEAD) {
      const overhead = parseFloat(process.env.LLMKG_TELEMETRY_MAX_OVERHEAD);
      if (!isNaN(overhead) && overhead >= 0 && overhead <= 10) {
        envConfig.maxOverhead = overhead;
      }
    }

    if (process.env.LLMKG_TELEMETRY_BUFFER_SIZE) {
      const bufferSize = parseInt(process.env.LLMKG_TELEMETRY_BUFFER_SIZE, 10);
      if (!isNaN(bufferSize) && bufferSize > 0) {
        envConfig.bufferSize = bufferSize;
      }
    }

    if (process.env.LLMKG_TELEMETRY_FLUSH_INTERVAL) {
      const flushInterval = parseInt(process.env.LLMKG_TELEMETRY_FLUSH_INTERVAL, 10);
      if (!isNaN(flushInterval) && flushInterval > 0) {
        envConfig.flushInterval = flushInterval;
      }
    }

    // Instrumentation settings
    const instrumentation: Partial<TelemetryConfig['instrumentation']> = {};
    if (process.env.LLMKG_TELEMETRY_INSTR_SDR !== undefined) {
      instrumentation.sdr = process.env.LLMKG_TELEMETRY_INSTR_SDR === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_INSTR_COGNITIVE !== undefined) {
      instrumentation.cognitive = process.env.LLMKG_TELEMETRY_INSTR_COGNITIVE === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_INSTR_NEURAL !== undefined) {
      instrumentation.neural = process.env.LLMKG_TELEMETRY_INSTR_NEURAL === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_INSTR_MEMORY !== undefined) {
      instrumentation.memory = process.env.LLMKG_TELEMETRY_INSTR_MEMORY === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_INSTR_ATTENTION !== undefined) {
      instrumentation.attention = process.env.LLMKG_TELEMETRY_INSTR_ATTENTION === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_INSTR_GRAPH !== undefined) {
      instrumentation.graph = process.env.LLMKG_TELEMETRY_INSTR_GRAPH === 'true';
    }

    if (Object.keys(instrumentation).length > 0) {
      envConfig.instrumentation = { ...DEFAULT_CONFIG.instrumentation, ...instrumentation };
    }

    // MCP server settings
    const mcpServers: Partial<TelemetryConfig['mcpServers']> = {};
    if (process.env.LLMKG_TELEMETRY_MCP_BRAIN_INSPIRED !== undefined) {
      mcpServers.brainInspired = process.env.LLMKG_TELEMETRY_MCP_BRAIN_INSPIRED === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_MCP_FEDERATED !== undefined) {
      mcpServers.federated = process.env.LLMKG_TELEMETRY_MCP_FEDERATED === 'true';
    }

    if (Object.keys(mcpServers).length > 0) {
      envConfig.mcpServers = { ...DEFAULT_CONFIG.mcpServers, ...mcpServers };
    }

    // Performance settings
    const performance: Partial<TelemetryConfig['performance']> = {};
    if (process.env.LLMKG_TELEMETRY_PERF_MONITORING !== undefined) {
      performance.enableImpactMonitoring = process.env.LLMKG_TELEMETRY_PERF_MONITORING === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_SAMPLING_RATE) {
      const samplingRate = parseFloat(process.env.LLMKG_TELEMETRY_SAMPLING_RATE);
      if (!isNaN(samplingRate) && samplingRate >= 0 && samplingRate <= 1) {
        performance.samplingRate = samplingRate;
      }
    }
    if (process.env.LLMKG_TELEMETRY_ALERT_THRESHOLD) {
      const alertThreshold = parseFloat(process.env.LLMKG_TELEMETRY_ALERT_THRESHOLD);
      if (!isNaN(alertThreshold) && alertThreshold >= 0 && alertThreshold <= 1) {
        performance.alertThreshold = alertThreshold;
      }
    }

    if (Object.keys(performance).length > 0) {
      envConfig.performance = { ...DEFAULT_CONFIG.performance, ...performance };
    }

    // Collection settings
    const collection: Partial<TelemetryConfig['collection']> = {};
    if (process.env.LLMKG_TELEMETRY_METRICS !== undefined) {
      collection.enableMetrics = process.env.LLMKG_TELEMETRY_METRICS === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_TRACES !== undefined) {
      collection.enableTraces = process.env.LLMKG_TELEMETRY_TRACES === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_LOGS !== undefined) {
      collection.enableLogs = process.env.LLMKG_TELEMETRY_LOGS === 'true';
    }
    if (process.env.LLMKG_TELEMETRY_RETENTION) {
      const retention = parseInt(process.env.LLMKG_TELEMETRY_RETENTION, 10);
      if (!isNaN(retention) && retention > 0) {
        collection.retentionPeriod = retention;
      }
    }

    if (Object.keys(collection).length > 0) {
      envConfig.collection = { ...DEFAULT_CONFIG.collection, ...collection };
    }

    return { ...DEFAULT_CONFIG, ...envConfig };
  }

  /**
   * Get current configuration
   */
  getConfig(): TelemetryConfig {
    return { ...this.config };
  }

  /**
   * Update configuration at runtime
   */
  updateConfig(updates: Partial<TelemetryConfig>): void {
    this.config = { ...this.config, ...updates };
    this.notifyConfigChange();
  }

  /**
   * Check if telemetry is enabled
   */
  isEnabled(): boolean {
    return this.config.enabled;
  }

  /**
   * Check if specific instrumentation is enabled
   */
  isInstrumentationEnabled(type: keyof TelemetryConfig['instrumentation']): boolean {
    return this.config.enabled && this.config.instrumentation[type];
  }

  /**
   * Check if MCP server monitoring is enabled
   */
  isMCPServerEnabled(server: keyof TelemetryConfig['mcpServers']): boolean {
    return this.config.enabled && this.config.mcpServers[server];
  }

  /**
   * Get configuration for specific telemetry level
   */
  getConfigForLevel(level: 'minimal' | 'standard' | 'verbose'): Partial<TelemetryConfig> {
    const baseConfig = this.getConfig();

    switch (level) {
      case 'minimal':
        return {
          ...baseConfig,
          level,
          instrumentation: {
            ...baseConfig.instrumentation,
            neural: false,
            attention: false,
          },
          collection: {
            ...baseConfig.collection,
            enableTraces: false,
            enableLogs: false,
          },
          performance: {
            ...baseConfig.performance,
            samplingRate: 0.01, // 1% sampling
          },
        };

      case 'verbose':
        return {
          ...baseConfig,
          level,
          collection: {
            ...baseConfig.collection,
            enableTraces: true,
            enableLogs: true,
          },
          performance: {
            ...baseConfig.performance,
            samplingRate: 1.0, // 100% sampling
          },
        };

      default: // standard
        return baseConfig;
    }
  }

  /**
   * Register listener for configuration changes
   */
  onConfigChange(listener: (config: TelemetryConfig) => void): void {
    this.configChangeListeners.push(listener);
  }

  /**
   * Unregister configuration change listener
   */
  offConfigChange(listener: (config: TelemetryConfig) => void): void {
    const index = this.configChangeListeners.indexOf(listener);
    if (index !== -1) {
      this.configChangeListeners.splice(index, 1);
    }
  }

  /**
   * Notify all listeners of configuration changes
   */
  private notifyConfigChange(): void {
    this.configChangeListeners.forEach(listener => {
      try {
        listener(this.config);
      } catch (error) {
        console.warn('Error notifying config change listener:', error);
      }
    });
  }

  /**
   * Validate configuration values
   */
  validateConfig(config: Partial<TelemetryConfig>): string[] {
    const errors: string[] = [];

    if (config.maxOverhead !== undefined) {
      if (config.maxOverhead < 0 || config.maxOverhead > 10) {
        errors.push('maxOverhead must be between 0 and 10');
      }
    }

    if (config.bufferSize !== undefined) {
      if (config.bufferSize <= 0 || !Number.isInteger(config.bufferSize)) {
        errors.push('bufferSize must be a positive integer');
      }
    }

    if (config.flushInterval !== undefined) {
      if (config.flushInterval <= 0 || !Number.isInteger(config.flushInterval)) {
        errors.push('flushInterval must be a positive integer');
      }
    }

    if (config.performance?.samplingRate !== undefined) {
      if (config.performance.samplingRate < 0 || config.performance.samplingRate > 1) {
        errors.push('performance.samplingRate must be between 0 and 1');
      }
    }

    if (config.performance?.alertThreshold !== undefined) {
      if (config.performance.alertThreshold < 0 || config.performance.alertThreshold > 1) {
        errors.push('performance.alertThreshold must be between 0 and 1');
      }
    }

    return errors;
  }

  /**
   * Export configuration to environment variable format
   */
  exportToEnv(): Record<string, string> {
    const env: Record<string, string> = {};

    env.LLMKG_TELEMETRY_ENABLED = this.config.enabled.toString();
    env.LLMKG_TELEMETRY_LEVEL = this.config.level;
    env.LLMKG_TELEMETRY_MAX_OVERHEAD = this.config.maxOverhead.toString();
    env.LLMKG_TELEMETRY_BUFFER_SIZE = this.config.bufferSize.toString();
    env.LLMKG_TELEMETRY_FLUSH_INTERVAL = this.config.flushInterval.toString();

    // Instrumentation
    env.LLMKG_TELEMETRY_INSTR_SDR = this.config.instrumentation.sdr.toString();
    env.LLMKG_TELEMETRY_INSTR_COGNITIVE = this.config.instrumentation.cognitive.toString();
    env.LLMKG_TELEMETRY_INSTR_NEURAL = this.config.instrumentation.neural.toString();
    env.LLMKG_TELEMETRY_INSTR_MEMORY = this.config.instrumentation.memory.toString();
    env.LLMKG_TELEMETRY_INSTR_ATTENTION = this.config.instrumentation.attention.toString();
    env.LLMKG_TELEMETRY_INSTR_GRAPH = this.config.instrumentation.graph.toString();

    // MCP servers
    env.LLMKG_TELEMETRY_MCP_BRAIN_INSPIRED = this.config.mcpServers.brainInspired.toString();
    env.LLMKG_TELEMETRY_MCP_FEDERATED = this.config.mcpServers.federated.toString();

    // Performance
    env.LLMKG_TELEMETRY_PERF_MONITORING = this.config.performance.enableImpactMonitoring.toString();
    env.LLMKG_TELEMETRY_SAMPLING_RATE = this.config.performance.samplingRate.toString();
    env.LLMKG_TELEMETRY_ALERT_THRESHOLD = this.config.performance.alertThreshold.toString();

    // Collection
    env.LLMKG_TELEMETRY_METRICS = this.config.collection.enableMetrics.toString();
    env.LLMKG_TELEMETRY_TRACES = this.config.collection.enableTraces.toString();
    env.LLMKG_TELEMETRY_LOGS = this.config.collection.enableLogs.toString();
    env.LLMKG_TELEMETRY_RETENTION = this.config.collection.retentionPeriod.toString();

    return env;
  }
}

// Global configuration instance
export const telemetryConfig = new TelemetryConfigManager();