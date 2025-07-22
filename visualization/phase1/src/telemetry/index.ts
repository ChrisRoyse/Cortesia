/**
 * LLMKG Telemetry System - Main Export Module
 * 
 * Non-intrusive telemetry injection system for LLMKG visualization dashboard.
 * Provides comprehensive monitoring without requiring any modifications to
 * LLMKG's core Rust codebase.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

// Core configuration management
export {
  TelemetryConfig,
  TelemetryConfigManager,
  telemetryConfig,
  DEFAULT_CONFIG,
} from './config.js';

// Data recording and buffering
export {
  TelemetryEvent,
  TelemetryBatch,
  TelemetryRecorder,
  PerformanceImpact,
  telemetryRecorder,
} from './recorder.js';

// MCP proxy system
export {
  MCPCall,
  TelemetryData,
  ProxyConfig,
  MCPTelemetryProxy,
  ToolInstrumentation,
  mcpTelemetryProxy,
} from './proxy.js';

// Environment-based injection
export {
  InjectionTarget,
  InjectionResult,
  RuntimeHook,
  TelemetryInjector,
  telemetryInjector,
} from './injector.js';

// LLMKG operation instrumentation
export {
  InstrumentationPoint,
  OperationMetrics,
  PerformanceProfile,
  LLMKGInstrumentation,
  llmkgInstrumentation,
} from './instrumentation.js';

// Central orchestration manager
export {
  TelemetryManagerConfig,
  TelemetrySnapshot,
  TelemetryAlert,
  TelemetryManager,
  telemetryManager,
  DEFAULT_MANAGER_CONFIG,
} from './manager.js';

/**
 * Quick-start function for basic telemetry setup
 * 
 * @example
 * ```typescript
 * import { initializeTelemetry } from '@llmkg/visualization/telemetry';
 * 
 * // Initialize with default settings
 * await initializeTelemetry();
 * 
 * // Initialize with custom configuration
 * await initializeTelemetry({
 *   level: 'verbose',
 *   maxOverhead: 0.5,
 *   instrumentation: {
 *     sdr: true,
 *     cognitive: true,
 *     neural: false,
 *   }
 * });
 * ```
 */
export async function initializeTelemetry(config?: Partial<TelemetryConfig>): Promise<void> {
  const { telemetryConfig, telemetryManager } = await import('./config.js').then(async () => ({
    telemetryConfig: (await import('./config.js')).telemetryConfig,
    telemetryManager: (await import('./manager.js')).telemetryManager,
  }));

  // Update configuration if provided
  if (config) {
    telemetryConfig.updateConfig(config);
  }

  // Initialize the telemetry system
  await telemetryManager.initialize();
}

/**
 * Quick-start function for instrumenting MCP clients
 * 
 * @example
 * ```typescript
 * import { wrapMCPClient } from '@llmkg/visualization/telemetry';
 * import { MCPClient } from '@llmkg/visualization/mcp';
 * 
 * const client = new MCPClient(config);
 * const instrumentedClient = wrapMCPClient(client);
 * ```
 */
export function wrapMCPClient(client: any): any {
  const { telemetryManager } = require('./manager.js');
  return telemetryManager.wrapMCPClient(client);
}

/**
 * Quick-start function for instrumenting tool functions
 * 
 * @example
 * ```typescript
 * import { wrapTool } from '@llmkg/visualization/telemetry';
 * 
 * const instrumentedTool = wrapTool('sdr_encode', originalSDREncodeFunction, 'sdr');
 * ```
 */
export function wrapTool(
  toolName: string,
  toolFunction: Function,
  category?: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph'
): Function {
  const { telemetryManager } = require('./manager.js');
  return telemetryManager.wrapTool(toolName, toolFunction, category);
}

/**
 * Quick-start function for instrumenting operations
 * 
 * @example
 * ```typescript
 * import { instrument } from '@llmkg/visualization/telemetry';
 * 
 * const result = instrument('sdr', 'encode_operation', () => {
 *   return performSDREncoding(data);
 * });
 * ```
 */
export function instrument<T>(
  category: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph',
  operationName: string,
  operation: () => T,
  metadata?: Record<string, any>
): T {
  const { telemetryManager } = require('./manager.js');
  return telemetryManager.instrument(category, operationName, operation, metadata);
}

/**
 * Quick-start function for instrumenting async operations
 * 
 * @example
 * ```typescript
 * import { instrumentAsync } from '@llmkg/visualization/telemetry';
 * 
 * const result = await instrumentAsync('graph', 'query_operation', async () => {
 *   return await performGraphQuery(query);
 * });
 * ```
 */
export async function instrumentAsync<T>(
  category: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph',
  operationName: string,
  operation: () => Promise<T>,
  metadata?: Record<string, any>
): Promise<T> {
  const { telemetryManager } = require('./manager.js');
  return telemetryManager.instrumentAsync(category, operationName, operation, metadata);
}

/**
 * Get current telemetry system status
 * 
 * @example
 * ```typescript
 * import { getTelemetryStatus } from '@llmkg/visualization/telemetry';
 * 
 * const status = getTelemetryStatus();
 * console.log(`Telemetry enabled: ${status.enabled}`);
 * console.log(`Current overhead: ${status.overhead}%`);
 * ```
 */
export function getTelemetryStatus(): {
  enabled: boolean;
  overhead: number;
  operationCount: number;
  errorRate: number;
  healthy: boolean;
} {
  const { telemetryManager } = require('./manager.js');
  const snapshot = telemetryManager.getSnapshot();
  const health = telemetryManager.getHealthStatus();
  
  return {
    enabled: snapshot.system.isEnabled,
    overhead: snapshot.performance.overallOverhead,
    operationCount: snapshot.operations.totalOperations,
    errorRate: snapshot.operations.errorRate,
    healthy: health.healthy,
  };
}

/**
 * Shutdown telemetry system
 * 
 * @example
 * ```typescript
 * import { shutdownTelemetry } from '@llmkg/visualization/telemetry';
 * 
 * await shutdownTelemetry();
 * ```
 */
export async function shutdownTelemetry(): Promise<void> {
  const { telemetryManager } = require('./manager.js');
  await telemetryManager.shutdown();
}

/**
 * Environment variable documentation
 */
export const ENVIRONMENT_VARIABLES = {
  // Core telemetry settings
  LLMKG_TELEMETRY_ENABLED: 'Enable/disable telemetry collection (true/false)',
  LLMKG_TELEMETRY_LEVEL: 'Telemetry collection level (minimal/standard/verbose)',
  LLMKG_TELEMETRY_MAX_OVERHEAD: 'Maximum performance overhead percentage (0-10)',
  LLMKG_TELEMETRY_BUFFER_SIZE: 'Buffer size for telemetry data (number)',
  LLMKG_TELEMETRY_FLUSH_INTERVAL: 'Flush interval in milliseconds (number)',

  // Instrumentation settings
  LLMKG_TELEMETRY_INSTR_SDR: 'Enable SDR operation instrumentation (true/false)',
  LLMKG_TELEMETRY_INSTR_COGNITIVE: 'Enable cognitive pattern instrumentation (true/false)',
  LLMKG_TELEMETRY_INSTR_NEURAL: 'Enable neural processing instrumentation (true/false)',
  LLMKG_TELEMETRY_INSTR_MEMORY: 'Enable memory system instrumentation (true/false)',
  LLMKG_TELEMETRY_INSTR_ATTENTION: 'Enable attention mechanism instrumentation (true/false)',
  LLMKG_TELEMETRY_INSTR_GRAPH: 'Enable graph query instrumentation (true/false)',

  // MCP server settings
  LLMKG_TELEMETRY_MCP_BRAIN_INSPIRED: 'Enable BrainInspiredMCPServer monitoring (true/false)',
  LLMKG_TELEMETRY_MCP_FEDERATED: 'Enable FederatedMCPServer monitoring (true/false)',

  // Performance settings
  LLMKG_TELEMETRY_PERF_MONITORING: 'Enable performance impact monitoring (true/false)',
  LLMKG_TELEMETRY_SAMPLING_RATE: 'Sampling rate for performance monitoring (0-1)',
  LLMKG_TELEMETRY_ALERT_THRESHOLD: 'Alert threshold as fraction of max overhead (0-1)',

  // Data collection settings
  LLMKG_TELEMETRY_METRICS: 'Enable metric collection (true/false)',
  LLMKG_TELEMETRY_TRACES: 'Enable trace collection (true/false)',
  LLMKG_TELEMETRY_LOGS: 'Enable log collection (true/false)',
  LLMKG_TELEMETRY_RETENTION: 'Data retention period in milliseconds (number)',

  // Injection targets
  LLMKG_TELEMETRY_INJECTION_TARGETS: 'JSON array of injection targets (JSON string)',
} as const;

/**
 * Usage examples and best practices
 */
export const USAGE_EXAMPLES = {
  basicSetup: `
// Basic telemetry setup
import { initializeTelemetry } from '@llmkg/visualization/telemetry';

await initializeTelemetry({
  level: 'standard',
  maxOverhead: 1.0,
  instrumentation: {
    sdr: true,
    cognitive: true,
    neural: true,
    memory: true,
    attention: false, // Disable for lower overhead
    graph: true,
  },
});
  `,

  mcpClientInstrumentation: `
// Instrument MCP client
import { wrapMCPClient } from '@llmkg/visualization/telemetry';
import { MCPClient } from '@llmkg/visualization/mcp';

const client = new MCPClient(config);
const instrumentedClient = wrapMCPClient(client);

// Use instrumented client normally
const tools = await instrumentedClient.listTools();
  `,

  operationInstrumentation: `
// Instrument LLMKG operations
import { instrument, instrumentAsync } from '@llmkg/visualization/telemetry';

// Synchronous operation
const encodedSDR = instrument('sdr', 'encode_data', () => {
  return sdrEncoder.encode(inputData);
}, { dataSize: inputData.length });

// Asynchronous operation  
const graphResult = await instrumentAsync('graph', 'complex_query', async () => {
  return await knowledgeGraph.query(complexQuery);
}, { queryComplexity: 'high' });
  `,

  environmentConfiguration: `
# Environment variable configuration
export LLMKG_TELEMETRY_ENABLED=true
export LLMKG_TELEMETRY_LEVEL=standard
export LLMKG_TELEMETRY_MAX_OVERHEAD=1.0
export LLMKG_TELEMETRY_INSTR_SDR=true
export LLMKG_TELEMETRY_INSTR_COGNITIVE=true
export LLMKG_TELEMETRY_INSTR_NEURAL=false
export LLMKG_TELEMETRY_PERF_MONITORING=true
export LLMKG_TELEMETRY_SAMPLING_RATE=0.1
  `,

  monitoringAndAlerts: `
// Monitor telemetry system health
import { getTelemetryStatus, telemetryManager } from '@llmkg/visualization/telemetry';

// Check system status
const status = getTelemetryStatus();
if (!status.healthy) {
  console.warn('Telemetry system health issues detected');
}

// Listen for alerts
telemetryManager.on('alert_created', (alert) => {
  console.error(\`Telemetry alert: \${alert.message}\`, alert.details);
});

// Get detailed health information
const health = telemetryManager.getHealthStatus();
if (health.issues.length > 0) {
  console.warn('Issues:', health.issues);
  console.log('Recommendations:', health.recommendations);
}
  `,
} as const;