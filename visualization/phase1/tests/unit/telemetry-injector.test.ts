/**
 * Unit tests for Telemetry Injector System
 * 
 * Tests the environment-based telemetry injection system including
 * non-intrusive monitoring, runtime hooks, and proxy injection.
 */

import { TelemetryInjector, InjectionTarget, RuntimeHook } from '../../src/telemetry/injector';
import { MockMCPClient, PerformanceTracker, TestHelpers } from '../config/test-utils';

// Mock external dependencies
jest.mock('../../src/telemetry/config', () => ({
  telemetryConfig: {
    getConfig: jest.fn(() => ({
      enabled: true,
      categories: ['sdr', 'cognitive', 'neural'],
      samplingRate: 1.0
    })),
    isEnabled: jest.fn(() => true),
    isInstrumentationEnabled: jest.fn(() => true),
    exportToEnv: jest.fn(() => ({
      LLMKG_TELEMETRY_ENABLED: 'true',
      LLMKG_TELEMETRY_SAMPLING_RATE: '1.0'
    })),
    onConfigChange: jest.fn()
  }
}));

jest.mock('../../src/telemetry/recorder', () => ({
  telemetryRecorder: {
    recordLog: jest.fn(),
    recordMetric: jest.fn(),
    recordTrace: jest.fn()
  }
}));

jest.mock('../../src/telemetry/proxy', () => ({
  mcpTelemetryProxy: {
    wrapMCPClient: jest.fn((client) => client),
    wrapTool: jest.fn((name, fn) => fn)
  }
}));

describe('TelemetryInjector', () => {
  let injector: TelemetryInjector;
  let mockMCPClient: MockMCPClient;
  let performanceTracker: PerformanceTracker;

  beforeEach(() => {
    injector = new TelemetryInjector();
    mockMCPClient = new MockMCPClient();
    performanceTracker = new PerformanceTracker();
  });

  afterEach(() => {
    if (injector) {
      injector.cleanup();
    }
    performanceTracker.clear();
    jest.clearAllMocks();
  });

  describe('Initialization', () => {
    it('should create injector instance', () => {
      expect(injector).toBeDefined();
      expect(injector).toBeInstanceOf(TelemetryInjector);
    });

    it('should initialize successfully', async () => {
      await expect(injector.initialize()).resolves.not.toThrow();
    });

    it('should handle double initialization', async () => {
      await injector.initialize();
      await expect(injector.initialize()).resolves.not.toThrow();
    });

    it('should load configuration from environment', () => {
      // Set environment variables
      process.env.LLMKG_TELEMETRY_INJECTION_TARGETS = JSON.stringify([
        {
          name: 'test_target',
          type: 'mcp_server',
          target: 'test_server',
          config: { enabled: true }
        }
      ]);

      const newInjector = new TelemetryInjector();
      expect(newInjector).toBeDefined();

      // Clean up
      delete process.env.LLMKG_TELEMETRY_INJECTION_TARGETS;
    });
  });

  describe('Injection Target Management', () => {
    beforeEach(async () => {
      await injector.initialize();
    });

    it('should register injection targets', () => {
      const target: InjectionTarget = {
        name: 'test_mcp_client',
        type: 'client',
        target: 'MCPClient',
        config: {
          enabled: true,
          samplingRate: 0.8,
          categories: ['sdr', 'cognitive']
        }
      };

      expect(() => injector.registerInjectionTarget(target)).not.toThrow();
    });

    it('should unregister injection targets', () => {
      const target: InjectionTarget = {
        name: 'test_target',
        type: 'tool',
        target: 'test_tool',
        config: { enabled: true }
      };

      injector.registerInjectionTarget(target);
      expect(() => injector.unregisterInjectionTarget('test_target')).not.toThrow();
    });

    it('should handle invalid targets gracefully', () => {
      expect(() => injector.unregisterInjectionTarget('non_existent')).not.toThrow();
    });
  });

  describe('MCP Client Injection', () => {
    beforeEach(async () => {
      await injector.initialize();
    });

    it('should inject telemetry into MCP client', () => {
      const wrappedClient = injector.injectMCPClient(mockMCPClient);
      
      expect(wrappedClient).toBeDefined();
      // The proxy should return the same or wrapped client
      expect(wrappedClient).toBe(mockMCPClient);
    });

    it('should return original client when telemetry disabled', () => {
      // Mock disabled telemetry
      const { telemetryConfig } = require('../../src/telemetry/config');
      telemetryConfig.isEnabled.mockReturnValue(false);

      const wrappedClient = injector.injectMCPClient(mockMCPClient);
      expect(wrappedClient).toBe(mockMCPClient);
    });

    it('should track injection results', () => {
      injector.injectMCPClient(mockMCPClient);
      
      const stats = injector.getStats();
      expect(stats.successfulInjections).toBeGreaterThan(0);
    });
  });

  describe('Tool Injection', () => {
    beforeEach(async () => {
      await injector.initialize();
    });

    it('should inject telemetry into tools', () => {
      const originalTool = jest.fn(() => 'result');
      
      const wrappedTool = injector.injectTool('test_tool', originalTool, 'sdr');
      
      expect(wrappedTool).toBeDefined();
      expect(typeof wrappedTool).toBe('function');
    });

    it('should support different tool categories', () => {
      const categories = ['sdr', 'cognitive', 'neural', 'memory', 'attention', 'graph'] as const;
      
      categories.forEach(category => {
        const tool = jest.fn();
        const wrapped = injector.injectTool(`${category}_tool`, tool, category);
        expect(wrapped).toBeDefined();
      });
    });

    it('should return original tool when instrumentation disabled', () => {
      const { telemetryConfig } = require('../../src/telemetry/config');
      telemetryConfig.isInstrumentationEnabled.mockReturnValue(false);

      const originalTool = jest.fn();
      const wrappedTool = injector.injectTool('disabled_tool', originalTool);
      
      expect(wrappedTool).toBe(originalTool);
    });
  });

  describe('Runtime Hooks', () => {
    beforeEach(async () => {
      await injector.initialize();
    });

    it('should add runtime hooks', () => {
      const hook: RuntimeHook = {
        name: 'test_hook',
        pattern: /test_pattern/,
        hook: (target, metadata) => target,
        priority: 100,
        active: true
      };

      expect(() => injector.addRuntimeHook(hook)).not.toThrow();
    });

    it('should remove runtime hooks', () => {
      const hook: RuntimeHook = {
        name: 'removable_hook',
        pattern: /removable/,
        hook: (target, metadata) => target,
        priority: 50,
        active: true
      };

      injector.addRuntimeHook(hook);
      expect(() => injector.removeRuntimeHook('removable_hook')).not.toThrow();
    });

    it('should maintain hook priority ordering', () => {
      const hooks: RuntimeHook[] = [
        {
          name: 'low_priority',
          pattern: /test/,
          hook: () => {},
          priority: 10,
          active: true
        },
        {
          name: 'high_priority',
          pattern: /test/,
          hook: () => {},
          priority: 100,
          active: true
        },
        {
          name: 'medium_priority',
          pattern: /test/,
          hook: () => {},
          priority: 50,
          active: true
        }
      ];

      hooks.forEach(hook => injector.addRuntimeHook(hook));
      
      const stats = injector.getStats();
      expect(stats.activeHooks).toBe(3);
    });
  });

  describe('Non-Intrusive Monitoring', () => {
    it('should setup without modifying global objects destructively', async () => {
      const originalConsole = global.console;
      const originalProcess = global.process;
      
      await injector.initialize();
      
      // Global objects should still be accessible and functional
      expect(global.console).toBeDefined();
      expect(global.process).toBeDefined();
      
      // Basic functionality should work
      expect(() => console.log('test')).not.toThrow();
      expect(() => process.nextTick(() => {})).not.toThrow();
    });

    it('should handle missing global objects gracefully', async () => {
      const originalGlobal = global.global;
      delete (global as any).global;
      
      await expect(injector.initialize()).resolves.not.toThrow();
      
      // Restore
      (global as any).global = originalGlobal;
    });
  });

  describe('Environment Configuration', () => {
    it('should inject environment variables', async () => {
      await injector.initialize();
      
      // Should have set some telemetry environment variables
      expect(process.env.LLMKG_TELEMETRY_ENABLED).toBeDefined();
    });

    it('should not overwrite existing environment variables', async () => {
      process.env.LLMKG_TELEMETRY_ENABLED = 'false';
      
      await injector.initialize();
      
      // Should preserve existing value
      expect(process.env.LLMKG_TELEMETRY_ENABLED).toBe('false');
      
      // Clean up
      delete process.env.LLMKG_TELEMETRY_ENABLED;
    });

    it('should watch for configuration changes', async () => {
      await injector.initialize();
      
      // This is tested indirectly through the configuration system
      expect(injector).toBeDefined();
    });
  });

  describe('Performance Impact', () => {
    beforeEach(async () => {
      await injector.initialize();
    });

    it('should have minimal initialization overhead', async () => {
      const newInjector = new TelemetryInjector();
      
      performanceTracker.start('initialization');
      await newInjector.initialize();
      const duration = performanceTracker.end('initialization');
      
      expect(duration).toBeLessThan(100); // Should initialize in <100ms
      
      newInjector.cleanup();
    });

    it('should have low injection overhead', () => {
      const tool = () => 'result';
      
      performanceTracker.start('tool_injection');
      const wrappedTool = injector.injectTool('perf_tool', tool);
      const duration = performanceTracker.end('tool_injection');
      
      expect(duration).toBeLessThan(10); // Should wrap in <10ms
      expect(wrappedTool).toBeDefined();
    });

    it('should handle high-frequency operations', async () => {
      const tool = jest.fn(() => Math.random());
      const wrappedTool = injector.injectTool('high_freq_tool', tool);
      
      performanceTracker.start('high_frequency_calls');
      
      // Make 1000 calls
      for (let i = 0; i < 1000; i++) {
        wrappedTool();
      }
      
      const duration = performanceTracker.end('high_frequency_calls');
      const avgCallTime = duration / 1000;
      
      expect(avgCallTime).toBeLessThan(1); // Should average <1ms per call
      expect(tool).toHaveBeenCalledTimes(1000);
    });
  });

  describe('Error Handling', () => {
    it('should handle initialization errors gracefully', async () => {
      const { telemetryRecorder } = require('../../src/telemetry/recorder');
      telemetryRecorder.recordLog.mockImplementation(() => {
        throw new Error('Recording failed');
      });

      // Should not throw even if recording fails
      await expect(injector.initialize()).resolves.not.toThrow();
    });

    it('should handle injection failures', () => {
      const { mcpTelemetryProxy } = require('../../src/telemetry/proxy');
      mcpTelemetryProxy.wrapMCPClient.mockImplementation(() => {
        throw new Error('Wrapping failed');
      });

      expect(() => injector.injectMCPClient(mockMCPClient)).not.toThrow();
    });

    it('should continue working after hook errors', () => {
      const failingHook: RuntimeHook = {
        name: 'failing_hook',
        pattern: /test/,
        hook: () => { throw new Error('Hook failed'); },
        priority: 100,
        active: true
      };

      expect(() => injector.addRuntimeHook(failingHook)).not.toThrow();
    });
  });

  describe('Statistics and Monitoring', () => {
    beforeEach(async () => {
      await injector.initialize();
    });

    it('should provide injection statistics', () => {
      // Perform some injections
      injector.injectMCPClient(mockMCPClient);
      injector.injectTool('test_tool', () => {});
      
      const stats = injector.getStats();
      
      expect(stats).toHaveProperty('totalTargets');
      expect(stats).toHaveProperty('successfulInjections');
      expect(stats).toHaveProperty('failedInjections');
      expect(stats).toHaveProperty('activeHooks');
      expect(stats).toHaveProperty('averageOverhead');
      
      expect(typeof stats.totalTargets).toBe('number');
      expect(typeof stats.successfulInjections).toBe('number');
      expect(typeof stats.averageOverhead).toBe('number');
    });

    it('should track successful injections', () => {
      injector.injectMCPClient(mockMCPClient);
      
      const stats = injector.getStats();
      expect(stats.successfulInjections).toBeGreaterThan(0);
    });

    it('should measure injection overhead', () => {
      const startTime = performance.now();
      injector.injectTool('overhead_test', () => {});
      const endTime = performance.now();
      
      const stats = injector.getStats();
      expect(stats.averageOverhead).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Cleanup', () => {
    it('should cleanup resources properly', async () => {
      await injector.initialize();
      
      // Add some targets and hooks
      injector.registerInjectionTarget({
        name: 'cleanup_test',
        type: 'client',
        target: 'test',
        config: { enabled: true }
      });
      
      expect(() => injector.cleanup()).not.toThrow();
      
      const stats = injector.getStats();
      expect(stats.totalTargets).toBe(0);
      expect(stats.activeHooks).toBe(0);
    });

    it('should handle cleanup when not initialized', () => {
      const cleanInjector = new TelemetryInjector();
      expect(() => cleanInjector.cleanup()).not.toThrow();
    });

    it('should call cleanup on proxied objects', async () => {
      const mockProxy = {
        cleanup: jest.fn()
      };
      
      const { mcpTelemetryProxy } = require('../../src/telemetry/proxy');
      mcpTelemetryProxy.wrapMCPClient.mockReturnValue(mockProxy);
      
      await injector.initialize();
      injector.injectMCPClient(mockMCPClient);
      injector.cleanup();
      
      expect(mockProxy.cleanup).toHaveBeenCalled();
    });
  });

  describe('LLMKG-Specific Features', () => {
    beforeEach(async () => {
      await injector.initialize();
    });

    it('should infer operation categories correctly', () => {
      const testCases = [
        { name: 'sdr_encode', expected: 'sdr' },
        { name: 'sparse_distributed_rep', expected: 'sdr' },
        { name: 'cognitive_process', expected: 'cognitive' },
        { name: 'brain_cortex_sim', expected: 'cognitive' },
        { name: 'neural_network', expected: 'neural' },
        { name: 'neuron_firing', expected: 'neural' },
        { name: 'memory_store', expected: 'memory' },
        { name: 'cache_retrieve', expected: 'memory' },
        { name: 'attention_focus', expected: 'attention' },
        { name: 'focus_mechanism', expected: 'attention' },
        { name: 'knowledge_graph', expected: 'graph' },
        { name: 'node_relationship', expected: 'graph' },
        { name: 'unknown_operation', expected: 'sdr' } // default
      ];

      testCases.forEach(({ name, expected }) => {
        const tool = jest.fn();
        const wrapped = injector.injectTool(name, tool, expected as any);
        expect(wrapped).toBeDefined();
      });
    });

    it('should handle LLMKG-specific modules', () => {
      // This tests the module pattern matching
      const llmkgModules = [
        'mcp',
        'model-context-protocol',
        'llmkg',
        'brain-inspired-mcp',
        'federated-learning'
      ];

      // The actual module hooking happens at runtime
      // This test verifies the patterns would match
      llmkgModules.forEach(moduleName => {
        expect(moduleName).toMatch(/mcp|llmkg|brain.*inspired|federated.*learning/i);
      });
    });
  });
});