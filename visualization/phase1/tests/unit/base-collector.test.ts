/**
 * Unit tests for Base Data Collector
 * 
 * Tests the abstract base collector implementation including buffering,
 * aggregation, event handling, and high-frequency data processing.
 */

import { BaseCollector, CollectorConfig, CollectedData, CircularBuffer, DataAggregator } from '../../src/collectors/base';
import { MockMCPClient, TestCollector, PerformanceTracker, TestHelpers } from '../config/test-utils';

describe('BaseCollector', () => {
  let mcpClient: MockMCPClient;
  let collector: TestCollector;
  let performanceTracker: PerformanceTracker;

  beforeEach(() => {
    mcpClient = new MockMCPClient();
    performanceTracker = new PerformanceTracker();
  });

  afterEach(async () => {
    if (collector && collector.isRunning()) {
      await collector.stop();
    }
    performanceTracker.clear();
  });

  describe('Initialization', () => {
    it('should create collector with default configuration', () => {
      collector = new TestCollector(mcpClient);
      expect(collector).toBeInstanceOf(BaseCollector);
      expect(collector.getStats().totalCollected).toBe(0);
    });

    it('should create collector with custom configuration', () => {
      const config: Partial<CollectorConfig> = {
        name: 'custom-collector',
        collectionInterval: 50,
        bufferSize: 5000,
        maxBufferAge: 30000,
        autoFlush: false,
        sampleRate: 0.5
      };

      collector = new TestCollector(mcpClient, config);
      expect(collector).toBeDefined();
    });

    it('should initialize with correct default values', () => {
      collector = new TestCollector(mcpClient);
      const stats = collector.getStats();
      
      expect(stats.totalCollected).toBe(0);
      expect(stats.successfulCollections).toBe(0);
      expect(stats.failedCollections).toBe(0);
      expect(stats.successRate).toBe(0);
    });
  });

  describe('CircularBuffer', () => {
    let buffer: CircularBuffer<number>;

    beforeEach(() => {
      buffer = new CircularBuffer<number>(5);
    });

    it('should add items correctly', () => {
      expect(buffer.add(1)).toBe(true);
      expect(buffer.add(2)).toBe(true);
      expect(buffer.getStats().size).toBe(2);
    });

    it('should handle buffer overflow', () => {
      // Fill buffer
      for (let i = 1; i <= 5; i++) {
        buffer.add(i);
      }
      expect(buffer.getStats().size).toBe(5);
      expect(buffer.isFull()).toBe(true);

      // Add one more (should overwrite oldest)
      buffer.add(6);
      expect(buffer.getStats().size).toBe(5);
      
      const items = buffer.peek();
      expect(items).toEqual([2, 3, 4, 5, 6]);
    });

    it('should peek items without removing them', () => {
      buffer.add(1);
      buffer.add(2);
      buffer.add(3);

      const peeked = buffer.peek(2);
      expect(peeked).toEqual([1, 2]);
      expect(buffer.getStats().size).toBe(3);
    });

    it('should drain items correctly', () => {
      buffer.add(1);
      buffer.add(2);
      buffer.add(3);

      const drained = buffer.drain(2);
      expect(drained).toEqual([1, 2]);
      expect(buffer.getStats().size).toBe(1);
    });

    it('should clear buffer correctly', () => {
      buffer.add(1);
      buffer.add(2);
      buffer.clear();
      
      expect(buffer.isEmpty()).toBe(true);
      expect(buffer.getStats().size).toBe(0);
    });
  });

  describe('DataAggregator', () => {
    let aggregator: DataAggregator;

    beforeEach(() => {
      aggregator = new DataAggregator();
    });

    it('should add values and calculate statistics', () => {
      aggregator.addValue(10);
      aggregator.addValue(20);
      aggregator.addValue(30);

      const stats = aggregator.getStatistics();
      expect(stats.count).toBe(3);
      expect(stats.mean).toBe(20);
      expect(stats.min).toBe(10);
      expect(stats.max).toBe(30);
      expect(stats.sum).toBe(60);
    });

    it('should calculate rate correctly', async () => {
      const startTime = Date.now();
      
      for (let i = 0; i < 10; i++) {
        aggregator.addValue(i, startTime + i * 100);
      }

      const rate = aggregator.getRate(1000);
      expect(rate).toBeGreaterThan(5); // Should be around 10 events/sec
    });

    it('should handle empty aggregator', () => {
      const stats = aggregator.getStatistics();
      expect(stats.count).toBe(0);
      expect(stats.mean).toBe(0);
      expect(stats.min).toBe(0);
      expect(stats.max).toBe(0);
    });

    it('should clear data correctly', () => {
      aggregator.addValue(100);
      aggregator.clear();
      
      const stats = aggregator.getStatistics();
      expect(stats.count).toBe(0);
    });
  });

  describe('Collection Process', () => {
    beforeEach(() => {
      collector = new TestCollector(mcpClient, {
        name: 'test-collector',
        collectionInterval: 50,
        autoFlush: false
      });
    });

    it('should start and stop collection', async () => {
      expect(collector.isRunning()).toBe(false);
      
      await collector.start();
      expect(collector.isRunning()).toBe(true);
      
      await collector.stop();
      expect(collector.isRunning()).toBe(false);
    });

    it('should collect data at specified intervals', async () => {
      const collectedSpy = jest.fn();
      collector.on('data:collected', collectedSpy);

      await collector.start();
      
      // Wait for multiple collections
      await TestHelpers.waitFor(() => collectedSpy.mock.calls.length >= 3, 1000);
      
      await collector.stop();

      expect(collectedSpy.mock.calls.length).toBeGreaterThanOrEqual(3);
    });

    it('should handle collection errors gracefully', async () => {
      const errorSpy = jest.fn();
      collector.on('collection:error', errorSpy);

      // Set up collector to throw error
      collector.setCollectFunction(async () => {
        throw new Error('Collection failed');
      });

      await collector.start();
      
      // Wait for error to occur
      await TestHelpers.waitFor(() => errorSpy.mock.calls.length > 0, 1000);
      
      await collector.stop();

      expect(errorSpy).toHaveBeenCalled();
      expect(collector.getStats().failedCollections).toBeGreaterThan(0);
    });

    it('should update statistics correctly', async () => {
      await collector.start();
      
      await TestHelpers.waitFor(() => collector.getStats().totalCollected > 0, 1000);
      
      await collector.stop();

      const stats = collector.getStats();
      expect(stats.totalCollected).toBeGreaterThan(0);
      expect(stats.successfulCollections).toBeGreaterThan(0);
      expect(stats.successRate).toBeGreaterThan(0);
    });

    it('should maintain performance metrics', async () => {
      await collector.start();
      
      await TestHelpers.waitFor(() => collector.getStats().totalCollected >= 5, 1000);
      
      await collector.stop();

      const stats = collector.getStats();
      expect(stats.averageProcessingTime).toBeGreaterThan(0);
      expect(stats.eventsPerSecond).toBeGreaterThan(0);
    });
  });

  describe('Data Validation', () => {
    beforeEach(() => {
      collector = new TestCollector(mcpClient);
    });

    it('should validate collected data correctly', async () => {
      const validData: CollectedData = {
        id: 'test-id',
        timestamp: Date.now(),
        source: 'test-source',
        type: 'test-type',
        data: { value: 123 },
        metadata: {
          collector: 'test-collector',
          method: 'test',
          tags: { test: 'true' }
        }
      };

      // Use reflection to access protected method
      const isValid = (collector as any).validateData(validData);
      expect(isValid).toBe(true);
    });

    it('should reject invalid data', async () => {
      const invalidDataSamples = [
        null,
        undefined,
        { id: 'test' }, // Missing required fields
        { id: 'test', timestamp: 'invalid' }, // Wrong type
        { 
          id: 'test',
          timestamp: Date.now(),
          source: 'test',
          type: 'test',
          data: {},
          metadata: null // Invalid metadata
        }
      ];

      for (const invalidData of invalidDataSamples) {
        const isValid = (collector as any).validateData(invalidData);
        expect(isValid).toBe(false);
      }
    });
  });

  describe('Buffering and Flushing', () => {
    beforeEach(() => {
      collector = new TestCollector(mcpClient, {
        name: 'test-collector',
        collectionInterval: 50,
        bufferSize: 10,
        autoFlush: true,
        flushInterval: 200
      });
    });

    it('should buffer collected data', async () => {
      await collector.start();
      
      await TestHelpers.waitFor(() => collector.getBufferContents().length > 0, 1000);
      
      const bufferContents = collector.getBufferContents();
      expect(bufferContents.length).toBeGreaterThan(0);
      
      await collector.stop();
    });

    it('should auto-flush buffered data', async () => {
      const flushedSpy = jest.fn();
      collector.on('data:flushed', flushedSpy);

      await collector.start();
      
      // Wait for auto-flush to occur
      await TestHelpers.waitFor(() => flushedSpy.mock.calls.length > 0, 1000);
      
      await collector.stop();

      expect(flushedSpy).toHaveBeenCalled();
    });

    it('should flush on buffer overflow', async () => {
      const flushedSpy = jest.fn();
      collector.on('data:flushed', flushedSpy);

      // Create collector with small buffer
      collector = new TestCollector(mcpClient, {
        name: 'test-collector',
        collectionInterval: 10, // Fast collection
        bufferSize: 3, // Small buffer
        autoFlush: false // Disable auto-flush to test overflow flush
      });
      collector.on('data:flushed', flushedSpy);

      await collector.start();
      
      // Wait for buffer overflow flush
      await TestHelpers.waitFor(() => flushedSpy.mock.calls.length > 0, 1000);
      
      await collector.stop();

      expect(flushedSpy).toHaveBeenCalled();
    });
  });

  describe('Sampling', () => {
    it('should respect sampling rate', async () => {
      // Create collector with 50% sampling rate
      collector = new TestCollector(mcpClient, {
        name: 'test-collector',
        collectionInterval: 20,
        sampleRate: 0.5,
        autoFlush: false
      });

      await collector.start();
      
      // Wait for collections
      await new Promise(resolve => setTimeout(resolve, 500));
      
      await collector.stop();

      const stats = collector.getStats();
      const expectedCollections = Math.floor(500 / 20); // 25 potential collections
      const actualCollections = stats.totalCollected;
      
      // With 50% sampling, should collect roughly half (allow for randomness)
      expect(actualCollections).toBeLessThan(expectedCollections * 0.8);
    });
  });

  describe('Performance Requirements', () => {
    it('should achieve high-frequency data collection (>10 Hz)', async () => {
      collector = new TestCollector(mcpClient, {
        name: 'high-freq-collector',
        collectionInterval: 10, // 100 Hz target
        autoFlush: false
      });

      performanceTracker.start('high_frequency_collection');
      
      await collector.start();
      
      // Run for 1 second
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      await collector.stop();
      
      const duration = performanceTracker.end('high_frequency_collection');
      const stats = collector.getStats();
      const frequency = stats.totalCollected / (duration / 1000);
      
      expect(frequency).toBeGreaterThan(10); // Should achieve >10 Hz
    });

    it('should maintain low processing latency', async () => {
      collector = new TestCollector(mcpClient, {
        name: 'latency-test-collector',
        collectionInterval: 50
      });

      await collector.start();
      
      await TestHelpers.waitFor(() => collector.getStats().totalCollected >= 10, 2000);
      
      await collector.stop();

      const stats = collector.getStats();
      expect(stats.averageProcessingTime).toBeLessThan(10); // <10ms processing time
    });

    it('should handle memory efficiently', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      collector = new TestCollector(mcpClient, {
        name: 'memory-test-collector',
        collectionInterval: 10,
        bufferSize: 1000,
        maxMemoryUsage: 50 // 50MB limit
      });

      await collector.start();
      
      // Run for a while to generate data
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      await collector.stop();

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = (finalMemory - initialMemory) / (1024 * 1024); // Convert to MB
      
      expect(memoryIncrease).toBeLessThan(50); // Should stay under memory limit
    });
  });

  describe('Event System', () => {
    beforeEach(() => {
      collector = new TestCollector(mcpClient, {
        name: 'event-test-collector',
        collectionInterval: 100
      });
    });

    it('should emit lifecycle events', async () => {
      const startedSpy = jest.fn();
      const stoppedSpy = jest.fn();
      
      collector.on('started', startedSpy);
      collector.on('stopped', stoppedSpy);

      await collector.start();
      await collector.stop();

      expect(startedSpy).toHaveBeenCalled();
      expect(stoppedSpy).toHaveBeenCalled();
    });

    it('should emit data collection events', async () => {
      const collectedSpy = jest.fn();
      const telemetrySpy = jest.fn();
      
      collector.on('data:collected', collectedSpy);
      collector.on('telemetry', telemetrySpy);

      await collector.start();
      
      await TestHelpers.waitFor(() => collectedSpy.mock.calls.length > 0, 1000);
      
      await collector.stop();

      expect(collectedSpy).toHaveBeenCalled();
    });

    it('should emit configuration events', () => {
      const configuredSpy = jest.fn();
      collector.on('configured', configuredSpy);

      collector.configure({
        collectionInterval: 200,
        bufferSize: 5000
      });

      expect(configuredSpy).toHaveBeenCalled();
    });
  });

  describe('Health Monitoring', () => {
    beforeEach(() => {
      collector = new TestCollector(mcpClient);
    });

    it('should provide health status', async () => {
      await collector.start();
      
      await TestHelpers.waitFor(() => collector.getStats().totalCollected > 0, 1000);
      
      const health = collector.getHealthStatus();
      
      expect(health).toHaveProperty('isActive');
      expect(health).toHaveProperty('health');
      expect(health).toHaveProperty('stats');
      expect(health).toHaveProperty('uptime');
      expect(health.isActive).toBe(true);
      expect(['healthy', 'warning', 'critical']).toContain(health.health);
      
      await collector.stop();
    });

    it('should detect unhealthy conditions', async () => {
      // Set up collector to fail frequently
      collector.setCollectFunction(async () => {
        if (Math.random() < 0.8) { // 80% failure rate
          throw new Error('Simulated failure');
        }
        return [];
      });

      await collector.start();
      
      // Wait for failures to accumulate
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const health = collector.getHealthStatus();
      expect(['warning', 'critical']).toContain(health.health);
      
      await collector.stop();
    });
  });
});