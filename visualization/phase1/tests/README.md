# LLMKG Visualization Phase 1 - Testing Infrastructure

Comprehensive testing infrastructure for the LLMKG visualization dashboard Phase 1 components, ensuring production-ready quality and performance validation.

## Overview

This testing suite provides complete coverage for all Phase 1 visualization components:

- **MCP Client** - Model Context Protocol client for LLMKG communication
- **Data Collection Agents** - Specialized collectors for cognitive patterns, neural activity, memory systems
- **Telemetry Injection System** - Non-intrusive telemetry and instrumentation
- **WebSocket Communication** - High-performance real-time data streaming

## Test Structure

```
tests/
├── config/           # Test configuration and utilities
├── unit/             # Component unit tests
├── integration/      # System integration tests
├── performance/      # Performance and load tests
├── mocks/           # Mock servers and data generators
├── e2e/             # End-to-end pipeline tests
└── README.md        # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)

Individual component testing with >90% code coverage:

- `mcp-client.test.ts` - MCP client functionality, connection management, data generation
- `base-collector.test.ts` - Base collector features, buffering, aggregation, performance
- `telemetry-injector.test.ts` - Telemetry injection, runtime hooks, non-intrusive monitoring
- `websocket-server.test.ts` - WebSocket server, message handling, client management

### Integration Tests (`tests/integration/`)

System-level component interaction testing:

- `system-integration.test.ts` - Complete data pipeline testing, component coordination, error handling

### Performance Tests (`tests/performance/`)

Performance validation against LLMKG requirements:

- `latency-performance.test.ts` - <100ms latency requirement validation
- `throughput-performance.test.ts` - >1000 messages/second throughput validation

### End-to-End Tests (`tests/e2e/`)

Complete pipeline testing with realistic LLMKG scenarios:

- `complete-pipeline.test.ts` - Full data flow from LLMKG through visualization streaming

### Mock Infrastructure (`tests/mocks/`)

Realistic test doubles for LLMKG components:

- `brain-inspired-server.ts` - Mock BrainInspiredMCPServer with cognitive operations

### Test Configuration (`tests/config/`)

Shared testing utilities and setup:

- `jest-setup.ts` - Global test configuration, mocks, custom matchers
- `test-utils.ts` - Performance tracking, data generators, test helpers

## Running Tests

### All Tests
```bash
npm test
```

### Test Categories
```bash
# Unit tests
npm run test:unit

# Integration tests  
npm run test:integration

# Performance tests (with extended timeout)
npm run test:performance

# End-to-end tests (with extended timeout)
npm run test:e2e

# Coverage report
npm run test:coverage

# CI/CD mode
npm run test:ci
```

### Watch Mode
```bash
npm run test:watch
```

## Key Features

### LLMKG-Specific Testing

- **Cognitive Pattern Data** - Validates cortical region processing, activation levels, neural connections
- **SDR Operations** - Tests Sparse Distributed Representation encoding/decoding with proper sparsity
- **Neural Activity** - Validates firing rates, membrane potentials, synaptic inputs
- **Memory Systems** - Tests episodic/semantic memory consolidation and retrieval
- **Attention Mechanisms** - Validates focus strength, salience maps, attention switching
- **Knowledge Graphs** - Tests graph operations, relationship queries, node updates

### Performance Requirements Validation

- **Latency Testing** - Validates <100ms end-to-end processing latency
- **Throughput Testing** - Validates >1000 messages/second system throughput
- **Memory Efficiency** - Monitors memory usage under sustained load
- **Scalability** - Tests performance with multiple concurrent components

### Real-World Scenarios

- **Component Failures** - Tests graceful degradation and recovery
- **Network Issues** - Validates reconnection and data persistence
- **Memory Pressure** - Tests performance under resource constraints
- **High Load** - Validates sustained performance under peak loads

## Mock LLMKG Server

The `MockBrainInspiredMCPServer` provides realistic simulation of LLMKG operations:

### Available Tools
- `sdr_encode` - Sparse Distributed Representation encoding
- `sdr_decode` - SDR semantic decoding  
- `cognitive_process` - Cortical region simulation
- `neural_simulate` - Neural network activity simulation
- `memory_store` - Memory consolidation operations
- `memory_retrieve` - Memory recall operations
- `attention_focus` - Attention mechanism control
- `graph_query` - Knowledge graph queries
- `graph_update` - Knowledge graph updates

### Configuration
```typescript
const mockServer = new MockBrainInspiredMCPServer();

// Configure processing delays
mockServer.configureProcessingDelay(1, 10); // 1-10ms range

// Enable/disable telemetry
mockServer.setTelemetryEnabled(true);

await mockServer.start();
```

## Performance Tracking

Built-in performance measurement utilities:

```typescript
import { PerformanceTracker } from '../config/test-utils';

const tracker = new PerformanceTracker();

tracker.start('operation_name');
// ... perform operation
const duration = tracker.end('operation_name');

const stats = tracker.getStats('operation_name');
// Returns: { count, average, min, max, p95, p99 }
```

## Data Generators

LLMKG-specific test data generation:

```typescript
import { LLMKGDataGenerator } from '../config/test-utils';

// Generate cognitive pattern data
const cognitiveData = LLMKGDataGenerator.generateCognitivePattern({
  cortical_region: 'prefrontal',
  activation_level: 0.8
});

// Generate SDR data
const sdrData = LLMKGDataGenerator.generateSDRData({
  size: 2048,
  sparsity: 0.02
});

// Generate mixed batch
const batch = LLMKGDataGenerator.generateMixedLLMKGBatch(100);
```

## Test Utilities

Helper functions for async testing:

```typescript
import { TestHelpers } from '../config/test-utils';

// Wait for condition
await TestHelpers.waitFor(() => condition, 5000);

// Measure execution time
const { result, duration } = await TestHelpers.measureTime(async () => {
  return await someOperation();
});

// Generate test data
const testData = TestHelpers.generateTestData('cognitive', 10);
```

## Custom Jest Matchers

Extended matchers for specialized testing:

```typescript
// Range validation
expect(latency).toBeWithinRange(0, 100);

// Async timing validation  
expect(mockFunction).toHaveBeenCalledWithinTime(1000);
```

## Coverage Requirements

- **Unit Tests**: >90% code coverage
- **Integration Tests**: All component interactions covered
- **Performance Tests**: All latency and throughput requirements validated
- **E2E Tests**: Complete data pipeline scenarios covered

## CI/CD Integration

Optimized for continuous integration:

```bash
# Run in CI environment
npm run test:ci

# Generates:
# - Test results in JUnit format
# - Coverage reports in LCOV format
# - Performance benchmarks
# - Error logs and diagnostics
```

## Debugging Tests

### Verbose Output
```bash
JEST_VERBOSE=true npm test
```

### Debug Specific Test
```bash
npm test -- --testNamePattern="should achieve low latency"
```

### Coverage Analysis
```bash
npm run test:coverage
# Opens coverage/lcov-report/index.html
```

## Performance Benchmarks

Expected performance characteristics:

- **MCP Client**: Data generation <10ms latency, >100 Hz frequency
- **Collectors**: Processing <20ms latency, >50 Hz collection rate  
- **WebSocket**: Message broadcast <5ms latency, >2000 messages/second
- **End-to-End**: Complete pipeline <100ms latency, >1000 operations/second

## Error Scenarios

Comprehensive error condition testing:

- Network disconnections and reconnections
- Memory pressure and resource constraints  
- Component failures and recovery
- Invalid data handling
- Concurrent access conflicts
- Performance degradation detection

## Best Practices

1. **Test Isolation** - Each test runs independently with clean setup/teardown
2. **Realistic Data** - Use LLMKG-specific data patterns and operations
3. **Performance Focus** - All tests validate latency and throughput requirements
4. **Error Resilience** - Test failure modes and recovery scenarios
5. **Documentation** - Clear test descriptions and expected behaviors

## Contributing

When adding new tests:

1. Follow existing patterns in each test category
2. Use the provided utilities and mock infrastructure
3. Validate both functionality and performance requirements
4. Include error scenarios and edge cases
5. Update documentation for new test utilities

## Support

For questions about the testing infrastructure:

1. Review existing test patterns in each category
2. Check the mock infrastructure for LLMKG simulation
3. Use the provided performance tracking and data generation utilities
4. Follow the established patterns for async testing and error handling