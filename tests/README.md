# Phase 4 Test Suite

This directory contains comprehensive tests for the Phase 4 Self-Organization & Learning implementation of LLMKG.

## Test Structure

### 1. `phase4_realistic_tests.rs`
- **Purpose**: Core functionality tests with realistic expectations
- **Features**:
  - Hebbian learning with measurable weight changes
  - Homeostasis with actual stability effects
  - Optimization with rollback scenarios
  - Memory usage bounds verification
  - DeepSeek API integration

### 2. `phase4_deepseek_integration.rs`
- **Purpose**: Integration with DeepSeek LLM for enhanced testing
- **Features**:
  - Knowledge generation using LLM
  - Query quality evaluation
  - Adaptive learning with LLM feedback
  - Continuous learning loops
  - Performance benchmarking

### 3. `phase4_comprehensive_tests.rs`
- **Purpose**: Original comprehensive test coverage
- **Note**: Uses test helpers, some functionality may be stubbed

### 4. `phase4_advanced_stress_tests.rs`
- **Purpose**: High-load and edge case testing
- **Features**:
  - 5000+ activation event processing
  - Chaotic stability testing
  - Complex refactoring scenarios
  - Emergency adaptation
  - Meta-learning capabilities

### 5. `phase4_integration_scenarios.rs`
- **Purpose**: Real-world usage scenarios
- **Scenarios**:
  - AI Research Assistant
  - Multi-User Collaborative Learning
  - Performance Degradation Recovery
  - Long-term Knowledge Retention
  - Adaptive Response Generation

### 6. `phase4_test_utils.rs`
- **Purpose**: Shared testing utilities
- **Utilities**:
  - Test metrics collection
  - Graph validation
  - Test data builders
  - Performance profiling
  - Assertion macros

## Running the Tests

### Prerequisites
1. Create a `.env` file in the project root:
```env
DEEPSEEK_API_KEY=sk-a67cb9f8a3d741d086bcfd0760de7ad6
DEEPSEEK_API_URL=https://api.deepseek.com/v1
```

2. Ensure Rust and Cargo are installed

### Run All Tests (Windows PowerShell)
```powershell
.\tests\run_phase4_tests.ps1
```

### Run All Tests (Linux/macOS)
```bash
./tests/run_phase4_tests.sh
```

### Run Specific Test Suite
```bash
# Run realistic tests only
cargo test --test phase4_realistic_tests

# Run DeepSeek integration tests
cargo test --test phase4_deepseek_integration

# Run with verbose output
cargo test --test phase4_realistic_tests -- --nocapture
```

### Include Stress Tests
```powershell
# PowerShell
.\tests\run_phase4_tests.ps1 --include-stress

# Bash
./tests/run_phase4_tests.sh --include-stress
```

## Test Configuration

Environment variables (set in `.env`):
- `MIN_LEARNING_EFFICIENCY`: Minimum acceptable learning efficiency (default: 0.15)
- `MIN_CONFIDENCE_THRESHOLD`: Minimum query confidence (default: 0.5)
- `MAX_MEMORY_OVERHEAD_PERCENT`: Maximum memory increase (default: 20%)
- `SYNTHETIC_ENTITY_COUNT`: Number of entities for tests (default: 100)

## Key Improvements from Original Tests

1. **Realistic Assertions**: Tests verify actual behavior changes, not just positive values
2. **No Phantom Methods**: Only uses actual API methods that exist in the implementation
3. **Proper Error Handling**: Tests handle both success and failure cases
4. **Measurable Metrics**: Performance and learning effectiveness are measured, not assumed
5. **DeepSeek Integration**: Uses real LLM for knowledge generation and quality evaluation
6. **Resource Management**: Tests verify memory usage stays within bounds
7. **Time-based Validation**: Avoids flaky time-based assertions

## Debugging Failed Tests

1. Check the `.env` file exists and contains valid API key
2. Run with `--nocapture` to see detailed output
3. Check test logs for specific failure reasons
4. Verify network connectivity for DeepSeek API tests
5. Use `RUST_BACKTRACE=1` for detailed error traces

## Performance Benchmarks

Run benchmarks separately:
```bash
cargo test --test phase4_realistic_tests --release -- benchmark
```

Expected performance targets:
- Hebbian learning: < 10ms per event
- Query processing: < 500ms average
- Memory overhead: < 20% increase
- Learning efficiency: > 15% improvement