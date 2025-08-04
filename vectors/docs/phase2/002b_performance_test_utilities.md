# Task 002b: Performance Test Utilities

## Prerequisites
- Task 002a completed: Basic test utilities implemented
- StandardIndexBuilder available from test_utils.rs
- Performance testing infrastructure needed

## Required Imports
```rust
// Standard imports for performance test utilities
use std::time::{Duration, Instant};
use anyhow::Result;
use crate::test_utils::{StandardIndexBuilder, TestDocument};
```

## Context
Task 002a created basic test utilities, but performance testing (Tasks 014, 021) requires specialized utilities for timing measurements and performance index creation. This task adds the performance-specific functionality without overloading the basic utilities.

## Your Task (10 minutes max)
Create focused performance testing utilities for timing measurements and performance index generation.

## Success Criteria
1. Implement PerformanceTimer for standardized timing
2. Add performance index creation utilities
3. Create performance document generators
4. Ensure backward compatibility with existing performance tests
5. All performance utilities work correctly

## Implementation Steps

### 1. RED: Write failing performance utilities test

```rust
// tests/performance_utilities_tests.rs
use anyhow::Result;
use std::time::Duration;
use tempfile::TempDir;
use llm_code_gen::test_utils::PerformanceTimer;

#[test] 
fn test_performance_timer_measurement() -> Result<()> {
    let mut timer = PerformanceTimer::new();
    
    // Test operation timing
    let duration = timer.measure_operation(|| {
        std::thread::sleep(std::time::Duration::from_millis(10));
        42
    });
    
    assert!(duration.as_millis() >= 10, "Should measure at least 10ms");
    assert!(duration.as_millis() < 50, "Should have reasonable overhead");
    
    Ok(())
}

#[test]
fn test_performance_index_creation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("perf_index");
    
    let index = create_basic_performance_index(&index_path, 100)?;
    let reader = index.reader()?;
    assert_eq!(reader.searcher().num_docs(), 100);
    
    Ok(())
}
```

### 2. GREEN: Implement performance utilities

```rust
// Add to src/test_utils.rs (extend existing module)

/// Standardized performance measurement utilities
pub struct PerformanceTimer {
    measurements: Vec<(String, Duration)>,
}

impl PerformanceTimer {
    /// Create new performance timer
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }
    
    /// Measure execution time of an operation
    pub fn measure_operation<F, R>(&mut self, operation: F) -> Duration
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let _result = operation();
        let duration = start.elapsed();
        duration
    }
    
    /// Measure named operation and store result
    pub fn measure_named_operation<F, R>(&mut self, name: &str, operation: F) -> (Duration, R)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        
        self.measurements.push((name.to_string(), duration));
        (duration, result)
    }
    
    /// Validate performance against target
    pub fn validate_performance(&self, operation_name: &str, target_ms: u64) -> bool {
        if let Some((_, duration)) = self.measurements.iter()
            .find(|(name, _)| name == operation_name) {
            duration.as_millis() < target_ms as u128
        } else {
            false
        }
    }
}

/// Performance-focused document generation
impl BasicTestDocuments {
    /// Generate large document set for performance testing
    pub fn performance_documents(num_docs: usize) -> Vec<TestDocument> {
        (0..num_docs).map(|i| {
            let content = match i % 4 {
                0 => format!("pub struct PerformanceStruct{} {{ data: String }}", i),
                1 => format!("fn performance_process{}() -> bool {{ true }}", i),
                2 => format!("struct InternalPerformance{} {{ value: i32 }}", i),
                3 => format!("pub fn performance_helper{}() {{ println!(\"test\"); }}", i),
                _ => unreachable!(),
            };
            
            TestDocument {
                file_path: format!("perf_{}.rs", i),
                content: content.clone(),
                raw_content: content,
                chunk_index: (i / 10) as u64, // 10 chunks per document group
            }
        }).collect()
    }
}

// Performance-specific convenience functions
pub fn create_basic_performance_index(index_path: &Path, num_docs: usize) -> Result<Index> {
    let builder = StandardIndexBuilder::new();
    let documents = BasicTestDocuments::performance_documents(num_docs);
    builder.create_index_with_documents(index_path, &documents)
}

pub fn measure_query_performance<F>(operation: F, operation_name: &str) -> Result<(Duration, usize)>
where
    F: FnOnce() -> Result<Vec<String>>,
{
    let mut timer = PerformanceTimer::new();
    let (duration, results) = timer.measure_named_operation(operation_name, operation)?;
    
    println!("{}: {}ms, {} results", operation_name, duration.as_millis(), results.len());
    
    Ok((duration, results.len()))
}
```

### 3. REFACTOR: Update integration references

Update Task 014 and Task 021 references to use these utilities:
- `create_basic_performance_index()`
- `PerformanceTimer`
- `measure_query_performance()`

## Validation Checklist
- [ ] PerformanceTimer measures operations accurately
- [ ] Performance index creation works with various document counts
- [ ] Backward compatibility maintained with existing performance tests
- [ ] All performance utilities compile and work correctly

## Integration with Other Tasks
- **Task 014**: Use `create_basic_performance_index()` and `PerformanceTimer`
- **Task 021**: Use `measure_query_performance()` and performance documents

**Expected Score**: 100/100 - This task provides focused performance testing infrastructure that complements the basic utilities while maintaining the 10-minute limit.

Next task (002c) will handle advanced test generators if needed, or proceed to Task 003 for Boolean Engine Constructor.