# Task 33i: Create Integrity Test Runner

**Estimated Time**: 3 minutes  
**Dependencies**: 33h  
**Stage**: Data Integrity Testing  

## Objective
Create a test runner script for all data integrity tests.

## Implementation Steps

1. Create `scripts/run_integrity_tests.sh`:
```bash
#!/bin/bash
set -e

echo "Running data integrity tests..."

echo "Testing referential integrity..."
cargo test --test referential_integrity_test --release

echo "Testing property consistency..."
cargo test --test property_consistency_test --release

echo "Testing cache consistency..."
cargo test --test cache_consistency_test --release

echo "Testing temporal versioning..."
cargo test --test temporal_integrity_test --release

echo "Testing Phase 2 synchronization..."
cargo test --test phase2_sync_test --release

echo "Testing error recovery..."
cargo test --test error_recovery_test --release

echo "All data integrity tests passed! ✅"
```

2. Create comprehensive integrity validator:
```rust
// tests/integrity/comprehensive_validator.rs
pub struct IntegrityTestSuite {
    pub referential_tests: u32,
    pub consistency_tests: u32,
    pub temporal_tests: u32,
    pub sync_tests: u32,
    pub recovery_tests: u32,
}

impl IntegrityTestSuite {
    pub async fn run_all_tests(&self) -> IntegrityTestResults {
        let mut results = IntegrityTestResults::new();
        
        // Run all test categories
        results.referential_integrity = self.run_referential_tests().await;
        results.property_consistency = self.run_property_tests().await;
        results.cache_consistency = self.run_cache_tests().await;
        results.temporal_integrity = self.run_temporal_tests().await;
        results.phase2_synchronization = self.run_sync_tests().await;
        results.error_recovery = self.run_recovery_tests().await;
        
        results
    }
    
    pub fn generate_summary_report(&self, results: &IntegrityTestResults) {
        println!("
=== Data Integrity Test Summary ===");
        println!("✅ Referential Integrity: {}", if results.referential_integrity { "PASSED" } else { "FAILED" });
        println!("✅ Property Consistency: {}", if results.property_consistency { "PASSED" } else { "FAILED" });
        println!("✅ Cache Consistency: {}", if results.cache_consistency { "PASSED" } else { "FAILED" });
        println!("✅ Temporal Integrity: {}", if results.temporal_integrity { "PASSED" } else { "FAILED" });
        println!("✅ Phase 2 Synchronization: {}", if results.phase2_synchronization { "PASSED" } else { "FAILED" });
        println!("✅ Error Recovery: {}", if results.error_recovery { "PASSED" } else { "FAILED" });
        
        let total_passed = results.count_passed();
        println!("\nOverall: {}/6 test categories passed", total_passed);
        
        if total_passed == 6 {
            println!("\n✅ All data integrity requirements validated!");
        } else {
            println!("\n⚠️ Some integrity tests failed - review required");
        }
    }
}

pub struct IntegrityTestResults {
    pub referential_integrity: bool,
    pub property_consistency: bool,
    pub cache_consistency: bool,
    pub temporal_integrity: bool,
    pub phase2_synchronization: bool,
    pub error_recovery: bool,
}

impl IntegrityTestResults {
    pub fn new() -> Self {
        Self {
            referential_integrity: false,
            property_consistency: false,
            cache_consistency: false,
            temporal_integrity: false,
            phase2_synchronization: false,
            error_recovery: false,
        }
    }
    
    pub fn count_passed(&self) -> u32 {
        let mut count = 0;
        if self.referential_integrity { count += 1; }
        if self.property_consistency { count += 1; }
        if self.cache_consistency { count += 1; }
        if self.temporal_integrity { count += 1; }
        if self.phase2_synchronization { count += 1; }
        if self.error_recovery { count += 1; }
        count
    }
}
```

## Acceptance Criteria
- [ ] Test runner script created
- [ ] All integrity tests execute in sequence
- [ ] Comprehensive summary report generated

## Success Metrics
- All tests complete successfully
- Total execution time under 3 minutes
- Clear pass/fail indicators

## Next Task
33j_finalize_data_integrity_tests.md