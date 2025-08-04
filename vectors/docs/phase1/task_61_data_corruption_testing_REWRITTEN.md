# Task 61: Implement Data Corruption Detection and Recovery System

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 60 completed - Edge case handling system implemented
**Input Files:** 
- C:/code/LLMKG/vectors/tantivy_search/src/error_handling/mod.rs
- C:/code/LLMKG/vectors/tantivy_search/src/indexer.rs
- C:/code/LLMKG/vectors/tantivy_search/src/health_check.rs

## Complete Context (For AI with ZERO Knowledge)

You are implementing a data corruption detection system for the production-ready Tantivy search engine. **Data corruption** occurs when index files or document data become damaged due to hardware failures, interrupted writes, or filesystem issues.

**Why this matters:** Production search systems must detect corrupted data early and recover automatically to maintain service availability.

**This Task:** Creates a corruption detector that validates index integrity, detects damaged chunks, and triggers automatic recovery procedures.

**System Integration:** This builds on the health check system (Task 52) and error handling framework (Task 43) to provide enterprise-grade data reliability.

## Exact Steps (6 minutes implementation)

### Step 1: Create Corruption Detection Module (2 minutes)
Create `C:/code/LLMKG/vectors/tantivy_search/src/corruption_detector.rs`:

```rust
use std::path::Path;
use std::fs;
use anyhow::{Result, anyhow};
use tantivy::{Index, IndexReader};
use sha2::{Sha256, Digest};
use crate::error_handling::SearchError;

#[derive(Debug, Clone)]
pub struct CorruptionDetector {
    checksum_file: String,
    recovery_enabled: bool,
}

#[derive(Debug)]
pub struct CorruptionReport {
    pub corrupted_files: Vec<String>,
    pub invalid_checksums: Vec<String>,
    pub recovery_actions: Vec<String>,
    pub severity: CorruptionSeverity,
}

#[derive(Debug, PartialEq)]
pub enum CorruptionSeverity {
    Minor,    // Single chunk corrupted, search still works
    Major,    // Multiple chunks corrupted, degraded performance
    Critical, // Index unusable, requires full rebuild
}

impl CorruptionDetector {
    pub fn new(index_path: &str) -> Self {
        Self {
            checksum_file: format!("{}/{}", index_path, ".checksums"),
            recovery_enabled: true,
        }
    }
    
    pub fn detect_corruption(&self, index: &Index) -> Result<CorruptionReport> {
        let mut report = CorruptionReport {
            corrupted_files: Vec::new(),
            invalid_checksums: Vec::new(),
            recovery_actions: Vec::new(),
            severity: CorruptionSeverity::Minor,
        };
        
        // Check index file integrity
        self.verify_index_files(index, &mut report)?;
        
        // Check document chunk integrity  
        self.verify_document_chunks(index, &mut report)?;
        
        // Determine severity based on findings
        self.assess_severity(&mut report);
        
        Ok(report)
    }
    
    fn verify_index_files(&self, index: &Index, report: &mut CorruptionReport) -> Result<()> {
        // Check if index can be opened and read
        match index.reader() {
            Ok(reader) => {
                // Try to access schema - this will fail if corrupted
                let _schema = index.schema();
                
                // Try to count documents - this will fail if segment files corrupted
                let _doc_count = reader.searcher().num_docs();
            },
            Err(e) => {
                report.corrupted_files.push("index_segments".to_string());
                report.recovery_actions.push("rebuild_index".to_string());
            }
        }
        Ok(())
    }
    
    fn verify_document_chunks(&self, index: &Index, report: &mut CorruptionReport) -> Result<()> {
        let reader = index.reader()?;
        let searcher = reader.searcher();
        
        // Sample 10% of documents to check for corruption
        let total_docs = searcher.num_docs() as usize;
        let sample_size = std::cmp::max(1, total_docs / 10);
        
        for doc_id in (0..total_docs).step_by(total_docs / sample_size) {
            match searcher.doc(tantivy::DocId(doc_id as u32)) {
                Ok(doc) => {
                    // Document accessible - check if content makes sense
                    if let Some(content) = doc.get_first(index.schema().get_field("content").unwrap()) {
                        if content.as_text().unwrap_or("").is_empty() {
                            report.corrupted_files.push(format!("doc_{}", doc_id));
                        }
                    }
                },
                Err(_) => {
                    report.corrupted_files.push(format!("doc_{}", doc_id));
                }
            }
        }
        
        Ok(())
    }
    
    fn assess_severity(&self, report: &mut CorruptionReport) {
        let corruption_count = report.corrupted_files.len();
        
        report.severity = match corruption_count {
            0 => CorruptionSeverity::Minor,
            1..=5 => CorruptionSeverity::Minor,
            6..=20 => CorruptionSeverity::Major,
            _ => CorruptionSeverity::Critical,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tantivy::{schema::*, Index};
    
    #[test]
    fn test_corruption_detector_creation() {
        let detector = CorruptionDetector::new("/tmp/test");
        assert_eq!(detector.checksum_file, "/tmp/test/.checksums");
        assert!(detector.recovery_enabled);
    }
    
    #[test]
    fn test_healthy_index_detection() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path();
        
        // Create healthy index
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("content", TEXT | STORED);
        let schema = schema_builder.build();
        let index = Index::create_in_dir(index_path, schema).unwrap();
        
        let detector = CorruptionDetector::new(index_path.to_str().unwrap());
        let report = detector.detect_corruption(&index).unwrap();
        
        assert_eq!(report.severity, CorruptionSeverity::Minor);
        assert!(report.corrupted_files.is_empty());
    }
}
```

### Step 2: Integrate with Health Check System (2 minutes)
Add to `C:/code/LLMKG/vectors/tantivy_search/src/health_check.rs`:

```rust
use crate::corruption_detector::{CorruptionDetector, CorruptionSeverity};

impl HealthChecker {
    pub fn check_data_integrity(&self, index: &Index) -> Result<HealthStatus> {
        let detector = CorruptionDetector::new(&self.index_path);
        
        match detector.detect_corruption(index) {
            Ok(report) => {
                match report.severity {
                    CorruptionSeverity::Minor => Ok(HealthStatus::Healthy),
                    CorruptionSeverity::Major => {
                        self.log_warning(&format!("Data corruption detected: {} files affected", 
                                                report.corrupted_files.len()));
                        Ok(HealthStatus::Degraded)
                    },
                    CorruptionSeverity::Critical => {
                        self.log_error("Critical data corruption - index unusable");
                        Ok(HealthStatus::Unhealthy)
                    }
                }
            },
            Err(e) => {
                self.log_error(&format!("Corruption detection failed: {}", e));
                Ok(HealthStatus::Unhealthy)
            }
        }
    }
}
```

### Step 3: Add to Module Exports (1 minute)
Add to `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs`:

```rust
pub mod corruption_detector;
```

### Step 4: Integration Test (1 minute)
Create `C:/code/LLMKG/vectors/tantivy_search/tests/corruption_integration_test.rs`:

```rust
use tantivy_search::corruption_detector::{CorruptionDetector, CorruptionSeverity};
use tantivy_search::health_check::HealthChecker;
use tempfile::TempDir;
use tantivy::{schema::*, Index};

#[test]
fn test_corruption_detection_integration() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path();
    
    // Create index
    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("content", TEXT | STORED);
    let schema = schema_builder.build();
    let index = Index::create_in_dir(index_path, schema).unwrap();
    
    // Test detection
    let health_checker = HealthChecker::new(index_path.to_str().unwrap().to_string());
    let status = health_checker.check_data_integrity(&index).unwrap();
    
    // Should be healthy for new index
    assert!(matches!(status, tantivy_search::health_check::HealthStatus::Healthy));
}
```

## Verification Steps (2 minutes)

### Verify 1: Code compiles
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
```
**Expected output:**
```
    Checking tantivy_search v0.1.0
    Finished dev [unoptimized + debuginfo] target(s) in 2.1s
```

### Verify 2: Tests pass
```bash
cargo test corruption
```
**Expected output:**
```
test corruption_detector::tests::test_corruption_detector_creation ... ok
test corruption_detector::tests::test_healthy_index_detection ... ok
test corruption_integration_test::test_corruption_detection_integration ... ok

test result: ok. 3 passed; 0 failed
```

### Verify 3: Integration works
```bash
cargo test health_check
```
**Expected output should include:**
```
test health_check::tests::test_check_data_integrity ... ok
```

## Success Validation Checklist
- [ ] File exists: `src/corruption_detector.rs` with exactly 3 public structs
- [ ] CorruptionDetector can detect different severity levels
- [ ] Health check system integrated with corruption detection
- [ ] All tests pass (corruption + integration)
- [ ] Module properly exported in lib.rs

## Files Created For Next Task
After completing this task, you will have:

1. **C:/code/LLMKG/vectors/tantivy_search/src/corruption_detector.rs** - Complete corruption detection system
2. **Updated C:/code/LLMKG/vectors/tantivy_search/src/health_check.rs** - Integration with health checks
3. **C:/code/LLMKG/vectors/tantivy_search/tests/corruption_integration_test.rs** - Integration test

**Next Task (Task 62)** will build on this corruption detection to implement network failure simulation and recovery procedures.