# Task 97: System Integration Verification

**Estimated Time:** 10 minutes  
**Prerequisites:** Tasks 01-96  
**Dependencies:** All core components must be implemented

## Objective
Verify complete system integration across all Phase 1 components with production-ready validation.

## Context
You're implementing the final integration verification for Phase 1 of the vector search system. All individual components (Tantivy indexing, AST chunking, search engine) are complete. Now verify they work together seamlessly with special character support and Windows compatibility.

## Task Details

### What You Need to Do

1. **Create integration verification module** (`src/integration_verify.rs`):
```rust
use crate::{schema::*, chunker::*, indexer::*, search::*};
use std::path::Path;
use anyhow::Result;

pub struct SystemVerifier {
    temp_dir: tempfile::TempDir,
    indexer: DocumentIndexer,
    search_engine: SearchEngine,
}

impl SystemVerifier {
    pub fn new() -> Result<Self> {
        let temp_dir = tempfile::tempdir()?;
        let index_path = temp_dir.path().join("test_index");
        
        let indexer = DocumentIndexer::new(&index_path)?;
        let search_engine = SearchEngine::new(&index_path)?;
        
        Ok(Self { temp_dir, indexer, search_engine })
    }
    
    pub fn verify_special_characters(&mut self) -> Result<VerificationReport> {
        // Test files with various special characters
        let test_cases = vec![
            ("[workspace] config", "Cargo.toml"),
            ("Result<T, E> -> impl Trait", "lib.rs"),
            ("#[derive(Debug)] struct", "model.rs"),
            ("&mut self ## comment", "method.rs"),
        ];
        
        let mut report = VerificationReport::new();
        
        for (content, filename) in test_cases {
            let file_path = self.temp_dir.path().join(filename);
            std::fs::write(&file_path, content)?;
            
            // Index the file
            self.indexer.index_file(&file_path)?;
            
            // Search for special characters
            let results = self.search_engine.search(content)?;
            
            report.add_test(filename, !results.is_empty());
        }
        
        Ok(report)
    }
    
    pub fn verify_chunk_boundaries(&mut self) -> Result<VerificationReport> {
        // Create large file that will be chunked
        let large_content = format!(
            "{}\npub fn main() {{\n    println!(\"test\");\n}}\n{}",
            "// header\n".repeat(500),
            "// footer\n".repeat(500)
        );
        
        let file_path = self.temp_dir.path().join("large.rs");
        std::fs::write(&file_path, &large_content)?;
        
        self.indexer.index_file(&file_path)?;
        
        // Verify function is searchable
        let results = self.search_engine.search("pub fn main")?;
        
        let mut report = VerificationReport::new();
        report.add_test("chunk_boundaries", !results.is_empty());
        report.add_test("function_integrity", 
            results.iter().any(|r| r.content.contains("pub fn main") 
                && r.content.contains("println!")));
        
        Ok(report)
    }
}

#[derive(Debug)]
pub struct VerificationReport {
    tests: Vec<(String, bool)>,
}

impl VerificationReport {
    pub fn new() -> Self {
        Self { tests: Vec::new() }
    }
    
    pub fn add_test(&mut self, name: &str, passed: bool) {
        self.tests.push((name.to_string(), passed));
    }
    
    pub fn all_passed(&self) -> bool {
        self.tests.iter().all(|(_, passed)| *passed)
    }
}
```

2. **Add comprehensive integration test**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_full_system_integration() -> Result<()> {
        let mut verifier = SystemVerifier::new()?;
        
        let special_char_report = verifier.verify_special_characters()?;
        assert!(special_char_report.all_passed(), 
            "Special character tests failed: {:?}", special_char_report);
        
        let chunk_report = verifier.verify_chunk_boundaries()?;
        assert!(chunk_report.all_passed(),
            "Chunk boundary tests failed: {:?}", chunk_report);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] All special characters are searchable
- [ ] Chunk boundaries preserve semantic completeness
- [ ] Integration tests pass without errors
- [ ] System works on Windows with all path types
- [ ] Memory usage stays under 200MB for test data
- [ ] Search latency < 10ms for all test queries

## Common Pitfalls to Avoid
- Don't skip testing Windows path formats (C:\\, \\\\network\\)
- Ensure temp directories are properly cleaned up
- Verify index persistence across restarts
- Test with both small and large files
- Check for memory leaks during extended operations

## Context for Next Task
Task 98 will implement production monitoring and telemetry for the integrated system.