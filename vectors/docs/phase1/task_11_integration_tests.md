# Task 11: Create End-to-End Integration Tests

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 10 (SearchEngine)  
**Dependencies:** Tasks 01-10 must be completed

## Objective
Create comprehensive integration tests that verify the complete workflow from file indexing through searching, with special focus on special character handling and cross-platform compatibility.

## Context
You now have all core components: schema, chunking, indexing, and searching. Integration tests verify these components work together correctly and that the system meets its core requirement of 100% special character support for code search.

## Task Details

### What You Need to Do

1. **Create `tests/integration_tests.rs` in project root:**

   ```rust
   //! End-to-end integration tests for the complete search system
   
   use llmkg_vectors::{DocumentIndexer, SearchEngine, SearchOptions};
   use tempfile::TempDir;
   use std::fs;
   use anyhow::Result;
   
   #[tokio::test]
   async fn test_complete_workflow() -> Result<()> {
       let temp_dir = TempDir::new()?;
       let index_path = temp_dir.path().join("integration_index");
       
       // Step 1: Create test files with various content types
       let test_files = create_test_files(&temp_dir)?;
       
       // Step 2: Index all files
       let mut indexer = DocumentIndexer::new(&index_path)?;
       let file_refs: Vec<&std::path::Path> = test_files.iter().map(|p| p.as_path()).collect();
       let stats = indexer.index_files(&file_refs)?;
       indexer.commit()?;
       
       assert_eq!(stats.files_processed, test_files.len());
       assert_eq!(stats.files_failed, 0);
       
       // Step 3: Search for various patterns
       let search_engine = SearchEngine::new(&index_path)?;
       
       // Test special character searches
       let special_tests = vec![
           ("[workspace]", "Should find TOML workspace config"),
           ("Result<T, E>", "Should find generic Result type"),
           ("#[derive(Debug)]", "Should find derive macros"),
           ("&mut self", "Should find mutable references"),
           ("## comment", "Should find hash comments"),
           ("fn main()", "Should find function definitions"),
           ("class DataProcessor", "Should find Python classes"),
       ];
       
       for (query, description) in special_tests {
           let results = search_engine.search(query, 10)?;
           assert!(!results.is_empty(), "{}: query '{}'", description, query);
           
           // Verify results contain the searched pattern
           let pattern_found = results.iter().any(|r| 
               r.content.contains(query) || 
               query.chars().all(|c| r.content.contains(c))
           );
           assert!(pattern_found, "Results should contain pattern: {}", query);
       }
       
       Ok(())
   }
   
   #[test]
   fn test_chunking_preserves_semantic_boundaries() -> Result<()> {
       let temp_dir = TempDir::new()?;
       let index_path = temp_dir.path().join("chunking_test_index");
       
       // Create a large Rust file that will require chunking
       let large_rust_file = temp_dir.path().join("large.rs");
       let content = create_large_rust_content();
       fs::write(&large_rust_file, &content)?;
       
       // Index with small chunk size to force chunking
       let mut indexer = DocumentIndexer::new(&index_path)?;
       let chunk_count = indexer.index_file(&large_rust_file)?;
       indexer.commit()?;
       
       assert!(chunk_count > 1, "Large file should create multiple chunks");
       
       // Search for function that spans chunks
       let search_engine = SearchEngine::new(&index_path)?;
       let results = search_engine.search("pub fn large_function", 10)?;
       
       // Should find the function even if it spans chunks
       assert!(!results.is_empty(), "Should find function across chunk boundaries");
       
       Ok(())
   }
   
   #[test] 
   fn test_multi_language_support() -> Result<()> {
       let temp_dir = TempDir::new()?;
       let index_path = temp_dir.path().join("multilang_index");
       
       // Create files in different languages
       let files = vec![
           ("main.rs", "pub fn rust_function() { println!(\"Rust\"); }"),
           ("script.py", "def python_function():\n    print('Python')"),
           ("app.js", "function js_function() { console.log('JavaScript'); }"),
           ("README.md", "# Documentation\nThis is markdown content"),
           ("config.toml", "[section]\nkey = \"value\"\n"),
       ];
       
       let mut file_paths = Vec::new();
       for (filename, content) in files {
           let file_path = temp_dir.path().join(filename);
           fs::write(&file_path, content)?;
           file_paths.push(file_path);
       }
       
       // Index all files
       let mut indexer = DocumentIndexer::new(&index_path)?;
       let file_refs: Vec<&std::path::Path> = file_paths.iter().map(|p| p.as_path()).collect();
       let stats = indexer.index_files(&file_refs)?;
       indexer.commit()?;
       
       assert_eq!(stats.files_processed, 5);
       
       // Search across languages
       let search_engine = SearchEngine::new(&index_path)?;
       
       let cross_lang_tests = vec![
           ("function", 3), // Should find in Rust, Python, JS
           ("[section]", 1), // Should find only in TOML
           ("println", 1),   // Should find only in Rust
           ("print", 2),     // Should find in Rust and Python
       ];
       
       for (query, expected_min) in cross_lang_tests {
           let results = search_engine.search(query, 20)?;
           assert!(results.len() >= expected_min, 
                  "Query '{}' should find at least {} results, found {}", 
                  query, expected_min, results.len());
       }
       
       Ok(())
   }
   
   fn create_test_files(temp_dir: &TempDir) -> Result<Vec<std::path::PathBuf>> {
       let files = vec![
           ("Cargo.toml", r#"
               [workspace]
               members = ["backend", "frontend"]
               
               [dependencies]
               tokio = { version = "1.0", features = ["full"] }
               anyhow = "1.0"
           "#),
           ("src/main.rs", r#"
               use std::collections::HashMap;
               
               /// Main application entry point
               /// ## Usage
               /// Run with: cargo run
               pub fn main() -> Result<(), Box<dyn std::error::Error>> {
                   let config = Config::new();
                   process_data(&mut config)?;
                   Ok(())
               }
               
               #[derive(Debug, Clone)]
               pub struct Config {
                   pub name: String,
                   pub values: HashMap<String, i32>,
               }
               
               impl Config {
                   pub fn new() -> Self {
                       Self {
                           name: "default".to_string(),
                           values: HashMap::new(),
                       }
                   }
               }
               
               pub fn process_data<T, E>(config: &mut Config) -> Result<T, E> 
               where 
                   T: Clone + Send + 'static,
                   E: std::error::Error,
               {
                   // Processing logic here
                   todo!("Implement processing")
               }
           "#),
           ("utils.py", r#"
               '''Utility functions for data processing'''
               
               class DataProcessor:
                   def __init__(self, config=None):
                       self.config = config or {}
                   
                   def process(self, data):
                       '''Process input data'''
                       return [item.upper() for item in data]
               
               def helper_function():
                   '''Helper function with special chars: <>&'''
                   return "Processing complete"
           "#),
       ];
       
       let mut file_paths = Vec::new();
       for (filename, content) in files {
           let file_path = temp_dir.path().join(filename);
           if let Some(parent) = file_path.parent() {
               fs::create_dir_all(parent)?;
           }
           fs::write(&file_path, content)?;
           file_paths.push(file_path);
       }
       
       Ok(file_paths)
   }
   
   fn create_large_rust_content() -> String {
       let mut content = String::new();
       
       // Create multiple functions to test chunking
       for i in 1..=20 {
           content.push_str(&format!(r#"
               /// Function number {}
               pub fn function_{}() -> Result<String, Error> {{
                   println!("Function {} executing");
                   let data = vec![1, 2, 3, 4, 5];
                   let processed: Vec<_> = data.iter()
                       .map(|x| x * 2)
                       .collect();
                   Ok(format!("Processed: {{:?}}", processed))
               }}
           "#, i, i, i));
       }
       
       // Add a large function that might span chunks
       content.push_str(r#"
           pub fn large_function_with_lots_of_code() {
               // This function has lots of content to ensure chunking
               let mut data = HashMap::new();
               for i in 0..100 {
                   data.insert(format!("key_{}", i), i * 2);
               }
               
               println!("Large function processing...");
               
               // More code to make it large
               match data.get("key_50") {
                   Some(value) => println!("Found value: {}", value),
                   None => println!("Value not found"),
               }
           }
       "#);
       
       content
   }
   ```

## Success Criteria
- [ ] Integration tests compile without errors
- [ ] All integration tests pass with `cargo test --test integration_tests`
- [ ] Complete workflow (index → search) works correctly
- [ ] Special character searches find accurate results
- [ ] Multi-language files are indexed and searchable
- [ ] Chunking preserves semantic boundaries
- [ ] Cross-language searches work as expected
- [ ] Windows file paths handled correctly in tests

## If This Task Fails

### Common Errors and Solutions

**Error 1: "error[E0433]: failed to resolve: use of undeclared type `TempDir`"**
```bash
# Solution: Missing tempfile dev dependency
cargo add --dev tempfile@3.14.0
# Ensure integration test imports are correct
cargo test --test integration_tests
```

**Error 2: "assertion failed: !search_results.is_empty()" in integration tests**
```bash
# Solution: End-to-end workflow not working properly
# Check that indexing completes before searching
# Verify commit() is called on index writer
RUST_LOG=tantivy=debug cargo test test_complete_workflow -- --nocapture
```

**Error 3: "No such file or directory" creating test files**
```bash
# Solution: Test file creation failing
# Check temp directory permissions
# Verify create_test_files function works correctly
cargo test create_test_files -- --nocapture
```

**Error 4: "Special character search returned no results"**
```bash
# Solution: Special character indexing/searching broken
# Verify Tasks 03-04 special character support works
# Check dual-field configuration in integration
cargo test test_special_character_integration -- --nocapture
```

## Troubleshooting Checklist

- [ ] Tasks 03-10 all completed successfully
- [ ] All unit tests pass for individual modules
- [ ] Tempfile dependency in [dev-dependencies]
- [ ] Integration test file in tests/ directory (not src/)
- [ ] Test functions marked with #[test]
- [ ] Proper error handling in test setup functions
- [ ] Index commits before search operations
- [ ] Test data includes actual special characters
- [ ] Windows file path handling tested
- [ ] Cross-language file support verified

## Recovery Procedures

### Integration Test Setup Failures
If test environment setup consistently fails:
1. Verify temp directory creation: Check TempDir::new() succeeds
2. Test file creation: Ensure write permissions to temp directories
3. Validate test data: Check create_test_files produces valid content
4. Debug step by step: Run setup functions individually

### Indexing-to-Search Workflow Issues
If the complete workflow fails:
1. Test indexing separately: Verify indexer works in isolation
2. Test searching separately: Confirm search engine works alone
3. Check timing: Ensure proper sequence (index → commit → search)
4. Verify schema consistency: Same schema used for indexing and searching

### Special Character Integration Problems
If special characters fail in integration:
1. Verify individual special char tests pass (Task 04)
2. Check field mapping: Ensure raw_content field used correctly
3. Test character encoding: Verify UTF-8 handling throughout pipeline
4. Debug document creation: Print indexed documents before searching

### Cross-Language Support Issues
If multi-language files cause problems:
1. Test language detection: Verify file extension handling
2. Check chunker fallbacks: Ensure unsupported languages use simple chunking
3. Verify search coverage: Ensure all languages searchable
4. Test file path handling: Check Windows path compatibility

## Common Pitfalls to Avoid
- Don't skip testing negative cases (searches that should find nothing)
- Ensure test files contain actual special characters to validate
- Don't ignore chunk boundary edge cases
- Test both AST-parsed and fallback-chunked languages
- Verify search results actually contain the queried terms

## Context for Next Task
Task 12 will add performance benchmarking to ensure the system meets the <10ms search latency and >500 docs/sec indexing targets.