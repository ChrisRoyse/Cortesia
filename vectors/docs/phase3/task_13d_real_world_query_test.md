# Task 13d: Real World Query Test

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: task_13c_method_comparison_test.md**

## Context
You are implementing tests for complex, real-world search scenarios that include special characters, generic types, and advanced Rust patterns. This validates that the search system handles the complex syntax patterns found in actual neuromorphic codebases.

## Your Task
Implement comprehensive tests for special characters and complex query patterns commonly found in Rust code.

## Required Implementation
Add this test function to the integration_tests module in `crates/vector-search/src/lib.rs`:

```rust
#[tokio::test]
async fn test_real_world_query_patterns() {
    let temp_dir = TempDir::new().unwrap();
    create_test_workspace(&temp_dir).unwrap();
    
    let workspace_path = temp_dir.path();
    let index_path = temp_dir.path().join("real_world_test_index");
    
    // Index the test workspace
    index_directory(workspace_path, &index_path).await.unwrap();
    let search_engine = SearchEngine::new(&index_path).unwrap();
    
    // Test special character search - workspace sections
    let workspace_results = search_engine.search_text("[workspace]").unwrap();
    assert!(!workspace_results.is_empty(), "Should find workspace brackets");
    
    // Test generic type patterns
    let generic_results = search_engine.search_text("Result<T, E>").unwrap();
    assert!(!generic_results.is_empty(), "Should find generic types");
    
    // Test attribute patterns
    let derive_results = search_engine.search_text("#[derive(Debug)]").unwrap();
    assert!(!derive_results.is_empty(), "Should find derive attributes");
    
    // Test complex nested generics
    let complex_generic_results = search_engine.search_text("HashMap<String, Vec<u64>>").unwrap();
    assert!(!complex_generic_results.is_empty(), "Should find complex generics");
    
    // Test function signature patterns
    let function_results = search_engine.search_text("fn main() -> Result<(), Box<dyn Error>>").unwrap();
    assert!(!function_results.is_empty(), "Should find function signatures");
    
    // Test async trait patterns
    let async_trait_results = search_engine.search_text("#[async_trait::async_trait]").unwrap();
    assert!(!async_trait_results.is_empty(), "Should find async trait attributes");
    
    println!("✓ Real-world query test passed:");
    println!("  - Workspace patterns: {} results", workspace_results.len());
    println!("  - Generic types: {} results", generic_results.len());
    println!("  - Derive attributes: {} results", derive_results.len());
    println!("  - Complex generics: {} results", complex_generic_results.len());
    println!("  - Function signatures: {} results", function_results.len());
    println!("  - Async traits: {} results", async_trait_results.len());
}
```

## Success Criteria
- [ ] Special character searches work correctly
- [ ] Generic type patterns found accurately
- [ ] Attribute patterns recognized properly
- [ ] Complex nested generics handled correctly
- [ ] Function signatures searchable
- [ ] Async trait patterns detected
- [ ] All assertions pass
- [ ] Test provides useful diagnostic output

## Validation
Run `cargo test -p vector-search test_real_world_query_patterns` - should pass.

## Next Task
Task 13e will implement error handling integration tests to ensure robust failure recovery.