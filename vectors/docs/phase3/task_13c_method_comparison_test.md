# Task 13c: Method Comparison Test

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: task_13b_basic_engine_integration_test.md**

## Context
You are creating tests that validate different search methods work correctly and consistently. This ensures that language filtering, file filtering, and different query types all produce expected results when used with the same indexed content.

## Your Task
Implement comprehensive tests for different search methods and filtering capabilities.

## Required Implementation
Add this test function to the integration_tests module in `crates/vector-search/src/lib.rs`:

```rust
#[tokio::test]
async fn test_search_method_comparison() {
    let temp_dir = TempDir::new().unwrap();
    create_test_workspace(&temp_dir).unwrap();
    
    let workspace_path = temp_dir.path();
    let index_path = temp_dir.path().join("method_test_index");
    
    // Index the test workspace
    index_directory(workspace_path, &index_path).await.unwrap();
    let search_engine = SearchEngine::new(&index_path).unwrap();
    
    // Test basic text search
    let basic_results = search_engine.search_text("CorticalColumn").unwrap();
    assert!(!basic_results.is_empty(), "Should find CorticalColumn references");
    
    // Test language filtering
    let rust_results = search_engine.search_by_language("CorticalColumn", "rust").unwrap();
    assert!(!rust_results.is_empty(), "Should find Rust-specific content");
    
    let python_results = search_engine.search_by_language("NeuralBridge", "python").unwrap();
    assert!(!python_results.is_empty(), "Should find Python-specific content");
    
    // Test file filtering
    let cargo_results = search_engine.search_in_files("workspace", &["Cargo"]).unwrap();
    assert!(!cargo_results.is_empty(), "Should find Cargo.toml content");
    
    // Verify language availability
    let stats = search_engine.get_stats().unwrap();
    assert!(stats.available_languages.contains(&"rust".to_string()));
    assert!(stats.available_languages.contains(&"python".to_string()));
    assert!(stats.available_languages.contains(&"toml".to_string()));
    
    println!("✓ Method comparison test passed:");
    println!("  - Basic search: {} results", basic_results.len());
    println!("  - Rust filtering: {} results", rust_results.len());
    println!("  - Python filtering: {} results", python_results.len());
    println!("  - Available languages: {:?}", stats.available_languages);
}
```

## Success Criteria
- [ ] Multiple search methods tested
- [ ] Language filtering verified
- [ ] File filtering verified
- [ ] Statistics validation included
- [ ] All search types return expected results
- [ ] Language availability properly tracked
- [ ] Test passes when executed

## Validation
Run `cargo test -p vector-search test_search_method_comparison` - should pass.

## Next Task
Task 13d will implement real-world query tests using complex patterns and special characters.