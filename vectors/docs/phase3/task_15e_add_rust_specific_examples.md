# Task 15e: Add Rust-Specific Code Search Examples

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: Task 15d completion**

## Context
Rust developers need specialized search patterns for code exploration, dependency analysis, and pattern recognition within Rust codebases. This task creates comprehensive examples for searching Rust code structures, syntax patterns, and common Rust idioms.

## Your Task
Create detailed examples demonstrating Rust-specific search patterns including function signatures, trait implementations, macro usage, error handling patterns, and async/await constructs.

## Required Implementation

```rust
//! # Rust-Specific Code Search Examples
//! 
//! Demonstrates specialized search patterns for Rust codebases including
//! function signatures, traits, macros, async patterns, and error handling.

use vector_search::{
    CodeQuery,
    RustParser,
    SyntaxQuery,
    SemanticQuery,
    SearchEngine
};

/// Function and method search patterns
pub async fn rust_function_search_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Basic function signature search
    let function_query = CodeQuery::rust()
        .function_signature("fn.*process.*data")
        .with_field("code");
    
    let functions = engine.search_code(&function_query).await?;
    println!("Function signatures found: {}", functions.len());
    
    // Async function patterns
    let async_functions = CodeQuery::rust()
        .pattern(r"async\s+fn\s+\w+")
        .with_return_type("Result<")
        .with_field("code");
    
    let async_results = engine.search_code(&async_functions).await?;
    
    // Generic function search
    let generic_functions = CodeQuery::rust()
        .function_signature(r"fn\s+\w+<[^>]+>")
        .with_complexity_limit(20)  // Avoid overly complex signatures
        .with_field("code");
    
    let generic_results = engine.search_code(&generic_functions).await?;
    
    // Method implementation search
    let impl_methods = CodeQuery::rust()
        .impl_block("impl.*MyStruct")
        .function_within_impl("fn.*new")
        .with_field("code");
    
    let methods = engine.search_code(&impl_methods).await?;
    
    // Constructor patterns
    let constructors = CodeQuery::rust()
        .pattern(r"fn\s+new\s*\(")
        .or_pattern(r"fn\s+default\s*\(")
        .or_pattern(r"fn\s+from\s*\(")
        .with_field("code");
    
    let constructor_results = engine.search_code(&constructors).await?;
    
    // Display function signatures with context
    for result in functions.iter().take(3) {
        if let Some(signature) = result.extract_function_signature() {
            println!("Function: {}", signature);
            if let Some(context) = result.get_context_lines(2) {
                println!("Context:\n{}", context);
            }
        }
    }
    
    Ok(())
}

/// Trait and implementation search patterns
pub async fn rust_trait_search_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Trait definition search
    let trait_definitions = CodeQuery::rust()
        .trait_definition("trait.*Serialize")
        .with_associated_types(true)
        .with_field("code");
    
    let traits = engine.search_code(&trait_definitions).await?;
    println!("Trait definitions: {}", traits.len());
    
    // Trait implementation search
    let trait_impls = CodeQuery::rust()
        .impl_trait("impl.*Display.*for")
        .with_generic_constraints(true)
        .with_field("code");
    
    let implementations = engine.search_code(&trait_impls).await?;
    
    // Derive macro usage
    let derive_patterns = CodeQuery::rust()
        .derive_macro("#\\[derive\\(.*Debug.*\\)\\]")
        .with_following_struct(true)
        .with_field("code");
    
    let derives = engine.search_code(&derive_patterns).await?;
    
    // Generic trait bounds
    let bounded_generics = CodeQuery::rust()
        .pattern(r"where\s+\w+:\s*\w+")
        .or_pattern(r"<\w+:\s*\w+>")
        .with_complexity_analysis(true)
        .with_field("code");
    
    let bounds = engine.search_code(&bounded_generics).await?;
    
    // Associated type usage
    let associated_types = CodeQuery::rust()
        .pattern(r"type\s+\w+\s*=")
        .within_trait_impl(true)
        .with_field("code");
    
    let assoc_types = engine.search_code(&associated_types).await?;
    
    Ok(())
}

/// Error handling and Result patterns
pub async fn rust_error_handling_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Result type usage patterns
    let result_patterns = CodeQuery::rust()
        .return_type("Result<")
        .with_error_propagation("?")
        .with_field("code");
    
    let results = engine.search_code(&result_patterns).await?;
    println!("Result patterns: {}", results.len());
    
    // Error propagation with ?
    let error_propagation = CodeQuery::rust()
        .pattern(r"\w+\([^)]*\)\?")
        .with_context_analysis(true)
        .with_field("code");
    
    let propagation = engine.search_code(&error_propagation).await?;
    
    // Match error handling
    let match_errors = CodeQuery::rust()
        .match_expression("match.*result")
        .with_error_arm("Err")
        .with_success_arm("Ok")
        .with_field("code");
    
    let matches = engine.search_code(&match_errors).await?;
    
    // Custom error types
    let custom_errors = CodeQuery::rust()
        .enum_definition("enum.*Error")
        .or_struct_definition("struct.*Error")
        .with_error_trait_impl(true)
        .with_field("code");
    
    let error_types = engine.search_code(&custom_errors).await?;
    
    // Error conversion patterns
    let error_conversions = CodeQuery::rust()
        .impl_from("impl From<")
        .for_error_type(true)
        .with_field("code");
    
    let conversions = engine.search_code(&error_conversions).await?;
    
    Ok(())
}

/// Async/await and concurrency patterns
pub async fn rust_async_patterns_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Basic async/await usage
    let async_await = CodeQuery::rust()
        .async_function("async fn")
        .with_await_usage("await")
        .with_field("code");
    
    let async_results = engine.search_code(&async_await).await?;
    println!("Async/await patterns: {}", async_results.len());
    
    // Tokio runtime patterns
    let tokio_patterns = CodeQuery::rust()
        .pattern(r"#\[tokio::main\]")
        .or_pattern(r"tokio::spawn")
        .or_pattern(r"Runtime::new")
        .with_field("code");
    
    let tokio_usage = engine.search_code(&tokio_patterns).await?;
    
    // Channel usage patterns
    let channel_patterns = CodeQuery::rust()
        .pattern(r"mpsc::(channel|unbounded)")
        .or_pattern(r"oneshot::channel")
        .or_pattern(r"broadcast::channel")
        .with_usage_context(true)
        .with_field("code");
    
    let channels = engine.search_code(&channel_patterns).await?;
    
    // Mutex and Arc patterns
    let concurrency_primitives = CodeQuery::rust()
        .pattern(r"Arc<Mutex<")
        .or_pattern(r"RwLock<")
        .or_pattern(r"AtomicBool")
        .with_thread_safety_analysis(true)
        .with_field("code");
    
    let primitives = engine.search_code(&concurrency_primitives).await?;
    
    // Future and Stream patterns
    let futures = CodeQuery::rust()
        .pattern(r"impl\s+Future")
        .or_pattern(r"Pin<Box<dyn Future")
        .or_pattern(r"Stream<Item")
        .with_field("code");
    
    let future_impls = engine.search_code(&futures).await?;
    
    Ok(())
}

/// Macro definition and usage patterns
pub async fn rust_macro_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Declarative macro definitions
    let macro_definitions = CodeQuery::rust()
        .macro_definition("macro_rules!")
        .with_pattern_matching(true)
        .with_field("code");
    
    let macros = engine.search_code(&macro_definitions).await?;
    println!("Macro definitions: {}", macros.len());
    
    // Procedural macro patterns
    let proc_macros = CodeQuery::rust()
        .pattern(r"#\[proc_macro")
        .or_pattern(r"#\[derive\(")
        .or_pattern(r"#\[proc_macro_attribute\]")
        .with_field("code");
    
    let proc_macro_usage = engine.search_code(&proc_macros).await?;
    
    // Common macro invocations
    let macro_invocations = CodeQuery::rust()
        .pattern(r"println!\(")
        .or_pattern(r"vec!\[")
        .or_pattern(r"format!\(")
        .or_pattern(r"assert!")
        .with_expansion_context(true)
        .with_field("code");
    
    let invocations = engine.search_code(&macro_invocations).await?;
    
    // Custom derive macros
    let custom_derives = CodeQuery::rust()
        .derive_macro("#\\[derive\\(.*\\w+.*\\)\\]")
        .exclude_std_derives(true)  // Focus on custom derives
        .with_field("code");
    
    let custom_derive_usage = engine.search_code(&custom_derives).await?;
    
    Ok(())
}

/// Dependency and crate usage patterns
pub async fn rust_dependency_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // External crate usage
    let extern_crates = CodeQuery::rust()
        .use_statement("use.*::")
        .with_crate_analysis(true)
        .with_field("code");
    
    let uses = engine.search_code(&extern_crates).await?;
    println!("Use statements: {}", uses.len());
    
    // Serde patterns
    let serde_patterns = CodeQuery::rust()
        .pattern(r"#\[derive\(.*Serialize.*\)\\]")
        .or_pattern(r"#\\[serde\\(")
        .or_pattern(r"serde_json::")
        .with_field("code");
    
    let serde_usage = engine.search_code(&serde_patterns).await?;
    
    // Database ORM patterns  
    let database_patterns = CodeQuery::rust()
        .pattern(r"#\[derive\(.*Queryable.*\)\\]")
        .or_pattern(r"diesel::")
        .or_pattern(r"sqlx::")
        .with_field("code");
    
    let db_usage = engine.search_code(&database_patterns).await?;
    
    // Web framework patterns
    let web_patterns = CodeQuery::rust()
        .pattern(r"#\[get\(")
        .or_pattern(r"#\[post\(")
        .or_pattern(r"actix_web::")
        .or_pattern(r"warp::")
        .with_field("code");
    
    let web_usage = engine.search_code(&web_patterns).await?;
    
    // Test patterns
    let test_patterns = CodeQuery::rust()
        .pattern(r"#\[test\]")
        .or_pattern(r"#\[tokio::test\]")
        .or_pattern(r"assert_eq!")
        .or_pattern(r"mock")
        .with_field("code");
    
    let test_usage = engine.search_code(&test_patterns).await?;
    
    Ok(())
}

/// Advanced Rust-specific search combinations
pub async fn advanced_rust_search_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Complex function analysis
    let complex_analysis = CodeQuery::rust()
        .function_signature("fn.*")
        .with_complexity_metrics(true)
        .with_dependency_analysis(true)
        .with_performance_hints(true)
        .filter_by_complexity(10, 50)  // Medium complexity range
        .with_field("code");
    
    let complex_functions = engine.search_code(&complex_analysis).await?;
    
    // Lifetime and borrowing patterns
    let lifetime_patterns = CodeQuery::rust()
        .pattern(r"<'[a-z]+>")
        .or_pattern(r"&'[a-z]+")
        .with_borrow_checker_analysis(true)
        .with_field("code");
    
    let lifetimes = engine.search_code(&lifetime_patterns).await?;
    
    // Unsafe code detection
    let unsafe_patterns = CodeQuery::rust()
        .unsafe_block("unsafe")
        .with_safety_analysis(true)
        .with_field("code");
    
    let unsafe_code = engine.search_code(&unsafe_patterns).await?;
    
    // Performance-critical patterns
    let performance_patterns = CodeQuery::rust()
        .pattern(r"#\[inline")
        .or_pattern(r"SIMD")
        .or_pattern(r"MaybeUninit")
        .with_performance_analysis(true)
        .with_field("code");
    
    let perf_code = engine.search_code(&performance_patterns).await?;
    
    // Generate comprehensive analysis report
    let total_patterns = complex_functions.len() + lifetimes.len() + 
                        unsafe_code.len() + perf_code.len();
    
    println!("Advanced Rust analysis completed: {} patterns found", total_patterns);
    
    // Extract insights from results
    for result in complex_functions.iter().take(3) {
        if let Some(metrics) = result.get_complexity_metrics() {
            println!("Function complexity: cyclomatic={}, cognitive={}", 
                    metrics.cyclomatic, metrics.cognitive);
        }
    }
    
    Ok(())
}

/// Performance guidelines for Rust code search
pub fn rust_search_performance_tips() {
    println!("
    Rust Code Search Performance Tips:
    
    1. Syntax Parsing:
       - Use tree-sitter for accurate syntax analysis
       - Cache parsed ASTs for repeated searches
       - Limit search depth for large files
       
    2. Pattern Optimization:
       - Use specific patterns over broad wildcards
       - Combine related patterns in single queries
       - Filter by file extensions (.rs) early
       
    3. Semantic Analysis:
       - Enable semantic analysis only when needed
       - Use symbol tables for identifier searches
       - Cache type information for repeated queries
       
    4. Memory Management:
       - Stream large codebases rather than loading all
       - Use parallel processing for independent files
       - Clear caches periodically to prevent memory leaks
       
    5. Index Optimization:
       - Create separate indices for code vs documentation
       - Use specialized tokenizers for Rust syntax
       - Index function signatures separately from bodies
    ");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_rust_function_detection() {
        let engine = create_rust_test_engine().await;
        
        let query = CodeQuery::rust()
            .function_signature("fn test_function")
            .with_field("code");
            
        let results = engine.search_code(&query).await.unwrap();
        
        // Verify all results are valid Rust functions
        for result in results {
            let code = result.get_field("code").unwrap();
            assert!(code.contains("fn test_function"));
            assert!(is_valid_rust_syntax(&code));
        }
    }
    
    #[tokio::test]
    async fn test_async_pattern_accuracy() {
        let engine = create_rust_test_engine().await;
        
        let query = CodeQuery::rust()
            .async_function("async fn")
            .with_await_usage("await");
            
        let results = engine.search_code(&query).await.unwrap();
        
        // Verify async patterns are correctly identified
        for result in results {
            let code = result.get_field("code").unwrap();
            assert!(code.contains("async fn"));
            assert!(code.contains(".await") || code.contains("await!"));
        }
    }
}
```

## Success Criteria
- [ ] Rust function and method search patterns implemented
- [ ] Trait definition and implementation search examples
- [ ] Error handling and Result pattern searches
- [ ] Async/await and concurrency pattern detection
- [ ] Macro definition and usage search capabilities
- [ ] Dependency and crate usage pattern recognition
- [ ] Advanced Rust-specific search combinations
- [ ] Performance optimization guidelines for code search

## Validation
Test Rust-specific search functionality:
```bash
cargo test test_rust_function_detection
cargo test test_async_pattern_accuracy

# Test with real Rust codebase
cargo run --example rust_codebase_analysis -- /path/to/rust/project
```

## Next Task
Task 15f will create the final documentation guide file with quick start instructions and comprehensive usage examples.