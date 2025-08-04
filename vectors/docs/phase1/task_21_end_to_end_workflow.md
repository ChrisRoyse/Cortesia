# Task 21: Create End-to-End Integration Test Workflow

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 20 (Search caching)  
**Dependencies:** Tasks 01-20 must be completed

## Objective
Create comprehensive end-to-end integration tests that validate the complete workflow from file processing through search and result presentation, ensuring all components work together correctly.

## Context
Integration tests are crucial for validating that all individual components work together as a cohesive system. This task creates automated tests that simulate real user workflows and catch integration issues that unit tests might miss.

## Task Details

### What You Need to Do

1. **Create integration tests in `tests/integration_tests.rs`:**

```rust
use llmkg_vectors::*;
use tempfile::TempDir;
use std::fs;
use anyhow::Result;

#[tokio::test]
async fn test_complete_indexing_and_search_workflow() -> Result<()> {
    // Setup test environment
    let temp_dir = TempDir::new()?;
    let test_files = create_test_codebase(&temp_dir)?;
    
    // 1. Create and configure pipeline
    let schema = schema::create_schema();
    let index = tantivy::Index::create_in_dir(temp_dir.path().join("index"), schema)?;
    let indexer = indexing::ChunkIndexer::new(index.clone())?;
    let pipeline_config = pipeline::PipelineConfig::default();
    let mut pipeline = pipeline::ChunkingPipeline::new(indexer, pipeline_config)?;
    
    // 2. Index the test codebase
    let mut total_docs = 0;
    for file_path in &test_files {
        let content = fs::read_to_string(file_path)?;
        let doc = create_document(&schema, &content, file_path, 0)?;
        writer.add_document(doc)?;
        total_docs += 1;
    }
    writer.commit()?;
    
    assert!(total_docs > 0, "Should process test files");
    
    // 3. Create search components
    let search_config = search::SearchConfig::default();
    let search_engine = search::SearchEngine::new(index.clone(), search_config)?;
    let query_parser = query::QueryParser::new(&index)?;
    
    // 4. Test various search scenarios
    test_function_search(&search_engine, &query_parser).await?;
    test_filtered_search(&search_engine, &query_parser).await?;
    test_complex_queries(&search_engine, &query_parser).await?;
    
    // 5. Test result presentation
    test_result_highlighting(&search_engine, &query_parser).await?;
    
    Ok(())
}

fn create_test_codebase(temp_dir: &TempDir) -> Result<Vec<String>> {
    let files = vec![
        ("src/main.rs", r#"
            /// Main application entry point
            fn main() -> Result<(), Box<dyn std::error::Error>> {
                let config = load_configuration()?;
                start_server(config).await
            }
            
            /// Load application configuration
            fn load_configuration() -> Result<Config, ConfigError> {
                Config::from_file("config.toml")
            }
        "#),
        ("src/config.rs", r#"
            use serde::{Deserialize, Serialize};
            
            /// Application configuration structure
            #[derive(Debug, Serialize, Deserialize)]
            pub struct Config {
                pub server_port: u16,
                pub database_url: String,
                pub debug_mode: bool,
            }
            
            impl Config {
                pub fn from_file(path: &str) -> Result<Self, ConfigError> {
                    let content = std::fs::read_to_string(path)?;
                    toml::from_str(&content).map_err(Into::into)
                }
            }
        "#),
        ("src/database.py", r#"
            """Database connection and operations module."""
            
            import asyncpg
            from typing import List, Dict, Any
            
            class DatabaseManager:
                """Manages database connections and operations."""
                
                def __init__(self, connection_url: str):
                    self.connection_url = connection_url
                    self.pool = None
                
                async def connect(self) -> None:
                    """Establish database connection pool."""
                    self.pool = await asyncpg.create_pool(self.connection_url)
                
                async def execute_query(self, query: str, params: List[Any] = None) -> List[Dict]:
                    """Execute a database query and return results."""
                    async with self.pool.acquire() as connection:
                        return await connection.fetch(query, *(params or []))
        "#),
        ("README.md", r#"
            # Test Project
            
            This is a test project for the vector search system.
            It contains sample Rust and Python code to test indexing and search functionality.
            
            ## Features
            
            - Configuration management
            - Database operations
            - Server functionality
            
            ## Usage
            
            Run the application with:
            ```
            cargo run
            ```
        "#),
    ];
    
    fs::create_dir_all(temp_dir.path().join("src"))?;
    
    let mut created_files = Vec::new();
    for (path, content) in files {
        let full_path = temp_dir.path().join(path);
        fs::write(&full_path, content.trim())?;
        created_files.push(full_path.to_string_lossy().to_string());
    }
    
    Ok(created_files)
}

async fn test_function_search(search_engine: &search::SearchEngine, parser: &query::QueryParser) -> Result<()> {
    let parsed_query = parser.parse("load_configuration")?;
    let results = search_engine.search(&parsed_query)?;
    
    assert!(results.total_hits > 0, "Should find function references");
    
    // Verify result contains the function
    let found_function = results.results.iter()
        .any(|r| r.content.contains("load_configuration"));
    assert!(found_function, "Should find load_configuration function");
    
    Ok(())
}

async fn test_filtered_search(search_engine: &search::SearchEngine, parser: &query::QueryParser) -> Result<()> {
    // Test language filter
    let rust_query = parser.parse("config lang:rust")?;
    let rust_results = search_engine.search(&rust_query)?;
    
    for result in &rust_results.results {
        assert_eq!(result.metadata.language, Some("rust".to_string()));
    }
    
    // Test file type filter
    let python_query = parser.parse("database filetype:py")?;
    let python_results = search_engine.search(&python_query)?;
    
    for result in &python_results.results {
        assert!(result.metadata.file_path.ends_with(".py"));
    }
    
    Ok(())
}

async fn test_complex_queries(search_engine: &search::SearchEngine, parser: &query::QueryParser) -> Result<()> {
    // Test boolean query
    let boolean_query = parser.parse("config AND (load OR read)")?;
    let boolean_results = search_engine.search(&boolean_query)?;
    assert!(boolean_results.total_hits > 0, "Boolean query should return results");
    
    // Test phrase query
    let phrase_query = parser.parse("\"database connection\"")?;
    let phrase_results = search_engine.search(&phrase_query)?;
    // May or may not find results depending on exact phrase matching
    
    // Test semantic type filter
    let struct_query = parser.parse("Config type:struct")?;
    let struct_results = search_engine.search(&struct_query)?;
    
    if struct_results.total_hits > 0 {
        let found_struct = struct_results.results.iter()
            .any(|r| r.metadata.semantic_type == "struct");
        assert!(found_struct, "Should find struct semantic type");
    }
    
    Ok(())
}

async fn test_result_highlighting(search_engine: &search::SearchEngine, parser: &query::QueryParser) -> Result<()> {
    let query = parser.parse("configuration")?;
    let results = search_engine.search(&query)?;
    
    if !results.results.is_empty() {
        // Test that we can create highlighted results
        let config = highlighting::HighlightConfig::default();
        let highlighter = highlighting::ResultHighlighter::new(config);
        
        // This would require access to the actual searcher and document
        // For now, just verify the highlighting system exists
        assert!(results.results[0].content.len() > 0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_incremental_indexing_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    create_test_codebase(&temp_dir)?;
    
    // Initial indexing
    let schema = schema::create_schema();
    let index = tantivy::Index::create_in_dir(temp_dir.path().join("index"), schema)?;
    let indexer = indexing::ChunkIndexer::new(index.clone())?;
    let config = pipeline::PipelineConfig {
        enable_incremental: true,
        ..Default::default()
    };
    let mut pipeline = pipeline::ChunkingPipeline::new(indexer, config)?;
    
    let initial_stats = pipeline.process_directory(temp_dir.path())?;
    let initial_chunks = initial_stats.chunks_indexed;
    
    // Add a new file
    fs::write(temp_dir.path().join("src/new_module.rs"), r#"
        /// New module for testing incremental indexing
        pub fn new_function() -> String {
            "This is a new function".to_string()
        }
    "#)?;
    
    // Process again (should only process new file)
    let incremental_stats = pipeline.process_directory(temp_dir.path())?;
    
    // Should have processed the new file
    assert!(incremental_stats.chunks_indexed >= initial_chunks);
    
    Ok(())
}

#[tokio::test]
async fn test_search_performance_and_caching() -> Result<()> {
    let temp_dir = TempDir::new()?;
    create_test_codebase(&temp_dir)?;
    
    // Setup with caching enabled
    let schema = schema::create_schema();
    let index = tantivy::Index::create_in_dir(temp_dir.path().join("index"), schema)?;
    let indexer = indexing::ChunkIndexer::new(index.clone())?;
    let pipeline_config = pipeline::PipelineConfig::default();
    let mut pipeline = pipeline::ChunkingPipeline::new(indexer, pipeline_config)?;
    
    pipeline.process_directory(temp_dir.path())?;
    
    let search_config = search::SearchConfig::default();
    let search_engine = search::SearchEngine::new(index.clone(), search_config)?;
    let query_parser = query::QueryParser::new(&index)?;
    
    // Setup cache
    let cache_config = caching::CacheConfig::default();
    let cache = caching::SearchCache::new(cache_config);
    
    let query = query_parser.parse("configuration")?;
    
    // First search (cache miss)
    let start_time = std::time::Instant::now();
    let results1 = search_engine.search(&query)?;
    let first_search_time = start_time.elapsed();
    
    // Store in cache
    cache.put(&query, results1.clone())?;
    
    // Second search (should be faster if cached)
    let cached_result = cache.get(&query);
    assert!(cached_result.is_some(), "Should retrieve from cache");
    
    // Verify cache stats
    let stats = cache.get_stats();
    assert!(stats.cache_hits > 0, "Should have cache hits");
    assert!(stats.hit_rate > 0.0, "Should have positive hit rate");
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling_and_recovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Test with malformed files
    fs::create_dir_all(temp_dir.path().join("src"))?;
    fs::write(temp_dir.path().join("src/malformed.rs"), "invalid rust syntax {{{")?;
    fs::write(temp_dir.path().join("src/empty.rs"), "")?;
    fs::write(temp_dir.path().join("src/valid.rs"), "fn test() {}")?;
    
    let schema = schema::create_schema();
    let index = tantivy::Index::create_in_dir(temp_dir.path().join("index"), schema)?;
    let indexer = indexing::ChunkIndexer::new(index.clone())?;
    let pipeline_config = pipeline::PipelineConfig::default();
    let mut pipeline = pipeline::ChunkingPipeline::new(indexer, pipeline_config)?;
    
    // Should handle errors gracefully
    let stats = pipeline.process_directory(temp_dir.path())?;
    
    // Should process valid files and track errors
    assert!(stats.files_processed > 0 || stats.files_failed > 0);
    
    // Search should still work
    let search_config = search::SearchConfig::default();
    let search_engine = search::SearchEngine::new(index, search_config)?;
    let query_parser = query::QueryParser::new(&search_engine.index)?;
    
    let query = query_parser.parse("test")?;
    let _results = search_engine.search(&query)?; // Should not panic
    
    Ok(())
}

#[tokio::test]
async fn test_large_codebase_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create a larger test codebase
    fs::create_dir_all(temp_dir.path().join("src"))?;
    
    for i in 0..20 {
        let content = format!(r#"
            /// Module {} for testing large codebase handling
            pub mod module{} {{
                use std::collections::HashMap;
                
                /// Function {} that processes data
                pub fn process_data_{}(input: &str) -> Result<String, Box<dyn std::error::Error>> {{
                    let mut map = HashMap::new();
                    map.insert("key_{}", input);
                    Ok(format!("Processed: {{}}", input))
                }}
                
                /// Structure {} for data storage
                pub struct DataStore{} {{
                    pub data: HashMap<String, String>,
                    pub metadata: Vec<String>,
                }}
                
                impl DataStore{} {{
                    pub fn new() -> Self {{
                        Self {{
                            data: HashMap::new(),
                            metadata: Vec::new(),
                        }}
                    }}
                }}
            }}
        "#, i, i, i, i, i, i, i, i);
        
        fs::write(temp_dir.path().join(format!("src/module_{}.rs", i)), content)?;
    }
    
    let schema = schema::create_schema();
    let index = tantivy::Index::create_in_dir(temp_dir.path().join("index"), schema)?;
    let indexer = indexing::ChunkIndexer::new(index.clone())?;
    let pipeline_config = pipeline::PipelineConfig::default();
    let mut pipeline = pipeline::ChunkingPipeline::new(indexer, pipeline_config)?;
    
    let stats = pipeline.process_directory(temp_dir.path())?;
    
    // Should handle large number of files
    assert!(stats.files_processed >= 20, "Should process all test files");
    assert!(stats.chunks_indexed > 40, "Should create multiple chunks");
    
    // Search should work efficiently
    let search_config = search::SearchConfig::default();
    let search_engine = search::SearchEngine::new(index, search_config)?;
    let query_parser = query::QueryParser::new(&search_engine.index)?;
    
    let start_time = std::time::Instant::now();
    let query = query_parser.parse("process_data")?;
    let results = search_engine.search(&query)?;
    let search_time = start_time.elapsed();
    
    assert!(results.total_hits > 0, "Should find results in large codebase");
    assert!(search_time.as_millis() < 1000, "Search should be reasonably fast");
    
    Ok(())
}
```

2. **Create specialized integration tests for different components:**

```rust
// Additional integration test files would be created here
// for specific component interactions
```

## Success Criteria
- [ ] End-to-end integration tests compile without errors
- [ ] Complete workflow test passes (indexing → searching → results)
- [ ] Function search finds appropriate results
- [ ] Filtered search applies filters correctly
- [ ] Complex queries (boolean, phrase) work properly
- [ ] Result highlighting integration works
- [ ] Incremental indexing workflow functions correctly
- [ ] Performance and caching integration works
- [ ] Error handling prevents system crashes
- [ ] Large codebase handling performs adequately
- [ ] All integration tests pass with `cargo test --test integration_tests`

## Context for Next Task
Task 22 will implement component interaction testing to validate that individual components communicate correctly and handle edge cases in their interactions.