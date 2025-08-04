# Task 036: Implement Tantivy Baseline

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The Tantivy baseline provides performance comparison against a standalone Tantivy implementation to isolate Tantivy's performance characteristics from the integrated system's overhead.

## Project Structure
```
src/
  validation/
    baseline.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the standalone Tantivy performance baseline by creating temporary indices, measuring indexing speed, and executing search operations. This provides a pure Tantivy performance baseline independent of system integration overhead.

## Requirements
1. Extend `src/validation/baseline.rs` with Tantivy standalone implementation
2. Create temporary Tantivy indices for testing
3. Measure both indexing time and search performance
4. Implement proper index cleanup and resource management
5. Handle memory usage measurement accurately
6. Support different schema configurations for comparison
7. Provide indexing speed metrics (files per second)

## Expected Code Structure
```rust
// Add to baseline.rs file

use tantivy::{
    collector::TopDocs,
    directory::MmapDirectory,
    doc,
    query::QueryParser,
    schema::{Schema, TextField, FAST, STORED, TEXT},
    Index, IndexWriter, ReloadPolicy,
};
use std::fs;
use std::path::Path;

impl BaselineBenchmark {
    async fn run_tantivy_standalone(&self, query: &str, start_time: Instant) -> Result<BaselineResult> {
        let query_start = Instant::now();
        
        // Create temporary index directory
        let temp_index_dir = self.config.temp_dir.join(format!("tantivy_baseline_{}", 
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis()));
        
        std::fs::create_dir_all(&temp_index_dir)
            .context("Failed to create temporary Tantivy index directory")?;
        
        let result = match self.create_and_search_tantivy_index(&temp_index_dir, query, query_start).await {
            Ok(result) => result,
            Err(e) => BaselineResult {
                tool: BaselineTool::TantivyStandalone,
                query: query.to_string(),
                execution_time: query_start.elapsed(),
                results_count: 0,
                success: false,
                error_message: Some(e.to_string()),
                memory_usage_mb: 0.0,
                index_time: None,
            }
        };
        
        // Cleanup temporary index
        if temp_index_dir.exists() {
            let _ = std::fs::remove_dir_all(&temp_index_dir);
        }
        
        Ok(result)
    }
    
    async fn create_and_search_tantivy_index(
        &self,
        index_dir: &Path,
        query: &str,
        query_start: Instant
    ) -> Result<BaselineResult> {
        // Create schema
        let schema = self.create_tantivy_schema();
        
        // Create index
        let directory = MmapDirectory::open(index_dir)
            .context("Failed to open Tantivy directory")?;
        let index = Index::open_or_create(directory, schema.clone())
            .context("Failed to create Tantivy index")?;
        
        // Index documents and measure indexing time
        let indexing_start = Instant::now();
        let (indexed_docs, indexing_memory) = self.index_documents_tantivy(&index, &schema).await?;
        let indexing_time = indexing_start.elapsed();
        
        // Perform search and measure search time
        let search_start = Instant::now();
        let (results_count, search_memory) = self.search_tantivy_index(&index, &schema, query).await?;
        let search_time = search_start.elapsed();
        
        let total_execution_time = query_start.elapsed();
        let max_memory = indexing_memory.max(search_memory);
        
        println!("Tantivy baseline - Indexed {} docs in {:?}, searched in {:?}", 
                 indexed_docs, indexing_time, search_time);
        
        Ok(BaselineResult {
            tool: BaselineTool::TantivyStandalone,
            query: query.to_string(),
            execution_time: total_execution_time,
            results_count,
            success: true,
            error_message: None,
            memory_usage_mb: max_memory,
            index_time: Some(indexing_time),
        })
    }
    
    fn create_tantivy_schema(&self) -> Schema {
        let mut schema_builder = Schema::builder();
        
        // Add text field for content - this matches typical search use cases
        schema_builder.add_text_field("content", TEXT | STORED);
        
        // Add path field for file identification
        schema_builder.add_text_field("path", STORED | FAST);
        
        // Add title field if we extract it
        schema_builder.add_text_field("title", TEXT | STORED | FAST);
        
        schema_builder.build()
    }
    
    async fn index_documents_tantivy(&self, index: &Index, schema: &Schema) -> Result<(usize, f64)> {
        let mut index_writer = index.writer(50_000_000) // 50MB heap
            .context("Failed to create Tantivy index writer")?;
        
        let content_field = schema.get_field("content")
            .context("Content field not found in schema")?;
        let path_field = schema.get_field("path")
            .context("Path field not found in schema")?;
        let title_field = schema.get_field("title")
            .context("Title field not found in schema")?;
        
        let mut indexed_count = 0;
        let mut max_memory = 0.0f64;
        
        // Track memory usage during indexing
        let memory_monitor = tokio::spawn(async move {
            let mut system = sysinfo::System::new();
            let mut max_mem = 0.0f64;
            
            for _ in 0..100 { // Monitor for a reasonable time
                system.refresh_memory();
                let current_mem = system.used_memory() as f64 / (1024.0 * 1024.0);
                max_mem = max_mem.max(current_mem);
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            
            max_mem
        });
        
        // Recursively index files in test data directory
        indexed_count = self.index_directory_recursive(&mut index_writer, &self.test_data_dir, 
                                                      content_field, path_field, title_field).await?;
        
        // Commit the index
        index_writer.commit()
            .context("Failed to commit Tantivy index")?;
        
        // Get memory usage
        memory_monitor.abort();
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        max_memory = system.used_memory() as f64 / (1024.0 * 1024.0);
        
        Ok((indexed_count, max_memory))
    }
    
    async fn index_directory_recursive(
        &self,
        index_writer: &mut IndexWriter,
        dir: &Path,
        content_field: tantivy::schema::Field,
        path_field: tantivy::schema::Field,
        title_field: tantivy::schema::Field,
    ) -> Result<usize> {
        let mut count = 0;
        
        if !dir.is_dir() {
            return Ok(0);
        }
        
        let entries = fs::read_dir(dir)
            .context("Failed to read directory")?;
        
        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();
            
            if path.is_dir() {
                // Recursively index subdirectories
                count += self.index_directory_recursive(index_writer, &path, 
                                                      content_field, path_field, title_field).await?;
            } else if self.is_text_file(&path) {
                // Index text files
                if let Ok(content) = fs::read_to_string(&path) {
                    let title = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("untitled");
                    
                    let path_str = path.to_string_lossy();
                    
                    let doc = doc!(
                        content_field => content,
                        path_field => path_str.as_ref(),
                        title_field => title
                    );
                    
                    index_writer.add_document(doc)
                        .context("Failed to add document to Tantivy index")?;
                    
                    count += 1;
                    
                    // Commit periodically to avoid memory buildup
                    if count % 1000 == 0 {
                        index_writer.commit()
                            .context("Failed to commit Tantivy index during indexing")?;
                    }
                }
            }
        }
        
        Ok(count)
    }
    
    fn is_text_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            let ext = extension.to_string_lossy().to_lowercase();
            matches!(ext.as_str(), 
                "txt" | "md" | "rst" | "log" | "csv" | "json" | "xml" | "html" | 
                "js" | "ts" | "rs" | "py" | "java" | "cpp" | "c" | "h" |
                "yml" | "yaml" | "toml" | "ini" | "cfg" | "conf")
        } else {
            // Files without extension - check if they look like text
            if let Ok(content) = fs::read(path) {
                // Simple heuristic: if more than 95% of bytes are printable ASCII, consider it text
                let printable_count = content.iter()
                    .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
                    .count();
                
                content.len() > 0 && (printable_count as f64 / content.len() as f64) > 0.95
            } else {
                false
            }
        }
    }
    
    async fn search_tantivy_index(&self, index: &Index, schema: &Schema, query: &str) -> Result<(usize, f64)> {
        let reader = index.reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()
            .context("Failed to create Tantivy index reader")?;
        
        let searcher = reader.searcher();
        
        let content_field = schema.get_field("content")
            .context("Content field not found")?;
        
        let query_parser = QueryParser::for_index(index, vec![content_field]);
        
        // Parse the query - handle potential parsing errors
        let parsed_query = match query_parser.parse_query(query) {
            Ok(q) => q,
            Err(_) => {
                // If query parsing fails, try a simpler term query
                let term_query = tantivy::query::TermQuery::new(
                    tantivy::Term::from_field_text(content_field, query),
                    tantivy::schema::IndexRecordOption::Basic
                );
                Box::new(term_query)
            }
        };
        
        let mut max_memory = 0.0f64;
        let mut system = sysinfo::System::new();
        
        // Measure memory before search
        system.refresh_memory();
        let pre_search_memory = system.used_memory() as f64 / (1024.0 * 1024.0);
        
        // Execute search - get top 10000 results to ensure we capture all matches
        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(10000))
            .context("Failed to execute Tantivy search")?;
        
        let results_count = top_docs.len();
        
        // Measure memory after search
        system.refresh_memory();
        let post_search_memory = system.used_memory() as f64 / (1024.0 * 1024.0);
        max_memory = post_search_memory.max(pre_search_memory);
        
        Ok((results_count, max_memory))
    }
    
    // Advanced search with different query types
    async fn search_tantivy_advanced(&self, index: &Index, schema: &Schema, query: &str) -> Result<(usize, f64)> {
        let reader = index.reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()
            .context("Failed to create Tantivy index reader")?;
        
        let searcher = reader.searcher();
        
        let content_field = schema.get_field("content")?;
        let title_field = schema.get_field("title")?;
        
        let query_parser = QueryParser::new(
            index.schema(),
            vec![content_field, title_field],
            index.tokenizers().clone(),
        );
        
        // Try different query parsing strategies
        let queries_to_try = vec![
            query.to_string(),                           // Original query
            format!("\"{}\"", query),                    // Phrase query
            query.split_whitespace().collect::<Vec<_>>()
                .join(" AND "),                          // AND query
            query.split_whitespace().collect::<Vec<_>>()
                .join(" OR "),                           // OR query
        ];
        
        let mut best_results_count = 0;
        let mut max_memory = 0.0f64;
        let mut system = sysinfo::System::new();
        
        for query_variant in queries_to_try {
            if let Ok(parsed_query) = query_parser.parse_query(&query_variant) {
                system.refresh_memory();
                let pre_memory = system.used_memory() as f64 / (1024.0 * 1024.0);
                
                if let Ok(top_docs) = searcher.search(&parsed_query, &TopDocs::with_limit(10000)) {
                    best_results_count = best_results_count.max(top_docs.len());
                    
                    system.refresh_memory();
                    let post_memory = system.used_memory() as f64 / (1024.0 * 1024.0);
                    max_memory = max_memory.max(post_memory).max(pre_memory);
                }
            }
        }
        
        Ok((best_results_count, max_memory))
    }
    
    // Test method to validate Tantivy standalone functionality
    pub async fn test_tantivy_standalone(&self) -> Result<bool> {
        let test_dir = self.config.temp_dir.join("tantivy_test");
        std::fs::create_dir_all(&test_dir)?;
        
        // Create a test file
        let test_file = test_dir.join("test.txt");
        std::fs::write(&test_file, "This is a test document for Tantivy validation\ntest content here")?;
        
        // Try to create and search a small index
        let temp_index_dir = self.config.temp_dir.join("tantivy_test_index");
        std::fs::create_dir_all(&temp_index_dir)?;
        
        let result = match self.create_and_search_tantivy_index(&temp_index_dir, "test", Instant::now()).await {
            Ok(baseline_result) => baseline_result.success && baseline_result.results_count > 0,
            Err(_) => false,
        };
        
        // Cleanup
        let _ = std::fs::remove_dir_all(&test_dir);
        let _ = std::fs::remove_dir_all(&temp_index_dir);
        
        Ok(result)
    }
    
    // Method to get Tantivy indexing performance metrics
    pub async fn measure_tantivy_indexing_performance(&self, data_dir: &Path) -> Result<TantivyIndexingMetrics> {
        let temp_index_dir = self.config.temp_dir.join("tantivy_indexing_perf");
        std::fs::create_dir_all(&temp_index_dir)?;
        
        let schema = self.create_tantivy_schema();
        let directory = MmapDirectory::open(&temp_index_dir)?;
        let index = Index::open_or_create(directory, schema.clone())?;
        
        let indexing_start = Instant::now();
        let (indexed_docs, max_memory) = self.index_documents_tantivy(&index, &schema).await?;
        let indexing_time = indexing_start.elapsed();
        
        let indexing_rate = if indexing_time.as_secs_f64() > 0.0 {
            indexed_docs as f64 / indexing_time.as_secs_f64()
        } else {
            0.0
        };
        
        // Get index size
        let index_size_bytes = self.calculate_directory_size(&temp_index_dir)?;
        
        let metrics = TantivyIndexingMetrics {
            documents_indexed: indexed_docs,
            indexing_time,
            indexing_rate_docs_per_sec: indexing_rate,
            index_size_mb: index_size_bytes as f64 / (1024.0 * 1024.0),
            peak_memory_mb: max_memory,
        };
        
        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_index_dir);
        
        Ok(metrics)
    }
    
    fn calculate_directory_size(&self, dir: &Path) -> Result<u64> {
        let mut size = 0;
        
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    size += self.calculate_directory_size(&path)?;
                } else {
                    size += entry.metadata()?.len();
                }
            }
        }
        
        Ok(size)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TantivyIndexingMetrics {
    pub documents_indexed: usize,
    pub indexing_time: Duration,
    pub indexing_rate_docs_per_sec: f64,
    pub index_size_mb: f64,
    pub peak_memory_mb: f64,
}
```

## Dependencies to Add
```toml
[dependencies]
tantivy = "0.21"
```

## Success Criteria
- Tantivy standalone indexing works correctly
- Search functionality returns accurate result counts
- Memory usage monitoring provides meaningful data
- Indexing performance metrics are calculated accurately
- Index cleanup prevents disk space leaks
- Error handling covers schema creation, indexing, and search failures
- Performance measurements are consistent and reliable

## Time Limit
10 minutes maximum