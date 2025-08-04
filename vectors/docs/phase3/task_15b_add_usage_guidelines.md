# Task 15b: Add Usage Guidelines

**Estimated Time: 8 minutes**  
**Lines of Code: ~25**
**Prerequisites: Task 15a completion**

## Context
After establishing basic examples, developers need comprehensive usage guidelines that cover performance characteristics, best practices, and common patterns. This task creates documentation that helps developers use the vector search system efficiently and avoid common pitfalls.

## Your Task
Create a comprehensive usage guidelines document that covers performance characteristics, memory management, indexing strategies, and search optimization techniques.

## Required Implementation

```markdown
# Vector Search Usage Guidelines

## Performance Characteristics

### Memory Management
- **Index Memory**: Plan for 2-3x document size in RAM for optimal performance
- **Batch Size**: Use batch sizes of 1000-10000 documents for best throughput
- **Commit Strategy**: Manual commits every 10k-100k documents for production

```rust
// Recommended memory configuration
let config = IndexConfig::new()
    .with_memory_budget(1_000_000_000) // 1GB for large datasets
    .with_commit_on_add(false)         // Manual commit control
    .with_merge_policy_log_merge(10);  // Control segment merging
```

### Indexing Performance Guidelines

#### Document Size Optimization
- **Small Documents** (<1KB): Batch in groups of 10,000
- **Medium Documents** (1KB-100KB): Batch in groups of 1,000  
- **Large Documents** (>100KB): Process individually with streaming

```rust
async fn optimized_indexing(documents: Vec<Document>) -> Result<(), SearchError> {
    let mut engine = SearchEngine::new(config).await?;
    let batch_size = calculate_optimal_batch_size(&documents);
    
    for chunk in documents.chunks(batch_size) {
        engine.add_documents_batch(chunk.to_vec()).await?;
        
        // Commit periodically, not on every batch
        if chunk.len() % 10000 == 0 {
            engine.commit().await?;
        }
    }
    
    engine.commit().await?; // Final commit
    Ok(())
}

fn calculate_optimal_batch_size(documents: &[Document]) -> usize {
    let avg_size = documents.iter()
        .map(|d| d.estimated_size())
        .sum::<usize>() / documents.len();
        
    match avg_size {
        0..=1024 => 10000,      // Small docs
        1025..=102400 => 1000,  // Medium docs  
        _ => 100                // Large docs
    }
}
```

## Search Optimization Best Practices

### Query Performance
1. **Use specific field targeting** when possible
2. **Limit result sets** with appropriate limits
3. **Cache frequent queries** at application level
4. **Use filters before text search** for better performance

```rust
// Optimized query pattern
let query = SearchQuery::new("machine learning")
    .with_field_boost("title", 2.0)        // Boost title matches
    .with_limit(20)                        // Reasonable limit
    .with_filter("category", "AI")         // Pre-filter
    .with_explain(false);                  // Disable unless debugging

let results = engine.search(&query).await?;
```

### Memory Usage Patterns
- **Reader Instances**: Reuse searcher instances across queries
- **Index Warming**: Perform dummy searches after index updates
- **Memory Monitoring**: Track RSS memory usage during indexing

## Common Anti-Patterns to Avoid

### ❌ Inefficient Patterns
```rust
// DON'T: Commit after every document
for doc in documents {
    engine.add_document(doc).await?;
    engine.commit().await?; // Expensive!
}

// DON'T: Create new engine instances repeatedly  
for query in queries {
    let engine = SearchEngine::new(config).await?; // Wasteful!
    let results = engine.search(&query).await?;
}
```

### ✅ Efficient Patterns
```rust
// DO: Batch commits
let mut count = 0;
for doc in documents {
    engine.add_document(doc).await?;
    count += 1;
    
    if count % 1000 == 0 {
        engine.commit().await?;
    }
}

// DO: Reuse engine instances
let engine = SearchEngine::new(config).await?;
for query in queries {
    let results = engine.search(&query).await?;
    // Process results...
}
```

## Resource Management

### Connection Lifecycle
```rust
pub struct SearchManager {
    engine: SearchEngine,
    config: IndexConfig,
}

impl SearchManager {
    pub async fn new() -> Result<Self, SearchError> {
        let config = IndexConfig::production_defaults();
        let engine = SearchEngine::new(config.clone()).await?;
        
        Ok(Self { engine, config })
    }
    
    pub async fn health_check(&self) -> bool {
        self.engine.index_health().await.is_ok()
    }
    
    pub async fn refresh_if_needed(&mut self) -> Result<(), SearchError> {
        if self.engine.needs_refresh().await? {
            self.engine.refresh().await?;
        }
        Ok(())
    }
}
```

## Production Deployment Guidelines

### Configuration for Production
- Set appropriate memory limits based on available RAM
- Configure logging for performance monitoring
- Enable metric collection for operational insights
- Use connection pooling for high-concurrency scenarios

### Monitoring Key Metrics
- Query latency (p50, p95, p99)
- Index size and growth rate
- Memory usage patterns
- Error rates and types

These guidelines ensure optimal performance and reliable operation in production environments.
```

## Success Criteria
- [ ] Performance characteristics documented with specific numbers
- [ ] Memory management guidelines with code examples
- [ ] Indexing optimization strategies provided
- [ ] Search performance best practices outlined
- [ ] Anti-patterns identified with corrections
- [ ] Production deployment guidance included

## Validation
Verify guidelines accuracy:
```bash
# Test memory recommendations with sample data
cargo run --example memory_profiling

# Validate batch size recommendations  
cargo run --example batch_optimization
```

## Next Task
Task 15c will add proximity and phrase search examples to demonstrate advanced search capabilities.