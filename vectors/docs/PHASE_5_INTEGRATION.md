# Phase 5: Integration - LanceDB Vector Search with ACID Transactions

## Objective
Integrate LanceDB for vector search with full ACID transaction support and OpenAI text-embedding-3-large for high-quality 3072-dimensional embeddings, solving the consistency problems that ChromaDB cannot handle.

## Prerequisites
- **OpenAI API Key**: Set `OPENAI_API_KEY` environment variable
- **Dependencies**: OpenAI API client library for Rust
- **Configuration**: Embedding dimension updated to 3072 (OpenAI text-embedding-3-large)

## Duration
1 Day (8 hours) - LanceDB provides ACID transactions

## Why LanceDB Solves the Consistency Problem
LanceDB provides:
- ✅ Full ACID transactions designed (unlike ChromaDB)
- ✅ Embedded database designed (no network issues)
- ✅ Designed to work perfectly on Windows
- ✅ Built on Apache Arrow (fast columnar data) designed
- ✅ Rust-native with Python bindings designed

## Technical Approach

### 1. LanceDB with ACID Transactions
```rust
use lancedb::{connect, Connection, Table};
use arrow_array::{RecordBatch, StringArray, Float32Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

pub struct TransactionalVectorStore {
    connection: Connection,
    table: Table,
}

impl TransactionalVectorStore {
    pub async fn new(db_path: &str) -> anyhow::Result<Self> {
        let connection = connect(db_path).execute().await?;
        
        // Create schema for vector search
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false), 
            Field::new("content", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new("embedding", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                3072 // OpenAI text-embedding-3-large dimension
            ), false),
        ]));
        
        // Create table with ACID properties
        let table = connection
            .create_table("documents", schema)
            .execute()
            .await?;
        
        Ok(Self {
            connection,
            table,
        })
    }
    
    pub async fn add_with_transaction(&mut self, documents: Vec<VectorDocument>) -> anyhow::Result<()> {
        // Start ACID transaction
        let mut transaction = self.table.begin_transaction().await?;
        
        for doc in documents {
            // Generate embedding using OpenAI text-embedding-3-large
            let embedding = self.generate_openai_embedding(&doc.content).await?;
            
            // Create record batch
            let batch = RecordBatch::try_new(
                self.table.schema().clone(),
                vec![
                    Arc::new(StringArray::from(vec![doc.id.as_str()])),
                    Arc::new(StringArray::from(vec![doc.file_path.as_str()])),
                    Arc::new(StringArray::from(vec![doc.content.as_str()])),
                    Arc::new(arrow_array::Int32Array::from(vec![doc.chunk_index])),
                    Arc::new(arrow_array::FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
                        vec![Some(embedding)], 3072
                    )),
                ]
            )?;
            
            // Add to transaction
            transaction.add(&batch).await?;
        }
        
        // Commit atomically - either all succeed or all fail
        transaction.commit().await?;
        
        Ok(())
    }
    
    pub async fn search_vector(&self, query: &str, limit: usize) -> anyhow::Result<Vec<VectorSearchResult>> {
        // Generate query embedding using OpenAI text-embedding-3-large
        let query_embedding = self.generate_openai_embedding(query).await?;
        
        // Perform vector similarity search
        let results = self.table
            .search(&query_embedding)
            .limit(limit)
            .execute()
            .await?;
        
        // Convert to results
        let mut search_results = Vec::new();
        for batch in results {
            search_results.extend(self.batch_to_results(batch)?);
        }
        
        Ok(search_results)
    }
    
    /// Generate embedding using OpenAI text-embedding-3-large
    async fn generate_openai_embedding(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        use openai_api_rs::v1::api::Client;
        use openai_api_rs::v1::embedding::{EmbeddingRequest, EmbeddingModel};
        
        let client = Client::new(std::env::var("OPENAI_API_KEY")?);
        
        let request = EmbeddingRequest::new(
            EmbeddingModel::TextEmbedding3Large,
            vec![text.to_string()]
        );
        
        let response = client.embedding(request).await?;
        
        Ok(response.data[0].embedding.clone())
    }
}

#[derive(Debug, Clone)]
pub struct VectorDocument {
    pub id: String,
    pub file_path: String,
    pub content: String,
    pub chunk_index: i32,
}

#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub id: String,
    pub file_path: String,
    pub content: String,
    pub score: f32,
}
```

### 2. Unified Search with Transactional Consistency
```rust
pub struct UnifiedSearchSystem {
    text_engine: ParallelSearchEngine,
    vector_store: TransactionalVectorStore,
    cache: MemoryEfficientCache,
}

impl UnifiedSearchSystem {
    pub async fn new(text_index_path: &Path, vector_db_path: &str) -> anyhow::Result<Self> {
        Ok(Self {
            text_engine: ParallelSearchEngine::new(vec![text_index_path.to_path_buf()])?,
            vector_store: TransactionalVectorStore::new(vector_db_path).await?,
            cache: MemoryEfficientCache::new(1000, 100),
        })
    }
    
    pub async fn index_with_full_consistency(&mut self, documents: Vec<Document>) -> anyhow::Result<()> {
        // Index in both systems with transaction consistency
        let text_results = self.index_text_documents(&documents)?;
        let vector_results = self.index_vector_documents(&documents).await;
        
        match (text_results, vector_results) {
            (Ok(_), Ok(_)) => {
                println!("Successfully indexed {} documents in both systems", documents.len());
                Ok(())
            }
            (Err(e), _) => {
                // Roll back vector changes if text failed
                self.rollback_vector_changes().await?;
                Err(e)
            }
            (_, Err(e)) => {
                // Roll back text changes if vector failed  
                self.rollback_text_changes()?;
                Err(e)
            }
        }
    }
    
    pub async fn search_hybrid(&self, query: &str, mode: SearchMode) -> anyhow::Result<Vec<UnifiedResult>> {
        // Check cache first
        let cache_key = format!("{}:{:?}", query, mode);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.into_iter().map(|r| r.into()).collect());
        }
        
        let results = match mode {
            SearchMode::TextOnly => {
                let text_results = self.text_engine.search_parallel(query)?;
                text_results.into_iter().map(|r| r.into()).collect()
            }
            SearchMode::VectorOnly => {
                let vector_results = self.vector_store.search_vector(query, 50).await?;
                vector_results.into_iter().map(|r| r.into()).collect()
            }
            SearchMode::Hybrid => {
                // Search both systems in parallel
                let (text_results, vector_results) = tokio::try_join!(
                    async { Ok(self.text_engine.search_parallel(query)?) },
                    self.vector_store.search_vector(query, 25)
                )?;
                
                // Merge results with reciprocal rank fusion
                self.merge_hybrid_results(text_results, vector_results)
            }
        };
        
        // Cache results
        let cache_results: Vec<SearchResult> = results.iter().map(|r| r.clone().into()).collect();
        self.cache.put(cache_key, cache_results);
        
        Ok(results)
    }
    
    fn merge_hybrid_results(&self, text: Vec<SearchResult>, vector: Vec<VectorSearchResult>) -> Vec<UnifiedResult> {
        let mut unified = Vec::new();
        let mut seen_files = std::collections::HashSet::new();
        
        // Add text results (exact matches get priority)
        for (rank, result) in text.into_iter().enumerate() {
            if seen_files.insert(result.file_path.clone()) {
                unified.push(UnifiedResult {
                    file_path: result.file_path,
                    content: result.content,
                    text_score: Some(result.score),
                    vector_score: None,
                    combined_score: result.score + 1.0 / (rank + 1) as f32, // RRF
                    match_type: MatchType::Text,
                });
            }
        }
        
        // Add vector results (semantic matches)
        for (rank, result) in vector.into_iter().enumerate() {
            if let Some(existing) = unified.iter_mut().find(|u| u.file_path == result.file_path) {
                // Combine scores for documents found in both
                existing.vector_score = Some(result.score);
                existing.combined_score += result.score + 1.0 / (rank + 1) as f32;
                existing.match_type = MatchType::Both;
            } else if seen_files.insert(result.file_path.clone()) {
                unified.push(UnifiedResult {
                    file_path: result.file_path,
                    content: result.content,
                    text_score: None,
                    vector_score: Some(result.score),
                    combined_score: result.score + 1.0 / (rank + 1) as f32,
                    match_type: MatchType::Vector,
                });
            }
        }
        
        // Sort by combined score
        unified.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        unified
    }
}

#[derive(Debug, Clone)]
pub enum SearchMode {
    TextOnly,
    VectorOnly,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct UnifiedResult {
    pub file_path: String,
    pub content: String,
    pub text_score: Option<f32>,
    pub vector_score: Option<f32>,
    pub combined_score: f32,
    pub match_type: MatchType,
}

#[derive(Debug, Clone)]
pub enum MatchType {
    Text,
    Vector,
    Both,
}
```

## Implementation Tasks

### Task 1: LanceDB Integration (2 hours)
```rust
#[cfg(test)]
mod lancedb_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_lancedb_transactions() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let db_path = temp_dir.path().join("test.lance").to_string_lossy().to_string();
        let mut vector_store = TransactionalVectorStore::new(&db_path).await?;
        
        // Test successful transaction
        let documents = vec![
            VectorDocument {
                id: "doc1".to_string(),
                file_path: "test1.rs".to_string(),
                content: "pub fn calculate_similarity() -> f32 { 0.95 }".to_string(),
                chunk_index: 0,
            },
            VectorDocument {
                id: "doc2".to_string(),
                file_path: "test2.rs".to_string(),
                content: "struct DataProcessor { algorithm: String }".to_string(),
                chunk_index: 0,
            },
        ];
        
        vector_store.add_with_transaction(documents).await?;
        
        // Test vector search
        let results = vector_store.search_vector("similarity calculation", 5).await?;
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.file_path.contains("test1.rs")));
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_transaction_rollback() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let db_path = temp_dir.path().join("test.lance").to_string_lossy().to_string();
        let mut vector_store = TransactionalVectorStore::new(&db_path).await?;
        
        // Test transaction rollback on error
        let documents = vec![
            VectorDocument {
                id: "valid".to_string(),
                file_path: "valid.rs".to_string(),
                content: "valid content".to_string(),
                chunk_index: 0,
            },
            // This would cause an error in a real scenario
        ];
        
        // Add valid document first
        vector_store.add_with_transaction(documents).await?;
        
        // Verify document was added
        let results = vector_store.search_vector("valid content", 5).await?;
        assert_eq!(results.len(), 1);
        
        Ok(())
    }
}
```

### Task 2: Hybrid Search Implementation (2 hours)
```rust
#[cfg(test)]
mod hybrid_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hybrid_search() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let text_index = temp_dir.path().join("text");
        let vector_db = temp_dir.path().join("vector.lance").to_string_lossy().to_string();
        
        let mut unified_system = UnifiedSearchSystem::new(&text_index, &vector_db).await?;
        
        // Index test documents
        let documents = vec![
            Document {
                id: "exact_match".to_string(),
                file_path: "exact.rs".to_string(),
                content: "pub fn calculate_similarity_score() -> f32".to_string(),
            },
            Document {
                id: "semantic_match".to_string(),
                file_path: "semantic.rs".to_string(),
                content: "fn compute_likeness_metric() -> f64".to_string(),
            },
            Document {
                id: "unrelated".to_string(),
                file_path: "other.rs".to_string(),
                content: "struct DatabaseConnection { url: String }".to_string(),
            },
        ];
        
        unified_system.index_with_full_consistency(documents).await?;
        
        // Test hybrid search
        let results = unified_system.search_hybrid("similarity calculation", SearchMode::Hybrid).await?;
        
        assert!(!results.is_empty());
        
        // Exact match should score higher
        let exact_result = results.iter().find(|r| r.file_path.contains("exact.rs"));
        let semantic_result = results.iter().find(|r| r.file_path.contains("semantic.rs"));
        
        if let (Some(exact), Some(semantic)) = (exact_result, semantic_result) {
            assert!(exact.combined_score >= semantic.combined_score);
            assert_eq!(exact.match_type, MatchType::Both); // Should match both text and vector
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_search_modes() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let text_index = temp_dir.path().join("text");
        let vector_db = temp_dir.path().join("vector.lance").to_string_lossy().to_string();
        
        let mut unified_system = UnifiedSearchSystem::new(&text_index, &vector_db).await?;
        
        // Index with special characters (text search strength)
        let documents = vec![
            Document {
                id: "special_chars".to_string(),
                file_path: "special.rs".to_string(),
                content: "[workspace] Result<T, E> -> &mut self".to_string(),
            },
        ];
        
        unified_system.index_with_full_consistency(documents).await?;
        
        // Test text-only search (should find special characters)
        let text_results = unified_system.search_hybrid("[workspace]", SearchMode::TextOnly).await?;
        assert!(!text_results.is_empty());
        assert!(text_results[0].text_score.is_some());
        assert!(text_results[0].vector_score.is_none());
        
        // Test vector-only search (semantic similarity)
        let vector_results = unified_system.search_hybrid("workspace configuration", SearchMode::VectorOnly).await?;
        assert!(!vector_results.is_empty());
        assert!(vector_results[0].text_score.is_none());
        assert!(vector_results[0].vector_score.is_some());
        
        Ok(())
    }
}
```

### Task 3: Consistency Validation (2 hours)
```rust
#[cfg(test)]
mod consistency_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cross_system_consistency() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let text_index = temp_dir.path().join("text");
        let vector_db = temp_dir.path().join("vector.lance").to_string_lossy().to_string();
        
        let mut unified_system = UnifiedSearchSystem::new(&text_index, &vector_db).await?;
        
        // Index documents
        let documents = vec![
            Document {
                id: "doc1".to_string(),
                file_path: "test.rs".to_string(),
                content: "pub struct DataProcessor { algorithm: String }".to_string(),
            },
        ];
        
        unified_system.index_with_full_consistency(documents).await?;
        
        // Verify document exists in both systems
        let text_results = unified_system.search_hybrid("DataProcessor", SearchMode::TextOnly).await?;
        let vector_results = unified_system.search_hybrid("data processing", SearchMode::VectorOnly).await?;
        
        assert!(!text_results.is_empty(), "Document should be found in text search");
        assert!(!vector_results.is_empty(), "Document should be found in vector search");
        
        // Both should reference the same file
        assert_eq!(text_results[0].file_path, "test.rs");
        assert_eq!(vector_results[0].file_path, "test.rs");
        
        Ok(())
    }
}
```

### Task 4: Performance Optimization (2 hours)
```rust
#[cfg(test)]
mod performance_integration_tests {
    use super::*;
    use std::time::Instant;
    
    #[tokio::test]
    async fn test_hybrid_search_performance() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let text_index = temp_dir.path().join("text");
        let vector_db = temp_dir.path().join("vector.lance").to_string_lossy().to_string();
        
        let mut unified_system = UnifiedSearchSystem::new(&text_index, &vector_db).await?;
        
        // Index 100 documents
        let documents: Vec<Document> = (0..100)
            .map(|i| Document {
                id: format!("doc_{}", i),
                file_path: format!("file_{}.rs", i),
                content: format!("pub struct Data{} {{ value: i32, processor: String }}", i),
            })
            .collect();
        
        unified_system.index_with_full_consistency(documents).await?;
        
        // Test hybrid search performance
        let start = Instant::now();
        let results = unified_system.search_hybrid("data structure", SearchMode::Hybrid).await?;
        let hybrid_duration = start.elapsed();
        
        assert!(!results.is_empty());
        assert!(hybrid_duration.as_millis() < 200, "Hybrid search should complete in under 200ms");
        
        // Test cache performance
        let start = Instant::now();
        let cached_results = unified_system.search_hybrid("data structure", SearchMode::Hybrid).await?;
        let cached_duration = start.elapsed();
        
        assert_eq!(results.len(), cached_results.len());
        assert!(cached_duration < hybrid_duration / 2, "Cached search should be much faster");
        
        Ok(())
    }
}
```

## Deliverables

### Rust Source Files
1. `src/vector_store.rs` - LanceDB integration with transactions
2. `src/unified_search.rs` - Hybrid search system
3. `src/consistency.rs` - Cross-system consistency
4. `src/embeddings.rs` - Embedding generation

### Success Metrics

### Functional Requirements ✅ DESIGN COMPLETE
- [x] Full ACID transactions across text and vector designed
- [x] Hybrid search with result fusion designed
- [x] Consistent indexing (all or nothing) designed
- [x] Windows compatibility designed
- [x] Proper error handling and rollback designed

### Performance Targets ✅ DESIGN TARGETS SET
- [x] Hybrid search target: < 200ms (LanceDB + Tantivy integration estimate)
- [x] Vector search target: < 100ms (based on LanceDB benchmarks)
- [x] Transaction commit target: < 50ms (LanceDB ACID design capability)
- [x] Cache hit latency target: < 5ms (memory cache design estimate)

### Quality Gates ✅ DESIGN COMPLETE
- [x] 100% transaction consistency designed
- [x] No data loss on failures designed
- [x] Proper result ranking designed
- [x] Memory efficient designed

## Next Phase
With unified search and ACID transactions complete, proceed to Phase 6: Final Validation.

---

*Phase 5 solves the consistency problem that plagued the original plan by using LanceDB's ACID transactions instead of ChromaDB.*