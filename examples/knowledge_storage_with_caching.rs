use llmkg::enhanced_knowledge_storage::production::caching::*;
use serde::{Serialize, Deserialize};
use std::time::Duration;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDocument {
    pub id: String,
    pub chunks: Vec<DocumentChunk>,
    pub summary: String,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub query: String,
    pub relevant_chunks: Vec<DocumentChunk>,
    pub reasoning_chain: Vec<String>,
    pub confidence_score: f64,
}

/// Example Enhanced Knowledge Storage System with Caching
pub struct CachedKnowledgeStorage {
    cache: MultiLevelCache,
}

impl CachedKnowledgeStorage {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let cache = CacheConfigBuilder::new()
            .l1_capacity(1000)
            .l1_max_bytes(50 * 1024 * 1024)  // 50MB L1
            .l2_cache_dir("./knowledge_cache")
            .l2_max_bytes(500 * 1024 * 1024) // 500MB L2
            .write_strategy(WriteStrategy::WriteThrough)
            .compression_level(6)
            .build()
            .await?;
        
        Ok(Self { cache })
    }

    /// Process a document with intelligent caching
    pub async fn process_document(&self, document_id: &str, raw_content: &str) -> Result<ProcessedDocument, Box<dyn std::error::Error + Send + Sync>> {
        let cache_key = format!("processed_doc:{}", document_id);
        
        // Try to get from cache first
        if let Some(cached_doc) = self.cache.get::<ProcessedDocument>(&cache_key).await {
            println!("✓ Retrieved processed document from cache: {}", document_id);
            return Ok(cached_doc);
        }
        
        println!("Processing document from scratch: {}", document_id);
        
        // Simulate document processing
        let processed_doc = self.simulate_document_processing(document_id, raw_content).await;
        
        // Cache the result with 24-hour TTL
        self.cache.put(
            cache_key,
            processed_doc.clone(),
            Some(Duration::from_secs(86400))
        ).await;
        
        println!("✓ Document processed and cached: {}", document_id);
        Ok(processed_doc)
    }

    /// Execute a query with caching for reasoning results
    pub async fn execute_query(&self, query: &str) -> Result<QueryResult, Box<dyn std::error::Error + Send + Sync>> {
        let query_hash = self.calculate_query_hash(query);
        let cache_key = format!("query_result:{}", query_hash);
        
        // Check cache first
        if let Some(cached_result) = self.cache.get::<QueryResult>(&cache_key).await {
            println!("✓ Retrieved query result from cache for: {}", query);
            return Ok(cached_result);
        }
        
        println!("Executing query from scratch: {}", query);
        
        // Simulate query execution with reasoning
        let query_result = self.simulate_query_execution(query).await;
        
        // Cache with shorter TTL for query results (1 hour)
        self.cache.put(
            cache_key,
            query_result.clone(),
            Some(Duration::from_secs(3600))
        ).await;
        
        println!("✓ Query executed and result cached");
        Ok(query_result)
    }

    /// Get semantic embeddings with caching
    pub async fn get_embeddings(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let text_hash = self.calculate_text_hash(text);
        let cache_key = format!("embedding:{}", text_hash);
        
        if let Some(cached_embedding) = self.cache.get::<Vec<f32>>(&cache_key).await {
            return Ok(cached_embedding);
        }
        
        // Simulate embedding computation
        let embedding = self.simulate_embedding_computation(text).await;
        
        // Cache embeddings with 2-hour TTL
        self.cache.put(
            cache_key,
            embedding.clone(),
            Some(Duration::from_secs(7200))
        ).await;
        
        Ok(embedding)
    }

    /// Invalidate cache entries for a specific document
    pub async fn invalidate_document_cache(&self, document_id: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let pattern = format!("^(processed_doc|embedding):.*{}.*", regex::escape(document_id));
        let invalidated = self.cache.invalidate_pattern(&pattern).await?;
        println!("Invalidated {} cache entries for document: {}", invalidated, document_id);
        Ok(invalidated)
    }

    /// Get cache performance statistics
    pub async fn get_cache_stats(&self) -> CacheStatistics {
        self.cache.get_statistics().await
    }

    /// Warm cache with frequently accessed documents
    pub async fn warm_cache_with_popular_documents(&self, document_ids: Vec<String>) {
        let mut warmup_data = Vec::new();
        
        for doc_id in document_ids {
            // Simulate loading popular document data
            let doc_data = format!("Popular document content for {}", doc_id);
            let processed_doc = self.simulate_document_processing(&doc_id, &doc_data).await;
            
            if let Ok(serialized) = bincode::serialize(&processed_doc) {
                warmup_data.push((format!("processed_doc:{}", doc_id), serialized));
            }
        }
        
        self.cache.warm_cache(warmup_data).await;
        println!("✓ Cache warmed with popular documents");
    }

    // Simulation helpers
    async fn simulate_document_processing(&self, doc_id: &str, content: &str) -> ProcessedDocument {
        tokio::time::sleep(Duration::from_millis(100)).await; // Simulate processing time
        
        let chunks = vec![
            DocumentChunk {
                id: format!("{}_chunk_1", doc_id),
                content: content[..content.len().min(100)].to_string(),
                embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5],
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("type".to_string(), "text".to_string());
                    map.insert("source".to_string(), doc_id.to_string());
                    map
                },
            }
        ];
        
        ProcessedDocument {
            id: doc_id.to_string(),
            chunks,
            summary: format!("Summary of document {}", doc_id),
            processing_time_ms: 100,
        }
    }

    async fn simulate_query_execution(&self, query: &str) -> QueryResult {
        tokio::time::sleep(Duration::from_millis(50)).await; // Simulate reasoning time
        
        QueryResult {
            query: query.to_string(),
            relevant_chunks: vec![], // Would contain actual matching chunks
            reasoning_chain: vec![
                "Analyzed query intent".to_string(),
                "Retrieved relevant documents".to_string(),
                "Applied reasoning algorithms".to_string(),
                "Ranked results by relevance".to_string(),
            ],
            confidence_score: 0.85,
        }
    }

    async fn simulate_embedding_computation(&self, text: &str) -> Vec<f32> {
        tokio::time::sleep(Duration::from_millis(20)).await; // Simulate embedding computation
        
        // Simple hash-based embedding simulation
        let mut embedding = vec![0.0; 384]; // Typical embedding dimension
        let text_bytes = text.as_bytes();
        for (i, &byte) in text_bytes.iter().enumerate() {
            embedding[i % 384] += byte as f32 / 255.0;
        }
        embedding
    }

    fn calculate_query_hash(&self, query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn calculate_text_hash(&self, text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("🧠 Enhanced Knowledge Storage System with Caching Demo");
    println!("=" .repeat(60));

    // Initialize the cached knowledge storage system
    let knowledge_storage = CachedKnowledgeStorage::new().await?;
    println!("✓ Knowledge storage system with caching initialized");

    // Example 1: Document Processing with Caching
    println!("\n📄 Document Processing with Caching");
    println!("-" .repeat(40));
    
    let doc1_id = "research_paper_001";
    let doc1_content = "This is a research paper about artificial intelligence and machine learning. It covers various topics including neural networks, deep learning, and natural language processing. The paper presents novel approaches to knowledge representation and reasoning.";
    
    // First processing (cache miss)
    let start = std::time::Instant::now();
    let processed_doc1 = knowledge_storage.process_document(doc1_id, doc1_content).await?;
    let first_duration = start.elapsed();
    println!("First processing took: {:?}", first_duration);
    
    // Second processing (cache hit)
    let start = std::time::Instant::now();
    let processed_doc1_cached = knowledge_storage.process_document(doc1_id, doc1_content).await?;
    let second_duration = start.elapsed();
    println!("Second processing took: {:?}", second_duration);
    println!("Speedup: {:.1}x faster", first_duration.as_millis() as f64 / second_duration.as_millis() as f64);

    // Example 2: Query Execution with Caching
    println!("\n🔍 Query Execution with Caching");
    println!("-" .repeat(40));
    
    let query = "What are the latest developments in neural networks?";
    
    // First query execution (cache miss)
    let start = std::time::Instant::now();
    let query_result1 = knowledge_storage.execute_query(query).await?;
    let first_query_duration = start.elapsed();
    println!("First query took: {:?}", first_query_duration);
    
    // Second query execution (cache hit)
    let start = std::time::Instant::now();
    let query_result2 = knowledge_storage.execute_query(query).await?;
    let second_query_duration = start.elapsed();
    println!("Second query took: {:?}", second_query_duration);
    println!("Query speedup: {:.1}x faster", first_query_duration.as_millis() as f64 / second_query_duration.as_millis() as f64);

    // Example 3: Embedding Caching
    println!("\n🎯 Embedding Computation with Caching");
    println!("-" .repeat(40));
    
    let text_for_embedding = "Machine learning models require large amounts of training data";
    
    let start = std::time::Instant::now();
    let embedding1 = knowledge_storage.get_embeddings(text_for_embedding).await?;
    let embedding_duration1 = start.elapsed();
    println!("First embedding computation took: {:?}", embedding_duration1);
    
    let start = std::time::Instant::now();
    let embedding2 = knowledge_storage.get_embeddings(text_for_embedding).await?;
    let embedding_duration2 = start.elapsed();
    println!("Second embedding computation took: {:?}", embedding_duration2);
    println!("Embedding dimension: {}", embedding1.len());

    // Example 4: Cache Warming
    println!("\n🔥 Cache Warming with Popular Documents");
    println!("-" .repeat(40));
    
    let popular_docs = vec![
        "popular_doc_1".to_string(),
        "popular_doc_2".to_string(),
        "popular_doc_3".to_string(),
    ];
    
    knowledge_storage.warm_cache_with_popular_documents(popular_docs).await;

    // Example 5: Cache Statistics and Performance
    println!("\n📊 Cache Performance Statistics");
    println!("-" .repeat(40));
    
    let stats = knowledge_storage.get_cache_stats().await;
    println!("Cache Statistics:");
    println!("  Total Requests: {}", stats.total_requests);
    println!("  Hit Rate: {:.2}%", stats.hit_rate() * 100.0);
    println!("  L1 Cache: {} entries, {:.2} MB", stats.l1_entry_count, stats.l1_size_bytes as f64 / 1024.0 / 1024.0);
    println!("  L2 Cache: {} entries, {:.2} MB", stats.l2_entry_count, stats.l2_size_bytes as f64 / 1024.0 / 1024.0);
    println!("  Compression Ratio: {:.2}", stats.compression_ratio());
    
    if stats.stampede_preventions > 0 {
        println!("  Cache Stampede Preventions: {}", stats.stampede_preventions);
    }

    // Example 6: Cache Invalidation
    println!("\n🗑️  Cache Invalidation");
    println!("-" .repeat(40));
    
    knowledge_storage.invalidate_document_cache(doc1_id).await?;
    
    // Verify invalidation by processing the document again
    let start = std::time::Instant::now();
    let _reprocessed_doc = knowledge_storage.process_document(doc1_id, doc1_content).await?;
    let reprocess_duration = start.elapsed();
    println!("Reprocessing after invalidation took: {:?}", reprocess_duration);

    // Final statistics
    println!("\n📈 Final Cache Statistics");
    println!("-" .repeat(40));
    
    let final_stats = knowledge_storage.get_cache_stats().await;
    println!("Final Statistics:");
    println!("  Total Requests: {}", final_stats.total_requests);
    println!("  Overall Hit Rate: {:.2}%", final_stats.hit_rate() * 100.0);
    println!("  Cache Effectiveness: {}", if final_stats.hit_rate() > 0.5 { "Excellent ✓" } else { "Needs Optimization ⚠" });

    println!("\n✅ Cached Knowledge Storage Demo Completed Successfully!");
    println!("The multi-level caching system significantly improves performance");
    println!("for document processing, query execution, and embedding computation.");

    Ok(())
}