use crate::core::triple::KnowledgeNode;
use crate::error::Result;
use crate::text::TextCompressor;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Fast text summarization for knowledge chunks
/// Designed for sub-millisecond performance, not neural models
pub struct NeuralSummarizer {
    compressor: TextCompressor,
    cache: Arc<RwLock<HashMap<String, CachedSummary>>>,
}

impl NeuralSummarizer {
    pub fn new() -> Self {
        Self {
            compressor: TextCompressor::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn summarize_chunk(&self, chunk: &str) -> Result<String> {
        // Check cache first
        let cache_key = self.generate_cache_key(chunk);
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                if !cached.is_expired() {
                    return Ok(cached.summary.clone());
                }
            }
        }

        // Use fast text compression instead of neural model
        let summary = self.compressor.compress(chunk);
        
        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, CachedSummary::new(summary.clone()));
        }

        Ok(summary)
    }

    pub async fn summarize_node(&self, node: &KnowledgeNode) -> Result<String> {
        match &node.content {
            crate::core::triple::NodeContent::Chunk { text, .. } => {
                self.summarize_chunk(text).await
            }
            crate::core::triple::NodeContent::Triple(triple) => {
                Ok(triple.to_natural_language())
            }
            crate::core::triple::NodeContent::Entity { name, description, .. } => {
                Ok(format!("{}: {}", name, self.compressor.compress(description)))
            }
            crate::core::triple::NodeContent::Relationship { predicate, description, .. } => {
                Ok(format!("{}: {}", predicate, description))
            }
        }
    }

    pub async fn summarize_path(&self, nodes: &[KnowledgeNode]) -> Result<String> {
        let mut summaries = Vec::new();
        for node in nodes {
            summaries.push(self.summarize_node(node).await?);
        }
        
        // Combine summaries
        let combined = summaries.join(" ");
        Ok(self.compressor.compress(&combined))
    }

    fn generate_cache_key(&self, chunk: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        chunk.hash(&mut hasher);
        format!("chunk_{:x}", hasher.finish())
    }

    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        let total = cache.len();
        let expired = cache.values().filter(|entry| entry.is_expired()).count();
        (total, expired)
    }
}

#[derive(Clone)]
struct CachedSummary {
    summary: String,
    created_at: std::time::Instant,
    ttl: std::time::Duration,
}

impl CachedSummary {
    fn new(summary: String) -> Self {
        Self {
            summary,
            created_at: std::time::Instant::now(),
            ttl: std::time::Duration::from_secs(3600), // 1 hour cache
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

impl Default for NeuralSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fast_summarization() {
        let summarizer = NeuralSummarizer::new();
        
        let text = "This is a test chunk that contains some information about testing. ".repeat(10);
        
        let start = std::time::Instant::now();
        let summary = summarizer.summarize_chunk(&text).await.unwrap();
        let elapsed = start.elapsed();
        
        // Should be very fast
        assert!(elapsed.as_micros() < 1000); // Under 1ms
        assert!(!summary.is_empty());
        assert!(summary.len() < text.len());
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let summarizer = NeuralSummarizer::new();
        
        let text = "Cache test text";
        
        // First call - not cached
        let result1 = summarizer.summarize_chunk(text).await.unwrap();
        
        // Second call - should be cached and faster
        let start = std::time::Instant::now();
        let result2 = summarizer.summarize_chunk(text).await.unwrap();
        let elapsed = start.elapsed();
        
        assert_eq!(result1, result2);
        assert!(elapsed.as_micros() < 100); // Cache hit should be very fast
    }
}