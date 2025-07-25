//! Native Rust embeddings implementation for semantic similarity

use std::sync::Arc;
use crate::models::rust_tokenizer::{RustTokenizer, TokenizedInput};
use crate::models::rust_bert_models::{Matrix, RustBertModel, EmbeddingLayer, SelfAttention, FeedForward};
use crate::models::{ModelError, Result};

/// Sentence transformer model for generating embeddings
#[derive(Debug, Clone)]
pub struct RustSentenceTransformer {
    pub bert: RustBertModel,
    pub pooling_strategy: PoolingStrategy,
    pub normalize: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum PoolingStrategy {
    /// Use [CLS] token representation
    CLS,
    /// Mean pooling over all tokens
    Mean,
    /// Max pooling over all tokens
    Max,
    /// Mean pooling with attention mask
    MeanWithMask,
}

impl RustSentenceTransformer {
    pub fn new(vocab_size: usize, hidden_size: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            bert: RustBertModel::new(vocab_size, hidden_size, num_layers, num_heads),
            pooling_strategy: PoolingStrategy::Mean,
            normalize: true,
        }
    }
    
    /// Generate embeddings for a batch of texts
    pub fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        
        for text in texts {
            let embedding = self.encode_single(text)?;
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }
    
    /// Generate embedding for a single text
    pub fn encode_single(&self, text: &str) -> Result<Vec<f32>> {
        let tokenized = self.bert.tokenizer.encode(text, true);
        let hidden_states = self.bert.forward(&tokenized.input_ids);
        
        let pooled = match self.pooling_strategy {
            PoolingStrategy::CLS => self.cls_pooling(&hidden_states),
            PoolingStrategy::Mean => self.mean_pooling(&hidden_states),
            PoolingStrategy::Max => self.max_pooling(&hidden_states),
            PoolingStrategy::MeanWithMask => self.mean_pooling_with_mask(&hidden_states, &tokenized.attention_mask),
        };
        
        let final_embedding = if self.normalize {
            self.normalize_vector(&pooled)
        } else {
            pooled
        };
        
        Ok(final_embedding)
    }
    
    fn cls_pooling(&self, hidden_states: &Matrix) -> Vec<f32> {
        if hidden_states.rows > 0 {
            (0..self.bert.hidden_size).map(|i| hidden_states.get(0, i)).collect()
        } else {
            vec![0.0; self.bert.hidden_size]
        }
    }
    
    fn mean_pooling(&self, hidden_states: &Matrix) -> Vec<f32> {
        let mut pooled = vec![0.0; self.bert.hidden_size];
        
        if hidden_states.rows == 0 {
            return pooled;
        }
        
        for col in 0..self.bert.hidden_size {
            let mut sum = 0.0;
            for row in 0..hidden_states.rows {
                sum += hidden_states.get(row, col);
            }
            pooled[col] = sum / hidden_states.rows as f32;
        }
        
        pooled
    }
    
    fn max_pooling(&self, hidden_states: &Matrix) -> Vec<f32> {
        let mut pooled = vec![f32::NEG_INFINITY; self.bert.hidden_size];
        
        for col in 0..self.bert.hidden_size {
            for row in 0..hidden_states.rows {
                let val = hidden_states.get(row, col);
                if val > pooled[col] {
                    pooled[col] = val;
                }
            }
        }
        
        // Handle case where all values were -inf
        for val in &mut pooled {
            if *val == f32::NEG_INFINITY {
                *val = 0.0;
            }
        }
        
        pooled
    }
    
    fn mean_pooling_with_mask(&self, hidden_states: &Matrix, attention_mask: &[f32]) -> Vec<f32> {
        let mut pooled = vec![0.0; self.bert.hidden_size];
        let mut total_weight = 0.0;
        
        for col in 0..self.bert.hidden_size {
            let mut weighted_sum = 0.0;
            for row in 0..hidden_states.rows.min(attention_mask.len()) {
                let weight = attention_mask[row];
                weighted_sum += hidden_states.get(row, col) * weight;
                if col == 0 { // Count total weight only once
                    total_weight += weight;
                }
            }
            pooled[col] = if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                0.0
            };
        }
        
        pooled
    }
    
    fn normalize_vector(&self, vector: &[f32]) -> Vec<f32> {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.iter().map(|x| x / norm).collect()
        } else {
            vector.to_vec()
        }
    }
    
    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
    
    /// Find most similar texts to a query
    pub fn find_similar(&self, query: &str, candidates: &[&str], top_k: usize) -> Result<Vec<(usize, f32)>> {
        let query_embedding = self.encode_single(query)?;
        let candidate_embeddings = self.encode(candidates)?;
        
        let mut similarities = Vec::new();
        for (i, candidate_embedding) in candidate_embeddings.iter().enumerate() {
            let similarity = self.cosine_similarity(&query_embedding, candidate_embedding);
            similarities.push((i, similarity));
        }
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top-k
        similarities.truncate(top_k);
        Ok(similarities)
    }
}

/// MiniLM model for lightweight embeddings
#[derive(Debug, Clone)]
pub struct RustMiniLM {
    pub transformer: RustSentenceTransformer,
}

impl RustMiniLM {
    pub fn new() -> Self {
        let vocab_size = 30522;
        let hidden_size = 384;    // MiniLM uses smaller hidden size
        let num_layers = 6;       // Fewer layers than BERT
        let num_heads = 12;
        
        Self {
            transformer: RustSentenceTransformer::new(vocab_size, hidden_size, num_layers, num_heads),
        }
    }
    
    pub fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.transformer.encode(texts)
    }
    
    pub fn encode_single(&self, text: &str) -> Result<Vec<f32>> {
        self.transformer.encode_single(text)
    }
    
    pub fn similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let embedding1 = self.encode_single(text1)?;
        let embedding2 = self.encode_single(text2)?;
        Ok(self.transformer.cosine_similarity(&embedding1, &embedding2))
    }
    
    pub fn find_similar(&self, query: &str, candidates: &[&str], top_k: usize) -> Result<Vec<(usize, f32)>> {
        self.transformer.find_similar(query, candidates, top_k)
    }
}

/// Embedding cache for performance optimization
#[derive(Debug)]
pub struct EmbeddingCache {
    cache: std::collections::HashMap<String, Vec<f32>>,
    max_size: usize,
}

impl EmbeddingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size,
        }
    }
    
    pub fn get(&self, text: &str) -> Option<&Vec<f32>> {
        self.cache.get(text)
    }
    
    pub fn insert(&mut self, text: String, embedding: Vec<f32>) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: remove first entry
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(text, embedding);
    }
    
    pub fn clear(&mut self) {
        self.cache.clear();
    }
    
    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

/// Cached sentence transformer
#[derive(Debug)]
pub struct CachedSentenceTransformer {
    pub model: RustMiniLM,
    pub cache: EmbeddingCache,
}

impl CachedSentenceTransformer {
    pub fn new(cache_size: usize) -> Self {
        Self {
            model: RustMiniLM::new(),
            cache: EmbeddingCache::new(cache_size),
        }
    }
    
    pub fn encode_single(&mut self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }
        
        let embedding = self.model.encode_single(text)?;
        self.cache.insert(text.to_string(), embedding.clone());
        Ok(embedding)
    }
    
    pub fn encode(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.encode_single(text)?);
        }
        Ok(embeddings)
    }
    
    pub fn similarity(&mut self, text1: &str, text2: &str) -> Result<f32> {
        let embedding1 = self.encode_single(text1)?;
        let embedding2 = self.encode_single(text2)?;
        Ok(self.model.transformer.cosine_similarity(&embedding1, &embedding2))
    }
}

/// Semantic search engine using embeddings
#[derive(Debug)]
pub struct SemanticSearchEngine {
    pub model: RustMiniLM,
    pub document_embeddings: Vec<Vec<f32>>,
    pub documents: Vec<String>,
}

impl SemanticSearchEngine {
    pub fn new() -> Self {
        Self {
            model: RustMiniLM::new(),
            document_embeddings: Vec::new(),
            documents: Vec::new(),
        }
    }
    
    pub fn add_documents(&mut self, documents: &[&str]) -> Result<()> {
        for doc in documents {
            let embedding = self.model.encode_single(doc)?;
            self.document_embeddings.push(embedding);
            self.documents.push(doc.to_string());
        }
        Ok(())
    }
    
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<(String, f32)>> {
        if self.documents.is_empty() {
            return Ok(Vec::new());
        }
        
        let query_embedding = self.model.encode_single(query)?;
        let mut similarities = Vec::new();
        
        for (i, doc_embedding) in self.document_embeddings.iter().enumerate() {
            let similarity = self.model.transformer.cosine_similarity(&query_embedding, doc_embedding);
            similarities.push((i, similarity));
        }
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top-k with document text
        let results = similarities.into_iter()
            .take(top_k)
            .map(|(idx, sim)| (self.documents[idx].clone(), sim))
            .collect();
        
        Ok(results)
    }
    
    pub fn clear(&mut self) {
        self.document_embeddings.clear();
        self.documents.clear();
    }
    
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sentence_transformer() {
        let model = RustSentenceTransformer::new(1000, 128, 2, 4);
        
        let texts = vec!["Hello world", "Goodbye world"];
        let embeddings = model.encode(&texts).unwrap();
        
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 128);
        assert_eq!(embeddings[1].len(), 128);
    }
    
    #[test]
    fn test_pooling_strategies() {
        let model = RustSentenceTransformer::new(1000, 64, 1, 2);
        
        let text = "This is a test sentence";
        let tokenized = model.bert.tokenizer.encode(text, true);
        let hidden_states = model.bert.forward(&tokenized.input_ids);
        
        let cls_pooled = model.cls_pooling(&hidden_states);
        let mean_pooled = model.mean_pooling(&hidden_states);
        let max_pooled = model.max_pooling(&hidden_states);
        
        assert_eq!(cls_pooled.len(), 64);
        assert_eq!(mean_pooled.len(), 64);
        assert_eq!(max_pooled.len(), 64);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let model = RustSentenceTransformer::new(100, 64, 1, 2);
        
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let vec3 = vec![-1.0, -2.0, -3.0];
        
        let sim1 = model.cosine_similarity(&vec1, &vec2);
        let sim2 = model.cosine_similarity(&vec1, &vec3);
        
        assert!((sim1 - 1.0).abs() < 1e-6); // Should be 1.0 (identical)
        assert!((sim2 + 1.0).abs() < 1e-6); // Should be -1.0 (opposite)
    }
    
    #[test]
    fn test_minilm() {
        let model = RustMiniLM::new();
        
        let text1 = "The cat sits on the mat";
        let text2 = "A cat is sitting on a mat";
        let text3 = "Dogs are running in the park";
        
        let embedding1 = model.encode_single(text1).unwrap();
        let embedding2 = model.encode_single(text2).unwrap();
        let embedding3 = model.encode_single(text3).unwrap();
        
        assert_eq!(embedding1.len(), 384); // MiniLM embedding size
        
        let sim1_2 = model.transformer.cosine_similarity(&embedding1, &embedding2);
        let sim1_3 = model.transformer.cosine_similarity(&embedding1, &embedding3);
        
        // Similar sentences should have higher similarity
        println!("Similarity 1-2: {}", sim1_2);
        println!("Similarity 1-3: {}", sim1_3);
        
        // Basic sanity check - similar sentences should have positive similarity
        assert!(sim1_2 > 0.0);
    }
    
    #[test]
    fn test_embedding_cache() {
        let mut cache = EmbeddingCache::new(2);
        
        cache.insert("hello".to_string(), vec![1.0, 2.0, 3.0]);
        cache.insert("world".to_string(), vec![4.0, 5.0, 6.0]);
        
        assert_eq!(cache.size(), 2);
        assert!(cache.get("hello").is_some());
        assert!(cache.get("world").is_some());
        
        // Adding third item should evict first
        cache.insert("test".to_string(), vec![7.0, 8.0, 9.0]);
        assert_eq!(cache.size(), 2);
    }
    
    #[test]
    fn test_cached_transformer() {
        let mut model = CachedSentenceTransformer::new(10);
        
        let text = "Test sentence for caching";
        
        // First call should compute embedding
        let embedding1 = model.encode_single(text).unwrap();
        assert_eq!(model.cache.size(), 1);
        
        // Second call should use cache
        let embedding2 = model.encode_single(text).unwrap();
        assert_eq!(embedding1, embedding2);
        assert_eq!(model.cache.size(), 1);
    }
    
    #[test]
    fn test_semantic_search() {
        let mut search_engine = SemanticSearchEngine::new();
        
        let documents = vec![
            "Albert Einstein was a physicist",
            "Marie Curie was a chemist",
            "The weather is nice today",
            "Physics is the study of matter and energy",
        ];
        
        search_engine.add_documents(&documents).unwrap();
        assert_eq!(search_engine.document_count(), 4);
        
        let results = search_engine.search("scientist", 2).unwrap();
        assert_eq!(results.len(), 2);
        
        println!("Search results for 'scientist':");
        for (doc, score) in &results {
            println!("  {}: {:.3}", doc, score);
        }
        
        // Basic sanity check
        assert!(results[0].1 > 0.0); // Should have positive similarity
    }
    
    #[test]
    fn test_find_similar() {
        let model = RustMiniLM::new();
        
        let query = "famous scientist";
        let candidates = vec![
            "Albert Einstein was a physicist",
            "I like pizza",
            "Marie Curie discovered radium",
            "The weather is sunny",
        ];
        
        let results = model.find_similar(query, &candidates, 2).unwrap();
        assert_eq!(results.len(), 2);
        
        println!("Most similar to '{}': ", query);
        for (idx, score) in &results {
            println!("  {}: {:.3}", candidates[*idx], score);
        }
        
        // Should find scientist-related texts more similar
        assert!(results[0].1 > 0.0);
    }
}