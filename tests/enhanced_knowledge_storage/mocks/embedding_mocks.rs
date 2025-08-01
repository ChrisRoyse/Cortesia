//! Embedding System Mocks
//! 
//! Mock implementations for embedding generation and similarity search
//! components used in semantic knowledge storage and retrieval.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock embedding generator implementation
pub struct MockEmbeddingGenerator {
    embedding_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    call_log: Arc<Mutex<Vec<String>>>,
    embedding_dimension: usize,
    generation_delay_ms: u64,
}

impl MockEmbeddingGenerator {
    pub fn new(dimension: usize) -> Self {
        Self {
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
            embedding_dimension: dimension,
            generation_delay_ms: 0,
        }
    }
    
    pub fn with_delay(dimension: usize, delay_ms: u64) -> Self {
        Self {
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
            embedding_dimension: dimension,
            generation_delay_ms: delay_ms,
        }
    }
    
    pub fn generate_embedding(&self, text: &str) -> Vec<f32> {
        self.call_log.lock().unwrap().push(format!("generate_embedding: {} chars", text.len()));
        
        // Check cache first
        if let Some(cached) = self.embedding_cache.lock().unwrap().get(text) {
            return cached.clone();
        }
        
        // Simulate generation delay
        if self.generation_delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.generation_delay_ms));
        }
        
        // Generate mock embedding based on text hash
        let hash = simple_hash(text);
        let embedding = (0..self.embedding_dimension)
            .map(|i| ((hash + i) % 1000) as f32 / 1000.0)
            .collect::<Vec<f32>>();
        
        // Cache the result
        self.embedding_cache.lock().unwrap().insert(text.to_string(), embedding.clone());
        
        embedding
    }
    
    pub fn batch_generate_embeddings(&self, texts: &[String]) -> Vec<Vec<f32>> {
        self.call_log.lock().unwrap().push(format!("batch_generate_embeddings: {} texts", texts.len()));
        
        texts.iter()
            .map(|text| self.generate_embedding(text))
            .collect()
    }
    
    pub fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
    
    pub fn clear_call_log(&self) {
        self.call_log.lock().unwrap().clear();
    }
    
    pub fn get_dimension(&self) -> usize {
        self.embedding_dimension
    }
}

/// Mock similarity calculator implementation
pub struct MockSimilarityCalculator {
    call_log: Arc<Mutex<Vec<String>>>,
}

impl MockSimilarityCalculator {
    pub fn new() -> Self {
        Self {
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        self.call_log.lock().unwrap().push("cosine_similarity".to_string());
        
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    pub fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.call_log.lock().unwrap().push("euclidean_distance".to_string());
        
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    pub fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
}

/// Mock embedding index for fast similarity search
pub struct MockEmbeddingIndex {
    embeddings: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    call_log: Arc<Mutex<Vec<String>>>,
    dimension: usize,
}

impl MockEmbeddingIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            embeddings: Arc::new(Mutex::new(HashMap::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
            dimension,
        }
    }
    
    pub fn add_embedding(&self, id: String, embedding: Vec<f32>) -> Result<(), String> {
        if embedding.len() != self.dimension {
            return Err(format!("Expected dimension {}, got {}", self.dimension, embedding.len()));
        }
        
        self.call_log.lock().unwrap().push(format!("add_embedding: {}", id));
        self.embeddings.lock().unwrap().insert(id, embedding);
        Ok(())
    }
    
    pub fn search_similar(&self, query_embedding: &[f32], k: usize) -> Vec<SimilarityMatch> {
        self.call_log.lock().unwrap().push(format!("search_similar: k={}", k));
        
        let embeddings = self.embeddings.lock().unwrap();
        let mut results = Vec::new();
        
        for (id, embedding) in embeddings.iter() {
            let similarity = self.calculate_cosine_similarity(query_embedding, embedding);
            results.push(SimilarityMatch {
                id: id.clone(),
                similarity,
            });
        }
        
        // Sort by similarity (descending) and take top k
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        
        results
    }
    
    fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    pub fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
}

/// Similarity match result for testing
#[derive(Debug, Clone)]
pub struct SimilarityMatch {
    pub id: String,
    pub similarity: f32,
}

/// Helper functions for creating mock embedding components
pub fn create_mock_embedding_generator(dimension: usize) -> MockEmbeddingGenerator {
    MockEmbeddingGenerator::new(dimension)
}

pub fn create_mock_embedding_generator_with_delay(dimension: usize, delay_ms: u64) -> MockEmbeddingGenerator {
    MockEmbeddingGenerator::with_delay(dimension, delay_ms)
}

pub fn create_mock_similarity_calculator() -> MockSimilarityCalculator {
    MockSimilarityCalculator::new()
}

pub fn create_mock_embedding_index(dimension: usize) -> MockEmbeddingIndex {
    MockEmbeddingIndex::new(dimension)
}

/// Test helper for setting up embedding mocks with sample data
pub fn setup_embedding_mocks_with_sample_data() -> (MockEmbeddingGenerator, MockEmbeddingIndex) {
    let generator = create_mock_embedding_generator(384);
    let index = create_mock_embedding_index(384);
    
    // Add some sample embeddings
    let sample_texts = vec![
        "artificial intelligence",
        "machine learning",
        "neural networks",
        "deep learning",
    ];
    
    for (i, text) in sample_texts.iter().enumerate() {
        let embedding = generator.generate_embedding(text);
        let _ = index.add_embedding(format!("sample_{}", i), embedding);
    }
    
    (generator, index)
}

/// Simple hash function for generating deterministic mock embeddings
fn simple_hash(text: &str) -> usize {
    text.chars()
        .enumerate()
        .map(|(i, c)| (c as usize) * (i + 1))
        .sum()
}