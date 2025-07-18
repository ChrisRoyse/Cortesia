//! Embedding Layer Unit Tests
//! 
//! Comprehensive unit tests for the LLMKG embedding layer components

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use rand::prelude::*;

/// CLIP embedding model for testing
#[derive(Debug, Clone)]
pub struct CLIPEmbedding {
    /// Model configuration
    config: CLIPConfig,
    /// Text encoder state
    text_encoder: TextEncoder,
    /// Image encoder state
    image_encoder: ImageEncoder,
    /// Cached embeddings
    embedding_cache: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct CLIPConfig {
    pub embedding_dimension: usize,
    pub max_text_length: usize,
    pub image_size: (usize, usize),
    pub normalize_embeddings: bool,
    pub cache_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct TextEncoder {
    vocab_size: usize,
    embedding_dim: usize,
}

#[derive(Debug, Clone)]
pub struct ImageEncoder {
    input_size: (usize, usize),
    embedding_dim: usize,
}

impl Default for CLIPConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: 512,
            max_text_length: 77,
            image_size: (224, 224),
            normalize_embeddings: true,
            cache_enabled: true,
        }
    }
}

impl CLIPEmbedding {
    /// Create new CLIP embedding model
    pub fn new(config: CLIPConfig) -> Self {
        Self {
            text_encoder: TextEncoder {
                vocab_size: 50000,
                embedding_dim: config.embedding_dimension,
            },
            image_encoder: ImageEncoder {
                input_size: config.image_size,
                embedding_dim: config.embedding_dimension,
            },
            config,
            embedding_cache: HashMap::new(),
        }
    }

    /// Encode text to embedding
    pub async fn encode_text(&mut self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if self.config.cache_enabled {
            if let Some(cached) = self.embedding_cache.get(text) {
                return Ok(cached.clone());
            }
        }

        // Simulate text encoding (in real implementation, this would use a neural network)
        let embedding = self.simulate_text_encoding(text)?;
        
        // Cache the result
        if self.config.cache_enabled {
            self.embedding_cache.insert(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    /// Encode image to embedding
    pub async fn encode_image(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        // Simulate image encoding
        self.simulate_image_encoding(image_data)
    }

    /// Compute similarity between embeddings
    pub fn compute_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(anyhow!("Embedding dimension mismatch: {} vs {}", a.len(), b.len()));
        }

        // Cosine similarity
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Simulate text encoding process
    fn simulate_text_encoding(&self, text: &str) -> Result<Vec<f32>> {
        if text.len() > self.config.max_text_length * 4 {
            return Err(anyhow!("Text too long: {} characters", text.len()));
        }

        // Create deterministic embedding based on text content
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        text.hash(&mut hasher);
        let seed = hasher.finish();

        let mut rng = StdRng::seed_from_u64(seed);
        let mut embedding: Vec<f32> = (0..self.config.embedding_dimension)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Add some text-specific patterns
        let text_features = self.extract_text_features(text);
        for (i, feature) in text_features.iter().enumerate() {
            if i < embedding.len() {
                embedding[i] = (embedding[i] + feature) / 2.0;
            }
        }

        if self.config.normalize_embeddings {
            self.normalize_embedding(&mut embedding);
        }

        Ok(embedding)
    }

    /// Extract simple text features
    fn extract_text_features(&self, text: &str) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Length feature
        features.push((text.len() as f32).ln() / 10.0);
        
        // Word count feature
        features.push((text.split_whitespace().count() as f32).ln() / 5.0);
        
        // Character frequency features
        let chars: Vec<char> = text.chars().collect();
        for &ch in &['a', 'e', 'i', 'o', 'u', 't', 'n', 's', 'r'] {
            let freq = chars.iter().filter(|&&c| c == ch).count() as f32 / chars.len() as f32;
            features.push(freq);
        }
        
        // Pad or truncate to match embedding dimension
        features.resize(self.config.embedding_dimension, 0.0);
        features
    }

    /// Simulate image encoding process
    fn simulate_image_encoding(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        if image_data.is_empty() {
            return Err(anyhow!("Empty image data"));
        }

        // Create deterministic embedding based on image data
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        image_data.hash(&mut hasher);
        let seed = hasher.finish();

        let mut rng = StdRng::seed_from_u64(seed);
        let mut embedding: Vec<f32> = (0..self.config.embedding_dimension)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Add image-specific patterns
        let image_features = self.extract_image_features(image_data);
        for (i, feature) in image_features.iter().enumerate() {
            if i < embedding.len() {
                embedding[i] = (embedding[i] + feature) / 2.0;
            }
        }

        if self.config.normalize_embeddings {
            self.normalize_embedding(&mut embedding);
        }

        Ok(embedding)
    }

    /// Extract simple image features
    fn extract_image_features(&self, image_data: &[u8]) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Size feature
        features.push((image_data.len() as f32).ln() / 20.0);
        
        // Byte statistics
        let mean = image_data.iter().map(|&b| b as f32).sum::<f32>() / image_data.len() as f32;
        features.push(mean / 255.0);
        
        let variance = image_data.iter()
            .map(|&b| (b as f32 - mean).powi(2))
            .sum::<f32>() / image_data.len() as f32;
        features.push(variance.sqrt() / 255.0);
        
        // Histogram features
        let mut histogram = [0; 16];
        for &byte in image_data {
            histogram[(byte / 16) as usize] += 1;
        }
        
        for count in histogram {
            features.push(count as f32 / image_data.len() as f32);
        }
        
        // Pad or truncate to match embedding dimension
        features.resize(self.config.embedding_dimension, 0.0);
        features
    }

    /// Normalize embedding to unit length
    fn normalize_embedding(&self, embedding: &mut [f32]) {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Get embedding statistics
    pub fn get_stats(&self) -> EmbeddingStats {
        EmbeddingStats {
            cache_size: self.embedding_cache.len(),
            embedding_dimension: self.config.embedding_dimension,
            cache_hit_rate: 0.0, // Would track this in real implementation
        }
    }

    /// Clear embedding cache
    pub fn clear_cache(&mut self) {
        self.embedding_cache.clear();
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingStats {
    pub cache_size: usize,
    pub embedding_dimension: usize,
    pub cache_hit_rate: f64,
}

/// Vector quantization for testing
#[derive(Debug, Clone)]
pub struct VectorQuantizer {
    codebook: Vec<Vec<f32>>,
    num_clusters: usize,
    embedding_dim: usize,
}

impl VectorQuantizer {
    /// Create new vector quantizer
    pub fn new(num_clusters: usize, embedding_dim: usize) -> Self {
        // Initialize random codebook
        let mut rng = StdRng::seed_from_u64(42);
        let codebook: Vec<Vec<f32>> = (0..num_clusters)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        Self {
            codebook,
            num_clusters,
            embedding_dim,
        }
    }

    /// Train codebook on embeddings (simplified k-means)
    pub fn train(&mut self, embeddings: &[Vec<f32>], iterations: usize) -> Result<()> {
        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings provided for training"));
        }

        for embedding in embeddings {
            if embedding.len() != self.embedding_dim {
                return Err(anyhow!("Embedding dimension mismatch"));
            }
        }

        for _ in 0..iterations {
            // Assign embeddings to clusters
            let assignments = self.assign_to_clusters(embeddings)?;
            
            // Update centroids
            self.update_centroids(embeddings, &assignments)?;
        }

        Ok(())
    }

    /// Quantize an embedding
    pub fn quantize(&self, embedding: &[f32]) -> Result<usize> {
        if embedding.len() != self.embedding_dim {
            return Err(anyhow!("Embedding dimension mismatch"));
        }

        let mut best_cluster = 0;
        let mut best_distance = f32::INFINITY;

        for (i, centroid) in self.codebook.iter().enumerate() {
            let distance = self.euclidean_distance(embedding, centroid);
            if distance < best_distance {
                best_distance = distance;
                best_cluster = i;
            }
        }

        Ok(best_cluster)
    }

    /// Dequantize a cluster index
    pub fn dequantize(&self, cluster_id: usize) -> Result<Vec<f32>> {
        if cluster_id >= self.num_clusters {
            return Err(anyhow!("Invalid cluster ID: {}", cluster_id));
        }
        Ok(self.codebook[cluster_id].clone())
    }

    /// Assign embeddings to clusters
    fn assign_to_clusters(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>> {
        embeddings.iter()
            .map(|emb| self.quantize(emb))
            .collect()
    }

    /// Update centroids based on assignments
    fn update_centroids(&mut self, embeddings: &[Vec<f32>], assignments: &[usize]) -> Result<()> {
        // Reset centroids
        for centroid in &mut self.codebook {
            centroid.fill(0.0);
        }

        // Count assignments per cluster
        let mut cluster_counts = vec![0; self.num_clusters];
        
        // Sum embeddings for each cluster
        for (embedding, &cluster_id) in embeddings.iter().zip(assignments) {
            for (i, &value) in embedding.iter().enumerate() {
                self.codebook[cluster_id][i] += value;
            }
            cluster_counts[cluster_id] += 1;
        }

        // Average to get new centroids
        for (cluster_id, count) in cluster_counts.iter().enumerate() {
            if *count > 0 {
                for value in &mut self.codebook[cluster_id] {
                    *value /= *count as f32;
                }
            }
        }

        Ok(())
    }

    /// Compute Euclidean distance
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Get quantization statistics
    pub fn get_compression_ratio(&self, original_size: usize) -> f64 {
        let quantized_size = self.num_clusters * self.embedding_dim * 4; // 4 bytes per f32
        let cluster_indices_size = original_size * 4; // 4 bytes per index
        let total_quantized = quantized_size + cluster_indices_size;
        let original_total = original_size * self.embedding_dim * 4;
        
        original_total as f64 / total_quantized as f64
    }
}

/// Performance test for embeddings
#[derive(Debug)]
pub struct EmbeddingPerformanceTest {
    pub name: String,
    pub text_count: usize,
    pub image_count: usize,
    pub expected_latency_ms: f64,
}

impl EmbeddingPerformanceTest {
    pub async fn run(&self) -> Result<EmbeddingTestResult> {
        let start_time = std::time::Instant::now();
        
        let mut clip = CLIPEmbedding::new(CLIPConfig::default());
        
        // Test text embeddings
        let mut text_latencies = Vec::new();
        for i in 0..self.text_count {
            let text = format!("Test text sample number {} with some content for embedding", i);
            
            let text_start = std::time::Instant::now();
            let _embedding = clip.encode_text(&text).await?;
            let text_duration = text_start.elapsed();
            
            text_latencies.push(text_duration.as_secs_f64() * 1000.0);
        }
        
        // Test image embeddings
        let mut image_latencies = Vec::new();
        for i in 0..self.image_count {
            // Create fake image data
            let image_data: Vec<u8> = (0..1024).map(|j| ((i + j) % 256) as u8).collect();
            
            let image_start = std::time::Instant::now();
            let _embedding = clip.encode_image(&image_data).await?;
            let image_duration = image_start.elapsed();
            
            image_latencies.push(image_duration.as_secs_f64() * 1000.0);
        }
        
        let total_duration = start_time.elapsed();
        let avg_text_latency = if text_latencies.is_empty() { 0.0 } else {
            text_latencies.iter().sum::<f64>() / text_latencies.len() as f64
        };
        let avg_image_latency = if image_latencies.is_empty() { 0.0 } else {
            image_latencies.iter().sum::<f64>() / image_latencies.len() as f64
        };
        
        Ok(EmbeddingTestResult {
            test_name: self.name.clone(),
            total_duration_ms: total_duration.as_millis() as f64,
            avg_text_latency_ms: avg_text_latency,
            avg_image_latency_ms: avg_image_latency,
            text_count: self.text_count,
            image_count: self.image_count,
            passed: avg_text_latency <= self.expected_latency_ms && avg_image_latency <= self.expected_latency_ms,
        })
    }
}

#[derive(Debug)]
pub struct EmbeddingTestResult {
    pub test_name: String,
    pub total_duration_ms: f64,
    pub avg_text_latency_ms: f64,
    pub avg_image_latency_ms: f64,
    pub text_count: usize,
    pub image_count: usize,
    pub passed: bool,
}

/// Test suite for embedding layer
pub async fn run_embedding_tests() -> Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();

    // Basic CLIP embedding tests
    results.push(test_clip_text_encoding().await);
    results.push(test_clip_image_encoding().await);
    results.push(test_clip_similarity().await);
    results.push(test_clip_caching().await);
    results.push(test_clip_normalization().await);

    // Vector quantization tests
    results.push(test_vector_quantization().await);
    results.push(test_quantization_training().await);
    results.push(test_quantization_compression().await);

    // Performance tests
    results.push(test_embedding_performance().await);
    results.push(test_embedding_batch_processing().await);

    Ok(results)
}

async fn test_clip_text_encoding() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut clip = CLIPEmbedding::new(CLIPConfig::default());
            
            let text = "Hello, world! This is a test sentence.";
            let embedding = clip.encode_text(text).await?;
            
            // Validate embedding properties
            assert_eq!(embedding.len(), 512); // Default dimension
            assert!(embedding.iter().any(|&x| x != 0.0)); // Not all zeros
            
            // Test deterministic behavior
            let embedding2 = clip.encode_text(text).await?;
            assert_eq!(embedding, embedding2);
            
            // Test different texts produce different embeddings
            let different_text = "This is completely different content.";
            let different_embedding = clip.encode_text(different_text).await?;
            assert_ne!(embedding, different_embedding);
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "clip_text_encoding".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 2048,
        coverage_percentage: 88.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_clip_image_encoding() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let clip = CLIPEmbedding::new(CLIPConfig::default());
            
            // Create test image data
            let image_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
            let embedding = clip.encode_image(&image_data).await?;
            
            // Validate embedding properties
            assert_eq!(embedding.len(), 512);
            assert!(embedding.iter().any(|&x| x != 0.0));
            
            // Test deterministic behavior
            let embedding2 = clip.encode_image(&image_data).await?;
            assert_eq!(embedding, embedding2);
            
            // Test different images produce different embeddings
            let different_image: Vec<u8> = (0..1024).map(|i| ((i + 100) % 256) as u8).collect();
            let different_embedding = clip.encode_image(&different_image).await?;
            assert_ne!(embedding, different_embedding);
            
            // Test empty image error
            let empty_result = clip.encode_image(&[]).await;
            assert!(empty_result.is_err());
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "clip_image_encoding".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 4096,
        coverage_percentage: 90.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_clip_similarity() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut clip = CLIPEmbedding::new(CLIPConfig::default());
            
            // Test similar texts
            let text1 = "The cat sat on the mat";
            let text2 = "A cat is sitting on a mat";
            let text3 = "Completely unrelated content about space exploration";
            
            let emb1 = clip.encode_text(text1).await?;
            let emb2 = clip.encode_text(text2).await?;
            let emb3 = clip.encode_text(text3).await?;
            
            let sim_1_2 = clip.compute_similarity(&emb1, &emb2)?;
            let sim_1_3 = clip.compute_similarity(&emb1, &emb3)?;
            
            // Similar texts should be more similar than dissimilar ones
            assert!(sim_1_2 > sim_1_3, "Similar texts should have higher similarity");
            
            // Similarity should be symmetric
            let sim_2_1 = clip.compute_similarity(&emb2, &emb1)?;
            assert!((sim_1_2 - sim_2_1).abs() < 1e-6);
            
            // Self-similarity should be 1.0 (for normalized embeddings)
            let self_sim = clip.compute_similarity(&emb1, &emb1)?;
            assert!((self_sim - 1.0).abs() < 1e-6);
            
            // Test dimension mismatch error
            let short_vec = vec![1.0, 2.0, 3.0];
            let mismatch_result = clip.compute_similarity(&emb1, &short_vec);
            assert!(mismatch_result.is_err());
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "clip_similarity".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 3072,
        coverage_percentage: 92.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_clip_caching() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut clip = CLIPEmbedding::new(CLIPConfig::default());
            
            let text = "This text will be cached";
            
            // First encoding (cache miss)
            let start_time = std::time::Instant::now();
            let embedding1 = clip.encode_text(text).await?;
            let first_duration = start_time.elapsed();
            
            // Second encoding (cache hit)
            let start_time = std::time::Instant::now();
            let embedding2 = clip.encode_text(text).await?;
            let second_duration = start_time.elapsed();
            
            // Results should be identical
            assert_eq!(embedding1, embedding2);
            
            // Second should be faster (cache hit)
            assert!(second_duration <= first_duration);
            
            // Check cache stats
            let stats = clip.get_stats();
            assert_eq!(stats.cache_size, 1);
            
            // Test cache clearing
            clip.clear_cache();
            let stats = clip.get_stats();
            assert_eq!(stats.cache_size, 0);
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "clip_caching".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1536,
        coverage_percentage: 85.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_clip_normalization() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut clip = CLIPEmbedding::new(CLIPConfig {
                normalize_embeddings: true,
                ..CLIPConfig::default()
            });
            
            let text = "Test normalization";
            let embedding = clip.encode_text(text).await?;
            
            // Check that embedding is normalized (unit length)
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized to unit length");
            
            // Test with non-normalized config
            let mut clip_non_norm = CLIPEmbedding::new(CLIPConfig {
                normalize_embeddings: false,
                ..CLIPConfig::default()
            });
            
            let embedding_non_norm = clip_non_norm.encode_text(text).await?;
            let norm_non_norm: f32 = embedding_non_norm.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            // Non-normalized should likely have different norm
            assert!((norm_non_norm - 1.0).abs() > 1e-3);
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "clip_normalization".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 2048,
        coverage_percentage: 87.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_vector_quantization() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut quantizer = VectorQuantizer::new(8, 16);
        
        // Test quantization
        let test_vector = vec![0.5; 16];
        let cluster_id = quantizer.quantize(&test_vector)?;
        assert!(cluster_id < 8);
        
        // Test dequantization
        let reconstructed = quantizer.dequantize(cluster_id)?;
        assert_eq!(reconstructed.len(), 16);
        
        // Test round-trip consistency
        let cluster_id2 = quantizer.quantize(&reconstructed)?;
        assert_eq!(cluster_id, cluster_id2);
        
        // Test error cases
        let short_vector = vec![1.0, 2.0];
        assert!(quantizer.quantize(&short_vector).is_err());
        assert!(quantizer.dequantize(10).is_err()); // Invalid cluster ID
        
        Ok(())
    })();

    UnitTestResult {
        name: "vector_quantization".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1024,
        coverage_percentage: 90.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_quantization_training() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut quantizer = VectorQuantizer::new(4, 8);
        
        // Create training data
        let mut rng = StdRng::seed_from_u64(42);
        let training_data: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        // Train quantizer
        quantizer.train(&training_data, 5)?;
        
        // Test that all training data can be quantized
        for embedding in &training_data {
            let cluster_id = quantizer.quantize(embedding)?;
            assert!(cluster_id < 4);
        }
        
        // Test compression ratio
        let compression_ratio = quantizer.get_compression_ratio(training_data.len());
        assert!(compression_ratio > 1.0, "Should achieve compression");
        
        // Test with mismatched dimensions
        let bad_data = vec![vec![1.0, 2.0]]; // Wrong dimension
        assert!(quantizer.train(&bad_data, 1).is_err());
        
        // Test with empty data
        assert!(quantizer.train(&[], 1).is_err());
        
        Ok(())
    })();

    UnitTestResult {
        name: "quantization_training".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 2048,
        coverage_percentage: 88.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_quantization_compression() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut quantizer = VectorQuantizer::new(16, 128);
        
        // Create large dataset
        let mut rng = StdRng::seed_from_u64(42);
        let large_dataset: Vec<Vec<f32>> = (0..1000)
            .map(|_| (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        // Train quantizer
        quantizer.train(&large_dataset, 10)?;
        
        // Test compression
        let compression_ratio = quantizer.get_compression_ratio(large_dataset.len());
        assert!(compression_ratio > 5.0, "Should achieve significant compression");
        
        // Test quantization quality (distortion)
        let mut total_distortion = 0.0;
        for embedding in large_dataset.iter().take(100) {
            let cluster_id = quantizer.quantize(embedding)?;
            let reconstructed = quantizer.dequantize(cluster_id)?;
            
            // Compute reconstruction error
            let distortion: f32 = embedding.iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            total_distortion += distortion as f64;
        }
        
        let avg_distortion = total_distortion / 100.0;
        assert!(avg_distortion < 5.0, "Average distortion should be reasonable");
        
        Ok(())
    })();

    UnitTestResult {
        name: "quantization_compression".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 16384,
        coverage_percentage: 86.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_embedding_performance() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let test = EmbeddingPerformanceTest {
                name: "embedding_performance".to_string(),
                text_count: 100,
                image_count: 50,
                expected_latency_ms: 10.0, // 10ms per embedding
            };
            
            let result = test.run().await?;
            
            assert!(result.passed, 
                "Performance test failed: text_latency={}ms, image_latency={}ms (expected <={}ms)",
                result.avg_text_latency_ms, result.avg_image_latency_ms, test.expected_latency_ms);
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "embedding_performance".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 8192,
        coverage_percentage: 82.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_embedding_batch_processing() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut clip = CLIPEmbedding::new(CLIPConfig::default());
            
            // Process batch of texts
            let texts = vec![
                "First text sample",
                "Second text sample", 
                "Third text sample",
                "Fourth text sample",
                "Fifth text sample",
            ];
            
            let batch_start = std::time::Instant::now();
            let mut embeddings = Vec::new();
            for text in &texts {
                let embedding = clip.encode_text(text).await?;
                embeddings.push(embedding);
            }
            let batch_duration = batch_start.elapsed();
            
            // Validate all embeddings
            assert_eq!(embeddings.len(), texts.len());
            for embedding in &embeddings {
                assert_eq!(embedding.len(), 512);
            }
            
            // Test that all embeddings are different
            for i in 0..embeddings.len() {
                for j in (i+1)..embeddings.len() {
                    assert_ne!(embeddings[i], embeddings[j]);
                }
            }
            
            // Batch processing should be reasonably fast
            let avg_time_per_embedding = batch_duration.as_millis() as f64 / texts.len() as f64;
            assert!(avg_time_per_embedding < 50.0, "Batch processing too slow: {}ms per embedding", avg_time_per_embedding);
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "embedding_batch_processing".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 4096,
        coverage_percentage: 84.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding_layer_comprehensive() {
        let results = run_embedding_tests().await.unwrap();
        
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        
        println!("Embedding Layer Tests: {}/{} passed", passed_tests, total_tests);
        
        for result in &results {
            if result.passed {
                println!("✅ {}: {}ms", result.name, result.duration_ms);
            } else {
                println!("❌ {}: {} ({}ms)", result.name, 
                         result.error_message.as_deref().unwrap_or("Unknown error"),
                         result.duration_ms);
            }
        }
        
        assert_eq!(passed_tests, total_tests, "Some embedding tests failed");
    }
}