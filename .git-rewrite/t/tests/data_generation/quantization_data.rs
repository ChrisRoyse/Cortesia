//! Quantization Test Data Generation
//! 
//! Provides generation of test data specifically for product quantization accuracy testing.

use crate::infrastructure::deterministic_rng::DeterministicRng;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Test set for quantization validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationTestSet {
    pub original_vectors: Vec<Vec<f32>>,
    pub expected_approximations: Vec<Vec<f32>>,
    pub expected_distances: Vec<DistanceComparison>,
    pub compression_ratio: f64,
    pub quantization_parameters: QuantizationParams,
}

/// Parameters for quantization testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    pub vector_dimension: usize,
    pub subvector_count: usize,
    pub codebook_size: usize,
    pub subvector_dimension: usize,
}

/// Distance comparison for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceComparison {
    pub vector1_idx: u64,
    pub vector2_idx: u64,
    pub original_distance: f32,
    pub approximated_distance: f32,
    pub compression_error: f32,
    pub relative_error: f32,
}

/// SIMD test set for vectorized operations
#[derive(Debug, Clone)]
pub struct SimdTestSet {
    pub aligned_vectors: Vec<Vec<f32>>,
    pub edge_case_vectors: Vec<Vec<f32>>,
    pub expected_dot_products: Vec<f32>,
    pub expected_norms: Vec<f32>,
    pub alignment_metadata: AlignmentMetadata,
}

/// Metadata about vector alignment for SIMD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentMetadata {
    pub simd_width: usize,
    pub alignment_bytes: usize,
    pub padding_info: Vec<PaddingInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingInfo {
    pub vector_idx: usize,
    pub original_length: usize,
    pub padded_length: usize,
    pub padding_value: f32,
}

/// Specialized data for compression analysis
#[derive(Debug, Clone)]
pub struct CompressionTestData {
    pub baseline_vectors: Vec<Vec<f32>>,
    pub compression_levels: Vec<CompressionLevel>,
    pub expected_metrics: CompressionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionLevel {
    pub level_id: u32,
    pub bits_per_component: u8,
    pub expected_quality: f32,
    pub expected_compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetrics {
    pub mse_by_level: HashMap<u32, f64>,
    pub psnr_by_level: HashMap<u32, f64>,
    pub compression_ratios: HashMap<u32, f64>,
    pub reconstruction_times: HashMap<u32, f64>, // In microseconds
}

/// Quantization data generator
pub struct QuantizationDataGenerator {
    rng: DeterministicRng,
}

impl QuantizationDataGenerator {
    /// Create a new quantization data generator
    pub fn new(seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        rng.set_label("quantization_data_generator".to_string());
        
        Self { rng }
    }

    /// Generate test data specifically for product quantization accuracy
    pub fn generate_product_quantization_test_data(
        &mut self,
        vector_count: u64,
        dimension: usize,
        codebook_size: usize,
    ) -> Result<QuantizationTestSet> {
        if vector_count == 0 {
            return Err(anyhow!("Vector count must be positive"));
        }
        if dimension == 0 {
            return Err(anyhow!("Dimension must be positive"));
        }
        if codebook_size == 0 || codebook_size > 256 {
            return Err(anyhow!("Codebook size must be between 1 and 256"));
        }

        // Calculate subvector parameters
        let subvector_count = 8; // Standard for 96-dim: 8 subvectors of 12 dims each
        let subvector_dimension = dimension / subvector_count;
        
        if dimension % subvector_count != 0 {
            return Err(anyhow!("Dimension must be divisible by subvector count"));
        }

        let params = QuantizationParams {
            vector_dimension: dimension,
            subvector_count,
            codebook_size,
            subvector_dimension,
        };

        // Generate structured vectors for predictable quantization behavior
        let mut original_vectors = Vec::new();
        let mut expected_approximations = Vec::new();
        let mut expected_distances = Vec::new();

        for i in 0..vector_count {
            let vector = self.generate_structured_vector(dimension, i)?;
            let approximation = self.compute_expected_pq_approximation(&vector, &params)?;
            
            original_vectors.push(vector.clone());
            expected_approximations.push(approximation);
        }

        // Precompute expected distances for similarity search validation
        for i in 0..vector_count {
            for j in (i + 1)..vector_count {
                let original_dist = self.euclidean_distance(&original_vectors[i as usize], &original_vectors[j as usize]);
                let approx_dist = self.euclidean_distance(&expected_approximations[i as usize], &expected_approximations[j as usize]);
                let compression_error = (approx_dist - original_dist).abs();
                let relative_error = if original_dist > 1e-10 {
                    compression_error / original_dist
                } else {
                    0.0
                };

                expected_distances.push(DistanceComparison {
                    vector1_idx: i,
                    vector2_idx: j,
                    original_distance: original_dist,
                    approximated_distance: approx_dist,
                    compression_error,
                    relative_error,
                });
            }
        }

        let compression_ratio = self.calculate_compression_ratio(dimension, codebook_size, subvector_count);

        Ok(QuantizationTestSet {
            original_vectors,
            expected_approximations,
            expected_distances,
            compression_ratio,
            quantization_parameters: params,
        })
    }

    /// Generate vectors specifically for testing SIMD operations
    pub fn generate_simd_test_vectors(&mut self, count: u64) -> Result<SimdTestSet> {
        if count == 0 {
            return Err(anyhow!("Vector count must be positive"));
        }

        const SIMD_WIDTH: usize = 8; // AVX2 width for f32
        const ALIGNMENT_BYTES: usize = 32; // 32-byte alignment for AVX2

        let mut aligned_vectors = Vec::new();
        let mut edge_case_vectors = Vec::new();
        let mut expected_dot_products = Vec::new();
        let mut expected_norms = Vec::new();
        let mut padding_info = Vec::new();

        // Generate regular aligned vectors
        for i in 0..count {
            let dimension = if i % 3 == 0 {
                128 // Standard size
            } else if i % 3 == 1 {
                96  // Common embedding size
            } else {
                64  // Smaller size
            };

            // Ensure dimension is aligned to SIMD width
            let aligned_dimension = ((dimension + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
            let mut vector = self.generate_random_normal_vector(dimension)?;
            
            // Pad to aligned dimension
            let original_length = vector.len();
            vector.resize(aligned_dimension, 0.0);
            
            if aligned_dimension > original_length {
                padding_info.push(PaddingInfo {
                    vector_idx: i as usize,
                    original_length,
                    padded_length: aligned_dimension,
                    padding_value: 0.0,
                });
            }

            // Calculate expected results
            let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
            expected_norms.push(norm);

            aligned_vectors.push(vector);
        }

        // Generate edge case vectors for robustness testing
        edge_case_vectors.extend(vec![
            vec![0.0; SIMD_WIDTH], // All zeros
            vec![1.0; SIMD_WIDTH], // All ones
            vec![f32::INFINITY; SIMD_WIDTH], // All infinities
            vec![f32::NEG_INFINITY; SIMD_WIDTH], // All negative infinities
            vec![f32::NAN; SIMD_WIDTH], // All NaNs
            {
                let mut v = vec![0.0; SIMD_WIDTH];
                v[0] = f32::MAX;
                v[1] = f32::MIN;
                v
            }, // Min/max values
            {
                let mut v = vec![0.0; SIMD_WIDTH];
                for i in 0..SIMD_WIDTH {
                    v[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
                }
                v
            }, // Alternating pattern
            {
                let mut v = vec![0.0; SIMD_WIDTH];
                v[0] = 1e-10; // Very small positive
                v[1] = -1e-10; // Very small negative
                v
            }, // Near-zero values
        ]);

        // Calculate expected dot products for pairs
        for i in 0..aligned_vectors.len().min(10) {
            for j in (i + 1)..aligned_vectors.len().min(10) {
                let dot_product = aligned_vectors[i].iter()
                    .zip(aligned_vectors[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>();
                expected_dot_products.push(dot_product);
            }
        }

        let alignment_metadata = AlignmentMetadata {
            simd_width: SIMD_WIDTH,
            alignment_bytes: ALIGNMENT_BYTES,
            padding_info,
        };

        Ok(SimdTestSet {
            aligned_vectors,
            edge_case_vectors,
            expected_dot_products,
            expected_norms,
            alignment_metadata,
        })
    }

    /// Generate data for compression ratio analysis
    pub fn generate_compression_test_data(&mut self, vector_count: u64, dimension: usize) -> Result<CompressionTestData> {
        if vector_count == 0 || dimension == 0 {
            return Err(anyhow!("Vector count and dimension must be positive"));
        }

        // Generate baseline vectors with varied characteristics
        let mut baseline_vectors = Vec::new();
        for i in 0..vector_count {
            let vector = match i % 4 {
                0 => self.generate_random_normal_vector(dimension)?, // Normal distribution
                1 => self.generate_uniform_vector(dimension)?,       // Uniform distribution
                2 => self.generate_sparse_vector(dimension, 0.1)?,   // Sparse vector (10% non-zero)
                _ => self.generate_clustered_vector(dimension, i)?,   // Clustered around centroid
            };
            baseline_vectors.push(vector);
        }

        // Define compression levels to test
        let compression_levels = vec![
            CompressionLevel {
                level_id: 0,
                bits_per_component: 8,
                expected_quality: 0.95,
                expected_compression_ratio: 4.0, // 32-bit to 8-bit
            },
            CompressionLevel {
                level_id: 1,
                bits_per_component: 4,
                expected_quality: 0.85,
                expected_compression_ratio: 8.0, // 32-bit to 4-bit
            },
            CompressionLevel {
                level_id: 2,
                bits_per_component: 2,
                expected_quality: 0.7,
                expected_compression_ratio: 16.0, // 32-bit to 2-bit
            },
            CompressionLevel {
                level_id: 3,
                bits_per_component: 1,
                expected_quality: 0.5,
                expected_compression_ratio: 32.0, // 32-bit to 1-bit
            },
        ];

        // Calculate expected metrics
        let expected_metrics = self.calculate_compression_metrics(&baseline_vectors, &compression_levels)?;

        Ok(CompressionTestData {
            baseline_vectors,
            compression_levels,
            expected_metrics,
        })
    }

    /// Generate structured vector with predictable quantization behavior
    fn generate_structured_vector(&mut self, dimension: usize, vector_index: u64) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; dimension];
        
        // Create patterns that are predictable for quantization
        let pattern_type = vector_index % 4;
        
        match pattern_type {
            0 => {
                // Smooth gradient pattern
                for i in 0..dimension {
                    vector[i] = (i as f32 / dimension as f32) * 2.0 - 1.0; // -1 to 1
                }
            },
            1 => {
                // Sinusoidal pattern
                for i in 0..dimension {
                    let freq = 2.0 * std::f32::consts::PI * (vector_index as f32 + 1.0) / 10.0;
                    vector[i] = (freq * i as f32 / dimension as f32).sin();
                }
            },
            2 => {
                // Step function pattern
                let step_size = dimension / 8;
                for i in 0..dimension {
                    let step = i / step_size;
                    vector[i] = if step % 2 == 0 { 1.0 } else { -1.0 };
                }
            },
            _ => {
                // Random normal with controlled variance
                for i in 0..dimension {
                    vector[i] = self.rng.normal(0.0, 0.5) as f32;
                }
            }
        }

        // Add small amount of noise for realism
        for i in 0..dimension {
            vector[i] += self.rng.normal(0.0, 0.01) as f32;
        }

        // Normalize to unit vector
        let norm = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for i in 0..dimension {
                vector[i] /= norm;
            }
        }

        Ok(vector)
    }

    /// Compute expected product quantization approximation
    fn compute_expected_pq_approximation(&mut self, vector: &[f32], params: &QuantizationParams) -> Result<Vec<f32>> {
        let mut approximation = vec![0.0; vector.len()];
        
        // Simulate PQ by quantizing each subvector
        for sub_idx in 0..params.subvector_count {
            let start_idx = sub_idx * params.subvector_dimension;
            let end_idx = ((sub_idx + 1) * params.subvector_dimension).min(vector.len());
            
            if start_idx >= vector.len() {
                break;
            }

            let subvector = &vector[start_idx..end_idx];
            
            // Simulate quantization by clustering to nearest centroid
            let quantized_subvector = self.quantize_subvector(subvector, params.codebook_size)?;
            
            // Copy quantized values to approximation
            for (i, &value) in quantized_subvector.iter().enumerate() {
                if start_idx + i < approximation.len() {
                    approximation[start_idx + i] = value;
                }
            }
        }

        Ok(approximation)
    }

    /// Simulate subvector quantization
    fn quantize_subvector(&mut self, subvector: &[f32], codebook_size: usize) -> Result<Vec<f32>> {
        if subvector.is_empty() {
            return Ok(Vec::new());
        }

        // Generate deterministic codebook for this subvector
        let mut codebook = Vec::new();
        for i in 0..codebook_size {
            let mut centroid = vec![0.0; subvector.len()];
            let angle = 2.0 * std::f64::consts::PI * i as f64 / codebook_size as f64;
            
            for j in 0..subvector.len() {
                centroid[j] = (angle + j as f64).cos() as f32;
            }
            
            // Normalize centroid
            let norm = centroid.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for j in 0..centroid.len() {
                    centroid[j] /= norm;
                }
            }
            
            codebook.push(centroid);
        }

        // Find nearest centroid
        let mut best_centroid = &codebook[0];
        let mut best_distance = self.euclidean_distance(subvector, &codebook[0]);
        
        for centroid in &codebook[1..] {
            let distance = self.euclidean_distance(subvector, centroid);
            if distance < best_distance {
                best_distance = distance;
                best_centroid = centroid;
            }
        }

        Ok(best_centroid.clone())
    }

    /// Generate random normal vector
    fn generate_random_normal_vector(&mut self, dimension: usize) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; dimension];
        for i in 0..dimension {
            vector[i] = self.rng.normal(0.0, 1.0) as f32;
        }
        Ok(vector)
    }

    /// Generate uniform random vector
    fn generate_uniform_vector(&mut self, dimension: usize) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; dimension];
        for i in 0..dimension {
            vector[i] = self.rng.range_f64(-1.0, 1.0) as f32;
        }
        Ok(vector)
    }

    /// Generate sparse vector with given sparsity
    fn generate_sparse_vector(&mut self, dimension: usize, sparsity: f32) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; dimension];
        let non_zero_count = (dimension as f32 * sparsity) as usize;
        
        for _ in 0..non_zero_count {
            let idx = self.rng.next_usize(dimension);
            vector[idx] = self.rng.normal(0.0, 1.0) as f32;
        }
        
        Ok(vector)
    }

    /// Generate clustered vector around a centroid
    fn generate_clustered_vector(&mut self, dimension: usize, cluster_id: u64) -> Result<Vec<f32>> {
        // Generate deterministic centroid
        let mut centroid = vec![0.0; dimension];
        let cluster_angle = 2.0 * std::f64::consts::PI * (cluster_id % 8) as f64 / 8.0;
        
        for i in 0..dimension {
            centroid[i] = (cluster_angle + i as f64 * 0.1).cos() as f32;
        }

        // Add noise around centroid
        let mut vector = vec![0.0; dimension];
        for i in 0..dimension {
            vector[i] = centroid[i] + self.rng.normal(0.0, 0.1) as f32;
        }

        Ok(vector)
    }

    /// Calculate compression ratio
    fn calculate_compression_ratio(&self, dimension: usize, codebook_size: usize, subvector_count: usize) -> f64 {
        let original_bits = dimension * 32; // 32-bit floats
        let compressed_bits = subvector_count * (codebook_size as f64).log2().ceil() as usize;
        
        if compressed_bits > 0 {
            original_bits as f64 / compressed_bits as f64
        } else {
            1.0
        }
    }

    /// Calculate expected compression metrics
    fn calculate_compression_metrics(&mut self, vectors: &[Vec<f32>], levels: &[CompressionLevel]) -> Result<CompressionMetrics> {
        let mut mse_by_level = HashMap::new();
        let mut psnr_by_level = HashMap::new();
        let mut compression_ratios = HashMap::new();
        let mut reconstruction_times = HashMap::new();

        for level in levels {
            // Simulate compression and reconstruction
            let mut total_mse = 0.0;
            let mut sample_count = 0;

            for vector in vectors.iter().take(100) { // Sample subset for efficiency
                let compressed = self.simulate_compression(vector, level.bits_per_component)?;
                let reconstructed = self.simulate_decompression(&compressed, vector.len(), level.bits_per_component)?;
                
                // Calculate MSE
                let mse = vector.iter()
                    .zip(reconstructed.iter())
                    .map(|(&orig, &recon)| (orig - recon).powi(2))
                    .sum::<f32>() / vector.len() as f32;
                
                total_mse += mse as f64;
                sample_count += 1;
            }

            let avg_mse = total_mse / sample_count as f64;
            let psnr = if avg_mse > 1e-10 {
                20.0 * (1.0 / avg_mse.sqrt()).log10()
            } else {
                100.0 // Very high PSNR for perfect reconstruction
            };

            mse_by_level.insert(level.level_id, avg_mse);
            psnr_by_level.insert(level.level_id, psnr);
            compression_ratios.insert(level.level_id, level.expected_compression_ratio);
            
            // Estimate reconstruction time (microseconds per vector)
            let base_time = 10.0; // Base overhead
            let bit_time = 0.1 * (8.0 / level.bits_per_component as f64); // Inverse relationship
            reconstruction_times.insert(level.level_id, base_time + bit_time);
        }

        Ok(CompressionMetrics {
            mse_by_level,
            psnr_by_level,
            compression_ratios,
            reconstruction_times,
        })
    }

    /// Simulate compression to given bit depth
    fn simulate_compression(&mut self, vector: &[f32], bits_per_component: u8) -> Result<Vec<u8>> {
        let levels = 1u32 << bits_per_component;
        let mut compressed = Vec::new();

        for &value in vector {
            // Quantize to specified bit depth
            let normalized = (value + 1.0) / 2.0; // Normalize to [0, 1]
            let quantized = (normalized * (levels - 1) as f32).round() as u32;
            let clamped = quantized.min(levels - 1);
            compressed.push(clamped as u8);
        }

        Ok(compressed)
    }

    /// Simulate decompression from given bit depth
    fn simulate_decompression(&mut self, compressed: &[u8], original_length: usize, bits_per_component: u8) -> Result<Vec<f32>> {
        let levels = 1u32 << bits_per_component;
        let mut decompressed = Vec::new();

        for &quantized_value in compressed.iter().take(original_length) {
            // Dequantize back to float
            let normalized = quantized_value as f32 / (levels - 1) as f32;
            let value = normalized * 2.0 - 1.0; // Denormalize to [-1, 1]
            decompressed.push(value);
        }

        Ok(decompressed)
    }

    /// Calculate Euclidean distance between two vectors
    fn euclidean_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return f32::INFINITY;
        }

        v1.iter()
            .zip(v2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_quantization_test_data() {
        let mut generator = QuantizationDataGenerator::new(42);
        
        let test_set = generator.generate_product_quantization_test_data(50, 96, 256).unwrap();
        
        assert_eq!(test_set.original_vectors.len(), 50);
        assert_eq!(test_set.expected_approximations.len(), 50);
        assert!(test_set.expected_distances.len() > 0);
        assert!(test_set.compression_ratio > 1.0);
        
        // Verify vector dimensions
        for vector in &test_set.original_vectors {
            assert_eq!(vector.len(), 96);
        }
        
        // Verify approximations have same dimensions
        for approx in &test_set.expected_approximations {
            assert_eq!(approx.len(), 96);
        }
    }

    #[test]
    fn test_simd_test_vectors() {
        let mut generator = QuantizationDataGenerator::new(42);
        
        let test_set = generator.generate_simd_test_vectors(20).unwrap();
        
        assert_eq!(test_set.aligned_vectors.len(), 20);
        assert!(test_set.edge_case_vectors.len() > 0);
        assert!(test_set.expected_norms.len() > 0);
        
        // Verify SIMD alignment
        for vector in &test_set.aligned_vectors {
            assert_eq!(vector.len() % test_set.alignment_metadata.simd_width, 0);
        }
    }

    #[test]
    fn test_compression_test_data() {
        let mut generator = QuantizationDataGenerator::new(42);
        
        let test_data = generator.generate_compression_test_data(30, 64).unwrap();
        
        assert_eq!(test_data.baseline_vectors.len(), 30);
        assert!(test_data.compression_levels.len() > 0);
        
        // Verify compression levels are ordered
        let mut prev_bits = 0;
        for level in &test_data.compression_levels {
            assert!(level.bits_per_component > prev_bits);
            assert!(level.expected_compression_ratio > 1.0);
            prev_bits = level.bits_per_component;
        }
    }

    #[test]
    fn test_structured_vector_generation() {
        let mut generator = QuantizationDataGenerator::new(42);
        
        let vector1 = generator.generate_structured_vector(128, 0).unwrap();
        let vector2 = generator.generate_structured_vector(128, 1).unwrap();
        
        assert_eq!(vector1.len(), 128);
        assert_eq!(vector2.len(), 128);
        
        // Vectors should be different (different patterns)
        let distance = generator.euclidean_distance(&vector1, &vector2);
        assert!(distance > 0.1);
        
        // Vectors should be normalized
        let norm1 = vector1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2 = vector2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm1 - 1.0).abs() < 1e-6);
        assert!((norm2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantization_approximation() {
        let mut generator = QuantizationDataGenerator::new(42);
        
        let params = QuantizationParams {
            vector_dimension: 96,
            subvector_count: 8,
            codebook_size: 256,
            subvector_dimension: 12,
        };
        
        let vector = generator.generate_structured_vector(96, 0).unwrap();
        let approximation = generator.compute_expected_pq_approximation(&vector, &params).unwrap();
        
        assert_eq!(approximation.len(), 96);
        
        // Approximation should be reasonably close to original
        let distance = generator.euclidean_distance(&vector, &approximation);
        assert!(distance < 2.0); // Should be reasonable approximation
    }

    #[test]
    fn test_compression_metrics() {
        let mut generator = QuantizationDataGenerator::new(42);
        
        let vectors = vec![
            vec![1.0, 0.5, -0.5, -1.0],
            vec![0.8, 0.3, -0.3, -0.8],
        ];
        
        let levels = vec![
            CompressionLevel {
                level_id: 0,
                bits_per_component: 8,
                expected_quality: 0.95,
                expected_compression_ratio: 4.0,
            },
        ];
        
        let metrics = generator.calculate_compression_metrics(&vectors, &levels).unwrap();
        
        assert!(metrics.mse_by_level.contains_key(&0));
        assert!(metrics.psnr_by_level.contains_key(&0));
        assert!(metrics.compression_ratios.contains_key(&0));
        
        // MSE should be positive
        assert!(metrics.mse_by_level[&0] >= 0.0);
    }

    #[test]
    fn test_deterministic_generation() {
        let mut gen1 = QuantizationDataGenerator::new(12345);
        let mut gen2 = QuantizationDataGenerator::new(12345);
        
        let test_set1 = gen1.generate_product_quantization_test_data(10, 32, 128).unwrap();
        let test_set2 = gen2.generate_product_quantization_test_data(10, 32, 128).unwrap();
        
        // Same seed should produce identical results
        for (v1, v2) in test_set1.original_vectors.iter().zip(test_set2.original_vectors.iter()) {
            for (&a, &b) in v1.iter().zip(v2.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_invalid_parameters() {
        let mut generator = QuantizationDataGenerator::new(42);
        
        // Test zero vector count
        assert!(generator.generate_product_quantization_test_data(0, 96, 256).is_err());
        
        // Test zero dimension
        assert!(generator.generate_product_quantization_test_data(10, 0, 256).is_err());
        
        // Test invalid codebook size
        assert!(generator.generate_product_quantization_test_data(10, 96, 0).is_err());
        assert!(generator.generate_product_quantization_test_data(10, 96, 300).is_err());
        
        // Test dimension not divisible by subvector count
        assert!(generator.generate_product_quantization_test_data(10, 97, 256).is_err());
    }
}