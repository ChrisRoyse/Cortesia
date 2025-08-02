use std::collections::HashSet;
use rand::prelude::*;
use crate::error::{Result, GraphError};
use crate::core::sdr_types::{SDR, SDRConfig};

impl SDR {
    pub fn new(active_bits: HashSet<usize>, total_bits: usize) -> Self {
        Self {
            active_bits,
            total_bits,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Create SDR from dense vector
    pub fn from_dense_vector(vector: &[f32], config: &SDRConfig) -> Self {
        // Find indices of top-k values to create sparse representation
        let mut indexed_values: Vec<(usize, f32)> = vector.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        // Sort by value (descending)
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top active_bits indices
        let active_bits: HashSet<usize> = indexed_values.iter()
            .take(config.active_bits)
            .map(|(i, _)| *i)
            .collect();

        Self::new(active_bits, config.total_bits)
    }

    /// Create random SDR
    pub fn random(config: &SDRConfig) -> Self {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let mut indices: Vec<usize> = (0..config.total_bits).collect();
        indices.shuffle(&mut thread_rng());
        
        let active_bits: HashSet<usize> = indices.into_iter()
            .take(config.active_bits)
            .collect();

        Self::new(active_bits, config.total_bits)
    }

    /// Create random SDR with provided RNG
    pub fn random_with_rng<R: Rng>(config: &SDRConfig, rng: &mut R) -> Self {
        use rand::seq::SliceRandom;
        
        let mut indices: Vec<usize> = (0..config.total_bits).collect();
        indices.shuffle(rng);
        
        let active_bits: HashSet<usize> = indices.into_iter()
            .take(config.active_bits)
            .collect();

        Self::new(active_bits, config.total_bits)
    }

    /// Calculate overlap with another SDR
    pub fn overlap(&self, other: &SDR) -> f32 {
        if self.total_bits != other.total_bits {
            return 0.0;
        }

        let intersection_size = self.active_bits.intersection(&other.active_bits).count();
        let union_size = self.active_bits.union(&other.active_bits).count();
        
        if union_size == 0 {
            0.0
        } else {
            intersection_size as f32 / union_size as f32
        }
    }

    /// Calculate Jaccard similarity
    pub fn jaccard_similarity(&self, other: &SDR) -> f32 {
        if self.total_bits != other.total_bits {
            return 0.0;
        }

        let intersection_size = self.active_bits.intersection(&other.active_bits).count();
        let union_size = self.active_bits.union(&other.active_bits).count();
        
        if union_size == 0 {
            0.0
        } else {
            intersection_size as f32 / union_size as f32
        }
    }

    /// Calculate cosine similarity (treating SDR as binary vector)
    pub fn cosine_similarity(&self, other: &SDR) -> f32 {
        if self.total_bits != other.total_bits {
            return 0.0;
        }

        let intersection_size = self.active_bits.intersection(&other.active_bits).count();
        let norm_self = (self.active_bits.len() as f32).sqrt();
        let norm_other = (other.active_bits.len() as f32).sqrt();
        
        if norm_self == 0.0 || norm_other == 0.0 {
            0.0
        } else {
            intersection_size as f32 / (norm_self * norm_other)
        }
    }

    /// Convert to dense vector
    pub fn to_dense_vector(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.total_bits];
        for &bit in &self.active_bits {
            if bit < dense.len() {
                dense[bit] = 1.0;
            }
        }
        dense
    }

    /// Get sparsity (fraction of active bits)
    pub fn sparsity(&self) -> f32 {
        self.active_bits.len() as f32 / self.total_bits as f32
    }

    /// Union with another SDR
    pub fn union(&self, other: &SDR) -> Result<SDR> {
        if self.total_bits != other.total_bits {
            return Err(GraphError::InvalidInput("SDR dimensions must match".to_string()));
        }

        let active_bits = self.active_bits.union(&other.active_bits).cloned().collect();
        Ok(SDR::new(active_bits, self.total_bits))
    }

    /// Intersection with another SDR
    pub fn intersection(&self, other: &SDR) -> Result<SDR> {
        if self.total_bits != other.total_bits {
            return Err(GraphError::InvalidInput("SDR dimensions must match".to_string()));
        }

        let active_bits = self.active_bits.intersection(&other.active_bits).cloned().collect();
        Ok(SDR::new(active_bits, self.total_bits))
    }
}