//! Similarity Computation Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::embedding::similarity::*;

#[cfg(test)]
mod similarity_tests {
    use super::*;

    #[test]
    fn test_similarity_metrics() {
        let mut rng = DeterministicRng::new(SIMD_TEST_SEED);
        let dim = 128;
        
        let vec1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let vec2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        // Test cosine similarity
        let cosine_sim = cosine_similarity(&vec1, &vec2);
        assert!(cosine_sim >= -1.0 && cosine_sim <= 1.0);
        
        // Test dot product similarity  
        let dot_sim = dot_product_similarity(&vec1, &vec2);
        assert!(dot_sim.is_finite());
        
        // Test euclidean distance
        let euclidean_dist = euclidean_distance(&vec1, &vec2);
        assert!(euclidean_dist >= 0.0);
    }
}