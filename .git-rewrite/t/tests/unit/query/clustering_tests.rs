//! Clustering Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::query::clustering::*;

#[cfg(test)]
mod clustering_tests {
    use super::*;

    #[test]
    fn test_entity_clustering() {
        let mut rng = DeterministicRng::new(RAG_TEST_SEED);
        let clusterer = KMeansClusterer::new(5);
        
        // Generate test entities
        let entities = (0..100).map(|i| {
            let embedding: Vec<f32> = (0..64).map(|_| rng.gen_range(-1.0..1.0)).collect();
            ClusterableEntity::new(format!("entity_{}", i), embedding)
        }).collect();
        
        let clusters = clusterer.cluster(&entities).unwrap();
        
        assert_eq!(clusters.len(), 5);
        assert!(clusters.iter().all(|cluster| !cluster.entities.is_empty()));
    }
}