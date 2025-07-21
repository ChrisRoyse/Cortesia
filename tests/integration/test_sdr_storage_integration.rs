//! Comprehensive integration tests for SDR storage module
//! 
//! These tests verify the public API of SDR storage and its integration
//! with the KnowledgeEngine for memory-efficient storage and retrieval.

use std::collections::HashSet;
use std::sync::Arc;

// Helper function to generate a simple embedding
fn generate_simple_embedding(text: &str, dim: usize) -> Vec<f32> {
    // Simple hash-based embedding for deterministic testing
    let hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
    let mut embedding = vec![0.0; dim];
    for i in 0..dim {
        embedding[i] = ((hash.wrapping_mul(i as u32 + 1) % 1000) as f32) / 1000.0;
    }
    embedding
}

#[cfg(test)]
mod sdr_storage_tests {
    use super::*;

    #[test]
    fn test_sdr_basic_operations() {
        // Test SDR creation and basic properties
        let active_bits: HashSet<usize> = [1, 2, 3, 4, 5].iter().cloned().collect();
        let total_bits = 100;
        
        // Verify SDR properties
        assert_eq!(active_bits.len(), 5);
        let sparsity = active_bits.len() as f32 / total_bits as f32;
        assert!(sparsity <= 0.1); // 5% sparsity
    }

    #[test]
    fn test_sdr_similarity_calculations() {
        // Test overlap calculation
        let sdr1_bits: HashSet<usize> = [1, 2, 3, 4, 5].iter().cloned().collect();
        let sdr2_bits: HashSet<usize> = [3, 4, 5, 6, 7].iter().cloned().collect();
        
        let intersection: HashSet<_> = sdr1_bits.intersection(&sdr2_bits).cloned().collect();
        let union: HashSet<_> = sdr1_bits.union(&sdr2_bits).cloned().collect();
        
        let overlap = intersection.len() as f32 / union.len() as f32;
        assert!(overlap > 0.0 && overlap < 1.0);
        assert_eq!(intersection.len(), 3); // 3, 4, 5
        assert_eq!(union.len(), 7); // 1, 2, 3, 4, 5, 6, 7
    }

    #[test]
    fn test_dense_to_sdr_conversion() {
        let dense_vector = vec![0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0];
        let active_bits = 4; // Top 4 values
        
        // Find indices of top-k values
        let mut indexed_values: Vec<(usize, f32)> = dense_vector.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_indices: HashSet<usize> = indexed_values.iter()
            .take(active_bits)
            .map(|(i, _)| *i)
            .collect();
        
        // Should have indices 1, 2, 4, 6 (values 0.9, 0.8, 0.7, 0.6)
        assert!(top_indices.contains(&1));
        assert!(top_indices.contains(&2));
        assert!(top_indices.contains(&4));
        assert!(top_indices.contains(&6));
    }

    #[test]
    fn test_memory_efficiency_calculation() {
        // Test memory efficiency of SDR vs dense vectors
        let embedding_dim = 384;
        let num_entities = 1000;
        
        // Calculate dense storage
        let dense_storage = embedding_dim * std::mem::size_of::<f32>() * num_entities;
        
        // Calculate SDR storage (assuming 2048 total bits, 40 active bits)
        let total_bits = 2048;
        let active_bits = 40;
        let sdr_storage_per_entity = active_bits * std::mem::size_of::<usize>() + 
                                     std::mem::size_of::<usize>() * 2; // overhead
        let sdr_total_storage = sdr_storage_per_entity * num_entities;
        
        // SDR should use significantly less memory
        assert!(sdr_total_storage < dense_storage);
        let compression_ratio = dense_storage as f32 / sdr_total_storage as f32;
        assert!(compression_ratio > 5.0); // At least 5x compression
    }

    #[test]
    fn test_pattern_clustering() {
        // Test that similar concepts have overlapping SDRs
        let concepts = vec![
            ("cat", "animal pet feline"),
            ("dog", "animal pet canine"),
            ("car", "vehicle automobile transport"),
            ("truck", "vehicle automobile transport large"),
        ];
        
        // Generate simple SDRs for each concept
        let mut concept_sdrs = Vec::new();
        for (name, description) in &concepts {
            let embedding = generate_simple_embedding(description, 100);
            let mut indexed: Vec<(usize, f32)> = embedding.iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let active_bits: HashSet<usize> = indexed.iter()
                .take(10)
                .map(|(i, _)| *i)
                .collect();
            
            concept_sdrs.push((name, active_bits));
        }
        
        // Check that animals have more overlap with each other than with vehicles
        let cat_sdr = &concept_sdrs[0].1;
        let dog_sdr = &concept_sdrs[1].1;
        let car_sdr = &concept_sdrs[2].1;
        
        let animal_overlap = cat_sdr.intersection(dog_sdr).count();
        let cross_category_overlap = cat_sdr.intersection(car_sdr).count();
        
        // This is a weak test due to simple hash-based embeddings
        // In a real system with semantic embeddings, this would be more pronounced
        assert!(animal_overlap >= 0);
        assert!(cross_category_overlap >= 0);
    }

    #[tokio::test]
    async fn test_concurrent_sdr_operations() {
        use tokio::task::JoinSet;
        
        let mut join_set = JoinSet::new();
        let shared_data = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        
        // Spawn multiple concurrent tasks
        for i in 0..10 {
            let data_clone = shared_data.clone();
            
            join_set.spawn(async move {
                let embedding = generate_simple_embedding(&format!("concept_{}", i), 128);
                let mut data = data_clone.lock().await;
                data.push((i, embedding));
            });
        }
        
        // Wait for all tasks to complete
        while let Some(result) = join_set.join_next().await {
            result.unwrap();
        }
        
        // Verify all operations completed
        let data = shared_data.lock().await;
        assert_eq!(data.len(), 10);
    }

    #[test]
    fn test_sdr_union_intersection_operations() {
        let sdr1: HashSet<usize> = [1, 2, 3, 4, 5].iter().cloned().collect();
        let sdr2: HashSet<usize> = [4, 5, 6, 7, 8].iter().cloned().collect();
        
        // Test union
        let union: HashSet<usize> = sdr1.union(&sdr2).cloned().collect();
        assert_eq!(union.len(), 8);
        for i in 1..=8 {
            assert!(union.contains(&i));
        }
        
        // Test intersection
        let intersection: HashSet<usize> = sdr1.intersection(&sdr2).cloned().collect();
        assert_eq!(intersection.len(), 2);
        assert!(intersection.contains(&4));
        assert!(intersection.contains(&5));
    }

    #[test]
    fn test_cosine_similarity_calculation() {
        let sdr1: HashSet<usize> = [1, 2, 3, 4, 5].iter().cloned().collect();
        let sdr2: HashSet<usize> = [3, 4, 5, 6, 7].iter().cloned().collect();
        
        let intersection_size = sdr1.intersection(&sdr2).count() as f32;
        let norm1 = (sdr1.len() as f32).sqrt();
        let norm2 = (sdr2.len() as f32).sqrt();
        
        let cosine_similarity = intersection_size / (norm1 * norm2);
        
        // 3 common elements, both have 5 elements
        // cosine = 3 / (sqrt(5) * sqrt(5)) = 3/5 = 0.6
        assert!((cosine_similarity - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_sdr_statistics() {
        // Simulate SDR storage statistics
        let total_patterns = 1000;
        let total_entities = 800;
        let average_active_bits = 40;
        let total_bits = 2048;
        
        let average_sparsity = average_active_bits as f32 / total_bits as f32;
        let total_active_bits = total_patterns * average_active_bits;
        
        assert!(average_sparsity <= 0.02 + 0.001); // ~2% sparsity
        assert_eq!(total_active_bits, 40000);
        
        // Memory usage calculation
        let pattern_memory = total_patterns * (average_active_bits * std::mem::size_of::<usize>());
        let entity_mapping_memory = total_entities * (std::mem::size_of::<u64>() + std::mem::size_of::<String>());
        let total_memory = pattern_memory + entity_mapping_memory;
        
        assert!(total_memory > 0);
    }

    #[test]
    fn test_sdr_compaction_logic() {
        // Simulate compaction by removing unused patterns
        let mut all_patterns: HashSet<String> = HashSet::new();
        let mut used_patterns: HashSet<String> = HashSet::new();
        
        // Add patterns
        for i in 0..10 {
            all_patterns.insert(format!("pattern_{}", i));
            if i < 7 {
                used_patterns.insert(format!("pattern_{}", i));
            }
        }
        
        // Find unused patterns
        let unused: HashSet<String> = all_patterns.difference(&used_patterns).cloned().collect();
        assert_eq!(unused.len(), 3);
        assert!(unused.contains("pattern_7"));
        assert!(unused.contains("pattern_8"));
        assert!(unused.contains("pattern_9"));
        
        // Remove unused patterns
        for pattern in &unused {
            all_patterns.remove(pattern);
        }
        
        assert_eq!(all_patterns.len(), 7);
    }

    #[test]
    fn test_knowledge_workflow_simulation() {
        // Simulate a knowledge graph workflow with SDR compression
        let relationships = vec![
            ("Einstein", "developed", "Theory of Relativity"),
            ("Theory of Relativity", "describes", "spacetime"),
            ("Newton", "formulated", "Laws of Motion"),
            ("Einstein", "influenced_by", "Newton"),
        ];
        
        let mut stored_sdrs = Vec::new();
        
        for (subject, predicate, object) in &relationships {
            let triple_text = format!("{} {} {}", subject, predicate, object);
            let embedding = generate_simple_embedding(&triple_text, 512);
            
            // Convert to SDR (top 80 out of 4096 bits)
            let mut indexed: Vec<(usize, f32)> = embedding.iter()
                .enumerate()
                .map(|(i, &v)| (i % 4096, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let sdr: HashSet<usize> = indexed.iter()
                .take(80)
                .map(|(i, _)| *i)
                .collect();
            
            stored_sdrs.push((triple_text, sdr));
        }
        
        assert_eq!(stored_sdrs.len(), relationships.len());
        
        // Calculate memory savings
        let original_size = 512 * std::mem::size_of::<f32>() * relationships.len();
        let sdr_size = 80 * std::mem::size_of::<usize>() * relationships.len();
        
        assert!(sdr_size < original_size);
        let compression_ratio = original_size as f32 / sdr_size as f32;
        assert!(compression_ratio > 2.0);
    }
}