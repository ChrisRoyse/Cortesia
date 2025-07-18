# Phase 3: Unit Testing Framework

## Overview

Phase 3 creates a comprehensive unit testing framework that validates every individual component of the LLMKG system in isolation. This phase ensures 100% code coverage with deterministic, predictable outcomes for all unit-level functionality.

## Objectives

1. **Complete Code Coverage**: Test every function, method, and code path
2. **Component Isolation**: Test each module independently with mocked dependencies
3. **Edge Case Coverage**: Test all boundary conditions and error scenarios
4. **Performance Validation**: Verify unit-level performance characteristics
5. **Deterministic Results**: Ensure all tests produce identical, predictable outcomes
6. **Regression Prevention**: Catch breaking changes at the unit level

## Detailed Implementation Plan

### 1. Core Graph Operations Testing

#### 1.1 Entity Management Tests
**File**: `tests/unit/core/entity_tests.rs`

```rust
mod entity_tests {
    use super::*;
    use crate::test_infrastructure::*;
    
    #[test]
    fn test_entity_creation_deterministic() {
        let mut rng = DeterministicRng::new(ENTITY_TEST_SEED);
        
        // Test 1: Basic entity creation
        let entity_key = EntityKey::from_hash("test_entity_1");
        let entity = Entity::new(entity_key, "Test Entity".to_string());
        
        assert_eq!(entity.key(), entity_key);
        assert_eq!(entity.name(), "Test Entity");
        assert_eq!(entity.attributes().len(), 0);
        assert_eq!(entity.memory_usage(), EXPECTED_EMPTY_ENTITY_SIZE);
        
        // Test 2: Entity with attributes
        let mut entity_with_attrs = Entity::new(
            EntityKey::from_hash("test_entity_2"), 
            "Entity with Attributes".to_string()
        );
        
        entity_with_attrs.add_attribute("type", "test");
        entity_with_attrs.add_attribute("value", "42");
        entity_with_attrs.add_attribute("flag", "true");
        
        assert_eq!(entity_with_attrs.attributes().len(), 3);
        assert_eq!(entity_with_attrs.get_attribute("type"), Some("test"));
        assert_eq!(entity_with_attrs.get_attribute("value"), Some("42"));
        assert_eq!(entity_with_attrs.get_attribute("nonexistent"), None);
        
        // Verify memory usage calculation
        let expected_memory = EXPECTED_EMPTY_ENTITY_SIZE 
            + "type".len() + "test".len() 
            + "value".len() + "42".len()
            + "flag".len() + "true".len()
            + ATTRIBUTE_OVERHEAD * 3;
        assert_eq!(entity_with_attrs.memory_usage(), expected_memory);
        
        // Test 3: Entity serialization determinism
        let serialized1 = entity_with_attrs.serialize();
        let serialized2 = entity_with_attrs.serialize();
        assert_eq!(serialized1, serialized2);
        
        let deserialized = Entity::deserialize(&serialized1).unwrap();
        assert_eq!(entity_with_attrs, deserialized);
    }
    
    #[test]
    fn test_entity_key_generation() {
        // Test deterministic key generation
        let key1 = EntityKey::from_hash("identical_input");
        let key2 = EntityKey::from_hash("identical_input");
        assert_eq!(key1, key2);
        
        // Test key uniqueness
        let key3 = EntityKey::from_hash("different_input");
        assert_ne!(key1, key3);
        
        // Test key collision resistance (birthday paradox test)
        let mut keys = HashSet::new();
        for i in 0..100000 {
            let key = EntityKey::from_hash(format!("key_{}", i));
            assert!(!keys.contains(&key), "Key collision detected at iteration {}", i);
            keys.insert(key);
        }
    }
    
    #[test]
    fn test_entity_attribute_edge_cases() {
        let mut entity = Entity::new(EntityKey::from_hash("test"), "Test".to_string());
        
        // Test empty string values
        entity.add_attribute("empty", "");
        assert_eq!(entity.get_attribute("empty"), Some(""));
        
        // Test Unicode values
        entity.add_attribute("unicode", "🦀 Rust 测试 🌟");
        assert_eq!(entity.get_attribute("unicode"), Some("🦀 Rust 测试 🌟"));
        
        // Test very long values
        let long_value = "x".repeat(10000);
        entity.add_attribute("long", &long_value);
        assert_eq!(entity.get_attribute("long"), Some(&long_value));
        
        // Test attribute overwriting
        entity.add_attribute("overwrite", "original");
        entity.add_attribute("overwrite", "updated");
        assert_eq!(entity.get_attribute("overwrite"), Some("updated"));
        
        // Test case sensitivity
        entity.add_attribute("CaseSensitive", "value1");
        entity.add_attribute("casesensitive", "value2");
        assert_eq!(entity.get_attribute("CaseSensitive"), Some("value1"));
        assert_eq!(entity.get_attribute("casesensitive"), Some("value2"));
    }
    
    #[test]
    fn test_entity_memory_management() {
        let mut entity = Entity::new(EntityKey::from_hash("memory_test"), "Memory Test".to_string());
        let initial_memory = entity.memory_usage();
        
        // Add attributes and verify memory growth
        entity.add_attribute("attr1", "value1");
        let memory_after_1 = entity.memory_usage();
        assert!(memory_after_1 > initial_memory);
        
        entity.add_attribute("attr2", "value2");
        let memory_after_2 = entity.memory_usage();
        assert!(memory_after_2 > memory_after_1);
        
        // Remove attribute and verify memory decrease
        entity.remove_attribute("attr1");
        let memory_after_removal = entity.memory_usage();
        assert!(memory_after_removal < memory_after_2);
        
        // Verify memory calculation accuracy
        let expected_memory = calculate_expected_entity_memory(&entity);
        assert_eq!(entity.memory_usage(), expected_memory);
    }
}
```

#### 1.2 Graph Structure Tests
**File**: `tests/unit/core/graph_tests.rs`

```rust
mod graph_tests {
    use super::*;
    use crate::test_infrastructure::*;
    
    #[test]
    fn test_graph_basic_operations() {
        let mut graph = KnowledgeGraph::new();
        
        // Test 1: Empty graph properties
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.relationship_count(), 0);
        assert_eq!(graph.memory_usage(), EXPECTED_EMPTY_GRAPH_SIZE);
        
        // Test 2: Add single entity
        let entity_key = EntityKey::from_hash("entity_1");
        let entity = Entity::new(entity_key, "Entity 1".to_string());
        
        let add_result = graph.add_entity(entity.clone());
        assert!(add_result.is_ok());
        assert_eq!(graph.entity_count(), 1);
        assert!(graph.contains_entity(entity_key));
        
        // Test 3: Retrieve entity
        let retrieved = graph.get_entity(entity_key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), &entity);
        
        // Test 4: Add duplicate entity (should fail)
        let duplicate_result = graph.add_entity(entity.clone());
        assert!(duplicate_result.is_err());
        assert_eq!(graph.entity_count(), 1); // Count unchanged
        
        // Test 5: Add second entity
        let entity2_key = EntityKey::from_hash("entity_2");
        let entity2 = Entity::new(entity2_key, "Entity 2".to_string());
        
        graph.add_entity(entity2).unwrap();
        assert_eq!(graph.entity_count(), 2);
        
        // Test 6: Add relationship
        let relationship = Relationship::new("connects".to_string(), 1.0, RelationshipType::Directed);
        let rel_result = graph.add_relationship(entity_key, entity2_key, relationship);
        assert!(rel_result.is_ok());
        assert_eq!(graph.relationship_count(), 1);
        
        // Test 7: Verify relationship existence
        let relationships = graph.get_relationships(entity_key);
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].target(), entity2_key);
        assert_eq!(relationships[0].relationship().name(), "connects");
    }
    
    #[test]
    fn test_graph_csr_storage_format() {
        let mut graph = KnowledgeGraph::new();
        let (entities, relationships) = create_test_graph_data(100, 200);
        
        // Add all entities
        for entity in entities {
            graph.add_entity(entity).unwrap();
        }
        
        // Add all relationships
        for (source, target, rel) in relationships {
            graph.add_relationship(source, target, rel).unwrap();
        }
        
        // Test CSR format properties
        let csr_storage = graph.get_csr_storage();
        
        // Verify row offsets are monotonically increasing
        let offsets = csr_storage.row_offsets();
        for i in 1..offsets.len() {
            assert!(offsets[i] >= offsets[i-1], "CSR offsets not monotonic at index {}", i);
        }
        
        // Verify column indices are valid
        let columns = csr_storage.column_indices();
        let max_entity_id = graph.entity_count() as u32;
        for &col_idx in columns {
            assert!(col_idx < max_entity_id, "Invalid column index: {}", col_idx);
        }
        
        // Verify data integrity
        let total_relationships = offsets[offsets.len() - 1];
        assert_eq!(total_relationships as usize, columns.len());
        assert_eq!(total_relationships as u64, graph.relationship_count());
        
        // Test cache-friendly access patterns
        let access_time = measure_sequential_access_time(&csr_storage);
        let random_access_time = measure_random_access_time(&csr_storage);
        
        // Sequential access should be significantly faster
        assert!(access_time < random_access_time * 0.8, 
                "CSR format not showing cache-friendly behavior");
    }
    
    #[test]
    fn test_graph_memory_efficiency() {
        let mut graph = KnowledgeGraph::new();
        let entity_count = 1000u64;
        let relationship_count = 2000u64;
        
        // Add entities and track memory growth
        let mut memory_measurements = Vec::new();
        
        for i in 0..entity_count {
            let entity_key = EntityKey::from_hash(format!("entity_{}", i));
            let entity = Entity::new(entity_key, format!("Entity {}", i));
            graph.add_entity(entity).unwrap();
            
            if i % 100 == 0 {
                memory_measurements.push((i, graph.memory_usage()));
            }
        }
        
        // Verify linear memory growth for entities
        for window in memory_measurements.windows(2) {
            let (count1, memory1) = window[0];
            let (count2, memory2) = window[1];
            
            let entity_diff = count2 - count1;
            let memory_diff = memory2 - memory1;
            let memory_per_entity = memory_diff / entity_diff;
            
            // Should be close to expected entity size
            assert!(memory_per_entity <= EXPECTED_ENTITY_SIZE_UPPER_BOUND,
                   "Memory per entity {} exceeds bound {}", 
                   memory_per_entity, EXPECTED_ENTITY_SIZE_UPPER_BOUND);
        }
        
        // Test target: < 70 bytes per entity
        let final_memory = graph.memory_usage();
        let memory_per_entity = final_memory / entity_count;
        assert!(memory_per_entity < 70, 
               "Memory per entity {} exceeds 70 byte target", memory_per_entity);
    }
    
    #[test]
    fn test_graph_concurrent_access() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let graph = Arc::new(Mutex::new(KnowledgeGraph::new()));
        let entity_count = 100;
        let thread_count = 4;
        
        // Prepare entities for each thread
        let entities_per_thread = entity_count / thread_count;
        let mut handles = Vec::new();
        
        for thread_id in 0..thread_count {
            let graph_clone = Arc::clone(&graph);
            
            let handle = thread::spawn(move || {
                let start_idx = thread_id * entities_per_thread;
                let end_idx = start_idx + entities_per_thread;
                
                for i in start_idx..end_idx {
                    let entity_key = EntityKey::from_hash(format!("thread_{}_{}", thread_id, i));
                    let entity = Entity::new(entity_key, format!("Entity {}", i));
                    
                    let mut graph = graph_clone.lock().unwrap();
                    graph.add_entity(entity).unwrap();
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify final state
        let graph = graph.lock().unwrap();
        assert_eq!(graph.entity_count(), entity_count as u64);
        
        // Verify all entities are present and accessible
        for thread_id in 0..thread_count {
            for i in 0..entities_per_thread {
                let entity_key = EntityKey::from_hash(format!("thread_{}_{}", thread_id, i));
                assert!(graph.contains_entity(entity_key));
            }
        }
    }
}
```

### 2. Storage Layer Testing

#### 2.1 CSR Format Tests
**File**: `tests/unit/storage/csr_tests.rs`

```rust
mod csr_tests {
    use super::*;
    use crate::test_infrastructure::*;
    
    #[test]
    fn test_csr_construction_deterministic() {
        let mut rng = DeterministicRng::new(CSR_TEST_SEED);
        
        // Create test adjacency matrix
        let n = 100;
        let density = 0.1;
        let adjacency_matrix = generate_random_adjacency_matrix(&mut rng, n, density);
        
        // Convert to CSR format
        let csr = CompressedSparseRow::from_adjacency_matrix(&adjacency_matrix);
        
        // Test 1: Verify structure integrity
        assert_eq!(csr.num_rows(), n);
        assert_eq!(csr.row_offsets().len(), n + 1);
        
        // Test 2: Verify data consistency
        let expected_nnz = count_nonzeros(&adjacency_matrix);
        assert_eq!(csr.num_nonzeros(), expected_nnz);
        assert_eq!(csr.column_indices().len(), expected_nnz);
        assert_eq!(csr.values().len(), expected_nnz);
        
        // Test 3: Verify CSR properties
        let offsets = csr.row_offsets();
        for i in 1..offsets.len() {
            assert!(offsets[i] >= offsets[i-1], "Row offsets not monotonic");
        }
        
        // Test 4: Verify round-trip conversion
        let reconstructed_matrix = csr.to_adjacency_matrix();
        assert_matrices_equal(&adjacency_matrix, &reconstructed_matrix);
        
        // Test 5: Test deterministic construction
        let csr2 = CompressedSparseRow::from_adjacency_matrix(&adjacency_matrix);
        assert_eq!(csr, csr2);
    }
    
    #[test]
    fn test_csr_access_patterns() {
        let csr = create_test_csr_matrix(1000, 0.05);
        
        // Test 1: Sequential row access
        let start_time = std::time::Instant::now();
        for row in 0..csr.num_rows() {
            let _row_data = csr.get_row(row);
        }
        let sequential_time = start_time.elapsed();
        
        // Test 2: Random row access
        let mut rng = DeterministicRng::new(ACCESS_PATTERN_SEED);
        let random_rows: Vec<usize> = (0..csr.num_rows())
            .map(|_| rng.gen_range(0..csr.num_rows()))
            .collect();
        
        let start_time = std::time::Instant::now();
        for &row in &random_rows {
            let _row_data = csr.get_row(row);
        }
        let random_time = start_time.elapsed();
        
        // Sequential access should be faster (cache-friendly)
        let ratio = random_time.as_nanos() as f64 / sequential_time.as_nanos() as f64;
        assert!(ratio > 1.2, "CSR format not showing cache advantage: ratio {}", ratio);
        
        // Test 3: Column access performance
        let start_time = std::time::Instant::now();
        for col in 0..csr.num_rows() {
            let _col_elements = csr.get_column_elements(col);
        }
        let column_time = start_time.elapsed();
        
        // Row access should be much faster than column access
        let row_col_ratio = column_time.as_nanos() as f64 / sequential_time.as_nanos() as f64;
        assert!(row_col_ratio > 5.0, "Column access not showing expected slowdown: ratio {}", row_col_ratio);
    }
    
    #[test]
    fn test_csr_memory_layout() {
        let sizes = vec![10, 100, 1000, 10000];
        let density = 0.05;
        
        for &size in &sizes {
            let csr = create_test_csr_matrix(size, density);
            
            // Calculate expected memory usage
            let expected_offsets_size = (size + 1) * std::mem::size_of::<usize>();
            let nnz = csr.num_nonzeros();
            let expected_indices_size = nnz * std::mem::size_of::<usize>();
            let expected_values_size = nnz * std::mem::size_of::<f32>();
            let expected_total = expected_offsets_size + expected_indices_size + expected_values_size;
            
            let actual_memory = csr.memory_usage();
            
            // Allow small overhead for struct metadata
            let overhead_ratio = actual_memory as f64 / expected_total as f64;
            assert!(overhead_ratio < 1.1, 
                   "CSR memory overhead too high: {} vs {} (ratio: {})", 
                   actual_memory, expected_total, overhead_ratio);
            
            // Verify memory usage scales linearly with non-zeros
            let memory_per_nnz = actual_memory as f64 / nnz as f64;
            let expected_per_nnz = (std::mem::size_of::<usize>() + std::mem::size_of::<f32>()) as f64;
            
            assert!((memory_per_nnz - expected_per_nnz).abs() < expected_per_nnz * 0.2,
                   "Memory per non-zero element incorrect: {} vs {}", 
                   memory_per_nnz, expected_per_nnz);
        }
    }
    
    #[test]
    fn test_csr_operations() {
        let mut rng = DeterministicRng::new(CSR_OPERATIONS_SEED);
        let csr = create_test_csr_matrix(500, 0.08);
        
        // Test 1: Matrix-vector multiplication
        let vector: Vec<f32> = (0..csr.num_rows()).map(|i| (i as f32) * 0.1).collect();
        let result = csr.multiply_vector(&vector);
        
        // Verify result dimensions
        assert_eq!(result.len(), csr.num_rows());
        
        // Verify against naive implementation
        let expected_result = naive_matrix_vector_multiply(&csr, &vector);
        for (i, (&actual, &expected)) in result.iter().zip(expected_result.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6, 
                   "Matrix-vector multiplication mismatch at index {}: {} vs {}", 
                   i, actual, expected);
        }
        
        // Test 2: Sparse matrix addition
        let csr2 = create_test_csr_matrix_with_seed(500, 0.08, CSR_OPERATIONS_SEED + 1);
        let sum_result = csr.add(&csr2);
        
        // Verify dimensions
        assert_eq!(sum_result.num_rows(), csr.num_rows());
        
        // Verify addition correctness by spot-checking
        for _ in 0..100 {
            let row = rng.gen_range(0..csr.num_rows());
            let col = rng.gen_range(0..csr.num_rows());
            
            let val1 = csr.get_element(row, col);
            let val2 = csr2.get_element(row, col);
            let sum_val = sum_result.get_element(row, col);
            
            assert!((sum_val - (val1 + val2)).abs() < 1e-6,
                   "Matrix addition incorrect at ({}, {}): {} vs {}", 
                   row, col, sum_val, val1 + val2);
        }
    }
}
```

#### 2.2 Bloom Filter Tests
**File**: `tests/unit/storage/bloom_tests.rs`

```rust
mod bloom_tests {
    use super::*;
    use crate::test_infrastructure::*;
    
    #[test]
    fn test_bloom_filter_basic_operations() {
        let expected_items = 10000;
        let false_positive_rate = 0.01;
        let mut bloom = BloomFilter::new(expected_items, false_positive_rate);
        
        // Test 1: Empty filter properties
        assert_eq!(bloom.len(), 0);
        assert!(!bloom.contains("nonexistent"));
        
        // Test 2: Add items and test membership
        let test_items: Vec<String> = (0..1000).map(|i| format!("item_{}", i)).collect();
        
        for item in &test_items {
            bloom.insert(item);
        }
        
        assert_eq!(bloom.len(), test_items.len());
        
        // All inserted items should be found (no false negatives)
        for item in &test_items {
            assert!(bloom.contains(item), "False negative for item: {}", item);
        }
        
        // Test 3: False positive rate validation
        let false_positive_count = count_false_positives(&bloom, &test_items, 10000);
        let actual_fp_rate = false_positive_count as f64 / 10000.0;
        
        // Should be close to expected rate (within 50% margin)
        assert!(actual_fp_rate <= false_positive_rate * 1.5,
               "False positive rate too high: {} vs expected {}", 
               actual_fp_rate, false_positive_rate);
        
        // Test 4: Memory usage
        let expected_memory = calculate_expected_bloom_memory(expected_items, false_positive_rate);
        let actual_memory = bloom.memory_usage();
        
        assert!((actual_memory as f64 - expected_memory as f64).abs() < expected_memory as f64 * 0.1,
               "Bloom filter memory usage incorrect: {} vs {}", actual_memory, expected_memory);
    }
    
    #[test]
    fn test_bloom_filter_deterministic() {
        let items = vec!["apple", "banana", "cherry", "date", "elderberry"];
        
        // Create two identical bloom filters
        let mut bloom1 = BloomFilter::new(100, 0.01);
        let mut bloom2 = BloomFilter::new(100, 0.01);
        
        // Add same items to both
        for item in &items {
            bloom1.insert(item);
            bloom2.insert(item);
        }
        
        // Should have identical internal state
        assert_eq!(bloom1.get_bit_array(), bloom2.get_bit_array());
        assert_eq!(bloom1.len(), bloom2.len());
        
        // Should give identical results for all queries
        for i in 0..1000 {
            let test_item = format!("test_item_{}", i);
            assert_eq!(bloom1.contains(&test_item), bloom2.contains(&test_item));
        }
    }
    
    #[test]
    fn test_bloom_filter_serialization() {
        let mut bloom = BloomFilter::new(1000, 0.01);
        let items: Vec<String> = (0..500).map(|i| format!("serialize_test_{}", i)).collect();
        
        for item in &items {
            bloom.insert(item);
        }
        
        // Serialize and deserialize
        let serialized = bloom.serialize().unwrap();
        let deserialized = BloomFilter::deserialize(&serialized).unwrap();
        
        // Should be functionally identical
        assert_eq!(bloom.len(), deserialized.len());
        
        // Test membership for all original items
        for item in &items {
            assert_eq!(bloom.contains(item), deserialized.contains(item));
        }
        
        // Test on random queries
        let mut rng = DeterministicRng::new(BLOOM_SERIALIZE_SEED);
        for _ in 0..1000 {
            let test_item = format!("random_test_{}", rng.gen::<u64>());
            assert_eq!(bloom.contains(&test_item), deserialized.contains(&test_item));
        }
    }
    
    #[test]
    fn test_bloom_filter_capacity_growth() {
        let initial_capacity = 100;
        let mut bloom = BloomFilter::new(initial_capacity, 0.01);
        
        // Fill beyond initial capacity
        for i in 0..(initial_capacity * 3) {
            bloom.insert(&format!("capacity_test_{}", i));
        }
        
        // Should still function correctly (though with higher false positive rate)
        for i in 0..(initial_capacity * 3) {
            let item = format!("capacity_test_{}", i);
            assert!(bloom.contains(&item), "Lost item after capacity exceeded: {}", item);
        }
        
        // Check that false positive rate hasn't become too high
        let fp_count = count_false_positives(&bloom, &[], 10000);
        let fp_rate = fp_count as f64 / 10000.0;
        
        // Should be reasonable even when overloaded
        assert!(fp_rate < 0.5, "False positive rate too high after overload: {}", fp_rate);
    }
    
    #[test]
    fn test_bloom_filter_hash_function_quality() {
        let mut bloom = BloomFilter::new(10000, 0.001);
        
        // Test hash distribution for sequential inputs
        let sequential_items: Vec<String> = (0..1000).map(|i| format!("{}", i)).collect();
        for item in &sequential_items {
            bloom.insert(item);
        }
        
        // Test hash distribution for similar strings
        let similar_items: Vec<String> = (0..1000).map(|i| format!("prefix_{}", i)).collect();
        for item in &similar_items {
            bloom.insert(item);
        }
        
        // Verify bit distribution is reasonably uniform
        let bit_array = bloom.get_bit_array();
        let set_bits = bit_array.iter().filter(|&&bit| bit).count();
        let total_bits = bit_array.len();
        let set_ratio = set_bits as f64 / total_bits as f64;
        
        // Should be roughly 50% for good hash distribution
        assert!(set_ratio > 0.3 && set_ratio < 0.7,
               "Poor hash distribution: {}% bits set", set_ratio * 100.0);
        
        // Test avalanche effect (small input changes cause large hash changes)
        test_hash_avalanche_effect(&bloom);
    }
}

fn count_false_positives(bloom: &BloomFilter, known_items: &[String], test_count: usize) -> usize {
    let mut false_positives = 0;
    let mut rng = DeterministicRng::new(FALSE_POSITIVE_SEED);
    
    for _ in 0..test_count {
        let test_item = format!("fp_test_{}", rng.gen::<u64>());
        if !known_items.contains(&test_item) && bloom.contains(&test_item) {
            false_positives += 1;
        }
    }
    
    false_positives
}
```

### 3. Embedding System Testing

#### 3.1 Vector Quantization Tests
**File**: `tests/unit/embedding/quantization_tests.rs`

```rust
mod quantization_tests {
    use super::*;
    use crate::test_infrastructure::*;
    
    #[test]
    fn test_product_quantization_accuracy() {
        let mut rng = DeterministicRng::new(PQ_TEST_SEED);
        let dimension = 128;
        let codebook_size = 256;
        let vector_count = 1000;
        
        // Generate test vectors with known properties
        let test_vectors = generate_clustered_vectors(&mut rng, vector_count, dimension, 10);
        
        // Train product quantizer
        let mut quantizer = ProductQuantizer::new(dimension, codebook_size);
        quantizer.train(&test_vectors).unwrap();
        
        // Test 1: Quantization and reconstruction
        let mut total_error = 0.0;
        let mut max_error = 0.0;
        
        for vector in &test_vectors {
            let quantized = quantizer.quantize(vector);
            let reconstructed = quantizer.reconstruct(&quantized);
            
            assert_eq!(reconstructed.len(), vector.len());
            
            let error = euclidean_distance(vector, &reconstructed);
            total_error += error;
            max_error = max_error.max(error);
        }
        
        let average_error = total_error / vector_count as f32;
        
        // Verify compression quality
        assert!(average_error < EXPECTED_PQ_AVERAGE_ERROR,
               "Average quantization error too high: {} vs {}", 
               average_error, EXPECTED_PQ_AVERAGE_ERROR);
        
        assert!(max_error < EXPECTED_PQ_MAX_ERROR,
               "Maximum quantization error too high: {} vs {}", 
               max_error, EXPECTED_PQ_MAX_ERROR);
        
        // Test 2: Compression ratio
        let original_size = vector_count * dimension * std::mem::size_of::<f32>();
        let compressed_size = quantizer.compressed_size(vector_count);
        let compression_ratio = original_size as f64 / compressed_size as f64;
        
        assert!(compression_ratio >= EXPECTED_MIN_COMPRESSION_RATIO,
               "Compression ratio too low: {} vs {}", 
               compression_ratio, EXPECTED_MIN_COMPRESSION_RATIO);
        
        // Test 3: Deterministic behavior
        let quantized1 = quantizer.quantize(&test_vectors[0]);
        let quantized2 = quantizer.quantize(&test_vectors[0]);
        assert_eq!(quantized1, quantized2);
        
        let reconstructed1 = quantizer.reconstruct(&quantized1);
        let reconstructed2 = quantizer.reconstruct(&quantized1);
        assert_vectors_equal(&reconstructed1, &reconstructed2, 1e-10);
    }
    
    #[test]
    fn test_quantizer_training_convergence() {
        let mut rng = DeterministicRng::new(PQ_TRAINING_SEED);
        let dimension = 64;
        let codebook_size = 128;
        
        // Generate training data with clear cluster structure
        let training_vectors = generate_clear_clusters(&mut rng, 5000, dimension, 8);
        
        let mut quantizer = ProductQuantizer::new(dimension, codebook_size);
        
        // Train with convergence monitoring
        let training_result = quantizer.train_with_monitoring(&training_vectors);
        
        assert!(training_result.converged, "Quantizer training did not converge");
        assert!(training_result.final_loss < EXPECTED_FINAL_TRAINING_LOSS,
               "Training loss too high: {} vs {}", 
               training_result.final_loss, EXPECTED_FINAL_TRAINING_LOSS);
        
        // Verify training stability
        assert!(training_result.loss_variance < EXPECTED_LOSS_VARIANCE,
               "Training loss too unstable: variance {}", training_result.loss_variance);
        
        // Test codebook quality
        verify_codebook_quality(&quantizer, &training_vectors);
    }
    
    #[test]
    fn test_quantization_edge_cases() {
        let dimension = 32;
        let codebook_size = 64;
        let mut quantizer = ProductQuantizer::new(dimension, codebook_size);
        
        // Test 1: Zero vector
        let zero_vector = vec![0.0; dimension];
        let training_data = vec![zero_vector.clone(); 100];
        quantizer.train(&training_data).unwrap();
        
        let quantized = quantizer.quantize(&zero_vector);
        let reconstructed = quantizer.reconstruct(&quantized);
        
        for &value in &reconstructed {
            assert!(value.abs() < 1e-6, "Zero vector reconstruction error: {}", value);
        }
        
        // Test 2: Unit vectors
        for dim in 0..dimension {
            let mut unit_vector = vec![0.0; dimension];
            unit_vector[dim] = 1.0;
            
            let quantized = quantizer.quantize(&unit_vector);
            let reconstructed = quantizer.reconstruct(&quantized);
            
            let error = euclidean_distance(&unit_vector, &reconstructed);
            assert!(error < 1.0, "Unit vector quantization error too high: {}", error);
        }
        
        // Test 3: Very large values
        let large_vector: Vec<f32> = (0..dimension).map(|i| (i as f32) * 1000.0).collect();
        let quantized = quantizer.quantize(&large_vector);
        let reconstructed = quantizer.reconstruct(&quantized);
        
        // Should handle large values gracefully
        let relative_error = euclidean_distance(&large_vector, &reconstructed) / 
                           euclidean_norm(&large_vector);
        assert!(relative_error < 0.5, "Large value quantization relative error: {}", relative_error);
        
        // Test 4: NaN and infinity handling
        let mut nan_vector = vec![1.0; dimension];
        nan_vector[0] = f32::NAN;
        
        let result = quantizer.quantize(&nan_vector);
        // Should either handle gracefully or return error
        assert!(result.is_empty() || !result.iter().any(|&x| x > codebook_size as u8));
        
        let mut inf_vector = vec![1.0; dimension];
        inf_vector[0] = f32::INFINITY;
        
        let result = quantizer.quantize(&inf_vector);
        assert!(result.is_empty() || !result.iter().any(|&x| x > codebook_size as u8));
    }
    
    #[test]
    fn test_quantization_memory_efficiency() {
        let dimensions = vec![32, 64, 128, 256];
        let codebook_size = 256;
        let vector_count = 1000;
        
        for &dim in &dimensions {
            let mut quantizer = ProductQuantizer::new(dim, codebook_size);
            
            // Generate training data
            let mut rng = DeterministicRng::new(PQ_MEMORY_SEED + dim as u64);
            let training_data = generate_random_vectors(&mut rng, vector_count, dim);
            
            quantizer.train(&training_data).unwrap();
            
            // Calculate memory usage
            let quantizer_memory = quantizer.memory_usage();
            let vector_memory = vector_count * quantizer.compressed_vector_size();
            let total_compressed_memory = quantizer_memory + vector_memory;
            
            let original_memory = vector_count * dim * std::mem::size_of::<f32>();
            let compression_ratio = original_memory as f64 / total_compressed_memory as f64;
            
            println!("Dimension {}: Compression ratio {:.2}x", dim, compression_ratio);
            
            // Should achieve significant compression
            assert!(compression_ratio >= 10.0,
                   "Insufficient compression for dimension {}: {:.2}x", dim, compression_ratio);
            
            // Memory usage should scale predictably
            let expected_quantizer_memory = calculate_expected_quantizer_memory(dim, codebook_size);
            let memory_ratio = quantizer_memory as f64 / expected_quantizer_memory as f64;
            
            assert!(memory_ratio > 0.8 && memory_ratio < 1.2,
                   "Quantizer memory usage unexpected: {} vs {} (ratio: {:.2})", 
                   quantizer_memory, expected_quantizer_memory, memory_ratio);
        }
    }
}
```

#### 3.2 SIMD Operations Tests
**File**: `tests/unit/embedding/simd_tests.rs`

```rust
mod simd_tests {
    use super::*;
    use crate::test_infrastructure::*;
    
    #[test]
    fn test_simd_distance_computation() {
        let dimensions = vec![32, 64, 128, 256, 512, 1024];
        
        for &dim in &dimensions {
            let mut rng = DeterministicRng::new(SIMD_TEST_SEED + dim as u64);
            
            // Generate test vectors
            let vector1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let vector2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            
            // Compute distance using SIMD and scalar implementations
            let simd_distance = simd_euclidean_distance(&vector1, &vector2);
            let scalar_distance = scalar_euclidean_distance(&vector1, &vector2);
            
            // Results should be nearly identical
            let difference = (simd_distance - scalar_distance).abs();
            assert!(difference < 1e-5, 
                   "SIMD vs scalar distance mismatch for dim {}: {} vs {} (diff: {})", 
                   dim, simd_distance, scalar_distance, difference);
            
            // Test performance improvement
            let simd_time = measure_distance_computation_time(&vector1, &vector2, true);
            let scalar_time = measure_distance_computation_time(&vector1, &vector2, false);
            
            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            println!("SIMD speedup for dimension {}: {:.2}x", dim, speedup);
            
            // Should see significant speedup for larger dimensions
            if dim >= 128 {
                assert!(speedup > 2.0, "Insufficient SIMD speedup for dimension {}: {:.2}x", dim, speedup);
            }
        }
    }
    
    #[test]
    fn test_simd_batch_operations() {
        let dimension = 128;
        let batch_size = 1000;
        let query_count = 100;
        
        let mut rng = DeterministicRng::new(SIMD_BATCH_SEED);
        
        // Generate database vectors
        let database: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        // Generate query vectors
        let queries: Vec<Vec<f32>> = (0..query_count)
            .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        
        // Test batch distance computation
        for query in &queries {
            let simd_distances = simd_batch_distances(query, &database);
            let scalar_distances: Vec<f32> = database.iter()
                .map(|db_vec| scalar_euclidean_distance(query, db_vec))
                .collect();
            
            assert_eq!(simd_distances.len(), scalar_distances.len());
            
            for (i, (&simd_dist, &scalar_dist)) in simd_distances.iter().zip(scalar_distances.iter()).enumerate() {
                let difference = (simd_dist - scalar_dist).abs();
                assert!(difference < 1e-4, 
                       "Batch distance mismatch at index {}: {} vs {}", 
                       i, simd_dist, scalar_dist);
            }
        }
        
        // Test performance
        let simd_time = measure_batch_distance_time(&queries[0], &database, true);
        let scalar_time = measure_batch_distance_time(&queries[0], &database, false);
        
        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        assert!(speedup > 3.0, "Insufficient batch SIMD speedup: {:.2}x", speedup);
    }
    
    #[test]
    fn test_simd_alignment_requirements() {
        // Test various alignment scenarios
        let dimension = 256;
        let mut rng = DeterministicRng::new(SIMD_ALIGNMENT_SEED);
        
        // Test aligned vectors
        let aligned_vec1 = create_aligned_vector(dimension, &mut rng);
        let aligned_vec2 = create_aligned_vector(dimension, &mut rng);
        
        let aligned_distance = simd_euclidean_distance(&aligned_vec1, &aligned_vec2);
        
        // Test unaligned vectors (should still work correctly)
        let unaligned_vec1 = create_unaligned_vector(dimension, &mut rng);
        let unaligned_vec2 = create_unaligned_vector(dimension, &mut rng);
        
        let unaligned_distance = simd_euclidean_distance(&unaligned_vec1, &unaligned_vec2);
        
        // Results should be consistent regardless of alignment
        let scalar_distance1 = scalar_euclidean_distance(&aligned_vec1, &aligned_vec2);
        let scalar_distance2 = scalar_euclidean_distance(&unaligned_vec1, &unaligned_vec2);
        
        assert!((aligned_distance - scalar_distance1).abs() < 1e-5);
        assert!((unaligned_distance - scalar_distance2).abs() < 1e-5);
        
        // Test mixed alignment
        let mixed_distance = simd_euclidean_distance(&aligned_vec1, &unaligned_vec2);
        let scalar_mixed = scalar_euclidean_distance(&aligned_vec1, &unaligned_vec2);
        
        assert!((mixed_distance - scalar_mixed).abs() < 1e-5);
    }
    
    #[test]
    fn test_simd_vector_operations() {
        let dimension = 128;
        let mut rng = DeterministicRng::new(SIMD_VECTOR_OPS_SEED);
        
        let vec1: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let vec2: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let scalar = 2.5f32;
        
        // Test vector addition
        let simd_add = simd_vector_add(&vec1, &vec2);
        let scalar_add = scalar_vector_add(&vec1, &vec2);
        verify_vector_equality(&simd_add, &scalar_add, 1e-6);
        
        // Test vector subtraction
        let simd_sub = simd_vector_subtract(&vec1, &vec2);
        let scalar_sub = scalar_vector_subtract(&vec1, &vec2);
        verify_vector_equality(&simd_sub, &scalar_sub, 1e-6);
        
        // Test scalar multiplication
        let simd_mul = simd_scalar_multiply(&vec1, scalar);
        let scalar_mul = scalar_scalar_multiply(&vec1, scalar);
        verify_vector_equality(&simd_mul, &scalar_mul, 1e-6);
        
        // Test dot product
        let simd_dot = simd_dot_product(&vec1, &vec2);
        let scalar_dot = scalar_dot_product(&vec1, &vec2);
        assert!((simd_dot - scalar_dot).abs() < 1e-5, 
               "Dot product mismatch: {} vs {}", simd_dot, scalar_dot);
        
        // Test vector normalization
        let simd_norm = simd_normalize(&vec1);
        let scalar_norm = scalar_normalize(&vec1);
        verify_vector_equality(&simd_norm, &scalar_norm, 1e-6);
        
        // Verify normalized vector has unit length
        let norm_length = simd_dot_product(&simd_norm, &simd_norm).sqrt();
        assert!((norm_length - 1.0).abs() < 1e-6, "Normalized vector length: {}", norm_length);
    }
}
```

### 4. Query Engine Testing

#### 4.1 Graph RAG Tests  
**File**: `tests/unit/query/rag_tests.rs`

```rust
mod rag_tests {
    use super::*;
    use crate::test_infrastructure::*;
    
    #[test]
    fn test_graph_rag_context_assembly() {
        // Create test graph with known structure
        let (graph, embeddings) = create_academic_paper_graph(100, 200);
        let rag_engine = GraphRagEngine::new(graph, embeddings);
        
        // Test query with known expected context
        let query_entity = EntityKey::from_hash("paper_central");
        let rag_params = RagParameters {
            max_context_entities: 10,
            max_graph_depth: 2,
            similarity_threshold: 0.7,
            diversity_factor: 0.3,
        };
        
        let context = rag_engine.assemble_context(query_entity, &rag_params);
        
        // Verify context properties
        assert!(context.entities.len() <= rag_params.max_context_entities);
        assert!(!context.entities.is_empty());
        assert!(context.entities.contains(&query_entity));
        
        // Verify all entities are within graph distance limit
        for &entity in &context.entities {
            if entity != query_entity {
                let distance = graph.shortest_path_length(query_entity, entity);
                assert!(distance.is_some(), "Context entity not reachable: {:?}", entity);
                assert!(distance.unwrap() <= rag_params.max_graph_depth,
                       "Context entity too far: distance {}", distance.unwrap());
            }
        }
        
        // Verify relevance scores are reasonable
        assert_eq!(context.relevance_scores.len(), context.entities.len());
        for &score in &context.relevance_scores {
            assert!(score >= 0.0 && score <= 1.0, "Invalid relevance score: {}", score);
        }
        
        // Verify diversity in context
        let diversity_score = calculate_context_diversity(&context, &embeddings);
        assert!(diversity_score >= rag_params.diversity_factor * 0.8,
               "Insufficient context diversity: {}", diversity_score);
    }
    
    #[test]
    fn test_rag_similarity_search_integration() {
        let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(500, 1000);
        let rag_engine = GraphRagEngine::new(graph, embeddings);
        
        // Test with embedding-based query
        let query_embedding = vec![0.1, 0.2, 0.3, 0.4]; // Known test embedding
        let similarity_results = rag_engine.similarity_search(&query_embedding, 20);
        
        assert_eq!(similarity_results.len(), 20);
        
        // Verify results are sorted by similarity (descending)
        for window in similarity_results.windows(2) {
            assert!(window[0].similarity >= window[1].similarity,
                   "Similarity results not properly sorted");
        }
        
        // Verify all similarities are valid
        for result in &similarity_results {
            assert!(result.similarity >= 0.0 && result.similarity <= 1.0,
                   "Invalid similarity score: {}", result.similarity);
        }
        
        // Test integration with graph context expansion
        let top_similar_entity = similarity_results[0].entity;
        let context = rag_engine.expand_from_similarity_seed(
            top_similar_entity, 
            &RagParameters::default()
        );
        
        // Context should include the similar entity
        assert!(context.entities.contains(&top_similar_entity));
        
        // Context should be expanded beyond just similar entities
        assert!(context.entities.len() > similarity_results.len() / 2);
    }
    
    #[test]
    fn test_rag_context_quality_metrics() {
        let (graph, embeddings) = create_test_knowledge_graph_with_clusters(200, 400, 5);
        let rag_engine = GraphRagEngine::new(graph, embeddings);
        
        // Query from a known cluster center
        let cluster_center = EntityKey::from_hash("cluster_0_center");
        let context = rag_engine.assemble_context(cluster_center, &RagParameters::default());
        
        // Calculate quality metrics
        let quality_metrics = rag_engine.evaluate_context_quality(&context, cluster_center);
        
        // Test coverage metric
        assert!(quality_metrics.coverage >= 0.0 && quality_metrics.coverage <= 1.0);
        
        // For a cluster center, coverage should be high
        assert!(quality_metrics.coverage >= 0.7,
               "Low coverage for cluster center query: {}", quality_metrics.coverage);
        
        // Test diversity metric
        assert!(quality_metrics.diversity >= 0.0 && quality_metrics.diversity <= 1.0);
        
        // Test relevance metric
        assert!(quality_metrics.relevance >= 0.0 && quality_metrics.relevance <= 1.0);
        
        // For a well-formed cluster, relevance should be high
        assert!(quality_metrics.relevance >= 0.6,
               "Low relevance for cluster query: {}", quality_metrics.relevance);
        
        // Test coherence metric (semantic consistency)
        assert!(quality_metrics.coherence >= 0.0 && quality_metrics.coherence <= 1.0);
        
        // Test novelty metric (information richness)
        assert!(quality_metrics.novelty >= 0.0 && quality_metrics.novelty <= 1.0);
        
        // Verify metric relationships make sense
        // High relevance + high coverage should generally correlate with high coherence
        if quality_metrics.relevance > 0.8 && quality_metrics.coverage > 0.8 {
            assert!(quality_metrics.coherence > 0.5,
                   "High relevance/coverage should yield reasonable coherence");
        }
    }
    
    #[test]
    fn test_rag_multi_strategy_integration() {
        let (graph, embeddings) = create_complex_test_graph(300, 600);
        let rag_engine = GraphRagEngine::new(graph, embeddings);
        
        let query_entity = EntityKey::from_hash("multi_strategy_test");
        
        // Test different retrieval strategies
        let embedding_strategy = RagStrategy::EmbeddingSimilarity {
            similarity_threshold: 0.8,
            max_candidates: 15,
        };
        
        let graph_strategy = RagStrategy::GraphTraversal {
            max_depth: 3,
            relationship_weights: true,
        };
        
        let hybrid_strategy = RagStrategy::Hybrid {
            embedding_weight: 0.6,
            graph_weight: 0.4,
            max_context_size: 12,
        };
        
        // Execute different strategies
        let embedding_context = rag_engine.assemble_context_with_strategy(
            query_entity, &embedding_strategy
        );
        let graph_context = rag_engine.assemble_context_with_strategy(
            query_entity, &graph_strategy  
        );
        let hybrid_context = rag_engine.assemble_context_with_strategy(
            query_entity, &hybrid_strategy
        );
        
        // Verify strategy differences
        assert_ne!(embedding_context.entities, graph_context.entities);
        
        // Hybrid should combine aspects of both
        let embedding_overlap = calculate_set_overlap(&hybrid_context.entities, &embedding_context.entities);
        let graph_overlap = calculate_set_overlap(&hybrid_context.entities, &graph_context.entities);
        
        assert!(embedding_overlap > 0.2, "Hybrid strategy should include embedding results");
        assert!(graph_overlap > 0.2, "Hybrid strategy should include graph results");
        
        // Compare context quality across strategies
        let embedding_quality = rag_engine.evaluate_context_quality(&embedding_context, query_entity);
        let graph_quality = rag_engine.evaluate_context_quality(&graph_context, query_entity);
        let hybrid_quality = rag_engine.evaluate_context_quality(&hybrid_context, query_entity);
        
        // Hybrid should generally perform well across multiple metrics
        assert!(hybrid_quality.overall_score >= 
               (embedding_quality.overall_score * 0.8).max(graph_quality.overall_score * 0.8),
               "Hybrid strategy should be competitive with specialized strategies");
    }
    
    #[test]
    fn test_rag_performance_characteristics() {
        let sizes = vec![100, 500, 1000, 2000];
        
        for &size in &sizes {
            let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(size, size * 2);
            let rag_engine = GraphRagEngine::new(graph, embeddings);
            
            let query_entity = EntityKey::from_hash(format!("perf_test_{}", size));
            
            // Measure context assembly time
            let start_time = std::time::Instant::now();
            let context = rag_engine.assemble_context(query_entity, &RagParameters::default());
            let assembly_time = start_time.elapsed();
            
            println!("RAG context assembly for {} entities: {:?}", size, assembly_time);
            
            // Should scale sub-linearly with graph size
            let time_per_entity = assembly_time.as_nanos() as f64 / size as f64;
            
            // For size = 100, establish baseline
            if size == 100 {
                assert!(assembly_time < Duration::from_millis(50),
                       "RAG assembly too slow for small graph: {:?}", assembly_time);
            }
            
            // For larger sizes, should not grow linearly
            if size >= 1000 {
                assert!(assembly_time < Duration::from_millis(500),
                       "RAG assembly too slow for graph size {}: {:?}", size, assembly_time);
            }
            
            // Verify context quality doesn't degrade with size
            let quality = rag_engine.evaluate_context_quality(&context, query_entity);
            assert!(quality.overall_score >= 0.5,
                   "Context quality degraded for size {}: {}", size, quality.overall_score);
            
            // Memory usage should be reasonable
            let memory_usage = rag_engine.memory_usage();
            let memory_per_entity = memory_usage / size as u64;
            
            assert!(memory_per_entity < 1000, // < 1KB per entity
                   "Memory usage too high: {} bytes per entity", memory_per_entity);
        }
    }
}
```

## Implementation Strategy

### Week 1: Core Component Tests
**Days 1-2**: Entity and graph structure unit tests
**Days 3-4**: Storage layer tests (CSR, bloom filters, indexing)
**Days 5**: Memory management and core utilities tests

### Week 2: Advanced Component Tests  
**Days 6-7**: Embedding system tests (quantization, SIMD)
**Days 8-9**: Query engine tests (RAG, optimization, clustering)
**Days 10**: Federation, MCP, and WASM tests

## Test Coverage Strategy

### Automated Coverage Analysis
```rust
// Coverage configuration in Cargo.toml
[tool.coverage]
target_coverage = 100
exclude_patterns = [
    "tests/*",
    "examples/*", 
    "benches/*"
]

fail_under = 95  // Fail CI if coverage drops below 95%
```

### Coverage Validation
- **Line Coverage**: Every executable line must be tested
- **Branch Coverage**: Every conditional branch must be exercised  
- **Path Coverage**: All execution paths through complex functions
- **Boundary Coverage**: All edge cases and error conditions

## Success Criteria

### Functional Requirements
- ✅ 100% line coverage across all modules
- ✅ All unit tests pass deterministically 
- ✅ Complete isolation between test cases
- ✅ All edge cases and error conditions tested

### Performance Requirements
- ✅ Unit test suite completes in <10 minutes
- ✅ Individual tests complete in <1 second  
- ✅ Memory usage during testing stays bounded
- ✅ No memory leaks detected in any test

### Quality Requirements
- ✅ All tests are self-validating with clear assertions
- ✅ Test failure messages are diagnostic and actionable
- ✅ Tests are maintainable and well-documented
- ✅ New features automatically get corresponding unit tests

This comprehensive unit testing framework ensures every component of LLMKG is thoroughly validated in isolation, providing a solid foundation for the integration and end-to-end testing phases that follow.