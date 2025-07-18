//! Bloom Filter Unit Tests
//!
//! Comprehensive tests for bloom filter implementation including
//! false positive rates, serialization, hash function quality,
//! and performance characteristics.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::storage::bloom::BloomFilter;

#[cfg(test)]
mod bloom_tests {
    use super::*;

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

    #[test]
    fn test_bloom_filter_false_positive_rates() {
        let test_cases = vec![
            (1000, 0.001),
            (1000, 0.01),
            (1000, 0.1),
            (10000, 0.001),
            (10000, 0.01),
        ];
        
        for (capacity, target_fp_rate) in test_cases {
            let mut bloom = BloomFilter::new(capacity, target_fp_rate);
            
            // Fill to capacity
            for i in 0..capacity {
                bloom.insert(&format!("item_{}", i));
            }
            
            // Test false positive rate
            let fp_count = count_false_positives(&bloom, &[], 100000);
            let actual_fp_rate = fp_count as f64 / 100000.0;
            
            println!("Capacity: {}, Target FP: {:.3}, Actual FP: {:.3}", 
                    capacity, target_fp_rate, actual_fp_rate);
            
            // Should be within reasonable bounds
            assert!(actual_fp_rate <= target_fp_rate * 2.0,
                   "False positive rate too high: {:.3} vs target {:.3}",
                   actual_fp_rate, target_fp_rate);
            
            // Verify no false negatives
            for i in 0..capacity {
                let item = format!("item_{}", i);
                assert!(bloom.contains(&item), "False negative for: {}", item);
            }
        }
    }

    #[test]
    fn test_bloom_filter_optimal_parameters() {
        let test_cases = vec![
            (1000, 0.01),
            (10000, 0.01),
            (100000, 0.001),
        ];
        
        for (expected_items, fp_rate) in test_cases {
            let bloom = BloomFilter::new(expected_items, fp_rate);
            
            let optimal_m = bloom.optimal_bit_array_size(expected_items, fp_rate);
            let optimal_k = bloom.optimal_hash_functions(expected_items, optimal_m);
            
            // Verify parameters are in reasonable ranges
            assert!(optimal_m > expected_items, "Bit array should be larger than item count");
            assert!(optimal_k >= 1 && optimal_k <= 20, "Hash function count should be reasonable: {}", optimal_k);
            
            // Verify actual parameters match or are close to optimal
            let actual_m = bloom.bit_array_size();
            let actual_k = bloom.hash_function_count();
            
            let m_ratio = actual_m as f64 / optimal_m as f64;
            assert!(m_ratio > 0.8 && m_ratio < 1.2,
                   "Bit array size not optimal: {} vs {} (ratio: {:.2})",
                   actual_m, optimal_m, m_ratio);
            
            assert!(actual_k == optimal_k || (actual_k as i32 - optimal_k as i32).abs() <= 1,
                   "Hash function count not optimal: {} vs {}", actual_k, optimal_k);
        }
    }

    #[test]
    fn test_bloom_filter_performance() {
        let item_count = 100000;
        let fp_rate = 0.01;
        let mut bloom = BloomFilter::new(item_count, fp_rate);
        
        // Test insertion performance
        let items: Vec<String> = (0..item_count).map(|i| format!("perf_item_{}", i)).collect();
        
        let (_, insertion_time) = measure_execution_time(|| {
            for item in &items {
                bloom.insert(item);
            }
        });
        
        println!("Bloom insertion time for {} items: {:?}", item_count, insertion_time);
        let insertions_per_second = item_count as f64 / insertion_time.as_secs_f64();
        assert!(insertions_per_second > 100000.0, "Insertion too slow: {:.0} ops/sec", insertions_per_second);
        
        // Test lookup performance
        let (_, lookup_time) = measure_execution_time(|| {
            for item in &items {
                let _ = bloom.contains(item);
            }
        });
        
        println!("Bloom lookup time for {} items: {:?}", item_count, lookup_time);
        let lookups_per_second = item_count as f64 / lookup_time.as_secs_f64();
        assert!(lookups_per_second > 200000.0, "Lookup too slow: {:.0} ops/sec", lookups_per_second);
        
        // Test batch operations
        let batch_size = 10000;
        let batch_items: Vec<&str> = items.iter().take(batch_size).map(|s| s.as_str()).collect();
        
        let (_, batch_lookup_time) = measure_execution_time(|| {
            let _results = bloom.contains_batch(&batch_items);
        });
        
        println!("Bloom batch lookup time for {} items: {:?}", batch_size, batch_lookup_time);
        let batch_lookups_per_second = batch_size as f64 / batch_lookup_time.as_secs_f64();
        assert!(batch_lookups_per_second > 500000.0, "Batch lookup too slow: {:.0} ops/sec", batch_lookups_per_second);
    }

    #[test]
    fn test_bloom_filter_concurrent_access() {
        use std::sync::{Arc, RwLock};
        use std::thread;
        
        let bloom = Arc::new(RwLock::new(BloomFilter::new(10000, 0.01)));
        let thread_count = 4;
        let items_per_thread = 1000;
        
        // Insert phase - multiple writers
        let mut insert_handles = Vec::new();
        
        for thread_id in 0..thread_count {
            let bloom_clone = Arc::clone(&bloom);
            
            let handle = thread::spawn(move || {
                for i in 0..items_per_thread {
                    let item = format!("thread_{}_item_{}", thread_id, i);
                    let mut bloom = bloom_clone.write().unwrap();
                    bloom.insert(&item);
                }
            });
            
            insert_handles.push(handle);
        }
        
        // Wait for all insertions
        for handle in insert_handles {
            handle.join().unwrap();
        }
        
        // Read phase - multiple readers
        let mut read_handles = Vec::new();
        
        for thread_id in 0..thread_count {
            let bloom_clone = Arc::clone(&bloom);
            
            let handle = thread::spawn(move || {
                let bloom = bloom_clone.read().unwrap();
                let mut found_count = 0;
                
                for i in 0..items_per_thread {
                    let item = format!("thread_{}_item_{}", thread_id, i);
                    if bloom.contains(&item) {
                        found_count += 1;
                    }
                }
                
                found_count
            });
            
            read_handles.push(handle);
        }
        
        // Verify all reads found their items
        for handle in read_handles {
            let found_count = handle.join().unwrap();
            assert_eq!(found_count, items_per_thread, "Thread didn't find all its items");
        }
        
        // Verify final state
        let bloom = bloom.read().unwrap();
        assert_eq!(bloom.len(), thread_count * items_per_thread);
    }

    #[test]
    fn test_bloom_filter_edge_cases() {
        // Test very small capacity
        let mut small_bloom = BloomFilter::new(1, 0.1);
        small_bloom.insert("single_item");
        assert!(small_bloom.contains("single_item"));
        
        // Test very high false positive rate
        let mut high_fp_bloom = BloomFilter::new(100, 0.9);
        high_fp_bloom.insert("test");
        assert!(high_fp_bloom.contains("test"));
        
        // Test very low false positive rate
        let low_fp_bloom = BloomFilter::new(100, 0.0001);
        assert!(low_fp_bloom.bit_array_size() > 1000); // Should use many bits
        
        // Test empty string
        let mut empty_bloom = BloomFilter::new(100, 0.01);
        empty_bloom.insert("");
        assert!(empty_bloom.contains(""));
        
        // Test very long string
        let long_string = "x".repeat(10000);
        let mut long_bloom = BloomFilter::new(100, 0.01);
        long_bloom.insert(&long_string);
        assert!(long_bloom.contains(&long_string));
        
        // Test unicode strings
        let unicode_items = vec!["🦀", "测试", "🌟", "Ελληνικά", "العربية"];
        let mut unicode_bloom = BloomFilter::new(100, 0.01);
        
        for item in &unicode_items {
            unicode_bloom.insert(item);
        }
        
        for item in &unicode_items {
            assert!(unicode_bloom.contains(item), "Unicode item not found: {}", item);
        }
    }

    #[test]
    fn test_bloom_filter_memory_efficiency() {
        let sizes = vec![1000, 10000, 100000];
        let fp_rate = 0.01;
        
        for &size in &sizes {
            let bloom = BloomFilter::new(size, fp_rate);
            let memory_usage = bloom.memory_usage();
            let memory_per_item = memory_usage as f64 / size as f64;
            
            println!("Size: {}, Memory per item: {:.2} bytes", size, memory_per_item);
            
            // Should be reasonable memory usage
            assert!(memory_per_item < 20.0, "Memory per item too high: {:.2} bytes", memory_per_item);
            
            // Verify against theoretical minimum
            let theoretical_bits = -(size as f64 * fp_rate.ln()) / (2.0_f64.ln().powi(2));
            let theoretical_bytes = theoretical_bits / 8.0;
            let efficiency = theoretical_bytes / memory_per_item;
            
            assert!(efficiency > 0.5, "Memory efficiency too low: {:.2}", efficiency);
            
            // Test memory growth is linear
            if size >= 10000 {
                let smaller_bloom = BloomFilter::new(size / 10, fp_rate);
                let smaller_memory = smaller_bloom.memory_usage() as f64;
                let growth_ratio = memory_usage as f64 / smaller_memory;
                
                // Should be close to 10x (linear growth)
                assert!(growth_ratio > 8.0 && growth_ratio < 12.0,
                       "Non-linear memory growth: {:.2}x", growth_ratio);
            }
        }
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

fn calculate_expected_bloom_memory(expected_items: usize, fp_rate: f64) -> usize {
    // Calculate optimal bit array size
    let m = (-((expected_items as f64) * fp_rate.ln()) / (2.0_f64.ln().powi(2))).ceil() as usize;
    
    // Add overhead for structure metadata
    let bit_array_bytes = (m + 7) / 8; // Round up to nearest byte
    let metadata_overhead = 64; // Estimated overhead
    
    bit_array_bytes + metadata_overhead
}

fn test_hash_avalanche_effect(bloom: &BloomFilter) {
    let test_strings = vec![
        ("test", "test1"),
        ("hello", "hallo"),
        ("data", "data1"),
        ("abcd", "abce"),
    ];
    
    for (str1, str2) in test_strings {
        let hashes1 = bloom.compute_hashes(str1);
        let hashes2 = bloom.compute_hashes(str2);
        
        // Count different hash values
        let different_hashes = hashes1.iter()
            .zip(hashes2.iter())
            .filter(|(h1, h2)| h1 != h2)
            .count();
        
        // Should have significant differences for small input changes
        let difference_ratio = different_hashes as f64 / hashes1.len() as f64;
        assert!(difference_ratio > 0.5,
               "Poor avalanche effect for '{}' vs '{}': {:.2}% different",
               str1, str2, difference_ratio * 100.0);
    }
}