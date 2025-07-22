use std::hash::{Hash, Hasher};
use ahash::AHasher;

pub struct BloomFilter {
    bit_array: Vec<u64>,
    bit_count: usize,
    hash_count: usize,
}

impl BloomFilter {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let bit_count = Self::optimal_bit_count(expected_items, false_positive_rate);
        let hash_count = Self::optimal_hash_count(expected_items, bit_count);
        
        Self {
            bit_array: vec![0u64; (bit_count + 63) / 64],
            bit_count,
            hash_count,
        }
    }
    
    fn optimal_bit_count(items: usize, fp_rate: f64) -> usize {
        let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        ((items as f64) * fp_rate.ln() / -ln2_squared).ceil() as usize
    }
    
    fn optimal_hash_count(items: usize, bits: usize) -> usize {
        let ratio = bits as f64 / items as f64;
        (ratio * std::f64::consts::LN_2).round().max(1.0) as usize
    }
    
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let hashes = self.hash(item);
        
        for i in 0..self.hash_count {
            let bit_index = hashes[i] % self.bit_count;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            self.bit_array[word_index] |= 1u64 << bit_offset;
        }
    }
    
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.hash(item);
        
        for i in 0..self.hash_count {
            let bit_index = hashes[i] % self.bit_count;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            
            if (self.bit_array[word_index] & (1u64 << bit_offset)) == 0 {
                return false;
            }
        }
        
        true
    }
    
    fn hash<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut hashes = Vec::with_capacity(self.hash_count);
        
        // Use double hashing technique
        let mut hasher1 = AHasher::default();
        item.hash(&mut hasher1);
        let h1 = hasher1.finish() as usize;
        
        let mut hasher2 = AHasher::default();
        hasher2.write_u64(h1 as u64);
        let h2 = hasher2.finish() as usize;
        
        for i in 0..self.hash_count {
            hashes.push(h1.wrapping_add(i.wrapping_mul(h2)));
        }
        
        hashes
    }
    
    pub fn clear(&mut self) {
        self.bit_array.fill(0);
    }
    
    pub fn fill_ratio(&self) -> f64 {
        let set_bits: u32 = self.bit_array.iter()
            .map(|&word| word.count_ones())
            .sum();
        set_bits as f64 / self.bit_count as f64
    }
    
    pub fn estimated_items(&self) -> f64 {
        let x = self.fill_ratio();
        if x == 0.0 {
            return 0.0;
        }
        
        let ln_1_minus_x = (1.0 - x).ln();
        -(self.bit_count as f64) * ln_1_minus_x / self.hash_count as f64
    }
    
    pub fn memory_usage(&self) -> usize {
        self.bit_array.capacity() * std::mem::size_of::<u64>()
    }
    
    /// Get the capacity of the bloom filter (in bits)
    pub fn capacity(&self) -> usize {
        self.bit_count
    }
    
    /// Add edge (not applicable - BloomFilter is a probabilistic data structure)
    pub fn add_edge(&mut self, _from: u32, _to: u32, _weight: f32) -> crate::error::Result<()> {
        Err(crate::error::GraphError::UnsupportedOperation(
            "BloomFilter is a probabilistic data structure, not a graph. Use CSRGraph for edges.".to_string()
        ))
    }
    
    /// Update entity (not applicable - BloomFilter only tracks membership)
    pub fn update_entity(&mut self, _id: u32, _data: Vec<u8>) -> crate::error::Result<()> {
        Err(crate::error::GraphError::UnsupportedOperation(
            "BloomFilter only tracks membership, cannot update entities.".to_string()
        ))
    }
    
    /// Remove (not supported - BloomFilter doesn't support deletion)
    pub fn remove(&mut self, _id: u32) -> crate::error::Result<()> {
        Err(crate::error::GraphError::UnsupportedOperation(
            "Standard BloomFilter does not support removal. Use CountingBloomFilter instead.".to_string()
        ))
    }
    
    /// Check if filter contains an entity (wrapper around contains)
    pub fn contains_entity(&self, id: u32) -> bool {
        self.contains(&id)
    }
    
    /// Get encoded size
    pub fn encoded_size(&self) -> usize {
        // Size needed to serialize this bloom filter
        std::mem::size_of::<usize>() * 3 + // bit_count, hash_count, array length
        self.bit_array.len() * std::mem::size_of::<u64>()
    }
}

pub struct CountingBloomFilter {
    counters: Vec<u8>,
    counter_bits: usize,
    counter_mask: u8,
    hash_count: usize,
}

impl CountingBloomFilter {
    pub fn new(expected_items: usize, false_positive_rate: f64, counter_bits: usize) -> Self {
        assert!(counter_bits <= 8 && counter_bits > 0);
        
        let bit_count = BloomFilter::optimal_bit_count(expected_items, false_positive_rate);
        let hash_count = BloomFilter::optimal_hash_count(expected_items, bit_count);
        let counter_mask = (1u8 << counter_bits) - 1;
        
        Self {
            counters: vec![0u8; bit_count],
            counter_bits,
            counter_mask,
            hash_count,
        }
    }
    
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let hashes = self.hash(item);
        
        for i in 0..self.hash_count {
            let index = hashes[i] % self.counters.len();
            if self.counters[index] < self.counter_mask {
                self.counters[index] += 1;
            }
        }
    }
    
    pub fn remove<T: Hash>(&mut self, item: &T) {
        let hashes = self.hash(item);
        
        for i in 0..self.hash_count {
            let index = hashes[i] % self.counters.len();
            if self.counters[index] > 0 {
                self.counters[index] -= 1;
            }
        }
    }
    
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.hash(item);
        
        for i in 0..self.hash_count {
            let index = hashes[i] % self.counters.len();
            if self.counters[index] == 0 {
                return false;
            }
        }
        
        true
    }
    
    fn hash<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut hashes = Vec::with_capacity(self.hash_count);
        
        let mut hasher1 = AHasher::default();
        item.hash(&mut hasher1);
        let h1 = hasher1.finish() as usize;
        
        let mut hasher2 = AHasher::default();
        hasher2.write_u64(h1 as u64);
        let h2 = hasher2.finish() as usize;
        
        for i in 0..self.hash_count {
            hashes.push(h1.wrapping_add(i.wrapping_mul(h2)));
        }
        
        hashes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_optimal_bit_count() {
        // Test basic functionality
        let bit_count = BloomFilter::optimal_bit_count(1000, 0.01);
        assert!(bit_count > 0);
        assert!(bit_count > 1000); // Should be larger than item count for low FP rate
        
        // Test mathematical property: more items should require more bits
        let bit_count_small = BloomFilter::optimal_bit_count(100, 0.01);
        let bit_count_large = BloomFilter::optimal_bit_count(10000, 0.01);
        assert!(bit_count_large > bit_count_small);
        
        // Test mathematical property: lower FP rate should require more bits
        let bit_count_high_fp = BloomFilter::optimal_bit_count(1000, 0.1);
        let bit_count_low_fp = BloomFilter::optimal_bit_count(1000, 0.001);
        assert!(bit_count_low_fp > bit_count_high_fp);
        
        // Edge cases
        assert!(BloomFilter::optimal_bit_count(1, 0.5) > 0);
        assert!(BloomFilter::optimal_bit_count(1000000, 0.001) > 0);
    }

    #[test]
    fn test_optimal_hash_count() {
        // Test basic functionality
        let hash_count = BloomFilter::optimal_hash_count(1000, 10000);
        assert!(hash_count >= 1);
        
        // Test mathematical property: higher bit-to-item ratio should require more hashes
        let hash_count_low_ratio = BloomFilter::optimal_hash_count(1000, 1000);
        let hash_count_high_ratio = BloomFilter::optimal_hash_count(1000, 10000);
        assert!(hash_count_high_ratio >= hash_count_low_ratio);
        
        // Should return at least 1 hash function
        assert!(BloomFilter::optimal_hash_count(10000, 1000) >= 1);
        
        // Edge cases
        assert_eq!(BloomFilter::optimal_hash_count(1000, 1), 1); // Minimum is 1
        assert!(BloomFilter::optimal_hash_count(1, 1000000) > 1);
    }

    #[test]
    fn test_hash_function_consistency() {
        let filter = BloomFilter::new(100, 0.01);
        let test_item = "test_string";
        
        // Hash function should be deterministic
        let hashes1 = filter.hash(&test_item);
        let hashes2 = filter.hash(&test_item);
        assert_eq!(hashes1, hashes2);
        
        // Should generate the correct number of hashes
        assert_eq!(hashes1.len(), filter.hash_count);
        
        // Different inputs should produce different hashes (with high probability)
        let hashes_different = filter.hash(&"different_string");
        assert_ne!(hashes1, hashes_different);
    }

    #[test]
    fn test_hash_function_distribution() {
        let filter = BloomFilter::new(1000, 0.01);
        let mut all_hash_values = HashSet::new();
        
        // Test with many different inputs
        for i in 0..100 {
            let item = format!("test_item_{}", i);
            let hashes = filter.hash(&item);
            
            for hash_value in hashes {
                all_hash_values.insert(hash_value);
            }
        }
        
        // Should produce diverse hash values (basic distribution test)
        assert!(all_hash_values.len() > 50, "Hash function should produce diverse values");
    }

    #[test]
    fn test_hash_function_double_hashing() {
        let filter = BloomFilter::new(100, 0.01);
        let test_item = 42u32;
        let hashes = filter.hash(&test_item);
        
        // For multiple hash functions, values should be different
        if hashes.len() > 1 {
            let unique_hashes: HashSet<_> = hashes.iter().collect();
            assert_eq!(unique_hashes.len(), hashes.len(), "All hash values should be unique");
        }
    }

    #[test]
    fn test_bloom_filter_insert_contains() {
        let mut filter = BloomFilter::new(100, 0.01);
        
        // Initially should not contain anything
        assert!(!filter.contains(&"test"));
        assert!(!filter.contains(&42));
        
        // After insertion, should contain the item
        filter.insert(&"test");
        assert!(filter.contains(&"test"));
        
        filter.insert(&42);
        assert!(filter.contains(&42));
        
        // Should still work with the first item
        assert!(filter.contains(&"test"));
    }

    #[test]
    fn test_false_positive_rate_property() {
        let expected_items = 1000;
        let fp_rate = 0.01;
        let mut filter = BloomFilter::new(expected_items, fp_rate);
        
        // Insert exactly the expected number of items
        let inserted_items: Vec<String> = (0..expected_items).map(|i| format!("item_{}", i)).collect();
        for item in &inserted_items {
            filter.insert(item);
        }
        
        // All inserted items should be found
        for item in &inserted_items {
            assert!(filter.contains(item), "Inserted item should always be found");
        }
        
        // Test false positive rate with non-inserted items
        let test_items: Vec<String> = (expected_items..expected_items + 1000)
            .map(|i| format!("test_item_{}", i))
            .collect();
        
        let false_positives = test_items.iter()
            .filter(|item| filter.contains(item))
            .count();
        
        let actual_fp_rate = false_positives as f64 / test_items.len() as f64;
        
        // Allow some tolerance due to randomness, but should be reasonably close
        assert!(actual_fp_rate <= fp_rate * 3.0, 
            "False positive rate {} should be reasonably close to expected {}", 
            actual_fp_rate, fp_rate);
    }

    #[test]
    fn test_fill_ratio() {
        let mut filter = BloomFilter::new(100, 0.1);
        
        // Initially should be empty
        assert_eq!(filter.fill_ratio(), 0.0);
        
        // After some insertions, should increase
        filter.insert(&1);
        filter.insert(&2);
        filter.insert(&3);
        
        let fill_ratio = filter.fill_ratio();
        assert!(fill_ratio > 0.0);
        assert!(fill_ratio <= 1.0);
    }

    #[test]
    fn test_estimated_items() {
        let mut filter = BloomFilter::new(100, 0.1);
        
        // Initially should estimate 0 items
        assert_eq!(filter.estimated_items(), 0.0);
        
        // Insert some items and check estimation
        for i in 0..10 {
            filter.insert(&i);
        }
        
        let estimated = filter.estimated_items();
        assert!(estimated > 0.0);
        // Should be a reasonable estimate (within an order of magnitude)
        assert!(estimated >= 1.0 && estimated <= 100.0);
    }

    #[test]
    fn test_clear() {
        let mut filter = BloomFilter::new(100, 0.1);
        
        // Insert items
        filter.insert(&"test1");
        filter.insert(&"test2");
        assert!(filter.contains(&"test1"));
        assert!(filter.contains(&"test2"));
        
        // Clear and verify
        filter.clear();
        assert!(!filter.contains(&"test1"));
        assert!(!filter.contains(&"test2"));
        assert_eq!(filter.fill_ratio(), 0.0);
        assert_eq!(filter.estimated_items(), 0.0);
    }

    #[test]
    fn test_memory_usage() {
        let filter = BloomFilter::new(1000, 0.01);
        let memory_usage = filter.memory_usage();
        
        assert!(memory_usage > 0);
        // Should be roughly the size of the bit array
        let expected_min_size = filter.bit_array.len() * std::mem::size_of::<u64>();
        assert!(memory_usage >= expected_min_size);
    }

    #[test]
    fn test_counting_bloom_filter_basic() {
        let mut filter = CountingBloomFilter::new(100, 0.01, 4);
        
        // Initially should not contain anything
        assert!(!filter.contains(&"test"));
        
        // After insertion, should contain the item
        filter.insert(&"test");
        assert!(filter.contains(&"test"));
        
        // Should support removal
        filter.remove(&"test");
        assert!(!filter.contains(&"test"));
    }

    #[test]
    fn test_counting_bloom_filter_multiple_insertions() {
        let mut filter = CountingBloomFilter::new(100, 0.01, 4);
        
        // Insert same item multiple times
        filter.insert(&"test");
        filter.insert(&"test");
        filter.insert(&"test");
        assert!(filter.contains(&"test"));
        
        // Remove once - should still contain
        filter.remove(&"test");
        assert!(filter.contains(&"test"));
        
        // Remove again - should still contain
        filter.remove(&"test");
        assert!(filter.contains(&"test"));
        
        // Remove third time - should no longer contain
        filter.remove(&"test");
        assert!(!filter.contains(&"test"));
    }

    #[test]
    fn test_counting_bloom_filter_counter_overflow() {
        let mut filter = CountingBloomFilter::new(100, 0.1, 2); // 2-bit counters, max value 3
        
        // Insert many times to test overflow handling
        for _ in 0..10 {
            filter.insert(&"test");
        }
        
        // Should still contain the item
        assert!(filter.contains(&"test"));
        
        // Removing should work correctly even after overflow
        for _ in 0..5 {
            filter.remove(&"test");
        }
        
        // Might still contain due to counter saturation
        // This tests that the implementation handles overflow gracefully
    }

    #[test]
    fn test_bit_array_word_boundaries() {
        let mut filter = BloomFilter::new(100, 0.1);
        
        // Insert items that might span u64 word boundaries
        for i in 0..200 {
            filter.insert(&i);
        }
        
        // All items should be findable
        for i in 0..200 {
            assert!(filter.contains(&i), "Item {} should be found", i);
        }
    }

    #[test]
    fn test_different_data_types() {
        let mut filter = BloomFilter::new(100, 0.1);
        
        // Test with different hashable types
        filter.insert(&42u32);
        filter.insert(&"string");
        filter.insert(&vec![1, 2, 3]);
        filter.insert(&(1, 2, 3));
        
        assert!(filter.contains(&42u32));
        assert!(filter.contains(&"string"));
        assert!(filter.contains(&vec![1, 2, 3]));
        assert!(filter.contains(&(1, 2, 3)));
        
        // Different values should not be found
        assert!(!filter.contains(&43u32));
        assert!(!filter.contains(&"different"));
    }

    #[cfg(test)]
    mod property_tests {
        use super::*;
        
        #[test]
        fn test_no_false_negatives() {
            let mut filter = BloomFilter::new(50, 0.1);
            let test_items: Vec<i32> = (0..50).collect();
            
            // Insert all items
            for item in &test_items {
                filter.insert(item);
            }
            
            // All inserted items must be found (no false negatives allowed)
            for item in &test_items {
                assert!(filter.contains(item), "Bloom filter must never have false negatives");
            }
        }
        
        #[test]
        fn test_mathematical_bounds() {
            let items = 1000;
            let fp_rate = 0.01;
            
            let bit_count = BloomFilter::optimal_bit_count(items, fp_rate);
            let hash_count = BloomFilter::optimal_hash_count(items, bit_count);
            
            // Mathematical bounds from Bloom filter theory
            assert!(hash_count as f64 <= (bit_count as f64 / items as f64) * std::f64::consts::LN_2 + 1.0);
            assert!(hash_count >= 1);
            
            // Bit count should follow the mathematical formula approximately
            let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
            let expected_bits = ((items as f64) * fp_rate.ln() / -ln2_squared).ceil() as usize;
            assert_eq!(bit_count, expected_bits);
        }
    }
}