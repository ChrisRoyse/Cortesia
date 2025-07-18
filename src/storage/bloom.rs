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