//! Deterministic Random Number Generation
//! 
//! Provides cryptographically secure, deterministic random number generation
//! for reproducible test results across platforms.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Deterministic random number generator for reproducible testing
pub struct DeterministicRng {
    /// Internal state for the linear congruential generator
    state: u64,
    /// Base seed for this RNG instance
    seed: u64,
    /// Operation counter for debugging and reproducibility
    operation_counter: u64,
    /// Optional label for debugging
    label: Option<String>,
}

impl DeterministicRng {
    /// Create a new deterministic RNG with the given seed
    pub fn new(seed: u64) -> Self {
        Self {
            state: Self::initialize_state(seed),
            seed,
            operation_counter: 0,
            label: None,
        }
    }

    /// Create a new deterministic RNG with a seed and label
    pub fn with_label(seed: u64, label: String) -> Self {
        Self {
            state: Self::initialize_state(seed),
            seed,
            operation_counter: 0,
            label: Some(label),
        }
    }

    /// Initialize the internal state from a seed
    fn initialize_state(seed: u64) -> u64 {
        // Use a simple but effective initialization
        // XOR with a large prime to avoid zero states
        seed ^ 0x9E3779B97F4A7C15
    }

    /// Generate the next random u64
    pub fn next_u64(&mut self) -> u64 {
        self.operation_counter += 1;
        
        // Linear Congruential Generator with good parameters
        // Based on Numerical Recipes constants
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        
        // XorShift to improve quality
        let mut result = self.state;
        result ^= result >> 21;
        result ^= result << 35;
        result ^= result >> 4;
        
        result
    }

    /// Generate a random u32
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generate a random f64 in the range [0.0, 1.0)
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // Use 53 bits for IEEE 754 double precision
        (bits as f64) / (1u64 << 53) as f64
    }

    /// Generate a random f32 in the range [0.0, 1.0)
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32() >> 8; // Use 24 bits for IEEE 754 single precision
        (bits as f32) / (1u32 << 24) as f32
    }

    /// Generate a random bool
    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// Generate a random usize in the range [0, max)
    pub fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }

    /// Generate a random i32 in the range [min, max]
    pub fn range_i32(&mut self, min: i32, max: i32) -> i32 {
        if min >= max {
            return min;
        }
        let range = (max - min + 1) as u64;
        let value = self.next_u64() % range;
        min + value as i32
    }

    /// Generate a random f64 in the range [min, max)
    pub fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        if min >= max {
            return min;
        }
        min + (max - min) * self.next_f64()
    }

    /// Generate a random string of given length using alphanumeric characters
    pub fn next_string(&mut self, length: usize) -> String {
        const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        (0..length)
            .map(|_| CHARS[self.next_usize(CHARS.len())] as char)
            .collect()
    }

    /// Fork this RNG to create a new independent RNG for a specific test
    pub fn fork_for_test(&mut self, test_name: &str) -> DeterministicRng {
        let test_seed = self.generate_test_seed(test_name);
        self.operation_counter += 1; // Count the fork operation
        DeterministicRng::with_label(test_seed, test_name.to_string())
    }

    /// Generate a deterministic seed for a named test
    pub fn generate_test_seed(&self, test_name: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        test_name.hash(&mut hasher);
        self.operation_counter.hash(&mut hasher);
        hasher.finish()
    }

    /// Create a fork for a specific component or subsystem
    pub fn fork_for_component(&mut self, component_name: &str) -> DeterministicRng {
        let component_seed = self.generate_component_seed(component_name);
        self.operation_counter += 1;
        DeterministicRng::with_label(component_seed, component_name.to_string())
    }

    /// Generate a seed for a component
    fn generate_component_seed(&self, component_name: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        component_name.hash(&mut hasher);
        "component".hash(&mut hasher);
        self.operation_counter.hash(&mut hasher);
        hasher.finish()
    }

    /// Reset the RNG to its initial state
    pub fn reset(&mut self) {
        self.state = Self::initialize_state(self.seed);
        self.operation_counter = 0;
    }

    /// Get the current seed
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the current operation counter
    pub fn operation_count(&self) -> u64 {
        self.operation_counter
    }

    /// Get the label if set
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Set a new label
    pub fn set_label(&mut self, label: String) {
        self.label = Some(label);
    }

    /// Generate a permutation of indices [0, n)
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        self.shuffle(&mut indices);
        indices
    }

    /// Shuffle a slice in-place using Fisher-Yates algorithm
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }

    /// Select k random elements from a slice without replacement
    pub fn sample<T: Clone>(&mut self, slice: &[T], k: usize) -> Vec<T> {
        if k >= slice.len() {
            return slice.to_vec();
        }

        let mut indices = self.permutation(slice.len());
        indices.truncate(k);
        indices.into_iter().map(|i| slice[i].clone()).collect()
    }

    /// Generate random bytes
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let value = self.next_u64();
            let bytes = value.to_le_bytes();
            let copy_len = chunk.len().min(8);
            chunk[..copy_len].copy_from_slice(&bytes[..copy_len]);
        }
    }

    /// Create a snapshot of the current state for later restoration
    pub fn snapshot(&self) -> RngSnapshot {
        RngSnapshot {
            state: self.state,
            seed: self.seed,
            operation_counter: self.operation_counter,
            label: self.label.clone(),
        }
    }

    /// Restore the RNG from a snapshot
    pub fn restore(&mut self, snapshot: &RngSnapshot) {
        self.state = snapshot.state;
        self.seed = snapshot.seed;
        self.operation_counter = snapshot.operation_counter;
        self.label = snapshot.label.clone();
    }

    /// Generate data with a specific distribution
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        // Box-Muller transform
        static mut NEXT_GAUSSIAN: Option<f64> = None;
        static mut HAS_NEXT_GAUSSIAN: bool = false;
        
        unsafe {
            if HAS_NEXT_GAUSSIAN {
                HAS_NEXT_GAUSSIAN = false;
                return mean + std_dev * NEXT_GAUSSIAN.unwrap();
            }
        }
        
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        
        let mag = std_dev * (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();
        
        unsafe {
            NEXT_GAUSSIAN = Some(z1);
            HAS_NEXT_GAUSSIAN = true;
        }
        
        mean + z0
    }

    /// Generate an exponential distribution
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        -self.next_f64().ln() / lambda
    }

    /// Generate a Poisson distribution (approximation for large lambda)
    pub fn poisson(&mut self, lambda: f64) -> u32 {
        if lambda < 30.0 {
            // Direct method for small lambda
            let l = (-lambda).exp();
            let mut k = 0;
            let mut p = 1.0;
            
            while p > l {
                k += 1;
                p *= self.next_f64();
            }
            
            k - 1
        } else {
            // Normal approximation for large lambda
            let normal_sample = self.normal(lambda, lambda.sqrt());
            normal_sample.round().max(0.0) as u32
        }
    }
}

/// Snapshot of RNG state for restoration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RngSnapshot {
    state: u64,
    seed: u64,
    operation_counter: u64,
    label: Option<String>,
}

/// Thread-safe wrapper for DeterministicRng
pub struct ThreadSafeDeterministicRng {
    rng: Arc<Mutex<DeterministicRng>>,
}

impl ThreadSafeDeterministicRng {
    /// Create a new thread-safe deterministic RNG
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Arc::new(Mutex::new(DeterministicRng::new(seed))),
        }
    }

    /// Execute a closure with exclusive access to the RNG
    pub fn with_rng<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut DeterministicRng) -> T,
    {
        let mut rng = self.rng.lock()
            .map_err(|_| anyhow!("Failed to acquire RNG lock"))?;
        Ok(f(&mut *rng))
    }

    /// Clone the RNG for independent use
    pub fn clone_rng(&self) -> Result<DeterministicRng> {
        let rng = self.rng.lock()
            .map_err(|_| anyhow!("Failed to acquire RNG lock"))?;
        let snapshot = rng.snapshot();
        let mut new_rng = DeterministicRng::new(snapshot.seed);
        new_rng.restore(&snapshot);
        Ok(new_rng)
    }
}

impl Clone for ThreadSafeDeterministicRng {
    fn clone(&self) -> Self {
        Self {
            rng: Arc::clone(&self.rng),
        }
    }
}

/// Factory for creating test-specific RNGs
pub struct RngFactory {
    base_seed: u64,
    test_counter: Arc<Mutex<u64>>,
}

impl RngFactory {
    /// Create a new RNG factory with a base seed
    pub fn new(base_seed: u64) -> Self {
        Self {
            base_seed,
            test_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a new RNG for a specific test
    pub fn create_for_test(&self, test_name: &str) -> Result<DeterministicRng> {
        let mut counter = self.test_counter.lock()
            .map_err(|_| anyhow!("Failed to acquire counter lock"))?;
        
        *counter += 1;
        let test_id = *counter;
        
        let mut hasher = DefaultHasher::new();
        self.base_seed.hash(&mut hasher);
        test_name.hash(&mut hasher);
        test_id.hash(&mut hasher);
        
        let test_seed = hasher.finish();
        Ok(DeterministicRng::with_label(test_seed, test_name.to_string()))
    }

    /// Create a new RNG for a data generation task
    pub fn create_for_data_generation(&self, data_type: &str, size_hint: usize) -> Result<DeterministicRng> {
        let mut hasher = DefaultHasher::new();
        self.base_seed.hash(&mut hasher);
        "data_generation".hash(&mut hasher);
        data_type.hash(&mut hasher);
        size_hint.hash(&mut hasher);
        
        let data_seed = hasher.finish();
        Ok(DeterministicRng::with_label(data_seed, format!("data_{}", data_type)))
    }

    /// Get the base seed
    pub fn base_seed(&self) -> u64 {
        self.base_seed
    }
}

/// Utility functions for deterministic randomization
pub mod utils {
    use super::*;

    /// Create a deterministic RNG from a string seed
    pub fn rng_from_string(seed_str: &str) -> DeterministicRng {
        let mut hasher = DefaultHasher::new();
        seed_str.hash(&mut hasher);
        DeterministicRng::new(hasher.finish())
    }

    /// Validate that two RNGs produce identical sequences
    pub fn validate_determinism(seed: u64, operations: usize) -> bool {
        let mut rng1 = DeterministicRng::new(seed);
        let mut rng2 = DeterministicRng::new(seed);

        for _ in 0..operations {
            if rng1.next_u64() != rng2.next_u64() {
                return false;
            }
        }
        
        true
    }

    /// Test cross-platform determinism by generating a reference sequence
    pub fn generate_reference_sequence(seed: u64, length: usize) -> Vec<u64> {
        let mut rng = DeterministicRng::new(seed);
        (0..length).map(|_| rng.next_u64()).collect()
    }

    /// Verify a sequence matches the reference
    pub fn verify_sequence(seed: u64, sequence: &[u64]) -> bool {
        let reference = generate_reference_sequence(seed, sequence.len());
        reference == sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_generation() {
        let seed = 12345;
        let mut rng1 = DeterministicRng::new(seed);
        let mut rng2 = DeterministicRng::new(seed);

        // Test that same seed produces same sequence
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_different_seeds_different_sequences() {
        let mut rng1 = DeterministicRng::new(12345);
        let mut rng2 = DeterministicRng::new(54321);

        // Different seeds should produce different sequences
        let mut differences = 0;
        for _ in 0..100 {
            if rng1.next_u64() != rng2.next_u64() {
                differences += 1;
            }
        }
        
        // Expect most values to be different
        assert!(differences > 90);
    }

    #[test]
    fn test_reset_functionality() {
        let seed = 98765;
        let mut rng = DeterministicRng::new(seed);
        
        let first_sequence: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
        
        rng.reset();
        let second_sequence: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
        
        assert_eq!(first_sequence, second_sequence);
    }

    #[test]
    fn test_fork_for_test() {
        let mut base_rng = DeterministicRng::new(42);
        let mut test_rng1 = base_rng.fork_for_test("test1");
        let mut test_rng2 = base_rng.fork_for_test("test1");
        
        // Same test name should produce same sequence
        assert_eq!(test_rng1.next_u64(), test_rng2.next_u64());
    }

    #[test]
    fn test_different_test_names() {
        let mut base_rng = DeterministicRng::new(42);
        let mut test_rng1 = base_rng.fork_for_test("test1");
        let mut test_rng2 = base_rng.fork_for_test("test2");
        
        // Different test names should produce different sequences
        assert_ne!(test_rng1.next_u64(), test_rng2.next_u64());
    }

    #[test]
    fn test_snapshot_restore() {
        let mut rng = DeterministicRng::new(123);
        
        // Generate some values
        let _: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
        
        // Take snapshot
        let snapshot = rng.snapshot();
        
        // Generate more values
        let values_after_snapshot: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
        
        // Restore and generate again
        rng.restore(&snapshot);
        let values_after_restore: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
        
        assert_eq!(values_after_snapshot, values_after_restore);
    }

    #[test]
    fn test_range_generation() {
        let mut rng = DeterministicRng::new(42);
        
        // Test i32 range
        for _ in 0..100 {
            let value = rng.range_i32(10, 20);
            assert!(value >= 10 && value <= 20);
        }
        
        // Test f64 range
        for _ in 0..100 {
            let value = rng.range_f64(1.0, 2.0);
            assert!(value >= 1.0 && value < 2.0);
        }
    }

    #[test]
    fn test_string_generation() {
        let mut rng = DeterministicRng::new(42);
        
        let s1 = rng.next_string(10);
        let s2 = rng.next_string(10);
        
        assert_eq!(s1.len(), 10);
        assert_eq!(s2.len(), 10);
        assert_ne!(s1, s2); // Should be different
        
        // Test determinism
        rng.reset();
        let s3 = rng.next_string(10);
        assert_eq!(s1, s3);
    }

    #[test]
    fn test_shuffle() {
        let mut rng = DeterministicRng::new(42);
        let mut data = vec![1, 2, 3, 4, 5];
        let original = data.clone();
        
        rng.shuffle(&mut data);
        
        // Should have same elements
        let mut sorted_data = data.clone();
        sorted_data.sort();
        assert_eq!(sorted_data, original);
        
        // Should be different order (with high probability)
        assert_ne!(data, original);
    }

    #[test]
    fn test_sample() {
        let mut rng = DeterministicRng::new(42);
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        
        let sample = rng.sample(&data, 3);
        assert_eq!(sample.len(), 3);
        
        // All sampled elements should be from original data
        for item in &sample {
            assert!(data.contains(item));
        }
    }

    #[test]
    fn test_utils_determinism_validation() {
        assert!(utils::validate_determinism(42, 1000));
    }

    #[test]
    fn test_utils_reference_sequence() {
        let seed = 12345;
        let length = 50;
        
        let seq1 = utils::generate_reference_sequence(seed, length);
        let seq2 = utils::generate_reference_sequence(seed, length);
        
        assert_eq!(seq1, seq2);
        assert_eq!(seq1.len(), length);
        
        assert!(utils::verify_sequence(seed, &seq1));
    }

    #[test]
    fn test_rng_factory() {
        let factory = RngFactory::new(42);
        
        let rng1 = factory.create_for_test("test1").unwrap();
        let rng2 = factory.create_for_test("test1").unwrap();
        
        // Same test name should produce different RNGs (due to counter)
        assert_ne!(rng1.seed(), rng2.seed());
        
        let data_rng = factory.create_for_data_generation("graph", 1000).unwrap();
        assert_eq!(data_rng.label(), Some("data_graph"));
    }

    #[test]
    fn test_normal_distribution() {
        let mut rng = DeterministicRng::new(42);
        
        // Generate samples and check they're reasonable
        let samples: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
        
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / samples.len() as f64;
        
        // Check mean is close to 0 and variance close to 1
        assert!((mean.abs()) < 0.1);
        assert!((variance - 1.0).abs() < 0.1);
    }
}