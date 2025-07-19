use std::time::Duration;

/// Benchmark result for zero-copy vs standard access performance
#[derive(Debug)]
pub struct BenchmarkResult {
    pub zero_copy_time: Duration,
    pub standard_time: Duration,
    pub zero_copy_time_ns: u64,
    pub standard_time_ns: u64,
    pub iterations: usize,
    pub speedup: f64,
}

impl BenchmarkResult {
    pub fn zero_copy_ops_per_sec(&self) -> f64 {
        self.iterations as f64 / self.zero_copy_time.as_secs_f64()
    }
    
    pub fn standard_ops_per_sec(&self) -> f64 {
        self.iterations as f64 / self.standard_time.as_secs_f64()
    }
}

/// Safe entity information extracted from zero-copy storage
#[derive(Debug, Clone)]
pub struct ZeroCopyEntityInfo {
    pub id: u32,
    pub type_id: u16,
    pub degree: u16,
    pub properties: String,
    pub embedding: Vec<f32>,
}

impl ZeroCopyEntityInfo {
    pub fn id(&self) -> u32 {
        self.id
    }
    
    pub fn type_id(&self) -> u16 {
        self.type_id
    }
    
    pub fn properties(&self) -> &str {
        &self.properties
    }
}

/// Search result with zero-copy entity reference
#[derive(Debug, Clone)]
pub struct ZeroCopySearchResult {
    pub entity_id: u32,
    pub similarity: f32,
    pub entity_info: ZeroCopyEntityInfo,
}

impl PartialEq for ZeroCopySearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for ZeroCopySearchResult {}

impl PartialOrd for ZeroCopySearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.similarity.partial_cmp(&other.similarity)
    }
}

impl Ord for ZeroCopySearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.similarity.partial_cmp(&other.similarity).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Benchmark result for zero-copy performance
#[derive(Debug, Clone)]
pub struct ZeroCopyBenchmark {
    pub zero_copy_ms: f64,
    pub standard_ms: f64,
    pub speedup: f64,
}

impl ZeroCopyBenchmark {
    pub fn zero_copy_ops_per_sec(&self) -> f64 {
        1000.0 / self.zero_copy_ms
    }
    
    pub fn standard_ops_per_sec(&self) -> f64 {
        1000.0 / self.standard_ms
    }
}