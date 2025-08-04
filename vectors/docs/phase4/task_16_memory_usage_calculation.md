# Task 16: Implement Memory Usage Calculation

## Context
You are implementing Phase 4 of a vector indexing system. The cache put method was implemented in the previous task with basic memory estimation. Now you need to implement accurate memory usage calculation and tracking for different types of search results.

## Current State
- `src/cache.rs` exists with `MemoryEfficientCache` struct
- `put()` and `get()` methods implemented with basic size estimation
- Basic eviction policies are in place
- Memory tracking exists but needs refinement

## Task Objective
Implement accurate memory usage calculation for cache entries, including proper sizing for different content types, metadata overhead, and system memory alignment.

## Implementation Requirements

### 1. Enhance CacheEntry size estimation
Replace the basic `estimate_size` method with a more accurate implementation:
```rust
impl CacheEntry {
    fn estimate_size(results: &[SearchResult]) -> usize {
        let mut total_size = 0;
        
        // Base struct overhead
        total_size += std::mem::size_of::<CacheEntry>();
        total_size += std::mem::size_of::<Vec<SearchResult>>();
        
        // SearchResult overhead and content
        for result in results {
            // Base SearchResult struct size
            total_size += std::mem::size_of::<SearchResult>();
            
            // String data (heap allocated)
            total_size += result.file_path.capacity();
            total_size += result.content.capacity();
            
            // Add padding for memory alignment (8-byte alignment typical)
            total_size = Self::align_to_8_bytes(total_size);
        }
        
        // Add HashMap overhead for the entry itself
        total_size += std::mem::size_of::<String>(); // Key overhead
        total_size += 32; // HashMap entry overhead estimate
        
        // Add 10% overhead for fragmentation and other allocator overhead
        total_size + (total_size / 10)
    }
    
    fn align_to_8_bytes(size: usize) -> usize {
        (size + 7) & !7
    }
    
    // Method to recalculate size after access (in case content changed)
    fn recalculate_size(&mut self) {
        self.estimated_size = Self::estimate_size(&self.results);
    }
    
    // Get detailed size breakdown for debugging
    fn size_breakdown(&self) -> MemorySizeBreakdown {
        let mut breakdown = MemorySizeBreakdown::new();
        
        breakdown.struct_overhead = std::mem::size_of::<CacheEntry>() + 
                                   std::mem::size_of::<Vec<SearchResult>>();
        
        for result in &self.results {
            breakdown.result_structs += std::mem::size_of::<SearchResult>();
            breakdown.file_paths += result.file_path.capacity();
            breakdown.content_data += result.content.capacity();
        }
        
        breakdown.alignment_padding = Self::align_to_8_bytes(
            breakdown.struct_overhead + breakdown.result_structs + 
            breakdown.file_paths + breakdown.content_data
        ) - (breakdown.struct_overhead + breakdown.result_structs + 
             breakdown.file_paths + breakdown.content_data);
        
        breakdown.fragmentation_overhead = breakdown.total_raw() / 10;
        breakdown
    }
}
```

### 2. Add memory size breakdown struct
Add this struct before the `CacheEntry` implementation:
```rust
#[derive(Debug, Clone)]
pub struct MemorySizeBreakdown {
    pub struct_overhead: usize,
    pub result_structs: usize,
    pub file_paths: usize,
    pub content_data: usize,
    pub alignment_padding: usize,
    pub fragmentation_overhead: usize,
}

impl MemorySizeBreakdown {
    fn new() -> Self {
        Self {
            struct_overhead: 0,
            result_structs: 0,
            file_paths: 0,
            content_data: 0,
            alignment_padding: 0,
            fragmentation_overhead: 0,
        }
    }
    
    pub fn total_raw(&self) -> usize {
        self.struct_overhead + self.result_structs + self.file_paths + self.content_data
    }
    
    pub fn total_with_overhead(&self) -> usize {
        self.total_raw() + self.alignment_padding + self.fragmentation_overhead
    }
    
    pub fn format_breakdown(&self) -> String {
        format!(
            "Memory Breakdown:\n  Struct overhead: {} bytes\n  Result structs: {} bytes\n  File paths: {} bytes\n  Content data: {} bytes\n  Alignment padding: {} bytes\n  Fragmentation overhead: {} bytes\n  Total: {} bytes ({:.2} KB)",
            self.struct_overhead,
            self.result_structs,
            self.file_paths,
            self.content_data,
            self.alignment_padding,
            self.fragmentation_overhead,
            self.total_with_overhead(),
            self.total_with_overhead() as f64 / 1024.0
        )
    }
}
```

### 3. Add memory profiling methods to cache
Add these methods to the `MemoryEfficientCache` implementation:
```rust
pub fn get_memory_profile(&self) -> CacheMemoryProfile {
    let cache = self.query_cache.read().unwrap();
    let mut profile = CacheMemoryProfile::new();
    
    for (query, entry) in cache.iter() {
        let breakdown = entry.size_breakdown();
        profile.add_entry(query, &breakdown);
    }
    
    profile.total_reported_usage = *self.current_memory_usage.read().unwrap();
    profile
}

pub fn memory_efficiency_report(&self) -> String {
    let profile = self.get_memory_profile();
    let stats = self.get_stats();
    
    format!(
        "Cache Memory Efficiency Report:\n{}\n\nTop Memory Consumers:\n{}\n\nMemory Distribution:\n{}",
        stats.format_memory_summary(self.max_memory_mb),
        profile.format_top_consumers(5),
        profile.format_memory_distribution()
    )
}

pub fn validate_memory_consistency(&self) -> Result<(), String> {
    let cache = self.query_cache.read().unwrap();
    let reported_usage = *self.current_memory_usage.read().unwrap();
    
    let mut calculated_usage = 0;
    for entry in cache.values() {
        calculated_usage += entry.estimated_size;
    }
    
    let difference = (reported_usage as i64 - calculated_usage as i64).abs() as usize;
    let tolerance = calculated_usage / 100; // 1% tolerance
    
    if difference > tolerance.max(1024) { // At least 1KB tolerance
        Err(format!(
            "Memory inconsistency detected: reported {} bytes, calculated {} bytes (difference: {} bytes)",
            reported_usage, calculated_usage, difference
        ))
    } else {
        Ok(())
    }
}
```

### 4. Add cache memory profile struct
Add this struct before the `MemoryEfficientCache` implementation:
```rust
#[derive(Debug)]
pub struct CacheMemoryProfile {
    pub entry_count: usize,
    pub total_calculated_usage: usize,
    pub total_reported_usage: usize,
    pub avg_entry_size: usize,
    pub largest_entry_size: usize,
    pub smallest_entry_size: usize,
    pub largest_entry_query: String,
    pub content_data_bytes: usize,
    pub file_path_bytes: usize,
    pub struct_overhead_bytes: usize,
    pub fragmentation_bytes: usize,
}

impl CacheMemoryProfile {
    fn new() -> Self {
        Self {
            entry_count: 0,
            total_calculated_usage: 0,
            total_reported_usage: 0,
            avg_entry_size: 0,
            largest_entry_size: 0,
            smallest_entry_size: usize::MAX,
            largest_entry_query: String::new(),
            content_data_bytes: 0,
            file_path_bytes: 0,
            struct_overhead_bytes: 0,
            fragmentation_bytes: 0,
        }
    }
    
    fn add_entry(&mut self, query: &str, breakdown: &MemorySizeBreakdown) {
        let entry_size = breakdown.total_with_overhead();
        
        self.entry_count += 1;
        self.total_calculated_usage += entry_size;
        self.content_data_bytes += breakdown.content_data;
        self.file_path_bytes += breakdown.file_paths;
        self.struct_overhead_bytes += breakdown.struct_overhead + breakdown.result_structs;
        self.fragmentation_bytes += breakdown.alignment_padding + breakdown.fragmentation_overhead;
        
        if entry_size > self.largest_entry_size {
            self.largest_entry_size = entry_size;
            self.largest_entry_query = query.to_string();
        }
        
        if entry_size < self.smallest_entry_size {
            self.smallest_entry_size = entry_size;
        }
        
        self.avg_entry_size = if self.entry_count > 0 {
            self.total_calculated_usage / self.entry_count
        } else {
            0
        };
    }
    
    pub fn format_top_consumers(&self, limit: usize) -> String {
        format!(
            "  Largest entry: '{}' ({:.2} KB)\n  Average entry size: {:.2} KB\n  Smallest entry: {:.2} KB",
            if self.largest_entry_query.len() > 50 { 
                &self.largest_entry_query[..50] 
            } else { 
                &self.largest_entry_query 
            },
            self.largest_entry_size as f64 / 1024.0,
            self.avg_entry_size as f64 / 1024.0,
            if self.smallest_entry_size == usize::MAX { 0.0 } else { self.smallest_entry_size as f64 / 1024.0 }
        )
    }
    
    pub fn format_memory_distribution(&self) -> String {
        if self.total_calculated_usage == 0 {
            return "  No memory usage data available".to_string();
        }
        
        let content_pct = (self.content_data_bytes as f64 / self.total_calculated_usage as f64) * 100.0;
        let paths_pct = (self.file_path_bytes as f64 / self.total_calculated_usage as f64) * 100.0;
        let overhead_pct = (self.struct_overhead_bytes as f64 / self.total_calculated_usage as f64) * 100.0;
        let frag_pct = (self.fragmentation_bytes as f64 / self.total_calculated_usage as f64) * 100.0;
        
        format!(
            "  Content data: {:.1}% ({:.2} KB)\n  File paths: {:.1}% ({:.2} KB)\n  Struct overhead: {:.1}% ({:.2} KB)\n  Fragmentation/alignment: {:.1}% ({:.2} KB)",
            content_pct, self.content_data_bytes as f64 / 1024.0,
            paths_pct, self.file_path_bytes as f64 / 1024.0,
            overhead_pct, self.struct_overhead_bytes as f64 / 1024.0,
            frag_pct, self.fragmentation_bytes as f64 / 1024.0
        )
    }
}
```

### 5. Enhance CacheStats with memory details
Update the `CacheStats` struct and add methods:
```rust
impl CacheStats {
    pub fn format_memory_summary(&self, max_memory_mb: usize) -> String {
        format!(
            "Memory Usage: {:.2}/{} MB ({:.1}% full)\nEntries: {} | Hit Rate: {:.1}% ({} hits, {} misses)",
            self.memory_usage_mb,
            max_memory_mb,
            self.memory_utilization(max_memory_mb) * 100.0,
            self.entries,
            self.hit_rate * 100.0,
            self.total_hits,
            self.total_misses
        )
    }
    
    pub fn memory_pressure_level(&self, max_memory_mb: usize) -> MemoryPressureLevel {
        let utilization = self.memory_utilization(max_memory_mb);
        
        if utilization < 0.5 {
            MemoryPressureLevel::Low
        } else if utilization < 0.8 {
            MemoryPressureLevel::Medium
        } else if utilization < 0.95 {
            MemoryPressureLevel::High
        } else {
            MemoryPressureLevel::Critical
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryPressureLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl MemoryPressureLevel {
    pub fn should_trigger_eviction(&self) -> bool {
        matches!(self, MemoryPressureLevel::High | MemoryPressureLevel::Critical)
    }
    
    pub fn eviction_aggressiveness(&self) -> f64 {
        match self {
            MemoryPressureLevel::Low => 0.0,
            MemoryPressureLevel::Medium => 0.1,
            MemoryPressureLevel::High => 0.25,
            MemoryPressureLevel::Critical => 0.5,
        }
    }
}
```

### 6. Add comprehensive memory calculation tests
Add these tests to the test module:
```rust
#[test]
fn test_memory_size_estimation_accuracy() {
    let small_result = SearchResult {
        file_path: "a.rs".to_string(),
        content: "x".to_string(),
        chunk_index: 0,
        score: 1.0,
    };
    
    let large_result = SearchResult {
        file_path: "very_long_file_path_with_many_directories/subdir/file.rs".to_string(),
        content: "x".repeat(1000),
        chunk_index: 0,
        score: 1.0,
    };
    
    let small_entry = CacheEntry::new(vec![small_result]);
    let large_entry = CacheEntry::new(vec![large_result]);
    
    // Large entry should be significantly larger
    assert!(large_entry.estimated_size > small_entry.estimated_size * 10);
    
    // Both should have reasonable overhead
    assert!(small_entry.estimated_size > 100); // At least 100 bytes overhead
    assert!(large_entry.estimated_size > 1000); // At least content size
}

#[test]
fn test_memory_size_breakdown() {
    let results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "pub fn test() {}".to_string(),
            chunk_index: 0,
            score: 1.0,
        },
        SearchResult {
            file_path: "main.rs".to_string(),
            content: "fn main() { println!(\"Hello\"); }".to_string(),
            chunk_index: 1,
            score: 0.9,
        }
    ];
    
    let entry = CacheEntry::new(results);
    let breakdown = entry.size_breakdown();
    
    // All components should be non-zero
    assert!(breakdown.struct_overhead > 0);
    assert!(breakdown.result_structs > 0);
    assert!(breakdown.file_paths > 0);
    assert!(breakdown.content_data > 0);
    
    // Total should match estimated size
    assert_eq!(breakdown.total_with_overhead(), entry.estimated_size);
    
    // Breakdown should format properly
    let formatted = breakdown.format_breakdown();
    assert!(formatted.contains("Memory Breakdown"));
    assert!(formatted.contains("bytes"));
}

#[test]
fn test_cache_memory_profile() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    let small_results = vec![
        SearchResult {
            file_path: "small.rs".to_string(),
            content: "tiny".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    let large_results = vec![
        SearchResult {
            file_path: "large_file_with_long_name.rs".to_string(),
            content: "x".repeat(500),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    cache.put("small_query".to_string(), small_results);
    cache.put("large_query".to_string(), large_results);
    
    let profile = cache.get_memory_profile();
    
    assert_eq!(profile.entry_count, 2);
    assert!(profile.total_calculated_usage > 0);
    assert!(profile.largest_entry_size > profile.avg_entry_size);
    assert!(profile.content_data_bytes > 0);
    assert!(profile.file_path_bytes > 0);
}

#[test]
fn test_memory_consistency_validation() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    // Empty cache should validate
    assert!(cache.validate_memory_consistency().is_ok());
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add entries and validate
    cache.put("query1".to_string(), test_results.clone());
    cache.put("query2".to_string(), test_results);
    
    assert!(cache.validate_memory_consistency().is_ok());
}

#[test]
fn test_memory_pressure_levels() {
    let stats_low = CacheStats {
        entries: 10,
        memory_usage_bytes: 20 * 1024 * 1024, // 20MB
        memory_usage_mb: 20.0,
        hit_rate: 0.8,
        total_hits: 80,
        total_misses: 20,
    };
    
    let stats_high = CacheStats {
        entries: 100,
        memory_usage_bytes: 90 * 1024 * 1024, // 90MB
        memory_usage_mb: 90.0,
        hit_rate: 0.9,
        total_hits: 900,
        total_misses: 100,
    };
    
    assert_eq!(stats_low.memory_pressure_level(100), MemoryPressureLevel::Low);
    assert_eq!(stats_high.memory_pressure_level(100), MemoryPressureLevel::High);
    
    assert!(!stats_low.memory_pressure_level(100).should_trigger_eviction());
    assert!(stats_high.memory_pressure_level(100).should_trigger_eviction());
}

#[test]
fn test_memory_efficiency_report() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "test content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    cache.put("test_query".to_string(), test_results);
    
    let report = cache.memory_efficiency_report();
    
    assert!(report.contains("Cache Memory Efficiency Report"));
    assert!(report.contains("Memory Usage"));
    assert!(report.contains("Top Memory Consumers"));
    assert!(report.contains("Memory Distribution"));
}
```

## Success Criteria
- [ ] Accurate memory size estimation for different content types
- [ ] Memory breakdown analysis provides detailed insights
- [ ] Memory profiling identifies largest consumers
- [ ] Memory consistency validation catches discrepancies
- [ ] Memory pressure levels guide eviction decisions
- [ ] All tests pass with accurate size calculations
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Memory alignment is important for accurate estimation
- Include fragmentation overhead for realistic sizing
- Size breakdown helps optimize memory usage
- Memory pressure levels enable smart eviction
- Validation catches memory tracking bugs early