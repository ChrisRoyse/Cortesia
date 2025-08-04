# Task 24: Implement Cache Cleanup Optimization

## Context
You are implementing Phase 4 of a vector indexing system. Search engine integration was implemented in the previous task. Now you need to implement comprehensive cache cleanup optimization with background maintenance, fragmentation reduction, and performance optimization.

## Current State
- `src/cache.rs` exists with complete cache implementation and statistics
- `src/cached_search.rs` provides search engine integration
- Cache operations work but need optimization for long-term performance
- Need background cleanup, defragmentation, and maintenance routines

## Task Objective
Implement comprehensive cache cleanup optimization with background maintenance tasks, memory defragmentation, performance optimization, and automated cleanup strategies.

## Implementation Requirements

### 1. Add cache maintenance and cleanup structures
Add this maintenance module to `src/cache.rs`:
```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct CacheMaintenanceConfig {
    pub enable_background_cleanup: bool,
    pub cleanup_interval_seconds: u64,
    pub enable_defragmentation: bool,
    pub defrag_threshold_percent: f64,
    pub enable_ttl_cleanup: bool,
    pub default_ttl_seconds: u64,
    pub enable_access_pattern_optimization: bool,
    pub optimization_interval_seconds: u64,
    pub enable_memory_pressure_cleanup: bool,
    pub memory_pressure_threshold: f64,
    pub aggressive_cleanup_threshold: f64,
}

impl CacheMaintenanceConfig {
    pub fn default() -> Self {
        Self {
            enable_background_cleanup: true,
            cleanup_interval_seconds: 300, // 5 minutes
            enable_defragmentation: true,
            defrag_threshold_percent: 20.0,
            enable_ttl_cleanup: true,
            default_ttl_seconds: 3600, // 1 hour
            enable_access_pattern_optimization: true,
            optimization_interval_seconds: 900, // 15 minutes
            enable_memory_pressure_cleanup: true,
            memory_pressure_threshold: 0.8,
            aggressive_cleanup_threshold: 0.95,
        }
    }
    
    pub fn performance_focused() -> Self {
        Self {
            enable_background_cleanup: true,
            cleanup_interval_seconds: 120, // 2 minutes - more frequent
            enable_defragmentation: true,
            defrag_threshold_percent: 15.0, // More aggressive defrag
            enable_ttl_cleanup: true,
            default_ttl_seconds: 7200, // 2 hours - longer TTL
            enable_access_pattern_optimization: true,
            optimization_interval_seconds: 600, // 10 minutes - more frequent
            enable_memory_pressure_cleanup: true,
            memory_pressure_threshold: 0.75, // Earlier intervention
            aggressive_cleanup_threshold: 0.9,
        }
    }
    
    pub fn memory_conservative() -> Self {
        Self {
            enable_background_cleanup: true,
            cleanup_interval_seconds: 60, // 1 minute - very frequent
            enable_defragmentation: true,
            defrag_threshold_percent: 10.0, // Very aggressive defrag
            enable_ttl_cleanup: true,
            default_ttl_seconds: 1800, // 30 minutes - shorter TTL
            enable_access_pattern_optimization: true,
            optimization_interval_seconds: 300, // 5 minutes
            enable_memory_pressure_cleanup: true,
            memory_pressure_threshold: 0.6, // Very early intervention
            aggressive_cleanup_threshold: 0.8,
        }
    }
}

#[derive(Debug)]
pub struct CacheMaintenanceManager {
    cache: Arc<MemoryEfficientCache>,
    config: CacheMaintenanceConfig,
    running: Arc<AtomicBool>,
    cleanup_thread: Option<JoinHandle<()>>,
    ttl_tracker: Arc<Mutex<TTLTracker>>,
    access_optimizer: Arc<Mutex<AccessPatternOptimizer>>,
    maintenance_stats: Arc<Mutex<MaintenanceStats>>,
}

impl CacheMaintenanceManager {
    pub fn new(cache: Arc<MemoryEfficientCache>, config: CacheMaintenanceConfig) -> Self {
        let ttl_tracker = Arc::new(Mutex::new(TTLTracker::new(config.default_ttl_seconds)));
        let access_optimizer = Arc::new(Mutex::new(AccessPatternOptimizer::new()));
        let maintenance_stats = Arc::new(Mutex::new(MaintenanceStats::new()));
        
        Self {
            cache,
            config,
            running: Arc::new(AtomicBool::new(false)),
            cleanup_thread: None,
            ttl_tracker,
            access_optimizer,
            maintenance_stats,
        }
    }
    
    pub fn start_background_maintenance(&mut self) {
        if !self.config.enable_background_cleanup {
            return;
        }
        
        if self.running.load(Ordering::Relaxed) {
            return; // Already running
        }
        
        self.running.store(true, Ordering::Relaxed);
        
        let cache = Arc::clone(&self.cache);
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let ttl_tracker = Arc::clone(&self.ttl_tracker);
        let access_optimizer = Arc::clone(&self.access_optimizer);
        let maintenance_stats = Arc::clone(&self.maintenance_stats);
        
        let handle = thread::spawn(move || {
            Self::maintenance_loop(
                cache,
                config,
                running,
                ttl_tracker,
                access_optimizer,
                maintenance_stats,
            );
        });
        
        self.cleanup_thread = Some(handle);
    }
    
    pub fn stop_background_maintenance(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        
        if let Some(handle) = self.cleanup_thread.take() {
            let _ = handle.join();
        }
    }
    
    fn maintenance_loop(
        cache: Arc<MemoryEfficientCache>,
        config: CacheMaintenanceConfig,
        running: Arc<AtomicBool>,
        ttl_tracker: Arc<Mutex<TTLTracker>>,
        access_optimizer: Arc<Mutex<AccessPatternOptimizer>>,
        maintenance_stats: Arc<Mutex<MaintenanceStats>>,
    ) {
        let mut last_cleanup = Instant::now();
        let mut last_optimization = Instant::now();
        
        while running.load(Ordering::Relaxed) {
            let now = Instant::now();
            
            // Regular cleanup cycle
            if now.duration_since(last_cleanup).as_secs() >= config.cleanup_interval_seconds {
                let cleanup_result = Self::perform_cleanup_cycle(&cache, &config, &ttl_tracker);
                
                if let Ok(mut stats) = maintenance_stats.lock() {
                    stats.record_cleanup_cycle(cleanup_result);
                }
                
                last_cleanup = now;
            }
            
            // Optimization cycle
            if config.enable_access_pattern_optimization &&
               now.duration_since(last_optimization).as_secs() >= config.optimization_interval_seconds {
                
                let optimization_result = Self::perform_optimization_cycle(&cache, &access_optimizer);
                
                if let Ok(mut stats) = maintenance_stats.lock() {
                    stats.record_optimization_cycle(optimization_result);
                }
                
                last_optimization = now;
            }
            
            // Sleep for a short interval
            thread::sleep(Duration::from_secs(10));
        }
    }
    
    fn perform_cleanup_cycle(
        cache: &Arc<MemoryEfficientCache>,
        config: &CacheMaintenanceConfig,
        ttl_tracker: &Arc<Mutex<TTLTracker>>,
    ) -> CleanupResult {
        let start_time = Instant::now();
        let mut result = CleanupResult::new();
        
        // TTL-based cleanup
        if config.enable_ttl_cleanup {
            if let Ok(mut tracker) = ttl_tracker.lock() {
                let expired_keys = tracker.get_expired_keys();
                for key in &expired_keys {
                    if cache.remove(key) {
                        result.ttl_expired_removed += 1;
                    }
                }
                tracker.cleanup_expired_entries(&expired_keys);
            }
        }
        
        // Memory pressure cleanup
        if config.enable_memory_pressure_cleanup {
            let stats = cache.get_stats();
            let memory_utilization = stats.memory_utilization(cache.max_memory_mb());
            
            if memory_utilization > config.memory_pressure_threshold {
                let cleanup_count = Self::perform_memory_pressure_cleanup(
                    cache,
                    memory_utilization,
                    config.aggressive_cleanup_threshold,
                );
                result.memory_pressure_removed = cleanup_count;
            }
        }
        
        // Defragmentation
        if config.enable_defragmentation {
            let fragmentation_result = Self::perform_defragmentation(cache, config.defrag_threshold_percent);
            result.defragmentation_performed = fragmentation_result.performed;
            result.memory_reclaimed_bytes = fragmentation_result.bytes_reclaimed;
        }
        
        result.duration = start_time.elapsed();
        
        // Validate cache consistency after cleanup
        if let Err(e) = cache.validate_cache() {
            result.consistency_errors.push(e);
        }
        
        result
    }
    
    fn perform_memory_pressure_cleanup(
        cache: &Arc<MemoryEfficientCache>,
        memory_utilization: f64,
        aggressive_threshold: f64,
    ) -> usize {
        let cleanup_ratio = if memory_utilization > aggressive_threshold {
            0.3 // Remove 30% of entries
        } else {
            0.15 // Remove 15% of entries
        };
        
        let current_entries = cache.current_entries();
        let target_removals = ((current_entries as f64) * cleanup_ratio) as usize;
        
        // Get cache entries and sort by access patterns for smart removal
        let candidates = cache.get_eviction_candidates_for_cleanup(target_removals);
        
        let mut removed_count = 0;
        for candidate in candidates {
            if cache.remove(&candidate) {
                removed_count += 1;
            }
        }
        
        removed_count
    }
    
    fn perform_defragmentation(cache: &Arc<MemoryEfficientCache>, threshold_percent: f64) -> DefragmentationResult {
        let start_memory = cache.current_memory_usage();
        let fragmentation_ratio = cache.calculate_fragmentation_ratio();
        
        if fragmentation_ratio * 100.0 < threshold_percent {
            return DefragmentationResult {
                performed: false,
                bytes_reclaimed: 0,
                fragmentation_before: fragmentation_ratio,
                fragmentation_after: fragmentation_ratio,
            };
        }
        
        // Perform defragmentation by rebuilding cache entries
        let defrag_result = cache.defragment_cache();
        let end_memory = cache.current_memory_usage();
        let bytes_reclaimed = start_memory.saturating_sub(end_memory);
        
        DefragmentationResult {
            performed: true,
            bytes_reclaimed,
            fragmentation_before: fragmentation_ratio,
            fragmentation_after: cache.calculate_fragmentation_ratio(),
        }
    }
    
    fn perform_optimization_cycle(
        cache: &Arc<MemoryEfficientCache>,
        access_optimizer: &Arc<Mutex<AccessPatternOptimizer>>,
    ) -> OptimizationResult {
        let start_time = Instant::now();
        let mut result = OptimizationResult::new();
        
        if let Ok(mut optimizer) = access_optimizer.lock() {
            // Analyze access patterns
            let access_analysis = optimizer.analyze_access_patterns(cache);
            result.access_pattern_analysis = Some(access_analysis.clone());
            
            // Apply optimizations based on analysis
            if access_analysis.hot_data_ratio < 0.8 {
                let reorg_result = optimizer.reorganize_hot_data(cache);
                result.hot_data_reorganizations = reorg_result.reorganized_entries;
            }
            
            // Optimize eviction policies based on patterns
            if access_analysis.temporal_locality_score < 0.6 {
                optimizer.suggest_eviction_policy_adjustment(cache);
                result.policy_adjustments_suggested = true;
            }
        }
        
        result.duration = start_time.elapsed();
        result
    }
    
    pub fn perform_manual_cleanup(&self) -> CleanupResult {
        Self::perform_cleanup_cycle(&self.cache, &self.config, &self.ttl_tracker)
    }
    
    pub fn get_maintenance_stats(&self) -> MaintenanceStats {
        if let Ok(stats) = self.maintenance_stats.lock() {
            stats.clone()
        } else {
            MaintenanceStats::new()
        }
    }
    
    pub fn add_entry_with_ttl(&self, key: String, ttl_seconds: u64) {
        if let Ok(mut tracker) = self.ttl_tracker.lock() {
            tracker.add_entry(key, ttl_seconds);
        }
    }
    
    pub fn extend_entry_ttl(&self, key: &str, additional_seconds: u64) {
        if let Ok(mut tracker) = self.ttl_tracker.lock() {
            tracker.extend_ttl(key, additional_seconds);
        }
    }
}

impl Drop for CacheMaintenanceManager {
    fn drop(&mut self) {
        self.stop_background_maintenance();
    }
}

#[derive(Debug, Clone)]
pub struct TTLTracker {
    entries: HashMap<String, SystemTime>,
    default_ttl_seconds: u64,
}

impl TTLTracker {
    fn new(default_ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            default_ttl_seconds,
        }
    }
    
    fn add_entry(&mut self, key: String, ttl_seconds: u64) {
        let expiry = SystemTime::now() + Duration::from_secs(ttl_seconds);
        self.entries.insert(key, expiry);
    }
    
    fn extend_ttl(&mut self, key: &str, additional_seconds: u64) {
        if let Some(expiry) = self.entries.get_mut(key) {
            *expiry = SystemTime::now() + Duration::from_secs(additional_seconds);
        }
    }
    
    fn get_expired_keys(&self) -> Vec<String> {
        let now = SystemTime::now();
        self.entries
            .iter()
            .filter_map(|(key, expiry)| {
                if now > *expiry {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect()
    }
    
    fn cleanup_expired_entries(&mut self, expired_keys: &[String]) {
        for key in expired_keys {
            self.entries.remove(key);
        }
    }
}

#[derive(Debug, Clone)]
pub struct AccessPatternOptimizer {
    access_history: VecDeque<AccessEvent>,
    hot_data_threshold: usize,
    max_history_size: usize,
}

impl AccessPatternOptimizer {
    fn new() -> Self {
        Self {
            access_history: VecDeque::new(),
            hot_data_threshold: 10,
            max_history_size: 10000,
        }
    }
    
    fn analyze_access_patterns(&mut self, cache: &MemoryEfficientCache) -> AccessPatternAnalysis {
        let stats = cache.get_stats();
        let total_accesses = stats.total_hits + stats.total_misses;
        
        // Calculate various metrics
        let hot_data_ratio = self.calculate_hot_data_ratio();
        let temporal_locality_score = self.calculate_temporal_locality();
        let access_distribution = self.analyze_access_distribution();
        
        AccessPatternAnalysis {
            total_accesses,
            hot_data_ratio,
            temporal_locality_score,
            access_distribution,
            cache_efficiency: stats.hit_rate,
            recommendations: self.generate_recommendations(hot_data_ratio, temporal_locality_score),
        }
    }
    
    fn calculate_hot_data_ratio(&self) -> f64 {
        if self.access_history.is_empty() {
            return 1.0;
        }
        
        let mut access_counts: HashMap<String, usize> = HashMap::new();
        for event in &self.access_history {
            *access_counts.entry(event.key.clone()).or_insert(0) += 1;
        }
        
        let hot_entries = access_counts.values().filter(|&&count| count >= self.hot_data_threshold).count();
        hot_entries as f64 / access_counts.len() as f64
    }
    
    fn calculate_temporal_locality(&self) -> f64 {
        if self.access_history.len() < 10 {
            return 0.5; // Default neutral score
        }
        
        let recent_window = 100;
        let recent_accesses: std::collections::HashSet<_> = self.access_history
            .iter()
            .rev()
            .take(recent_window)
            .map(|event| &event.key)
            .collect();
        
        let total_unique_recent = recent_accesses.len();
        let window_size = self.access_history.len().min(recent_window);
        
        // Higher score means better temporal locality (fewer unique keys in recent accesses)
        1.0 - (total_unique_recent as f64 / window_size as f64)
    }
    
    fn analyze_access_distribution(&self) -> AccessDistribution {
        let mut key_counts: HashMap<String, usize> = HashMap::new();
        for event in &self.access_history {
            *key_counts.entry(event.key.clone()).or_insert(0) += 1;
        }
        
        let mut counts: Vec<usize> = key_counts.values().cloned().collect();
        counts.sort_unstable();
        
        let total_keys = counts.len();
        let total_accesses: usize = counts.iter().sum();
        
        AccessDistribution {
            total_unique_keys: total_keys,
            total_accesses,
            most_accessed_count: counts.last().copied().unwrap_or(0),
            least_accessed_count: counts.first().copied().unwrap_or(0),
            median_access_count: if !counts.is_empty() {
                counts[total_keys / 2]
            } else {
                0
            },
        }
    }
    
    fn generate_recommendations(&self, hot_data_ratio: f64, temporal_locality: f64) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if hot_data_ratio < 0.2 {
            recommendations.push("Consider increasing cache size - low hot data ratio indicates frequent evictions".to_string());
        }
        
        if temporal_locality < 0.3 {
            recommendations.push("Poor temporal locality - consider LRU eviction policy".to_string());
        } else if temporal_locality > 0.8 {
            recommendations.push("Excellent temporal locality - LFU policy might be more effective".to_string());
        }
        
        if self.access_history.len() > self.max_history_size / 2 {
            recommendations.push("Consider reducing access history size for better performance".to_string());
        }
        
        recommendations
    }
    
    fn reorganize_hot_data(&mut self, cache: &MemoryEfficientCache) -> ReorganizationResult {
        // This would involve reorganizing cache data structures for better access patterns
        // For now, return a placeholder result
        ReorganizationResult {
            reorganized_entries: 0,
            performance_improvement_estimate: 0.0,
        }
    }
    
    fn suggest_eviction_policy_adjustment(&self, cache: &MemoryEfficientCache) {
        // This would analyze current eviction policy effectiveness
        // and suggest adjustments
        println!("Analyzing eviction policy effectiveness...");
    }
}

#[derive(Debug, Clone)]
pub struct AccessEvent {
    key: String,
    timestamp: Instant,
    was_hit: bool,
}

#[derive(Debug, Clone)]
pub struct AccessPatternAnalysis {
    pub total_accesses: usize,
    pub hot_data_ratio: f64,
    pub temporal_locality_score: f64,
    pub access_distribution: AccessDistribution,
    pub cache_efficiency: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AccessDistribution {
    pub total_unique_keys: usize,
    pub total_accesses: usize,
    pub most_accessed_count: usize,
    pub least_accessed_count: usize,
    pub median_access_count: usize,
}

#[derive(Debug, Clone)]
pub struct ReorganizationResult {
    pub reorganized_entries: usize,
    pub performance_improvement_estimate: f64,
}

#[derive(Debug, Clone)]
pub struct MaintenanceStats {
    pub total_cleanup_cycles: usize,
    pub total_optimization_cycles: usize,
    pub ttl_expired_removed: usize,
    pub memory_pressure_removed: usize,
    pub defragmentations_performed: usize,
    pub total_memory_reclaimed_bytes: usize,
    pub total_cleanup_duration: Duration,
    pub total_optimization_duration: Duration,
    pub consistency_errors: usize,
    pub last_cleanup_time: Option<SystemTime>,
    pub last_optimization_time: Option<SystemTime>,
}

impl MaintenanceStats {
    fn new() -> Self {
        Self {
            total_cleanup_cycles: 0,
            total_optimization_cycles: 0,
            ttl_expired_removed: 0,
            memory_pressure_removed: 0,
            defragmentations_performed: 0,
            total_memory_reclaimed_bytes: 0,
            total_cleanup_duration: Duration::ZERO,
            total_optimization_duration: Duration::ZERO,
            consistency_errors: 0,
            last_cleanup_time: None,
            last_optimization_time: None,
        }
    }
    
    fn record_cleanup_cycle(&mut self, result: CleanupResult) {
        self.total_cleanup_cycles += 1;
        self.ttl_expired_removed += result.ttl_expired_removed;
        self.memory_pressure_removed += result.memory_pressure_removed;
        if result.defragmentation_performed {
            self.defragmentations_performed += 1;
        }
        self.total_memory_reclaimed_bytes += result.memory_reclaimed_bytes;
        self.total_cleanup_duration += result.duration;
        self.consistency_errors += result.consistency_errors.len();
        self.last_cleanup_time = Some(SystemTime::now());
    }
    
    fn record_optimization_cycle(&mut self, result: OptimizationResult) {
        self.total_optimization_cycles += 1;
        self.total_optimization_duration += result.duration;
        self.last_optimization_time = Some(SystemTime::now());
    }
    
    pub fn format_maintenance_report(&self) -> String {
        let avg_cleanup_duration = if self.total_cleanup_cycles > 0 {
            self.total_cleanup_duration.as_secs_f64() / self.total_cleanup_cycles as f64
        } else {
            0.0
        };
        
        let avg_optimization_duration = if self.total_optimization_cycles > 0 {
            self.total_optimization_duration.as_secs_f64() / self.total_optimization_cycles as f64
        } else {
            0.0
        };
        
        format!(
            "Cache Maintenance Report:\n\
             \nCleanup Operations:\n\
             Total Cycles: {}\n\
             TTL Expired Removed: {}\n\
             Memory Pressure Removed: {}\n\
             Defragmentations: {}\n\
             Memory Reclaimed: {:.2} MB\n\
             Avg Cleanup Duration: {:.2}s\n\
             \nOptimization Operations:\n\
             Total Cycles: {}\n\
             Avg Optimization Duration: {:.2}s\n\
             \nHealth:\n\
             Consistency Errors: {}\n\
             Last Cleanup: {}\n\
             Last Optimization: {}",
            self.total_cleanup_cycles,
            self.ttl_expired_removed,
            self.memory_pressure_removed,
            self.defragmentations_performed,
            self.total_memory_reclaimed_bytes as f64 / (1024.0 * 1024.0),
            avg_cleanup_duration,
            self.total_optimization_cycles,
            avg_optimization_duration,
            self.consistency_errors,
            self.format_time_option(self.last_cleanup_time),
            self.format_time_option(self.last_optimization_time)
        )
    }
    
    fn format_time_option(&self, time_opt: Option<SystemTime>) -> String {
        match time_opt {
            Some(time) => {
                if let Ok(duration) = time.elapsed() {
                    format!("{:.1} minutes ago", duration.as_secs_f64() / 60.0)
                } else {
                    "recently".to_string()
                }
            }
            None => "never".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CleanupResult {
    pub ttl_expired_removed: usize,
    pub memory_pressure_removed: usize,
    pub defragmentation_performed: bool,
    pub memory_reclaimed_bytes: usize,
    pub duration: Duration,
    pub consistency_errors: Vec<String>,
}

impl CleanupResult {
    fn new() -> Self {
        Self {
            ttl_expired_removed: 0,
            memory_pressure_removed: 0,
            defragmentation_performed: false,
            memory_reclaimed_bytes: 0,
            duration: Duration::ZERO,
            consistency_errors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DefragmentationResult {
    pub performed: bool,
    pub bytes_reclaimed: usize,
    pub fragmentation_before: f64,
    pub fragmentation_after: f64,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub access_pattern_analysis: Option<AccessPatternAnalysis>,
    pub hot_data_reorganizations: usize,
    pub policy_adjustments_suggested: bool,
    pub duration: Duration,
}

impl OptimizationResult {
    fn new() -> Self {
        Self {
            access_pattern_analysis: None,
            hot_data_reorganizations: 0,
            policy_adjustments_suggested: false,
            duration: Duration::ZERO,
        }
    }
}
```

### 2. Add cleanup optimization methods to MemoryEfficientCache
Add these methods to the existing `MemoryEfficientCache` implementation:
```rust
impl MemoryEfficientCache {
    pub fn get_eviction_candidates_for_cleanup(&self, target_count: usize) -> Vec<String> {
        let cache = self.query_cache.read().unwrap();
        
        if cache.is_empty() || target_count == 0 {
            return Vec::new();
        }
        
        // Prioritize removal of least recently used and largest entries
        let mut candidates: Vec<_> = cache.iter().collect();
        
        // Sort by combined score: age + size + access frequency
        candidates.sort_by(|a, b| {
            let a_score = self.calculate_removal_score(a.1);
            let b_score = self.calculate_removal_score(b.1);
            b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        candidates
            .into_iter()
            .take(target_count)
            .map(|(key, _)| key.clone())
            .collect()
    }
    
    fn calculate_removal_score(&self, entry: &CacheEntry) -> f64 {
        let now = std::time::Instant::now();
        
        // Age score (higher = older, more likely to remove)
        let age_seconds = now.duration_since(entry.timestamp).as_secs_f64();
        let age_score = age_seconds / 3600.0; // Normalize to hours
        
        // Size score (higher = larger, more likely to remove for memory pressure)
        let size_score = entry.estimated_size as f64 / (1024.0 * 1024.0); // Normalize to MB
        
        // Access frequency score (lower = less accessed, more likely to remove)
        let access_score = 1.0 / (entry.access_count as f64 + 1.0);
        
        // Combined weighted score
        (age_score * 0.4) + (size_score * 0.3) + (access_score * 0.3)
    }
    
    pub fn calculate_fragmentation_ratio(&self) -> f64 {
        let cache = self.query_cache.read().unwrap();
        let memory_usage = *self.current_memory_usage.read().unwrap();
        
        if memory_usage == 0 || cache.is_empty() {
            return 0.0;
        }
        
        // Calculate actual data size vs total memory usage
        let actual_data_size: usize = cache.values()
            .map(|entry| entry.results.iter()
                .map(|r| r.file_path.len() + r.content.len())
                .sum::<usize>())
            .sum();
        
        let overhead = memory_usage.saturating_sub(actual_data_size);
        overhead as f64 / memory_usage as f64
    }
    
    pub fn defragment_cache(&self) -> bool {
        // This is a simplified defragmentation - in practice, this would involve
        // more sophisticated memory reallocation and compaction
        
        let mut cache = self.query_cache.write().unwrap();
        let mut memory_usage = self.current_memory_usage.write().unwrap();
        
        // Recalculate all entry sizes and update memory usage
        let mut new_memory_usage = 0;
        for entry in cache.values_mut() {
            entry.recalculate_size();
            new_memory_usage += entry.estimated_size;
        }
        
        let memory_saved = memory_usage.saturating_sub(new_memory_usage);
        *memory_usage = new_memory_usage;
        
        memory_saved > 0
    }
    
    pub fn optimize_for_access_patterns(&self, hot_keys: &[String]) -> usize {
        // This would reorganize internal data structures based on access patterns
        // For now, we'll just ensure hot keys are not candidates for eviction
        
        let cache = self.query_cache.read().unwrap();
        let mut optimized_count = 0;
        
        for key in hot_keys {
            if let Some(entry) = cache.get(key) {
                // In a real implementation, this might move the entry to a
                // faster access structure or adjust its priority
                optimized_count += 1;
            }
        }
        
        optimized_count
    }
    
    pub fn get_cache_health_metrics(&self) -> CacheHealthMetrics {
        let stats = self.get_stats();
        let fragmentation_ratio = self.calculate_fragmentation_ratio();
        let memory_utilization = stats.memory_utilization(self.max_memory_mb);
        let concurrent_stats = self.get_concurrent_stats();
        
        CacheHealthMetrics {
            memory_utilization,
            fragmentation_ratio,
            hit_rate: stats.hit_rate,
            entry_utilization: stats.entries as f64 / self.max_entries as f64,
            avg_entry_size: if stats.entries > 0 {
                stats.memory_usage_bytes / stats.entries
            } else {
                0
            },
            contention_level: if concurrent_stats.has_high_contention() {
                ContentionLevel::High
            } else if concurrent_stats.read_contention_count > 0 || concurrent_stats.write_contention_count > 0 {
                ContentionLevel::Medium
            } else {
                ContentionLevel::Low
            },
            health_score: self.calculate_overall_health_score(&stats, fragmentation_ratio, memory_utilization),
        }
    }
    
    fn calculate_overall_health_score(&self, stats: &CacheStats, fragmentation: f64, memory_util: f64) -> f64 {
        // Score from 0.0 to 1.0 based on multiple factors
        let hit_rate_score = stats.hit_rate;
        let memory_score = 1.0 - memory_util.min(1.0);
        let fragmentation_score = 1.0 - fragmentation.min(1.0);
        let utilization_score = if memory_util > 0.9 { 0.5 } else { 1.0 };
        
        (hit_rate_score * 0.4) + (memory_score * 0.2) + (fragmentation_score * 0.2) + (utilization_score * 0.2)
    }
}

#[derive(Debug, Clone)]
pub struct CacheHealthMetrics {
    pub memory_utilization: f64,
    pub fragmentation_ratio: f64,
    pub hit_rate: f64,
    pub entry_utilization: f64,
    pub avg_entry_size: usize,
    pub contention_level: ContentionLevel,
    pub health_score: f64,
}

impl CacheHealthMetrics {
    pub fn format_health_report(&self) -> String {
        let health_status = match self.health_score {
            score if score >= 0.8 => "EXCELLENT",
            score if score >= 0.6 => "GOOD", 
            score if score >= 0.4 => "FAIR",
            score if score >= 0.2 => "POOR",
            _ => "CRITICAL",
        };
        
        format!(
            "Cache Health Report:\n\
             Overall Health: {} ({:.1}%)\n\
             \nKey Metrics:\n\
             Memory Utilization: {:.1}%\n\
             Fragmentation Ratio: {:.1}%\n\
             Hit Rate: {:.1}%\n\
             Entry Utilization: {:.1}%\n\
             Avg Entry Size: {:.2} KB\n\
             Contention Level: {:?}\n\
             \nRecommendations:\n{}",
            health_status, self.health_score * 100.0,
            self.memory_utilization * 100.0,
            self.fragmentation_ratio * 100.0,
            self.hit_rate * 100.0,
            self.entry_utilization * 100.0,
            self.avg_entry_size as f64 / 1024.0,
            self.contention_level,
            self.generate_health_recommendations()
        )
    }
    
    fn generate_health_recommendations(&self) -> String {
        let mut recommendations = Vec::new();
        
        if self.memory_utilization > 0.9 {
            recommendations.push("• Consider increasing cache memory limit");
        }
        
        if self.fragmentation_ratio > 0.3 {
            recommendations.push("• Enable defragmentation or increase defrag frequency");
        }
        
        if self.hit_rate < 0.5 {
            recommendations.push("• Review cache sizing and eviction policies");
        }
        
        if self.contention_level == ContentionLevel::High {
            recommendations.push("• Optimize concurrent access patterns");
        }
        
        if self.entry_utilization > 0.95 {
            recommendations.push("• Consider increasing max entry limit");
        }
        
        if recommendations.is_empty() {
            recommendations.push("• Cache is operating optimally");
        }
        
        recommendations.join("\n")
    }
}
```

### 3. Add comprehensive cleanup optimization tests
Add these tests to the cache module:
```rust
#[cfg(test)]
mod cleanup_optimization_tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_cache_maintenance_manager_basic() {
        let cache = Arc::new(MemoryEfficientCache::new(100, 10));
        let config = CacheMaintenanceConfig::default();
        let mut manager = CacheMaintenanceManager::new(Arc::clone(&cache), config);
        
        // Add some test data
        let test_results = vec![
            SearchResult {
                file_path: "test.rs".to_string(),
                content: "test content".to_string(),
                chunk_index: 0,
                score: 1.0,
            }
        ];
        
        cache.put("test_key".to_string(), test_results);
        manager.add_entry_with_ttl("test_key".to_string(), 1); // 1 second TTL
        
        // Wait for TTL to expire
        std::thread::sleep(Duration::from_secs(2));
        
        // Perform manual cleanup
        let cleanup_result = manager.perform_manual_cleanup();
        
        assert!(cleanup_result.ttl_expired_removed >= 1);
        assert!(cleanup_result.duration > Duration::ZERO);
        
        // Key should be removed from cache
        assert!(cache.get("test_key").is_none());
    }
    
    #[test]
    fn test_background_maintenance() {
        let cache = Arc::new(MemoryEfficientCache::new(100, 10));
        let mut config = CacheMaintenanceConfig::default();
        config.cleanup_interval_seconds = 1; // Very frequent for testing
        
        let mut manager = CacheMaintenanceManager::new(Arc::clone(&cache), config);
        
        // Add test data
        for i in 0..10 {
            let test_results = vec![
                SearchResult {
                    file_path: format!("test_{}.rs", i),
                    content: format!("content {}", i),
                    chunk_index: 0,
                    score: 1.0,
                }
            ];
            cache.put(format!("key_{}", i), test_results);
            manager.add_entry_with_ttl(format!("key_{}", i), 2); // 2 second TTL
        }
        
        // Start background maintenance
        manager.start_background_maintenance();
        
        // Wait for cleanup to occur
        std::thread::sleep(Duration::from_secs(3));
        
        // Stop maintenance
        manager.stop_background_maintenance();
        
        // Check stats
        let stats = manager.get_maintenance_stats();
        assert!(stats.total_cleanup_cycles > 0);
        
        println!("{}", stats.format_maintenance_report());
    }
    
    #[test]
    fn test_memory_pressure_cleanup() {
        let cache = Arc::new(MemoryEfficientCache::new(100, 1)); // Small memory limit
        let config = CacheMaintenanceConfig::memory_conservative();
        let manager = CacheMaintenanceManager::new(Arc::clone(&cache), config);
        
        // Fill cache with large entries to create memory pressure
        let large_content = "x".repeat(50000); // 50KB per entry
        for i in 0..20 {
            let test_results = vec![
                SearchResult {
                    file_path: format!("large_{}.rs", i),
                    content: large_content.clone(),
                    chunk_index: 0,
                    score: 1.0,
                }
            ];
            let success = cache.put(format!("large_key_{}", i), test_results);
            if !success {
                break; // Memory limit reached
            }
        }
        
        let initial_entries = cache.current_entries();
        let initial_memory = cache.current_memory_usage_mb();
        
        // Perform cleanup under memory pressure
        let cleanup_result = manager.perform_manual_cleanup();
        
        let final_entries = cache.current_entries();
        let final_memory = cache.current_memory_usage_mb();
        
        // Should have cleaned up some entries due to memory pressure
        assert!(cleanup_result.memory_pressure_removed > 0 || final_entries < initial_entries);
        assert!(final_memory <= initial_memory);
        
        println!("Memory pressure cleanup: {} -> {} entries, {:.2} -> {:.2} MB",
                 initial_entries, final_entries, initial_memory, final_memory);
    }
    
    #[test]
    fn test_cache_defragmentation() {
        let cache = Arc::new(MemoryEfficientCache::new(100, 10));
        
        // Add and remove entries to create fragmentation
        for i in 0..50 {
            let test_results = vec![
                SearchResult {
                    file_path: format!("frag_{}.rs", i),
                    content: "x".repeat(1000 * (i % 5 + 1)), // Variable sizes
                    chunk_index: 0,
                    score: 1.0,
                }
            ];
            cache.put(format!("frag_key_{}", i), test_results);
        }
        
        // Remove every other entry to create fragmentation
        for i in (0..50).step_by(2) {
            cache.remove(&format!("frag_key_{}", i));
        }
        
        let initial_fragmentation = cache.calculate_fragmentation_ratio();
        let initial_memory = cache.current_memory_usage();
        
        // Perform defragmentation
        let defrag_success = cache.defragment_cache();
        
        let final_fragmentation = cache.calculate_fragmentation_ratio();
        let final_memory = cache.current_memory_usage();
        
        if initial_fragmentation > 0.1 {
            assert!(defrag_success);
            assert!(final_fragmentation <= initial_fragmentation);
            assert!(final_memory <= initial_memory);
        }
        
        println!("Defragmentation: {:.1}% -> {:.1}% fragmentation, {} -> {} bytes",
                 initial_fragmentation * 100.0, final_fragmentation * 100.0,
                 initial_memory, final_memory);
    }
    
    #[test]
    fn test_access_pattern_optimization() {
        let cache = Arc::new(MemoryEfficientCache::new(100, 10));
        
        // Create test data with known access patterns
        let test_results = vec![
            SearchResult {
                file_path: "hot.rs".to_string(),
                content: "hot content".to_string(),
                chunk_index: 0,
                score: 1.0,
            }
        ];
        
        // Add entries
        for i in 0..10 {
            cache.put(format!("key_{}", i), test_results.clone());
        }
        
        // Simulate hot access pattern
        let hot_keys = vec!["key_0".to_string(), "key_1".to_string(), "key_2".to_string()];
        for _ in 0..10 {
            for key in &hot_keys {
                cache.get(key);
            }
        }
        
        // Optimize for access patterns
        let optimized_count = cache.optimize_for_access_patterns(&hot_keys);
        
        assert_eq!(optimized_count, 3);
        
        // Test access pattern analyzer
        let mut optimizer = AccessPatternOptimizer::new();
        let analysis = optimizer.analyze_access_patterns(&cache);
        
        assert!(analysis.total_accesses > 0);
        assert!(analysis.cache_efficiency > 0.0);
        assert!(!analysis.recommendations.is_empty());
        
        println!("Access pattern analysis:");
        println!("  Hot data ratio: {:.2}", analysis.hot_data_ratio);
        println!("  Temporal locality: {:.2}", analysis.temporal_locality_score);
        println!("  Recommendations: {:?}", analysis.recommendations);
    }
    
    #[test]
    fn test_cache_health_metrics() {
        let cache = Arc::new(MemoryEfficientCache::new(100, 10));
        
        // Add various types of data
        for i in 0..30 {
            let test_results = vec![
                SearchResult {
                    file_path: format!("health_{}.rs", i),
                    content: "x".repeat(1000 * (i % 3 + 1)),
                    chunk_index: 0,
                    score: 1.0,
                }
            ];
            cache.put(format!("health_key_{}", i), test_results);
        }
        
        // Generate some access patterns
        for i in 0..15 {
            cache.get(&format!("health_key_{}", i));
        }
        
        let health_metrics = cache.get_cache_health_metrics();
        
        assert!(health_metrics.memory_utilization >= 0.0);
        assert!(health_metrics.fragmentation_ratio >= 0.0);
        assert!(health_metrics.hit_rate >= 0.0);
        assert!(health_metrics.health_score >= 0.0 && health_metrics.health_score <= 1.0);
        
        let health_report = health_metrics.format_health_report();
        
        assert!(health_report.contains("Cache Health Report"));
        assert!(health_report.contains("Overall Health"));
        assert!(health_report.contains("Recommendations"));
        
        println!("{}", health_report);
    }
    
    #[test]
    fn test_ttl_tracker_functionality() {
        let mut tracker = TTLTracker::new(60); // 1 minute default TTL
        
        // Add entries with different TTLs
        tracker.add_entry("short_ttl".to_string(), 1); // 1 second
        tracker.add_entry("long_ttl".to_string(), 3600); // 1 hour
        
        // No entries should be expired immediately
        let expired = tracker.get_expired_keys();
        assert!(expired.is_empty());
        
        // Wait for short TTL to expire
        std::thread::sleep(Duration::from_secs(2));
        
        let expired = tracker.get_expired_keys();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0], "short_ttl");
        
        // Extend TTL for an entry
        tracker.extend_ttl("long_ttl", 1800); // Add 30 minutes
        
        // Clean up expired entries
        tracker.cleanup_expired_entries(&expired);
        
        let expired_after_cleanup = tracker.get_expired_keys();
        assert!(expired_after_cleanup.is_empty());
    }
    
    #[test]
    fn test_maintenance_statistics() {
        let mut stats = MaintenanceStats::new();
        
        // Record some cleanup cycles
        let cleanup_result = CleanupResult {
            ttl_expired_removed: 5,
            memory_pressure_removed: 3,
            defragmentation_performed: true,
            memory_reclaimed_bytes: 1024 * 1024, // 1MB
            duration: Duration::from_millis(500),
            consistency_errors: Vec::new(),
        };
        
        stats.record_cleanup_cycle(cleanup_result);
        
        // Record optimization cycle
        let optimization_result = OptimizationResult {
            access_pattern_analysis: None,
            hot_data_reorganizations: 2,
            policy_adjustments_suggested: true,
            duration: Duration::from_millis(200),
        };
        
        stats.record_optimization_cycle(optimization_result);
        
        // Verify statistics
        assert_eq!(stats.total_cleanup_cycles, 1);
        assert_eq!(stats.total_optimization_cycles, 1);
        assert_eq!(stats.ttl_expired_removed, 5);
        assert_eq!(stats.memory_pressure_removed, 3);
        assert_eq!(stats.defragmentations_performed, 1);
        assert_eq!(stats.total_memory_reclaimed_bytes, 1024 * 1024);
        
        let report = stats.format_maintenance_report();
        assert!(report.contains("Total Cycles: 1"));
        assert!(report.contains("Memory Reclaimed: 1.00 MB"));
        
        println!("{}", report);
    }
}
```

## Success Criteria
- [ ] Comprehensive cache maintenance manager implemented
- [ ] Background cleanup operations with configurable intervals
- [ ] TTL-based expiration tracking and cleanup
- [ ] Memory pressure detection and cleanup
- [ ] Cache defragmentation reduces memory overhead
- [ ] Access pattern optimization improves performance
- [ ] Health metrics provide actionable insights
- [ ] Maintenance statistics track cleanup effectiveness
- [ ] All cleanup optimization tests pass
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Background maintenance runs in separate thread
- TTL tracking enables time-based cache expiration
- Memory pressure cleanup prevents system resource exhaustion
- Defragmentation reduces memory fragmentation overhead
- Access pattern optimization improves cache hit rates
- Health metrics provide operational insights
- Maintenance statistics enable performance monitoring