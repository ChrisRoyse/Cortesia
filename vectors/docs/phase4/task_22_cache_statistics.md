# Task 22: Implement Cache Statistics

## Context
You are implementing Phase 4 of a vector indexing system. Concurrent cache access was implemented in the previous task. Now you need to implement comprehensive cache statistics with detailed analytics, performance metrics, and operational insights.

## Current State
- `src/cache.rs` exists with full concurrent cache implementation
- Basic hit/miss statistics and concurrent access tracking are implemented
- Need comprehensive statistics collection and analysis
- Need performance analytics and trend detection

## Task Objective
Implement comprehensive cache statistics with detailed analytics, performance trends, operational metrics, and reporting capabilities.

## Implementation Requirements

### 1. Add comprehensive statistics collection structures
Add these statistics structures to the cache module:
```rust
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    // Basic counters
    pub hit_count: usize,
    pub miss_count: usize,
    pub put_count: usize,
    pub eviction_count: usize,
    pub error_count: usize,
    
    // Performance metrics
    pub avg_get_duration_ns: f64,
    pub avg_put_duration_ns: f64,
    pub p95_get_duration_ns: f64,
    pub p95_put_duration_ns: f64,
    pub max_get_duration_ns: u64,
    pub max_put_duration_ns: u64,
    
    // Memory and capacity
    pub current_entries: usize,
    pub max_entries_reached: usize,
    pub current_memory_bytes: usize,
    pub max_memory_bytes_reached: usize,
    pub total_bytes_evicted: usize,
    pub total_entries_evicted: usize,
    
    // Access patterns
    pub unique_queries_seen: usize,
    pub query_frequency_distribution: HashMap<String, usize>,
    pub temporal_access_pattern: Vec<TemporalAccessPoint>,
    
    // Efficiency metrics
    pub memory_efficiency_ratio: f64,  // useful_data / total_memory
    pub cache_effectiveness_score: f64, // hit_rate * memory_efficiency
    pub query_diversity_index: f64,     // unique_queries / total_queries
    
    // Operational stats
    pub uptime_seconds: u64,
    pub operations_per_second: f64,
    pub memory_growth_rate_mb_per_hour: f64,
    pub eviction_rate_per_hour: f64,
    
    // Recent performance windows
    pub recent_hit_rates: VecDeque<f64>,
    pub recent_memory_usage: VecDeque<usize>,
    pub recent_operation_counts: VecDeque<usize>,
}

#[derive(Debug, Clone)]
pub struct TemporalAccessPoint {
    pub timestamp: SystemTime,
    pub hits: usize,
    pub misses: usize,
    pub puts: usize,
    pub memory_usage: usize,
}

#[derive(Debug, Clone)]
pub struct QueryStatistics {
    pub query: String,
    pub hit_count: usize,
    pub miss_count: usize,
    pub last_access: SystemTime,
    pub first_access: SystemTime,
    pub avg_result_size: usize,
    pub total_bytes_served: usize,
    pub access_frequency_per_hour: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub current_value: f64,
    pub previous_value: f64,
    pub change_percent: f64,
    pub trend_direction: TrendDirection,
    pub is_significant: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Declining,
    Stable,
    Unknown,
}

#[derive(Debug)]
pub struct StatisticsCollector {
    start_time: SystemTime,
    get_durations: VecDeque<u64>,      // Store in nanoseconds
    put_durations: VecDeque<u64>,
    temporal_points: VecDeque<TemporalAccessPoint>,
    query_stats: HashMap<String, QueryStatistics>,
    max_stored_durations: usize,
    max_temporal_points: usize,
    max_tracked_queries: usize,
    last_snapshot_time: SystemTime,
    previous_statistics: Option<CacheStatistics>,
}

impl StatisticsCollector {
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            get_durations: VecDeque::new(),
            put_durations: VecDeque::new(),
            temporal_points: VecDeque::new(),
            query_stats: HashMap::new(),
            max_stored_durations: 1000,
            max_temporal_points: 288, // 24 hours in 5-minute intervals
            max_tracked_queries: 500,
            last_snapshot_time: SystemTime::now(),
            previous_statistics: None,
        }
    }
    
    pub fn record_get_operation(&mut self, query: &str, was_hit: bool, duration: Duration, result_size: usize) {
        let duration_ns = duration.as_nanos() as u64;
        self.get_durations.push_back(duration_ns);
        
        // Maintain size limit
        if self.get_durations.len() > self.max_stored_durations {
            self.get_durations.pop_front();
        }
        
        // Update query statistics
        self.update_query_stats(query, was_hit, result_size);
    }
    
    pub fn record_put_operation(&mut self, query: &str, duration: Duration, data_size: usize) {
        let duration_ns = duration.as_nanos() as u64;
        self.put_durations.push_back(duration_ns);
        
        if self.put_durations.len() > self.max_stored_durations {
            self.put_durations.pop_front();
        }
        
        // Update query stats for puts
        let entry = self.query_stats.entry(query.to_string()).or_insert_with(|| {
            QueryStatistics {
                query: query.to_string(),
                hit_count: 0,
                miss_count: 0,
                last_access: SystemTime::now(),
                first_access: SystemTime::now(),
                avg_result_size: 0,
                total_bytes_served: 0,
                access_frequency_per_hour: 0.0,
            }
        });
        
        entry.avg_result_size = if entry.hit_count + entry.miss_count > 0 {
            (entry.total_bytes_served + data_size) / (entry.hit_count + entry.miss_count + 1)
        } else {
            data_size
        };
    }
    
    fn update_query_stats(&mut self, query: &str, was_hit: bool, result_size: usize) {
        let now = SystemTime::now();
        let entry = self.query_stats.entry(query.to_string()).or_insert_with(|| {
            QueryStatistics {
                query: query.to_string(),
                hit_count: 0,
                miss_count: 0,
                last_access: now,
                first_access: now,
                avg_result_size: 0,
                total_bytes_served: 0,
                access_frequency_per_hour: 0.0,
            }
        });
        
        if was_hit {
            entry.hit_count += 1;
            entry.total_bytes_served += result_size;
        } else {
            entry.miss_count += 1;
        }
        
        entry.last_access = now;
        
        // Update average result size
        if entry.hit_count > 0 {
            entry.avg_result_size = entry.total_bytes_served / entry.hit_count;
        }
        
        // Calculate access frequency
        if let Ok(duration) = now.duration_since(entry.first_access) {
            let hours = duration.as_secs_f64() / 3600.0;
            if hours > 0.0 {
                entry.access_frequency_per_hour = (entry.hit_count + entry.miss_count) as f64 / hours;
            }
        }
        
        // Maintain query limit
        if self.query_stats.len() > self.max_tracked_queries {
            // Remove least recently accessed query
            let oldest_query = self.query_stats
                .iter()
                .min_by_key(|(_, stats)| stats.last_access)
                .map(|(query, _)| query.clone());
            
            if let Some(query) = oldest_query {
                self.query_stats.remove(&query);
            }
        }
    }
    
    pub fn take_temporal_snapshot(&mut self, cache_stats: &CacheStats) {
        let now = SystemTime::now();
        let point = TemporalAccessPoint {
            timestamp: now,
            hits: cache_stats.total_hits,
            misses: cache_stats.total_misses,
            puts: 0, // Would need to track this separately
            memory_usage: cache_stats.memory_usage_bytes,
        };
        
        self.temporal_points.push_back(point);
        
        if self.temporal_points.len() > self.max_temporal_points {
            self.temporal_points.pop_front();
        }
    }
    
    pub fn generate_statistics(&mut self, cache_stats: &CacheStats) -> CacheStatistics {
        let now = SystemTime::now();
        let uptime = now.duration_since(self.start_time)
            .unwrap_or(Duration::ZERO);
        
        // Calculate performance metrics
        let (avg_get_duration, p95_get_duration, max_get_duration) = self.calculate_duration_stats(&self.get_durations);
        let (avg_put_duration, p95_put_duration, max_put_duration) = self.calculate_duration_stats(&self.put_durations);
        
        // Calculate derived metrics
        let total_operations = cache_stats.total_hits + cache_stats.total_misses;
        let operations_per_second = if uptime.as_secs() > 0 {
            total_operations as f64 / uptime.as_secs_f64()
        } else {
            0.0
        };
        
        let memory_efficiency_ratio = self.calculate_memory_efficiency(cache_stats);
        let cache_effectiveness_score = cache_stats.hit_rate * memory_efficiency_ratio;
        let query_diversity_index = if total_operations > 0 {
            self.query_stats.len() as f64 / total_operations as f64
        } else {
            0.0
        };
        
        // Growth rates
        let (memory_growth_rate, eviction_rate) = self.calculate_growth_rates(uptime);
        
        // Recent windows
        let recent_hit_rates = self.calculate_recent_hit_rates();
        let recent_memory_usage = self.extract_recent_memory_usage();
        let recent_operation_counts = self.calculate_recent_operation_counts();
        
        let statistics = CacheStatistics {
            hit_count: cache_stats.total_hits,
            miss_count: cache_stats.total_misses,
            put_count: 0, // Would need separate tracking
            eviction_count: 0, // Would get from cache eviction stats
            error_count: 0, // Would need separate tracking
            
            avg_get_duration_ns: avg_get_duration,
            avg_put_duration_ns: avg_put_duration,
            p95_get_duration_ns: p95_get_duration,
            p95_put_duration_ns: p95_put_duration,
            max_get_duration_ns: max_get_duration,
            max_put_duration_ns: max_put_duration,
            
            current_entries: cache_stats.entries,
            max_entries_reached: cache_stats.entries, // Would need historical max
            current_memory_bytes: cache_stats.memory_usage_bytes,
            max_memory_bytes_reached: cache_stats.memory_usage_bytes, // Would need historical max
            total_bytes_evicted: 0, // Would get from cache eviction stats
            total_entries_evicted: 0,
            
            unique_queries_seen: self.query_stats.len(),
            query_frequency_distribution: self.get_query_frequency_distribution(),
            temporal_access_pattern: self.temporal_points.iter().cloned().collect(),
            
            memory_efficiency_ratio,
            cache_effectiveness_score,
            query_diversity_index,
            
            uptime_seconds: uptime.as_secs(),
            operations_per_second,
            memory_growth_rate_mb_per_hour: memory_growth_rate,
            eviction_rate_per_hour: eviction_rate,
            
            recent_hit_rates,
            recent_memory_usage,
            recent_operation_counts,
        };
        
        self.previous_statistics = Some(statistics.clone());
        self.last_snapshot_time = now;
        
        statistics
    }
    
    fn calculate_duration_stats(&self, durations: &VecDeque<u64>) -> (f64, f64, u64) {
        if durations.is_empty() {
            return (0.0, 0.0, 0);
        }
        
        let mut sorted: Vec<u64> = durations.iter().cloned().collect();
        sorted.sort_unstable();
        
        let avg = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64;
        let p95_index = (sorted.len() as f64 * 0.95) as usize;
        let p95 = sorted.get(p95_index).copied().unwrap_or(0) as f64;
        let max = sorted.last().copied().unwrap_or(0);
        
        (avg, p95, max)
    }
    
    fn calculate_memory_efficiency(&self, cache_stats: &CacheStats) -> f64 {
        if cache_stats.memory_usage_bytes == 0 {
            return 1.0;
        }
        
        // Estimate useful data vs overhead
        let estimated_useful_data = self.query_stats.values()
            .map(|stats| stats.total_bytes_served)
            .sum::<usize>();
        
        if estimated_useful_data > 0 {
            estimated_useful_data as f64 / cache_stats.memory_usage_bytes as f64
        } else {
            0.5 // Default assumption
        }
    }
    
    fn calculate_growth_rates(&self, uptime: Duration) -> (f64, f64) {
        let hours = uptime.as_secs_f64() / 3600.0;
        if hours <= 0.0 {
            return (0.0, 0.0);
        }
        
        // Memory growth rate (simplified)
        let memory_growth_rate = if let Some(first_point) = self.temporal_points.front() {
            if let Some(last_point) = self.temporal_points.back() {
                let memory_growth_bytes = (last_point.memory_usage as i64 - first_point.memory_usage as i64).max(0) as f64;
                let time_diff_hours = last_point.timestamp.duration_since(first_point.timestamp)
                    .unwrap_or(Duration::ZERO).as_secs_f64() / 3600.0;
                
                if time_diff_hours > 0.0 {
                    (memory_growth_bytes / (1024.0 * 1024.0)) / time_diff_hours
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        // Eviction rate (would need actual eviction data)
        let eviction_rate = 0.0; // Placeholder
        
        (memory_growth_rate, eviction_rate)
    }
    
    fn get_query_frequency_distribution(&self) -> HashMap<String, usize> {
        self.query_stats.iter()
            .map(|(query, stats)| (query.clone(), stats.hit_count + stats.miss_count))
            .collect()
    }
    
    fn calculate_recent_hit_rates(&self) -> VecDeque<f64> {
        let mut hit_rates = VecDeque::new();
        
        for window in self.temporal_points.windows(2) {
            if let [prev, curr] = window {
                let hits_delta = curr.hits.saturating_sub(prev.hits);
                let misses_delta = curr.misses.saturating_sub(prev.misses);
                let total_delta = hits_delta + misses_delta;
                
                let hit_rate = if total_delta > 0 {
                    hits_delta as f64 / total_delta as f64
                } else {
                    0.0
                };
                
                hit_rates.push_back(hit_rate);
            }
        }
        
        hit_rates
    }
    
    fn extract_recent_memory_usage(&self) -> VecDeque<usize> {
        self.temporal_points.iter().map(|point| point.memory_usage).collect()
    }
    
    fn calculate_recent_operation_counts(&self) -> VecDeque<usize> {
        let mut operation_counts = VecDeque::new();
        
        for window in self.temporal_points.windows(2) {
            if let [prev, curr] = window {
                let ops_delta = (curr.hits + curr.misses + curr.puts).saturating_sub(prev.hits + prev.misses + prev.puts);
                operation_counts.push_back(ops_delta);
            }
        }
        
        operation_counts
    }
}
```

### 2. Add performance trend analysis
Add these trend analysis methods:
```rust
impl StatisticsCollector {
    pub fn analyze_performance_trends(&self, current_stats: &CacheStatistics) -> Vec<PerformanceTrend> {
        let mut trends = Vec::new();
        
        if let Some(prev_stats) = &self.previous_statistics {
            trends.push(self.analyze_metric_trend(
                "Hit Rate",
                current_stats.hit_count as f64 / (current_stats.hit_count + current_stats.miss_count).max(1) as f64,
                prev_stats.hit_count as f64 / (prev_stats.hit_count + prev_stats.miss_count).max(1) as f64,
                5.0, // 5% threshold
            ));
            
            trends.push(self.analyze_metric_trend(
                "Average Get Duration",
                current_stats.avg_get_duration_ns / 1_000_000.0, // Convert to milliseconds
                prev_stats.avg_get_duration_ns / 1_000_000.0,
                10.0, // 10% threshold
            ));
            
            trends.push(self.analyze_metric_trend(
                "Memory Efficiency",
                current_stats.memory_efficiency_ratio,
                prev_stats.memory_efficiency_ratio,
                5.0,
            ));
            
            trends.push(self.analyze_metric_trend(
                "Operations Per Second",
                current_stats.operations_per_second,
                prev_stats.operations_per_second,
                15.0, // 15% threshold for throughput
            ));
            
            trends.push(self.analyze_metric_trend(
                "Cache Effectiveness",
                current_stats.cache_effectiveness_score,
                prev_stats.cache_effectiveness_score,
                10.0,
            ));
        }
        
        trends
    }
    
    fn analyze_metric_trend(&self, name: &str, current: f64, previous: f64, threshold_percent: f64) -> PerformanceTrend {
        let change_percent = if previous != 0.0 {
            ((current - previous) / previous) * 100.0
        } else if current > 0.0 {
            100.0 // New metric appeared
        } else {
            0.0
        };
        
        let trend_direction = if change_percent.abs() < 1.0 {
            TrendDirection::Stable
        } else if change_percent > 0.0 {
            if name.contains("Duration") {
                TrendDirection::Declining // Higher duration is worse
            } else {
                TrendDirection::Improving // Higher values generally better
            }
        } else {
            if name.contains("Duration") {
                TrendDirection::Improving // Lower duration is better
            } else {
                TrendDirection::Declining // Lower values generally worse
            }
        };
        
        let is_significant = change_percent.abs() > threshold_percent;
        
        PerformanceTrend {
            metric_name: name.to_string(),
            current_value: current,
            previous_value: previous,
            change_percent,
            trend_direction,
            is_significant,
        }
    }
    
    pub fn get_top_queries(&self, limit: usize) -> Vec<QueryStatistics> {
        let mut queries: Vec<_> = self.query_stats.values().cloned().collect();
        queries.sort_by(|a, b| {
            let a_total = a.hit_count + a.miss_count;
            let b_total = b.hit_count + b.miss_count;
            b_total.cmp(&a_total)
        });
        queries.into_iter().take(limit).collect()
    }
    
    pub fn get_performance_anomalies(&self, current_stats: &CacheStatistics) -> Vec<PerformanceAnomaly> {
        let mut anomalies = Vec::new();
        
        // Check for unusual hit rate
        let hit_rate = current_stats.hit_count as f64 / (current_stats.hit_count + current_stats.miss_count).max(1) as f64;
        if hit_rate < 0.1 {
            anomalies.push(PerformanceAnomaly {
                severity: AnomalySeverity::High,
                description: format!("Very low hit rate: {:.1}%", hit_rate * 100.0),
                recommendation: "Review query patterns and cache sizing".to_string(),
            });
        }
        
        // Check for high latency
        if current_stats.avg_get_duration_ns > 10_000_000.0 { // 10ms
            anomalies.push(PerformanceAnomaly {
                severity: AnomalySeverity::Medium,
                description: format!("High average get latency: {:.1}ms", current_stats.avg_get_duration_ns / 1_000_000.0),
                recommendation: "Investigate cache contention or memory pressure".to_string(),
            });
        }
        
        // Check for memory inefficiency
        if current_stats.memory_efficiency_ratio < 0.3 {
            anomalies.push(PerformanceAnomaly {
                severity: AnomalySeverity::Medium,
                description: format!("Low memory efficiency: {:.1}%", current_stats.memory_efficiency_ratio * 100.0),
                recommendation: "Review data structures and eviction policies".to_string(),
            });
        }
        
        // Check for excessive evictions
        if current_stats.eviction_rate_per_hour > 100.0 {
            anomalies.push(PerformanceAnomaly {
                severity: AnomalySeverity::High,
                description: format!("High eviction rate: {:.0}/hour", current_stats.eviction_rate_per_hour),
                recommendation: "Increase cache size or review access patterns".to_string(),
            });
        }
        
        anomalies
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    pub severity: AnomalySeverity,
    pub description: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}
```

### 3. Add statistics reporting capabilities
Add comprehensive reporting methods:
```rust
impl CacheStatistics {
    pub fn format_comprehensive_report(&self) -> String {
        format!(
            "=== CACHE PERFORMANCE REPORT ===\n\
             \nBasic Metrics:\n{}\
             \nPerformance Metrics:\n{}\
             \nMemory & Capacity:\n{}\
             \nAccess Patterns:\n{}\
             \nEfficiency Metrics:\n{}\
             \nOperational Stats:\n{}\
             \nRecent Trends:\n{}",
            self.format_basic_metrics(),
            self.format_performance_metrics(),
            self.format_memory_metrics(),
            self.format_access_patterns(),
            self.format_efficiency_metrics(),
            self.format_operational_stats(),
            self.format_recent_trends()
        )
    }
    
    fn format_basic_metrics(&self) -> String {
        let total_requests = self.hit_count + self.miss_count;
        let hit_rate = if total_requests > 0 {
            (self.hit_count as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };
        
        format!(
            "  Total Requests: {} ({} hits, {} misses)\n\
             Hit Rate: {:.1}%\n\
             Put Operations: {}\n\
             Evictions: {}\n\
             Errors: {}\n",
            total_requests, self.hit_count, self.miss_count,
            hit_rate, self.put_count, self.eviction_count, self.error_count
        )
    }
    
    fn format_performance_metrics(&self) -> String {
        format!(
            "  Average GET: {:.2}ms (P95: {:.2}ms, Max: {:.2}ms)\n\
             Average PUT: {:.2}ms (P95: {:.2}ms, Max: {:.2}ms)\n\
             Throughput: {:.0} ops/sec\n",
            self.avg_get_duration_ns / 1_000_000.0,
            self.p95_get_duration_ns / 1_000_000.0,
            self.max_get_duration_ns as f64 / 1_000_000.0,
            self.avg_put_duration_ns / 1_000_000.0,
            self.p95_put_duration_ns / 1_000_000.0,
            self.max_put_duration_ns as f64 / 1_000_000.0,
            self.operations_per_second
        )
    }
    
    fn format_memory_metrics(&self) -> String {
        format!(
            "  Current Memory: {:.2}MB ({} entries)\n\
             Peak Memory: {:.2}MB ({} entries max)\n\
             Evicted: {:.2}MB ({} entries)\n\
             Growth Rate: {:.2}MB/hour\n",
            self.current_memory_bytes as f64 / (1024.0 * 1024.0),
            self.current_entries,
            self.max_memory_bytes_reached as f64 / (1024.0 * 1024.0),
            self.max_entries_reached,
            self.total_bytes_evicted as f64 / (1024.0 * 1024.0),
            self.total_entries_evicted,
            self.memory_growth_rate_mb_per_hour
        )
    }
    
    fn format_access_patterns(&self) -> String {
        let avg_query_frequency = if !self.query_frequency_distribution.is_empty() {
            self.query_frequency_distribution.values().sum::<usize>() as f64 / 
            self.query_frequency_distribution.len() as f64
        } else {
            0.0
        };
        
        format!(
            "  Unique Queries: {}\n\
             Query Diversity: {:.3}\n\
             Avg Query Frequency: {:.1}\n\
             Temporal Data Points: {}\n",
            self.unique_queries_seen,
            self.query_diversity_index,
            avg_query_frequency,
            self.temporal_access_pattern.len()
        )
    }
    
    fn format_efficiency_metrics(&self) -> String {
        format!(
            "  Memory Efficiency: {:.1}%\n\
             Cache Effectiveness: {:.3}\n\
             Eviction Rate: {:.1}/hour\n",
            self.memory_efficiency_ratio * 100.0,
            self.cache_effectiveness_score,
            self.eviction_rate_per_hour
        )
    }
    
    fn format_operational_stats(&self) -> String {
        let uptime_hours = self.uptime_seconds as f64 / 3600.0;
        format!(
            "  Uptime: {:.1} hours\n\
             Operations/sec: {:.0}\n\
             Recent Hit Rate Trend: {}\n",
            uptime_hours,
            self.operations_per_second,
            self.format_recent_hit_rate_trend()
        )
    }
    
    fn format_recent_trends(&self) -> String {
        if self.recent_hit_rates.len() < 2 {
            return "  Insufficient data for trend analysis\n".to_string();
        }
        
        let recent_avg = self.recent_hit_rates.iter().sum::<f64>() / self.recent_hit_rates.len() as f64;
        let memory_trend = if let (Some(first), Some(last)) = (self.recent_memory_usage.front(), self.recent_memory_usage.back()) {
            if *last > *first {
                "increasing"
            } else if *last < *first {
                "decreasing"
            } else {
                "stable"
            }
        } else {
            "unknown"
        };
        
        format!(
            "  Recent Hit Rate: {:.1}%\n\
             Memory Trend: {}\n\
             Operation Count Trend: {}\n",
            recent_avg * 100.0,
            memory_trend,
            if self.recent_operation_counts.len() > 1 { "tracked" } else { "insufficient data" }
        )
    }
    
    fn format_recent_hit_rate_trend(&self) -> String {
        if self.recent_hit_rates.len() < 2 {
            return "no trend data".to_string();
        }
        
        let recent_avg = self.recent_hit_rates.iter().rev().take(5).sum::<f64>() / 
                        self.recent_hit_rates.len().min(5) as f64;
        format!("{:.1}%", recent_avg * 100.0)
    }
    
    pub fn format_json_report(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    pub fn export_csv_summary(&self) -> String {
        format!(
            "timestamp,hit_count,miss_count,hit_rate,avg_get_ms,avg_put_ms,memory_mb,entries,ops_per_sec\n\
             {},{},{},{:.3},{:.2},{:.2},{:.2},{},{:.0}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
            self.hit_count,
            self.miss_count,
            self.hit_count as f64 / (self.hit_count + self.miss_count).max(1) as f64,
            self.avg_get_duration_ns / 1_000_000.0,
            self.avg_put_duration_ns / 1_000_000.0,
            self.current_memory_bytes as f64 / (1024.0 * 1024.0),
            self.current_entries,
            self.operations_per_second
        )
    }
}

impl PerformanceTrend {
    pub fn format_trend_report(trends: &[PerformanceTrend]) -> String {
        let mut report = String::from("Performance Trend Analysis:\n");
        
        for trend in trends {
            let direction_symbol = match trend.trend_direction {
                TrendDirection::Improving => "↗",
                TrendDirection::Declining => "↘",
                TrendDirection::Stable => "→",
                TrendDirection::Unknown => "?",
            };
            
            let significance = if trend.is_significant { " (SIGNIFICANT)" } else { "" };
            
            report.push_str(&format!(
                "  {} {}: {:.2} → {:.2} ({:+.1}%){}{}\n",
                direction_symbol,
                trend.metric_name,
                trend.previous_value,
                trend.current_value,
                trend.change_percent,
                significance
            ));
        }
        
        report
    }
}

impl PerformanceAnomaly {
    pub fn format_anomaly_report(anomalies: &[PerformanceAnomaly]) -> String {
        if anomalies.is_empty() {
            return "No performance anomalies detected.\n".to_string();
        }
        
        let mut report = String::from("Performance Anomalies Detected:\n");
        
        for (i, anomaly) in anomalies.iter().enumerate() {
            let severity_symbol = match anomaly.severity {
                AnomalySeverity::Critical => "🔴",
                AnomalySeverity::High => "🟠",
                AnomalySeverity::Medium => "🟡",
                AnomalySeverity::Low => "🟢",
            };
            
            report.push_str(&format!(
                "  {}. {} {}\n     Recommendation: {}\n",
                i + 1,
                severity_symbol,
                anomaly.description,
                anomaly.recommendation
            ));
        }
        
        report
    }
}
```

### 4. Add comprehensive statistics tests
Add these statistics tests to the test module:
```rust
#[cfg(test)]
mod statistics_tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_statistics_collector_basic_functionality() {
        let mut collector = StatisticsCollector::new();
        let cache_stats = CacheStats {
            entries: 10,
            memory_usage_bytes: 1024 * 1024,
            memory_usage_mb: 1.0,
            hit_rate: 0.8,
            total_hits: 80,
            total_misses: 20,
        };
        
        // Record some operations
        collector.record_get_operation("query1", true, Duration::from_millis(5), 1024);
        collector.record_get_operation("query2", false, Duration::from_millis(3), 0);
        collector.record_put_operation("query3", Duration::from_millis(10), 2048);
        
        let statistics = collector.generate_statistics(&cache_stats);
        
        assert_eq!(statistics.hit_count, 80);
        assert_eq!(statistics.miss_count, 20);
        assert_eq!(statistics.current_entries, 10);
        assert_eq!(statistics.unique_queries_seen, 3);
        assert!(statistics.avg_get_duration_ns > 0.0);
        assert!(statistics.operations_per_second >= 0.0);
    }
    
    #[test]
    fn test_performance_trend_analysis() {
        let mut collector = StatisticsCollector::new();
        let cache_stats = CacheStats {
            entries: 10,
            memory_usage_bytes: 1024 * 1024,
            memory_usage_mb: 1.0,
            hit_rate: 0.7,
            total_hits: 70,
            total_misses: 30,
        };
        
        // Generate initial statistics
        let initial_stats = collector.generate_statistics(&cache_stats);
        
        // Simulate time passing and performance change
        std::thread::sleep(Duration::from_millis(10));
        
        let improved_cache_stats = CacheStats {
            entries: 15,
            memory_usage_bytes: 1024 * 1024,
            memory_usage_mb: 1.0,
            hit_rate: 0.9,  // Improved hit rate
            total_hits: 180,
            total_misses: 20,
        };
        
        let new_stats = collector.generate_statistics(&improved_cache_stats);
        let trends = collector.analyze_performance_trends(&new_stats);
        
        assert!(!trends.is_empty());
        
        // Find hit rate trend
        let hit_rate_trend = trends.iter().find(|t| t.metric_name == "Hit Rate");
        assert!(hit_rate_trend.is_some());
        
        let trend = hit_rate_trend.unwrap();
        assert_eq!(trend.trend_direction, TrendDirection::Improving);
        assert!(trend.change_percent > 0.0);
    }
    
    #[test]
    fn test_query_statistics_tracking() {
        let mut collector = StatisticsCollector::new();
        
        // Record multiple accesses to same queries
        collector.record_get_operation("popular_query", true, Duration::from_millis(2), 1024);
        collector.record_get_operation("popular_query", true, Duration::from_millis(3), 1024);
        collector.record_get_operation("popular_query", false, Duration::from_millis(5), 0);
        
        collector.record_get_operation("rare_query", true, Duration::from_millis(4), 512);
        
        let top_queries = collector.get_top_queries(10);
        
        assert_eq!(top_queries.len(), 2);
        
        // Popular query should be first
        let popular = &top_queries[0];
        assert_eq!(popular.query, "popular_query");
        assert_eq!(popular.hit_count, 2);
        assert_eq!(popular.miss_count, 1);
        assert!(popular.access_frequency_per_hour > 0.0);
        
        let rare = &top_queries[1];
        assert_eq!(rare.query, "rare_query");
        assert_eq!(rare.hit_count, 1);
        assert_eq!(rare.miss_count, 0);
    }
    
    #[test]
    fn test_temporal_snapshot_functionality() {
        let mut collector = StatisticsCollector::new();
        let cache_stats = CacheStats {
            entries: 5,
            memory_usage_bytes: 512 * 1024,
            memory_usage_mb: 0.5,
            hit_rate: 0.6,
            total_hits: 60,
            total_misses: 40,
        };
        
        // Take multiple temporal snapshots
        collector.take_temporal_snapshot(&cache_stats);
        
        std::thread::sleep(Duration::from_millis(5));
        
        let updated_stats = CacheStats {
            entries: 8,
            memory_usage_bytes: 768 * 1024,
            memory_usage_mb: 0.75,
            hit_rate: 0.7,
            total_hits: 140,
            total_misses: 60,
        };
        
        collector.take_temporal_snapshot(&updated_stats);
        
        let statistics = collector.generate_statistics(&updated_stats);
        
        assert_eq!(statistics.temporal_access_pattern.len(), 2);
        assert!(statistics.recent_hit_rates.len() >= 0); // May be empty if not enough data
    }
    
    #[test]
    fn test_performance_anomaly_detection() {
        let mut collector = StatisticsCollector::new();
        
        // Create statistics with anomalies
        let anomalous_stats = CacheStatistics {
            hit_count: 5,
            miss_count: 95,  // Very low hit rate
            put_count: 10,
            eviction_count: 0,
            error_count: 0,
            avg_get_duration_ns: 15_000_000.0,  // 15ms - high latency
            avg_put_duration_ns: 5_000_000.0,
            p95_get_duration_ns: 25_000_000.0,
            p95_put_duration_ns: 10_000_000.0,
            max_get_duration_ns: 50_000_000,
            max_put_duration_ns: 20_000_000,
            current_entries: 100,
            max_entries_reached: 100,
            current_memory_bytes: 10 * 1024 * 1024,
            max_memory_bytes_reached: 10 * 1024 * 1024,
            total_bytes_evicted: 0,
            total_entries_evicted: 0,
            unique_queries_seen: 50,
            query_frequency_distribution: HashMap::new(),
            temporal_access_pattern: Vec::new(),
            memory_efficiency_ratio: 0.2,  // Low efficiency
            cache_effectiveness_score: 0.02,
            query_diversity_index: 0.5,
            uptime_seconds: 3600,
            operations_per_second: 100.0,
            memory_growth_rate_mb_per_hour: 5.0,
            eviction_rate_per_hour: 150.0,  // High eviction rate
            recent_hit_rates: VecDeque::new(),
            recent_memory_usage: VecDeque::new(),
            recent_operation_counts: VecDeque::new(),
        };
        
        let anomalies = collector.get_performance_anomalies(&anomalous_stats);
        
        assert!(!anomalies.is_empty());
        
        // Should detect low hit rate
        let low_hit_rate = anomalies.iter().any(|a| a.description.contains("low hit rate"));
        assert!(low_hit_rate);
        
        // Should detect high latency
        let high_latency = anomalies.iter().any(|a| a.description.contains("latency"));
        assert!(high_latency);
        
        // Should detect low memory efficiency
        let low_efficiency = anomalies.iter().any(|a| a.description.contains("memory efficiency"));
        assert!(low_efficiency);
    }
    
    #[test]
    fn test_statistics_report_formatting() {
        let statistics = CacheStatistics {
            hit_count: 750,
            miss_count: 250,
            put_count: 100,
            eviction_count: 10,
            error_count: 2,
            avg_get_duration_ns: 2_500_000.0,
            avg_put_duration_ns: 8_000_000.0,
            p95_get_duration_ns: 5_000_000.0,
            p95_put_duration_ns: 15_000_000.0,
            max_get_duration_ns: 20_000_000,
            max_put_duration_ns: 50_000_000,
            current_entries: 50,
            max_entries_reached: 60,
            current_memory_bytes: 5 * 1024 * 1024,
            max_memory_bytes_reached: 6 * 1024 * 1024,
            total_bytes_evicted: 1024 * 1024,
            total_entries_evicted: 10,
            unique_queries_seen: 200,
            query_frequency_distribution: HashMap::new(),
            temporal_access_pattern: Vec::new(),
            memory_efficiency_ratio: 0.75,
            cache_effectiveness_score: 0.6,
            query_diversity_index: 0.2,
            uptime_seconds: 7200,
            operations_per_second: 138.9,
            memory_growth_rate_mb_per_hour: 1.5,
            eviction_rate_per_hour: 5.0,
            recent_hit_rates: VecDeque::new(),
            recent_memory_usage: VecDeque::new(),
            recent_operation_counts: VecDeque::new(),
        };
        
        let report = statistics.format_comprehensive_report();
        
        assert!(report.contains("CACHE PERFORMANCE REPORT"));
        assert!(report.contains("Hit Rate: 75.0%"));
        assert!(report.contains("Average GET: 2.50ms"));
        assert!(report.contains("Current Memory: 5.00MB"));
        assert!(report.contains("Memory Efficiency: 75.0%"));
        assert!(report.contains("Operations/sec: 139"));
        
        // Test CSV export
        let csv = statistics.export_csv_summary();
        assert!(csv.contains("hit_count,miss_count"));
        assert!(csv.contains("750,250"));
        
        println!("Sample comprehensive report:\n{}", report);
        println!("\nSample CSV export:\n{}", csv);
    }
    
    #[test]
    fn test_duration_statistics_calculation() {
        let mut collector = StatisticsCollector::new();
        
        // Add some duration samples
        let durations = vec![
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(5),
            Duration::from_millis(10),
            Duration::from_millis(15),
            Duration::from_millis(20),
            Duration::from_millis(25),
            Duration::from_millis(30),
            Duration::from_millis(100), // Outlier
        ];
        
        for duration in durations {
            collector.record_get_operation("test_query", true, duration, 1024);
        }
        
        let cache_stats = CacheStats {
            entries: 1,
            memory_usage_bytes: 1024,
            memory_usage_mb: 0.001,
            hit_rate: 1.0,
            total_hits: 10,
            total_misses: 0,
        };
        
        let statistics = collector.generate_statistics(&cache_stats);
        
        // Verify duration statistics
        assert!(statistics.avg_get_duration_ns > 0.0);
        assert!(statistics.p95_get_duration_ns > statistics.avg_get_duration_ns);
        assert!(statistics.max_get_duration_ns >= statistics.p95_get_duration_ns as u64);
        
        println!("Duration stats - Avg: {:.2}ms, P95: {:.2}ms, Max: {:.2}ms",
                 statistics.avg_get_duration_ns / 1_000_000.0,
                 statistics.p95_get_duration_ns / 1_000_000.0,
                 statistics.max_get_duration_ns as f64 / 1_000_000.0);
    }
}
```

## Success Criteria
- [ ] Comprehensive statistics collection implemented
- [ ] Performance trend analysis detects improvements and regressions
- [ ] Query-level statistics provide actionable insights
- [ ] Temporal access patterns are tracked accurately
- [ ] Performance anomaly detection identifies issues
- [ ] Multiple report formats (text, JSON, CSV) available
- [ ] All statistics tests pass validation
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Statistics provide deep insights into cache behavior
- Trend analysis helps detect performance regressions
- Query-level analytics identify optimization opportunities
- Anomaly detection prevents performance issues
- Multiple export formats enable integration with monitoring systems
- Temporal tracking enables capacity planning