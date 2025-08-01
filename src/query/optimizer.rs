use std::time::Duration;

pub struct QueryOptimizer {
    query_history: Vec<QueryStats>,
    optimization_settings: OptimizationSettings,
}

#[derive(Debug, Clone)]
pub struct QueryStats {
    pub query_embedding_hash: u64,
    pub execution_time: Duration,
    pub result_count: usize,
    pub cache_hit: bool,
    pub timestamp: std::time::Instant,
}

#[derive(Debug, Clone)]
pub struct OptimizationSettings {
    pub enable_caching: bool,
    pub cache_ttl: Duration,
    pub max_cache_size: usize,
    pub enable_query_rewriting: bool,
    pub enable_result_prefetching: bool,
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            max_cache_size: 1000,
            enable_query_rewriting: true,
            enable_result_prefetching: false,
        }
    }
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryOptimizer {
    pub fn new() -> Self {
        Self {
            query_history: Vec::new(),
            optimization_settings: OptimizationSettings::default(),
        }
    }
    
    pub fn with_settings(settings: OptimizationSettings) -> Self {
        Self {
            query_history: Vec::new(),
            optimization_settings: settings,
        }
    }
    
    pub fn record_query(&mut self, stats: QueryStats) {
        self.query_history.push(stats);
        
        // Keep only recent queries
        let cutoff = std::time::Instant::now() - Duration::from_secs(3600); // 1 hour
        self.query_history.retain(|stat| stat.timestamp > cutoff);
    }
    
    pub fn suggest_optimizations(&self) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        // Analyze query patterns
        let avg_execution_time = self.average_execution_time();
        let cache_hit_rate = self.cache_hit_rate();
        
        if avg_execution_time > Duration::from_millis(100) {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::ReduceQueryScope,
                description: "Average query time is high. Consider reducing max_entities or max_depth.".to_string(),
                impact: OptimizationImpact::High,
                estimated_improvement: 0.3,
            });
        }
        
        if cache_hit_rate < 0.5 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::IncreaseCacheSize,
                description: "Low cache hit rate. Consider increasing cache size or TTL.".to_string(),
                impact: OptimizationImpact::Medium,
                estimated_improvement: 0.2,
            });
        }
        
        // Check for repeated query patterns
        if self.has_repeated_patterns() {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::EnablePrefetching,
                description: "Detected repeated query patterns. Enable prefetching for better performance.".to_string(),
                impact: OptimizationImpact::Medium,
                estimated_improvement: 0.25,
            });
        }
        
        suggestions
    }
    
    fn average_execution_time(&self) -> Duration {
        if self.query_history.is_empty() {
            return Duration::ZERO;
        }
        
        let total: Duration = self.query_history.iter().map(|stat| stat.execution_time).sum();
        total / self.query_history.len() as u32
    }
    
    fn cache_hit_rate(&self) -> f64 {
        if self.query_history.is_empty() {
            return 0.0;
        }
        
        let cache_hits = self.query_history.iter().filter(|stat| stat.cache_hit).count();
        cache_hits as f64 / self.query_history.len() as f64
    }
    
    fn has_repeated_patterns(&self) -> bool {
        use std::collections::HashMap;
        
        let mut hash_counts: HashMap<u64, usize> = HashMap::new();
        for stat in &self.query_history {
            *hash_counts.entry(stat.query_embedding_hash).or_insert(0) += 1;
        }
        
        hash_counts.values().any(|&count| count > 3)
    }
    
    pub fn optimize_query_parameters(&self, max_entities: usize, max_depth: u8) -> (usize, u8) {
        let avg_time = self.average_execution_time();
        
        // Reduce parameters if queries are too slow
        if avg_time > Duration::from_millis(50) {
            let optimized_entities = (max_entities as f64 * 0.8) as usize;
            let optimized_depth = if max_depth > 2 { max_depth - 1 } else { max_depth };
            (optimized_entities.max(5), optimized_depth)
        } else {
            (max_entities, max_depth)
        }
    }
    
    pub fn performance_report(&self) -> PerformanceReport {
        PerformanceReport {
            total_queries: self.query_history.len(),
            average_execution_time: self.average_execution_time(),
            cache_hit_rate: self.cache_hit_rate(),
            queries_per_second: self.calculate_qps(),
            optimization_suggestions: self.suggest_optimizations(),
        }
    }
    
    fn calculate_qps(&self) -> f64 {
        if self.query_history.len() < 2 {
            return 0.0;
        }
        
        let recent_queries = self.query_history.iter()
            .rev()
            .take(100)
            .collect::<Vec<_>>();
        
        if recent_queries.len() < 2 {
            return 0.0;
        }
        
        let time_span = recent_queries.first().unwrap().timestamp
            .duration_since(recent_queries.last().unwrap().timestamp);
        
        if time_span.as_secs_f64() > 0.0 {
            recent_queries.len() as f64 / time_span.as_secs_f64()
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub suggestion_type: OptimizationType,
    pub description: String,
    pub impact: OptimizationImpact,
    pub estimated_improvement: f64, // 0.0 to 1.0
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    ReduceQueryScope,
    IncreaseCacheSize,
    EnablePrefetching,
    OptimizeEmbeddings,
    RebuildIndex,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationImpact {
    Low,
    Medium,
    High,
}

#[derive(Debug)]
pub struct PerformanceReport {
    pub total_queries: usize,
    pub average_execution_time: Duration,
    pub cache_hit_rate: f64,
    pub queries_per_second: f64,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}