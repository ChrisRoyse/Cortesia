//! Performance analysis and bottleneck detection

use super::types::*;
use crate::core::types::EntityKey;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

impl EfficiencyAnalyzer {
    /// Create new efficiency analyzer
    pub fn new() -> Self {
        Self {
            metrics_history: Vec::new(),
            baseline_metrics: PerformanceMetrics::default(),
            efficiency_threshold: 0.7,
            analysis_window: Duration::from_secs(300),
        }
    }

    /// Analyze current performance metrics
    pub fn analyze_performance(&mut self, current_metrics: &PerformanceMetrics) -> Result<f32> {
        // Calculate efficiency score
        let efficiency_score = self.calculate_efficiency_score(current_metrics);
        
        // Update metrics history
        self.metrics_history.push(current_metrics.clone());
        
        // Maintain history size
        if self.metrics_history.len() > 100 {
            self.metrics_history.remove(0);
        }
        
        Ok(efficiency_score)
    }

    /// Calculate efficiency score
    fn calculate_efficiency_score(&self, metrics: &PerformanceMetrics) -> f32 {
        let latency_score = self.calculate_latency_score(metrics);
        let memory_score = self.calculate_memory_score(metrics);
        let cache_score = metrics.cache_hit_rate;
        let throughput_score = metrics.throughput.min(1.0);
        let error_score = 1.0 - metrics.error_rate;
        
        let weights = [0.3, 0.2, 0.2, 0.2, 0.1];
        let scores = [latency_score, memory_score, cache_score, throughput_score, error_score];
        
        scores.iter().zip(weights.iter()).map(|(score, weight)| score * weight).sum()
    }

    /// Calculate latency score
    fn calculate_latency_score(&self, metrics: &PerformanceMetrics) -> f32 {
        let baseline_latency = self.baseline_metrics.query_latency.as_millis() as f32;
        let current_latency = metrics.query_latency.as_millis() as f32;
        
        if baseline_latency == 0.0 {
            return 1.0;
        }
        
        let ratio = current_latency / baseline_latency;
        (1.0 / ratio).min(1.0)
    }

    /// Calculate memory score
    fn calculate_memory_score(&self, metrics: &PerformanceMetrics) -> f32 {
        let baseline_memory = self.baseline_metrics.memory_usage as f32;
        let current_memory = metrics.memory_usage as f32;
        
        if baseline_memory == 0.0 {
            return 1.0;
        }
        
        let ratio = current_memory / baseline_memory;
        (1.0 / ratio).min(1.0)
    }

    /// Get performance trends
    pub fn get_performance_trends(&self) -> HashMap<String, f32> {
        let mut trends = HashMap::new();
        
        if self.metrics_history.len() < 2 {
            return trends;
        }
        
        let recent_metrics = &self.metrics_history[self.metrics_history.len() - 1];
        let previous_metrics = &self.metrics_history[self.metrics_history.len() - 2];
        
        // Calculate trends
        let latency_trend = self.calculate_trend(
            previous_metrics.query_latency.as_millis() as f32,
            recent_metrics.query_latency.as_millis() as f32
        );
        
        let memory_trend = self.calculate_trend(
            previous_metrics.memory_usage as f32,
            recent_metrics.memory_usage as f32
        );
        
        let cache_trend = self.calculate_trend(
            previous_metrics.cache_hit_rate,
            recent_metrics.cache_hit_rate
        );
        
        trends.insert("latency".to_string(), latency_trend);
        trends.insert("memory".to_string(), memory_trend);
        trends.insert("cache".to_string(), cache_trend);
        
        trends
    }

    /// Calculate trend between two values
    fn calculate_trend(&self, previous: f32, current: f32) -> f32 {
        if previous == 0.0 {
            return 0.0;
        }
        
        (current - previous) / previous
    }

    /// Detect performance anomalies
    pub fn detect_anomalies(&self, metrics: &PerformanceMetrics) -> Vec<String> {
        let mut anomalies = Vec::new();
        
        // Check latency anomaly
        if metrics.query_latency > Duration::from_millis(500) {
            anomalies.push("High query latency detected".to_string());
        }
        
        // Check memory anomaly
        if metrics.memory_usage > 1_000_000_000 { // 1GB
            anomalies.push("High memory usage detected".to_string());
        }
        
        // Check cache anomaly
        if metrics.cache_hit_rate < 0.3 {
            anomalies.push("Low cache hit rate detected".to_string());
        }
        
        // Check error rate anomaly
        if metrics.error_rate > 0.05 {
            anomalies.push("High error rate detected".to_string());
        }
        
        // Check resource utilization anomaly
        if metrics.resource_utilization > 0.9 {
            anomalies.push("High resource utilization detected".to_string());
        }
        
        anomalies
    }

    /// Get efficiency recommendations
    pub fn get_efficiency_recommendations(&self, metrics: &PerformanceMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Latency recommendations
        if metrics.query_latency > Duration::from_millis(200) {
            recommendations.push("Consider query optimization or indexing".to_string());
        }
        
        // Memory recommendations
        if metrics.memory_usage > 500_000_000 { // 500MB
            recommendations.push("Consider memory optimization or garbage collection".to_string());
        }
        
        // Cache recommendations
        if metrics.cache_hit_rate < 0.5 {
            recommendations.push("Consider cache optimization or warming".to_string());
        }
        
        // Throughput recommendations
        if metrics.throughput < 0.5 {
            recommendations.push("Consider parallel processing or load balancing".to_string());
        }
        
        recommendations
    }

    /// Update baseline metrics
    pub fn update_baseline(&mut self, metrics: &PerformanceMetrics) {
        self.baseline_metrics = metrics.clone();
    }

    /// Check if performance meets threshold
    pub fn meets_efficiency_threshold(&self, metrics: &PerformanceMetrics) -> bool {
        let efficiency_score = self.calculate_efficiency_score(metrics);
        efficiency_score >= self.efficiency_threshold
    }
}

impl BottleneckDetector {
    /// Create new bottleneck detector
    pub fn new() -> Self {
        Self {
            detection_sensitivity: 0.8,
            monitoring_window: Duration::from_secs(60),
            bottleneck_threshold: 0.7,
            identified_bottlenecks: Vec::new(),
        }
    }

    /// Detect bottlenecks in the system
    pub async fn detect_bottlenecks(
        &mut self,
        graph: &BrainEnhancedKnowledgeGraph,
        metrics: &PerformanceMetrics,
    ) -> Result<Vec<PerformanceBottleneck>> {
        let mut bottlenecks = Vec::new();
        
        // Query latency bottlenecks
        if let Some(bottleneck) = self.detect_query_latency_bottleneck(graph, metrics).await? {
            bottlenecks.push(bottleneck);
        }
        
        // Memory usage bottlenecks
        if let Some(bottleneck) = self.detect_memory_bottleneck(graph, metrics).await? {
            bottlenecks.push(bottleneck);
        }
        
        // Cache efficiency bottlenecks
        if let Some(bottleneck) = self.detect_cache_bottleneck(graph, metrics).await? {
            bottlenecks.push(bottleneck);
        }
        
        // Network traversal bottlenecks
        if let Some(bottleneck) = self.detect_traversal_bottleneck(graph, metrics).await? {
            bottlenecks.push(bottleneck);
        }
        
        // Update identified bottlenecks
        self.identified_bottlenecks = bottlenecks.clone();
        
        Ok(bottlenecks)
    }

    /// Detect query latency bottlenecks
    async fn detect_query_latency_bottleneck(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        metrics: &PerformanceMetrics,
    ) -> Result<Option<PerformanceBottleneck>> {
        if metrics.query_latency > Duration::from_millis(300) {
            // Find entities with high query times
            let slow_entities = self.find_slow_query_entities(graph).await?;
            
            let bottleneck = PerformanceBottleneck {
                bottleneck_type: BottleneckType::QueryLatency,
                severity: (metrics.query_latency.as_millis() as f32 / 1000.0).min(1.0),
                affected_entities: slow_entities,
                suggested_optimization: OptimizationType::QueryOptimization,
                estimated_improvement: 0.3,
                detection_time: Instant::now(),
            };
            
            return Ok(Some(bottleneck));
        }
        
        Ok(None)
    }

    /// Detect memory bottlenecks
    async fn detect_memory_bottleneck(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        metrics: &PerformanceMetrics,
    ) -> Result<Option<PerformanceBottleneck>> {
        if metrics.memory_usage > 800_000_000 { // 800MB
            // Find entities with high memory usage
            let memory_heavy_entities = self.find_memory_heavy_entities(graph).await?;
            
            let bottleneck = PerformanceBottleneck {
                bottleneck_type: BottleneckType::MemoryUsage,
                severity: (metrics.memory_usage as f32 / 1_000_000_000.0).min(1.0),
                affected_entities: memory_heavy_entities,
                suggested_optimization: OptimizationType::MemoryOptimization,
                estimated_improvement: 0.25,
                detection_time: Instant::now(),
            };
            
            return Ok(Some(bottleneck));
        }
        
        Ok(None)
    }

    /// Detect cache bottlenecks
    async fn detect_cache_bottleneck(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        metrics: &PerformanceMetrics,
    ) -> Result<Option<PerformanceBottleneck>> {
        if metrics.cache_hit_rate < 0.4 {
            // Find entities with poor cache performance
            let cache_poor_entities = self.find_cache_poor_entities(graph).await?;
            
            let bottleneck = PerformanceBottleneck {
                bottleneck_type: BottleneckType::CacheEfficiency,
                severity: 1.0 - metrics.cache_hit_rate,
                affected_entities: cache_poor_entities,
                suggested_optimization: OptimizationType::CacheOptimization,
                estimated_improvement: 0.2,
                detection_time: Instant::now(),
            };
            
            return Ok(Some(bottleneck));
        }
        
        Ok(None)
    }

    /// Detect network traversal bottlenecks
    async fn detect_traversal_bottleneck(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        metrics: &PerformanceMetrics,
    ) -> Result<Option<PerformanceBottleneck>> {
        if metrics.throughput < 0.3 {
            // Find entities with complex traversal patterns
            let complex_entities = self.find_complex_traversal_entities(graph).await?;
            
            let bottleneck = PerformanceBottleneck {
                bottleneck_type: BottleneckType::NetworkTraversal,
                severity: 1.0 - metrics.throughput,
                affected_entities: complex_entities,
                suggested_optimization: OptimizationType::HierarchyConsolidation,
                estimated_improvement: 0.15,
                detection_time: Instant::now(),
            };
            
            return Ok(Some(bottleneck));
        }
        
        Ok(None)
    }

    /// Find entities with slow query performance
    async fn find_slow_query_entities(&self, graph: &BrainEnhancedKnowledgeGraph) -> Result<Vec<EntityKey>> {
        let mut slow_entities = Vec::new();
        let entity_keys = graph.get_all_entity_keys();
        
        // Sample entities for performance testing
        let sample_size = (entity_keys.len() / 10).max(1);
        
        for i in (0..entity_keys.len()).step_by(entity_keys.len() / sample_size) {
            let entity_key = entity_keys[i];
            
            // Simulate query time check
            let neighbors = graph.get_neighbors(entity_key).await;
            if neighbors.len() > 100 {
                slow_entities.push(entity_key);
            }
        }
        
        Ok(slow_entities)
    }

    /// Find entities with high memory usage
    async fn find_memory_heavy_entities(&self, graph: &BrainEnhancedKnowledgeGraph) -> Result<Vec<EntityKey>> {
        let mut memory_heavy = Vec::new();
        let entity_keys = graph.get_all_entity_keys();
        
        // Sample entities for memory analysis
        let sample_size = (entity_keys.len() / 20).max(1);
        
        for i in (0..entity_keys.len()).step_by(entity_keys.len() / sample_size) {
            let entity_key = entity_keys[i];
            
            // Check entity data size
            if let Some(entity_data) = graph.get_entity_data(entity_key) {
                if entity_data.properties.len() > 50 {
                    memory_heavy.push(entity_key);
                }
            }
        }
        
        Ok(memory_heavy)
    }

    /// Find entities with poor cache performance
    async fn find_cache_poor_entities(&self, graph: &BrainEnhancedKnowledgeGraph) -> Result<Vec<EntityKey>> {
        let mut cache_poor = Vec::new();
        let entity_keys = graph.get_all_entity_keys();
        
        // Sample entities for cache analysis
        let sample_size = (entity_keys.len() / 15).max(1);
        
        for i in (0..entity_keys.len()).step_by(entity_keys.len() / sample_size) {
            let entity_key = entity_keys[i];
            
            // Simulate cache performance check
            let neighbors = graph.get_neighbors(entity_key).await;
            if neighbors.len() > 50 && neighbors.len() < 200 {
                cache_poor.push(entity_key);
            }
        }
        
        Ok(cache_poor)
    }

    /// Find entities with complex traversal patterns
    async fn find_complex_traversal_entities(&self, graph: &BrainEnhancedKnowledgeGraph) -> Result<Vec<EntityKey>> {
        let mut complex_entities = Vec::new();
        let entity_keys = graph.get_all_entity_keys();
        
        // Sample entities for traversal analysis
        let sample_size = (entity_keys.len() / 25).max(1);
        
        for i in (0..entity_keys.len()).step_by(entity_keys.len() / sample_size) {
            let entity_key = entity_keys[i];
            
            // Check traversal complexity
            let neighbors = graph.get_neighbors(entity_key).await;
            if neighbors.len() > 200 {
                complex_entities.push(entity_key);
            }
        }
        
        Ok(complex_entities)
    }

    /// Get bottleneck severity distribution
    pub fn get_severity_distribution(&self) -> HashMap<BottleneckType, f32> {
        let mut distribution = HashMap::new();
        
        for bottleneck in &self.identified_bottlenecks {
            let current_severity = distribution.get(&bottleneck.bottleneck_type).unwrap_or(&0.0);
            distribution.insert(bottleneck.bottleneck_type.clone(), current_severity + bottleneck.severity);
        }
        
        distribution
    }

    /// Get bottleneck recommendations
    pub fn get_bottleneck_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        for bottleneck in &self.identified_bottlenecks {
            let recommendation = match bottleneck.bottleneck_type {
                BottleneckType::QueryLatency => "Optimize query patterns and consider indexing".to_string(),
                BottleneckType::MemoryUsage => "Implement memory optimization strategies".to_string(),
                BottleneckType::CacheEfficiency => "Improve cache hit rates through better caching strategies".to_string(),
                BottleneckType::NetworkTraversal => "Optimize graph traversal patterns".to_string(),
                BottleneckType::IndexFragmentation => "Rebuild or optimize indices".to_string(),
                BottleneckType::ResourceContention => "Implement better resource management".to_string(),
            };
            
            recommendations.push(recommendation);
        }
        
        recommendations
    }
}

impl Default for EfficiencyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BottleneckDetector {
    fn default() -> Self {
        Self::new()
    }
}