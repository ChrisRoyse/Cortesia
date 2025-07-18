//! Performance monitoring for adaptive learning system

use super::types::*;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use anyhow::Result;

/// Performance monitoring system
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    pub metrics_collector: Arc<RwLock<MetricsCollector>>,
    pub baseline_performance: Arc<RwLock<PerformanceData>>,
    pub performance_history: Arc<RwLock<VecDeque<PerformanceData>>>,
    pub alert_thresholds: AlertThresholds,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(alert_thresholds: AlertThresholds) -> Self {
        Self {
            metrics_collector: Arc::new(RwLock::new(MetricsCollector::default())),
            baseline_performance: Arc::new(RwLock::new(PerformanceData::default())),
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
            alert_thresholds,
        }
    }
    
    /// Record new performance data
    pub fn record_performance(&self, data: PerformanceData) -> Result<()> {
        let mut history = self.performance_history.write().unwrap();
        history.push_back(data);
        
        // Keep only recent data (last 1000 entries)
        if history.len() > 1000 {
            history.pop_front();
        }
        
        Ok(())
    }
    
    /// Get current performance snapshot
    pub fn get_current_snapshot(&self) -> Result<PerformanceSnapshot> {
        let history = self.performance_history.read().unwrap();
        let metrics = self.metrics_collector.read().unwrap();
        
        if let Some(latest) = history.back() {
            let overall_score = self.calculate_overall_score(latest);
            let component_scores = self.calculate_component_scores(latest, &metrics);
            let bottlenecks = self.identify_current_bottlenecks(latest, &metrics);
            
            Ok(PerformanceSnapshot {
                timestamp: SystemTime::now(),
                overall_performance_score: overall_score,
                component_scores,
                bottlenecks,
                system_health: latest.system_stability,
            })
        } else {
            // Return default snapshot if no data
            Ok(PerformanceSnapshot {
                timestamp: SystemTime::now(),
                overall_performance_score: 0.8,
                component_scores: std::collections::HashMap::new(),
                bottlenecks: Vec::new(),
                system_health: 0.8,
            })
        }
    }
    
    /// Calculate overall performance score
    fn calculate_overall_score(&self, data: &PerformanceData) -> f32 {
        let mut score = 0.0;
        let mut components = 0;
        
        // Latency score
        if !data.query_latencies.is_empty() {
            let avg_latency = data.query_latencies.iter()
                .map(|d| d.as_millis() as f32)
                .sum::<f32>() / data.query_latencies.len() as f32;
            
            let latency_score = if avg_latency < 100.0 {
                1.0
            } else if avg_latency < 500.0 {
                1.0 - (avg_latency - 100.0) / 400.0
            } else {
                0.0
            };
            
            score += latency_score * 0.3;
            components += 1;
        }
        
        // Memory score
        if !data.memory_usage.is_empty() {
            let avg_memory = data.memory_usage.iter().sum::<f32>() / data.memory_usage.len() as f32;
            let memory_score = if avg_memory < 0.8 {
                1.0 - avg_memory
            } else {
                0.2
            };
            
            score += memory_score * 0.2;
            components += 1;
        }
        
        // Accuracy score
        if !data.accuracy_scores.is_empty() {
            let avg_accuracy = data.accuracy_scores.iter().sum::<f32>() / data.accuracy_scores.len() as f32;
            score += avg_accuracy * 0.3;
            components += 1;
        }
        
        // User satisfaction score
        if !data.user_satisfaction.is_empty() {
            let avg_satisfaction = data.user_satisfaction.iter().sum::<f32>() / data.user_satisfaction.len() as f32;
            score += avg_satisfaction * 0.2;
            components += 1;
        }
        
        if components > 0 {
            score
        } else {
            0.8 // Default score
        }
    }
    
    /// Calculate component-specific scores
    fn calculate_component_scores(
        &self,
        data: &PerformanceData,
        metrics: &MetricsCollector,
    ) -> std::collections::HashMap<String, f32> {
        let mut scores = std::collections::HashMap::new();
        
        // Query processing score
        if metrics.query_metrics.total_queries > 0 {
            let success_rate = metrics.query_metrics.successful_queries as f32 / 
                             metrics.query_metrics.total_queries as f32;
            scores.insert("query_processing".to_string(), success_rate);
        }
        
        // Cognitive patterns scores
        for (pattern, &success_rate) in &metrics.cognitive_metrics.pattern_success_rates {
            scores.insert(format!("pattern_{:?}", pattern), success_rate);
        }
        
        // System components scores
        scores.insert("memory_efficiency".to_string(), 1.0 - metrics.system_metrics.memory_usage);
        scores.insert("cpu_efficiency".to_string(), 1.0 - metrics.system_metrics.cpu_utilization);
        scores.insert("storage_efficiency".to_string(), metrics.system_metrics.storage_efficiency);
        
        // User interaction scores
        if !data.user_satisfaction.is_empty() {
            let avg_satisfaction = data.user_satisfaction.iter().sum::<f32>() / data.user_satisfaction.len() as f32;
            scores.insert("user_interaction".to_string(), avg_satisfaction);
        }
        
        scores
    }
    
    /// Identify current bottlenecks
    fn identify_current_bottlenecks(
        &self,
        data: &PerformanceData,
        metrics: &MetricsCollector,
    ) -> Vec<String> {
        let mut bottlenecks = Vec::new();
        
        // Check latency bottlenecks
        if !data.query_latencies.is_empty() {
            let avg_latency = data.query_latencies.iter()
                .map(|d| d.as_millis() as f32)
                .sum::<f32>() / data.query_latencies.len() as f32;
            
            if avg_latency > self.alert_thresholds.latency_threshold.as_millis() as f32 {
                bottlenecks.push("High query latency".to_string());
            }
        }
        
        // Check memory bottlenecks
        if metrics.system_metrics.memory_usage > self.alert_thresholds.memory_threshold {
            bottlenecks.push("High memory usage".to_string());
        }
        
        // Check CPU bottlenecks
        if metrics.system_metrics.cpu_utilization > self.alert_thresholds.cpu_threshold {
            bottlenecks.push("High CPU utilization".to_string());
        }
        
        // Check user satisfaction bottlenecks
        if !data.user_satisfaction.is_empty() {
            let avg_satisfaction = data.user_satisfaction.iter().sum::<f32>() / data.user_satisfaction.len() as f32;
            if avg_satisfaction < self.alert_thresholds.satisfaction_threshold {
                bottlenecks.push("Low user satisfaction".to_string());
            }
        }
        
        // Check error rate bottlenecks
        for (error_type, &rate) in &metrics.system_metrics.error_rates {
            if rate > self.alert_thresholds.error_rate_threshold {
                bottlenecks.push(format!("High {} error rate", error_type));
            }
        }
        
        bottlenecks
    }
    
    /// Get performance trend over time
    pub fn get_performance_trend(&self, window: Duration) -> Result<Vec<f32>> {
        let history = self.performance_history.read().unwrap();
        let cutoff_time = SystemTime::now() - window;
        
        let mut trend = Vec::new();
        
        for data in history.iter() {
            // For trend calculation, we need to estimate timestamp
            // In a real implementation, PerformanceData would have a timestamp field
            let score = self.calculate_overall_score(data);
            trend.push(score);
        }
        
        Ok(trend)
    }
    
    /// Detect performance anomalies
    pub fn detect_anomalies(&self) -> Result<Vec<PerformanceBottleneck>> {
        let snapshot = self.get_current_snapshot()?;
        let mut bottlenecks = Vec::new();
        
        // Check for performance drops
        if snapshot.overall_performance_score < 0.6 {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::Cognitive,
                severity: 1.0 - snapshot.overall_performance_score,
                description: "Overall performance below acceptable threshold".to_string(),
                potential_solutions: vec![
                    "Optimize query processing".to_string(),
                    "Tune cognitive patterns".to_string(),
                    "Adjust system parameters".to_string(),
                ],
            });
        }
        
        // Check for memory bottlenecks
        if let Some(&memory_score) = snapshot.component_scores.get("memory_efficiency") {
            if memory_score < 0.5 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::Memory,
                    severity: 1.0 - memory_score,
                    description: "Memory efficiency below threshold".to_string(),
                    potential_solutions: vec![
                        "Optimize memory allocation".to_string(),
                        "Implement memory pooling".to_string(),
                        "Garbage collection tuning".to_string(),
                    ],
                });
            }
        }
        
        // Check for CPU bottlenecks
        if let Some(&cpu_score) = snapshot.component_scores.get("cpu_efficiency") {
            if cpu_score < 0.5 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::Computation,
                    severity: 1.0 - cpu_score,
                    description: "CPU utilization too high".to_string(),
                    potential_solutions: vec![
                        "Optimize algorithms".to_string(),
                        "Parallelize processing".to_string(),
                        "Reduce computational complexity".to_string(),
                    ],
                });
            }
        }
        
        // Check for user satisfaction bottlenecks
        if let Some(&user_score) = snapshot.component_scores.get("user_interaction") {
            if user_score < 0.7 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::User,
                    severity: 1.0 - user_score,
                    description: "User satisfaction below threshold".to_string(),
                    potential_solutions: vec![
                        "Improve response quality".to_string(),
                        "Reduce response time".to_string(),
                        "Enhance user experience".to_string(),
                    ],
                });
            }
        }
        
        Ok(bottlenecks)
    }
    
    /// Update metrics collector
    pub fn update_metrics(&self, new_metrics: MetricsCollector) -> Result<()> {
        let mut metrics = self.metrics_collector.write().unwrap();
        *metrics = new_metrics;
        Ok(())
    }
    
    /// Get baseline performance
    pub fn get_baseline(&self) -> PerformanceData {
        self.baseline_performance.read().unwrap().clone()
    }
    
    /// Set baseline performance
    pub fn set_baseline(&self, baseline: PerformanceData) -> Result<()> {
        let mut baseline_ref = self.baseline_performance.write().unwrap();
        *baseline_ref = baseline;
        Ok(())
    }
    
    /// Check if alert should be triggered
    pub fn should_trigger_alert(&self, data: &PerformanceData) -> bool {
        // Check latency alerts
        if !data.query_latencies.is_empty() {
            let avg_latency = data.query_latencies.iter()
                .map(|d| d.as_millis() as f32)
                .sum::<f32>() / data.query_latencies.len() as f32;
            
            if avg_latency > self.alert_thresholds.latency_threshold.as_millis() as f32 {
                return true;
            }
        }
        
        // Check satisfaction alerts
        if !data.user_satisfaction.is_empty() {
            let avg_satisfaction = data.user_satisfaction.iter().sum::<f32>() / data.user_satisfaction.len() as f32;
            if avg_satisfaction < self.alert_thresholds.satisfaction_threshold {
                return true;
            }
        }
        
        // Check system stability alerts
        if data.system_stability < 0.7 {
            return true;
        }
        
        false
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> Result<String> {
        let snapshot = self.get_current_snapshot()?;
        let trend = self.get_performance_trend(Duration::from_secs(3600))?;
        let anomalies = self.detect_anomalies()?;
        
        let mut report = String::new();
        
        report.push_str("Adaptive Learning Performance Report\n");
        report.push_str("===================================\n\n");
        
        report.push_str(&format!("Overall Performance Score: {:.2}\n", snapshot.overall_performance_score));
        report.push_str(&format!("System Health: {:.2}\n", snapshot.system_health));
        report.push_str(&format!("Timestamp: {:?}\n\n", snapshot.timestamp));
        
        report.push_str("Component Scores:\n");
        for (component, score) in &snapshot.component_scores {
            report.push_str(&format!("  {}: {:.2}\n", component, score));
        }
        
        report.push_str("\nCurrent Bottlenecks:\n");
        for bottleneck in &snapshot.bottlenecks {
            report.push_str(&format!("  - {}\n", bottleneck));
        }
        
        report.push_str("\nPerformance Anomalies:\n");
        for anomaly in &anomalies {
            report.push_str(&format!("  - {} (severity: {:.2})\n", anomaly.description, anomaly.severity));
        }
        
        report.push_str(&format!("\nPerformance Trend: {} data points\n", trend.len()));
        if trend.len() > 1 {
            let trend_direction = if trend.last().unwrap() > trend.first().unwrap() {
                "improving"
            } else {
                "declining"
            };
            report.push_str(&format!("Trend Direction: {}\n", trend_direction));
        }
        
        Ok(report)
    }
}

impl Default for PerformanceData {
    fn default() -> Self {
        Self {
            query_latencies: Vec::new(),
            memory_usage: Vec::new(),
            accuracy_scores: Vec::new(),
            user_satisfaction: Vec::new(),
            system_stability: 0.8,
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new(AlertThresholds::default())
    }
}