//! Performance tracking and metrics for Phase 4 Learning System

use super::types::*;
use std::time::{Duration, SystemTime};

/// Comprehensive performance tracking for Phase 4 learning
#[derive(Debug, Clone)]
pub struct Phase4PerformanceTracker {
    pub learning_metrics: LearningMetrics,
    pub integration_metrics: IntegrationMetrics,
    pub efficiency_metrics: EfficiencyMetrics,
    pub quality_metrics: QualityMetrics,
    pub historical_performance: Vec<PerformanceData>,
}

/// Learning-specific performance metrics
#[derive(Debug, Clone)]
pub struct LearningMetrics {
    pub hebbian_learning_effectiveness: f32,
    pub homeostasis_stability_improvement: f32,
    pub optimization_efficiency_gains: f32,
    pub adaptive_learning_convergence_rate: f32,
    pub overall_learning_quality: f32,
}

/// Integration performance metrics
#[derive(Debug, Clone)]
pub struct IntegrationMetrics {
    pub phase3_compatibility_score: f32,
    pub inter_system_communication_quality: f32,
    pub coordination_effectiveness: f32,
    pub resource_sharing_efficiency: f32,
    pub conflict_resolution_success_rate: f32,
}

/// Efficiency performance metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    pub learning_overhead_percentage: f32,
    pub memory_efficiency_improvement: f32,
    pub computational_efficiency_improvement: f32,
    pub storage_optimization_gains: f32,
    pub query_performance_improvement: f32,
}

/// Quality performance metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub reasoning_quality_improvement: f32,
    pub user_satisfaction_improvement: f32,
    pub error_rate_reduction: f32,
    pub consistency_improvement: f32,
    pub robustness_improvement: f32,
}

/// Historical performance data point
#[derive(Debug, Clone)]
pub struct PerformanceData {
    pub timestamp: SystemTime,
    pub session_id: Option<uuid::Uuid>,
    pub learning_effectiveness: f32,
    pub system_health: f32,
    pub resource_efficiency: f32,
    pub user_satisfaction: f32,
    pub error_rate: f32,
    pub notes: String,
}

impl Phase4PerformanceTracker {
    /// Create new performance tracker
    pub fn new() -> Self {
        Self {
            learning_metrics: LearningMetrics::default(),
            integration_metrics: IntegrationMetrics::default(),
            efficiency_metrics: EfficiencyMetrics::default(),
            quality_metrics: QualityMetrics::default(),
            historical_performance: Vec::new(),
        }
    }
    
    /// Record performance data point
    pub fn record_performance(&mut self, data: PerformanceData) {
        self.historical_performance.push(data);
        
        // Keep only last 1000 data points to prevent unbounded growth
        if self.historical_performance.len() > 1000 {
            self.historical_performance.remove(0);
        }
        
        // Update current metrics based on recent performance
        self.update_current_metrics();
    }
    
    /// Update current metrics based on recent performance history
    fn update_current_metrics(&mut self) {
        if self.historical_performance.is_empty() {
            return;
        }
        
        // Calculate metrics from recent performance (last 10 data points)
        let recent_count = 10.min(self.historical_performance.len());
        let recent_data = &self.historical_performance[self.historical_performance.len() - recent_count..];
        
        // Learning metrics
        let avg_learning_effectiveness: f32 = recent_data.iter()
            .map(|d| d.learning_effectiveness)
            .sum::<f32>() / recent_count as f32;
        
        self.learning_metrics.overall_learning_quality = avg_learning_effectiveness;
        
        // Integration metrics
        let avg_system_health: f32 = recent_data.iter()
            .map(|d| d.system_health)
            .sum::<f32>() / recent_count as f32;
        
        self.integration_metrics.phase3_compatibility_score = avg_system_health;
        
        // Efficiency metrics
        let avg_resource_efficiency: f32 = recent_data.iter()
            .map(|d| d.resource_efficiency)
            .sum::<f32>() / recent_count as f32;
        
        self.efficiency_metrics.memory_efficiency_improvement = avg_resource_efficiency;
        
        // Quality metrics
        let avg_user_satisfaction: f32 = recent_data.iter()
            .map(|d| d.user_satisfaction)
            .sum::<f32>() / recent_count as f32;
        
        let avg_error_rate: f32 = recent_data.iter()
            .map(|d| d.error_rate)
            .sum::<f32>() / recent_count as f32;
        
        self.quality_metrics.user_satisfaction_improvement = avg_user_satisfaction;
        self.quality_metrics.error_rate_reduction = 1.0 - avg_error_rate;
    }
    
    /// Get overall performance score
    pub fn get_overall_score(&self) -> f32 {
        let learning_weight = 0.3;
        let integration_weight = 0.2;
        let efficiency_weight = 0.25;
        let quality_weight = 0.25;
        
        let learning_score = self.learning_metrics.overall_learning_quality;
        let integration_score = self.integration_metrics.phase3_compatibility_score;
        let efficiency_score = self.efficiency_metrics.memory_efficiency_improvement;
        let quality_score = (self.quality_metrics.user_satisfaction_improvement + 
                           self.quality_metrics.error_rate_reduction) / 2.0;
        
        learning_score * learning_weight +
        integration_score * integration_weight +
        efficiency_score * efficiency_weight +
        quality_score * quality_weight
    }
    
    /// Get performance trend over time
    pub fn get_performance_trend(&self, duration: Duration) -> Vec<f32> {
        let cutoff_time = SystemTime::now() - duration;
        
        self.historical_performance.iter()
            .filter(|data| data.timestamp > cutoff_time)
            .map(|data| {
                (data.learning_effectiveness + data.system_health + 
                 data.resource_efficiency + data.user_satisfaction) / 4.0
            })
            .collect()
    }
    
    /// Detect performance anomalies
    pub fn detect_anomalies(&self) -> Vec<String> {
        let mut anomalies = Vec::new();
        
        if self.learning_metrics.overall_learning_quality < 0.3 {
            anomalies.push("Low learning effectiveness detected".to_string());
        }
        
        if self.integration_metrics.coordination_effectiveness < 0.4 {
            anomalies.push("Poor coordination between learning systems".to_string());
        }
        
        if self.efficiency_metrics.learning_overhead_percentage > 0.8 {
            anomalies.push("High learning overhead impacting performance".to_string());
        }
        
        if self.quality_metrics.error_rate_reduction < 0.1 {
            anomalies.push("Error rate not improving with learning".to_string());
        }
        
        anomalies
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!(
            "Phase 4 Learning Performance Report\n\
             =====================================\n\
             Overall Score: {:.2}\n\
             Learning Quality: {:.2}\n\
             Integration Score: {:.2}\n\
             Efficiency Score: {:.2}\n\
             Quality Score: {:.2}\n\
             \n\
             Recent Anomalies: {}\n\
             Data Points: {}",
            self.get_overall_score(),
            self.learning_metrics.overall_learning_quality,
            self.integration_metrics.phase3_compatibility_score,
            self.efficiency_metrics.memory_efficiency_improvement,
            (self.quality_metrics.user_satisfaction_improvement + 
             self.quality_metrics.error_rate_reduction) / 2.0,
            self.detect_anomalies().join(", "),
            self.historical_performance.len()
        )
    }
}

impl Default for LearningMetrics {
    fn default() -> Self {
        Self {
            hebbian_learning_effectiveness: 0.5,
            homeostasis_stability_improvement: 0.5,
            optimization_efficiency_gains: 0.5,
            adaptive_learning_convergence_rate: 0.5,
            overall_learning_quality: 0.5,
        }
    }
}

impl Default for IntegrationMetrics {
    fn default() -> Self {
        Self {
            phase3_compatibility_score: 0.8,
            inter_system_communication_quality: 0.7,
            coordination_effectiveness: 0.6,
            resource_sharing_efficiency: 0.7,
            conflict_resolution_success_rate: 0.8,
        }
    }
}

impl Default for EfficiencyMetrics {
    fn default() -> Self {
        Self {
            learning_overhead_percentage: 0.2,
            memory_efficiency_improvement: 0.5,
            computational_efficiency_improvement: 0.5,
            storage_optimization_gains: 0.5,
            query_performance_improvement: 0.5,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            reasoning_quality_improvement: 0.5,
            user_satisfaction_improvement: 0.5,
            error_rate_reduction: 0.5,
            consistency_improvement: 0.5,
            robustness_improvement: 0.5,
        }
    }
}

impl Default for Phase4PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}