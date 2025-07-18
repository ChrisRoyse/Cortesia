//! Performance tracking for Phase 4 cognitive integration

use super::types::*;
use crate::cognitive::types::CognitivePatternType;
use std::sync::RwLock;
use std::collections::HashMap;
use std::time::SystemTime;
use anyhow::Result;

/// Tracks cognitive performance with learning integration
#[derive(Debug, Clone)]
pub struct CognitivePerformanceTracker {
    pub baseline_performance: PerformanceBaseline,
    pub current_performance: CurrentPerformance,
    pub performance_history: Vec<PerformanceData>,
    pub learning_impact_analysis: LearningImpactAnalysis,
}

impl CognitivePerformanceTracker {
    /// Create new cognitive performance tracker
    pub fn new() -> Self {
        let mut baseline_pattern_times = HashMap::new();
        let mut baseline_quality_scores = HashMap::new();
        
        // Initialize baseline performance for all cognitive patterns
        for pattern in [
            CognitivePatternType::Convergent,
            CognitivePatternType::Divergent,
            CognitivePatternType::Lateral,
            CognitivePatternType::Systems,
            CognitivePatternType::Critical,
            CognitivePatternType::Abstract,
            CognitivePatternType::Adaptive,
        ] {
            baseline_pattern_times.insert(pattern.clone(), std::time::Duration::from_millis(200));
            baseline_quality_scores.insert(pattern, 0.75);
        }
        
        Self {
            baseline_performance: PerformanceBaseline {
                pattern_response_times: baseline_pattern_times.clone(),
                pattern_quality_scores: baseline_quality_scores.clone(),
                overall_satisfaction: 0.75,
                error_rates: HashMap::new(),
                establishment_date: SystemTime::now(),
            },
            current_performance: CurrentPerformance {
                pattern_response_times: baseline_pattern_times,
                pattern_quality_scores: baseline_quality_scores,
                overall_satisfaction: 0.75,
                error_rates: HashMap::new(),
                last_updated: SystemTime::now(),
            },
            performance_history: Vec::new(),
            learning_impact_analysis: LearningImpactAnalysis {
                learning_contributions: HashMap::new(),
                adaptation_effectiveness: 0.8,
                optimization_benefits: HashMap::new(),
                user_satisfaction_improvements: 0.0,
            },
        }
    }
    
    /// Record performance data point
    pub fn record_performance(&mut self, data: PerformanceData) {
        self.performance_history.push(data);
        
        // Keep only last 1000 data points to prevent unbounded growth
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
        
        // Update current performance metrics
        self.update_current_metrics();
    }
    
    /// Update current performance metrics based on recent history
    fn update_current_metrics(&mut self) {
        if self.performance_history.is_empty() {
            return;
        }
        
        // Get recent performance data (last 10 points)
        let recent_count = 10.min(self.performance_history.len());
        let recent_data = &self.performance_history[self.performance_history.len() - recent_count..];
        
        // Update overall satisfaction
        let avg_satisfaction: f32 = recent_data.iter()
            .filter_map(|d| d.user_satisfaction.first())
            .sum::<f32>() / recent_count as f32;
        
        self.current_performance.overall_satisfaction = avg_satisfaction;
        self.current_performance.last_updated = SystemTime::now();
        
        // Update component-specific metrics
        for (component, scores) in recent_data.iter().flat_map(|d| d.component_scores.iter()) {
            if let Ok(pattern_type) = component.parse::<CognitivePatternType>() {
                self.current_performance.pattern_quality_scores.insert(pattern_type, *scores);
            }
        }
    }
    
    /// Get performance improvement since baseline
    pub fn get_performance_improvement(&self) -> f32 {
        self.current_performance.overall_satisfaction - self.baseline_performance.overall_satisfaction
    }
    
    /// Get pattern-specific performance improvements
    pub fn get_pattern_improvements(&self) -> HashMap<CognitivePatternType, f32> {
        let mut improvements = HashMap::new();
        
        for (pattern, &baseline_score) in &self.baseline_performance.pattern_quality_scores {
            if let Some(&current_score) = self.current_performance.pattern_quality_scores.get(pattern) {
                improvements.insert(pattern.clone(), current_score - baseline_score);
            }
        }
        
        improvements
    }
    
    /// Analyze learning impact on performance
    pub fn analyze_learning_impact(&mut self) -> Result<LearningImpactAnalysis> {
        // Calculate learning contributions
        let mut learning_contributions = HashMap::new();
        
        // Analyze hebbian learning contribution
        let hebbian_contribution = self.calculate_hebbian_contribution();
        learning_contributions.insert("hebbian_learning".to_string(), hebbian_contribution);
        
        // Analyze adaptive learning contribution
        let adaptive_contribution = self.calculate_adaptive_contribution();
        learning_contributions.insert("adaptive_learning".to_string(), adaptive_contribution);
        
        // Analyze optimization benefits
        let optimization_benefits = self.calculate_optimization_benefits();
        
        // Calculate overall adaptation effectiveness
        let adaptation_effectiveness = learning_contributions.values().sum::<f32>() / 
                                     learning_contributions.len() as f32;
        
        // Calculate user satisfaction improvements
        let user_satisfaction_improvements = self.get_performance_improvement();
        
        let analysis = LearningImpactAnalysis {
            learning_contributions,
            adaptation_effectiveness,
            optimization_benefits,
            user_satisfaction_improvements,
        };
        
        self.learning_impact_analysis = analysis.clone();
        Ok(analysis)
    }
    
    /// Calculate Hebbian learning contribution to performance
    fn calculate_hebbian_contribution(&self) -> f32 {
        // Simplified calculation based on pattern improvements
        let pattern_improvements = self.get_pattern_improvements();
        
        // Hebbian learning particularly benefits convergent and systems thinking
        let hebbian_patterns = [CognitivePatternType::Convergent, CognitivePatternType::Systems];
        let hebbian_improvement: f32 = hebbian_patterns.iter()
            .filter_map(|pattern| pattern_improvements.get(pattern))
            .sum();
        
        hebbian_improvement / hebbian_patterns.len() as f32
    }
    
    /// Calculate adaptive learning contribution to performance
    fn calculate_adaptive_contribution(&self) -> f32 {
        // Adaptive learning benefits all patterns, but especially adaptive pattern
        let pattern_improvements = self.get_pattern_improvements();
        
        pattern_improvements.get(&CognitivePatternType::Adaptive).unwrap_or(&0.0) * 2.0 +
        pattern_improvements.values().sum::<f32>() / pattern_improvements.len() as f32
    }
    
    /// Calculate optimization benefits
    fn calculate_optimization_benefits(&self) -> HashMap<String, f32> {
        let mut benefits = HashMap::new();
        
        // Calculate response time improvements
        let response_time_improvement = self.calculate_response_time_improvement();
        benefits.insert("response_time".to_string(), response_time_improvement);
        
        // Calculate accuracy improvements
        let accuracy_improvement = self.calculate_accuracy_improvement();
        benefits.insert("accuracy".to_string(), accuracy_improvement);
        
        // Calculate efficiency improvements
        let efficiency_improvement = self.calculate_efficiency_improvement();
        benefits.insert("efficiency".to_string(), efficiency_improvement);
        
        benefits
    }
    
    /// Calculate response time improvement
    fn calculate_response_time_improvement(&self) -> f32 {
        if self.performance_history.len() < 2 {
            return 0.0;
        }
        
        let recent = &self.performance_history[self.performance_history.len() - 1];
        let older = &self.performance_history[self.performance_history.len() / 2];
        
        let recent_avg = recent.throughput_metrics.average_response_time.as_millis() as f32;
        let older_avg = older.throughput_metrics.average_response_time.as_millis() as f32;
        
        if older_avg > 0.0 {
            (older_avg - recent_avg) / older_avg
        } else {
            0.0
        }
    }
    
    /// Calculate accuracy improvement
    fn calculate_accuracy_improvement(&self) -> f32 {
        if self.performance_history.len() < 2 {
            return 0.0;
        }
        
        let recent = &self.performance_history[self.performance_history.len() - 1];
        let older = &self.performance_history[self.performance_history.len() / 2];
        
        let recent_avg = if recent.accuracy_scores.is_empty() { 0.0 } else {
            recent.accuracy_scores.iter().sum::<f32>() / recent.accuracy_scores.len() as f32
        };
        
        let older_avg = if older.accuracy_scores.is_empty() { 0.0 } else {
            older.accuracy_scores.iter().sum::<f32>() / older.accuracy_scores.len() as f32
        };
        
        recent_avg - older_avg
    }
    
    /// Calculate efficiency improvement
    fn calculate_efficiency_improvement(&self) -> f32 {
        if self.performance_history.len() < 2 {
            return 0.0;
        }
        
        let recent = &self.performance_history[self.performance_history.len() - 1];
        let older = &self.performance_history[self.performance_history.len() / 2];
        
        let recent_efficiency = recent.throughput_metrics.queries_per_second;
        let older_efficiency = older.throughput_metrics.queries_per_second;
        
        if older_efficiency > 0.0 {
            (recent_efficiency - older_efficiency) / older_efficiency
        } else {
            0.0
        }
    }
    
    /// Generate performance report
    pub fn generate_performance_report(&self) -> String {
        let improvement = self.get_performance_improvement();
        let pattern_improvements = self.get_pattern_improvements();
        
        let mut report = format!(
            "Phase 4 Cognitive Performance Report\n\
             ===================================\n\
             Overall Satisfaction: {:.2} (baseline: {:.2})\n\
             Performance Improvement: {:.2}\n\
             \n\
             Pattern-Specific Improvements:\n",
            self.current_performance.overall_satisfaction,
            self.baseline_performance.overall_satisfaction,
            improvement
        );
        
        for (pattern, improvement) in pattern_improvements {
            report.push_str(&format!("  {:?}: {:.3}\n", pattern, improvement));
        }
        
        report.push_str(&format!(
            "\nLearning Impact Analysis:\n\
             Adaptation Effectiveness: {:.2}\n\
             User Satisfaction Improvements: {:.2}\n\
             \n\
             Performance History: {} data points\n",
            self.learning_impact_analysis.adaptation_effectiveness,
            self.learning_impact_analysis.user_satisfaction_improvements,
            self.performance_history.len()
        ));
        
        report
    }
    
    /// Check if performance has degraded below threshold
    pub fn check_performance_degradation(&self, threshold: f32) -> bool {
        let improvement = self.get_performance_improvement();
        improvement < -threshold
    }
    
    /// Get performance trends over time
    pub fn get_performance_trends(&self, window_size: usize) -> Vec<f32> {
        if self.performance_history.len() < window_size {
            return Vec::new();
        }
        
        let mut trends = Vec::new();
        
        for i in window_size..self.performance_history.len() {
            let window = &self.performance_history[i - window_size..i];
            let avg_performance = window.iter()
                .map(|d| d.overall_performance_score)
                .sum::<f32>() / window_size as f32;
            
            trends.push(avg_performance);
        }
        
        trends
    }
    
    /// Detect performance anomalies
    pub fn detect_anomalies(&self, threshold: f32) -> Vec<String> {
        let mut anomalies = Vec::new();
        
        // Check for sudden performance drops
        if self.performance_history.len() >= 2 {
            let current = &self.performance_history[self.performance_history.len() - 1];
            let previous = &self.performance_history[self.performance_history.len() - 2];
            
            let performance_drop = previous.overall_performance_score - current.overall_performance_score;
            
            if performance_drop > threshold {
                anomalies.push(format!("Performance drop detected: {:.2}", performance_drop));
            }
        }
        
        // Check for low satisfaction
        if self.current_performance.overall_satisfaction < 0.5 {
            anomalies.push("Low user satisfaction detected".to_string());
        }
        
        // Check for high error rates
        for (error_type, &rate) in &self.current_performance.error_rates {
            if rate > threshold {
                anomalies.push(format!("High error rate in {}: {:.2}", error_type, rate));
            }
        }
        
        anomalies
    }
}

impl Default for CognitivePerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}