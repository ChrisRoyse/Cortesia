//! Feedback aggregation and processing for adaptive learning

use super::types::*;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use anyhow::Result;
use uuid::Uuid;

/// Feedback aggregation system
#[derive(Debug, Clone)]
pub struct FeedbackAggregator {
    pub user_feedback: Arc<RwLock<VecDeque<UserFeedback>>>,
    pub system_feedback: Arc<RwLock<VecDeque<SystemFeedback>>>,
    pub feedback_config: FeedbackConfig,
}

impl FeedbackAggregator {
    /// Create new feedback aggregator
    pub fn new(config: FeedbackConfig) -> Self {
        Self {
            user_feedback: Arc::new(RwLock::new(VecDeque::new())),
            system_feedback: Arc::new(RwLock::new(VecDeque::new())),
            feedback_config: config,
        }
    }
    
    /// Add user feedback
    pub fn add_user_feedback(&self, feedback: UserFeedback) -> Result<()> {
        let mut user_feedback = self.user_feedback.write().unwrap();
        user_feedback.push_back(feedback);
        
        // Clean up old feedback
        self.cleanup_old_feedback(&mut user_feedback);
        
        Ok(())
    }
    
    /// Add system feedback
    pub fn add_system_feedback(&self, feedback: SystemFeedback) -> Result<()> {
        let mut system_feedback = self.system_feedback.write().unwrap();
        system_feedback.push_back(feedback);
        
        // Clean up old feedback
        self.cleanup_old_system_feedback(&mut system_feedback);
        
        Ok(())
    }
    
    /// Clean up old user feedback
    fn cleanup_old_feedback(&self, feedback: &mut VecDeque<UserFeedback>) {
        let cutoff_time = SystemTime::now() - self.feedback_config.feedback_retention_period;
        
        while let Some(front) = feedback.front() {
            if front.timestamp < cutoff_time {
                feedback.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Clean up old system feedback
    fn cleanup_old_system_feedback(&self, feedback: &mut VecDeque<SystemFeedback>) {
        let cutoff_time = SystemTime::now() - self.feedback_config.feedback_retention_period;
        
        while let Some(front) = feedback.front() {
            if front.timestamp < cutoff_time {
                feedback.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Aggregate feedback for analysis
    pub fn aggregate_feedback(&self) -> Result<AggregatedFeedback> {
        let user_feedback = self.user_feedback.read().unwrap();
        let system_feedback = self.system_feedback.read().unwrap();
        
        // Process user feedback
        let mut weighted_satisfaction = 0.0;
        let mut weighted_accuracy = 0.0;
        let mut weighted_response_quality = 0.0;
        let mut weighted_response_speed = 0.0;
        let mut total_weight = 0.0;
        
        for feedback in user_feedback.iter() {
            let weight = self.calculate_feedback_weight(feedback);
            
            weighted_satisfaction += feedback.satisfaction_score * weight;
            weighted_accuracy += feedback.accuracy_rating * weight;
            weighted_response_quality += feedback.response_quality * weight;
            weighted_response_speed += feedback.response_speed * weight;
            total_weight += weight;
        }
        
        let user_metrics = if total_weight > 0.0 {
            UserFeedbackMetrics {
                avg_satisfaction: weighted_satisfaction / total_weight,
                avg_accuracy: weighted_accuracy / total_weight,
                avg_response_quality: weighted_response_quality / total_weight,
                avg_response_speed: weighted_response_speed / total_weight,
                feedback_count: user_feedback.len(),
            }
        } else {
            UserFeedbackMetrics::default()
        };
        
        // Process system feedback
        let mut system_metrics = std::collections::HashMap::new();
        let mut severity_total = 0.0;
        let mut severity_count = 0;
        
        for feedback in system_feedback.iter() {
            let metric_key = format!("{}_{}", feedback.component, feedback.metric_type);
            system_metrics.insert(metric_key, feedback.value);
            
            severity_total += feedback.severity;
            severity_count += 1;
        }
        
        let avg_severity = if severity_count > 0 {
            severity_total / severity_count as f32
        } else {
            0.0
        };
        
        Ok(AggregatedFeedback {
            user_metrics,
            system_metrics,
            avg_system_severity: avg_severity,
            feedback_timestamp: SystemTime::now(),
        })
    }
    
    /// Calculate weight for feedback based on type and age
    fn calculate_feedback_weight(&self, feedback: &UserFeedback) -> f32 {
        let type_weight = match feedback.feedback_type {
            FeedbackType::Explicit => self.feedback_config.explicit_feedback_weight,
            FeedbackType::Implicit => self.feedback_config.implicit_feedback_weight,
            FeedbackType::System => self.feedback_config.system_feedback_weight,
        };
        
        // Apply temporal decay
        let age = SystemTime::now().duration_since(feedback.timestamp)
            .unwrap_or_default()
            .as_secs() as f32;
        
        let age_hours = age / 3600.0;
        let temporal_weight = self.feedback_config.temporal_decay_factor.powf(age_hours);
        
        type_weight * temporal_weight
    }
    
    /// Analyze user satisfaction trends
    pub async fn analyze_user_satisfaction(&self) -> Result<SatisfactionAnalysis> {
        let user_feedback = self.user_feedback.read().unwrap();
        
        if user_feedback.is_empty() {
            return Ok(SatisfactionAnalysis {
                satisfaction_trends: Vec::new(),
                problem_areas: Vec::new(),
                improvement_opportunities: Vec::new(),
            });
        }
        
        // Extract satisfaction scores
        let satisfaction_scores: Vec<f32> = user_feedback.iter()
            .map(|f| f.satisfaction_score)
            .collect();
        
        // Identify problem areas
        let mut problem_areas = Vec::new();
        
        let avg_response_quality = user_feedback.iter()
            .map(|f| f.response_quality)
            .sum::<f32>() / user_feedback.len() as f32;
        
        if avg_response_quality < 0.6 {
            problem_areas.push("Response Quality".to_string());
        }
        
        let avg_response_speed = user_feedback.iter()
            .map(|f| f.response_speed)
            .sum::<f32>() / user_feedback.len() as f32;
        
        if avg_response_speed < 0.6 {
            problem_areas.push("Response Speed".to_string());
        }
        
        // Identify improvement opportunities
        let mut improvement_opportunities = Vec::new();
        if avg_response_quality < 0.8 {
            improvement_opportunities.push("Enhance response accuracy".to_string());
        }
        if avg_response_speed < 0.8 {
            improvement_opportunities.push("Optimize response time".to_string());
        }
        
        Ok(SatisfactionAnalysis {
            satisfaction_trends: satisfaction_scores,
            problem_areas,
            improvement_opportunities,
        })
    }
    
    /// Correlate performance with user satisfaction
    pub async fn correlate_performance_outcomes(
        &self,
        performance_data: &PerformanceData,
    ) -> Result<CorrelationAnalysis> {
        let user_feedback = self.user_feedback.read().unwrap();
        
        if user_feedback.is_empty() || performance_data.query_latencies.is_empty() {
            return Ok(CorrelationAnalysis {
                performance_satisfaction_correlation: 0.0,
                speed_satisfaction_correlation: 0.0,
                accuracy_satisfaction_correlation: 0.0,
                significant_correlations: Vec::new(),
            });
        }
        
        // Calculate correlations between performance metrics and user satisfaction
        let avg_latency = performance_data.query_latencies.iter()
            .map(|d| d.as_millis() as f32)
            .sum::<f32>() / performance_data.query_latencies.len() as f32;
        
        let _avg_satisfaction = user_feedback.iter()
            .map(|f| f.satisfaction_score)
            .sum::<f32>() / user_feedback.len() as f32;
        
        let avg_accuracy = user_feedback.iter()
            .map(|f| f.accuracy_rating)
            .sum::<f32>() / user_feedback.len() as f32;
        
        // Simple correlation calculations (in real implementation, use proper statistical correlation)
        let speed_satisfaction_correlation = if avg_latency > 300.0 { -0.6 } else { 0.3 };
        let accuracy_satisfaction_correlation = avg_accuracy * 0.8;
        let performance_satisfaction_correlation = (speed_satisfaction_correlation + accuracy_satisfaction_correlation) / 2.0;
        
        let mut significant_correlations = Vec::new();
        if speed_satisfaction_correlation.abs() > 0.5 {
            significant_correlations.push(("Speed".to_string(), "Satisfaction".to_string(), speed_satisfaction_correlation));
        }
        if accuracy_satisfaction_correlation.abs() > 0.5 {
            significant_correlations.push(("Accuracy".to_string(), "Satisfaction".to_string(), accuracy_satisfaction_correlation));
        }
        
        Ok(CorrelationAnalysis {
            performance_satisfaction_correlation,
            speed_satisfaction_correlation,
            accuracy_satisfaction_correlation,
            significant_correlations,
        })
    }
    
    /// Get feedback summary
    pub fn get_feedback_summary(&self) -> Result<FeedbackSummary> {
        let user_feedback = self.user_feedback.read().unwrap();
        let system_feedback = self.system_feedback.read().unwrap();
        
        let user_feedback_count = user_feedback.len();
        let system_feedback_count = system_feedback.len();
        
        let avg_user_satisfaction = if user_feedback_count > 0 {
            user_feedback.iter()
                .map(|f| f.satisfaction_score)
                .sum::<f32>() / user_feedback_count as f32
        } else {
            0.0
        };
        
        let avg_system_severity = if system_feedback_count > 0 {
            system_feedback.iter()
                .map(|f| f.severity)
                .sum::<f32>() / system_feedback_count as f32
        } else {
            0.0
        };
        
        // Count feedback by type
        let mut explicit_count = 0;
        let mut implicit_count = 0;
        let mut system_user_count = 0;
        
        for feedback in user_feedback.iter() {
            match feedback.feedback_type {
                FeedbackType::Explicit => explicit_count += 1,
                FeedbackType::Implicit => implicit_count += 1,
                FeedbackType::System => system_user_count += 1,
            }
        }
        
        Ok(FeedbackSummary {
            total_user_feedback: user_feedback_count,
            total_system_feedback: system_feedback_count,
            avg_user_satisfaction,
            avg_system_severity,
            explicit_feedback_count: explicit_count,
            implicit_feedback_count: implicit_count,
            system_feedback_count: system_user_count,
        })
    }
    
    /// Create user feedback
    pub fn create_user_feedback(
        &self,
        satisfaction_score: f32,
        accuracy_rating: f32,
        response_quality: f32,
        response_speed: f32,
        context: String,
        feedback_type: FeedbackType,
    ) -> UserFeedback {
        UserFeedback {
            feedback_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            feedback_type,
            satisfaction_score,
            accuracy_rating,
            response_quality,
            response_speed,
            context,
            suggestions: Vec::new(),
        }
    }
    
    /// Create system feedback
    pub fn create_system_feedback(
        &self,
        component: String,
        metric_type: String,
        value: f32,
        severity: f32,
        context: std::collections::HashMap<String, String>,
    ) -> SystemFeedback {
        SystemFeedback {
            feedback_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            component,
            metric_type,
            value,
            severity,
            context,
        }
    }
    
    /// Get recent feedback
    pub fn get_recent_feedback(&self, limit: usize) -> (Vec<UserFeedback>, Vec<SystemFeedback>) {
        let user_feedback = self.user_feedback.read().unwrap();
        let system_feedback = self.system_feedback.read().unwrap();
        
        let recent_user: Vec<UserFeedback> = user_feedback.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect();
        
        let recent_system: Vec<SystemFeedback> = system_feedback.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect();
        
        (recent_user, recent_system)
    }
}

/// Aggregated feedback metrics
#[derive(Debug, Clone)]
pub struct AggregatedFeedback {
    pub user_metrics: UserFeedbackMetrics,
    pub system_metrics: std::collections::HashMap<String, f32>,
    pub avg_system_severity: f32,
    pub feedback_timestamp: SystemTime,
}

/// User feedback metrics
#[derive(Debug, Clone)]
pub struct UserFeedbackMetrics {
    pub avg_satisfaction: f32,
    pub avg_accuracy: f32,
    pub avg_response_quality: f32,
    pub avg_response_speed: f32,
    pub feedback_count: usize,
}

/// Feedback summary
#[derive(Debug, Clone)]
pub struct FeedbackSummary {
    pub total_user_feedback: usize,
    pub total_system_feedback: usize,
    pub avg_user_satisfaction: f32,
    pub avg_system_severity: f32,
    pub explicit_feedback_count: usize,
    pub implicit_feedback_count: usize,
    pub system_feedback_count: usize,
}

impl Default for UserFeedbackMetrics {
    fn default() -> Self {
        Self {
            avg_satisfaction: 0.8,
            avg_accuracy: 0.8,
            avg_response_quality: 0.8,
            avg_response_speed: 0.8,
            feedback_count: 0,
        }
    }
}

impl Default for FeedbackAggregator {
    fn default() -> Self {
        Self::new(FeedbackConfig::default())
    }
}