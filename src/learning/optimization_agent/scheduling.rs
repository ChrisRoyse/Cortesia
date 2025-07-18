//! Optimization scheduling and execution management

use super::types::*;
use crate::core::types::EntityKey;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

impl OptimizationScheduler {
    /// Create new optimization scheduler
    pub fn new() -> Self {
        Self {
            schedule_config: ScheduleConfig::default(),
            pending_optimizations: Vec::new(),
            execution_history: Vec::new(),
            priority_queue: Vec::new(),
        }
    }

    /// Schedule optimization for execution
    pub fn schedule_optimization(
        &mut self,
        opportunity: &OptimizationOpportunity,
        current_metrics: &PerformanceMetrics,
    ) -> Result<String> {
        let optimization_id = uuid::Uuid::new_v4().to_string();
        
        // Calculate priority
        let priority = self.calculate_priority(opportunity, current_metrics);
        
        // Create scheduled optimization
        let scheduled_optimization = ScheduledOptimization {
            optimization_id: optimization_id.clone(),
            optimization_type: opportunity.optimization_type.clone(),
            priority,
            scheduled_time: Instant::now(),
            estimated_duration: self.estimate_execution_duration(opportunity),
            safety_score: self.calculate_safety_score(opportunity),
            expected_improvement: opportunity.estimated_improvement,
        };
        
        // Add to pending optimizations
        self.pending_optimizations.push(scheduled_optimization.clone());
        
        // Add to priority queue
        let priority_item = PriorityItem {
            optimization_id: optimization_id.clone(),
            priority_score: priority,
            insertion_time: Instant::now(),
        };
        
        self.priority_queue.push(priority_item);
        
        // Sort priority queue
        self.sort_priority_queue();
        
        Ok(optimization_id)
    }

    /// Get next optimization to execute
    pub fn get_next_optimization(&mut self) -> Option<ScheduledOptimization> {
        // Check if we can execute more optimizations
        if self.get_active_optimizations().len() >= self.schedule_config.max_concurrent_optimizations {
            return None;
        }
        
        // Get highest priority optimization
        if let Some(priority_item) = self.priority_queue.pop() {
            // Find and remove from pending optimizations
            let index = self.pending_optimizations.iter()
                .position(|opt| opt.optimization_id == priority_item.optimization_id);
            
            if let Some(index) = index {
                return Some(self.pending_optimizations.remove(index));
            }
        }
        
        None
    }

    /// Record optimization execution
    pub fn record_execution(
        &mut self,
        optimization_id: &str,
        success: bool,
        actual_improvement: f32,
        duration: Duration,
        rollback_required: bool,
    ) {
        let execution_record = ExecutionRecord {
            optimization_id: optimization_id.to_string(),
            execution_time: Instant::now(),
            duration,
            success,
            actual_improvement,
            rollback_required,
        };
        
        self.execution_history.push(execution_record);
        
        // Maintain history size
        if self.execution_history.len() > 1000 {
            self.execution_history.remove(0);
        }
        
        // Remove from pending if still there
        self.pending_optimizations.retain(|opt| opt.optimization_id != optimization_id);
        self.priority_queue.retain(|item| item.optimization_id != optimization_id);
    }

    /// Calculate optimization priority
    fn calculate_priority(
        &self,
        opportunity: &OptimizationOpportunity,
        current_metrics: &PerformanceMetrics,
    ) -> f32 {
        let weights = &self.schedule_config.priority_weights;
        
        // Efficiency gain component
        let efficiency_gain = opportunity.estimated_improvement * weights.efficiency_gain;
        
        // Safety score component
        let safety_score = self.calculate_safety_score(opportunity) * weights.safety_score;
        
        // Execution cost component (inverted - lower cost = higher priority)
        let execution_cost = (1.0 - opportunity.implementation_cost) * weights.execution_cost;
        
        // User impact component
        let user_impact = self.calculate_user_impact(opportunity, current_metrics) * weights.user_impact;
        
        // Urgency multiplier
        let urgency_multiplier = self.calculate_urgency_multiplier(opportunity, current_metrics);
        
        let base_priority = efficiency_gain + safety_score + execution_cost + user_impact;
        
        base_priority * urgency_multiplier
    }

    /// Calculate safety score
    fn calculate_safety_score(&self, opportunity: &OptimizationOpportunity) -> f32 {
        match opportunity.risk_level {
            RiskLevel::Low => 1.0,
            RiskLevel::Medium => 0.8,
            RiskLevel::High => 0.6,
            RiskLevel::Critical => 0.3,
        }
    }

    /// Calculate user impact
    fn calculate_user_impact(&self, opportunity: &OptimizationOpportunity, current_metrics: &PerformanceMetrics) -> f32 {
        let mut impact = 0.0;
        
        // Higher impact if current performance is poor
        if current_metrics.query_latency > Duration::from_millis(300) {
            impact += 0.3;
        }
        
        if current_metrics.memory_usage > 800_000_000 {
            impact += 0.2;
        }
        
        if current_metrics.cache_hit_rate < 0.5 {
            impact += 0.2;
        }
        
        if current_metrics.error_rate > 0.02 {
            impact += 0.3;
        }
        
        // Adjustment based on optimization type
        match opportunity.optimization_type {
            OptimizationType::QueryOptimization => impact += 0.2,
            OptimizationType::MemoryOptimization => impact += 0.15,
            OptimizationType::CacheOptimization => impact += 0.1,
            _ => {},
        }
        
        impact.min(1.0)
    }

    /// Calculate urgency multiplier
    fn calculate_urgency_multiplier(&self, opportunity: &OptimizationOpportunity, current_metrics: &PerformanceMetrics) -> f32 {
        let mut multiplier = 1.0;
        
        // Critical system state
        if current_metrics.resource_utilization > 0.9 {
            multiplier *= 1.5;
        }
        
        if current_metrics.error_rate > 0.05 {
            multiplier *= 1.3;
        }
        
        // High-impact optimizations
        if opportunity.estimated_improvement > 0.5 {
            multiplier *= 1.2;
        }
        
        // Large-scale issues
        if opportunity.affected_entities.len() > 1000 {
            multiplier *= 1.1;
        }
        
        multiplier
    }

    /// Estimate execution duration
    fn estimate_execution_duration(&self, opportunity: &OptimizationOpportunity) -> Duration {
        let base_duration = match opportunity.optimization_type {
            OptimizationType::AttributeBubbling => Duration::from_secs(30),
            OptimizationType::ConnectionPruning => Duration::from_secs(60),
            OptimizationType::HierarchyConsolidation => Duration::from_secs(120),
            OptimizationType::SubgraphFactorization => Duration::from_secs(300),
            OptimizationType::IndexOptimization => Duration::from_secs(180),
            OptimizationType::CacheOptimization => Duration::from_secs(90),
            OptimizationType::QueryOptimization => Duration::from_secs(150),
            OptimizationType::MemoryOptimization => Duration::from_secs(200),
        };
        
        // Scale by number of affected entities
        let entity_factor = (opportunity.affected_entities.len() as f64 / 100.0).max(1.0);
        
        Duration::from_secs((base_duration.as_secs() as f64 * entity_factor) as u64)
    }

    /// Sort priority queue by priority score
    fn sort_priority_queue(&mut self) {
        self.priority_queue.sort_by(|a, b| {
            b.priority_score.partial_cmp(&a.priority_score).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get active optimizations
    fn get_active_optimizations(&self) -> Vec<&ScheduledOptimization> {
        // In a real implementation, this would track currently executing optimizations
        // For now, we'll return empty vector
        Vec::new()
    }

    /// Check if optimization can be executed now
    pub fn can_execute_optimization(&self, optimization_id: &str) -> bool {
        // Check cooldown period
        if let Some(last_execution) = self.execution_history.iter()
            .filter(|record| record.optimization_id == optimization_id)
            .max_by_key(|record| record.execution_time) {
            
            let time_since_last = Instant::now().duration_since(last_execution.execution_time);
            if time_since_last < self.schedule_config.cooldown_period {
                return false;
            }
        }
        
        // Check concurrent optimization limit
        if self.get_active_optimizations().len() >= self.schedule_config.max_concurrent_optimizations {
            return false;
        }
        
        true
    }

    /// Get optimization statistics
    pub fn get_optimization_statistics(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        
        stats.insert("pending_optimizations".to_string(), self.pending_optimizations.len() as f32);
        stats.insert("priority_queue_size".to_string(), self.priority_queue.len() as f32);
        stats.insert("execution_history_size".to_string(), self.execution_history.len() as f32);
        
        if !self.execution_history.is_empty() {
            let successful_executions = self.execution_history.iter().filter(|r| r.success).count();
            let success_rate = successful_executions as f32 / self.execution_history.len() as f32;
            stats.insert("success_rate".to_string(), success_rate);
            
            let avg_duration = self.execution_history.iter()
                .map(|r| r.duration.as_secs_f32())
                .sum::<f32>() / self.execution_history.len() as f32;
            stats.insert("average_duration_seconds".to_string(), avg_duration);
            
            let avg_improvement = self.execution_history.iter()
                .map(|r| r.actual_improvement)
                .sum::<f32>() / self.execution_history.len() as f32;
            stats.insert("average_improvement".to_string(), avg_improvement);
            
            let rollback_rate = self.execution_history.iter().filter(|r| r.rollback_required).count() as f32 / self.execution_history.len() as f32;
            stats.insert("rollback_rate".to_string(), rollback_rate);
        }
        
        stats
    }

    /// Get optimization type distribution
    pub fn get_optimization_type_distribution(&self) -> HashMap<OptimizationType, usize> {
        let mut distribution = HashMap::new();
        
        for optimization in &self.pending_optimizations {
            *distribution.entry(optimization.optimization_type.clone()).or_insert(0) += 1;
        }
        
        distribution
    }

    /// Get recent execution history
    pub fn get_recent_executions(&self, limit: usize) -> Vec<&ExecutionRecord> {
        self.execution_history.iter()
            .rev()
            .take(limit)
            .collect()
    }

    /// Cancel scheduled optimization
    pub fn cancel_optimization(&mut self, optimization_id: &str) -> bool {
        let pending_removed = self.pending_optimizations.retain(|opt| opt.optimization_id != optimization_id);
        let queue_removed = self.priority_queue.retain(|item| item.optimization_id != optimization_id);
        
        pending_removed || queue_removed
    }

    /// Update schedule configuration
    pub fn update_config(&mut self, new_config: ScheduleConfig) {
        self.schedule_config = new_config;
    }

    /// Get current schedule configuration
    pub fn get_config(&self) -> &ScheduleConfig {
        &self.schedule_config
    }

    /// Clear execution history
    pub fn clear_execution_history(&mut self) {
        self.execution_history.clear();
    }

    /// Get optimization by ID
    pub fn get_optimization(&self, optimization_id: &str) -> Option<&ScheduledOptimization> {
        self.pending_optimizations.iter()
            .find(|opt| opt.optimization_id == optimization_id)
    }

    /// Get high priority optimizations
    pub fn get_high_priority_optimizations(&self, threshold: f32) -> Vec<&ScheduledOptimization> {
        self.pending_optimizations.iter()
            .filter(|opt| opt.priority > threshold)
            .collect()
    }

    /// Get overdue optimizations
    pub fn get_overdue_optimizations(&self, max_wait_time: Duration) -> Vec<&ScheduledOptimization> {
        let now = Instant::now();
        self.pending_optimizations.iter()
            .filter(|opt| now.duration_since(opt.scheduled_time) > max_wait_time)
            .collect()
    }

    /// Reschedule optimization with new priority
    pub fn reschedule_optimization(&mut self, optimization_id: &str, new_priority: f32) -> bool {
        // Update in pending optimizations
        if let Some(optimization) = self.pending_optimizations.iter_mut()
            .find(|opt| opt.optimization_id == optimization_id) {
            optimization.priority = new_priority;
        }
        
        // Update in priority queue
        if let Some(queue_item) = self.priority_queue.iter_mut()
            .find(|item| item.optimization_id == optimization_id) {
            queue_item.priority_score = new_priority;
            
            // Re-sort queue
            self.sort_priority_queue();
            return true;
        }
        
        false
    }

    /// Get execution summary
    pub fn get_execution_summary(&self) -> HashMap<String, f32> {
        let mut summary = HashMap::new();
        
        if self.execution_history.is_empty() {
            return summary;
        }
        
        let total_executions = self.execution_history.len() as f32;
        let successful = self.execution_history.iter().filter(|r| r.success).count() as f32;
        let failed = total_executions - successful;
        let rollbacks = self.execution_history.iter().filter(|r| r.rollback_required).count() as f32;
        
        summary.insert("total_executions".to_string(), total_executions);
        summary.insert("successful_executions".to_string(), successful);
        summary.insert("failed_executions".to_string(), failed);
        summary.insert("rollback_executions".to_string(), rollbacks);
        summary.insert("success_percentage".to_string(), (successful / total_executions) * 100.0);
        summary.insert("rollback_percentage".to_string(), (rollbacks / total_executions) * 100.0);
        
        // Calculate total improvement
        let total_improvement = self.execution_history.iter()
            .map(|r| r.actual_improvement)
            .sum::<f32>();
        summary.insert("total_improvement".to_string(), total_improvement);
        
        // Calculate average execution time
        let avg_execution_time = self.execution_history.iter()
            .map(|r| r.duration.as_secs_f32())
            .sum::<f32>() / total_executions;
        summary.insert("average_execution_time".to_string(), avg_execution_time);
        
        summary
    }
}

impl Default for OptimizationScheduler {
    fn default() -> Self {
        Self::new()
    }
}