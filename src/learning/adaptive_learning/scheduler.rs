//! Learning task scheduling for adaptive learning system

use super::types::*;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use anyhow::Result;
use uuid::Uuid;

/// Learning task scheduler
#[derive(Debug, Clone)]
pub struct LearningScheduler {
    pub schedule_config: LearningScheduleConfig,
    pub scheduled_tasks: Arc<RwLock<VecDeque<ScheduledLearningTask>>>,
    pub execution_queue: Arc<RwLock<VecDeque<ScheduledLearningTask>>>,
    pub task_history: Arc<RwLock<Vec<CompletedTask>>>,
}

/// Completed task record
#[derive(Debug, Clone)]
pub struct CompletedTask {
    pub task_id: Uuid,
    pub task_type: LearningTaskType,
    pub completion_time: SystemTime,
    pub execution_duration: std::time::Duration,
    pub success: bool,
    pub performance_impact: f32,
}

impl LearningScheduler {
    /// Create new learning scheduler
    pub fn new(config: LearningScheduleConfig) -> Self {
        Self {
            schedule_config: config,
            scheduled_tasks: Arc::new(RwLock::new(VecDeque::new())),
            execution_queue: Arc::new(RwLock::new(VecDeque::new())),
            task_history: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Schedule a learning task
    pub async fn schedule_task(
        &self,
        task_type: LearningTaskType,
        priority: f32,
        scheduled_time: SystemTime,
    ) -> Result<Uuid> {
        let task = ScheduledLearningTask {
            task_id: Uuid::new_v4(),
            task_type: task_type.clone(),
            priority,
            scheduled_time,
            estimated_resources: self.estimate_resources(&task_type),
            dependencies: Vec::new(),
        };
        
        let task_id = task.task_id;
        
        // Insert task in priority order
        let mut scheduled_tasks = self.scheduled_tasks.write().unwrap();
        let insert_position = scheduled_tasks.iter()
            .position(|t| t.priority < priority)
            .unwrap_or(scheduled_tasks.len());
        
        scheduled_tasks.insert(insert_position, task);
        
        Ok(task_id)
    }
    
    /// Schedule emergency task with high priority
    pub async fn schedule_emergency_task(
        &self,
        task_type: LearningTaskType,
        emergency_context: &EmergencyContext,
    ) -> Result<Uuid> {
        let priority = self.calculate_emergency_priority(emergency_context);
        let scheduled_time = SystemTime::now(); // Immediate execution
        
        let task = ScheduledLearningTask {
            task_id: Uuid::new_v4(),
            task_type: task_type.clone(),
            priority,
            scheduled_time,
            estimated_resources: self.estimate_emergency_resources(&task_type, emergency_context),
            dependencies: Vec::new(),
        };
        
        let task_id = task.task_id;
        
        // Insert at front of queue for immediate execution
        let mut execution_queue = self.execution_queue.write().unwrap();
        execution_queue.push_front(task);
        
        Ok(task_id)
    }
    
    /// Calculate priority for emergency task
    fn calculate_emergency_priority(&self, emergency_context: &EmergencyContext) -> f32 {
        let base_priority = match emergency_context.trigger_type {
            EmergencyTrigger::SystemFailure => 1.0,
            EmergencyTrigger::PerformanceCollapse => 0.9,
            EmergencyTrigger::UserExodus => 0.8,
            EmergencyTrigger::ResourceExhaustion => 0.7,
        };
        
        base_priority * emergency_context.severity
    }
    
    /// Estimate resources for task type
    fn estimate_resources(&self, task_type: &LearningTaskType) -> ResourceRequirement {
        match task_type {
            LearningTaskType::EmergencyAdaptation => ResourceRequirement {
                memory_mb: 500.0,
                cpu_cores: 1.0,
                storage_mb: 100.0,
                network_bandwidth_mbps: 5.0,
            },
            LearningTaskType::HebbianLearning => ResourceRequirement {
                memory_mb: 200.0,
                cpu_cores: 0.5,
                storage_mb: 50.0,
                network_bandwidth_mbps: 1.0,
            },
            LearningTaskType::GraphOptimization => ResourceRequirement {
                memory_mb: 300.0,
                cpu_cores: 0.8,
                storage_mb: 75.0,
                network_bandwidth_mbps: 2.0,
            },
            LearningTaskType::ParameterTuning => ResourceRequirement {
                memory_mb: 150.0,
                cpu_cores: 0.3,
                storage_mb: 25.0,
                network_bandwidth_mbps: 0.5,
            },
            LearningTaskType::UserFeedbackIntegration => ResourceRequirement {
                memory_mb: 100.0,
                cpu_cores: 0.2,
                storage_mb: 20.0,
                network_bandwidth_mbps: 0.5,
            },
        }
    }
    
    /// Estimate resources for emergency task
    fn estimate_emergency_resources(&self, task_type: &LearningTaskType, emergency_context: &EmergencyContext) -> ResourceRequirement {
        let base_resources = self.estimate_resources(task_type);
        let emergency_multiplier = 1.0 + emergency_context.severity;
        
        ResourceRequirement {
            memory_mb: base_resources.memory_mb * emergency_multiplier,
            cpu_cores: base_resources.cpu_cores * emergency_multiplier,
            storage_mb: base_resources.storage_mb * emergency_multiplier,
            network_bandwidth_mbps: base_resources.network_bandwidth_mbps * emergency_multiplier,
        }
    }
    
    /// Get next task to execute
    pub fn get_next_task(&self) -> Option<ScheduledLearningTask> {
        // First check execution queue for high-priority tasks
        let mut execution_queue = self.execution_queue.write().unwrap();
        if let Some(task) = execution_queue.pop_front() {
            return Some(task);
        }
        
        // Then check scheduled tasks
        let mut scheduled_tasks = self.scheduled_tasks.write().unwrap();
        let now = SystemTime::now();
        
        // Find first task that's ready to execute
        for i in 0..scheduled_tasks.len() {
            if scheduled_tasks[i].scheduled_time <= now {
                return Some(scheduled_tasks.remove(i).unwrap());
            }
        }
        
        None
    }
    
    /// Check if resources are available for task
    pub fn check_resource_availability(&self, task: &ScheduledLearningTask) -> bool {
        let constraints = &self.schedule_config.resource_constraints;
        
        // Check if task would exceed resource limits
        if task.estimated_resources.memory_mb > constraints.max_memory_usage * 1000.0 {
            return false;
        }
        
        if task.estimated_resources.cpu_cores > constraints.max_cpu_usage {
            return false;
        }
        
        // Check concurrent task limit
        let history = self.task_history.read().unwrap();
        let recent_tasks = history.iter()
            .filter(|t| {
                let recent_time = SystemTime::now() - std::time::Duration::from_secs(3600);
                t.completion_time > recent_time
            })
            .count();
        
        if recent_tasks >= constraints.concurrent_task_limit {
            return false;
        }
        
        true
    }
    
    /// Record task completion
    pub fn record_completion(&self, task: &ScheduledLearningTask, success: bool, performance_impact: f32) -> Result<()> {
        let completed_task = CompletedTask {
            task_id: task.task_id,
            task_type: task.task_type.clone(),
            completion_time: SystemTime::now(),
            execution_duration: std::time::Duration::from_secs(60), // Would be measured
            success,
            performance_impact,
        };
        
        let mut history = self.task_history.write().unwrap();
        history.push(completed_task);
        
        // Keep only recent history (last 1000 tasks)
        if history.len() > 1000 {
            history.remove(0);
        }
        
        Ok(())
    }
    
    /// Get task statistics
    pub fn get_task_statistics(&self) -> TaskStatistics {
        let history = self.task_history.read().unwrap();
        let scheduled_tasks = self.scheduled_tasks.read().unwrap();
        let execution_queue = self.execution_queue.read().unwrap();
        
        let total_completed = history.len();
        let successful_tasks = history.iter().filter(|t| t.success).count();
        let success_rate = if total_completed > 0 {
            successful_tasks as f32 / total_completed as f32
        } else {
            0.0
        };
        
        let avg_performance_impact = if total_completed > 0 {
            history.iter().map(|t| t.performance_impact).sum::<f32>() / total_completed as f32
        } else {
            0.0
        };
        
        // Calculate task type distribution
        let mut task_type_counts = std::collections::HashMap::new();
        for task in history.iter() {
            *task_type_counts.entry(task.task_type.clone()).or_insert(0) += 1;
        }
        
        TaskStatistics {
            total_completed,
            successful_tasks,
            success_rate,
            avg_performance_impact,
            pending_tasks: scheduled_tasks.len(),
            queued_tasks: execution_queue.len(),
            task_type_distribution: task_type_counts,
        }
    }
    
    /// Get next scheduled time for task type
    pub fn get_next_cycle_time(&self, task_type: &LearningTaskType) -> SystemTime {
        let base_frequency = self.schedule_config.base_learning_frequency;
        let priority_weight = self.schedule_config.priority_weights
            .get(task_type)
            .unwrap_or(&0.5);
        
        // Higher priority tasks are scheduled more frequently
        let adjusted_frequency = base_frequency.mul_f32(1.0 / priority_weight);
        
        SystemTime::now() + adjusted_frequency
    }
    
    /// Adaptive scheduling based on system state
    pub fn adaptive_schedule(&self, system_load: f32, performance_score: f32) -> Result<()> {
        if !self.schedule_config.adaptive_scheduling {
            return Ok(());
        }
        
        // Adjust scheduling based on system state
        let mut scheduled_tasks = self.scheduled_tasks.write().unwrap();
        
        for task in scheduled_tasks.iter_mut() {
            // Delay tasks if system is overloaded
            if system_load > 0.8 {
                task.scheduled_time = task.scheduled_time + std::time::Duration::from_secs(300); // 5 minutes
            }
            
            // Expedite tasks if performance is poor
            if performance_score < 0.6 {
                match task.task_type {
                    LearningTaskType::HebbianLearning | LearningTaskType::GraphOptimization => {
                        task.scheduled_time = task.scheduled_time.saturating_sub(std::time::Duration::from_secs(600)); // 10 minutes earlier
                    }
                    _ => {}
                }
            }
        }
        
        Ok(())
    }
    
    /// Cancel scheduled task
    pub fn cancel_task(&self, task_id: Uuid) -> Result<bool> {
        let mut scheduled_tasks = self.scheduled_tasks.write().unwrap();
        
        if let Some(pos) = scheduled_tasks.iter().position(|t| t.task_id == task_id) {
            scheduled_tasks.remove(pos);
            return Ok(true);
        }
        
        let mut execution_queue = self.execution_queue.write().unwrap();
        if let Some(pos) = execution_queue.iter().position(|t| t.task_id == task_id) {
            execution_queue.remove(pos);
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Get pending tasks
    pub fn get_pending_tasks(&self) -> Vec<ScheduledLearningTask> {
        let scheduled_tasks = self.scheduled_tasks.read().unwrap();
        let execution_queue = self.execution_queue.read().unwrap();
        
        let mut pending = Vec::new();
        pending.extend(scheduled_tasks.iter().cloned());
        pending.extend(execution_queue.iter().cloned());
        
        // Sort by priority and scheduled time
        pending.sort_by(|a, b| {
            b.priority.partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.scheduled_time.cmp(&b.scheduled_time))
        });
        
        pending
    }
    
    /// Generate scheduler report
    pub fn generate_report(&self) -> String {
        let stats = self.get_task_statistics();
        let pending_tasks = self.get_pending_tasks();
        
        let mut report = String::new();
        
        report.push_str("Learning Scheduler Report\n");
        report.push_str("========================\n\n");
        
        report.push_str(&format!("Total Completed Tasks: {}\n", stats.total_completed));
        report.push_str(&format!("Successful Tasks: {}\n", stats.successful_tasks));
        report.push_str(&format!("Success Rate: {:.2}%\n", stats.success_rate * 100.0));
        report.push_str(&format!("Average Performance Impact: {:.2}\n", stats.avg_performance_impact));
        report.push_str(&format!("Pending Tasks: {}\n", stats.pending_tasks));
        report.push_str(&format!("Queued Tasks: {}\n", stats.queued_tasks));
        
        report.push_str("\nTask Type Distribution:\n");
        for (task_type, count) in &stats.task_type_distribution {
            report.push_str(&format!("  {:?}: {}\n", task_type, count));
        }
        
        report.push_str("\nNext Scheduled Tasks:\n");
        for (i, task) in pending_tasks.iter().take(5).enumerate() {
            report.push_str(&format!("  {}. {:?} (priority: {:.2})\n", 
                i + 1, task.task_type, task.priority));
        }
        
        report
    }
}

/// Task statistics
#[derive(Debug, Clone)]
pub struct TaskStatistics {
    pub total_completed: usize,
    pub successful_tasks: usize,
    pub success_rate: f32,
    pub avg_performance_impact: f32,
    pub pending_tasks: usize,
    pub queued_tasks: usize,
    pub task_type_distribution: std::collections::HashMap<LearningTaskType, usize>,
}

impl Default for LearningScheduler {
    fn default() -> Self {
        Self::new(LearningScheduleConfig::default())
    }
}