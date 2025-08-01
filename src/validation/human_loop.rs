use crate::core::triple::Triple;
use crate::error::{GraphError, Result};
use crate::validation::feedback::{FeedbackProcessor, ActiveLearningEngine, ValidationFeedback};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, oneshot};
use chrono::{DateTime, Utc};
use uuid;

/// Human-in-the-loop validation interface
pub struct HumanValidationInterface {
    validation_queue: Arc<RwLock<ValidationQueue>>,
    feedback_processor: Arc<FeedbackProcessor>,
    learning_engine: Arc<ActiveLearningEngine>,
    active_tasks: Arc<RwLock<HashMap<String, ValidationTask>>>,
    notification_sender: Arc<RwLock<Option<mpsc::Sender<ValidationNotification>>>>,
}

impl HumanValidationInterface {
    pub fn new(
        feedback_processor: Arc<FeedbackProcessor>,
        learning_engine: Arc<ActiveLearningEngine>,
    ) -> Self {
        Self {
            validation_queue: Arc::new(RwLock::new(ValidationQueue::new())),
            feedback_processor,
            learning_engine,
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            notification_sender: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn set_notification_channel(&self, sender: mpsc::Sender<ValidationNotification>) {
        let mut notification_sender = self.notification_sender.write().await;
        *notification_sender = Some(sender);
    }

    pub async fn request_validation(&self, item: ValidationItem) -> Result<ValidationResult> {
        let task_id = self.generate_task_id();
        let (response_sender, response_receiver) = oneshot::channel();

        let task = ValidationTask {
            id: task_id.clone(),
            item: item.clone(),
            created_at: Utc::now(),
            priority: self.calculate_priority(&item).await?,
            response_sender: Some(response_sender),
            status: ValidationStatus::Pending,
        };

        // Add to queue
        {
            let mut queue = self.validation_queue.write().await;
            queue.enqueue(task.clone())?;
        }

        // Add to active tasks
        {
            let mut active_tasks = self.active_tasks.write().await;
            active_tasks.insert(task_id.clone(), task);
        }

        // Send notification to human validators
        self.notify_validators(ValidationNotification::NewTask {
            task_id: task_id.clone(),
            item: item.clone(),
            priority: self.calculate_priority(&item).await?,
        }).await;

        // Wait for response or timeout
        match tokio::time::timeout(
            std::time::Duration::from_secs(300), // 5 minute timeout
            response_receiver
        ).await {
            Ok(Ok(result)) => {
                self.process_validation_result(&task_id, &result).await?;
                Ok(result)
            }
            Ok(Err(_)) => {
                // Channel closed without response
                Err(GraphError::ValidationTimeout("Validation task cancelled".to_string()))
            }
            Err(_) => {
                // Timeout
                self.handle_validation_timeout(&task_id).await?;
                Err(GraphError::ValidationTimeout("Validation request timed out".to_string()))
            }
        }
    }

    pub async fn submit_feedback(&self, task_id: &str, feedback: HumanFeedback) -> Result<()> {
        let validation_result = match feedback {
            HumanFeedback::Accept { confidence, notes } => {
                ValidationResult {
                    task_id: task_id.to_string(),
                    is_valid: true,
                    confidence,
                    human_notes: notes,
                    corrected_item: None,
                    timestamp: Utc::now(),
                }
            }
            HumanFeedback::Reject { reason, notes } => {
                ValidationResult {
                    task_id: task_id.to_string(),
                    is_valid: false,
                    confidence: 0.0,
                    human_notes: Some(format!("Rejected: {}. {}", reason, notes.unwrap_or_default())),
                    corrected_item: None,
                    timestamp: Utc::now(),
                }
            }
            HumanFeedback::Correct { corrected_item, notes } => {
                ValidationResult {
                    task_id: task_id.to_string(),
                    is_valid: false,
                    confidence: 0.9,
                    human_notes: notes,
                    corrected_item: Some(corrected_item),
                    timestamp: Utc::now(),
                }
            }
        };

        // Send result to waiting task
        {
            let mut active_tasks = self.active_tasks.write().await;
            if let Some(mut task) = active_tasks.remove(task_id) {
                if let Some(sender) = task.response_sender.take() {
                    let _ = sender.send(validation_result.clone());
                }
            }
        }

        // Process feedback for learning
        self.process_validation_result(task_id, &validation_result).await?;

        Ok(())
    }

    pub async fn get_pending_tasks(&self) -> Result<Vec<ValidationTask>> {
        let queue = self.validation_queue.read().await;
        Ok(queue.get_pending_tasks())
    }

    pub async fn get_task_by_id(&self, task_id: &str) -> Result<Option<ValidationTask>> {
        let active_tasks = self.active_tasks.read().await;
        Ok(active_tasks.get(task_id).cloned())
    }

    pub async fn get_queue_stats(&self) -> Result<ValidationQueueStats> {
        let queue = self.validation_queue.read().await;
        let active_tasks = self.active_tasks.read().await;
        
        Ok(ValidationQueueStats {
            pending_tasks: queue.len(),
            active_tasks: active_tasks.len(),
            total_processed: queue.total_processed,
            avg_processing_time: queue.avg_processing_time,
        })
    }

    pub async fn cancel_task(&self, task_id: &str) -> Result<bool> {
        let mut active_tasks = self.active_tasks.write().await;
        if let Some(mut task) = active_tasks.remove(task_id) {
            task.status = ValidationStatus::Cancelled;
            if let Some(sender) = task.response_sender.take() {
                let _ = sender.send(ValidationResult {
                    task_id: task_id.to_string(),
                    is_valid: false,
                    confidence: 0.0,
                    human_notes: Some("Task cancelled".to_string()),
                    corrected_item: None,
                    timestamp: Utc::now(),
                });
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn calculate_priority(&self, item: &ValidationItem) -> Result<ValidationPriority> {
        match item {
            ValidationItem::Triple(triple) => {
                // High priority for low-confidence triples
                if triple.confidence < 0.5 {
                    Ok(ValidationPriority::High)
                } else if triple.confidence < 0.8 {
                    Ok(ValidationPriority::Medium)
                } else {
                    Ok(ValidationPriority::Low)
                }
            }
            ValidationItem::Conflict { .. } => Ok(ValidationPriority::Critical),
            ValidationItem::EntityLink { confidence, .. } => {
                if *confidence < 0.6 {
                    Ok(ValidationPriority::High)
                } else {
                    Ok(ValidationPriority::Medium)
                }
            }
            ValidationItem::Custom { priority, .. } => Ok(*priority),
        }
    }

    async fn notify_validators(&self, notification: ValidationNotification) {
        let notification_sender = self.notification_sender.read().await;
        if let Some(sender) = notification_sender.as_ref() {
            let _ = sender.send(notification).await;
        }
    }

    async fn process_validation_result(&self, task_id: &str, result: &ValidationResult) -> Result<()> {
        // Convert to feedback for learning
        let feedback = ValidationFeedback {
            task_id: task_id.to_string(),
            original_confidence: 0.5, // Would get from original task
            human_validated: result.is_valid,
            human_confidence: result.confidence,
            correction_applied: result.corrected_item.is_some(),
            feedback_timestamp: result.timestamp,
        };

        // Process feedback
        self.feedback_processor.process_feedback(feedback).await?;

        // Update learning models
        self.learning_engine.update_from_feedback(task_id, result).await?;

        Ok(())
    }

    async fn handle_validation_timeout(&self, task_id: &str) -> Result<()> {
        let mut active_tasks = self.active_tasks.write().await;
        if let Some(mut task) = active_tasks.remove(task_id) {
            task.status = ValidationStatus::Timeout;
            
            // Use default low confidence for timeout
            let default_result = ValidationResult {
                task_id: task_id.to_string(),
                is_valid: false,
                confidence: 0.3,
                human_notes: Some("Validation timed out".to_string()),
                corrected_item: None,
                timestamp: Utc::now(),
            };

            if let Some(sender) = task.response_sender.take() {
                let _ = sender.send(default_result);
            }
        }

        Ok(())
    }

    fn generate_task_id(&self) -> String {
        format!("val_{}", uuid::Uuid::new_v4())
    }
}

/// Validation queue for managing pending tasks
pub struct ValidationQueue {
    queue: VecDeque<ValidationTask>,
    max_size: usize,
    pub total_processed: u64,
    pub avg_processing_time: f64,
}

impl Default for ValidationQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            max_size: 1000,
            total_processed: 0,
            avg_processing_time: 0.0,
        }
    }

    pub fn enqueue(&mut self, task: ValidationTask) -> Result<()> {
        if self.queue.len() >= self.max_size {
            return Err(GraphError::ResourceLimitExceeded(
                "Validation queue is full".to_string()
            ));
        }

        // Insert in priority order
        let insert_pos = self.queue.iter().position(|t| t.priority < task.priority)
            .unwrap_or(self.queue.len());
        
        self.queue.insert(insert_pos, task);
        Ok(())
    }

    pub fn dequeue(&mut self) -> Option<ValidationTask> {
        self.queue.pop_front()
    }

    pub fn get_pending_tasks(&self) -> Vec<ValidationTask> {
        self.queue.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Types of items that can be validated
#[derive(Debug, Clone)]
pub enum ValidationItem {
    Triple(Triple),
    Conflict {
        existing_triple: Triple,
        new_triple: Triple,
        conflict_type: String,
    },
    EntityLink {
        entity1: String,
        entity2: String,
        confidence: f32,
        reason: String,
    },
    Custom {
        data: serde_json::Value,
        description: String,
        priority: ValidationPriority,
    },
}

/// Validation task with metadata
#[derive(Debug)]
pub struct ValidationTask {
    pub id: String,
    pub item: ValidationItem,
    pub created_at: DateTime<Utc>,
    pub priority: ValidationPriority,
    pub response_sender: Option<oneshot::Sender<ValidationResult>>,
    pub status: ValidationStatus,
}

impl Clone for ValidationTask {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            item: self.item.clone(),
            created_at: self.created_at,
            priority: self.priority,
            response_sender: None, // Can't clone oneshot::Sender
            status: self.status.clone(),
        }
    }
}

/// Priority levels for validation tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Status of validation tasks
#[derive(Debug, Clone)]
pub enum ValidationStatus {
    Pending,
    InProgress,
    Completed,
    Cancelled,
    Timeout,
}

/// Result of human validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub task_id: String,
    pub is_valid: bool,
    pub confidence: f32,
    pub human_notes: Option<String>,
    pub corrected_item: Option<ValidationItem>,
    pub timestamp: DateTime<Utc>,
}

/// Human feedback types
#[derive(Debug, Clone)]
pub enum HumanFeedback {
    Accept {
        confidence: f32,
        notes: Option<String>,
    },
    Reject {
        reason: String,
        notes: Option<String>,
    },
    Correct {
        corrected_item: ValidationItem,
        notes: Option<String>,
    },
}

/// Notifications sent to human validators
#[derive(Debug, Clone)]
pub enum ValidationNotification {
    NewTask {
        task_id: String,
        item: ValidationItem,
        priority: ValidationPriority,
    },
    TaskCancelled {
        task_id: String,
    },
    QueueStatus {
        pending_count: usize,
        estimated_wait_time: std::time::Duration,
    },
}

/// Statistics about the validation queue
#[derive(Debug, Clone)]
pub struct ValidationQueueStats {
    pub pending_tasks: usize,
    pub active_tasks: usize,
    pub total_processed: u64,
    pub avg_processing_time: f64,
}

