use crate::error::Result;
use crate::validation::human_loop::{ValidationResult, ValidationItem};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

/// Feedback processor for learning from human validation
pub struct FeedbackProcessor {
    feedback_history: Arc<RwLock<Vec<ValidationFeedback>>>,
    confidence_models: Arc<RwLock<HashMap<String, ConfidenceModel>>>,
    learning_metrics: Arc<RwLock<LearningMetrics>>,
}

impl Default for FeedbackProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeedbackProcessor {
    pub fn new() -> Self {
        Self {
            feedback_history: Arc::new(RwLock::new(Vec::new())),
            confidence_models: Arc::new(RwLock::new(HashMap::new())),
            learning_metrics: Arc::new(RwLock::new(LearningMetrics::default())),
        }
    }

    pub async fn process_feedback(&self, feedback: ValidationFeedback) -> Result<()> {
        // Store feedback
        {
            let mut history = self.feedback_history.write().await;
            history.push(feedback.clone());
        }

        // Update confidence models
        self.update_confidence_models(&feedback).await?;

        // Update learning metrics
        self.update_learning_metrics(&feedback).await?;

        Ok(())
    }

    pub async fn get_confidence_score(&self, item_type: &str, features: &HashMap<String, f32>) -> Result<f32> {
        let models = self.confidence_models.read().await;
        
        if let Some(model) = models.get(item_type) {
            Ok(model.predict(features))
        } else {
            // Default confidence for unknown item types
            Ok(0.5)
        }
    }

    pub async fn get_learning_metrics(&self) -> Result<LearningMetrics> {
        let metrics = self.learning_metrics.read().await;
        Ok(metrics.clone())
    }

    async fn update_confidence_models(&self, feedback: &ValidationFeedback) -> Result<()> {
        let mut models = self.confidence_models.write().await;
        
        // Determine item type from feedback
        let item_type = "triple"; // Simplified - would extract from actual feedback
        
        let model = models.entry(item_type.to_string())
            .or_insert_with(|| ConfidenceModel::new(item_type.to_string()));
        
        // Update model with feedback
        model.update_from_feedback(feedback)?;
        
        Ok(())
    }

    async fn update_learning_metrics(&self, feedback: &ValidationFeedback) -> Result<()> {
        let mut metrics = self.learning_metrics.write().await;
        
        metrics.total_feedback_received += 1;
        
        if feedback.human_validated {
            metrics.positive_feedback += 1;
        } else {
            metrics.negative_feedback += 1;
        }
        
        if feedback.correction_applied {
            metrics.corrections_applied += 1;
        }
        
        // Update accuracy tracking
        let prediction_correct = (feedback.original_confidence > 0.5) == feedback.human_validated;
        if prediction_correct {
            metrics.correct_predictions += 1;
        }
        
        // Update running averages
        metrics.avg_human_confidence = (metrics.avg_human_confidence * (metrics.total_feedback_received - 1) as f32 + feedback.human_confidence) / metrics.total_feedback_received as f32;
        
        Ok(())
    }
}

/// Active learning engine for improving validation models
pub struct ActiveLearningEngine {
    learning_models: Arc<RwLock<HashMap<String, LearningModel>>>,
    uncertainty_samples: Arc<RwLock<Vec<UncertaintySample>>>,
    learning_stats: Arc<RwLock<LearningStats>>,
}

impl Default for ActiveLearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ActiveLearningEngine {
    pub fn new() -> Self {
        Self {
            learning_models: Arc::new(RwLock::new(HashMap::new())),
            uncertainty_samples: Arc::new(RwLock::new(Vec::new())),
            learning_stats: Arc::new(RwLock::new(LearningStats::default())),
        }
    }

    pub async fn update_from_feedback(&self, task_id: &str, result: &ValidationResult) -> Result<()> {
        // Create learning update from validation result
        let learning_update = self.create_learning_update(task_id, result).await?;
        
        // Update relevant models
        self.update_models(&learning_update).await?;
        
        // Update uncertainty tracking
        self.update_uncertainty_tracking(&learning_update).await?;
        
        Ok(())
    }

    pub async fn suggest_next_samples(&self, count: usize) -> Result<Vec<ValidationItem>> {
        let uncertainty_samples = self.uncertainty_samples.read().await;
        
        // Sort by uncertainty score (highest first)
        let mut sorted_samples: Vec<_> = uncertainty_samples.iter().collect();
        sorted_samples.sort_by(|a, b| b.uncertainty_score.partial_cmp(&a.uncertainty_score).unwrap());
        
        // Return top uncertain samples
        let suggested = sorted_samples.into_iter()
            .take(count)
            .map(|sample| sample.item.clone())
            .collect();
        
        Ok(suggested)
    }

    pub async fn get_learning_stats(&self) -> Result<LearningStats> {
        let stats = self.learning_stats.read().await;
        Ok(stats.clone())
    }

    async fn create_learning_update(&self, task_id: &str, result: &ValidationResult) -> Result<LearningUpdate> {
        Ok(LearningUpdate {
            task_id: task_id.to_string(),
            validation_result: result.clone(),
            learning_signal: self.extract_learning_signal(result),
            update_timestamp: Utc::now(),
        })
    }

    async fn update_models(&self, update: &LearningUpdate) -> Result<()> {
        let mut models = self.learning_models.write().await;
        
        // Get or create model for this type of validation
        let model_key = self.get_model_key(&update.validation_result.task_id);
        let model = models.entry(model_key.clone())
            .or_insert_with(|| LearningModel::new(model_key));
        
        // Update model with new data
        model.update(update)?;
        
        Ok(())
    }

    async fn update_uncertainty_tracking(&self, update: &LearningUpdate) -> Result<()> {
        let mut uncertainty_samples = self.uncertainty_samples.write().await;
        
        // Remove the sample that was just validated
        uncertainty_samples.retain(|sample| sample.task_id != update.task_id);
        
        // Update learning stats
        let mut stats = self.learning_stats.write().await;
        stats.total_updates += 1;
        stats.avg_confidence_improvement = (stats.avg_confidence_improvement + update.learning_signal.confidence_delta) / 2.0;
        
        Ok(())
    }

    fn extract_learning_signal(&self, result: &ValidationResult) -> LearningSignal {
        LearningSignal {
            confidence_delta: result.confidence - 0.5, // Assuming 0.5 was original
            validation_outcome: result.is_valid,
            correction_provided: result.corrected_item.is_some(),
            human_notes_provided: result.human_notes.is_some(),
        }
    }

    fn get_model_key(&self, task_id: &str) -> String {
        // Extract model key from task ID or use default
        format!("model_{}", task_id.chars().take(5).collect::<String>())
    }
}

/// Confidence model for predicting validation outcomes
#[derive(Debug, Clone)]
pub struct ConfidenceModel {
    model_type: String,
    feature_weights: HashMap<String, f32>,
    bias: f32,
    sample_count: usize,
}

impl ConfidenceModel {
    pub fn new(model_type: String) -> Self {
        Self {
            model_type,
            feature_weights: HashMap::new(),
            bias: 0.0,
            sample_count: 0,
        }
    }

    pub fn predict(&self, features: &HashMap<String, f32>) -> f32 {
        let mut score = self.bias;
        
        for (feature, value) in features {
            if let Some(weight) = self.feature_weights.get(feature) {
                score += weight * value;
            }
        }
        
        // Apply sigmoid to get probability
        1.0 / (1.0 + (-score).exp())
    }

    pub fn update_from_feedback(&mut self, feedback: &ValidationFeedback) -> Result<()> {
        self.sample_count += 1;
        
        // Simple gradient descent update
        let learning_rate = 0.01;
        let target = if feedback.human_validated { 1.0 } else { 0.0 };
        
        // Create features from feedback
        let features = self.extract_features(feedback);
        let prediction = self.predict(&features);
        let error = target - prediction;
        
        // Update weights
        for (feature, value) in features {
            let weight = self.feature_weights.entry(feature).or_insert(0.0);
            *weight += learning_rate * error * value;
        }
        
        // Update bias
        self.bias += learning_rate * error;
        
        Ok(())
    }

    fn extract_features(&self, feedback: &ValidationFeedback) -> HashMap<String, f32> {
        let mut features = HashMap::new();
        
        // Basic features
        features.insert("original_confidence".to_string(), feedback.original_confidence);
        features.insert("human_confidence".to_string(), feedback.human_confidence);
        features.insert("correction_applied".to_string(), if feedback.correction_applied { 1.0 } else { 0.0 });
        
        // Time-based features
        let hours_since_epoch = feedback.feedback_timestamp.timestamp() as f32 / 3600.0;
        features.insert("time_normalized".to_string(), hours_since_epoch % 24.0 / 24.0);
        
        features
    }
}

/// Learning model for active learning
#[derive(Debug, Clone)]
pub struct LearningModel {
    model_id: String,
    training_samples: Vec<LearningUpdate>,
    model_accuracy: f32,
}

impl LearningModel {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            training_samples: Vec::new(),
            model_accuracy: 0.5,
        }
    }

    pub fn update(&mut self, update: &LearningUpdate) -> Result<()> {
        self.training_samples.push(update.clone());
        
        // Simple accuracy update
        let correct_predictions = self.training_samples.iter()
            .filter(|sample| {
                let predicted_valid = sample.learning_signal.confidence_delta > 0.0;
                predicted_valid == sample.validation_result.is_valid
            })
            .count();
        
        self.model_accuracy = correct_predictions as f32 / self.training_samples.len() as f32;
        
        Ok(())
    }
}

/// Data structures for feedback processing

#[derive(Debug, Clone)]
pub struct ValidationFeedback {
    pub task_id: String,
    pub original_confidence: f32,
    pub human_validated: bool,
    pub human_confidence: f32,
    pub correction_applied: bool,
    pub feedback_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct LearningUpdate {
    pub task_id: String,
    pub validation_result: ValidationResult,
    pub learning_signal: LearningSignal,
    pub update_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct LearningSignal {
    pub confidence_delta: f32,
    pub validation_outcome: bool,
    pub correction_provided: bool,
    pub human_notes_provided: bool,
}

#[derive(Debug, Clone)]
pub struct UncertaintySample {
    pub task_id: String,
    pub item: ValidationItem,
    pub uncertainty_score: f32,
    pub model_predictions: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct LearningMetrics {
    pub total_feedback_received: u64,
    pub positive_feedback: u64,
    pub negative_feedback: u64,
    pub corrections_applied: u64,
    pub correct_predictions: u64,
    pub avg_human_confidence: f32,
}

impl LearningMetrics {
    pub fn accuracy(&self) -> f32 {
        if self.total_feedback_received == 0 {
            0.0
        } else {
            self.correct_predictions as f32 / self.total_feedback_received as f32
        }
    }
    
    pub fn positive_rate(&self) -> f32 {
        if self.total_feedback_received == 0 {
            0.0
        } else {
            self.positive_feedback as f32 / self.total_feedback_received as f32
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LearningStats {
    pub total_updates: u64,
    pub model_count: usize,
    pub avg_confidence_improvement: f32,
    pub uncertainty_samples_count: usize,
}

