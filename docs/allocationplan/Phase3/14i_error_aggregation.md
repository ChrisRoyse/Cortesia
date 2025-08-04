# Task 14i: Implement Error Aggregation

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 14h_circuit_breaker.md
**Stage**: Inheritance System

## Objective
Create error aggregation system to collect and analyze patterns across inheritance operations.

## Implementation
Create `src/inheritance/error_aggregation.rs`:

```rust
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use crate::inheritance::error_types::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    pub error_sequence: Vec<String>,
    pub frequency: u32,
    pub time_window_minutes: i64,
    pub affected_concepts: Vec<String>,
    pub severity_level: SeverityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    Low,
    Medium,
    High,
    Critical,
}

pub struct ErrorAggregator {
    error_buffer: Arc<RwLock<VecDeque<AggregatedError>>>,
    patterns: Arc<RwLock<HashMap<String, ErrorPattern>>>,
    config: AggregationConfig,
}

#[derive(Debug, Clone)]
pub struct AggregationConfig {
    pub buffer_size: usize,
    pub pattern_detection_window_minutes: i64,
    pub minimum_pattern_frequency: u32,
    pub analysis_interval_minutes: i64,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            pattern_detection_window_minutes: 60,
            minimum_pattern_frequency: 3,
            analysis_interval_minutes: 15,
        }
    }
}

#[derive(Debug, Clone)]
struct AggregatedError {
    pub error_type: String,
    pub concept_id: Option<String>,
    pub property_name: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub context_hash: String,
}

impl ErrorAggregator {
    pub fn new(config: AggregationConfig) -> Self {
        Self {
            error_buffer: Arc::new(RwLock::new(VecDeque::new())),
            patterns: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    pub async fn record_error(&self, error: &InheritanceErrorWithContext) {
        let aggregated = AggregatedError {
            error_type: format!("{:?}", error.error),
            concept_id: error.context.concept_id.clone(),
            property_name: error.context.property_name.clone(),
            timestamp: error.context.timestamp,
            context_hash: self.generate_context_hash(&error.context),
        };

        {
            let mut buffer = self.error_buffer.write().await;
            buffer.push_back(aggregated);
            
            // Maintain buffer size
            if buffer.len() > self.config.buffer_size {
                buffer.pop_front();
            }
        }

        // Trigger pattern analysis if enough time has passed
        self.analyze_patterns().await;
    }

    fn generate_context_hash(&self, context: &ErrorContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        context.operation.hash(&mut hasher);
        context.concept_id.hash(&mut hasher);
        context.property_name.hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }

    async fn analyze_patterns(&self) {
        let now = Utc::now();
        let cutoff_time = now - Duration::minutes(self.config.pattern_detection_window_minutes);
        
        let buffer = self.error_buffer.read().await;
        
        // Filter recent errors
        let recent_errors: Vec<_> = buffer.iter()
            .filter(|e| e.timestamp > cutoff_time)
            .collect();
        
        if recent_errors.len() < self.config.minimum_pattern_frequency as usize {
            return;
        }

        // Analyze sequential patterns
        self.detect_sequential_patterns(&recent_errors).await;
        
        // Analyze concept-based patterns
        self.detect_concept_patterns(&recent_errors).await;
        
        // Analyze time-based patterns
        self.detect_temporal_patterns(&recent_errors).await;
    }

    async fn detect_sequential_patterns(&self, errors: &[&AggregatedError]) {
        let mut sequences = HashMap::new();
        
        // Look for sequences of 2-3 errors
        for window_size in 2..=3 {
            for window in errors.windows(window_size) {
                let sequence: Vec<String> = window.iter()
                    .map(|e| e.error_type.clone())
                    .collect();
                
                let sequence_key = sequence.join(" -> ");
                *sequences.entry(sequence_key.clone()).or_insert(0) += 1;
                
                if sequences[&sequence_key] >= self.config.minimum_pattern_frequency {
                    let pattern = ErrorPattern {
                        error_sequence: sequence,
                        frequency: sequences[&sequence_key],
                        time_window_minutes: self.config.pattern_detection_window_minutes,
                        affected_concepts: self.extract_concepts(window),
                        severity_level: self.calculate_severity(window),
                    };
                    
                    self.patterns.write().await.insert(sequence_key, pattern);
                }
            }
        }
    }

    async fn detect_concept_patterns(&self, errors: &[&AggregatedError]) {
        let mut concept_errors = HashMap::new();
        
        for error in errors {
            if let Some(concept_id) = &error.concept_id {
                concept_errors.entry(concept_id.clone())
                    .or_insert_with(Vec::new)
                    .push(error.error_type.clone());
            }
        }
        
        for (concept_id, error_types) in concept_errors {
            if error_types.len() >= self.config.minimum_pattern_frequency as usize {
                let pattern_key = format!("concept_pattern_{}", concept_id);
                let pattern = ErrorPattern {
                    error_sequence: error_types,
                    frequency: error_types.len() as u32,
                    time_window_minutes: self.config.pattern_detection_window_minutes,
                    affected_concepts: vec![concept_id],
                    severity_level: SeverityLevel::Medium,
                };
                
                self.patterns.write().await.insert(pattern_key, pattern);
            }
        }
    }

    async fn detect_temporal_patterns(&self, errors: &[&AggregatedError]) {
        // Group errors by time buckets (e.g., 5-minute intervals)
        let mut time_buckets = HashMap::new();
        
        for error in errors {
            let bucket = error.timestamp.timestamp() / 300; // 5-minute buckets
            time_buckets.entry(bucket)
                .or_insert_with(Vec::new)
                .push(error.error_type.clone());
        }
        
        // Look for spikes in error frequency
        for (bucket, bucket_errors) in time_buckets {
            if bucket_errors.len() >= (self.config.minimum_pattern_frequency * 2) as usize {
                let pattern_key = format!("temporal_spike_{}", bucket);
                let pattern = ErrorPattern {
                    error_sequence: bucket_errors,
                    frequency: bucket_errors.len() as u32,
                    time_window_minutes: 5,
                    affected_concepts: Vec::new(),
                    severity_level: SeverityLevel::High,
                };
                
                self.patterns.write().await.insert(pattern_key, pattern);
            }
        }
    }

    fn extract_concepts(&self, errors: &[&AggregatedError]) -> Vec<String> {
        errors.iter()
            .filter_map(|e| e.concept_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    fn calculate_severity(&self, errors: &[&AggregatedError]) -> SeverityLevel {
        let has_cycle = errors.iter().any(|e| e.error_type.contains("CycleDetected"));
        let has_database = errors.iter().any(|e| e.error_type.contains("DatabaseError"));
        
        if has_cycle {
            SeverityLevel::Critical
        } else if has_database {
            SeverityLevel::High
        } else if errors.len() > 5 {
            SeverityLevel::Medium
        } else {
            SeverityLevel::Low
        }
    }

    pub async fn get_patterns(&self) -> HashMap<String, ErrorPattern> {
        self.patterns.read().await.clone()
    }

    pub async fn get_critical_patterns(&self) -> Vec<ErrorPattern> {
        self.patterns.read().await
            .values()
            .filter(|p| matches!(p.severity_level, SeverityLevel::Critical | SeverityLevel::High))
            .cloned()
            .collect()
    }

    pub async fn clear_old_patterns(&self, max_age_hours: i64) {
        let cutoff = Utc::now() - Duration::hours(max_age_hours);
        
        // In a real implementation, patterns would have timestamps
        // For now, just clear all patterns periodically
        if max_age_hours <= 24 {
            self.patterns.write().await.clear();
        }
    }
}
```

## Success Criteria
- Error patterns are detected automatically
- Pattern severity is correctly calculated
- Aggregation doesn't impact performance

## Next Task
14j_error_notification.md