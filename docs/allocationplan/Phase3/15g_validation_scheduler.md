# Task 15g: Implement Validation Scheduler

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 15f_validation_coordinator.md
**Stage**: Inheritance System

## Objective
Create scheduler for automated validation runs and monitoring.

## Implementation
Create `src/inheritance/validation/validation_scheduler.rs`:

```rust
use std::sync::Arc;
use tokio::{task::JoinHandle, time::{interval, Duration}};
use tokio::sync::mpsc;
use chrono::{DateTime, Utc};
use crate::inheritance::validation::validation_coordinator::{ValidationCoordinator, SystemValidationReport};

pub struct ValidationScheduler {
    coordinator: Arc<ValidationCoordinator>,
    config: SchedulerConfig,
    report_sender: mpsc::UnboundedSender<ScheduledValidationReport>,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub system_health_interval_minutes: u64,
    pub individual_concept_interval_hours: u64,
    pub batch_validation_interval_hours: u64,
    pub enable_background_validation: bool,
    pub max_concurrent_validations: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            system_health_interval_minutes: 15,
            individual_concept_interval_hours: 24,
            batch_validation_interval_hours: 6,
            enable_background_validation: true,
            max_concurrent_validations: 10,
        }
    }
}

#[derive(Debug)]
pub struct ScheduledValidationReport {
    pub validation_type: ValidationType,
    pub report: ValidationReportData,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
}

#[derive(Debug)]
pub enum ValidationType {
    SystemHealth,
    ConceptValidation,
    BatchValidation,
}

#[derive(Debug)]
pub enum ValidationReportData {
    SystemHealth(SystemValidationReport),
    Individual { concept_id: String, issues_found: usize },
    Batch { concepts_validated: usize, total_issues: usize },
}

impl ValidationScheduler {
    pub fn new(
        coordinator: Arc<ValidationCoordinator>,
        config: SchedulerConfig,
    ) -> (Self, mpsc::UnboundedReceiver<ScheduledValidationReport>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        let scheduler = Self {
            coordinator,
            config,
            report_sender: sender,
        };
        
        (scheduler, receiver)
    }

    pub fn start(&self) -> Vec<JoinHandle<()>> {
        let mut handles = Vec::new();
        
        if self.config.enable_background_validation {
            // Start system health validation
            handles.push(self.start_system_health_validation());
            
            // Start batch validation
            handles.push(self.start_batch_validation());
        }
        
        handles
    }

    fn start_system_health_validation(&self) -> JoinHandle<()> {
        let coordinator = self.coordinator.clone();
        let sender = self.report_sender.clone();
        let interval_duration = Duration::from_secs(self.config.system_health_interval_minutes * 60);
        
        tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            
            loop {
                interval.tick().await;
                
                let start_time = std::time::Instant::now();
                
                match coordinator.validate_system_health().await {
                    Ok(report) => {
                        let duration_ms = start_time.elapsed().as_millis() as u64;
                        
                        let scheduled_report = ScheduledValidationReport {
                            validation_type: ValidationType::SystemHealth,
                            report: ValidationReportData::SystemHealth(report),
                            timestamp: Utc::now(),
                            duration_ms,
                        };
                        
                        if let Err(e) = sender.send(scheduled_report) {
                            eprintln!("Failed to send system health report: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("System health validation failed: {}", e);
                    }
                }
            }
        })
    }

    fn start_batch_validation(&self) -> JoinHandle<()> {
        let coordinator = self.coordinator.clone();
        let sender = self.report_sender.clone();
        let interval_duration = Duration::from_secs(self.config.batch_validation_interval_hours * 3600);
        
        tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            
            loop {
                interval.tick().await;
                
                let start_time = std::time::Instant::now();
                
                // Get list of concepts to validate (in real implementation, this would query the database)
                let concept_ids = Self::get_concepts_for_validation().await;
                
                match coordinator.validate_batch_concepts(&concept_ids).await {
                    Ok(report) => {
                        let duration_ms = start_time.elapsed().as_millis() as u64;
                        
                        let scheduled_report = ScheduledValidationReport {
                            validation_type: ValidationType::BatchValidation,
                            report: ValidationReportData::Batch {
                                concepts_validated: report.total_concepts,
                                total_issues: report.total_issues,
                            },
                            timestamp: Utc::now(),
                            duration_ms,
                        };
                        
                        if let Err(e) = sender.send(scheduled_report) {
                            eprintln!("Failed to send batch validation report: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("Batch validation failed: {}", e);
                    }
                }
            }
        })
    }

    pub async fn schedule_concept_validation(&self, concept_id: String) -> Result<(), SchedulerError> {
        let coordinator = self.coordinator.clone();
        let sender = self.report_sender.clone();
        
        tokio::spawn(async move {
            let start_time = std::time::Instant::now();
            
            match coordinator.validate_concept(&concept_id).await {
                Ok(report) => {
                    let duration_ms = start_time.elapsed().as_millis() as u64;
                    
                    let scheduled_report = ScheduledValidationReport {
                        validation_type: ValidationType::ConceptValidation,
                        report: ValidationReportData::Individual {
                            concept_id: concept_id.clone(),
                            issues_found: report.total_issues(),
                        },
                        timestamp: Utc::now(),
                        duration_ms,
                    };
                    
                    if let Err(e) = sender.send(scheduled_report) {
                        eprintln!("Failed to send concept validation report: {}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Concept validation failed for {}: {}", concept_id, e);
                }
            }
        });
        
        Ok(())
    }

    pub async fn schedule_immediate_system_check(&self) -> Result<(), SchedulerError> {
        let coordinator = self.coordinator.clone();
        let sender = self.report_sender.clone();
        
        tokio::spawn(async move {
            let start_time = std::time::Instant::now();
            
            match coordinator.validate_system_health().await {
                Ok(report) => {
                    let duration_ms = start_time.elapsed().as_millis() as u64;
                    
                    let scheduled_report = ScheduledValidationReport {
                        validation_type: ValidationType::SystemHealth,
                        report: ValidationReportData::SystemHealth(report),
                        timestamp: Utc::now(),
                        duration_ms,
                    };
                    
                    if let Err(e) = sender.send(scheduled_report) {
                        eprintln!("Failed to send immediate system check report: {}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Immediate system check failed: {}", e);
                }
            }
        });
        
        Ok(())
    }

    async fn get_concepts_for_validation() -> Vec<String> {
        // In a real implementation, this would query the database for concepts
        // For now, return a sample list
        vec![
            "concept_1".to_string(),
            "concept_2".to_string(),
            "concept_3".to_string(),
        ]
    }

    pub fn get_config(&self) -> &SchedulerConfig {
        &self.config
    }
}

#[derive(Debug)]
pub enum SchedulerError {
    ConfigurationError(String),
    ValidationError(String),
    ChannelError(String),
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            SchedulerError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            SchedulerError::ChannelError(msg) => write!(f, "Channel error: {}", msg),
        }
    }
}

impl std::error::Error for SchedulerError {}

// Helper for consuming scheduled validation reports
pub struct ValidationReportConsumer {
    receiver: mpsc::UnboundedReceiver<ScheduledValidationReport>,
}

impl ValidationReportConsumer {
    pub fn new(receiver: mpsc::UnboundedReceiver<ScheduledValidationReport>) -> Self {
        Self { receiver }
    }

    pub async fn start_processing(mut self) {
        while let Some(report) = self.receiver.recv().await {
            self.process_report(report).await;
        }
    }

    async fn process_report(&self, report: ScheduledValidationReport) {
        match report.validation_type {
            ValidationType::SystemHealth => {
                if let ValidationReportData::SystemHealth(health_report) = &report.report {
                    tracing::info!(
                        health = ?health_report.overall_health,
                        total_issues = health_report.total_issues,
                        duration_ms = report.duration_ms,
                        "System health validation completed"
                    );
                    
                    if health_report.critical_issues > 0 {
                        tracing::error!(
                            critical_issues = health_report.critical_issues,
                            "Critical issues detected in system health check"
                        );
                    }
                }
            }
            ValidationType::ConceptValidation => {
                if let ValidationReportData::Individual { concept_id, issues_found } = &report.report {
                    tracing::info!(
                        concept_id = %concept_id,
                        issues_found = %issues_found,
                        duration_ms = report.duration_ms,
                        "Concept validation completed"
                    );
                }
            }
            ValidationType::BatchValidation => {
                if let ValidationReportData::Batch { concepts_validated, total_issues } = &report.report {
                    tracing::info!(
                        concepts_validated = %concepts_validated,
                        total_issues = %total_issues,
                        duration_ms = report.duration_ms,
                        "Batch validation completed"
                    );
                }
            }
        }
    }
}
```

## Success Criteria
- Schedules validation runs automatically
- Handles different validation types
- Reports results through channels

## Next Task
15h_validation_reporting.md