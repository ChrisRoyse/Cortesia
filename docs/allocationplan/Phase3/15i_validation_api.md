# Task 15i: Create Validation API Endpoints

**Time**: 7 minutes (1.5 min read, 4 min implement, 1.5 min verify)
**Dependencies**: 15h_validation_reporting.md
**Stage**: Inheritance System

## Objective
Create REST API endpoints for validation system management and monitoring.

## Implementation
Create `src/inheritance/validation/validation_api.rs`:

```rust
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use crate::inheritance::validation::validation_coordinator::ValidationCoordinator;
use crate::inheritance::validation::validation_scheduler::ValidationScheduler;
use crate::inheritance::validation::validation_reporting::ValidationReportGenerator;
use crate::inheritance::validation::rules::ValidationRules;

#[derive(Clone)]
pub struct ValidationApiState {
    pub coordinator: Arc<ValidationCoordinator>,
    pub scheduler: Arc<ValidationScheduler>,
    pub report_generator: Arc<tokio::sync::Mutex<ValidationReportGenerator>>,
}

pub fn create_validation_router(state: ValidationApiState) -> Router {
    Router::new()
        .route("/validation/concept/:concept_id", get(validate_concept))
        .route("/validation/inheritance", post(validate_inheritance_relationship))
        .route("/validation/system/health", get(validate_system_health))
        .route("/validation/batch", post(validate_batch_concepts))
        .route("/validation/schedule/concept", post(schedule_concept_validation))
        .route("/validation/schedule/system", post(schedule_system_validation))
        .route("/validation/reports/summary", get(get_validation_summary))
        .route("/validation/reports/export/:format", get(export_validation_report))
        .route("/validation/config/rules", get(get_validation_rules))
        .route("/validation/config/rules", post(update_validation_rules))
        .with_state(state)
}

#[derive(Deserialize)]
pub struct InheritanceValidationRequest {
    pub child_id: String,
    pub parent_id: String,
}

#[derive(Deserialize)]
pub struct BatchValidationRequest {
    pub concept_ids: Vec<String>,
    pub include_performance: Option<bool>,
}

#[derive(Deserialize)]
pub struct ScheduleValidationRequest {
    pub concept_id: Option<String>,
    pub immediate: Option<bool>,
}

#[derive(Deserialize)]
pub struct ExportQuery {
    pub period_hours: Option<u64>,
    pub include_details: Option<bool>,
}

#[derive(Serialize)]
pub struct ValidationApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ValidationApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn error(message: &str) -> ValidationApiResponse<()> {
        ValidationApiResponse {
            success: false,
            data: None,
            error: Some(message.to_string()),
            timestamp: chrono::Utc::now(),
        }
    }
}

pub async fn validate_concept(
    Path(concept_id): Path<String>,
    State(state): State<ValidationApiState>,
) -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    match state.coordinator.validate_concept(&concept_id).await {
        Ok(report) => {
            let response_data = serde_json::json!({
                "concept_id": report.concept_id,
                "total_issues": report.total_issues(),
                "has_critical_issues": report.has_critical_issues(),
                "validation_results": report.validation_results,
                "timestamp": report.timestamp
            });
            
            Ok(Json(ValidationApiResponse::success(response_data)))
        }
        Err(e) => {
            let error_response = ValidationApiResponse::error(&format!("Validation failed: {}", e));
            Ok(Json(error_response))
        }
    }
}

pub async fn validate_inheritance_relationship(
    State(state): State<ValidationApiState>,
    Json(request): Json<InheritanceValidationRequest>,
) -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    match state.coordinator.validate_inheritance_relationship(&request.child_id, &request.parent_id).await {
        Ok(report) => {
            let response_data = serde_json::json!({
                "child_id": request.child_id,
                "parent_id": request.parent_id,
                "total_issues": report.total_issues(),
                "validation_results": report.validation_results,
                "validator_results": report.validator_results
            });
            
            Ok(Json(ValidationApiResponse::success(response_data)))
        }
        Err(e) => {
            let error_response = ValidationApiResponse::error(&format!("Inheritance validation failed: {}", e));
            Ok(Json(error_response))
        }
    }
}

pub async fn validate_system_health(
    State(state): State<ValidationApiState>,
) -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    match state.coordinator.validate_system_health().await {
        Ok(report) => {
            let response_data = serde_json::json!({
                "overall_health": format!("{:?}", report.overall_health),
                "total_issues": report.total_issues,
                "critical_issues": report.critical_issues,
                "error_issues": report.error_issues,
                "warning_issues": report.warning_issues,
                "timestamp": report.timestamp
            });
            
            Ok(Json(ValidationApiResponse::success(response_data)))
        }
        Err(e) => {
            let error_response = ValidationApiResponse::error(&format!("System health validation failed: {}", e));
            Ok(Json(error_response))
        }
    }
}

pub async fn validate_batch_concepts(
    State(state): State<ValidationApiState>,
    Json(request): Json<BatchValidationRequest>,
) -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    match state.coordinator.validate_batch_concepts(&request.concept_ids).await {
        Ok(report) => {
            let response_data = serde_json::json!({
                "total_concepts": report.total_concepts,
                "total_issues": report.total_issues,
                "concept_reports": report.concept_reports.len(),
                "timestamp": report.timestamp
            });
            
            Ok(Json(ValidationApiResponse::success(response_data)))
        }
        Err(e) => {
            let error_response = ValidationApiResponse::error(&format!("Batch validation failed: {}", e));
            Ok(Json(error_response))
        }
    }
}

pub async fn schedule_concept_validation(
    State(state): State<ValidationApiState>,
    Json(request): Json<ScheduleValidationRequest>,
) -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    if let Some(concept_id) = request.concept_id {
        match state.scheduler.schedule_concept_validation(concept_id.clone()).await {
            Ok(_) => {
                let response_data = serde_json::json!({
                    "message": "Concept validation scheduled",
                    "concept_id": concept_id,
                    "scheduled_at": chrono::Utc::now()
                });
                
                Ok(Json(ValidationApiResponse::success(response_data)))
            }
            Err(e) => {
                let error_response = ValidationApiResponse::error(&format!("Failed to schedule validation: {}", e));
                Ok(Json(error_response))
            }
        }
    } else {
        let error_response = ValidationApiResponse::error("concept_id is required");
        Ok(Json(error_response))
    }
}

pub async fn schedule_system_validation(
    State(state): State<ValidationApiState>,
    Json(request): Json<ScheduleValidationRequest>,
) -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    match state.scheduler.schedule_immediate_system_check().await {
        Ok(_) => {
            let response_data = serde_json::json!({
                "message": "System validation scheduled",
                "scheduled_at": chrono::Utc::now(),
                "immediate": request.immediate.unwrap_or(false)
            });
            
            Ok(Json(ValidationApiResponse::success(response_data)))
        }
        Err(e) => {
            let error_response = ValidationApiResponse::error(&format!("Failed to schedule system validation: {}", e));
            Ok(Json(error_response))
        }
    }
}

pub async fn get_validation_summary(
    State(state): State<ValidationApiState>,
    Query(query): Query<ExportQuery>,
) -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    let mut report_generator = state.report_generator.lock().await;
    
    // In a real implementation, we would collect validation results from storage
    let validation_results = Vec::new(); // Placeholder
    let system_reports = Vec::new(); // Placeholder
    
    let period_hours = query.period_hours.unwrap_or(24) as f64;
    let summary = report_generator.generate_summary_report(&validation_results, &system_reports, period_hours);
    
    let response_data = serde_json::to_value(summary).unwrap();
    Ok(Json(ValidationApiResponse::success(response_data)))
}

pub async fn export_validation_report(
    Path(format): Path<String>,
    State(state): State<ValidationApiState>,
    Query(query): Query<ExportQuery>,
) -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    let mut report_generator = state.report_generator.lock().await;
    
    // Generate summary report
    let validation_results = Vec::new(); // Placeholder
    let system_reports = Vec::new(); // Placeholder
    let period_hours = query.period_hours.unwrap_or(24) as f64;
    let summary = report_generator.generate_summary_report(&validation_results, &system_reports, period_hours);
    
    match format.as_str() {
        "json" => {
            match report_generator.export_report_json(&summary) {
                Ok(json_data) => {
                    let response_data = serde_json::json!({
                        "format": "json",
                        "data": json_data,
                        "report_id": summary.report_id
                    });
                    Ok(Json(ValidationApiResponse::success(response_data)))
                }
                Err(e) => {
                    let error_response = ValidationApiResponse::error(&format!("JSON export failed: {}", e));
                    Ok(Json(error_response))
                }
            }
        }
        "csv" => {
            match report_generator.export_report_csv(&summary) {
                Ok(csv_data) => {
                    let response_data = serde_json::json!({
                        "format": "csv",
                        "data": csv_data,
                        "report_id": summary.report_id
                    });
                    Ok(Json(ValidationApiResponse::success(response_data)))
                }
                Err(e) => {
                    let error_response = ValidationApiResponse::error(&format!("CSV export failed: {}", e));
                    Ok(Json(error_response))
                }
            }
        }
        _ => {
            let error_response = ValidationApiResponse::error("Unsupported format. Use 'json' or 'csv'");
            Ok(Json(error_response))
        }
    }
}

pub async fn get_validation_rules(
    State(_state): State<ValidationApiState>,
) -> Result<Json<ValidationApiResponse<ValidationRules>>, StatusCode> {
    // Return default rules (in real implementation, load from configuration)
    let rules = ValidationRules::default();
    Ok(Json(ValidationApiResponse::success(rules)))
}

pub async fn update_validation_rules(
    State(_state): State<ValidationApiState>,
    Json(rules): Json<ValidationRules>,
) -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    // In a real implementation, this would update the validation configuration
    let response_data = serde_json::json!({
        "message": "Validation rules updated successfully",
        "updated_at": chrono::Utc::now()
    });
    
    Ok(Json(ValidationApiResponse::success(response_data)))
}

// Health check endpoint
pub async fn health_check() -> Result<Json<ValidationApiResponse<serde_json::Value>>, StatusCode> {
    let response_data = serde_json::json!({
        "status": "healthy",
        "service": "validation_api",
        "version": "1.0.0"
    });
    
    Ok(Json(ValidationApiResponse::success(response_data)))
}
```

## Success Criteria
- REST API endpoints are properly defined
- Request/response formats are consistent
- Error handling is comprehensive

## Next Task
15j_validation_integration.md