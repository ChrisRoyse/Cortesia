# Task 15h: Implement Validation Reporting

**Time**: 6 minutes
**Dependencies**: 15g_validation_scheduler.md
**Stage**: Inheritance System

## Objective
Create comprehensive reporting system for validation results.

## Implementation
Create `src/inheritance/validation/validation_reporting.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use crate::inheritance::validation::rules::{ValidationResult, ValidationSeverity};
use crate::inheritance::validation::validation_coordinator::{ValidationReport, SystemValidationReport, SystemHealth};

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationReportSummary {
    pub report_id: String,
    pub generation_time: DateTime<Utc>,
    pub validation_period: ValidationPeriod,
    pub overall_status: ValidationStatus,
    pub summary_statistics: SummaryStatistics,
    pub severity_breakdown: SeverityBreakdown,
    pub validator_performance: ValidatorPerformanceMetrics,
    pub recommendations: Vec<ValidationRecommendation>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationPeriod {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_hours: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ValidationStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SummaryStatistics {
    pub total_validations: u64,
    pub concepts_validated: u64,
    pub inheritance_relationships_checked: u64,
    pub total_issues_found: u64,
    pub issues_resolved: u64,
    pub success_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SeverityBreakdown {
    pub critical_count: u64,
    pub error_count: u64,
    pub warning_count: u64,
    pub info_count: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidatorPerformanceMetrics {
    pub structural_validator: ValidatorMetrics,
    pub semantic_validator: ValidatorMetrics,
    pub performance_validator: ValidatorMetrics,
    pub custom_rules: ValidatorMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidatorMetrics {
    pub executions: u64,
    pub average_execution_time_ms: f64,
    pub issues_found: u64,
    pub success_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationRecommendation {
    pub priority: RecommendationPriority,
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub affected_concepts: Vec<String>,
    pub suggested_actions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RecommendationPriority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Security,
    Structure,
    Semantics,
    Maintenance,
}

pub struct ValidationReportGenerator {
    report_history: Vec<ValidationReportSummary>,
    current_period_start: DateTime<Utc>,
}

impl ValidationReportGenerator {
    pub fn new() -> Self {
        Self {
            report_history: Vec::new(),
            current_period_start: Utc::now(),
        }
    }

    pub fn generate_summary_report(
        &mut self,
        validation_results: &[ValidationResult],
        system_reports: &[SystemValidationReport],
        period_hours: f64,
    ) -> ValidationReportSummary {
        let end_time = Utc::now();
        let start_time = self.current_period_start;
        
        let summary_stats = self.calculate_summary_statistics(validation_results, system_reports);
        let severity_breakdown = self.calculate_severity_breakdown(validation_results);
        let validator_performance = self.calculate_validator_performance(validation_results);
        let recommendations = self.generate_recommendations(validation_results, &severity_breakdown);
        
        let overall_status = self.determine_overall_status(&severity_breakdown, &summary_stats);
        
        let report = ValidationReportSummary {
            report_id: uuid::Uuid::new_v4().to_string(),
            generation_time: end_time,
            validation_period: ValidationPeriod {
                start_time,
                end_time,
                duration_hours: period_hours,
            },
            overall_status,
            summary_statistics: summary_stats,
            severity_breakdown,
            validator_performance,
            recommendations,
        };
        
        self.report_history.push(report.clone());
        self.current_period_start = end_time;
        
        report
    }

    fn calculate_summary_statistics(
        &self,
        validation_results: &[ValidationResult],
        system_reports: &[SystemValidationReport],
    ) -> SummaryStatistics {
        let total_validations = validation_results.len() + system_reports.len();
        
        let concepts_validated = validation_results.iter()
            .filter_map(|r| r.concept_id.as_ref())
            .collect::<std::collections::HashSet<_>>()
            .len();
        
        let total_issues = validation_results.len();
        let critical_and_error_issues = validation_results.iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Critical | ValidationSeverity::Error))
            .count();
        
        let success_rate = if total_validations > 0 {
            1.0 - (critical_and_error_issues as f64 / total_validations as f64)
        } else {
            1.0
        };
        
        SummaryStatistics {
            total_validations: total_validations as u64,
            concepts_validated: concepts_validated as u64,
            inheritance_relationships_checked: 0, // Would be tracked separately
            total_issues_found: total_issues as u64,
            issues_resolved: 0, // Would be tracked from previous runs
            success_rate,
        }
    }

    fn calculate_severity_breakdown(&self, validation_results: &[ValidationResult]) -> SeverityBreakdown {
        let mut breakdown = SeverityBreakdown {
            critical_count: 0,
            error_count: 0,
            warning_count: 0,
            info_count: 0,
        };
        
        for result in validation_results {
            match result.severity {
                ValidationSeverity::Critical => breakdown.critical_count += 1,
                ValidationSeverity::Error => breakdown.error_count += 1,
                ValidationSeverity::Warning => breakdown.warning_count += 1,
                ValidationSeverity::Info => breakdown.info_count += 1,
            }
        }
        
        breakdown
    }

    fn calculate_validator_performance(&self, validation_results: &[ValidationResult]) -> ValidatorPerformanceMetrics {
        let mut rule_counts = HashMap::new();
        
        for result in validation_results {
            *rule_counts.entry(result.rule_id.clone()).or_insert(0) += 1;
        }
        
        // Simulate performance metrics (in real implementation, these would be tracked)
        ValidatorPerformanceMetrics {
            structural_validator: ValidatorMetrics {
                executions: rule_counts.get("structural").unwrap_or(&0).clone(),
                average_execution_time_ms: 15.0,
                issues_found: rule_counts.get("structural").unwrap_or(&0).clone(),
                success_rate: 0.95,
            },
            semantic_validator: ValidatorMetrics {
                executions: rule_counts.get("semantic").unwrap_or(&0).clone(),
                average_execution_time_ms: 25.0,
                issues_found: rule_counts.get("semantic").unwrap_or(&0).clone(),
                success_rate: 0.92,
            },
            performance_validator: ValidatorMetrics {
                executions: rule_counts.get("performance").unwrap_or(&0).clone(),
                average_execution_time_ms: 8.0,
                issues_found: rule_counts.get("performance").unwrap_or(&0).clone(),
                success_rate: 0.98,
            },
            custom_rules: ValidatorMetrics {
                executions: rule_counts.get("custom").unwrap_or(&0).clone(),
                average_execution_time_ms: 12.0,
                issues_found: rule_counts.get("custom").unwrap_or(&0).clone(),
                success_rate: 0.90,
            },
        }
    }

    fn generate_recommendations(
        &self,
        validation_results: &[ValidationResult],
        severity_breakdown: &SeverityBreakdown,
    ) -> Vec<ValidationRecommendation> {
        let mut recommendations = Vec::new();
        
        if severity_breakdown.critical_count > 0 {
            recommendations.push(ValidationRecommendation {
                priority: RecommendationPriority::High,
                category: RecommendationCategory::Security,
                title: "Critical Issues Detected".to_string(),
                description: format!("{} critical issues found requiring immediate attention", severity_breakdown.critical_count),
                affected_concepts: self.extract_affected_concepts(validation_results, ValidationSeverity::Critical),
                suggested_actions: vec![
                    "Review and fix critical issues immediately".to_string(),
                    "Run validation again after fixes".to_string(),
                ],
            });
        }
        
        if severity_breakdown.warning_count > 20 {
            recommendations.push(ValidationRecommendation {
                priority: RecommendationPriority::Medium,
                category: RecommendationCategory::Maintenance,
                title: "High Warning Count".to_string(),
                description: format!("{} warnings detected, consider addressing to prevent future issues", severity_breakdown.warning_count),
                affected_concepts: Vec::new(),
                suggested_actions: vec![
                    "Review warning patterns".to_string(),
                    "Implement preventive measures".to_string(),
                ],
            });
        }
        
        // Add performance recommendations based on patterns
        let slow_resolution_count = validation_results.iter()
            .filter(|r| r.rule_id.contains("slow"))
            .count();
        
        if slow_resolution_count > 5 {
            recommendations.push(ValidationRecommendation {
                priority: RecommendationPriority::Medium,
                category: RecommendationCategory::Performance,
                title: "Performance Optimization Needed".to_string(),
                description: "Multiple slow operations detected".to_string(),
                affected_concepts: Vec::new(),
                suggested_actions: vec![
                    "Increase cache size".to_string(),
                    "Optimize inheritance hierarchies".to_string(),
                    "Consider parallel processing".to_string(),
                ],
            });
        }
        
        recommendations
    }

    fn extract_affected_concepts(&self, validation_results: &[ValidationResult], severity: ValidationSeverity) -> Vec<String> {
        validation_results.iter()
            .filter(|r| matches!(r.severity, severity))
            .filter_map(|r| r.concept_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    fn determine_overall_status(&self, severity_breakdown: &SeverityBreakdown, summary_stats: &SummaryStatistics) -> ValidationStatus {
        if severity_breakdown.critical_count > 0 {
            ValidationStatus::Critical
        } else if severity_breakdown.error_count > 10 || summary_stats.success_rate < 0.8 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Healthy
        }
    }

    pub fn export_report_json(&self, report: &ValidationReportSummary) -> Result<String, Box<dyn std::error::Error>> {
        Ok(serde_json::to_string_pretty(report)?)
    }

    pub fn export_report_csv(&self, report: &ValidationReportSummary) -> Result<String, Box<dyn std::error::Error>> {
        let mut csv = String::new();
        csv.push_str("Report ID,Generation Time,Duration Hours,Status,Total Validations,Critical,Errors,Warnings\n");
        csv.push_str(&format!(
            "{},{},{:.2},{:?},{},{},{},{}\n",
            report.report_id,
            report.generation_time,
            report.validation_period.duration_hours,
            report.overall_status,
            report.summary_statistics.total_validations,
            report.severity_breakdown.critical_count,
            report.severity_breakdown.error_count,
            report.severity_breakdown.warning_count
        ));
        
        Ok(csv)
    }

    pub fn get_historical_reports(&self) -> &[ValidationReportSummary] {
        &self.report_history
    }

    pub fn get_trend_analysis(&self, days: u32) -> TrendAnalysis {
        let cutoff_date = Utc::now() - chrono::Duration::days(days as i64);
        let recent_reports: Vec<_> = self.report_history.iter()
            .filter(|r| r.generation_time > cutoff_date)
            .collect();
        
        if recent_reports.is_empty() {
            return TrendAnalysis::default();
        }
        
        let avg_critical = recent_reports.iter()
            .map(|r| r.severity_breakdown.critical_count as f64)
            .sum::<f64>() / recent_reports.len() as f64;
        
        let avg_success_rate = recent_reports.iter()
            .map(|r| r.summary_statistics.success_rate)
            .sum::<f64>() / recent_reports.len() as f64;
        
        TrendAnalysis {
            period_days: days,
            average_critical_issues: avg_critical,
            average_success_rate: avg_success_rate,
            trend_direction: if avg_success_rate > 0.9 { TrendDirection::Improving } else { TrendDirection::Declining },
        }
    }
}

#[derive(Debug, Default)]
pub struct TrendAnalysis {
    pub period_days: u32,
    pub average_critical_issues: f64,
    pub average_success_rate: f64,
    pub trend_direction: TrendDirection,
}

#[derive(Debug, Default)]
pub enum TrendDirection {
    #[default]
    Stable,
    Improving,
    Declining,
}
```

## Success Criteria
- Generates comprehensive validation reports
- Provides trend analysis capabilities
- Exports reports in multiple formats

## Next Task
15i_validation_api.md