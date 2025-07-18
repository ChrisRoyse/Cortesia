//! Quality Assessment for E2E Simulations
//! 
//! Comprehensive quality assessment systems for evaluating the overall quality
//! and effectiveness of end-to-end simulation testing.

use super::simulation_environment::{E2ESimulationEnvironment, WorkflowResult};
use super::workflow_validators::{WorkflowValidationResult, ValidationSummary};
use super::performance_monitors::{PerformanceSummary, PerformanceTrends};
use super::health_monitors::{HealthReport, SystemHealthMetrics};
use anyhow::{Result, anyhow};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Overall simulation quality assessment
#[derive(Debug, Clone)]
pub struct SimulationQualityAssessment {
    pub assessment_id: String,
    pub simulation_name: String,
    pub overall_quality_score: f64,
    pub quality_grade: QualityGrade,
    pub assessment_dimensions: QualityDimensions,
    pub recommendations: Vec<QualityRecommendation>,
    pub assessment_timestamp: Instant,
    pub assessment_duration: Duration,
}

/// Quality assessment dimensions
#[derive(Debug, Clone)]
pub struct QualityDimensions {
    pub functional_quality: f64,
    pub performance_quality: f64,
    pub reliability_quality: f64,
    pub scalability_quality: f64,
    pub maintainability_quality: f64,
    pub usability_quality: f64,
    pub security_quality: f64,
    pub efficiency_quality: f64,
}

/// Quality grade levels
#[derive(Debug, Clone, PartialEq)]
pub enum QualityGrade {
    Excellent,  // 95-100%
    Good,       // 85-94%
    Acceptable, // 70-84%
    Poor,       // 50-69%
    Failing,    // Below 50%
}

/// Quality improvement recommendations
#[derive(Debug, Clone)]
pub struct QualityRecommendation {
    pub priority: RecommendationPriority,
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub impact_level: ImpactLevel,
    pub effort_estimate: EffortEstimate,
    pub implementation_steps: Vec<String>,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Recommendation categories
#[derive(Debug, Clone)]
pub enum RecommendationCategory {
    Performance,
    Reliability,
    Scalability,
    Security,
    Maintainability,
    Testing,
    Documentation,
    Monitoring,
}

/// Expected impact of implementing recommendation
#[derive(Debug, Clone)]
pub enum ImpactLevel {
    High,    // Significant improvement expected
    Medium,  // Moderate improvement expected
    Low,     // Minor improvement expected
}

/// Effort required to implement recommendation
#[derive(Debug, Clone)]
pub enum EffortEstimate {
    Low,     // 1-3 days
    Medium,  // 1-2 weeks
    High,    // 2-4 weeks
    VeryHigh, // 1+ months
}

/// Comprehensive quality assessor
pub struct SimulationQualityAssessor {
    assessment_criteria: QualityAssessmentCriteria,
    benchmarks: QualityBenchmarks,
}

/// Quality assessment criteria configuration
#[derive(Debug, Clone)]
pub struct QualityAssessmentCriteria {
    pub min_functional_score: f64,
    pub min_performance_score: f64,
    pub min_reliability_score: f64,
    pub min_scalability_score: f64,
    pub max_error_rate: f64,
    pub min_uptime: f64,
    pub max_response_time: Duration,
}

/// Quality benchmarks for comparison
#[derive(Debug, Clone)]
pub struct QualityBenchmarks {
    pub excellent_threshold: f64,
    pub good_threshold: f64,
    pub acceptable_threshold: f64,
    pub poor_threshold: f64,
    pub industry_standards: IndustryStandards,
}

/// Industry standard benchmarks
#[derive(Debug, Clone)]
pub struct IndustryStandards {
    pub enterprise_availability: f64,
    pub web_service_response_time: Duration,
    pub database_query_latency: Duration,
    pub api_throughput_qps: f64,
    pub memory_efficiency: f64,
}

impl SimulationQualityAssessor {
    pub fn new() -> Self {
        Self {
            assessment_criteria: QualityAssessmentCriteria::default(),
            benchmarks: QualityBenchmarks::default(),
        }
    }

    pub fn with_custom_criteria(criteria: QualityAssessmentCriteria) -> Self {
        Self {
            assessment_criteria: criteria,
            benchmarks: QualityBenchmarks::default(),
        }
    }

    /// Perform comprehensive quality assessment
    pub fn assess_simulation_quality(
        &self,
        simulation_name: &str,
        validation_results: &[WorkflowValidationResult],
        performance_summary: &PerformanceSummary,
        health_report: &HealthReport,
        trends: &PerformanceTrends,
    ) -> Result<SimulationQualityAssessment> {
        let start_time = Instant::now();
        let assessment_id = format!("qa_{}_{}", simulation_name, start_time.elapsed().as_millis());

        // Calculate quality dimensions
        let dimensions = self.calculate_quality_dimensions(
            validation_results, 
            performance_summary, 
            health_report, 
            trends
        )?;

        // Calculate overall quality score
        let overall_score = self.calculate_overall_quality_score(&dimensions);
        
        // Determine quality grade
        let quality_grade = self.determine_quality_grade(overall_score);

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &dimensions, 
            validation_results, 
            performance_summary, 
            health_report
        );

        Ok(SimulationQualityAssessment {
            assessment_id,
            simulation_name: simulation_name.to_string(),
            overall_quality_score: overall_score,
            quality_grade,
            assessment_dimensions: dimensions,
            recommendations,
            assessment_timestamp: start_time,
            assessment_duration: start_time.elapsed(),
        })
    }

    /// Calculate quality dimensions scores
    fn calculate_quality_dimensions(
        &self,
        validation_results: &[WorkflowValidationResult],
        performance_summary: &PerformanceSummary,
        health_report: &HealthReport,
        trends: &PerformanceTrends,
    ) -> Result<QualityDimensions> {
        // Functional quality based on validation results
        let functional_quality = if validation_results.is_empty() {
            0.5
        } else {
            validation_results.iter()
                .map(|r| r.validation_score)
                .sum::<f64>() / validation_results.len() as f64
        };

        // Performance quality based on performance metrics
        let performance_quality = self.calculate_performance_quality(performance_summary, trends);

        // Reliability quality based on health and success metrics
        let reliability_quality = self.calculate_reliability_quality(health_report, performance_summary);

        // Scalability quality based on trend analysis
        let scalability_quality = self.calculate_scalability_quality(trends, performance_summary);

        // Maintainability quality based on code and architecture quality
        let maintainability_quality = self.calculate_maintainability_quality(validation_results);

        // Usability quality based on API and interface design
        let usability_quality = self.calculate_usability_quality(validation_results, performance_summary);

        // Security quality based on error handling and data protection
        let security_quality = self.calculate_security_quality(validation_results, health_report);

        // Efficiency quality based on resource utilization
        let efficiency_quality = self.calculate_efficiency_quality(performance_summary);

        Ok(QualityDimensions {
            functional_quality,
            performance_quality,
            reliability_quality,
            scalability_quality,
            maintainability_quality,
            usability_quality,
            security_quality,
            efficiency_quality,
        })
    }

    fn calculate_performance_quality(&self, performance: &PerformanceSummary, trends: &PerformanceTrends) -> f64 {
        let mut score = 0.80; // Base score

        // Adjust based on execution time
        if performance.avg_execution_time <= Duration::from_secs(1) {
            score += 0.15;
        } else if performance.avg_execution_time <= Duration::from_secs(5) {
            score += 0.10;
        } else if performance.avg_execution_time > Duration::from_secs(30) {
            score -= 0.20;
        }

        // Adjust based on throughput
        if performance.current_throughput >= 100.0 {
            score += 0.10;
        } else if performance.current_throughput < 10.0 {
            score -= 0.15;
        }

        // Adjust based on success rate
        if performance.overall_success_rate >= 0.99 {
            score += 0.05;
        } else if performance.overall_success_rate < 0.95 {
            score -= 0.10;
        }

        // Consider performance trends
        match trends.throughput_trend {
            super::performance_monitors::TrendDirection::Increasing => score += 0.05,
            super::performance_monitors::TrendDirection::Decreasing => score -= 0.10,
            _ => {}
        }

        score.max(0.0).min(1.0)
    }

    fn calculate_reliability_quality(&self, health: &HealthReport, performance: &PerformanceSummary) -> f64 {
        let mut score = 0.75; // Base score

        // Health-based adjustments
        if health.uptime_percentage >= 99.9 {
            score += 0.20;
        } else if health.uptime_percentage >= 99.0 {
            score += 0.10;
        } else if health.uptime_percentage < 95.0 {
            score -= 0.25;
        }

        // System stability adjustments
        if health.system_stability_score >= 0.95 {
            score += 0.15;
        } else if health.system_stability_score < 0.80 {
            score -= 0.20;
        }

        // Success rate adjustments
        if performance.overall_success_rate >= 0.999 {
            score += 0.10;
        } else if performance.overall_success_rate < 0.90 {
            score -= 0.30;
        }

        score.max(0.0).min(1.0)
    }

    fn calculate_scalability_quality(&self, trends: &PerformanceTrends, performance: &PerformanceSummary) -> f64 {
        let mut score = 0.70; // Base score

        // Throughput trend analysis
        match trends.throughput_trend {
            super::performance_monitors::TrendDirection::Increasing => score += 0.20,
            super::performance_monitors::TrendDirection::Stable => score += 0.10,
            super::performance_monitors::TrendDirection::Decreasing => score -= 0.15,
        }

        // Memory usage trend analysis
        match trends.memory_usage_trend {
            super::performance_monitors::TrendDirection::Increasing => score -= 0.15,
            super::performance_monitors::TrendDirection::Stable => score += 0.10,
            super::performance_monitors::TrendDirection::Decreasing => score += 0.05,
        }

        // Resource efficiency consideration
        if performance.resource_efficiency >= 0.90 {
            score += 0.15;
        } else if performance.resource_efficiency < 0.70 {
            score -= 0.20;
        }

        score.max(0.0).min(1.0)
    }

    fn calculate_maintainability_quality(&self, validation_results: &[WorkflowValidationResult]) -> f64 {
        let mut score = 0.75; // Base score

        // Check for consistent quality across workflows
        if !validation_results.is_empty() {
            let scores: Vec<f64> = validation_results.iter().map(|r| r.validation_score).collect();
            let variance = self.calculate_variance(&scores);
            
            if variance < 0.01 { // Low variance indicates consistency
                score += 0.15;
            } else if variance > 0.05 {
                score -= 0.10;
            }

            // Check for error handling quality
            let avg_error_handling: f64 = validation_results.iter()
                .map(|r| r.metrics.error_handling_score)
                .sum::<f64>() / validation_results.len() as f64;
            
            if avg_error_handling >= 0.90 {
                score += 0.10;
            } else if avg_error_handling < 0.70 {
                score -= 0.15;
            }
        }

        score.max(0.0).min(1.0)
    }

    fn calculate_usability_quality(&self, validation_results: &[WorkflowValidationResult], performance: &PerformanceSummary) -> f64 {
        let mut score = 0.80; // Base score

        // Response time affects usability
        if performance.avg_execution_time <= Duration::from_secs(2) {
            score += 0.15;
        } else if performance.avg_execution_time > Duration::from_secs(10) {
            score -= 0.20;
        }

        // Consistency affects usability
        if !validation_results.is_empty() {
            let avg_consistency: f64 = validation_results.iter()
                .map(|r| r.metrics.consistency_score)
                .sum::<f64>() / validation_results.len() as f64;
            
            if avg_consistency >= 0.90 {
                score += 0.10;
            } else if avg_consistency < 0.70 {
                score -= 0.15;
            }
        }

        score.max(0.0).min(1.0)
    }

    fn calculate_security_quality(&self, validation_results: &[WorkflowValidationResult], health: &HealthReport) -> f64 {
        let mut score = 0.85; // Base score (assume good security by default)

        // Check for security-related validation issues
        for result in validation_results {
            for issue in &result.issues {
                if matches!(issue.issue_type, super::workflow_validators::IssueType::DataIntegrity) {
                    match issue.severity {
                        super::workflow_validators::IssueSeverity::Critical => score -= 0.25,
                        super::workflow_validators::IssueSeverity::High => score -= 0.15,
                        super::workflow_validators::IssueSeverity::Medium => score -= 0.05,
                        _ => {}
                    }
                }
            }
        }

        // System health affects security
        if health.system_stability_score < 0.80 {
            score -= 0.10; // Unstable systems can have security implications
        }

        score.max(0.0).min(1.0)
    }

    fn calculate_efficiency_quality(&self, performance: &PerformanceSummary) -> f64 {
        let mut score = performance.resource_efficiency;

        // Adjust based on memory usage
        if performance.peak_memory_usage_mb < 512 {
            score += 0.10;
        } else if performance.peak_memory_usage_mb > 2048 {
            score -= 0.15;
        }

        // Adjust based on CPU usage
        if performance.peak_cpu_usage_percentage < 50.0 {
            score += 0.05;
        } else if performance.peak_cpu_usage_percentage > 90.0 {
            score -= 0.10;
        }

        score.max(0.0).min(1.0)
    }

    fn calculate_overall_quality_score(&self, dimensions: &QualityDimensions) -> f64 {
        // Weighted average of all quality dimensions
        let weights = [
            (dimensions.functional_quality, 0.25),
            (dimensions.performance_quality, 0.20),
            (dimensions.reliability_quality, 0.20),
            (dimensions.scalability_quality, 0.15),
            (dimensions.maintainability_quality, 0.08),
            (dimensions.usability_quality, 0.05),
            (dimensions.security_quality, 0.04),
            (dimensions.efficiency_quality, 0.03),
        ];

        weights.iter().map(|(score, weight)| score * weight).sum()
    }

    fn determine_quality_grade(&self, score: f64) -> QualityGrade {
        match score {
            s if s >= 0.95 => QualityGrade::Excellent,
            s if s >= 0.85 => QualityGrade::Good,
            s if s >= 0.70 => QualityGrade::Acceptable,
            s if s >= 0.50 => QualityGrade::Poor,
            _ => QualityGrade::Failing,
        }
    }

    fn generate_recommendations(
        &self,
        dimensions: &QualityDimensions,
        validation_results: &[WorkflowValidationResult],
        performance_summary: &PerformanceSummary,
        health_report: &HealthReport,
    ) -> Vec<QualityRecommendation> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if dimensions.performance_quality < 0.80 {
            recommendations.push(QualityRecommendation {
                priority: if dimensions.performance_quality < 0.60 { 
                    RecommendationPriority::Critical 
                } else { 
                    RecommendationPriority::High 
                },
                category: RecommendationCategory::Performance,
                title: "Improve Performance Metrics".to_string(),
                description: "System performance is below acceptable thresholds".to_string(),
                impact_level: ImpactLevel::High,
                effort_estimate: EffortEstimate::Medium,
                implementation_steps: vec![
                    "Profile application to identify bottlenecks".to_string(),
                    "Optimize database queries and indexes".to_string(),
                    "Implement caching mechanisms".to_string(),
                    "Consider code optimization and algorithm improvements".to_string(),
                ],
            });
        }

        // Reliability recommendations
        if dimensions.reliability_quality < 0.85 {
            recommendations.push(QualityRecommendation {
                priority: RecommendationPriority::High,
                category: RecommendationCategory::Reliability,
                title: "Enhance System Reliability".to_string(),
                description: "System reliability needs improvement for production readiness".to_string(),
                impact_level: ImpactLevel::High,
                effort_estimate: EffortEstimate::High,
                implementation_steps: vec![
                    "Implement comprehensive error handling".to_string(),
                    "Add retry mechanisms with exponential backoff".to_string(),
                    "Implement circuit breakers for external dependencies".to_string(),
                    "Add health checks and monitoring".to_string(),
                    "Create disaster recovery procedures".to_string(),
                ],
            });
        }

        // Scalability recommendations
        if dimensions.scalability_quality < 0.75 {
            recommendations.push(QualityRecommendation {
                priority: RecommendationPriority::Medium,
                category: RecommendationCategory::Scalability,
                title: "Improve Scalability Architecture".to_string(),
                description: "System may not scale effectively under increased load".to_string(),
                impact_level: ImpactLevel::Medium,
                effort_estimate: EffortEstimate::High,
                implementation_steps: vec![
                    "Implement horizontal scaling capabilities".to_string(),
                    "Optimize resource utilization".to_string(),
                    "Add load balancing mechanisms".to_string(),
                    "Consider microservices architecture".to_string(),
                    "Implement auto-scaling policies".to_string(),
                ],
            });
        }

        // Security recommendations
        if dimensions.security_quality < 0.90 {
            recommendations.push(QualityRecommendation {
                priority: RecommendationPriority::High,
                category: RecommendationCategory::Security,
                title: "Strengthen Security Measures".to_string(),
                description: "Security measures need enhancement to meet enterprise standards".to_string(),
                impact_level: ImpactLevel::High,
                effort_estimate: EffortEstimate::Medium,
                implementation_steps: vec![
                    "Implement input validation and sanitization".to_string(),
                    "Add authentication and authorization mechanisms".to_string(),
                    "Encrypt sensitive data at rest and in transit".to_string(),
                    "Implement audit logging".to_string(),
                    "Regular security assessments and penetration testing".to_string(),
                ],
            });
        }

        // Monitoring recommendations
        if health_report.failed_health_checks > 0 {
            recommendations.push(QualityRecommendation {
                priority: RecommendationPriority::Medium,
                category: RecommendationCategory::Monitoring,
                title: "Enhance Monitoring and Alerting".to_string(),
                description: "Improve system monitoring to detect and respond to issues quickly".to_string(),
                impact_level: ImpactLevel::Medium,
                effort_estimate: EffortEstimate::Low,
                implementation_steps: vec![
                    "Implement comprehensive health checks".to_string(),
                    "Add performance metrics collection".to_string(),
                    "Set up alerting for critical issues".to_string(),
                    "Create operational dashboards".to_string(),
                    "Implement log aggregation and analysis".to_string(),
                ],
            });
        }

        // Testing recommendations
        let validation_issues: usize = validation_results.iter()
            .map(|r| r.issues.len())
            .sum();
        
        if validation_issues > 5 {
            recommendations.push(QualityRecommendation {
                priority: RecommendationPriority::Medium,
                category: RecommendationCategory::Testing,
                title: "Expand Test Coverage".to_string(),
                description: "Increase test coverage to catch more issues before production".to_string(),
                impact_level: ImpactLevel::Medium,
                effort_estimate: EffortEstimate::Medium,
                implementation_steps: vec![
                    "Add unit tests for critical components".to_string(),
                    "Implement integration tests".to_string(),
                    "Add performance tests".to_string(),
                    "Implement chaos engineering tests".to_string(),
                    "Set up continuous testing pipeline".to_string(),
                ],
            });
        }

        // Sort recommendations by priority
        recommendations.sort_by(|a, b| {
            use RecommendationPriority::*;
            match (&a.priority, &b.priority) {
                (Critical, Critical) | (High, High) | (Medium, Medium) | (Low, Low) => std::cmp::Ordering::Equal,
                (Critical, _) => std::cmp::Ordering::Less,
                (_, Critical) => std::cmp::Ordering::Greater,
                (High, _) => std::cmp::Ordering::Less,
                (_, High) => std::cmp::Ordering::Greater,
                (Medium, Low) => std::cmp::Ordering::Less,
                (Low, Medium) => std::cmp::Ordering::Greater,
            }
        });

        recommendations
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance
    }
}

impl QualityAssessmentCriteria {
    pub fn default() -> Self {
        Self {
            min_functional_score: 0.85,
            min_performance_score: 0.80,
            min_reliability_score: 0.90,
            min_scalability_score: 0.75,
            max_error_rate: 0.05,
            min_uptime: 99.0,
            max_response_time: Duration::from_secs(5),
        }
    }

    pub fn strict() -> Self {
        Self {
            min_functional_score: 0.95,
            min_performance_score: 0.90,
            min_reliability_score: 0.95,
            min_scalability_score: 0.85,
            max_error_rate: 0.01,
            min_uptime: 99.9,
            max_response_time: Duration::from_secs(2),
        }
    }

    pub fn lenient() -> Self {
        Self {
            min_functional_score: 0.70,
            min_performance_score: 0.65,
            min_reliability_score: 0.80,
            min_scalability_score: 0.60,
            max_error_rate: 0.10,
            min_uptime: 95.0,
            max_response_time: Duration::from_secs(10),
        }
    }
}

impl QualityBenchmarks {
    pub fn default() -> Self {
        Self {
            excellent_threshold: 0.95,
            good_threshold: 0.85,
            acceptable_threshold: 0.70,
            poor_threshold: 0.50,
            industry_standards: IndustryStandards::default(),
        }
    }
}

impl IndustryStandards {
    pub fn default() -> Self {
        Self {
            enterprise_availability: 99.9,
            web_service_response_time: Duration::from_millis(200),
            database_query_latency: Duration::from_millis(50),
            api_throughput_qps: 1000.0,
            memory_efficiency: 0.85,
        }
    }
}

impl SimulationQualityAssessment {
    /// Check if the simulation meets minimum quality standards
    pub fn meets_quality_standards(&self) -> bool {
        matches!(self.quality_grade, QualityGrade::Excellent | QualityGrade::Good | QualityGrade::Acceptable) &&
        self.overall_quality_score >= 0.70
    }

    /// Check if the simulation is production ready
    pub fn is_production_ready(&self) -> bool {
        matches!(self.quality_grade, QualityGrade::Excellent | QualityGrade::Good) &&
        self.overall_quality_score >= 0.85 &&
        self.assessment_dimensions.reliability_quality >= 0.90 &&
        self.assessment_dimensions.security_quality >= 0.85
    }

    /// Get critical recommendations that must be addressed
    pub fn get_critical_recommendations(&self) -> Vec<&QualityRecommendation> {
        self.recommendations.iter()
            .filter(|r| r.priority == RecommendationPriority::Critical)
            .collect()
    }

    /// Generate comprehensive quality report
    pub fn generate_quality_report(&self) -> String {
        let status = if self.is_production_ready() {
            "PRODUCTION READY"
        } else if self.meets_quality_standards() {
            "ACCEPTABLE QUALITY"
        } else {
            "NEEDS IMPROVEMENT"
        };

        let critical_count = self.get_critical_recommendations().len();
        let high_priority_count = self.recommendations.iter()
            .filter(|r| r.priority == RecommendationPriority::High)
            .count();

        format!(
            "Simulation Quality Assessment Report\n\
            =====================================\n\
            Simulation: {}\n\
            Assessment ID: {}\n\
            Overall Quality Score: {:.1}%\n\
            Quality Grade: {:?}\n\
            Status: {}\n\
            \n\
            Quality Dimensions:\n\
            - Functional Quality: {:.1}%\n\
            - Performance Quality: {:.1}%\n\
            - Reliability Quality: {:.1}%\n\
            - Scalability Quality: {:.1}%\n\
            - Maintainability Quality: {:.1}%\n\
            - Usability Quality: {:.1}%\n\
            - Security Quality: {:.1}%\n\
            - Efficiency Quality: {:.1}%\n\
            \n\
            Recommendations Summary:\n\
            - Critical Priority: {}\n\
            - High Priority: {}\n\
            - Total Recommendations: {}\n\
            \n\
            Assessment Duration: {:?}\n\
            Assessment Timestamp: {:?}",
            self.simulation_name,
            self.assessment_id,
            self.overall_quality_score * 100.0,
            self.quality_grade,
            status,
            self.assessment_dimensions.functional_quality * 100.0,
            self.assessment_dimensions.performance_quality * 100.0,
            self.assessment_dimensions.reliability_quality * 100.0,
            self.assessment_dimensions.scalability_quality * 100.0,
            self.assessment_dimensions.maintainability_quality * 100.0,
            self.assessment_dimensions.usability_quality * 100.0,
            self.assessment_dimensions.security_quality * 100.0,
            self.assessment_dimensions.efficiency_quality * 100.0,
            critical_count,
            high_priority_count,
            self.recommendations.len(),
            self.assessment_duration,
            self.assessment_timestamp,
        )
    }

    /// Generate detailed recommendations report
    pub fn generate_recommendations_report(&self) -> String {
        let mut report = String::from("Quality Improvement Recommendations\n");
        report.push_str("====================================\n\n");

        for (index, recommendation) in self.recommendations.iter().enumerate() {
            report.push_str(&format!(
                "{}. {} (Priority: {:?})\n\
                Category: {:?}\n\
                Description: {}\n\
                Expected Impact: {:?}\n\
                Effort Estimate: {:?}\n\
                Implementation Steps:\n{}\n\n",
                index + 1,
                recommendation.title,
                recommendation.priority,
                recommendation.category,
                recommendation.description,
                recommendation.impact_level,
                recommendation.effort_estimate,
                recommendation.implementation_steps.iter()
                    .enumerate()
                    .map(|(i, step)| format!("   {}. {}", i + 1, step))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::e2e::workflow_validators::{WorkflowValidationResult, WorkflowQualityMetrics};

    fn create_test_validation_result(score: f64, is_valid: bool) -> WorkflowValidationResult {
        WorkflowValidationResult {
            workflow_name: "test_workflow".to_string(),
            is_valid,
            validation_score: score,
            issues: Vec::new(),
            metrics: WorkflowQualityMetrics {
                accuracy_score: score,
                performance_score: score,
                reliability_score: if is_valid { 1.0 } else { 0.5 },
                scalability_score: score,
                consistency_score: score,
                error_handling_score: score,
                resource_efficiency_score: score,
                overall_quality_score: score,
            },
            validation_time: Duration::from_millis(100),
        }
    }

    fn create_test_performance_summary() -> PerformanceSummary {
        PerformanceSummary {
            total_workflows: 5,
            total_executions: 100,
            overall_success_rate: 0.95,
            avg_execution_time: Duration::from_secs(2),
            current_throughput: 50.0,
            resource_efficiency: 0.85,
            monitoring_duration: Duration::from_hours(1),
            peak_memory_usage_mb: 512,
            peak_cpu_usage_percentage: 70.0,
        }
    }

    fn create_test_health_report() -> HealthReport {
        HealthReport {
            uptime_percentage: 99.5,
            total_health_checks: 1000,
            failed_health_checks: 5,
            avg_response_time: Duration::from_millis(150),
            system_stability_score: 0.90,
        }
    }

    fn create_test_trends() -> PerformanceTrends {
        PerformanceTrends {
            throughput_trend: super::performance_monitors::TrendDirection::Stable,
            latency_trend: super::performance_monitors::TrendDirection::Stable,
            memory_usage_trend: super::performance_monitors::TrendDirection::Stable,
            cpu_usage_trend: super::performance_monitors::TrendDirection::Stable,
        }
    }

    #[test]
    fn test_quality_assessor_creation() {
        let assessor = SimulationQualityAssessor::new();
        assert_eq!(assessor.assessment_criteria.min_functional_score, 0.85);
        assert_eq!(assessor.benchmarks.excellent_threshold, 0.95);
    }

    #[test]
    fn test_quality_assessment_high_quality() {
        let assessor = SimulationQualityAssessor::new();
        let validation_results = vec![
            create_test_validation_result(0.90, true),
            create_test_validation_result(0.88, true),
        ];
        let performance = create_test_performance_summary();
        let health = create_test_health_report();
        let trends = create_test_trends();

        let assessment = assessor.assess_simulation_quality(
            "test_simulation",
            &validation_results,
            &performance,
            &health,
            &trends,
        ).unwrap();

        assert!(assessment.overall_quality_score > 0.80);
        assert!(matches!(assessment.quality_grade, QualityGrade::Good | QualityGrade::Excellent));
        assert!(assessment.meets_quality_standards());
    }

    #[test]
    fn test_quality_assessment_low_quality() {
        let assessor = SimulationQualityAssessor::new();
        let validation_results = vec![
            create_test_validation_result(0.60, false),
            create_test_validation_result(0.55, false),
        ];
        let mut performance = create_test_performance_summary();
        performance.overall_success_rate = 0.70;
        
        let mut health = create_test_health_report();
        health.uptime_percentage = 95.0;
        
        let trends = create_test_trends();

        let assessment = assessor.assess_simulation_quality(
            "test_simulation",
            &validation_results,
            &performance,
            &health,
            &trends,
        ).unwrap();

        assert!(assessment.overall_quality_score < 0.70);
        assert!(matches!(assessment.quality_grade, QualityGrade::Poor | QualityGrade::Failing));
        assert!(!assessment.meets_quality_standards());
        assert!(!assessment.is_production_ready());
    }

    #[test]
    fn test_quality_dimensions_calculation() {
        let assessor = SimulationQualityAssessor::new();
        let validation_results = vec![create_test_validation_result(0.85, true)];
        let performance = create_test_performance_summary();
        let health = create_test_health_report();
        let trends = create_test_trends();

        let dimensions = assessor.calculate_quality_dimensions(
            &validation_results,
            &performance,
            &health,
            &trends,
        ).unwrap();

        assert!(dimensions.functional_quality >= 0.80);
        assert!(dimensions.performance_quality >= 0.70);
        assert!(dimensions.reliability_quality >= 0.80);
        assert!(dimensions.scalability_quality >= 0.70);
    }

    #[test]
    fn test_quality_grade_determination() {
        let assessor = SimulationQualityAssessor::new();
        
        assert_eq!(assessor.determine_quality_grade(0.96), QualityGrade::Excellent);
        assert_eq!(assessor.determine_quality_grade(0.88), QualityGrade::Good);
        assert_eq!(assessor.determine_quality_grade(0.75), QualityGrade::Acceptable);
        assert_eq!(assessor.determine_quality_grade(0.60), QualityGrade::Poor);
        assert_eq!(assessor.determine_quality_grade(0.40), QualityGrade::Failing);
    }

    #[test]
    fn test_recommendations_generation() {
        let assessor = SimulationQualityAssessor::new();
        let validation_results = vec![create_test_validation_result(0.60, false)];
        let mut performance = create_test_performance_summary();
        performance.overall_success_rate = 0.80;
        
        let mut health = create_test_health_report();
        health.uptime_percentage = 95.0;
        health.failed_health_checks = 10;
        
        let trends = create_test_trends();

        let assessment = assessor.assess_simulation_quality(
            "test_simulation",
            &validation_results,
            &performance,
            &health,
            &trends,
        ).unwrap();

        assert!(!assessment.recommendations.is_empty());
        
        // Should have recommendations for performance, reliability, and monitoring
        let has_performance_rec = assessment.recommendations.iter()
            .any(|r| matches!(r.category, RecommendationCategory::Performance));
        let has_reliability_rec = assessment.recommendations.iter()
            .any(|r| matches!(r.category, RecommendationCategory::Reliability));
        let has_monitoring_rec = assessment.recommendations.iter()
            .any(|r| matches!(r.category, RecommendationCategory::Monitoring));

        assert!(has_performance_rec || has_reliability_rec || has_monitoring_rec);
    }

    #[test]
    fn test_quality_report_generation() {
        let assessor = SimulationQualityAssessor::new();
        let validation_results = vec![create_test_validation_result(0.85, true)];
        let performance = create_test_performance_summary();
        let health = create_test_health_report();
        let trends = create_test_trends();

        let assessment = assessor.assess_simulation_quality(
            "test_simulation",
            &validation_results,
            &performance,
            &health,
            &trends,
        ).unwrap();

        let report = assessment.generate_quality_report();
        
        assert!(report.contains("Simulation Quality Assessment Report"));
        assert!(report.contains("test_simulation"));
        assert!(report.contains("Quality Dimensions:"));
        assert!(report.contains("Recommendations Summary:"));
    }

    #[test]
    fn test_custom_criteria() {
        let strict_criteria = QualityAssessmentCriteria::strict();
        let assessor = SimulationQualityAssessor::with_custom_criteria(strict_criteria);
        
        assert_eq!(assessor.assessment_criteria.min_functional_score, 0.95);
        assert_eq!(assessor.assessment_criteria.min_reliability_score, 0.95);
        assert_eq!(assessor.assessment_criteria.max_error_rate, 0.01);
    }

    #[test]
    fn test_production_readiness_check() {
        let assessor = SimulationQualityAssessor::new();
        let validation_results = vec![
            create_test_validation_result(0.95, true),
            create_test_validation_result(0.93, true),
        ];
        let mut performance = create_test_performance_summary();
        performance.overall_success_rate = 0.99;
        
        let mut health = create_test_health_report();
        health.uptime_percentage = 99.9;
        health.system_stability_score = 0.95;
        
        let trends = create_test_trends();

        let assessment = assessor.assess_simulation_quality(
            "production_ready_simulation",
            &validation_results,
            &performance,
            &health,
            &trends,
        ).unwrap();

        assert!(assessment.is_production_ready());
        assert!(assessment.overall_quality_score >= 0.85);
        assert!(assessment.assessment_dimensions.reliability_quality >= 0.90);
    }
}