//! Workflow Validation for E2E Simulations
//! 
//! Comprehensive workflow validation systems for ensuring quality and correctness
//! of end-to-end simulation workflows.

use super::simulation_environment::{E2ESimulationEnvironment, WorkflowResult};
use anyhow::{Result, anyhow};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Workflow validation result
#[derive(Debug, Clone)]
pub struct WorkflowValidationResult {
    pub workflow_name: String,
    pub is_valid: bool,
    pub validation_score: f64,
    pub issues: Vec<ValidationIssue>,
    pub metrics: WorkflowQualityMetrics,
    pub validation_time: Duration,
}

/// Types of validation issues
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub suggested_fix: Option<String>,
    pub component: String,
}

/// Issue type categories
#[derive(Debug, Clone, PartialEq)]
pub enum IssueType {
    PerformanceIssue,
    AccuracyIssue,
    ReliabilityIssue,
    ScalabilityIssue,
    ConsistencyIssue,
    ResourceLeakage,
    DataIntegrity,
    ErrorHandling,
}

/// Issue severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Workflow quality metrics
#[derive(Debug, Clone)]
pub struct WorkflowQualityMetrics {
    pub accuracy_score: f64,
    pub performance_score: f64,
    pub reliability_score: f64,
    pub scalability_score: f64,
    pub consistency_score: f64,
    pub error_handling_score: f64,
    pub resource_efficiency_score: f64,
    pub overall_quality_score: f64,
}

/// Validation criteria for different workflow types
#[derive(Debug, Clone)]
pub struct ValidationCriteria {
    pub min_accuracy_score: f64,
    pub max_response_time: Duration,
    pub min_success_rate: f64,
    pub max_error_rate: f64,
    pub min_consistency_score: f64,
    pub max_memory_growth: f64,
    pub min_throughput: f64,
}

/// Research workflow validator
pub struct ResearchWorkflowValidator {
    criteria: ValidationCriteria,
}

impl ResearchWorkflowValidator {
    pub fn new() -> Self {
        Self {
            criteria: ValidationCriteria {
                min_accuracy_score: 0.85,
                max_response_time: Duration::from_secs(30),
                min_success_rate: 0.95,
                max_error_rate: 0.05,
                min_consistency_score: 0.90,
                max_memory_growth: 2.0,
                min_throughput: 10.0,
            },
        }
    }

    /// Validate literature review workflow
    pub fn validate_literature_review(&self, result: &WorkflowResult) -> WorkflowValidationResult {
        let start_time = Instant::now();
        let mut issues = Vec::new();
        let mut metrics = WorkflowQualityMetrics::default();

        // Validate response time
        if result.execution_time > self.criteria.max_response_time {
            issues.push(ValidationIssue {
                issue_type: IssueType::PerformanceIssue,
                severity: IssueSeverity::High,
                description: format!("Literature review took {:?}, exceeding maximum of {:?}", 
                    result.execution_time, self.criteria.max_response_time),
                suggested_fix: Some("Consider optimizing search algorithms or implementing caching".to_string()),
                component: "literature_search".to_string(),
            });
        }

        // Validate success rate
        let success_rate = if result.success { 1.0 } else { 0.0 };
        if success_rate < self.criteria.min_success_rate {
            issues.push(ValidationIssue {
                issue_type: IssueType::ReliabilityIssue,
                severity: IssueSeverity::Critical,
                description: format!("Success rate {:.2} below minimum {:.2}", 
                    success_rate, self.criteria.min_success_rate),
                suggested_fix: Some("Improve error handling and retry mechanisms".to_string()),
                component: "workflow_execution".to_string(),
            });
        }

        // Calculate accuracy score based on relevance of results
        metrics.accuracy_score = self.calculate_literature_accuracy(&result.output_data);
        if metrics.accuracy_score < self.criteria.min_accuracy_score {
            issues.push(ValidationIssue {
                issue_type: IssueType::AccuracyIssue,
                severity: IssueSeverity::High,
                description: format!("Literature relevance score {:.2} below minimum {:.2}", 
                    metrics.accuracy_score, self.criteria.min_accuracy_score),
                suggested_fix: Some("Improve search query generation and ranking algorithms".to_string()),
                component: "relevance_scoring".to_string(),
            });
        }

        metrics.performance_score = self.calculate_performance_score(result.execution_time);
        metrics.reliability_score = success_rate;
        metrics.consistency_score = self.calculate_consistency_score(&result.metrics);
        metrics.error_handling_score = self.calculate_error_handling_score(result);
        metrics.resource_efficiency_score = self.calculate_resource_efficiency(&result.metrics);
        metrics.scalability_score = self.calculate_scalability_score(&result.metrics);
        metrics.overall_quality_score = self.calculate_overall_score(&metrics);

        let is_valid = issues.iter().all(|issue| issue.severity != IssueSeverity::Critical);

        WorkflowValidationResult {
            workflow_name: "literature_review".to_string(),
            is_valid,
            validation_score: metrics.overall_quality_score,
            issues,
            metrics,
            validation_time: start_time.elapsed(),
        }
    }

    /// Validate citation analysis workflow
    pub fn validate_citation_analysis(&self, result: &WorkflowResult) -> WorkflowValidationResult {
        let start_time = Instant::now();
        let mut issues = Vec::new();
        let mut metrics = WorkflowQualityMetrics::default();

        // Check for data integrity issues
        if let Some(citation_data) = result.output_data.get("citations") {
            if citation_data.is_empty() {
                issues.push(ValidationIssue {
                    issue_type: IssueType::DataIntegrity,
                    severity: IssueSeverity::High,
                    description: "No citations found in analysis result".to_string(),
                    suggested_fix: Some("Verify input data quality and citation parsing logic".to_string()),
                    component: "citation_parser".to_string(),
                });
            }
        }

        // Validate network analysis quality
        metrics.accuracy_score = self.calculate_citation_network_quality(&result.output_data);
        metrics.consistency_score = self.calculate_citation_consistency(&result.output_data);
        
        if metrics.consistency_score < self.criteria.min_consistency_score {
            issues.push(ValidationIssue {
                issue_type: IssueType::ConsistencyIssue,
                severity: IssueSeverity::Medium,
                description: format!("Citation network consistency {:.2} below threshold {:.2}", 
                    metrics.consistency_score, self.criteria.min_consistency_score),
                suggested_fix: Some("Improve citation normalization and deduplication".to_string()),
                component: "network_analysis".to_string(),
            });
        }

        metrics.performance_score = self.calculate_performance_score(result.execution_time);
        metrics.reliability_score = if result.success { 1.0 } else { 0.0 };
        metrics.error_handling_score = self.calculate_error_handling_score(result);
        metrics.resource_efficiency_score = self.calculate_resource_efficiency(&result.metrics);
        metrics.scalability_score = self.calculate_scalability_score(&result.metrics);
        metrics.overall_quality_score = self.calculate_overall_score(&metrics);

        let is_valid = issues.iter().all(|issue| issue.severity != IssueSeverity::Critical);

        WorkflowValidationResult {
            workflow_name: "citation_analysis".to_string(),
            is_valid,
            validation_score: metrics.overall_quality_score,
            issues,
            metrics,
            validation_time: start_time.elapsed(),
        }
    }

    /// Validate collaboration analysis workflow
    pub fn validate_collaboration_analysis(&self, result: &WorkflowResult) -> WorkflowValidationResult {
        let start_time = Instant::now();
        let mut issues = Vec::new();
        let mut metrics = WorkflowQualityMetrics::default();

        // Validate collaboration network structure
        metrics.accuracy_score = self.calculate_collaboration_network_quality(&result.output_data);
        
        if let Some(network_data) = result.output_data.get("collaboration_network") {
            if network_data.is_empty() {
                issues.push(ValidationIssue {
                    issue_type: IssueType::DataIntegrity,
                    severity: IssueSeverity::Medium,
                    description: "Empty collaboration network detected".to_string(),
                    suggested_fix: Some("Check author extraction and relationship mapping logic".to_string()),
                    component: "collaboration_mapper".to_string(),
                });
            }
        }

        // Check for scalability issues with large networks
        if let Some(node_count) = result.metrics.get("node_count") {
            if let Ok(count) = node_count.parse::<u32>() {
                if count > 10000 {
                    let processing_time_per_node = result.execution_time.as_millis() as f64 / count as f64;
                    if processing_time_per_node > 10.0 { // 10ms per node
                        issues.push(ValidationIssue {
                            issue_type: IssueType::ScalabilityIssue,
                            severity: IssueSeverity::Medium,
                            description: format!("High processing time per node: {:.2}ms", processing_time_per_node),
                            suggested_fix: Some("Implement parallel processing for large networks".to_string()),
                            component: "network_processor".to_string(),
                        });
                    }
                }
            }
        }

        metrics.performance_score = self.calculate_performance_score(result.execution_time);
        metrics.reliability_score = if result.success { 1.0 } else { 0.0 };
        metrics.consistency_score = self.calculate_collaboration_consistency(&result.output_data);
        metrics.error_handling_score = self.calculate_error_handling_score(result);
        metrics.resource_efficiency_score = self.calculate_resource_efficiency(&result.metrics);
        metrics.scalability_score = self.calculate_scalability_score(&result.metrics);
        metrics.overall_quality_score = self.calculate_overall_score(&metrics);

        let is_valid = issues.iter().all(|issue| issue.severity != IssueSeverity::Critical);

        WorkflowValidationResult {
            workflow_name: "collaboration_analysis".to_string(),
            is_valid,
            validation_score: metrics.overall_quality_score,
            issues,
            metrics,
            validation_time: start_time.elapsed(),
        }
    }

    // Private helper methods

    fn calculate_literature_accuracy(&self, output_data: &HashMap<String, String>) -> f64 {
        // Simulate relevance scoring based on output quality
        if let Some(papers) = output_data.get("relevant_papers") {
            let paper_count = papers.split(',').count();
            if paper_count >= 10 {
                0.95 // High relevance for sufficient papers
            } else if paper_count >= 5 {
                0.85 // Medium relevance
            } else {
                0.70 // Low relevance
            }
        } else {
            0.50 // No papers found
        }
    }

    fn calculate_citation_network_quality(&self, output_data: &HashMap<String, String>) -> f64 {
        // Simulate network quality assessment
        if let Some(citations) = output_data.get("citations") {
            let citation_count = citations.split(',').count();
            let network_density = (citation_count as f64 / 100.0).min(1.0);
            0.8 + (network_density * 0.2) // Base score + density bonus
        } else {
            0.60
        }
    }

    fn calculate_citation_consistency(&self, output_data: &HashMap<String, String>) -> f64 {
        // Check for consistent citation formats and valid relationships
        if output_data.contains_key("citations") && output_data.contains_key("citation_graph") {
            0.90 // Good consistency if both present
        } else {
            0.70 // Partial consistency
        }
    }

    fn calculate_collaboration_network_quality(&self, output_data: &HashMap<String, String>) -> f64 {
        // Assess collaboration network structure quality
        if let Some(network) = output_data.get("collaboration_network") {
            if network.contains("clusters") && network.contains("centrality") {
                0.92 // High quality with clustering and centrality analysis
            } else {
                0.80 // Basic network structure
            }
        } else {
            0.65
        }
    }

    fn calculate_collaboration_consistency(&self, output_data: &HashMap<String, String>) -> f64 {
        // Check for consistent author naming and affiliation mapping
        if output_data.contains_key("author_mapping") && output_data.contains_key("affiliation_clusters") {
            0.88
        } else {
            0.75
        }
    }

    fn calculate_performance_score(&self, execution_time: Duration) -> f64 {
        let ratio = self.criteria.max_response_time.as_millis() as f64 / execution_time.as_millis() as f64;
        ratio.min(1.0).max(0.0)
    }

    fn calculate_consistency_score(&self, metrics: &HashMap<String, String>) -> f64 {
        // Simulate consistency measurement based on output variation
        if metrics.contains_key("output_variance") {
            0.85 // Good consistency if variance is tracked
        } else {
            0.80 // Default consistency
        }
    }

    fn calculate_error_handling_score(&self, result: &WorkflowResult) -> f64 {
        if result.success {
            if result.metrics.contains_key("errors_handled") {
                0.95 // Excellent error handling
            } else {
                0.85 // Good (no errors occurred)
            }
        } else {
            if result.metrics.contains_key("error_recovery_attempted") {
                0.70 // Attempted recovery
            } else {
                0.40 // Poor error handling
            }
        }
    }

    fn calculate_resource_efficiency(&self, metrics: &HashMap<String, String>) -> f64 {
        // Simulate resource efficiency calculation
        if let Some(memory_usage) = metrics.get("peak_memory_mb") {
            if let Ok(memory) = memory_usage.parse::<f64>() {
                if memory < 512.0 {
                    0.95 // Excellent efficiency
                } else if memory < 1024.0 {
                    0.85 // Good efficiency
                } else {
                    0.70 // Acceptable efficiency
                }
            } else {
                0.80
            }
        } else {
            0.80 // Default efficiency
        }
    }

    fn calculate_scalability_score(&self, metrics: &HashMap<String, String>) -> f64 {
        // Assess scalability based on processing patterns
        if let Some(throughput) = metrics.get("throughput") {
            if let Ok(tps) = throughput.parse::<f64>() {
                if tps >= self.criteria.min_throughput {
                    0.90
                } else {
                    0.70
                }
            } else {
                0.75
            }
        } else {
            0.75
        }
    }

    fn calculate_overall_score(&self, metrics: &WorkflowQualityMetrics) -> f64 {
        let weights = [
            (metrics.accuracy_score, 0.25),
            (metrics.performance_score, 0.20),
            (metrics.reliability_score, 0.20),
            (metrics.consistency_score, 0.15),
            (metrics.error_handling_score, 0.10),
            (metrics.resource_efficiency_score, 0.05),
            (metrics.scalability_score, 0.05),
        ];

        weights.iter().map(|(score, weight)| score * weight).sum()
    }
}

/// Content creation workflow validator
pub struct ContentWorkflowValidator {
    criteria: ValidationCriteria,
}

impl ContentWorkflowValidator {
    pub fn new() -> Self {
        Self {
            criteria: ValidationCriteria {
                min_accuracy_score: 0.80,
                max_response_time: Duration::from_secs(60),
                min_success_rate: 0.90,
                max_error_rate: 0.10,
                min_consistency_score: 0.85,
                max_memory_growth: 3.0,
                min_throughput: 5.0,
            },
        }
    }

    /// Validate article outline generation
    pub fn validate_outline_generation(&self, result: &WorkflowResult) -> WorkflowValidationResult {
        let start_time = Instant::now();
        let mut issues = Vec::new();
        let mut metrics = WorkflowQualityMetrics::default();

        // Validate outline structure
        metrics.accuracy_score = self.calculate_outline_quality(&result.output_data);
        if metrics.accuracy_score < self.criteria.min_accuracy_score {
            issues.push(ValidationIssue {
                issue_type: IssueType::AccuracyIssue,
                severity: IssueSeverity::High,
                description: format!("Outline quality score {:.2} below minimum {:.2}", 
                    metrics.accuracy_score, self.criteria.min_accuracy_score),
                suggested_fix: Some("Improve outline structure generation and coherence".to_string()),
                component: "outline_generator".to_string(),
            });
        }

        // Check for content completeness
        if let Some(outline) = result.output_data.get("outline") {
            if outline.len() < 100 {
                issues.push(ValidationIssue {
                    issue_type: IssueType::AccuracyIssue,
                    severity: IssueSeverity::Medium,
                    description: "Generated outline appears too brief".to_string(),
                    suggested_fix: Some("Enhance content generation to provide more detailed outlines".to_string()),
                    component: "content_generator".to_string(),
                });
            }
        }

        metrics.performance_score = self.calculate_performance_score(result.execution_time);
        metrics.reliability_score = if result.success { 1.0 } else { 0.0 };
        metrics.consistency_score = self.calculate_content_consistency(&result.output_data);
        metrics.error_handling_score = self.calculate_error_handling_score(result);
        metrics.resource_efficiency_score = self.calculate_resource_efficiency(&result.metrics);
        metrics.scalability_score = self.calculate_scalability_score(&result.metrics);
        metrics.overall_quality_score = self.calculate_overall_score(&metrics);

        let is_valid = issues.iter().all(|issue| issue.severity != IssueSeverity::Critical);

        WorkflowValidationResult {
            workflow_name: "outline_generation".to_string(),
            is_valid,
            validation_score: metrics.overall_quality_score,
            issues,
            metrics,
            validation_time: start_time.elapsed(),
        }
    }

    /// Validate FAQ generation workflow
    pub fn validate_faq_generation(&self, result: &WorkflowResult) -> WorkflowValidationResult {
        let start_time = Instant::now();
        let mut issues = Vec::new();
        let mut metrics = WorkflowQualityMetrics::default();

        // Validate FAQ quality and coverage
        metrics.accuracy_score = self.calculate_faq_quality(&result.output_data);
        
        if let Some(faqs) = result.output_data.get("faqs") {
            let faq_count = faqs.split("Q:").count() - 1;
            if faq_count < 5 {
                issues.push(ValidationIssue {
                    issue_type: IssueType::AccuracyIssue,
                    severity: IssueSeverity::Medium,
                    description: format!("Only {} FAQs generated, expected at least 5", faq_count),
                    suggested_fix: Some("Improve question generation coverage".to_string()),
                    component: "faq_generator".to_string(),
                });
            }
        }

        metrics.performance_score = self.calculate_performance_score(result.execution_time);
        metrics.reliability_score = if result.success { 1.0 } else { 0.0 };
        metrics.consistency_score = self.calculate_faq_consistency(&result.output_data);
        metrics.error_handling_score = self.calculate_error_handling_score(result);
        metrics.resource_efficiency_score = self.calculate_resource_efficiency(&result.metrics);
        metrics.scalability_score = self.calculate_scalability_score(&result.metrics);
        metrics.overall_quality_score = self.calculate_overall_score(&metrics);

        let is_valid = issues.iter().all(|issue| issue.severity != IssueSeverity::Critical);

        WorkflowValidationResult {
            workflow_name: "faq_generation".to_string(),
            is_valid,
            validation_score: metrics.overall_quality_score,
            issues,
            metrics,
            validation_time: start_time.elapsed(),
        }
    }

    /// Validate fact checking workflow
    pub fn validate_fact_checking(&self, result: &WorkflowResult) -> WorkflowValidationResult {
        let start_time = Instant::now();
        let mut issues = Vec::new();
        let mut metrics = WorkflowQualityMetrics::default();

        // Validate fact checking accuracy
        metrics.accuracy_score = self.calculate_fact_checking_accuracy(&result.output_data);
        if metrics.accuracy_score < 0.90 {
            issues.push(ValidationIssue {
                issue_type: IssueType::AccuracyIssue,
                severity: IssueSeverity::Critical,
                description: format!("Fact checking accuracy {:.2} below critical threshold 0.90", 
                    metrics.accuracy_score),
                suggested_fix: Some("Improve source verification and cross-referencing".to_string()),
                component: "fact_checker".to_string(),
            });
        }

        // Check for proper source attribution
        if !result.output_data.contains_key("sources") {
            issues.push(ValidationIssue {
                issue_type: IssueType::DataIntegrity,
                severity: IssueSeverity::High,
                description: "No source attribution found in fact checking result".to_string(),
                suggested_fix: Some("Ensure all fact checks include source references".to_string()),
                component: "source_tracker".to_string(),
            });
        }

        metrics.performance_score = self.calculate_performance_score(result.execution_time);
        metrics.reliability_score = if result.success { 1.0 } else { 0.0 };
        metrics.consistency_score = self.calculate_fact_checking_consistency(&result.output_data);
        metrics.error_handling_score = self.calculate_error_handling_score(result);
        metrics.resource_efficiency_score = self.calculate_resource_efficiency(&result.metrics);
        metrics.scalability_score = self.calculate_scalability_score(&result.metrics);
        metrics.overall_quality_score = self.calculate_overall_score(&metrics);

        let is_valid = issues.iter().all(|issue| issue.severity != IssueSeverity::Critical);

        WorkflowValidationResult {
            workflow_name: "fact_checking".to_string(),
            is_valid,
            validation_score: metrics.overall_quality_score,
            issues,
            metrics,
            validation_time: start_time.elapsed(),
        }
    }

    // Private helper methods

    fn calculate_outline_quality(&self, output_data: &HashMap<String, String>) -> f64 {
        if let Some(outline) = output_data.get("outline") {
            let has_structure = outline.contains("1.") || outline.contains("I.");
            let has_hierarchy = outline.contains("1.1") || outline.contains("A.");
            let has_content = outline.len() > 200;
            
            let mut score = 0.60;
            if has_structure { score += 0.15; }
            if has_hierarchy { score += 0.15; }
            if has_content { score += 0.10; }
            
            score
        } else {
            0.50
        }
    }

    fn calculate_faq_quality(&self, output_data: &HashMap<String, String>) -> f64 {
        if let Some(faqs) = output_data.get("faqs") {
            let question_count = faqs.split("Q:").count() - 1;
            let answer_count = faqs.split("A:").count() - 1;
            
            if question_count == answer_count && question_count >= 5 {
                0.85
            } else if question_count == answer_count {
                0.75
            } else {
                0.60
            }
        } else {
            0.50
        }
    }

    fn calculate_fact_checking_accuracy(&self, output_data: &HashMap<String, String>) -> f64 {
        // Simulate fact checking accuracy based on verification results
        if let Some(verification) = output_data.get("verification_results") {
            if verification.contains("verified") {
                0.95
            } else if verification.contains("partial") {
                0.80
            } else {
                0.65
            }
        } else {
            0.70
        }
    }

    fn calculate_content_consistency(&self, output_data: &HashMap<String, String>) -> f64 {
        // Check for consistent formatting and style
        if output_data.contains_key("outline") && output_data.contains_key("keywords") {
            0.85
        } else {
            0.75
        }
    }

    fn calculate_faq_consistency(&self, output_data: &HashMap<String, String>) -> f64 {
        if let Some(faqs) = output_data.get("faqs") {
            let q_count = faqs.split("Q:").count() - 1;
            let a_count = faqs.split("A:").count() - 1;
            
            if q_count == a_count {
                0.90
            } else {
                0.70
            }
        } else {
            0.75
        }
    }

    fn calculate_fact_checking_consistency(&self, output_data: &HashMap<String, String>) -> f64 {
        // Check for consistent verification methodology
        if output_data.contains_key("verification_method") && output_data.contains_key("confidence_scores") {
            0.88
        } else {
            0.75
        }
    }

    fn calculate_performance_score(&self, execution_time: Duration) -> f64 {
        let ratio = self.criteria.max_response_time.as_millis() as f64 / execution_time.as_millis() as f64;
        ratio.min(1.0).max(0.0)
    }

    fn calculate_error_handling_score(&self, result: &WorkflowResult) -> f64 {
        if result.success {
            if result.metrics.contains_key("errors_handled") {
                0.95
            } else {
                0.85
            }
        } else {
            if result.metrics.contains_key("error_recovery_attempted") {
                0.70
            } else {
                0.40
            }
        }
    }

    fn calculate_resource_efficiency(&self, metrics: &HashMap<String, String>) -> f64 {
        if let Some(memory_usage) = metrics.get("peak_memory_mb") {
            if let Ok(memory) = memory_usage.parse::<f64>() {
                if memory < 1024.0 {
                    0.90
                } else if memory < 2048.0 {
                    0.80
                } else {
                    0.65
                }
            } else {
                0.75
            }
        } else {
            0.75
        }
    }

    fn calculate_scalability_score(&self, metrics: &HashMap<String, String>) -> f64 {
        if let Some(throughput) = metrics.get("throughput") {
            if let Ok(tps) = throughput.parse::<f64>() {
                if tps >= self.criteria.min_throughput {
                    0.85
                } else {
                    0.70
                }
            } else {
                0.75
            }
        } else {
            0.75
        }
    }

    fn calculate_overall_score(&self, metrics: &WorkflowQualityMetrics) -> f64 {
        let weights = [
            (metrics.accuracy_score, 0.30),
            (metrics.performance_score, 0.20),
            (metrics.reliability_score, 0.20),
            (metrics.consistency_score, 0.15),
            (metrics.error_handling_score, 0.10),
            (metrics.resource_efficiency_score, 0.03),
            (metrics.scalability_score, 0.02),
        ];

        weights.iter().map(|(score, weight)| score * weight).sum()
    }
}

impl WorkflowQualityMetrics {
    pub fn default() -> Self {
        Self {
            accuracy_score: 0.0,
            performance_score: 0.0,
            reliability_score: 0.0,
            scalability_score: 0.0,
            consistency_score: 0.0,
            error_handling_score: 0.0,
            resource_efficiency_score: 0.0,
            overall_quality_score: 0.0,
        }
    }
}

/// Comprehensive workflow validation suite
pub struct WorkflowValidationSuite {
    research_validator: ResearchWorkflowValidator,
    content_validator: ContentWorkflowValidator,
}

impl WorkflowValidationSuite {
    pub fn new() -> Self {
        Self {
            research_validator: ResearchWorkflowValidator::new(),
            content_validator: ContentWorkflowValidator::new(),
        }
    }

    /// Validate any workflow result based on its type
    pub fn validate_workflow(&self, workflow_type: &str, result: &WorkflowResult) -> Result<WorkflowValidationResult> {
        match workflow_type {
            "literature_review" => Ok(self.research_validator.validate_literature_review(result)),
            "citation_analysis" => Ok(self.research_validator.validate_citation_analysis(result)),
            "collaboration_analysis" => Ok(self.research_validator.validate_collaboration_analysis(result)),
            "outline_generation" => Ok(self.content_validator.validate_outline_generation(result)),
            "faq_generation" => Ok(self.content_validator.validate_faq_generation(result)),
            "fact_checking" => Ok(self.content_validator.validate_fact_checking(result)),
            _ => Err(anyhow!("Unknown workflow type: {}", workflow_type)),
        }
    }

    /// Generate validation summary for multiple workflows
    pub fn generate_validation_summary(&self, results: &[WorkflowValidationResult]) -> ValidationSummary {
        let total_workflows = results.len();
        let valid_workflows = results.iter().filter(|r| r.is_valid).count();
        let avg_score = if total_workflows > 0 {
            results.iter().map(|r| r.validation_score).sum::<f64>() / total_workflows as f64
        } else {
            0.0
        };

        let critical_issues: Vec<&ValidationIssue> = results
            .iter()
            .flat_map(|r| &r.issues)
            .filter(|i| i.severity == IssueSeverity::Critical)
            .collect();

        let high_issues: Vec<&ValidationIssue> = results
            .iter()
            .flat_map(|r| &r.issues)
            .filter(|i| i.severity == IssueSeverity::High)
            .collect();

        ValidationSummary {
            total_workflows,
            valid_workflows,
            validation_pass_rate: valid_workflows as f64 / total_workflows as f64,
            average_quality_score: avg_score,
            critical_issue_count: critical_issues.len(),
            high_issue_count: high_issues.len(),
            most_common_issues: self.analyze_common_issues(results),
        }
    }

    fn analyze_common_issues(&self, results: &[WorkflowValidationResult]) -> Vec<(IssueType, usize)> {
        let mut issue_counts = HashMap::new();
        
        for result in results {
            for issue in &result.issues {
                *issue_counts.entry(issue.issue_type.clone()).or_insert(0) += 1;
            }
        }

        let mut sorted_issues: Vec<(IssueType, usize)> = issue_counts.into_iter().collect();
        sorted_issues.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_issues.into_iter().take(5).collect()
    }
}

/// Summary of validation results across multiple workflows
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    pub total_workflows: usize,
    pub valid_workflows: usize,
    pub validation_pass_rate: f64,
    pub average_quality_score: f64,
    pub critical_issue_count: usize,
    pub high_issue_count: usize,
    pub most_common_issues: Vec<(IssueType, usize)>,
}

impl ValidationSummary {
    /// Check if the overall validation passed
    pub fn is_validation_successful(&self) -> bool {
        self.validation_pass_rate >= 0.90 && 
        self.critical_issue_count == 0 && 
        self.average_quality_score >= 0.80
    }

    /// Generate human-readable summary
    pub fn generate_report(&self) -> String {
        format!(
            "Workflow Validation Summary:\n\
            Total Workflows: {}\n\
            Valid Workflows: {} ({:.1}%)\n\
            Average Quality Score: {:.2}\n\
            Critical Issues: {}\n\
            High Priority Issues: {}\n\
            Overall Status: {}\n\
            \nMost Common Issues:\n{}", 
            self.total_workflows,
            self.valid_workflows,
            self.validation_pass_rate * 100.0,
            self.average_quality_score,
            self.critical_issue_count,
            self.high_issue_count,
            if self.is_validation_successful() { "PASSED" } else { "FAILED" },
            self.most_common_issues.iter()
                .map(|(issue_type, count)| format!("  {:?}: {} occurrences", issue_type, count))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_workflow_result(success: bool, execution_time: Duration) -> WorkflowResult {
        let mut output_data = HashMap::new();
        output_data.insert("relevant_papers".to_string(), "paper1,paper2,paper3,paper4,paper5,paper6,paper7,paper8,paper9,paper10".to_string());
        
        let mut metrics = HashMap::new();
        metrics.insert("peak_memory_mb".to_string(), "256".to_string());
        metrics.insert("throughput".to_string(), "15.0".to_string());
        
        WorkflowResult {
            workflow_name: "test_workflow".to_string(),
            success,
            execution_time,
            output_data,
            metrics,
        }
    }

    #[test]
    fn test_research_workflow_validation() {
        let validator = ResearchWorkflowValidator::new();
        let result = create_test_workflow_result(true, Duration::from_secs(10));
        
        let validation = validator.validate_literature_review(&result);
        
        assert!(validation.is_valid);
        assert!(validation.validation_score > 0.8);
        assert_eq!(validation.workflow_name, "literature_review");
    }

    #[test]
    fn test_content_workflow_validation() {
        let validator = ContentWorkflowValidator::new();
        let mut result = create_test_workflow_result(true, Duration::from_secs(30));
        result.output_data.insert("outline".to_string(), 
            "1. Introduction\n1.1 Background\n1.2 Objectives\n2. Methods\n2.1 Data Collection\n2.2 Analysis\n3. Results\n3.1 Findings\n3.2 Discussion\n4. Conclusion".to_string());
        
        let validation = validator.validate_outline_generation(&result);
        
        assert!(validation.is_valid);
        assert!(validation.validation_score > 0.7);
    }

    #[test]
    fn test_validation_with_performance_issues() {
        let validator = ResearchWorkflowValidator::new();
        let result = create_test_workflow_result(true, Duration::from_secs(45)); // Exceeds max time
        
        let validation = validator.validate_literature_review(&result);
        
        assert!(!validation.issues.is_empty());
        let has_performance_issue = validation.issues.iter()
            .any(|issue| issue.issue_type == IssueType::PerformanceIssue);
        assert!(has_performance_issue);
    }

    #[test]
    fn test_validation_with_failure() {
        let validator = ResearchWorkflowValidator::new();
        let result = create_test_workflow_result(false, Duration::from_secs(10));
        
        let validation = validator.validate_literature_review(&result);
        
        assert!(!validation.is_valid);
        let has_reliability_issue = validation.issues.iter()
            .any(|issue| issue.issue_type == IssueType::ReliabilityIssue && issue.severity == IssueSeverity::Critical);
        assert!(has_reliability_issue);
    }

    #[test]
    fn test_validation_suite() {
        let suite = WorkflowValidationSuite::new();
        let result = create_test_workflow_result(true, Duration::from_secs(10));
        
        let validation = suite.validate_workflow("literature_review", &result).unwrap();
        assert!(validation.is_valid);
        
        // Test unknown workflow type
        let unknown_result = suite.validate_workflow("unknown_workflow", &result);
        assert!(unknown_result.is_err());
    }

    #[test]
    fn test_validation_summary() {
        let suite = WorkflowValidationSuite::new();
        let good_result = create_test_workflow_result(true, Duration::from_secs(10));
        let bad_result = create_test_workflow_result(false, Duration::from_secs(10));
        
        let validations = vec![
            suite.validate_workflow("literature_review", &good_result).unwrap(),
            suite.validate_workflow("citation_analysis", &bad_result).unwrap(),
        ];
        
        let summary = suite.generate_validation_summary(&validations);
        
        assert_eq!(summary.total_workflows, 2);
        assert_eq!(summary.valid_workflows, 1);
        assert_eq!(summary.validation_pass_rate, 0.5);
        assert!(!summary.is_validation_successful());
    }

    #[test]
    fn test_fact_checking_validation() {
        let validator = ContentWorkflowValidator::new();
        let mut result = create_test_workflow_result(true, Duration::from_secs(20));
        result.output_data.insert("verification_results".to_string(), "verified".to_string());
        result.output_data.insert("sources".to_string(), "source1,source2,source3".to_string());
        
        let validation = validator.validate_fact_checking(&result);
        
        assert!(validation.is_valid);
        assert!(validation.metrics.accuracy_score >= 0.90);
    }

    #[test]
    fn test_faq_generation_validation() {
        let validator = ContentWorkflowValidator::new();
        let mut result = create_test_workflow_result(true, Duration::from_secs(25));
        result.output_data.insert("faqs".to_string(), 
            "Q: What is this? A: This is a test. Q: How does it work? A: It works well. Q: Why use it? A: Because it's useful. Q: When to use? A: Use it now. Q: Where to find it? A: Find it here.".to_string());
        
        let validation = validator.validate_faq_generation(&result);
        
        assert!(validation.is_valid);
        assert!(validation.metrics.accuracy_score >= 0.80);
    }
}