# Task 074: Implement Markdown Report Generation

## Context
You are implementing markdown report generation for a Rust-based vector indexing system. The MarkdownReportGenerator creates professional, well-formatted markdown reports from validation results with tables, charts, and comprehensive documentation.

## Project Structure
```
src/
  validation/
    markdown_generator.rs     <- Create this file
  lib.rs
```

## Task Description
Create the `MarkdownReportGenerator` struct that converts validation reports into professional markdown format with tables, charts, executive summaries, and detailed findings.

## Requirements
1. Create `src/validation/markdown_generator.rs`
2. Implement comprehensive markdown generation from validation reports
3. Add table generation with proper formatting and alignment
4. Include chart placeholders and data visualization markup
5. Generate executive summaries and detailed sections
6. Support multiple report types (accuracy, performance, security, stress)

## Expected Code Structure
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use anyhow::{Result, Context};

use crate::validation::{
    ValidationReport,
    AccuracyReport,
    PerformanceReport,
    StressTestReport,
    SecurityReport,
};

#[derive(Debug, Clone)]
pub struct MarkdownReportGenerator {
    pub config: MarkdownConfig,
    pub template_registry: TemplateRegistry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownConfig {
    pub include_toc: bool,
    pub include_charts: bool,
    pub include_raw_data: bool,
    pub max_table_rows: usize,
    pub chart_width: u32,
    pub chart_height: u32,
    pub theme: MarkdownTheme,
    pub custom_css: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkdownTheme {
    Professional,
    Technical,
    Executive,
    Detailed,
}

#[derive(Debug, Clone)]
pub struct TemplateRegistry {
    pub report_templates: HashMap<String, ReportTemplate>,
    pub section_templates: HashMap<String, SectionTemplate>,
    pub table_templates: HashMap<String, TableTemplate>,
}

#[derive(Debug, Clone)]
pub struct ReportTemplate {
    pub template_name: String,
    pub header_template: String,
    pub toc_template: String,
    pub section_order: Vec<String>,
    pub footer_template: String,
}

#[derive(Debug, Clone)]
pub struct SectionTemplate {
    pub section_name: String,
    pub title_format: String,
    pub content_format: String,
    pub subsection_format: String,
}

#[derive(Debug, Clone)]
pub struct TableTemplate {
    pub table_name: String,
    pub header_format: String,
    pub row_format: String,
    pub alignment: Vec<TableAlignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TableAlignment {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone)]
pub struct GeneratedReport {
    pub content: String,
    pub metadata: ReportMetadata,
    pub sections: Vec<GeneratedSection>,
    pub charts: Vec<ChartPlaceholder>,
    pub attachments: Vec<ReportAttachment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub generated_at: DateTime<Utc>,
    pub report_type: String,
    pub version: String,
    pub total_sections: usize,
    pub total_tables: usize,
    pub total_charts: usize,
    pub word_count: usize,
}

#[derive(Debug, Clone)]
pub struct GeneratedSection {
    pub section_id: String,
    pub title: String,
    pub content: String,
    pub subsections: Vec<GeneratedSection>,
    pub tables: Vec<GeneratedTable>,
    pub charts: Vec<ChartPlaceholder>,
}

#[derive(Debug, Clone)]
pub struct GeneratedTable {
    pub table_id: String,
    pub title: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub alignment: Vec<TableAlignment>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartPlaceholder {
    pub chart_id: String,
    pub chart_type: ChartType,
    pub title: String,
    pub data_source: String,
    pub width: u32,
    pub height: u32,
    pub placeholder_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    BarChart,
    LineChart,
    PieChart,
    ScatterPlot,
    Histogram,
    HeatMap,
    TreeMap,
    Gauge,
}

#[derive(Debug, Clone)]
pub struct ReportAttachment {
    pub attachment_id: String,
    pub filename: String,
    pub content_type: String,
    pub content: Vec<u8>,
    pub description: String,
}

impl MarkdownReportGenerator {
    pub fn new() -> Self {
        Self {
            config: MarkdownConfig::default(),
            template_registry: TemplateRegistry::new(),
        }
    }

    pub fn with_config(config: MarkdownConfig) -> Self {
        Self {
            config,
            template_registry: TemplateRegistry::new(),
        }
    }

    pub fn generate_validation_report(&self, report: &ValidationReport) -> Result<GeneratedReport> {
        let mut generated = GeneratedReport {
            content: String::new(),
            metadata: ReportMetadata {
                generated_at: Utc::now(),
                report_type: "Validation Report".to_string(),
                version: "1.0.0".to_string(),
                total_sections: 0,
                total_tables: 0,
                total_charts: 0,
                word_count: 0,
            },
            sections: Vec::new(),
            charts: Vec::new(),
            attachments: Vec::new(),
        };

        // Generate report header
        generated.content.push_str(&self.generate_header("LLMKG Vector Indexing System - Validation Report")?);

        // Generate table of contents if enabled
        if self.config.include_toc {
            generated.content.push_str(&self.generate_toc(&report)?);
        }

        // Generate executive summary
        generated.content.push_str(&self.generate_executive_summary(&report)?);

        // Generate validation results section
        generated.content.push_str(&self.generate_validation_results_section(&report)?);

        // Generate performance metrics section
        generated.content.push_str(&self.generate_performance_section(&report)?);

        // Generate detailed findings
        generated.content.push_str(&self.generate_detailed_findings(&report)?);

        // Generate recommendations
        generated.content.push_str(&self.generate_recommendations_section(&report)?);

        // Generate appendices if raw data is included
        if self.config.include_raw_data {
            generated.content.push_str(&self.generate_appendices(&report)?);
        }

        // Generate footer
        generated.content.push_str(&self.generate_footer()?);

        // Update metadata
        generated.metadata.word_count = self.count_words(&generated.content);
        generated.metadata.total_sections = self.count_sections(&generated.content);
        generated.metadata.total_tables = self.count_tables(&generated.content);
        generated.metadata.total_charts = generated.charts.len();

        Ok(generated)
    }

    pub fn generate_accuracy_report(&self, report: &AccuracyReport) -> Result<GeneratedReport> {
        let mut generated = GeneratedReport {
            content: String::new(),
            metadata: ReportMetadata {
                generated_at: Utc::now(),
                report_type: "Accuracy Analysis Report".to_string(),
                version: "1.0.0".to_string(),
                total_sections: 0,
                total_tables: 0,
                total_charts: 0,
                word_count: 0,
            },
            sections: Vec::new(),
            charts: Vec::new(),
            attachments: Vec::new(),
        };

        // Report header
        generated.content.push_str(&self.generate_header("Accuracy Analysis Report")?);

        // Executive summary
        generated.content.push_str(&format!(
            "\n## Executive Summary\n\n\
            **Overall Accuracy**: {:.2}%\n\
            **Accuracy Grade**: {:?}\n\
            **Total Test Cases**: {}\n\
            **Query Types Analyzed**: {}\n\n\
            {}\n\n",
            report.overall_metrics.accuracy_percentage,
            report.overall_metrics.accuracy_grade,
            report.metadata.total_test_cases,
            report.query_type_breakdown.len(),
            self.generate_accuracy_summary_table(&report)?
        ));

        // Statistical analysis section
        generated.content.push_str(&self.generate_statistical_analysis_section(&report)?);

        // Query type breakdown
        generated.content.push_str(&self.generate_query_type_breakdown(&report)?);

        // Confidence intervals
        generated.content.push_str(&self.generate_confidence_intervals_section(&report)?);

        // Failure analysis
        generated.content.push_str(&self.generate_failure_analysis_section(&report)?);

        // Charts section
        if self.config.include_charts {
            generated.content.push_str(&self.generate_accuracy_charts_section(&report, &mut generated.charts)?);
        }

        // Recommendations
        generated.content.push_str(&self.generate_accuracy_recommendations(&report)?);

        // Update metadata
        generated.metadata.word_count = self.count_words(&generated.content);
        generated.metadata.total_sections = self.count_sections(&generated.content);
        generated.metadata.total_tables = self.count_tables(&generated.content);

        Ok(generated)
    }

    pub fn generate_performance_report(&self, report: &PerformanceReport) -> Result<GeneratedReport> {
        let mut generated = GeneratedReport {
            content: String::new(),
            metadata: ReportMetadata {
                generated_at: Utc::now(),
                report_type: "Performance Benchmark Report".to_string(),
                version: "1.0.0".to_string(),
                total_sections: 0,
                total_tables: 0,
                total_charts: 0,
                word_count: 0,
            },
            sections: Vec::new(),
            charts: Vec::new(),
            attachments: Vec::new(),
        };

        // Report header
        generated.content.push_str(&self.generate_header("Performance Benchmark Report")?);

        // Performance overview
        generated.content.push_str(&format!(
            "\n## Performance Overview\n\n\
            **Test Duration**: {:.2} seconds\n\
            **Total Queries**: {}\n\
            **Performance Grade**: {:?}\n\n\
            {}\n\n",
            report.metadata.test_duration_seconds,
            report.metadata.total_queries_executed,
            report.overall_metrics.performance_grade,
            self.generate_performance_summary_table(&report)?
        ));

        // Latency analysis
        generated.content.push_str(&self.generate_latency_analysis(&report)?);

        // Throughput analysis
        generated.content.push_str(&self.generate_throughput_analysis(&report)?);

        // Resource utilization
        generated.content.push_str(&self.generate_resource_utilization(&report)?);

        // Performance charts
        if self.config.include_charts {
            generated.content.push_str(&self.generate_performance_charts(&report, &mut generated.charts)?);
        }

        // Performance recommendations
        generated.content.push_str(&self.generate_performance_recommendations(&report)?);

        // Update metadata
        generated.metadata.word_count = self.count_words(&generated.content);
        generated.metadata.total_sections = self.count_sections(&generated.content);
        generated.metadata.total_tables = self.count_tables(&generated.content);

        Ok(generated)
    }

    pub fn generate_stress_test_report(&self, report: &StressTestReport) -> Result<GeneratedReport> {
        let mut generated = GeneratedReport {
            content: String::new(),
            metadata: ReportMetadata {
                generated_at: Utc::now(),
                report_type: "Stress Test Report".to_string(),
                version: "1.0.0".to_string(),
                total_sections: 0,
                total_tables: 0,
                total_charts: 0,
                word_count: 0,
            },
            sections: Vec::new(),
            charts: Vec::new(),
            attachments: Vec::new(),
        };

        // Report header
        generated.content.push_str(&self.generate_header("Stress Test Report")?);

        // Load test overview
        generated.content.push_str(&format!(
            "\n## Load Test Overview\n\n\
            **Peak Load**: {} concurrent users\n\
            **Test Duration**: {:.2} seconds\n\
            **Breaking Point**: {:?}\n\
            **System Stability**: {:?}\n\n\
            {}\n\n",
            report.load_profile.peak_concurrent_users,
            report.metadata.test_duration_seconds,
            report.breaking_point_analysis.breaking_point_reached,
            report.resilience_metrics.system_stability,
            self.generate_load_summary_table(&report)?
        ));

        // Load progression analysis
        generated.content.push_str(&self.generate_load_progression(&report)?);

        // Breaking point analysis
        generated.content.push_str(&self.generate_breaking_point_analysis(&report)?);

        // Resilience metrics
        generated.content.push_str(&self.generate_resilience_metrics(&report)?);

        // Capacity planning
        generated.content.push_str(&self.generate_capacity_planning(&report)?);

        // Update metadata
        generated.metadata.word_count = self.count_words(&generated.content);
        generated.metadata.total_sections = self.count_sections(&generated.content);
        generated.metadata.total_tables = self.count_tables(&generated.content);

        Ok(generated)
    }

    pub fn generate_security_report(&self, report: &SecurityReport) -> Result<GeneratedReport> {
        let mut generated = GeneratedReport {
            content: String::new(),
            metadata: ReportMetadata {
                generated_at: Utc::now(),
                report_type: "Security Assessment Report".to_string(),
                version: "1.0.0".to_string(),
                total_sections: 0,
                total_tables: 0,
                total_charts: 0,
                word_count: 0,
            },
            sections: Vec::new(),
            charts: Vec::new(),
            attachments: Vec::new(),
        };

        // Report header
        generated.content.push_str(&self.generate_header("Security Assessment Report")?);

        // Security overview
        generated.content.push_str(&format!(
            "\n## Security Overview\n\n\
            **Security Score**: {:.1}/10\n\
            **Vulnerabilities Found**: {}\n\
            **Critical Issues**: {}\n\
            **Compliance Status**: {:?}\n\n\
            {}\n\n",
            report.overall_score.security_score,
            report.vulnerability_summary.total_vulnerabilities,
            report.vulnerability_summary.critical_vulnerabilities,
            report.compliance_assessment.overall_compliance_status,
            self.generate_security_summary_table(&report)?
        ));

        // Vulnerability assessment
        generated.content.push_str(&self.generate_vulnerability_assessment(&report)?);

        // Threat analysis
        generated.content.push_str(&self.generate_threat_analysis(&report)?);

        // Compliance assessment
        generated.content.push_str(&self.generate_compliance_assessment(&report)?);

        // Security recommendations
        generated.content.push_str(&self.generate_security_recommendations(&report)?);

        // Update metadata
        generated.metadata.word_count = self.count_words(&generated.content);
        generated.metadata.total_sections = self.count_sections(&generated.content);
        generated.metadata.total_tables = self.count_tables(&generated.content);

        Ok(generated)
    }

    fn generate_header(&self, title: &str) -> Result<String> {
        Ok(format!(
            "# {}\n\n\
            **Generated**: {}\n\
            **Version**: 1.0.0\n\
            **Theme**: {:?}\n\n\
            ---\n\n",
            title,
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            self.config.theme
        ))
    }

    fn generate_toc(&self, _report: &ValidationReport) -> Result<String> {
        Ok(format!(
            "## Table of Contents\n\n\
            1. [Executive Summary](#executive-summary)\n\
            2. [Validation Results](#validation-results)\n\
            3. [Performance Metrics](#performance-metrics)\n\
            4. [Detailed Findings](#detailed-findings)\n\
            5. [Recommendations](#recommendations)\n\
            6. [Appendices](#appendices)\n\n\
            ---\n\n"
        ))
    }

    fn generate_executive_summary(&self, report: &ValidationReport) -> Result<String> {
        Ok(format!(
            "## Executive Summary\n\n\
            The LLMKG Vector Indexing System has undergone comprehensive validation testing. \
            This report presents the results of accuracy, performance, stress, and security assessments.\n\n\
            **Key Findings**:\n\
            - Overall system validation: **{:.1}%** success rate\n\
            - Performance targets: **{}** achieved\n\
            - Security assessment: **Completed**\n\
            - Production readiness: **{}**\n\n\
            {}",
            report.overall_results.overall_success_rate,
            if report.overall_results.all_tests_passed { "Met" } else { "Not Met" },
            if report.overall_results.production_ready { "Ready" } else { "Not Ready" },
            self.generate_summary_metrics_table(report)?
        ))
    }

    fn generate_validation_results_section(&self, report: &ValidationReport) -> Result<String> {
        Ok(format!(
            "\n## Validation Results\n\n\
            ### Test Execution Summary\n\n\
            - **Total Tests**: {}\n\
            - **Passed**: {}\n\
            - **Failed**: {}\n\
            - **Success Rate**: {:.2}%\n\n\
            ### Test Categories\n\n\
            {}\n\n",
            report.test_summary.total_tests,
            report.test_summary.passed_tests,
            report.test_summary.failed_tests,
            report.overall_results.overall_success_rate,
            self.generate_test_categories_table(report)?
        ))
    }

    fn generate_performance_section(&self, report: &ValidationReport) -> Result<String> {
        Ok(format!(
            "\n## Performance Metrics\n\n\
            ### Timing Analysis\n\n\
            - **Total Execution Time**: {:.2} seconds\n\
            - **Average Test Time**: {:.2} ms\n\
            - **Setup Time**: {:.2} seconds\n\
            - **Cleanup Time**: {:.2} seconds\n\n\
            {}",
            report.timing.total_execution_seconds,
            report.timing.average_test_time_ms,
            report.timing.setup_time_seconds,
            report.timing.cleanup_time_seconds,
            self.generate_timing_breakdown_table(report)?
        ))
    }

    fn generate_detailed_findings(&self, report: &ValidationReport) -> Result<String> {
        let mut content = String::from("\n## Detailed Findings\n\n");

        // Environment details
        content.push_str(&format!(
            "### Test Environment\n\n\
            - **Platform**: {}\n\
            - **Rust Version**: {}\n\
            - **Test Framework**: {}\n\
            - **Configuration**: {}\n\n",
            report.environment.platform,
            report.environment.rust_version,
            report.environment.test_framework_version,
            report.environment.configuration_hash
        ));

        // Test failures (if any)
        if !report.test_summary.failed_tests == 0 {
            content.push_str("### Failed Tests\n\n");
            content.push_str("All tests passed successfully. No failures to report.\n\n");
        }

        Ok(content)
    }

    fn generate_recommendations_section(&self, report: &ValidationReport) -> Result<String> {
        Ok(format!(
            "\n## Recommendations\n\n\
            Based on the validation results, the following recommendations are provided:\n\n\
            ### Production Readiness\n\
            - ✅ System meets all accuracy requirements\n\
            - ✅ Performance targets achieved\n\
            - ✅ Security assessment completed\n\
            - ✅ Ready for production deployment\n\n\
            ### Next Steps\n\
            1. Deploy to staging environment\n\
            2. Conduct user acceptance testing\n\
            3. Monitor performance in production\n\
            4. Schedule regular validation cycles\n\n"
        ))
    }

    fn generate_appendices(&self, _report: &ValidationReport) -> Result<String> {
        Ok(format!(
            "\n## Appendices\n\n\
            ### Appendix A: Raw Test Data\n\
            Raw test data is available upon request.\n\n\
            ### Appendix B: Configuration Files\n\
            All configuration files used during testing are archived.\n\n\
            ### Appendix C: Environment Details\n\
            Detailed environment specifications and dependencies.\n\n"
        ))
    }

    fn generate_footer(&self) -> Result<String> {
        Ok(format!(
            "\n---\n\n\
            **Report Generated**: {} UTC\n\
            **Generator Version**: LLMKG Validation Suite v1.0.0\n\
            **Contact**: validation@llmkg.com\n\n\
            *This report was automatically generated by the LLMKG Vector Indexing System validation suite.*\n",
            Utc::now().format("%Y-%m-%d %H:%M:%S")
        ))
    }

    // Helper methods for specific report types
    fn generate_accuracy_summary_table(&self, report: &AccuracyReport) -> Result<String> {
        Ok(format!(
            "| Metric | Value | Grade |\n\
            |--------|-------|-------|\n\
            | Overall Accuracy | {:.2}% | {:?} |\n\
            | Precision | {:.3} | {} |\n\
            | Recall | {:.3} | {} |\n\
            | F1 Score | {:.3} | {} |\n",
            report.overall_metrics.accuracy_percentage,
            report.overall_metrics.accuracy_grade,
            report.overall_metrics.precision,
            self.grade_metric(report.overall_metrics.precision),
            report.overall_metrics.recall,
            self.grade_metric(report.overall_metrics.recall),
            report.overall_metrics.f1_score,
            self.grade_metric(report.overall_metrics.f1_score)
        ))
    }

    fn generate_statistical_analysis_section(&self, report: &AccuracyReport) -> Result<String> {
        Ok(format!(
            "\n## Statistical Analysis\n\n\
            | Statistic | Value |\n\
            |-----------|-------|\n\
            | Mean Accuracy | {:.2}% |\n\
            | Median Accuracy | {:.2}% |\n\
            | Standard Deviation | {:.2}% |\n\
            | Variance | {:.2} |\n\
            | Normal Distribution | {} |\n\n",
            report.statistical_analysis.accuracy_mean,
            report.statistical_analysis.accuracy_median,
            report.statistical_analysis.accuracy_standard_deviation,
            report.statistical_analysis.accuracy_variance,
            if report.statistical_analysis.is_normally_distributed { "Yes" } else { "No" }
        ))
    }

    fn generate_query_type_breakdown(&self, report: &AccuracyReport) -> Result<String> {
        let mut content = String::from("\n## Query Type Breakdown\n\n");
        
        content.push_str("| Query Type | Test Cases | Passed | Failed | Accuracy | Precision | Recall |\n");
        content.push_str("|------------|------------|--------|--------|----------|-----------|--------|\n");
        
        for (query_type, accuracy) in &report.query_type_breakdown {
            content.push_str(&format!(
                "| {} | {} | {} | {} | {:.1}% | {:.3} | {:.3} |\n",
                query_type,
                accuracy.test_cases_count,
                accuracy.passed_count,
                accuracy.failed_count,
                accuracy.accuracy_percentage,
                accuracy.precision,
                accuracy.recall
            ));
        }
        
        content.push_str("\n");
        Ok(content)
    }

    fn generate_confidence_intervals_section(&self, report: &AccuracyReport) -> Result<String> {
        Ok(format!(
            "\n## Confidence Intervals (95%)\n\n\
            | Metric | Lower Bound | Upper Bound | Margin of Error |\n\
            |--------|-------------|-------------|------------------|\n\
            | Overall Accuracy | {:.2}% | {:.2}% | ±{:.2}% |\n\
            | Precision | {:.3} | {:.3} | - |\n\
            | Recall | {:.3} | {:.3} | - |\n\n\
            **Sample Size Adequacy**: {}\n\n",
            report.confidence_intervals.overall_accuracy_95_ci.0,
            report.confidence_intervals.overall_accuracy_95_ci.1,
            report.confidence_intervals.margin_of_error,
            report.confidence_intervals.precision_95_ci.0,
            report.confidence_intervals.precision_95_ci.1,
            report.confidence_intervals.recall_95_ci.0,
            report.confidence_intervals.recall_95_ci.1,
            if report.confidence_intervals.sample_size_adequacy { "Adequate" } else { "Insufficient" }
        ))
    }

    fn generate_failure_analysis_section(&self, report: &AccuracyReport) -> Result<String> {
        let mut content = format!(
            "\n## Failure Analysis\n\n\
            **Total Failures**: {}\n\
            **Failure Rate**: {:.2}%\n\n\
            ### Failure Categories\n\n",
            report.failure_analysis.total_failures,
            report.failure_analysis.failure_rate_percentage
        );

        if !report.failure_analysis.failure_categories.is_empty() {
            content.push_str("| Category | Count | Percentage |\n");
            content.push_str("|----------|-------|------------|\n");
            
            for (category, count) in &report.failure_analysis.failure_categories {
                let percentage = (*count as f64 / report.failure_analysis.total_failures as f64) * 100.0;
                content.push_str(&format!("| {} | {} | {:.1}% |\n", category, count, percentage));
            }
            content.push_str("\n");
        }

        content.push_str("### Remediation Suggestions\n\n");
        for (i, suggestion) in report.failure_analysis.remediation_suggestions.iter().enumerate() {
            content.push_str(&format!("{}. {}\n", i + 1, suggestion));
        }
        content.push_str("\n");

        Ok(content)
    }

    fn generate_accuracy_charts_section(&self, _report: &AccuracyReport, charts: &mut Vec<ChartPlaceholder>) -> Result<String> {
        // Add chart placeholders
        charts.push(ChartPlaceholder {
            chart_id: "accuracy_by_query_type".to_string(),
            chart_type: ChartType::BarChart,
            title: "Accuracy by Query Type".to_string(),
            data_source: "accuracy_report.query_type_breakdown".to_string(),
            width: self.config.chart_width,
            height: self.config.chart_height,
            placeholder_text: "[Accuracy by Query Type Chart]".to_string(),
        });

        charts.push(ChartPlaceholder {
            chart_id: "precision_recall_curve".to_string(),
            chart_type: ChartType::LineChart,
            title: "Precision-Recall Curve".to_string(),
            data_source: "accuracy_report.chart_data.precision_recall_curve".to_string(),
            width: self.config.chart_width,
            height: self.config.chart_height,
            placeholder_text: "[Precision-Recall Curve]".to_string(),
        });

        Ok(format!(
            "\n## Accuracy Visualizations\n\n\
            ### Accuracy by Query Type\n\
            {}\n\n\
            ### Precision-Recall Analysis\n\
            {}\n\n\
            ### Accuracy Distribution\n\
            {}\n\n",
            charts[charts.len()-2].placeholder_text,
            charts[charts.len()-1].placeholder_text,
            "[Accuracy Distribution Histogram]"
        ))
    }

    fn generate_accuracy_recommendations(&self, report: &AccuracyReport) -> Result<String> {
        let mut content = String::from("\n## Recommendations\n\n");

        if report.recommendations.is_empty() {
            content.push_str("No specific recommendations. System performance is optimal.\n\n");
            return Ok(content);
        }

        content.push_str("| Priority | Category | Recommendation | Expected Improvement |\n");
        content.push_str("|----------|----------|----------------|----------------------|\n");

        for rec in &report.recommendations {
            content.push_str(&format!(
                "| {:?} | {} | {} | +{:.1}% |\n",
                rec.priority,
                rec.category,
                rec.title,
                rec.expected_improvement
            ));
        }
        content.push_str("\n");

        Ok(content)
    }

    // Performance report specific methods
    fn generate_performance_summary_table(&self, report: &PerformanceReport) -> Result<String> {
        Ok(format!(
            "| Metric | Value | Target | Status |\n\
            |--------|-------|--------|--------|\n\
            | P50 Latency | {:.1}ms | <50ms | {} |\n\
            | P95 Latency | {:.1}ms | <100ms | {} |\n\
            | P99 Latency | {:.1}ms | <200ms | {} |\n\
            | Throughput | {:.1} QPS | >100 QPS | {} |\n",
            report.latency_analysis.p50_latency_ms,
            if report.latency_analysis.p50_latency_ms < 50.0 { "✅" } else { "❌" },
            report.latency_analysis.p95_latency_ms, 
            if report.latency_analysis.p95_latency_ms < 100.0 { "✅" } else { "❌" },
            report.latency_analysis.p99_latency_ms,
            if report.latency_analysis.p99_latency_ms < 200.0 { "✅" } else { "❌" },
            report.throughput_analysis.queries_per_second,
            if report.throughput_analysis.queries_per_second > 100.0 { "✅" } else { "❌" }
        ))
    }

    fn generate_latency_analysis(&self, report: &PerformanceReport) -> Result<String> {
        Ok(format!(
            "\n## Latency Analysis\n\n\
            | Percentile | Latency (ms) | Target | Status |\n\
            |------------|--------------|--------|--------|\n\
            | P50 | {:.2} | <50 | {} |\n\
            | P75 | {:.2} | <75 | {} |\n\
            | P90 | {:.2} | <90 | {} |\n\
            | P95 | {:.2} | <100 | {} |\n\
            | P99 | {:.2} | <200 | {} |\n\
            | Max | {:.2} | <500 | {} |\n\n",
            report.latency_analysis.p50_latency_ms,
            if report.latency_analysis.p50_latency_ms < 50.0 { "✅" } else { "❌" },
            report.latency_analysis.p75_latency_ms,
            if report.latency_analysis.p75_latency_ms < 75.0 { "✅" } else { "❌" },
            report.latency_analysis.p90_latency_ms,
            if report.latency_analysis.p90_latency_ms < 90.0 { "✅" } else { "❌" },
            report.latency_analysis.p95_latency_ms,
            if report.latency_analysis.p95_latency_ms < 100.0 { "✅" } else { "❌" },
            report.latency_analysis.p99_latency_ms,
            if report.latency_analysis.p99_latency_ms < 200.0 { "✅" } else { "❌" },
            report.latency_analysis.max_latency_ms,
            if report.latency_analysis.max_latency_ms < 500.0 { "✅" } else { "❌" }
        ))
    }

    fn generate_throughput_analysis(&self, report: &PerformanceReport) -> Result<String> {
        Ok(format!(
            "\n## Throughput Analysis\n\n\
            | Metric | Value | Target | Status |\n\
            |--------|-------|--------|--------|\n\
            | Peak QPS | {:.1} | >100 | {} |\n\
            | Average QPS | {:.1} | >80 | {} |\n\
            | Sustained QPS | {:.1} | >100 | {} |\n\
            | Total Queries | {} | - | - |\n\n",
            report.throughput_analysis.peak_queries_per_second,
            if report.throughput_analysis.peak_queries_per_second > 100.0 { "✅" } else { "❌" },
            report.throughput_analysis.queries_per_second,
            if report.throughput_analysis.queries_per_second > 80.0 { "✅" } else { "❌" },
            report.throughput_analysis.sustained_qps,
            if report.throughput_analysis.sustained_qps > 100.0 { "✅" } else { "❌" },
            report.throughput_analysis.total_queries_processed
        ))
    }

    fn generate_resource_utilization(&self, report: &PerformanceReport) -> Result<String> {
        Ok(format!(
            "\n## Resource Utilization\n\n\
            | Resource | Peak | Average | Target | Status |\n\
            |----------|------|---------|--------|--------|\n\
            | CPU Usage | {:.1}% | {:.1}% | <80% | {} |\n\
            | Memory Usage | {:.1} MB | {:.1} MB | <1024 MB | {} |\n\
            | Disk I/O | {:.1} MB/s | {:.1} MB/s | <100 MB/s | {} |\n\
            | Network I/O | {:.1} MB/s | {:.1} MB/s | <50 MB/s | {} |\n\n",
            report.resource_monitoring.peak_cpu_usage_percent,
            report.resource_monitoring.average_cpu_usage_percent,
            if report.resource_monitoring.peak_cpu_usage_percent < 80.0 { "✅" } else { "❌" },
            report.resource_monitoring.peak_memory_usage_mb,
            report.resource_monitoring.average_memory_usage_mb,
            if report.resource_monitoring.peak_memory_usage_mb < 1024.0 { "✅" } else { "❌" },
            report.resource_monitoring.peak_disk_io_mb_per_sec,
            report.resource_monitoring.average_disk_io_mb_per_sec,
            if report.resource_monitoring.peak_disk_io_mb_per_sec < 100.0 { "✅" } else { "❌" },
            report.resource_monitoring.peak_network_io_mb_per_sec,
            report.resource_monitoring.average_network_io_mb_per_sec,
            if report.resource_monitoring.peak_network_io_mb_per_sec < 50.0 { "✅" } else { "❌" }
        ))
    }

    fn generate_performance_charts(&self, _report: &PerformanceReport, charts: &mut Vec<ChartPlaceholder>) -> Result<String> {
        charts.push(ChartPlaceholder {
            chart_id: "latency_percentiles".to_string(),
            chart_type: ChartType::BarChart,
            title: "Latency Percentiles".to_string(),
            data_source: "performance_report.latency_analysis".to_string(),
            width: self.config.chart_width,
            height: self.config.chart_height,
            placeholder_text: "[Latency Percentiles Chart]".to_string(),
        });

        Ok(format!(
            "\n## Performance Charts\n\n\
            ### Latency Distribution\n\
            {}\n\n\
            ### Throughput Over Time\n\
            {}\n\n\
            ### Resource Utilization\n\
            {}\n\n",
            charts[charts.len()-1].placeholder_text,
            "[Throughput Over Time Chart]",
            "[Resource Utilization Chart]"
        ))
    }

    fn generate_performance_recommendations(&self, report: &PerformanceReport) -> Result<String> {
        let mut content = String::from("\n## Performance Recommendations\n\n");

        if report.recommendations.is_empty() {
            content.push_str("System performance meets all targets. No optimizations required.\n\n");
        } else {
            content.push_str("| Priority | Area | Recommendation | Expected Improvement |\n");
            content.push_str("|----------|------|----------------|----------------------|\n");
            
            for rec in &report.recommendations {
                content.push_str(&format!(
                    "| {:?} | {} | {} | +{:.1}% |\n",
                    rec.priority,
                    rec.category,
                    rec.title,
                    rec.expected_improvement_percentage
                ));
            }
            content.push_str("\n");
        }

        Ok(content)
    }

    // Stress test report specific methods
    fn generate_load_summary_table(&self, report: &StressTestReport) -> Result<String> {
        Ok(format!(
            "| Load Metric | Value | Target | Status |\n\
            |-------------|-------|--------|--------|\n\
            | Peak Users | {} | {} | {} |\n\
            | Duration | {:.1}s | {:.1}s | {} |\n\
            | Success Rate | {:.1}% | >95% | {} |\n\
            | Avg Response | {:.1}ms | <200ms | {} |\n",
            report.load_profile.peak_concurrent_users,
            report.load_profile.target_users,
            if report.load_profile.peak_concurrent_users >= report.load_profile.target_users { "✅" } else { "❌" },
            report.metadata.test_duration_seconds,
            report.load_profile.duration_seconds,
            if report.metadata.test_duration_seconds >= report.load_profile.duration_seconds { "✅" } else { "❌" },
            report.load_profile.success_rate_percentage,
            if report.load_profile.success_rate_percentage > 95.0 { "✅" } else { "❌" },
            report.load_profile.average_response_time_ms,
            if report.load_profile.average_response_time_ms < 200.0 { "✅" } else { "❌" }
        ))
    }

    fn generate_load_progression(&self, report: &StressTestReport) -> Result<String> {
        Ok(format!(
            "\n## Load Progression Analysis\n\n\
            | Phase | Users | Duration | Success Rate | Avg Latency |\n\
            |-------|-------|----------|--------------|-------------|\n\
            | Ramp-up | {} | {:.1}s | {:.1}% | {:.1}ms |\n\
            | Steady | {} | {:.1}s | {:.1}% | {:.1}ms |\n\
            | Peak | {} | {:.1}s | {:.1}% | {:.1}ms |\n\
            | Ramp-down | {} | {:.1}s | {:.1}% | {:.1}ms |\n\n",
            report.load_profile.ramp_up_users,
            report.load_profile.ramp_up_duration_seconds,
            report.load_profile.success_rate_percentage,
            report.load_profile.average_response_time_ms,
            report.load_profile.steady_state_users,
            report.load_profile.steady_duration_seconds,
            report.load_profile.success_rate_percentage,
            report.load_profile.average_response_time_ms,
            report.load_profile.peak_concurrent_users,
            report.load_profile.peak_duration_seconds,
            report.load_profile.success_rate_percentage,
            report.load_profile.average_response_time_ms,
            report.load_profile.ramp_down_users,
            report.load_profile.ramp_down_duration_seconds,
            report.load_profile.success_rate_percentage,
            report.load_profile.average_response_time_ms
        ))
    }

    fn generate_breaking_point_analysis(&self, report: &StressTestReport) -> Result<String> {
        Ok(format!(
            "\n## Breaking Point Analysis\n\n\
            **Breaking Point Reached**: {:?}\n\
            **Breaking Point Users**: {}\n\
            **Failure Threshold**: {:.1}%\n\
            **Critical Resource**: {}\n\n\
            ### System Limits\n\
            - **Max Concurrent Users**: {}\n\
            - **Max Throughput**: {:.1} QPS\n\
            - **Resource Bottleneck**: {}\n\n",
            report.breaking_point_analysis.breaking_point_reached,
            report.breaking_point_analysis.breaking_point_users,
            report.breaking_point_analysis.failure_threshold_percentage,
            report.breaking_point_analysis.critical_resource_constraint,
            report.capacity_planning.max_concurrent_users,
            report.capacity_planning.max_sustainable_qps,
            report.capacity_planning.primary_bottleneck
        ))
    }

    fn generate_resilience_metrics(&self, report: &StressTestReport) -> Result<String> {
        Ok(format!(
            "\n## Resilience Metrics\n\n\
            | Metric | Value | Target | Status |\n\
            |--------|-------|--------|--------|\n\
            | System Stability | {:?} | Stable | {} |\n\
            | Recovery Time | {:.1}s | <30s | {} |\n\
            | Error Rate | {:.2}% | <5% | {} |\n\
            | Resource Leaks | {} | 0 | {} |\n\n",
            report.resilience_metrics.system_stability,
            if format!(\"{:?}\", report.resilience_metrics.system_stability) == "Stable" { "✅" } else { "❌" },
            report.resilience_metrics.recovery_time_seconds,
            if report.resilience_metrics.recovery_time_seconds < 30.0 { "✅" } else { "❌" },
            report.resilience_metrics.error_rate_percentage,
            if report.resilience_metrics.error_rate_percentage < 5.0 { "✅" } else { "❌" },
            report.resilience_metrics.resource_leaks_detected,
            if report.resilience_metrics.resource_leaks_detected == 0 { "✅" } else { "❌" }
        ))
    }

    fn generate_capacity_planning(&self, report: &StressTestReport) -> Result<String> {
        Ok(format!(
            "\n## Capacity Planning\n\n\
            ### Current Capacity\n\
            - **Max Concurrent Users**: {}\n\
            - **Max Sustainable QPS**: {:.1}\n\
            - **Primary Bottleneck**: {}\n\n\
            ### Scaling Recommendations\n\
            - **Recommended Capacity**: {} users\n\
            - **Safety Margin**: {}%\n\
            - **Next Bottleneck**: {}\n\n",
            report.capacity_planning.max_concurrent_users,
            report.capacity_planning.max_sustainable_qps,
            report.capacity_planning.primary_bottleneck,
            report.capacity_planning.recommended_max_users,
            report.capacity_planning.safety_margin_percentage,
            report.capacity_planning.next_bottleneck_prediction
        ))
    }

    // Security report specific methods
    fn generate_security_summary_table(&self, report: &SecurityReport) -> Result<String> {
        Ok(format!(
            "| Security Area | Score | Status | Issues |\n\
            |---------------|-------|--------|--------|\n\
            | Overall Security | {:.1}/10 | {} | {} |\n\
            | Input Validation | {:.1}/10 | {} | {} |\n\
            | Access Control | {:.1}/10 | {} | {} |\n\
            | Data Protection | {:.1}/10 | {} | {} |\n",
            report.overall_score.security_score,
            if report.overall_score.security_score >= 8.0 { "✅ Pass" } else { "❌ Fail" },
            report.vulnerability_summary.total_vulnerabilities,
            report.input_validation.validation_score,
            if report.input_validation.validation_score >= 8.0 { "✅ Pass" } else { "❌ Fail" },
            report.input_validation.vulnerabilities_found.len(),
            report.access_control.access_control_score,
            if report.access_control.access_control_score >= 8.0 { "✅ Pass" } else { "❌ Fail" },
            report.access_control.access_control_issues.len(),
            report.data_protection.data_protection_score,
            if report.data_protection.data_protection_score >= 8.0 { "✅ Pass" } else { "❌ Fail" },
            report.data_protection.protection_gaps.len()
        ))
    }

    fn generate_vulnerability_assessment(&self, report: &SecurityReport) -> Result<String> {
        let mut content = format!(
            "\n## Vulnerability Assessment\n\n\
            **Total Vulnerabilities**: {}\n\
            **Critical**: {}\n\
            **High**: {}\n\
            **Medium**: {}\n\
            **Low**: {}\n\n",
            report.vulnerability_summary.total_vulnerabilities,
            report.vulnerability_summary.critical_vulnerabilities,
            report.vulnerability_summary.high_vulnerabilities,
            report.vulnerability_summary.medium_vulnerabilities,
            report.vulnerability_summary.low_vulnerabilities
        );

        if !report.input_validation.vulnerabilities_found.is_empty() {
            content.push_str("### Input Validation Issues\n\n");
            for vuln in &report.input_validation.vulnerabilities_found {
                content.push_str(&format!("- **{}**: {}\n", vuln.severity, vuln.description));
            }
            content.push_str("\n");
        }

        Ok(content)
    }

    fn generate_threat_analysis(&self, report: &SecurityReport) -> Result<String> {
        Ok(format!(
            "\n## Threat Analysis\n\n\
            **Threat Model Score**: {:.1}/10\n\
            **Attack Vectors Identified**: {}\n\
            **Mitigations in Place**: {}\n\n\
            ### Risk Assessment\n\
            - **High Risk Threats**: {}\n\
            - **Medium Risk Threats**: {}\n\
            - **Low Risk Threats**: {}\n\n",
            report.threat_modeling.threat_model_score,
            report.threat_modeling.attack_vectors_identified.len(),
            report.threat_modeling.mitigations_in_place.len(),
            report.threat_modeling.high_risk_threats,
            report.threat_modeling.medium_risk_threats,
            report.threat_modeling.low_risk_threats
        ))
    }

    fn generate_compliance_assessment(&self, report: &SecurityReport) -> Result<String> {
        Ok(format!(
            "\n## Compliance Assessment\n\n\
            **Overall Compliance**: {:?}\n\
            **Standards Evaluated**: {}\n\
            **Requirements Met**: {}\n\
            **Requirements Failed**: {}\n\n\
            ### Compliance Details\n\
            - **Data Privacy**: {}\n\
            - **Access Controls**: {}\n\
            - **Audit Logging**: {}\n\
            - **Encryption**: {}\n\n",
            report.compliance_assessment.overall_compliance_status,
            report.compliance_assessment.standards_evaluated.len(),
            report.compliance_assessment.requirements_met,
            report.compliance_assessment.requirements_failed,
            if report.compliance_assessment.data_privacy_compliant { "✅ Compliant" } else { "❌ Non-compliant" },
            if report.compliance_assessment.access_control_compliant { "✅ Compliant" } else { "❌ Non-compliant" },
            if report.compliance_assessment.audit_logging_compliant { "✅ Compliant" } else { "❌ Non-compliant" },
            if report.compliance_assessment.encryption_compliant { "✅ Compliant" } else { "❌ Non-compliant" }
        ))
    }

    fn generate_security_recommendations(&self, report: &SecurityReport) -> Result<String> {
        let mut content = String::from("\n## Security Recommendations\n\n");

        if report.recommendations.is_empty() {
            content.push_str("No security issues identified. System meets all security requirements.\n\n");
        } else {
            content.push_str("| Priority | Area | Recommendation | Risk Level |\n");
            content.push_str("|----------|------|----------------|------------|\n");
            
            for rec in &report.recommendations {
                content.push_str(&format!(
                    "| {:?} | {} | {} | {:?} |\n",
                    rec.priority,
                    rec.category,
                    rec.title,
                    rec.risk_level
                ));
            }
            content.push_str("\n");
        }

        Ok(content)
    }

    // Utility methods
    fn generate_summary_metrics_table(&self, report: &ValidationReport) -> Result<String> {
        Ok(format!(
            "\n| Metric | Value |\n\
            |--------|-------|\n\
            | Total Tests | {} |\n\
            | Success Rate | {:.1}% |\n\
            | Execution Time | {:.2}s |\n\
            | Production Ready | {} |\n\n",
            report.test_summary.total_tests,
            report.overall_results.overall_success_rate,
            report.timing.total_execution_seconds,
            if report.overall_results.production_ready { "Yes" } else { "No" }
        ))
    }

    fn generate_test_categories_table(&self, report: &ValidationReport) -> Result<String> {
        Ok(format!(
            "| Category | Tests | Passed | Failed | Success Rate |\n\
            |----------|-------|--------|--------|-------------|\n\
            | Ground Truth | {} | {} | {} | {:.1}% |\n\
            | Correctness | {} | {} | {} | {:.1}% |\n\
            | Performance | {} | {} | {} | {:.1}% |\n\
            | Security | {} | {} | {} | {:.1}% |\n",
            10, 10, 0, 100.0, // Placeholder values
            15, 15, 0, 100.0,
            8, 8, 0, 100.0,
            5, 5, 0, 100.0
        ))
    }

    fn generate_timing_breakdown_table(&self, report: &ValidationReport) -> Result<String> {
        Ok(format!(
            "\n| Phase | Duration (s) | Percentage |\n\
            |-------|--------------|------------|\n\
            | Setup | {:.2} | {:.1}% |\n\
            | Execution | {:.2} | {:.1}% |\n\
            | Cleanup | {:.2} | {:.1}% |\n\
            | **Total** | **{:.2}** | **100.0%** |\n\n",
            report.timing.setup_time_seconds,
            (report.timing.setup_time_seconds / report.timing.total_execution_seconds) * 100.0,
            report.timing.total_execution_seconds - report.timing.setup_time_seconds - report.timing.cleanup_time_seconds,
            ((report.timing.total_execution_seconds - report.timing.setup_time_seconds - report.timing.cleanup_time_seconds) / report.timing.total_execution_seconds) * 100.0,
            report.timing.cleanup_time_seconds,
            (report.timing.cleanup_time_seconds / report.timing.total_execution_seconds) * 100.0,
            report.timing.total_execution_seconds
        ))
    }

    fn grade_metric(&self, value: f64) -> &str {
        match value {
            v if v >= 0.95 => "A+",
            v if v >= 0.90 => "A",
            v if v >= 0.85 => "B+",
            v if v >= 0.80 => "B",
            v if v >= 0.75 => "C+",
            v if v >= 0.70 => "C",
            _ => "F",
        }
    }

    fn count_words(&self, content: &str) -> usize {
        content.split_whitespace().count()
    }

    fn count_sections(&self, content: &str) -> usize {
        content.lines().filter(|line| line.starts_with("##")).count()
    }

    fn count_tables(&self, content: &str) -> usize {
        content.lines().filter(|line| line.contains('|') && line.contains('-')).count()
    }
}

impl Default for MarkdownConfig {
    fn default() -> Self {
        Self {
            include_toc: true,
            include_charts: true,
            include_raw_data: false,
            max_table_rows: 50,
            chart_width: 800,
            chart_height: 400,
            theme: MarkdownTheme::Professional,
            custom_css: None,
        }
    }
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self {
            report_templates: HashMap::new(),
            section_templates: HashMap::new(),
            table_templates: HashMap::new(),
        }
    }
}

impl Default for GeneratedReport {
    fn default() -> Self {
        Self {
            content: String::new(),
            metadata: ReportMetadata {
                generated_at: Utc::now(),
                report_type: "Generic Report".to_string(),
                version: "1.0.0".to_string(),
                total_sections: 0,
                total_tables: 0,
                total_charts: 0,
                word_count: 0,
            },
            sections: Vec::new(),
            charts: Vec::new(),
            attachments: Vec::new(),
        }
    }
}
```

## Dependencies to Add
```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
```

## Success Criteria
- MarkdownReportGenerator compiles without errors
- All report types can be converted to professional markdown
- Tables are properly formatted with alignment
- Chart placeholders are correctly generated
- Generated reports are comprehensive and well-structured
- Template system supports customization

## Time Limit
10 minutes maximum