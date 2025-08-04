# Task 039: Implement Baseline Reporting

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The baseline reporting system generates comprehensive reports with visual performance comparison charts, detailed analysis, and export capabilities in multiple formats (Markdown, JSON, HTML).

## Project Structure
```
src/
  validation/
    baseline.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement comprehensive baseline reporting that generates professional-quality reports with performance comparison charts, statistical summaries, recommendations, and multi-format export capabilities. The reports should be suitable for both technical analysis and executive summaries.

## Requirements
1. Extend `src/validation/baseline.rs` with reporting functionality
2. Generate visual performance comparison charts (ASCII and HTML)
3. Create detailed analysis sections with statistical insights
4. Implement multi-format export (Markdown, JSON, HTML)
5. Add executive summary generation
6. Support customizable report templates
7. Include trend visualization and recommendation prioritization

## Expected Code Structure
```rust
// Add to baseline.rs file

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineReport {
    pub report_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub executive_summary: ExecutiveSummary,
    pub methodology: ReportMethodology,
    pub performance_analysis: PerformanceAnalysis,
    pub statistical_analysis: StatisticalAnalysis,
    pub recommendations: RecommendationSection,
    pub raw_data: RawDataSection,
    pub appendices: Vec<ReportAppendix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    pub key_findings: Vec<String>,
    pub performance_winners: BTreeMap<String, BaselineTool>,
    pub critical_recommendations: Vec<String>,
    pub overall_conclusion: String,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMethodology {
    pub test_environment: TestEnvironment,
    pub test_parameters: TestParameters,
    pub tools_tested: Vec<BaselineTool>,
    pub data_collection_approach: String,
    pub statistical_methods: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEnvironment {
    pub operating_system: String,
    pub cpu_info: String,
    pub memory_gb: f64,
    pub storage_type: String,
    pub test_data_characteristics: TestDataCharacteristics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataCharacteristics {
    pub total_files: usize,
    pub total_size_mb: f64,
    pub file_types: Vec<String>,
    pub average_file_size_kb: f64,
    pub largest_file_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestParameters {
    pub queries_tested: usize,
    pub warmup_runs: usize,
    pub measurement_runs: usize,
    pub timeout_seconds: u64,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub latency_analysis: MetricAnalysis,
    pub throughput_analysis: MetricAnalysis,
    pub memory_usage_analysis: MetricAnalysis,
    pub reliability_analysis: MetricAnalysis,
    pub overall_rankings: Vec<OverallRanking>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricAnalysis {
    pub metric_name: String,
    pub best_performer: BaselineTool,
    pub worst_performer: BaselineTool,
    pub performance_spread: f64,
    pub statistical_significance: bool,
    pub rankings: Vec<MetricRanking>,
    pub chart_data: ChartData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRanking {
    pub tool: BaselineTool,
    pub value: f64,
    pub rank: usize,
    pub percentile: f64,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub chart_type: ChartType,
    pub data_points: Vec<DataPoint>,
    pub ascii_chart: String,
    pub html_chart: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    BarChart,
    BoxPlot,
    ScatterPlot,
    LineChart,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub label: String,
    pub value: f64,
    pub confidence_interval: Option<(f64, f64)>,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallRanking {
    pub tool: BaselineTool,
    pub overall_score: f64,
    pub rank: usize,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub use_cases: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    pub significance_tests: Vec<StatisticalTestSummary>,
    pub effect_sizes: Vec<EffectSizeSummary>,
    pub confidence_intervals: Vec<ConfidenceIntervalSummary>,
    pub power_analysis: PowerAnalysisSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestSummary {
    pub test_name: String,
    pub comparison: String,
    pub p_value: f64,
    pub significant: bool,
    pub interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeSummary {
    pub comparison: String,
    pub effect_size: f64,
    pub magnitude: String,
    pub practical_significance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervalSummary {
    pub metric: String,
    pub tool: BaselineTool,
    pub point_estimate: f64,
    pub confidence_interval: (f64, f64),
    pub margin_of_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysisSummary {
    pub average_power: f64,
    pub minimum_power: f64,
    pub underpowered_tests: usize,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationSection {
    pub critical_actions: Vec<FormattedRecommendation>,
    pub performance_optimizations: Vec<FormattedRecommendation>,
    pub resource_optimizations: Vec<FormattedRecommendation>,
    pub reliability_improvements: Vec<FormattedRecommendation>,
    pub implementation_roadmap: Vec<RoadmapItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattedRecommendation {
    pub title: String,
    pub priority: String,
    pub description: String,
    pub rationale: String,
    pub expected_impact: String,
    pub implementation_effort: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadmapItem {
    pub phase: String,
    pub timeline: String,
    pub actions: Vec<String>,
    pub success_criteria: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDataSection {
    pub summary_statistics: BTreeMap<BaselineTool, ToolSummaryStats>,
    pub detailed_measurements: Option<String>, // Path to detailed CSV
    pub test_configuration: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSummaryStats {
    pub measurements: usize,
    pub success_rate: f64,
    pub mean_latency: f64,
    pub median_latency: f64,
    pub p95_latency: f64,
    pub throughput_qps: f64,
    pub memory_usage_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportAppendix {
    pub title: String,
    pub content_type: AppendixType,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AppendixType {
    StatisticalDetails,
    RawDataDump,
    Configuration,
    Methodology,
    Glossary,
}

impl BaselineComparison {
    pub fn generate_comprehensive_report(&self) -> Result<BaselineReport> {
        let report_id = format!("baseline_report_{}", 
            chrono::Utc::now().format("%Y%m%d_%H%M%S"));
        
        let report = BaselineReport {
            report_id: report_id.clone(),
            timestamp: chrono::Utc::now(),
            executive_summary: self.generate_executive_summary()?,
            methodology: self.generate_methodology_section()?,
            performance_analysis: self.generate_performance_analysis()?,
            statistical_analysis: self.generate_statistical_analysis()?,
            recommendations: self.generate_recommendation_section()?,
            raw_data: self.generate_raw_data_section()?,
            appendices: self.generate_appendices()?,
        };
        
        Ok(report)
    }
    
    fn generate_executive_summary(&self) -> Result<ExecutiveSummary> {
        let mut key_findings = Vec::new();
        let mut performance_winners = BTreeMap::new();
        let mut critical_recommendations = Vec::new();
        
        // Identify performance winners for each metric
        let metrics = ["Latency", "Throughput", "Memory Usage", "Reliability"];
        for metric in &metrics {
            if let Some(winner) = self.get_best_performer_for_metric(metric) {
                performance_winners.insert(metric.to_string(), winner);
                key_findings.push(format!("{} performs best in {}", 
                    format!("{:?}", winner), metric.to_lowercase()));
            }
        }
        
        // Extract critical recommendations
        critical_recommendations = self.recommendations.iter()
            .filter(|r| matches!(r.priority, RecommendationPriority::Critical | RecommendationPriority::High))
            .take(3)
            .map(|r| r.title.clone())
            .collect();
        
        // Generate overall conclusion
        let total_tests = self.statistical_tests.len();
        let significant_tests = self.statistical_tests.iter()
            .filter(|t| t.significant)
            .count();
        
        let overall_conclusion = format!(
            "Analysis of {} baseline tools across {} metrics revealed {} statistically significant performance differences. \
             {} shows the most consistent performance across metrics, while {} recommendations require immediate attention.",
            self.get_unique_tools().len(),
            metrics.len(),
            significant_tests,
            self.get_most_consistent_performer().map(|t| format!("{:?}", t)).unwrap_or("No tool".to_string()),
            critical_recommendations.len()
        );
        
        let confidence_score = if total_tests > 0 {
            (significant_tests as f64 / total_tests as f64) * 0.8 + 0.2
        } else {
            0.5
        };
        
        Ok(ExecutiveSummary {
            key_findings,
            performance_winners,
            critical_recommendations,
            overall_conclusion,
            confidence_score,
        })
    }
    
    fn get_best_performer_for_metric(&self, metric: &str) -> Option<BaselineTool> {
        match metric {
            "Latency" => self.comparative_analysis.fastest_tool.into(),
            "Memory Usage" => self.comparative_analysis.memory_efficient_tool.into(),
            "Reliability" => self.comparative_analysis.most_reliable_tool.into(),
            "Throughput" => {
                // Find tool with highest throughput
                self.baseline_results.iter()
                    .flat_map(|br| &br.tool_results)
                    .max_by(|a, b| a.1.throughput_qps.partial_cmp(&b.1.throughput_qps).unwrap())
                    .map(|(tool, _)| *tool)
            },
            _ => None,
        }
    }
    
    fn get_most_consistent_performer(&self) -> Option<BaselineTool> {
        // Find tool with best average ranking across all metrics
        let mut tool_rankings: BTreeMap<BaselineTool, Vec<usize>> = BTreeMap::new();
        
        for ranking in &self.performance_rankings {
            tool_rankings.entry(ranking.tool)
                .or_default()
                .push(ranking.rank);
        }
        
        tool_rankings.into_iter()
            .map(|(tool, ranks)| {
                let avg_rank = ranks.iter().sum::<usize>() as f64 / ranks.len() as f64;
                (tool, avg_rank)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(tool, _)| tool)
    }
    
    fn generate_methodology_section(&self) -> Result<ReportMethodology> {
        let test_env = if let Some(first_result) = self.baseline_results.first() {
            TestEnvironment {
                operating_system: first_result.system_info.os.clone(),
                cpu_info: format!("{} cores", first_result.system_info.cpu_cores),
                memory_gb: first_result.system_info.memory_gb,
                storage_type: "Standard".to_string(), // Could be enhanced
                test_data_characteristics: TestDataCharacteristics {
                    total_files: first_result.system_info.file_count,
                    total_size_mb: first_result.system_info.test_data_size_mb,
                    file_types: vec!["txt".to_string(), "md".to_string(), "rs".to_string()], // Example
                    average_file_size_kb: if first_result.system_info.file_count > 0 {
                        (first_result.system_info.test_data_size_mb * 1024.0) / first_result.system_info.file_count as f64
                    } else {
                        0.0
                    },
                    largest_file_mb: 0.0, // Would need to be calculated
                },
            }
        } else {
            TestEnvironment {
                operating_system: "Unknown".to_string(),
                cpu_info: "Unknown".to_string(),
                memory_gb: 0.0,
                storage_type: "Unknown".to_string(),
                test_data_characteristics: TestDataCharacteristics {
                    total_files: 0,
                    total_size_mb: 0.0,
                    file_types: vec![],
                    average_file_size_kb: 0.0,
                    largest_file_mb: 0.0,
                },
            }
        };
        
        let total_queries = self.baseline_results.iter()
            .flat_map(|br| br.tool_results.values())
            .map(|tr| tr.total_queries)
            .max()
            .unwrap_or(0);
        
        Ok(ReportMethodology {
            test_environment: test_env,
            test_parameters: TestParameters {
                queries_tested: total_queries,
                warmup_runs: 3, // Default from config
                measurement_runs: 10, // Default from config
                timeout_seconds: 30, // Default from config
                confidence_level: 0.95,
            },
            tools_tested: self.get_unique_tools(),
            data_collection_approach: "Controlled benchmarking with statistical validation".to_string(),
            statistical_methods: vec![
                "Welch's t-test".to_string(),
                "Mann-Whitney U test".to_string(),
                "Effect size calculation (Cohen's d)".to_string(),
                "Confidence intervals".to_string(),
            ],
        })
    }
    
    fn generate_performance_analysis(&self) -> Result<PerformanceAnalysis> {
        let latency_analysis = self.generate_metric_analysis("Latency")?;
        let throughput_analysis = self.generate_metric_analysis("Throughput")?;
        let memory_analysis = self.generate_metric_analysis("Memory Usage")?;
        let reliability_analysis = self.generate_metric_analysis("Reliability")?;
        
        let overall_rankings = self.generate_overall_rankings()?;
        
        Ok(PerformanceAnalysis {
            latency_analysis,
            throughput_analysis,
            memory_usage_analysis: memory_analysis,
            reliability_analysis,
            overall_rankings,
        })
    }
    
    fn generate_metric_analysis(&self, metric_name: &str) -> Result<MetricAnalysis> {
        let metric = match metric_name {
            "Latency" => ComparisonMetric::Latency,
            "Throughput" => ComparisonMetric::Throughput,
            "Memory Usage" => ComparisonMetric::MemoryUsage,
            "Reliability" => ComparisonMetric::SuccessRate,
            _ => bail!("Unknown metric: {}", metric_name),
        };
        
        let rankings: Vec<_> = self.performance_rankings.iter()
            .filter(|r| std::mem::discriminant(&r.metric) == std::mem::discriminant(&metric))
            .collect();
        
        let best_performer = rankings.iter()
            .min_by_key(|r| r.rank)
            .map(|r| r.tool)
            .unwrap_or(BaselineTool::Ripgrep);
        
        let worst_performer = rankings.iter()
            .max_by_key(|r| r.rank)
            .map(|r| r.tool)
            .unwrap_or(BaselineTool::Ripgrep);
        
        let performance_spread = if let (Some(best), Some(worst)) = (
            rankings.iter().min_by(|a, b| a.score.partial_cmp(&b.score).unwrap()),
            rankings.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        ) {
            ((worst.score - best.score) / best.score).abs()
        } else {
            0.0
        };
        
        let statistical_significance = self.statistical_tests.iter()
            .any(|t| std::mem::discriminant(&t.metric) == std::mem::discriminant(&metric) && t.significant);
        
        let metric_rankings: Vec<MetricRanking> = rankings.iter()
            .map(|r| MetricRanking {
                tool: r.tool,
                value: r.score,
                rank: r.rank,
                percentile: r.percentile,
                confidence_interval: r.confidence_interval,
            })
            .collect();
        
        let chart_data = self.generate_chart_data(&metric_rankings, metric_name)?;
        
        Ok(MetricAnalysis {
            metric_name: metric_name.to_string(),
            best_performer,
            worst_performer,
            performance_spread,
            statistical_significance,
            rankings: metric_rankings,
            chart_data,
        })
    }
    
    fn generate_chart_data(&self, rankings: &[MetricRanking], metric_name: &str) -> Result<ChartData> {
        let data_points: Vec<DataPoint> = rankings.iter()
            .map(|r| DataPoint {
                label: format!("{:?}", r.tool),
                value: r.value,
                confidence_interval: Some(r.confidence_interval),
                color: self.get_tool_color(r.tool),
            })
            .collect();
        
        let ascii_chart = self.generate_ascii_chart(&data_points, metric_name);
        let html_chart = self.generate_html_chart(&data_points, metric_name);
        
        Ok(ChartData {
            chart_type: ChartType::BarChart,
            data_points,
            ascii_chart,
            html_chart,
        })
    }
    
    fn get_tool_color(&self, tool: BaselineTool) -> String {
        match tool {
            BaselineTool::Ripgrep => "#FF6B6B".to_string(),
            BaselineTool::FindGrep => "#4ECDC4".to_string(),
            BaselineTool::TantivyStandalone => "#45B7D1".to_string(),
            BaselineTool::SystemSearch => "#96CEB4".to_string(),
            BaselineTool::WindowsPowerShell => "#FFEAA7".to_string(),
            BaselineTool::WindowsGrep => "#DDA0DD".to_string(),
        }
    }
    
    fn generate_ascii_chart(&self, data_points: &[DataPoint], metric_name: &str) -> String {
        if data_points.is_empty() {
            return "No data available".to_string();
        }
        
        let max_value = data_points.iter()
            .map(|dp| dp.value)
            .fold(0.0f64, |a, b| a.max(b));
        
        if max_value == 0.0 {
            return "No valid measurements".to_string();
        }
        
        let mut chart = format!("\n{} Performance Comparison\n", metric_name);
        chart.push_str(&"=".repeat(50));
        chart.push('\n');
        
        for dp in data_points {
            let bar_length = ((dp.value / max_value) * 40.0) as usize;
            let bar = "█".repeat(bar_length);
            chart.push_str(&format!(
                "{:<15} |{:<40} {:.2}\n",
                dp.label,
                bar,
                dp.value
            ));
        }
        
        chart
    }
    
    fn generate_html_chart(&self, data_points: &[DataPoint], metric_name: &str) -> String {
        let mut html = String::new();
        html.push_str(&format!(r#"
<div class="chart-container">
    <h3>{} Performance Comparison</h3>
    <svg width="600" height="400" viewBox="0 0 600 400">
"#, metric_name));
        
        let max_value = data_points.iter()
            .map(|dp| dp.value)
            .fold(0.0f64, |a, b| a.max(b));
        
        let bar_width = 80;
        let bar_spacing = 100;
        
        for (i, dp) in data_points.iter().enumerate() {
            let x = 50 + i * bar_spacing;
            let height = if max_value > 0.0 {
                ((dp.value / max_value) * 300.0) as usize
            } else {
                0
            };
            let y = 350 - height;
            
            html.push_str(&format!(r#"
        <rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="0.8"/>
        <text x="{}" y="375" text-anchor="middle" font-size="12">{}</text>
        <text x="{}" y="{}" text-anchor="middle" font-size="10">{:.2}</text>
"#, 
                x, y, bar_width, height, dp.color,
                x + bar_width / 2, dp.label,
                x + bar_width / 2, y - 5, dp.value
            ));
        }
        
        html.push_str(r#"
    </svg>
</div>
"#);
        
        html
    }
    
    fn generate_overall_rankings(&self) -> Result<Vec<OverallRanking>> {
        let tools = self.get_unique_tools();
        let mut rankings = Vec::new();
        
        for tool in tools {
            let tool_rankings: Vec<_> = self.performance_rankings.iter()
                .filter(|r| r.tool == tool)
                .collect();
            
            let overall_score = if !tool_rankings.is_empty() {
                let avg_rank = tool_rankings.iter()
                    .map(|r| r.rank as f64)
                    .sum::<f64>() / tool_rankings.len() as f64;
                // Convert rank to score (lower rank = higher score)
                (tool_rankings.len() as f64 + 1.0 - avg_rank) / tool_rankings.len() as f64
            } else {
                0.0
            };
            
            let strengths = self.identify_tool_strengths(tool);
            let weaknesses = self.identify_tool_weaknesses(tool);
            let use_cases = self.suggest_use_cases(tool, &strengths);
            
            rankings.push(OverallRanking {
                tool,
                overall_score,
                rank: 0, // Will be set after sorting
                strengths,
                weaknesses,
                use_cases,
            });
        }
        
        // Sort by overall score and assign ranks
        rankings.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());
        for (i, ranking) in rankings.iter_mut().enumerate() {
            ranking.rank = i + 1;
        }
        
        Ok(rankings)
    }
    
    fn identify_tool_strengths(&self, tool: BaselineTool) -> Vec<String> {
        let mut strengths = Vec::new();
        
        // Check if tool is best in any metric
        if self.comparative_analysis.fastest_tool == tool {
            strengths.push("Fastest search performance".to_string());
        }
        if self.comparative_analysis.most_reliable_tool == tool {
            strengths.push("Highest reliability".to_string());
        }
        if self.comparative_analysis.memory_efficient_tool == tool {
            strengths.push("Most memory efficient".to_string());
        }
        
        // Check rankings
        for ranking in &self.performance_rankings {
            if ranking.tool == tool && ranking.rank <= 2 {
                let metric_name = match ranking.metric {
                    ComparisonMetric::Latency => "Low latency",
                    ComparisonMetric::Throughput => "High throughput",
                    ComparisonMetric::MemoryUsage => "Low memory usage",
                    ComparisonMetric::SuccessRate => "High success rate",
                    ComparisonMetric::IndexTime => "Fast indexing",
                };
                strengths.push(metric_name.to_string());
            }
        }
        
        if strengths.is_empty() {
            strengths.push("Consistent performance".to_string());
        }
        
        strengths
    }
    
    fn identify_tool_weaknesses(&self, tool: BaselineTool) -> Vec<String> {
        let mut weaknesses = Vec::new();
        
        for ranking in &self.performance_rankings {
            if ranking.tool == tool && ranking.rank >= self.get_unique_tools().len() - 1 {
                let metric_name = match ranking.metric {
                    ComparisonMetric::Latency => "High latency",
                    ComparisonMetric::Throughput => "Low throughput", 
                    ComparisonMetric::MemoryUsage => "High memory usage",
                    ComparisonMetric::SuccessRate => "Low success rate",
                    ComparisonMetric::IndexTime => "Slow indexing",
                };
                weaknesses.push(metric_name.to_string());
            }
        }
        
        if weaknesses.is_empty() {
            weaknesses.push("No significant weaknesses identified".to_string());
        }
        
        weaknesses
    }
    
    fn suggest_use_cases(&self, tool: BaselineTool, strengths: &[String]) -> Vec<String> {
        let mut use_cases = Vec::new();
        
        match tool {
            BaselineTool::Ripgrep => {
                use_cases.push("Large codebase searching".to_string());
                use_cases.push("Fast text processing pipelines".to_string());
            },
            BaselineTool::TantivyStandalone => {
                use_cases.push("Full-text search applications".to_string());
                use_cases.push("Document indexing systems".to_string());
            },
            BaselineTool::FindGrep => {
                use_cases.push("Simple text searching".to_string());
                use_cases.push("System administration tasks".to_string());
            },
            BaselineTool::SystemSearch => {
                use_cases.push("Basic file searching".to_string());
                use_cases.push("Cross-platform compatibility".to_string());
            },
            BaselineTool::WindowsPowerShell => {
                use_cases.push("Windows system administration".to_string());
                use_cases.push("Scripted search operations".to_string());
            },
            BaselineTool::WindowsGrep => {
                use_cases.push("Windows text processing".to_string());
                use_cases.push("Git-based development workflows".to_string());
            },
        }
        
        // Add strength-based use cases
        for strength in strengths {
            if strength.contains("memory efficient") {
                use_cases.push("Resource-constrained environments".to_string());
            }
            if strength.contains("Fast") || strength.contains("fast") {
                use_cases.push("Time-critical applications".to_string());
            }
            if strength.contains("reliable") || strength.contains("reliability") {
                use_cases.push("Production systems requiring high uptime".to_string());
            }
        }
        
        use_cases.dedup();
        use_cases
    }
    
    fn generate_statistical_analysis(&self) -> Result<StatisticalAnalysis> {
        let significance_tests = self.statistical_tests.iter()
            .map(|test| StatisticalTestSummary {
                test_name: format!("{:?}", test.test_type),
                comparison: format!("{:?} vs {:?}", test.tool_a, test.tool_b),
                p_value: test.p_value,
                significant: test.significant,
                interpretation: test.interpretation.clone(),
            })
            .collect();
        
        let effect_sizes = self.statistical_tests.iter()
            .map(|test| EffectSizeSummary {
                comparison: format!("{:?} vs {:?}", test.tool_a, test.tool_b),
                effect_size: test.effect_size,
                magnitude: match test.effect_size {
                    x if x < 0.2 => "Negligible".to_string(),
                    x if x < 0.5 => "Small".to_string(),
                    x if x < 0.8 => "Medium".to_string(),
                    _ => "Large".to_string(),
                },
                practical_significance: test.effect_size > 0.5,
            })
            .collect();
        
        let confidence_intervals = self.performance_rankings.iter()
            .take(20) // Limit to avoid excessive data
            .map(|ranking| ConfidenceIntervalSummary {
                metric: format!("{:?}", ranking.metric),
                tool: ranking.tool,
                point_estimate: ranking.score,
                confidence_interval: ranking.confidence_interval,
                margin_of_error: (ranking.confidence_interval.1 - ranking.confidence_interval.0) / 2.0,
            })
            .collect();
        
        let average_power = if !self.statistical_tests.is_empty() {
            self.statistical_tests.iter().map(|t| t.power).sum::<f64>() / self.statistical_tests.len() as f64
        } else {
            0.0
        };
        
        let minimum_power = self.statistical_tests.iter()
            .map(|t| t.power)
            .fold(1.0f64, |a, b| a.min(b));
        
        let underpowered_tests = self.statistical_tests.iter()
            .filter(|t| t.power < 0.8)
            .count();
        
        let power_recommendations = if underpowered_tests > 0 {
            vec![
                format!("Increase sample size for {} underpowered tests", underpowered_tests),
                "Consider larger effect sizes for practical significance".to_string(),
            ]
        } else {
            vec!["Statistical power is adequate for all tests".to_string()]
        };
        
        Ok(StatisticalAnalysis {
            significance_tests,
            effect_sizes,
            confidence_intervals,
            power_analysis: PowerAnalysisSummary {
                average_power,
                minimum_power,
                underpowered_tests,
                recommendations: power_recommendations,
            },
        })
    }
    
    fn generate_recommendation_section(&self) -> Result<RecommendationSection> {
        let mut critical_actions = Vec::new();
        let mut performance_optimizations = Vec::new();
        let mut resource_optimizations = Vec::new();
        let mut reliability_improvements = Vec::new();
        
        for rec in &self.recommendations {
            let formatted = FormattedRecommendation {
                title: rec.title.clone(),
                priority: format!("{:?}", rec.priority),
                description: rec.description.clone(),
                rationale: rec.supporting_evidence.join("; "),
                expected_impact: rec.potential_impact.clone(),
                implementation_effort: rec.implementation_complexity.clone(),
                confidence: rec.confidence,
            };
            
            match rec.category {
                RecommendationCategory::Performance => performance_optimizations.push(formatted),
                RecommendationCategory::ResourceUsage => resource_optimizations.push(formatted),
                RecommendationCategory::Reliability => reliability_improvements.push(formatted),
                _ => {
                    if matches!(rec.priority, RecommendationPriority::Critical | RecommendationPriority::High) {
                        critical_actions.push(formatted);
                    }
                }
            }
        }
        
        let implementation_roadmap = vec![
            RoadmapItem {
                phase: "Phase 1: Critical Issues".to_string(),
                timeline: "Immediate (0-2 weeks)".to_string(),
                actions: critical_actions.iter().take(3).map(|r| r.title.clone()).collect(),
                success_criteria: vec!["All critical issues resolved".to_string()],
            },
            RoadmapItem {
                phase: "Phase 2: Performance Optimization".to_string(),
                timeline: "Short-term (2-6 weeks)".to_string(),
                actions: performance_optimizations.iter().take(3).map(|r| r.title.clone()).collect(),
                success_criteria: vec!["Measurable performance improvement".to_string()],
            },
            RoadmapItem {
                phase: "Phase 3: Resource Optimization".to_string(),
                timeline: "Medium-term (1-3 months)".to_string(),
                actions: resource_optimizations.iter().take(3).map(|r| r.title.clone()).collect(),
                success_criteria: vec!["Optimized resource utilization".to_string()],
            },
        ];
        
        Ok(RecommendationSection {
            critical_actions,
            performance_optimizations,
            resource_optimizations,
            reliability_improvements,
            implementation_roadmap,
        })
    }
    
    fn generate_raw_data_section(&self) -> Result<RawDataSection> {
        let mut summary_statistics = BTreeMap::new();
        
        for baseline_result in &self.baseline_results {
            for (tool, tool_result) in &baseline_result.tool_results {
                let stats = ToolSummaryStats {
                    measurements: tool_result.individual_results.len(),
                    success_rate: tool_result.success_rate,
                    mean_latency: tool_result.average_latency_ms,
                    median_latency: tool_result.median_latency_ms,
                    p95_latency: tool_result.p95_latency_ms,
                    throughput_qps: tool_result.throughput_qps,
                    memory_usage_mb: tool_result.average_memory_mb,
                };
                
                summary_statistics.insert(*tool, stats);
            }
        }
        
        Ok(RawDataSection {
            summary_statistics,
            detailed_measurements: None, // Could be implemented to save CSV
            test_configuration: "Standard baseline benchmark configuration".to_string(),
        })
    }
    
    fn generate_appendices(&self) -> Result<Vec<ReportAppendix>> {
        let mut appendices = Vec::new();
        
        // Statistical details appendix
        let statistical_details = self.format_statistical_details();
        appendices.push(ReportAppendix {
            title: "Statistical Analysis Details".to_string(),
            content_type: AppendixType::StatisticalDetails,
            content: statistical_details,
        });
        
        // Glossary appendix
        let glossary = self.generate_glossary();
        appendices.push(ReportAppendix {
            title: "Glossary of Terms".to_string(),
            content_type: AppendixType::Glossary,
            content: glossary,
        });
        
        Ok(appendices)
    }
    
    fn format_statistical_details(&self) -> String {
        let mut details = String::new();
        
        details.push_str("## Statistical Test Details\n\n");
        for test in &self.statistical_tests {
            details.push_str(&format!(
                "### {} - {} vs {}\n\n",
                format!("{:?}", test.test_type),
                format!("{:?}", test.tool_a),
                format!("{:?}", test.tool_b)
            ));
            details.push_str(&format!("- Test Statistic: {:.4}\n", test.test_statistic));
            details.push_str(&format!("- P-value: {:.6}\n", test.p_value));
            details.push_str(&format!("- Degrees of Freedom: {:.2}\n", test.degrees_of_freedom));
            details.push_str(&format!("- Effect Size: {:.4}\n", test.effect_size));
            details.push_str(&format!("- Statistical Power: {:.3}\n", test.power));
            details.push_str(&format!("- Significant: {}\n\n", test.significant));
        }
        
        details
    }
    
    fn generate_glossary(&self) -> String {
        r#"## Glossary

**Effect Size**: A measure of the magnitude of difference between groups, independent of sample size.

**P-value**: The probability of obtaining test results at least as extreme as observed, assuming the null hypothesis is true.

**Confidence Interval**: A range of values that likely contains the true population parameter with a specified level of confidence.

**Statistical Power**: The probability of correctly rejecting a false null hypothesis (avoiding Type II error).

**Welch's t-test**: A statistical test for comparing means of two groups with potentially unequal variances.

**Mann-Whitney U test**: A non-parametric test for comparing two independent samples.

**Throughput**: The number of queries processed per second (QPS).

**Latency**: The time taken to process a single query, measured in milliseconds.

**Percentile**: A value below which a given percentage of observations fall (e.g., P95 = 95th percentile).
"#.to_string()
    }
    
    // Export methods
    pub fn export_report_markdown(&self, report: &BaselineReport, path: impl AsRef<Path>) -> Result<()> {
        let markdown = self.format_report_as_markdown(report)?;
        fs::write(path, markdown)?;
        Ok(())
    }
    
    pub fn export_report_html(&self, report: &BaselineReport, path: impl AsRef<Path>) -> Result<()> {
        let html = self.format_report_as_html(report)?;
        fs::write(path, html)?;
        Ok(())
    }
    
    pub fn export_report_json(&self, report: &BaselineReport, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(report)?;
        fs::write(path, json)?;
        Ok(())
    }
    
    fn format_report_as_markdown(&self, report: &BaselineReport) -> Result<String> {
        let mut md = String::new();
        
        md.push_str(&format!("# Baseline Performance Report\n\n"));
        md.push_str(&format!("**Report ID:** {}\n", report.report_id));
        md.push_str(&format!("**Generated:** {}\n\n", report.timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        
        // Executive Summary
        md.push_str("## Executive Summary\n\n");
        md.push_str(&format!("**Overall Conclusion:** {}\n\n", report.executive_summary.overall_conclusion));
        md.push_str("### Key Findings\n\n");
        for finding in &report.executive_summary.key_findings {
            md.push_str(&format!("- {}\n", finding));
        }
        
        md.push_str("\n### Performance Winners\n\n");
        for (metric, tool) in &report.executive_summary.performance_winners {
            md.push_str(&format!("- **{}:** {:?}\n", metric, tool));
        }
        
        // Performance Analysis
        md.push_str("\n## Performance Analysis\n\n");
        
        // Add latency analysis
        md.push_str("### Latency Analysis\n\n");
        md.push_str(&report.performance_analysis.latency_analysis.ascii_chart);
        md.push_str("\n\n");
        
        // Recommendations
        md.push_str("## Recommendations\n\n");
        md.push_str("### Critical Actions\n\n");
        for (i, rec) in report.recommendations.critical_actions.iter().enumerate() {
            md.push_str(&format!("{}. **{}** (Priority: {})\n", i + 1, rec.title, rec.priority));
            md.push_str(&format!("   - {}\n", rec.description));
            md.push_str(&format!("   - Expected Impact: {}\n", rec.expected_impact));
            md.push_str(&format!("   - Confidence: {:.1}%\n\n", rec.confidence * 100.0));
        }
        
        Ok(md)
    }
    
    fn format_report_as_html(&self, report: &BaselineReport) -> Result<String> {
        let mut html = String::new();
        
        html.push_str(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Baseline Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .metric { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .recommendation { background: #e8f4fd; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }
        .critical { border-left-color: #f44336; background: #ffebee; }
        .chart-container { margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
"#);
        
        html.push_str(&format!(r#"
    <div class="header">
        <h1>Baseline Performance Report</h1>
        <p><strong>Report ID:</strong> {}</p>
        <p><strong>Generated:</strong> {}</p>
    </div>
"#, report.report_id, report.timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        
        // Executive Summary
        html.push_str(r#"
    <div class="section">
        <h2>Executive Summary</h2>
"#);
        html.push_str(&format!("<p><strong>Overall Conclusion:</strong> {}</p>", report.executive_summary.overall_conclusion));
        
        html.push_str("<h3>Key Findings</h3><ul>");
        for finding in &report.executive_summary.key_findings {
            html.push_str(&format!("<li>{}</li>", finding));
        }
        html.push_str("</ul>");
        
        // Performance Analysis with Charts
        html.push_str(r#"
    </div>
    <div class="section">
        <h2>Performance Analysis</h2>
"#);
        
        // Include HTML charts
        html.push_str(&report.performance_analysis.latency_analysis.chart_data.html_chart);
        
        // Recommendations
        html.push_str(r#"
    </div>
    <div class="section">
        <h2>Recommendations</h2>
        <h3>Critical Actions</h3>
"#);
        
        for rec in &report.recommendations.critical_actions.take(5) {
            html.push_str(&format!(r#"
        <div class="recommendation critical">
            <h4>{}</h4>
            <p><strong>Priority:</strong> {}</p>
            <p>{}</p>
            <p><strong>Expected Impact:</strong> {}</p>
            <p><strong>Confidence:</strong> {:.1}%</p>
        </div>
"#, rec.title, rec.priority, rec.description, rec.expected_impact, rec.confidence * 100.0));
        }
        
        html.push_str(r#"
    </div>
</body>
</html>
"#);
        
        Ok(html)
    }
}
```

## Success Criteria
- Report generation creates comprehensive, professional-quality documents
- ASCII charts provide clear visual performance comparisons
- HTML charts render correctly with proper styling
- Multi-format export (Markdown, JSON, HTML) works without errors
- Executive summary captures key insights accurately
- Statistical analysis section includes all relevant test details
- Recommendations are actionable and properly prioritized
- Report structure is logical and easy to navigate

## Time Limit
10 minutes maximum