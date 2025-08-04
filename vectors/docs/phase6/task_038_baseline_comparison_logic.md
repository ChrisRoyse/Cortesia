# Task 038: Implement Baseline Comparison Logic

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The baseline comparison logic performs statistical analysis between different baseline tools, detects performance improvements/degradations, and generates actionable recommendations based on comparative results.

## Project Structure
```
src/
  validation/
    baseline.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement comprehensive comparison logic that analyzes baseline results, performs statistical significance testing, calculates confidence intervals, detects performance trends, and generates recommendations. This system provides intelligent analysis of baseline performance data.

## Requirements
1. Extend `src/validation/baseline.rs` with comparison analysis logic
2. Implement statistical significance testing between tools
3. Add performance improvement/degradation detection
4. Create confidence interval calculations with proper error margins  
5. Generate actionable recommendations based on analysis
6. Support comparative ranking and scoring systems
7. Add trend analysis for performance over time

## Expected Code Structure
```rust
// Add to baseline.rs file

use std::collections::HashMap;
use statrs::distribution::{StudentsT, ContinuousCDF, Normal};
use statrs::statistics::{Statistics, OrderStatistics};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub comparison_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub baseline_results: Vec<BaselineResults>,
    pub comparative_analysis: ComparativeAnalysis,
    pub statistical_tests: Vec<StatisticalTest>,
    pub performance_rankings: Vec<PerformanceRanking>,
    pub recommendations: Vec<Recommendation>,
    pub trend_analysis: Option<TrendAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    pub test_type: TestType,
    pub tool_a: BaselineTool,
    pub tool_b: BaselineTool,
    pub metric: ComparisonMetric,
    pub p_value: f64,
    pub critical_value: f64,
    pub test_statistic: f64,
    pub degrees_of_freedom: f64,
    pub significant: bool,
    pub confidence_level: f64,
    pub effect_size: f64,
    pub power: f64,
    pub interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestType {
    TTest,
    MannWhitneyU,
    WelchsTTest,
    PairedTTest,
    KruskalWallis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonMetric {
    Latency,
    Throughput,
    MemoryUsage,
    SuccessRate,
    IndexTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRanking {
    pub tool: BaselineTool,
    pub metric: ComparisonMetric,
    pub rank: usize,
    pub score: f64,
    pub percentile: f64,
    pub relative_performance: f64, // Relative to best performer
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub priority: RecommendationPriority,
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub supporting_evidence: Vec<String>,
    pub confidence: f64,
    pub potential_impact: String,
    pub implementation_complexity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Reliability,
    ResourceUsage,
    Scalability,
    Configuration,
    Infrastructure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub metric: ComparisonMetric,
    pub tool: BaselineTool,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub regression_r_squared: f64,
    pub predicted_next_value: f64,
    pub prediction_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

impl BaselineComparison {
    pub fn new(comparison_id: String) -> Self {
        Self {
            comparison_id,
            timestamp: chrono::Utc::now(),
            baseline_results: Vec::new(),
            comparative_analysis: ComparativeAnalysis::default(),
            statistical_tests: Vec::new(),
            performance_rankings: Vec::new(),
            recommendations: Vec::new(),
            trend_analysis: None,
        }
    }
    
    pub fn add_baseline_results(&mut self, results: BaselineResults) {
        self.baseline_results.push(results);
    }
    
    pub async fn perform_comprehensive_analysis(&mut self) -> Result<()> {
        if self.baseline_results.len() < 2 {
            bail!("Need at least 2 baseline results to perform comparison");
        }
        
        // Perform statistical tests
        self.perform_statistical_tests()?;
        
        // Generate performance rankings
        self.generate_performance_rankings()?;
        
        // Create recommendations
        self.generate_recommendations()?;
        
        // Perform trend analysis if historical data is available
        if self.baseline_results.len() > 2 {
            self.perform_trend_analysis()?;
        }
        
        println!("Comprehensive baseline analysis completed with {} tests and {} recommendations", 
                 self.statistical_tests.len(), self.recommendations.len());
        
        Ok(())
    }
    
    fn perform_statistical_tests(&mut self) -> Result<()> {
        let metrics = vec![
            ComparisonMetric::Latency,
            ComparisonMetric::Throughput,
            ComparisonMetric::MemoryUsage,
            ComparisonMetric::SuccessRate,
        ];
        
        // Get all unique tool combinations
        let tools: Vec<BaselineTool> = self.baseline_results.iter()
            .flat_map(|br| br.tool_results.keys())
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        
        for metric in metrics {
            for i in 0..tools.len() {
                for j in (i + 1)..tools.len() {
                    let tool_a = tools[i];
                    let tool_b = tools[j];
                    
                    if let Some(test) = self.perform_pairwise_comparison(tool_a, tool_b, metric.clone())? {
                        self.statistical_tests.push(test);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn perform_pairwise_comparison(
        &self, 
        tool_a: BaselineTool, 
        tool_b: BaselineTool, 
        metric: ComparisonMetric
    ) -> Result<Option<StatisticalTest>> {
        let data_a = self.extract_metric_data(tool_a, &metric)?;
        let data_b = self.extract_metric_data(tool_b, &metric)?;
        
        if data_a.len() < 3 || data_b.len() < 3 {
            return Ok(None); // Not enough data for meaningful test
        }
        
        // Perform normality test to choose appropriate test
        let use_parametric = self.is_data_normal(&data_a) && self.is_data_normal(&data_b);
        
        let test = if use_parametric {
            self.perform_welchs_t_test(&data_a, &data_b, tool_a, tool_b, metric)?
        } else {
            self.perform_mann_whitney_u_test(&data_a, &data_b, tool_a, tool_b, metric)?
        };
        
        Ok(Some(test))
    }
    
    fn extract_metric_data(&self, tool: BaselineTool, metric: &ComparisonMetric) -> Result<Vec<f64>> {
        let mut data = Vec::new();
        
        for baseline_result in &self.baseline_results {
            if let Some(tool_result) = baseline_result.tool_results.get(&tool) {
                let values = match metric {
                    ComparisonMetric::Latency => {
                        tool_result.individual_results.iter()
                            .filter(|r| r.success)
                            .map(|r| r.execution_time.as_secs_f64() * 1000.0)
                            .collect()
                    },
                    ComparisonMetric::Throughput => {
                        vec![tool_result.throughput_qps]
                    },
                    ComparisonMetric::MemoryUsage => {
                        tool_result.individual_results.iter()
                            .filter(|r| r.success)
                            .map(|r| r.memory_usage_mb)
                            .collect()
                    },
                    ComparisonMetric::SuccessRate => {
                        vec![tool_result.success_rate]
                    },
                    ComparisonMetric::IndexTime => {
                        tool_result.individual_results.iter()
                            .filter_map(|r| r.index_time.map(|t| t.as_secs_f64() * 1000.0))
                            .collect()
                    },
                };
                
                data.extend(values);
            }
        }
        
        Ok(data)
    }
    
    fn is_data_normal(&self, data: &[f64]) -> bool {
        if data.len() < 8 {
            return false; // Too small for reliable normality test
        }
        
        // Simplified Shapiro-Wilk-like test
        let mean = data.mean();
        let std_dev = data.std_dev();
        
        if std_dev == 0.0 {
            return false;
        }
        
        // Count values within 1, 2, and 3 standard deviations
        let within_1_std = data.iter().filter(|&&x| (x - mean).abs() <= std_dev).count();
        let within_2_std = data.iter().filter(|&&x| (x - mean).abs() <= 2.0 * std_dev).count();
        let within_3_std = data.iter().filter(|&&x| (x - mean).abs() <= 3.0 * std_dev).count();
        
        let n = data.len() as f64;
        let pct_1_std = within_1_std as f64 / n;
        let pct_2_std = within_2_std as f64 / n;
        let pct_3_std = within_3_std as f64 / n;
        
        // For normal distribution: ~68% within 1σ, ~95% within 2σ, ~99.7% within 3σ
        pct_1_std >= 0.6 && pct_1_std <= 0.76 &&
        pct_2_std >= 0.90 && pct_2_std <= 0.98 &&
        pct_3_std >= 0.95
    }
    
    fn perform_welchs_t_test(
        &self,
        data_a: &[f64],
        data_b: &[f64],
        tool_a: BaselineTool,
        tool_b: BaselineTool,
        metric: ComparisonMetric,
    ) -> Result<StatisticalTest> {
        let mean_a = data_a.mean();
        let mean_b = data_b.mean();
        let var_a = data_a.variance();
        let var_b = data_b.variance();
        let n_a = data_a.len() as f64;
        let n_b = data_b.len() as f64;
        
        // Welch's t-test for unequal variances
        let pooled_se = (var_a / n_a + var_b / n_b).sqrt();
        let t_stat = (mean_a - mean_b) / pooled_se;
        
        // Welch-Satterthwaite equation for degrees of freedom
        let df = (var_a / n_a + var_b / n_b).powi(2) / 
                 ((var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0));
        
        let confidence_level = 0.95;
        let alpha = 1.0 - confidence_level;
        
        // Calculate p-value (two-tailed)
        let t_dist = StudentsT::new(0.0, 1.0, df)?;
        let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
        
        // Critical value for two-tailed test
        let critical_value = t_dist.inverse_cdf(1.0 - alpha / 2.0);
        
        let significant = p_value < alpha;
        
        // Effect size (Cohen's d)
        let pooled_std = ((var_a + var_b) / 2.0).sqrt();
        let effect_size = (mean_a - mean_b).abs() / pooled_std;
        
        // Statistical power (approximate)
        let power = self.calculate_statistical_power(effect_size, n_a, n_b, alpha);
        
        let interpretation = self.interpret_statistical_test(
            &metric, tool_a, tool_b, significant, effect_size, mean_a, mean_b
        );
        
        Ok(StatisticalTest {
            test_type: TestType::WelchsTTest,
            tool_a,
            tool_b,
            metric,
            p_value,
            critical_value,
            test_statistic: t_stat,
            degrees_of_freedom: df,
            significant,
            confidence_level,
            effect_size,
            power,
            interpretation,
        })
    }
    
    fn perform_mann_whitney_u_test(
        &self,
        data_a: &[f64],
        data_b: &[f64],
        tool_a: BaselineTool,
        tool_b: BaselineTool,
        metric: ComparisonMetric,
    ) -> Result<StatisticalTest> {
        // Simplified Mann-Whitney U test implementation
        let n_a = data_a.len() as f64;
        let n_b = data_b.len() as f64;
        let n_total = n_a + n_b;
        
        // Combine and rank all values
        let mut combined: Vec<(f64, usize)> = data_a.iter().map(|&x| (x, 0)).collect();
        combined.extend(data_b.iter().map(|&x| (x, 1)));
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Assign ranks (handle ties by averaging)
        let mut ranks = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i;
            while j < combined.len() && combined[j].0 == combined[i].0 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }
        
        // Calculate U statistics
        let r_a: f64 = combined.iter().zip(ranks.iter())
            .filter(|((_, group), _)| *group == 0)
            .map(|(_, rank)| rank)
            .sum();
        
        let u_a = r_a - n_a * (n_a + 1.0) / 2.0;
        let u_b = n_a * n_b - u_a;
        
        let u_stat = u_a.min(u_b);
        
        // Normal approximation for large samples
        let mean_u = n_a * n_b / 2.0;
        let std_u = (n_a * n_b * (n_total + 1.0) / 12.0).sqrt();
        
        let z_stat = (u_stat - mean_u) / std_u;
        
        // Calculate p-value using normal approximation
        let normal_dist = Normal::new(0.0, 1.0)?;
        let p_value = 2.0 * (1.0 - normal_dist.cdf(z_stat.abs()));
        
        let alpha = 0.05;
        let significant = p_value < alpha;
        
        // Effect size (r = Z / sqrt(N))
        let effect_size = z_stat.abs() / n_total.sqrt();
        
        let power = 0.8; // Placeholder - proper calculation is complex
        
        let interpretation = format!(
            "Mann-Whitney U test: {} vs {} for {}. U = {:.2}, Z = {:.2}, p = {:.4}",
            format!("{:?}", tool_a), format!("{:?}", tool_b), 
            format!("{:?}", metric), u_stat, z_stat, p_value
        );
        
        Ok(StatisticalTest {
            test_type: TestType::MannWhitneyU,
            tool_a,
            tool_b,
            metric,
            p_value,
            critical_value: 1.96, // Z critical value for α = 0.05
            test_statistic: z_stat,
            degrees_of_freedom: n_total - 2.0,
            significant,
            confidence_level: 0.95,
            effect_size,
            power,
            interpretation,
        })
    }
    
    fn calculate_statistical_power(&self, effect_size: f64, n_a: f64, n_b: f64, alpha: f64) -> f64 {
        // Simplified power calculation for t-test
        let harmonic_mean_n = 2.0 / (1.0 / n_a + 1.0 / n_b);
        let delta = effect_size * (harmonic_mean_n / 2.0).sqrt();
        
        // Approximate power using normal distribution
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let z_alpha = normal_dist.inverse_cdf(1.0 - alpha / 2.0);
        let power = 1.0 - normal_dist.cdf(z_alpha - delta) + normal_dist.cdf(-z_alpha - delta);
        
        power.max(0.0).min(1.0)
    }
    
    fn interpret_statistical_test(
        &self,
        metric: &ComparisonMetric,
        tool_a: BaselineTool,
        tool_b: BaselineTool,
        significant: bool,
        effect_size: f64,
        mean_a: f64,
        mean_b: f64,
    ) -> String {
        let better_tool = match metric {
            ComparisonMetric::Latency | ComparisonMetric::MemoryUsage => {
                if mean_a < mean_b { tool_a } else { tool_b }
            },
            ComparisonMetric::Throughput | ComparisonMetric::SuccessRate => {
                if mean_a > mean_b { tool_a } else { tool_b }
            },
            ComparisonMetric::IndexTime => {
                if mean_a < mean_b { tool_a } else { tool_b }
            },
        };
        
        let effect_magnitude = match effect_size {
            x if x < 0.2 => "negligible",
            x if x < 0.5 => "small",
            x if x < 0.8 => "medium",
            _ => "large",
        };
        
        if significant {
            format!(
                "{:?} significantly outperforms {:?} in {} with {} effect size (d = {:.3})",
                better_tool, 
                if better_tool == tool_a { tool_b } else { tool_a },
                format!("{:?}", metric).to_lowercase(),
                effect_magnitude,
                effect_size
            )
        } else {
            format!(
                "No significant difference between {:?} and {:?} in {} (effect size: {})",
                tool_a, tool_b, 
                format!("{:?}", metric).to_lowercase(),
                effect_magnitude
            )
        }
    }
    
    fn generate_performance_rankings(&mut self) -> Result<()> {
        let metrics = vec![
            ComparisonMetric::Latency,
            ComparisonMetric::Throughput, 
            ComparisonMetric::MemoryUsage,
            ComparisonMetric::SuccessRate,
        ];
        
        for metric in metrics {
            let mut tool_scores = Vec::new();
            
            // Calculate average performance for each tool
            for baseline_result in &self.baseline_results {
                for (tool, tool_result) in &baseline_result.tool_results {
                    let score = match metric {
                        ComparisonMetric::Latency => tool_result.average_latency_ms,
                        ComparisonMetric::Throughput => tool_result.throughput_qps,
                        ComparisonMetric::MemoryUsage => tool_result.average_memory_mb,
                        ComparisonMetric::SuccessRate => tool_result.success_rate,
                        ComparisonMetric::IndexTime => {
                            // Average index time if available
                            tool_result.individual_results.iter()
                                .filter_map(|r| r.index_time)
                                .map(|t| t.as_secs_f64() * 1000.0)
                                .sum::<f64>() / tool_result.individual_results.len().max(1) as f64
                        },
                    };
                    
                    tool_scores.push((*tool, score));
                }
            }
            
            // Sort and rank
            match metric {
                ComparisonMetric::Latency | ComparisonMetric::MemoryUsage | ComparisonMetric::IndexTime => {
                    tool_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // Lower is better
                },
                ComparisonMetric::Throughput | ComparisonMetric::SuccessRate => {
                    tool_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Higher is better
                },
            }
            
            let best_score = tool_scores.first().map(|(_, score)| *score).unwrap_or(0.0);
            
            for (rank, (tool, score)) in tool_scores.iter().enumerate() {
                let relative_performance = if best_score != 0.0 {
                    match metric {
                        ComparisonMetric::Latency | ComparisonMetric::MemoryUsage | ComparisonMetric::IndexTime => {
                            best_score / score // For "lower is better" metrics
                        },
                        ComparisonMetric::Throughput | ComparisonMetric::SuccessRate => {
                            score / best_score // For "higher is better" metrics
                        },
                    }
                } else {
                    1.0
                };
                
                // Calculate confidence interval (simplified)
                let data = self.extract_metric_data(*tool, &metric)?;
                let confidence_interval = if !data.is_empty() {
                    let mean = data.mean();
                    let std_err = data.std_dev() / (data.len() as f64).sqrt();
                    let margin = 1.96 * std_err; // 95% CI
                    (mean - margin, mean + margin)
                } else {
                    (*score, *score)
                };
                
                let percentile = (tool_scores.len() - rank) as f64 / tool_scores.len() as f64 * 100.0;
                
                self.performance_rankings.push(PerformanceRanking {
                    tool: *tool,
                    metric: metric.clone(),
                    rank: rank + 1,
                    score: *score,
                    percentile,
                    relative_performance,
                    confidence_interval,
                });
            }
        }
        
        Ok(())
    }
    
    fn generate_recommendations(&mut self) -> Result<()> {
        // Analyze statistical tests for recommendations
        for test in &self.statistical_tests {
            if test.significant && test.effect_size > 0.5 {
                let recommendation = self.create_performance_recommendation(test)?;
                self.recommendations.push(recommendation);
            }
        }
        
        // Analyze rankings for recommendations  
        self.create_ranking_based_recommendations()?;
        
        // Create resource usage recommendations
        self.create_resource_usage_recommendations()?;
        
        // Create reliability recommendations
        self.create_reliability_recommendations()?;
        
        Ok(())
    }
    
    fn create_performance_recommendation(&self, test: &StatisticalTest) -> Result<Recommendation> {
        let priority = match test.effect_size {
            x if x > 1.2 => RecommendationPriority::Critical,
            x if x > 0.8 => RecommendationPriority::High,
            x if x > 0.5 => RecommendationPriority::Medium,
            _ => RecommendationPriority::Low,
        };
        
        let title = format!(
            "Significant {} Performance Difference Detected",
            format!("{:?}", test.metric)
        );
        
        let description = format!(
            "Statistical analysis shows {} significantly outperforms {} in {} (p = {:.4}, effect size = {:.3})",
            format!("{:?}", test.tool_a), format!("{:?}", test.tool_b),
            format!("{:?}", test.metric).to_lowercase(), test.p_value, test.effect_size
        );
        
        let supporting_evidence = vec![
            format!("Statistical test: {:?}", test.test_type),
            format!("P-value: {:.6}", test.p_value),
            format!("Effect size: {:.3}", test.effect_size),
            format!("Statistical power: {:.3}", test.power),
        ];
        
        Ok(Recommendation {
            priority,
            category: RecommendationCategory::Performance,
            title,
            description,
            supporting_evidence,
            confidence: (1.0 - test.p_value).min(0.999),
            potential_impact: self.estimate_performance_impact(test.effect_size),
            implementation_complexity: "Low - Configuration Change".to_string(),
        })
    }
    
    fn create_ranking_based_recommendations(&mut self) -> Result<()> {
        // Find consistently poor performers
        let mut tool_average_ranks: HashMap<BaselineTool, Vec<usize>> = HashMap::new();
        
        for ranking in &self.performance_rankings {
            tool_average_ranks.entry(ranking.tool)
                .or_default()
                .push(ranking.rank);
        }
        
        for (tool, ranks) in tool_average_ranks {
            let avg_rank = ranks.iter().sum::<usize>() as f64 / ranks.len() as f64;
            let total_tools = self.get_unique_tools().len();
            
            if avg_rank > total_tools as f64 * 0.75 {
                let recommendation = Recommendation {
                    priority: RecommendationPriority::Medium,
                    category: RecommendationCategory::Performance,
                    title: format!("Poor Overall Performance: {:?}", tool),
                    description: format!(
                        "{:?} ranks poorly across multiple metrics (average rank: {:.1} out of {})",
                        tool, avg_rank, total_tools
                    ),
                    supporting_evidence: vec![
                        format!("Average rank: {:.1}", avg_rank),
                        format!("Metrics evaluated: {}", ranks.len()),
                    ],
                    confidence: 0.8,
                    potential_impact: "Medium - Consider alternative tools".to_string(),
                    implementation_complexity: "Medium - Tool Replacement".to_string(),
                };
                
                self.recommendations.push(recommendation);
            }
        }
        
        Ok(())
    }
    
    fn create_resource_usage_recommendations(&mut self) -> Result<()> {
        // Find tools with excessive memory usage
        let memory_rankings: Vec<_> = self.performance_rankings.iter()
            .filter(|r| matches!(r.metric, ComparisonMetric::MemoryUsage))
            .collect();
        
        if let Some(worst_memory) = memory_rankings.iter().max_by(|a, b| 
            a.score.partial_cmp(&b.score).unwrap()) {
            
            if worst_memory.relative_performance < 0.5 { // Uses 2x more memory than best
                let recommendation = Recommendation {
                    priority: RecommendationPriority::Medium,
                    category: RecommendationCategory::ResourceUsage,
                    title: format!("High Memory Usage: {:?}", worst_memory.tool),
                    description: format!(
                        "{:?} uses {:.1}MB memory on average, {:.1}x more than the most efficient tool",
                        worst_memory.tool, worst_memory.score, 1.0 / worst_memory.relative_performance
                    ),
                    supporting_evidence: vec![
                        format!("Memory usage: {:.1}MB", worst_memory.score),
                        format!("Relative performance: {:.3}", worst_memory.relative_performance),
                    ],
                    confidence: 0.9,
                    potential_impact: "Medium - Resource optimization".to_string(),
                    implementation_complexity: "Low - Configuration tuning".to_string(),
                };
                
                self.recommendations.push(recommendation);
            }
        }
        
        Ok(())
    }
    
    fn create_reliability_recommendations(&mut self) -> Result<()> {
        // Find tools with low success rates
        let reliability_rankings: Vec<_> = self.performance_rankings.iter()
            .filter(|r| matches!(r.metric, ComparisonMetric::SuccessRate))
            .collect();
        
        for ranking in reliability_rankings {
            if ranking.score < 95.0 { // Less than 95% success rate
                let priority = if ranking.score < 90.0 {
                    RecommendationPriority::High
                } else {
                    RecommendationPriority::Medium
                };
                
                let recommendation = Recommendation {
                    priority,
                    category: RecommendationCategory::Reliability,
                    title: format!("Low Success Rate: {:?}", ranking.tool),
                    description: format!(
                        "{:?} has a {:.1}% success rate, indicating reliability issues",
                        ranking.tool, ranking.score
                    ),
                    supporting_evidence: vec![
                        format!("Success rate: {:.1}%", ranking.score),
                        format!("Rank: {} out of {}", ranking.rank, reliability_rankings.len()),
                    ],
                    confidence: 0.95,
                    potential_impact: "High - Affects system reliability".to_string(),
                    implementation_complexity: "Medium - Requires investigation".to_string(),
                };
                
                self.recommendations.push(recommendation);
            }
        }
        
        Ok(())
    }
    
    fn estimate_performance_impact(&self, effect_size: f64) -> String {
        match effect_size {
            x if x > 1.2 => "Very High - Substantial performance difference".to_string(),
            x if x > 0.8 => "High - Notable performance improvement possible".to_string(),
            x if x > 0.5 => "Medium - Moderate performance gains expected".to_string(),
            x if x > 0.2 => "Low - Minor performance improvement".to_string(),
            _ => "Negligible - Minimal impact expected".to_string(),
        }
    }
    
    fn get_unique_tools(&self) -> Vec<BaselineTool> {
        self.baseline_results.iter()
            .flat_map(|br| br.tool_results.keys())
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
    
    fn perform_trend_analysis(&mut self) -> Result<()> {
        // This would analyze trends over time if historical data is available
        // For now, return Ok as this is a placeholder for future enhancement
        Ok(())
    }
    
    pub fn print_comparison_summary(&self) {
        println!("\n=== Baseline Comparison Analysis ===");
        println!("Comparison ID: {}", self.comparison_id);
        println!("Analyzed {} baseline results", self.baseline_results.len());
        println!("Performed {} statistical tests", self.statistical_tests.len());
        println!("Generated {} recommendations", self.recommendations.len());
        
        println!("\n--- Statistical Tests ---");
        for test in &self.statistical_tests {
            if test.significant {
                println!("✓ {}", test.interpretation);
            }
        }
        
        println!("\n--- Top Recommendations ---");
        let mut sorted_recommendations = self.recommendations.clone();
        sorted_recommendations.sort_by(|a, b| {
            let priority_order = |p: &RecommendationPriority| match p {
                RecommendationPriority::Critical => 0,
                RecommendationPriority::High => 1,
                RecommendationPriority::Medium => 2,
                RecommendationPriority::Low => 3,
                RecommendationPriority::Informational => 4,
            };
            priority_order(&a.priority).cmp(&priority_order(&b.priority))
        });
        
        for (i, rec) in sorted_recommendations.iter().take(5).enumerate() {
            println!("{}. [{:?}] {} (Confidence: {:.1}%)", 
                     i + 1, rec.priority, rec.title, rec.confidence * 100.0);
        }
    }
}
```

## Dependencies Already Added
The required statistical dependencies (statrs, chrono) were added in previous tasks.

## Success Criteria
- Statistical significance testing works correctly for different data distributions
- Performance rankings accurately reflect relative tool performance
- Recommendation generation provides actionable insights
- Confidence intervals are calculated with appropriate error margins
- Effect size calculations provide meaningful magnitude estimates
- Comparative analysis handles edge cases (missing data, identical performance)
- Output formatting is clear and professional

## Time Limit
10 minutes maximum