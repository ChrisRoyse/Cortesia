# Task 014: Intelligent Search Mode Determination Logic

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 005-013, specifically extending the CorrectnessValidator with intelligent SearchMode selection capabilities. The SearchModeAnalyzer provides automatic mode selection based on query analysis, hybrid optimization, and performance-driven recommendations.

## Project Structure
```
src/
  validation/
    correctness.rs  <- Extend this file
  lib.rs
```

## Task Description
Create the `SearchModeAnalyzer` that intelligently determines the optimal SearchMode for queries based on complexity analysis, performance characteristics, and configurable selection rules with fallback mechanisms.

## Requirements
1. Add to existing `src/validation/correctness.rs`
2. Implement `SearchModeAnalyzer` with query complexity scoring
3. Add performance-based mode selection with historical data
4. Include hybrid mode optimization logic
5. Support configuration-driven mode selection rules
6. Add fallback mechanisms for edge cases
7. Provide detailed analysis reporting and recommendations

## Expected Code Structure to Add
```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchMode {
    Vector,
    FullText,
    Hybrid,
    Auto, // Let the system decide
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryComplexity {
    pub word_count: usize,
    pub unique_word_count: usize,
    pub average_word_length: f64,
    pub has_special_chars: bool,
    pub has_boolean_operators: bool,
    pub has_quotes: bool,
    pub has_wildcards: bool,
    pub complexity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub vector_avg_latency: Duration,
    pub fulltext_avg_latency: Duration,
    pub hybrid_avg_latency: Duration,
    pub vector_success_rate: f64,
    pub fulltext_success_rate: f64,
    pub hybrid_success_rate: f64,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeSelectionRule {
    pub name: String,
    pub condition: RuleCondition,
    pub recommended_mode: SearchMode,
    pub priority: i32,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCondition {
    ComplexityThreshold { min: f64, max: f64 },
    WordCountRange { min: usize, max: usize },
    HasBooleanOperators,
    HasSpecialCharacters,
    PerformancePreference { metric: PerformanceMetric, threshold: f64 },
    HistoricalSuccess { mode: SearchMode, min_rate: f64 },
    Combined { conditions: Vec<RuleCondition>, operator: LogicalOperator },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    Latency,
    SuccessRate,
    Throughput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeRecommendation {
    pub recommended_mode: SearchMode,
    pub confidence: f64,
    pub reasoning: Vec<String>,
    pub fallback_modes: Vec<SearchMode>,
    pub expected_performance: ExpectedPerformance,
    pub rule_matches: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedPerformance {
    pub estimated_latency: Duration,
    pub estimated_success_rate: f64,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_intensity: f64,    // 0.0 to 1.0
    pub memory_usage: f64,     // 0.0 to 1.0
    pub io_intensity: f64,     // 0.0 to 1.0
}

pub struct SearchModeAnalyzer {
    rules: Vec<ModeSelectionRule>,
    performance_history: HashMap<SearchMode, PerformanceMetrics>,
    fallback_mode: SearchMode,
    complexity_weights: ComplexityWeights,
    enable_learning: bool,
}

#[derive(Debug, Clone)]
struct ComplexityWeights {
    word_count: f64,
    unique_ratio: f64,
    avg_word_length: f64,
    special_chars: f64,
    boolean_ops: f64,
    quotes: f64,
    wildcards: f64,
}

impl SearchModeAnalyzer {
    pub fn new() -> Self {
        Self {
            rules: Self::default_rules(),
            performance_history: HashMap::new(),
            fallback_mode: SearchMode::Hybrid,
            complexity_weights: ComplexityWeights {
                word_count: 0.2,
                unique_ratio: 0.15,
                avg_word_length: 0.1,
                special_chars: 0.15,
                boolean_ops: 0.2,
                quotes: 0.1,
                wildcards: 0.1,
            },
            enable_learning: true,
        }
    }
    
    pub fn with_rules(mut self, rules: Vec<ModeSelectionRule>) -> Self {
        self.rules = rules;
        self
    }
    
    pub fn with_fallback_mode(mut self, mode: SearchMode) -> Self {
        self.fallback_mode = mode;
        self
    }
    
    pub fn analyze_query(&self, query: &str) -> ModeRecommendation {
        let complexity = self.calculate_complexity(query);
        let mut reasoning = Vec::new();
        let mut rule_matches = Vec::new();
        let mut mode_scores: HashMap<SearchMode, f64> = HashMap::new();
        
        // Initialize base scores
        mode_scores.insert(SearchMode::Vector, 0.5);
        mode_scores.insert(SearchMode::FullText, 0.5);
        mode_scores.insert(SearchMode::Hybrid, 0.6); // Slight preference for hybrid
        
        // Apply rules in priority order
        let mut sorted_rules = self.rules.clone();
        sorted_rules.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        for rule in &sorted_rules {
            if !rule.enabled {
                continue;
            }
            
            if self.evaluate_condition(&rule.condition, &complexity, query) {
                rule_matches.push(rule.name.clone());
                
                let rule_weight = (rule.priority as f64 / 100.0).min(1.0);
                let current_score = mode_scores.get(&rule.recommended_mode).unwrap_or(&0.0);
                mode_scores.insert(rule.recommended_mode, current_score + rule_weight);
                
                reasoning.push(format!(
                    "Rule '{}' matched, recommending {:?} (weight: {:.2})", 
                    rule.name, rule.recommended_mode, rule_weight
                ));
            }
        }
        
        // Apply performance-based adjustments
        self.apply_performance_adjustments(&mut mode_scores, &mut reasoning);
        
        // Select the mode with highest score
        let recommended_mode = mode_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(mode, _)| *mode)
            .unwrap_or(self.fallback_mode);
        
        let confidence = mode_scores.get(&recommended_mode).unwrap_or(&0.0)
            .min(1.0).max(0.0);
        
        // Generate fallback modes
        let mut fallback_modes: Vec<(SearchMode, f64)> = mode_scores.iter()
            .filter(|(mode, _)| **mode != recommended_mode)
            .map(|(mode, score)| (*mode, *score))
            .collect();
        fallback_modes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let fallback_modes: Vec<SearchMode> = fallback_modes.into_iter()
            .map(|(mode, _)| mode)
            .collect();
        
        // Estimate performance
        let expected_performance = self.estimate_performance(recommended_mode, &complexity);
        
        if reasoning.is_empty() {
            reasoning.push(format!(
                "No specific rules matched, using default preference for {:?}", 
                recommended_mode
            ));
        }
        
        ModeRecommendation {
            recommended_mode,
            confidence,
            reasoning,
            fallback_modes,
            expected_performance,
            rule_matches,
        }
    }
    
    fn calculate_complexity(&self, query: &str) -> QueryComplexity {
        let words: Vec<&str> = query.split_whitespace().collect();
        let word_count = words.len();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let unique_word_count = unique_words.len();
        
        let average_word_length = if word_count > 0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count as f64
        } else {
            0.0
        };
        
        let has_special_chars = query.chars().any(|c| !c.is_alphanumeric() && !c.is_whitespace());
        let has_boolean_operators = query.to_lowercase().contains("and") || 
                                   query.to_lowercase().contains("or") || 
                                   query.to_lowercase().contains("not");
        let has_quotes = query.contains('"') || query.contains('\'');
        let has_wildcards = query.contains('*') || query.contains('?');
        
        // Calculate complexity score
        let mut complexity_score = 0.0;
        complexity_score += word_count as f64 * self.complexity_weights.word_count;
        complexity_score += (unique_word_count as f64 / word_count.max(1) as f64) * 
                           self.complexity_weights.unique_ratio;
        complexity_score += average_word_length * self.complexity_weights.avg_word_length;
        
        if has_special_chars {
            complexity_score += self.complexity_weights.special_chars;
        }
        if has_boolean_operators {
            complexity_score += self.complexity_weights.boolean_ops;
        }
        if has_quotes {
            complexity_score += self.complexity_weights.quotes;
        }
        if has_wildcards {
            complexity_score += self.complexity_weights.wildcards;
        }
        
        QueryComplexity {
            word_count,
            unique_word_count,
            average_word_length,
            has_special_chars,
            has_boolean_operators,
            has_quotes,
            has_wildcards,
            complexity_score,
        }
    }
    
    fn evaluate_condition(&self, condition: &RuleCondition, complexity: &QueryComplexity, 
                         query: &str) -> bool {
        match condition {
            RuleCondition::ComplexityThreshold { min, max } => {
                complexity.complexity_score >= *min && complexity.complexity_score <= *max
            },
            RuleCondition::WordCountRange { min, max } => {
                complexity.word_count >= *min && complexity.word_count <= *max
            },
            RuleCondition::HasBooleanOperators => complexity.has_boolean_operators,
            RuleCondition::HasSpecialCharacters => complexity.has_special_chars,
            RuleCondition::PerformancePreference { metric, threshold } => {
                self.check_performance_preference(metric, *threshold)
            },
            RuleCondition::HistoricalSuccess { mode, min_rate } => {
                if let Some(metrics) = self.performance_history.get(mode) {
                    match mode {
                        SearchMode::Vector => metrics.vector_success_rate >= *min_rate,
                        SearchMode::FullText => metrics.fulltext_success_rate >= *min_rate,
                        SearchMode::Hybrid => metrics.hybrid_success_rate >= *min_rate,
                        SearchMode::Auto => true, // Auto mode adapts
                    }
                } else {
                    false
                }
            },
            RuleCondition::Combined { conditions, operator } => {
                match operator {
                    LogicalOperator::And => {
                        conditions.iter().all(|c| self.evaluate_condition(c, complexity, query))
                    },
                    LogicalOperator::Or => {
                        conditions.iter().any(|c| self.evaluate_condition(c, complexity, query))
                    },
                    LogicalOperator::Not => {
                        !conditions.iter().any(|c| self.evaluate_condition(c, complexity, query))
                    },
                }
            },
        }
    }
    
    fn check_performance_preference(&self, metric: &PerformanceMetric, threshold: f64) -> bool {
        // This would check against historical performance data
        // For now, return a reasonable default
        match metric {
            PerformanceMetric::Latency => threshold > 100.0, // ms
            PerformanceMetric::SuccessRate => threshold > 0.8,
            PerformanceMetric::Throughput => threshold > 10.0, // queries/sec
        }
    }
    
    fn apply_performance_adjustments(&self, mode_scores: &mut HashMap<SearchMode, f64>, 
                                   reasoning: &mut Vec<String>) {
        for (mode, metrics) in &self.performance_history {
            if metrics.sample_count < 10 {
                continue; // Not enough data
            }
            
            let success_rate = match mode {
                SearchMode::Vector => metrics.vector_success_rate,
                SearchMode::FullText => metrics.fulltext_success_rate,
                SearchMode::Hybrid => metrics.hybrid_success_rate,
                SearchMode::Auto => 0.8, // Reasonable default
            };
            
            let latency_penalty = match mode {
                SearchMode::Vector => metrics.vector_avg_latency.as_millis() as f64 / 1000.0,
                SearchMode::FullText => metrics.fulltext_avg_latency.as_millis() as f64 / 1000.0,
                SearchMode::Hybrid => metrics.hybrid_avg_latency.as_millis() as f64 / 1000.0,
                SearchMode::Auto => 0.5, // Reasonable default
            };
            
            // Adjust score based on historical performance
            let performance_adjustment = success_rate - (latency_penalty * 0.1);
            let current_score = mode_scores.get(mode).unwrap_or(&0.0);
            mode_scores.insert(*mode, current_score + performance_adjustment * 0.3);
            
            reasoning.push(format!(
                "Performance adjustment for {:?}: success_rate={:.3}, latency={:.1}ms, adj={:.3}",
                mode, success_rate, latency_penalty * 1000.0, performance_adjustment * 0.3
            ));
        }
    }
    
    fn estimate_performance(&self, mode: SearchMode, complexity: &QueryComplexity) -> ExpectedPerformance {
        let base_latency = match mode {
            SearchMode::Vector => Duration::from_millis(50),
            SearchMode::FullText => Duration::from_millis(100),
            SearchMode::Hybrid => Duration::from_millis(150),
            SearchMode::Auto => Duration::from_millis(100),
        };
        
        // Adjust based on complexity
        let complexity_multiplier = 1.0 + (complexity.complexity_score * 0.5);
        let estimated_latency = Duration::from_millis(
            (base_latency.as_millis() as f64 * complexity_multiplier) as u64
        );
        
        let base_success_rate = match mode {
            SearchMode::Vector => 0.85,
            SearchMode::FullText => 0.90,
            SearchMode::Hybrid => 0.95,
            SearchMode::Auto => 0.88,
        };
        
        // Adjust success rate based on query characteristics
        let mut estimated_success_rate = base_success_rate;
        if complexity.has_boolean_operators && mode == SearchMode::Vector {
            estimated_success_rate *= 0.8; // Vector search struggles with boolean logic
        }
        if complexity.has_wildcards && mode == SearchMode::FullText {
            estimated_success_rate *= 1.1; // Full-text handles wildcards well
        }
        
        let resource_usage = match mode {
            SearchMode::Vector => ResourceUsage {
                cpu_intensity: 0.7,
                memory_usage: 0.8,
                io_intensity: 0.3,
            },
            SearchMode::FullText => ResourceUsage {
                cpu_intensity: 0.5,
                memory_usage: 0.4,
                io_intensity: 0.7,
            },
            SearchMode::Hybrid => ResourceUsage {
                cpu_intensity: 0.8,
                memory_usage: 0.9,
                io_intensity: 0.6,
            },
            SearchMode::Auto => ResourceUsage {
                cpu_intensity: 0.6,
                memory_usage: 0.6,
                io_intensity: 0.5,
            },
        };
        
        ExpectedPerformance {
            estimated_latency,
            estimated_success_rate,
            resource_usage,
        }
    }
    
    pub fn update_performance_metrics(&mut self, mode: SearchMode, latency: Duration, 
                                    success: bool) {
        if !self.enable_learning {
            return;
        }
        
        let metrics = self.performance_history.entry(mode).or_insert_with(|| {
            PerformanceMetrics {
                vector_avg_latency: Duration::from_millis(100),
                fulltext_avg_latency: Duration::from_millis(150),
                hybrid_avg_latency: Duration::from_millis(200),
                vector_success_rate: 0.85,
                fulltext_success_rate: 0.90,
                hybrid_success_rate: 0.95,
                sample_count: 0,
            }
        });
        
        let alpha = 0.1; // Learning rate
        metrics.sample_count += 1;
        
        match mode {
            SearchMode::Vector => {
                metrics.vector_avg_latency = Duration::from_millis(
                    ((1.0 - alpha) * metrics.vector_avg_latency.as_millis() as f64 + 
                     alpha * latency.as_millis() as f64) as u64
                );
                metrics.vector_success_rate = (1.0 - alpha) * metrics.vector_success_rate + 
                                             alpha * if success { 1.0 } else { 0.0 };
            },
            SearchMode::FullText => {
                metrics.fulltext_avg_latency = Duration::from_millis(
                    ((1.0 - alpha) * metrics.fulltext_avg_latency.as_millis() as f64 + 
                     alpha * latency.as_millis() as f64) as u64
                );
                metrics.fulltext_success_rate = (1.0 - alpha) * metrics.fulltext_success_rate + 
                                               alpha * if success { 1.0 } else { 0.0 };
            },
            SearchMode::Hybrid => {
                metrics.hybrid_avg_latency = Duration::from_millis(
                    ((1.0 - alpha) * metrics.hybrid_avg_latency.as_millis() as f64 + 
                     alpha * latency.as_millis() as f64) as u64
                );
                metrics.hybrid_success_rate = (1.0 - alpha) * metrics.hybrid_success_rate + 
                                             alpha * if success { 1.0 } else { 0.0 };
            },
            SearchMode::Auto => {
                // Auto mode metrics are aggregated from other modes
            },
        }
    }
    
    fn default_rules() -> Vec<ModeSelectionRule> {
        vec![
            ModeSelectionRule {
                name: "Simple queries prefer vector search".to_string(),
                condition: RuleCondition::Combined {
                    conditions: vec![
                        RuleCondition::WordCountRange { min: 1, max: 5 },
                        RuleCondition::ComplexityThreshold { min: 0.0, max: 2.0 },
                    ],
                    operator: LogicalOperator::And,
                },
                recommended_mode: SearchMode::Vector,
                priority: 70,
                enabled: true,
            },
            ModeSelectionRule {
                name: "Boolean operators prefer full-text search".to_string(),
                condition: RuleCondition::HasBooleanOperators,
                recommended_mode: SearchMode::FullText,
                priority: 80,
                enabled: true,
            },
            ModeSelectionRule {
                name: "Complex queries prefer hybrid search".to_string(),
                condition: RuleCondition::ComplexityThreshold { min: 3.0, max: 10.0 },
                recommended_mode: SearchMode::Hybrid,
                priority: 90,
                enabled: true,
            },
            ModeSelectionRule {
                name: "Long queries prefer hybrid search".to_string(),
                condition: RuleCondition::WordCountRange { min: 10, max: 1000 },
                recommended_mode: SearchMode::Hybrid,
                priority: 60,
                enabled: true,
            },
            ModeSelectionRule {
                name: "Wildcard queries prefer full-text search".to_string(),
                condition: RuleCondition::Combined {
                    conditions: vec![
                        RuleCondition::HasSpecialCharacters,
                    ],
                    operator: LogicalOperator::And,
                },
                recommended_mode: SearchMode::FullText,
                priority: 75,
                enabled: true,
            },
        ]
    }
    
    pub fn add_rule(&mut self, rule: ModeSelectionRule) {
        self.rules.push(rule);
    }
    
    pub fn remove_rule(&mut self, name: &str) {
        self.rules.retain(|rule| rule.name != name);
    }
    
    pub fn get_performance_summary(&self) -> HashMap<SearchMode, PerformanceMetrics> {
        self.performance_history.clone()
    }
}

impl Default for SearchModeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
```

## Success Criteria
- SearchModeAnalyzer struct compiles without errors
- Query complexity calculation accurately analyzes all aspects
- Rule evaluation system works correctly with logical operators
- Performance-based adjustments improve recommendations over time
- Mode selection provides reasonable recommendations with high confidence
- Fallback mechanisms handle edge cases appropriately
- Expected performance estimation aligns with actual performance
- Learning system improves recommendations based on historical data

## Time Limit
10 minutes maximum