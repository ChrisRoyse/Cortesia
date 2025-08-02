use std::sync::Arc;
use std::collections::HashMap as AHashMap;
use std::time::Instant;
use async_trait::async_trait;

use crate::cognitive::types::*;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
// Using pure graph operations for adaptive pattern selection
use crate::error::Result;

/// Ensemble result from merging multiple patterns
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    pub merged_answer: String,
    pub individual_contributions: Vec<PatternContribution>,
    pub confidence_analysis: ConfidenceDistribution,
}

/// Adaptive thinking pattern - selects optimal cognitive pattern based on query and context
pub struct AdaptiveThinking {
    pub graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub strategy_selector: Arc<StrategySelector>,
    pub ensemble_coordinator: Arc<EnsembleCoordinator>,
    // pub performance_tracker: Arc<PerformanceMonitor>,
}

impl AdaptiveThinking {
    pub fn new(
        graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        Self {
            graph,
            strategy_selector: Arc::new(StrategySelector::new()),
            ensemble_coordinator: Arc::new(EnsembleCoordinator::new()),
            // performance_tracker,
        }
    }
    
    pub async fn execute_adaptive_reasoning(
        &self,
        query: &str,
        context: Option<&str>,
        _available_patterns: Vec<CognitivePatternType>,
    ) -> Result<AdaptiveResult> {
        let start_time = Instant::now();
        
        // 1. Analyze query characteristics
        let query_analysis = self.analyze_query_characteristics(query, context).await?;
        
        // 2. Select optimal cognitive pattern(s)
        let strategy_selection = self.select_cognitive_strategies(
            query,
            query_analysis.clone(),
            _available_patterns,
        ).await?;
        
        // 3. Execute selected patterns (possibly in parallel)
        let pattern_results = self.execute_selected_patterns(
            query,
            context,
            &strategy_selection,
        ).await?;
        
        // 4. Merge results using ensemble methods
        let ensemble_result = self.merge_pattern_results(pattern_results).await?;
        
        // 5. Learn from outcome for future strategy selection
        let learning_update = self.update_strategy_performance(
            &query_analysis,
            &strategy_selection,
            &ensemble_result,
        ).await?;
        
        let _execution_time = start_time.elapsed();
        
        Ok(AdaptiveResult {
            final_answer: ensemble_result.merged_answer,
            strategy_used: strategy_selection,
            pattern_contributions: ensemble_result.individual_contributions,
            confidence_distribution: ensemble_result.confidence_analysis,
            learning_update,
        })
    }
    
    pub async fn analyze_query_characteristics(&self, query: &str, _context: Option<&str>) -> Result<QueryCharacteristics> {
        // Analyze query to determine optimal patterns
        let word_count = query.split_whitespace().count();
        let has_creative_words = query.to_lowercase().contains("creative") || 
                                 query.to_lowercase().contains("innovative");
        let has_factual_words = query.to_lowercase().contains("what is") || 
                               query.to_lowercase().contains("define");
        let _has_divergent_words = query.to_lowercase().contains("types") || 
                                 query.to_lowercase().contains("examples") ||
                                 query.to_lowercase().contains("kinds") ||
                                 query.to_lowercase().contains("varieties");
        
        Ok(QueryCharacteristics {
            complexity_score: (word_count as f32 / 20.0).min(1.0),
            ambiguity_level: if query.contains("?") { 0.3 } else { 0.7 },
            domain_specificity: 0.5,
            temporal_aspect: query.to_lowercase().contains("when") || 
                           query.to_lowercase().contains("time"),
            creative_requirement: if has_creative_words { 0.8 } else { 0.3 },
            factual_focus: if has_factual_words { 0.9 } else { 0.4 },
            abstraction_level: 0.5,
        })
    }
    
    pub async fn select_cognitive_strategies(
        &self,
        query: &str,
        query_analysis: QueryCharacteristics,
        _available_patterns: Vec<CognitivePatternType>,
    ) -> Result<StrategySelection> {
        let mut selected_patterns = Vec::new();
        
        // Simple strategy selection logic
        if query_analysis.factual_focus > 0.7 {
            selected_patterns.push(CognitivePatternType::Convergent);
        }
        
        if query_analysis.creative_requirement > 0.6 {
            selected_patterns.push(CognitivePatternType::Lateral);
            selected_patterns.push(CognitivePatternType::Divergent);
        }
        
        if query_analysis.complexity_score > 0.7 {
            selected_patterns.push(CognitivePatternType::Systems);
        }
        
        // Check for divergent queries (asking for multiple examples/types)
        let query_lower = query.to_lowercase();
        if (query_lower.contains("types") || query_lower.contains("examples") ||
           query_lower.contains("kinds") || query_lower.contains("varieties"))
            && !selected_patterns.contains(&CognitivePatternType::Divergent) {
                selected_patterns.push(CognitivePatternType::Divergent);
            }
        
        if selected_patterns.is_empty() {
            selected_patterns.push(CognitivePatternType::Convergent);
        }
        
        let final_patterns = selected_patterns;
        
        Ok(StrategySelection {
            selected_patterns: final_patterns.clone(),
            selection_confidence: self.calculate_selection_confidence(&final_patterns, &query_analysis),
            reasoning: format!("Heuristic selection: {} patterns", final_patterns.len()),
            execution_order: self.determine_execution_order(&final_patterns, &query_analysis),
        })
    }
}

#[async_trait]
impl CognitivePattern for AdaptiveThinking {
    async fn execute(
        &self,
        query: &str,
        context: Option<&str>,
        _parameters: PatternParameters,
    ) -> Result<PatternResult> {
        let start_time = Instant::now();
        
        // Execute adaptive reasoning
        let available_patterns = vec![
            CognitivePatternType::Convergent,
            CognitivePatternType::Divergent,
            CognitivePatternType::Lateral,
            CognitivePatternType::Systems,
            CognitivePatternType::Critical,
            CognitivePatternType::Abstract,
        ];
        
        let result = self.execute_adaptive_reasoning(query, context, available_patterns).await?;
        let execution_time = start_time.elapsed();
        
        Ok(PatternResult {
            pattern_type: CognitivePatternType::Adaptive,
            answer: result.final_answer,
            confidence: result.confidence_distribution.ensemble_confidence,
            reasoning_trace: Vec::new(),
            metadata: ResultMetadata {
                execution_time_ms: execution_time.as_millis() as u64,
                nodes_activated: 0,
                iterations_completed: 1,
                converged: true,
                total_energy: 0.0,
                additional_info: {
                    let mut info = AHashMap::new();
                    info.insert("query".to_string(), query.to_string());
                    info.insert("pattern".to_string(), "adaptive".to_string());
                    info.insert("selected_patterns".to_string(), format!("{:?}", result.strategy_used.selected_patterns));
                    info
                },
            },
        })
    }
    
    fn get_pattern_type(&self) -> CognitivePatternType {
        CognitivePatternType::Adaptive
    }
    
    fn get_optimal_use_cases(&self) -> Vec<String> {
        vec![
            "Automatic pattern selection".to_string(),
            "Complex query analysis".to_string(),
            "Meta-reasoning".to_string(),
            "Ensemble methods".to_string(),
        ]
    }
    
    fn estimate_complexity(&self, _query: &str) -> ComplexityEstimate {
        ComplexityEstimate {
            computational_complexity: 80,
            estimated_time_ms: 3000,
            memory_requirements_mb: 25,
            confidence: 0.9,
            parallelizable: true,
        }
    }
}

pub struct StrategySelector {
    // Implementation would use ML models to select optimal strategies
}

impl StrategySelector {
    fn new() -> Self {
        Self {}
    }
}

pub struct EnsembleCoordinator {
    // Implementation would coordinate multiple patterns
}

impl EnsembleCoordinator {
    fn new() -> Self {
        Self {}
    }
}

impl AdaptiveThinking {
    /// Execute selected cognitive patterns
    async fn execute_selected_patterns(
        &self,
        query: &str,
        context: Option<&str>,
        strategy: &StrategySelection,
    ) -> Result<Vec<PatternContribution>> {
        let mut contributions = Vec::new();
        
        for pattern_type in strategy.selected_patterns.iter() {
            let weight = 1.0 / strategy.selected_patterns.len() as f32;
            
            // For now, simulate pattern execution
            // In a full implementation, this would use the actual pattern instances
            let mock_result = self.simulate_pattern_execution(
                pattern_type,
                query,
                context,
            ).await?;
            
            contributions.push(PatternContribution {
                pattern_type: *pattern_type,
                contribution_weight: weight,
                partial_result: mock_result.answer,
                confidence: mock_result.confidence,
            });
        }
        
        Ok(contributions)
    }
    
    /// Merge results from multiple patterns using ensemble methods
    pub async fn merge_pattern_results(
        &self,
        pattern_results: Vec<PatternContribution>,
    ) -> Result<EnsembleResult> {
        if pattern_results.is_empty() {
            return Ok(EnsembleResult {
                merged_answer: "No patterns executed".to_string(),
                individual_contributions: Vec::new(),
                confidence_analysis: ConfidenceDistribution {
                    mean_confidence: 0.0,
                    variance: 0.0,
                    individual_confidences: Vec::new(),
                    ensemble_confidence: 0.0,
                },
            });
        }
        
        // Weight-based ensemble merging
        let total_weight: f32 = pattern_results.iter().map(|r| r.contribution_weight * r.confidence).sum();
        
        let merged_answer = if pattern_results.len() == 1 {
            pattern_results[0].partial_result.clone()
        } else {
            let primary_result = pattern_results.iter()
                .max_by(|a, b| (a.contribution_weight * a.confidence).partial_cmp(&(b.contribution_weight * b.confidence)).unwrap())
                .unwrap();
            
            format!(
                "Ensemble Analysis (primary: {:?}):\n\n{}",
                primary_result.pattern_type,
                primary_result.partial_result
            )
        };
        
        // Calculate confidence distribution
        let individual_confidences: Vec<f32> = pattern_results.iter().map(|r| r.confidence).collect();
        let mean_confidence = individual_confidences.iter().sum::<f32>() / individual_confidences.len() as f32;
        let variance = individual_confidences.iter()
            .map(|c| (c - mean_confidence).powi(2))
            .sum::<f32>() / individual_confidences.len() as f32;
        
        let ensemble_confidence = if total_weight > 0.0 {
            pattern_results.iter()
                .map(|r| r.confidence * r.contribution_weight)
                .sum::<f32>() / total_weight
        } else {
            mean_confidence
        };
        
        Ok(EnsembleResult {
            merged_answer,
            individual_contributions: pattern_results,
            confidence_analysis: ConfidenceDistribution {
                mean_confidence,
                variance,
                individual_confidences,
                ensemble_confidence,
            },
        })
    }
    
    /// Update strategy performance for learning
    async fn update_strategy_performance(
        &self,
        query_analysis: &QueryCharacteristics,
        strategy: &StrategySelection,
        ensemble_result: &EnsembleResult,
    ) -> Result<LearningUpdate> {
        // Calculate performance feedback
        let performance_feedback = ensemble_result.confidence_analysis.ensemble_confidence;
        
        // Assess strategy effectiveness
        let strategy_effectiveness = if ensemble_result.individual_contributions.len() > 1 {
            // Multi-pattern strategy - assess diversity and coherence
            let confidence_variance = ensemble_result.confidence_analysis.variance;
            let diversity_bonus = if confidence_variance < 0.1 { 0.1 } else { 0.0 };
            (performance_feedback + diversity_bonus).min(1.0)
        } else {
            // Single pattern strategy
            performance_feedback * 0.9 // Slight penalty for not using ensemble
        };
        
        // Generate recommended adjustments
        let mut recommended_adjustments = Vec::new();
        
        if strategy_effectiveness < 0.6 {
            recommended_adjustments.push("Consider adding more patterns to ensemble".to_string());
        }
        
        if ensemble_result.confidence_analysis.variance > 0.3 {
            recommended_adjustments.push("High variance in pattern confidences - review selection criteria".to_string());
        }
        
        if query_analysis.complexity_score > 0.8 && strategy.selected_patterns.len() < 3 {
            recommended_adjustments.push("Complex query may benefit from more patterns".to_string());
        }
        
        // Generate model updates (simplified)
        let model_updates = if strategy_effectiveness > 0.8 {
            vec![ModelUpdate {
                model_id: "pattern_selection".to_string(),
                update_type: UpdateType::WeightAdjustment,
                update_data: vec![1.0],
                confidence: 0.8,
            }]
        } else {
            vec![ModelUpdate {
                model_id: "pattern_selection".to_string(),
                update_type: UpdateType::ParameterTuning,
                update_data: vec![0.5],
                confidence: 0.6,
            }]
        };
        
        Ok(LearningUpdate {
            performance_feedback,
            strategy_effectiveness,
            recommended_adjustments,
            model_updates,
        })
    }
    
    
    /// Calculate pattern score based on query characteristics
    fn calculate_pattern_score(&self, pattern: &CognitivePatternType, analysis: &QueryCharacteristics) -> f32 {
        match pattern {
            CognitivePatternType::Convergent => {
                analysis.factual_focus * 0.8 + (1.0 - analysis.ambiguity_level) * 0.6
            }
            CognitivePatternType::Divergent => {
                analysis.creative_requirement * 0.9 + analysis.ambiguity_level * 0.5
            }
            CognitivePatternType::Lateral => {
                analysis.creative_requirement * 0.8 + analysis.abstraction_level * 0.7
            }
            CognitivePatternType::Systems => {
                analysis.complexity_score * 0.9 + analysis.domain_specificity * 0.6
            }
            CognitivePatternType::Critical => {
                analysis.ambiguity_level * 0.8 + analysis.factual_focus * 0.5
            }
            CognitivePatternType::Abstract => {
                analysis.abstraction_level * 0.9 + analysis.complexity_score * 0.7
            }
            CognitivePatternType::Adaptive => {
                0.3 // Lower score for adaptive calling itself
            }
            CognitivePatternType::ChainOfThought => {
                analysis.factual_focus * 0.7 + analysis.complexity_score * 0.6
            }
            CognitivePatternType::TreeOfThoughts => {
                analysis.creative_requirement * 0.7 + analysis.complexity_score * 0.8
            }
        }
    }
    
    /// Calculate confidence in pattern selection
    fn calculate_selection_confidence(
        &self,
        patterns: &[CognitivePatternType],
        analysis: &QueryCharacteristics,
    ) -> f32 {
        if patterns.is_empty() {
            return 0.0;
        }
        
        let avg_score = patterns.iter()
            .map(|p| self.calculate_pattern_score(p, analysis))
            .sum::<f32>() / patterns.len() as f32;
        
        // Bonus for appropriate number of patterns
        let pattern_count_bonus = match patterns.len() {
            1 => if analysis.complexity_score < 0.5 { 0.1 } else { -0.1 },
            2..=3 => 0.1,
            4 => 0.0,
            _ => -0.2,
        };
        
        (avg_score + pattern_count_bonus).clamp(0.0, 1.0)
    }
    
    /// Determine optimal execution order for patterns
    fn determine_execution_order(
        &self,
        patterns: &[CognitivePatternType],
        analysis: &QueryCharacteristics,
    ) -> Vec<usize> {
        let mut indexed_patterns: Vec<(usize, CognitivePatternType, f32)> = patterns.iter()
            .enumerate()
            .map(|(i, p)| (i, *p, self.calculate_pattern_score(p, analysis)))
            .collect();
        
        // Sort by score (highest first)
        indexed_patterns.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        indexed_patterns.into_iter().map(|(i, _, _)| i).collect()
    }
    
    /// Simulate pattern execution for testing
    async fn simulate_pattern_execution(
        &self,
        pattern_type: &CognitivePatternType,
        query: &str,
        _context: Option<&str>,
    ) -> Result<PatternResult> {
        // Simplified simulation - in reality would call actual pattern
        let confidence = match pattern_type {
            CognitivePatternType::Convergent => 0.8,
            CognitivePatternType::Divergent => 0.7,
            CognitivePatternType::Lateral => 0.6,
            CognitivePatternType::Systems => 0.75,
            CognitivePatternType::Critical => 0.85,
            CognitivePatternType::Abstract => 0.65,
            CognitivePatternType::Adaptive => 0.7,
            CognitivePatternType::ChainOfThought => 0.72,
            CognitivePatternType::TreeOfThoughts => 0.73,
        };
        
        Ok(PatternResult {
            pattern_type: *pattern_type,
            answer: format!("{pattern_type:?} analysis: {query}"),
            confidence,
            reasoning_trace: Vec::new(),
            metadata: ResultMetadata {
                execution_time_ms: 100,
                nodes_activated: 5,
                iterations_completed: 1,
                converged: true,
                total_energy: 0.5,
                additional_info: AHashMap::new(),
            },
        })
    }
}

// Individual pattern contribution to ensemble - use global PatternContribution from cognitive::types