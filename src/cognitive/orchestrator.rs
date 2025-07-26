use std::sync::Arc;
use std::collections::HashMap as AHashMap;

use crate::cognitive::types::*;
use crate::cognitive::{
    ConvergentThinking, DivergentThinking, LateralThinking, SystemsThinking,
    CriticalThinking, AbstractThinking, AdaptiveThinking,
};
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::monitoring::collectors::runtime_profiler::RuntimeProfiler;
use crate::trace_function;
// Neural server dependency removed - using pure graph operations
use crate::monitoring::performance::{PerformanceMonitor, Operation};
use crate::error::{Result, GraphError};

/// Central orchestrator for all cognitive patterns
pub struct CognitiveOrchestrator {
    patterns: AHashMap<CognitivePatternType, Arc<dyn CognitivePattern>>,
    adaptive_selector: Arc<AdaptiveThinking>,
    performance_monitor: Arc<PerformanceMonitor>,
    brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    config: CognitiveOrchestratorConfig,
    runtime_profiler: Option<Arc<RuntimeProfiler>>,
}

impl std::fmt::Debug for CognitiveOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CognitiveOrchestrator")
            .field("patterns", &self.patterns.keys().collect::<Vec<_>>())
            .field("adaptive_selector", &"AdaptiveThinking")
            .field("performance_monitor", &self.performance_monitor)
            .field("brain_graph", &"BrainEnhancedKnowledgeGraph")
            .field("config", &self.config)
            .finish()
    }
}

/// Configuration for the cognitive orchestrator
#[derive(Debug, Clone)]
pub struct CognitiveOrchestratorConfig {
    pub enable_adaptive_selection: bool,
    pub enable_ensemble_methods: bool,
    pub default_timeout_ms: u64,
    pub max_parallel_patterns: usize,
    pub performance_tracking: bool,
}

impl Default for CognitiveOrchestratorConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_selection: true,
            enable_ensemble_methods: true,
            default_timeout_ms: 5000,
            max_parallel_patterns: 3,
            performance_tracking: true,
        }
    }
}

impl CognitiveOrchestrator {
    /// Get the brain graph for external use
    pub fn get_brain_graph(&self) -> Arc<BrainEnhancedKnowledgeGraph> {
        self.brain_graph.clone()
    }
    
    /// Assess query complexity (0.0 = simple, 1.0 = very complex)
    fn assess_complexity(&self, query: &str) -> f32 {
        let word_count = query.split_whitespace().count();
        let question_marks = query.matches('?').count();
        let query_lower = query.to_lowercase();
        let complex_indicators = [
            "analyze", "evaluate", "compare", "synthesize", "relationship",
            "system", "complex", "intricate", "multifaceted"
        ];
        let complex_words = complex_indicators.iter().filter(|&word| query_lower.contains(word)).count();
        
        let base_complexity = (word_count as f32 / 20.0).min(1.0);
        let question_bonus = (question_marks as f32 * 0.1).min(0.3);
        let complexity_bonus = (complex_words as f32 * 0.15).min(0.4);
        
        (base_complexity + question_bonus + complexity_bonus).min(1.0)
    }
    
    /// Assess query ambiguity (0.0 = clear, 1.0 = very ambiguous)
    fn assess_ambiguity(&self, query: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let ambiguous_indicators = [
            "maybe", "perhaps", "possibly", "might", "could", "various",
            "different", "some", "many", "several", "general", "broad"
        ];
        let ambiguous_words = ambiguous_indicators.iter().filter(|&word| query_lower.contains(word)).count();
        
        let pronoun_indicators = [
            "it", "this", "that", "they", "them", "these", "those"
        ];
        let pronouns = pronoun_indicators.iter().filter(|&word| query_lower.contains(word)).count();
        
        let base_ambiguity = if query.contains('?') { 0.2 } else { 0.5 };
        let ambiguous_bonus = (ambiguous_words as f32 * 0.2).min(0.5);
        let pronoun_bonus = (pronouns as f32 * 0.1).min(0.3);
        
        (base_ambiguity + ambiguous_bonus + pronoun_bonus).min(1.0)
    }
    
    /// Assess creativity requirement (0.0 = factual, 1.0 = highly creative)
    fn assess_creativity_need(&self, query_lower: &str) -> f32 {
        let creative_indicators = [
            "creative", "innovative", "imagine", "invent", "design", "brainstorm",
            "alternatives", "possibilities", "novel", "unique", "original",
            "different ways", "new approaches", "think outside"
        ];
        
        let mut creativity_score: f32 = 0.0;
        for indicator in &creative_indicators {
            if query_lower.contains(indicator) {
                creativity_score += 0.3;
            }
        }
        
        // Questions about "what if" are highly creative
        if query_lower.contains("what if") || query_lower.contains("imagine if") {
            creativity_score += 0.5;
        }
        
        creativity_score.min(1.0f32)
    }
    
    /// Assess factual focus (0.0 = creative, 1.0 = highly factual)
    fn assess_factual_focus(&self, query_lower: &str) -> f32 {
        let factual_indicators = [
            "what is", "define", "definition", "fact", "truth", "evidence",
            "prove", "demonstrate", "show", "research", "study", "data",
            "statistics", "number", "measurement", "scientific"
        ];
        
        let mut factual_score: f32 = 0.0;
        for indicator in &factual_indicators {
            if query_lower.contains(indicator) {
                factual_score += 0.2;
            }
        }
        
        // Direct questions are often factual
        if query_lower.starts_with("what ") || query_lower.starts_with("when ") ||
           query_lower.starts_with("where ") || query_lower.starts_with("who ") {
            factual_score += 0.4;
        }
        
        factual_score.min(1.0f32)
    }
    
    /// Assess systems thinking requirement
    fn assess_systems_need(&self, query_lower: &str) -> f32 {
        let systems_indicators = [
            "system", "hierarchy", "relationship", "connection", "network",
            "interaction", "influence", "affect", "impact", "consequence",
            "structure", "organization", "pattern", "whole", "interconnect"
        ];
        
        let mut systems_score: f32 = 0.0;
        for indicator in &systems_indicators {
            if query_lower.contains(indicator) {
                systems_score += 0.25;
            }
        }
        
        systems_score.min(1.0f32)
    }
    
    /// Assess critical analysis requirement
    fn assess_critical_need(&self, query_lower: &str, context_lower: &Option<String>) -> f32 {
        let critical_indicators = [
            "evaluate", "assess", "judge", "critique", "analyze", "examine",
            "compare", "contrast", "argue", "debate", "evidence", "proof",
            "validate", "verify", "question", "challenge", "assumption"
        ];
        
        let mut critical_score: f32 = 0.0;
        for indicator in &critical_indicators {
            if query_lower.contains(indicator) {
                critical_score += 0.3;
            }
        }
        
        // Check context for critical thinking cues
        if let Some(context) = context_lower {
            for indicator in &critical_indicators {
                if context.contains(indicator) {
                    critical_score += 0.2;
                }
            }
        }
        
        critical_score.min(1.0f32)
    }
    
    /// Assess abstraction requirement
    fn assess_abstraction_need(&self, query_lower: &str) -> f32 {
        let abstraction_indicators = [
            "pattern", "concept", "principle", "theory", "model",
            "generalize", "abstract", "essence", "underlying", "fundamental",
            "category", "class", "type", "kind", "meta"
        ];
        
        let mut abstraction_score: f32 = 0.0;
        for indicator in &abstraction_indicators {
            if query_lower.contains(indicator) {
                abstraction_score += 0.25;
            }
        }
        
        abstraction_score.min(1.0f32)
    }
    
    /// Detect lateral thinking cues
    fn detect_lateral_cues(&self, query_lower: &str) -> Vec<String> {
        let lateral_patterns = [
            ("think outside", "unconventional thinking"),
            ("different perspective", "alternative viewpoint"),
            ("creative solution", "innovative approach"),
            ("unexpected", "surprising connections"),
            ("paradox", "contradictory elements"),
            ("analogy", "metaphorical thinking"),
        ];
        
        let mut cues = Vec::new();
        for (pattern, description) in &lateral_patterns {
            if query_lower.contains(pattern) {
                cues.push(description.to_string());
            }
        }
        
        cues
    }
    
    /// Detect convergent thinking indicators
    fn detect_convergent_indicators(&self, query_lower: &str) -> Vec<String> {
        let convergent_patterns = [
            ("single answer", "unique solution sought"),
            ("correct answer", "factual accuracy required"),
            ("precise", "precision needed"),
            ("exact", "exactness required"),
            ("specific", "specificity needed"),
            ("definition", "definitional clarity"),
        ];
        
        let mut indicators = Vec::new();
        for (pattern, description) in &convergent_patterns {
            if query_lower.contains(pattern) {
                indicators.push(description.to_string());
            }
        }
        
        indicators
    }
    
    /// Detect divergent thinking indicators
    fn detect_divergent_indicators(&self, query_lower: &str) -> Vec<String> {
        let divergent_patterns = [
            ("alternatives", "multiple options sought"),
            ("possibilities", "various possibilities"),
            ("different ways", "diverse approaches"),
            ("brainstorm", "idea generation"),
            ("explore", "exploration needed"),
            ("various", "variety sought"),
        ];
        
        let mut indicators = Vec::new();
        for (pattern, description) in &divergent_patterns {
            if query_lower.contains(pattern) {
                indicators.push(description.to_string());
            }
        }
        
        indicators
    }
    
    /// Calculate fitness score for a pattern given query analysis
    fn calculate_pattern_fitness(&self, pattern_type: CognitivePatternType, analysis: &QueryAnalysis) -> f32 {
        match pattern_type {
            CognitivePatternType::Convergent => {
                // Convergent thinking excels at factual, precise queries
                analysis.factual_focus * 0.4 +
                (1.0 - analysis.ambiguity_level) * 0.3 +
                (1.0 - analysis.creativity_requirement) * 0.3
            },
            CognitivePatternType::Divergent => {
                // Divergent thinking excels at creative, exploratory queries
                analysis.creativity_requirement * 0.4 +
                analysis.ambiguity_level * 0.2 +
                (!analysis.divergent_indicators.is_empty() as u8 as f32) * 0.4
            },
            CognitivePatternType::Lateral => {
                // Lateral thinking for unexpected connections and creative insights
                analysis.creativity_requirement * 0.3 +
                (!analysis.lateral_thinking_cues.is_empty() as u8 as f32) * 0.4 +
                analysis.ambiguity_level * 0.3
            },
            CognitivePatternType::Systems => {
                // Systems thinking for hierarchical and interconnected analysis
                analysis.systems_thinking_need * 0.5 +
                analysis.complexity_level * 0.3 +
                analysis.abstraction_level * 0.2
            },
            CognitivePatternType::Critical => {
                // Critical thinking for evaluation and evidence-based reasoning
                analysis.critical_analysis_need * 0.5 +
                analysis.factual_focus * 0.3 +
                (1.0 - analysis.ambiguity_level) * 0.2
            },
            CognitivePatternType::Abstract => {
                // Abstract thinking for pattern recognition and conceptual analysis
                analysis.abstraction_level * 0.5 +
                analysis.complexity_level * 0.3 +
                analysis.systems_thinking_need * 0.2
            },
            CognitivePatternType::Adaptive => {
                // Adaptive thinking as a meta-coordinator - moderate for all
                0.6 // Always moderately useful
            },
            _ => {
                // Other patterns get base score
                0.3
            }
        }
    }
    /// Add complementary patterns to enhance reasoning
    async fn add_complementary_patterns(
        &self,
        selected_patterns: &mut Vec<CognitivePatternType>,
        analysis: &QueryAnalysis,
        pattern_scores: &AHashMap<CognitivePatternType, f32>,
    ) -> Result<()> {
        // Create a set to track existing patterns
        let mut pattern_set: std::collections::HashSet<CognitivePatternType> = 
            selected_patterns.iter().cloned().collect();
        
        // Add critical thinking if factual focus is high and not already selected
        if analysis.factual_focus > 0.7 && !pattern_set.contains(&CognitivePatternType::Critical) {
            if let Some(&score) = pattern_scores.get(&CognitivePatternType::Critical) {
                if score > 0.4 {
                    selected_patterns.push(CognitivePatternType::Critical);
                    pattern_set.insert(CognitivePatternType::Critical);
                }
            }
        }
        
        // Add systems thinking for complex queries
        if analysis.complexity_level > 0.6 && !pattern_set.contains(&CognitivePatternType::Systems) {
            if let Some(&score) = pattern_scores.get(&CognitivePatternType::Systems) {
                if score > 0.4 {
                    selected_patterns.push(CognitivePatternType::Systems);
                    pattern_set.insert(CognitivePatternType::Systems);
                }
            }
        }
        
        // Add divergent thinking for creative queries
        if analysis.creativity_requirement > 0.6 && !pattern_set.contains(&CognitivePatternType::Divergent) {
            if let Some(&score) = pattern_scores.get(&CognitivePatternType::Divergent) {
                if score > 0.4 {
                    selected_patterns.push(CognitivePatternType::Divergent);
                    pattern_set.insert(CognitivePatternType::Divergent);
                }
            }
        }
        
        // Limit total patterns to prevent cognitive overload
        selected_patterns.truncate(self.config.max_parallel_patterns);
        
        Ok(())
    }
    
    /// Determine execution strategy based on patterns and analysis
    fn determine_execution_strategy(
        &self,
        patterns: &[CognitivePatternType],
        analysis: &QueryAnalysis,
    ) -> ExecutionStrategy {
        if patterns.len() == 1 {
            return ExecutionStrategy::Sequential;
        }
        
        // Use parallel for creative, divergent queries
        if analysis.creativity_requirement > 0.7 || analysis.ambiguity_level > 0.7 {
            return ExecutionStrategy::Parallel;
        }
        
        // Use hybrid for complex queries with a clear primary approach
        if analysis.complexity_level > 0.6 {
            return ExecutionStrategy::Hybrid;
        }
        
        // Default to sequential for balanced approach
        ExecutionStrategy::Sequential
    }
    
    /// Create pattern-specific parameters based on analysis
    fn create_pattern_parameters(&self, pattern_type: CognitivePatternType, analysis: &QueryAnalysis) -> PatternParameters {
        let mut params = PatternParameters::default();
        
        // Adjust parameters based on pattern type and query analysis
        match pattern_type {
            CognitivePatternType::Convergent => {
                params.max_depth = Some(if analysis.complexity_level > 0.7 { 7 } else { 5 });
                params.activation_threshold = Some(0.6); // Higher threshold for precision
            },
            CognitivePatternType::Divergent => {
                params.exploration_breadth = Some(if analysis.creativity_requirement > 0.7 { 15 } else { 10 });
                params.creativity_threshold = Some(analysis.creativity_requirement.max(0.3));
            },
            CognitivePatternType::Critical => {
                params.validation_level = Some(if analysis.critical_analysis_need > 0.8 {
                    ValidationLevel::Rigorous
                } else if analysis.critical_analysis_need > 0.5 {
                    ValidationLevel::Comprehensive
                } else {
                    ValidationLevel::Basic
                });
            },
            CognitivePatternType::Systems => {
                params.max_depth = Some(if analysis.systems_thinking_need > 0.8 { 8 } else { 6 });
            },
            _ => {
                // Use defaults for other patterns
            }
        }
        
        params
    }
    
    /// Calculate pattern weight for integration phase
    fn calculate_pattern_weight_for_integration(
        &self,
        pattern_type: CognitivePatternType,
        result: &PatternResult,
        analysis: &QueryAnalysis,
    ) -> f32 {
        let base_weight = result.confidence;
        let fitness_weight = self.calculate_pattern_fitness(pattern_type, analysis);
        let convergence_bonus = if result.metadata.converged { 0.1 } else { 0.0 };
        
        // Combine weights with emphasis on confidence and fitness
        (base_weight * 0.6 + fitness_weight * 0.4 + convergence_bonus).min(1.0)
    }
    
    /// Apply competitive inhibition between patterns
    fn apply_competitive_inhibition(
        &self,
        weighted_contributions: &mut Vec<(CognitivePatternType, PatternResult, f32)>,
    ) {
        // Sort by weight (highest first)
        weighted_contributions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply inhibition - stronger patterns suppress weaker ones
        for i in 0..weighted_contributions.len() {
            let current_weight = weighted_contributions[i].2;
            
            for j in (i + 1)..weighted_contributions.len() {
                let inhibition_strength = (current_weight - weighted_contributions[j].2).max(0.0) * 0.3;
                weighted_contributions[j].2 = (weighted_contributions[j].2 - inhibition_strength).max(0.1);
            }
        }
    }
    
    /// Synthesize final answer from weighted contributions
    async fn synthesize_final_answer(
        &self,
        query: &str,
        weighted_contributions: &[(CognitivePatternType, PatternResult, f32)],
        analysis: &QueryAnalysis,
    ) -> Result<String> {
        if weighted_contributions.is_empty() {
            return Ok("Unable to generate a response based on available patterns.".to_string());
        }
        
        // For single pattern, return its result
        if weighted_contributions.len() == 1 {
            return Ok(weighted_contributions[0].1.answer.clone());
        }
        
        // For multiple patterns, create an integrated response
        let total_weight: f32 = weighted_contributions.iter().map(|(_, _, w)| w).sum();
        
        if total_weight == 0.0 {
            return Ok(weighted_contributions[0].1.answer.clone());
        }
        
        // Determine synthesis strategy based on query analysis
        if analysis.factual_focus > 0.7 {
            // For factual queries, prioritize the highest confidence answer
            Ok(weighted_contributions[0].1.answer.clone())
        } else if analysis.creativity_requirement > 0.7 {
            // For creative queries, combine multiple perspectives
            self.synthesize_creative_response(query, weighted_contributions, total_weight).await
        } else {
            // Balanced synthesis for other cases
            self.synthesize_balanced_response(query, weighted_contributions, total_weight).await
        }
    }
    
    /// Synthesize creative response combining multiple perspectives
    async fn synthesize_creative_response(
        &self,
        _query: &str,
        weighted_contributions: &[(CognitivePatternType, PatternResult, f32)],
        total_weight: f32,
    ) -> Result<String> {
        let mut synthesis = String::new();
        synthesis.push_str("Integrated Analysis:\n\n");
        
        for (pattern_type, result, weight) in weighted_contributions {
            if *weight / total_weight > 0.15 { // Only include significant contributions
                let contribution_percentage = (*weight / total_weight * 100.0) as u32;
                let pattern_name = match pattern_type {
                    CognitivePatternType::Convergent => "Focused Analysis",
                    CognitivePatternType::Divergent => "Creative Exploration",
                    CognitivePatternType::Lateral => "Unconventional Insights",
                    CognitivePatternType::Systems => "Systems Perspective",
                    CognitivePatternType::Critical => "Critical Evaluation",
                    CognitivePatternType::Abstract => "Pattern Recognition",
                    _ => "Additional Analysis",
                };
                
                synthesis.push_str(&format!(
                    "{}% {}: {}\n\n",
                    contribution_percentage,
                    pattern_name,
                    result.answer.trim()
                ));
            }
        }
        
        Ok(synthesis.trim().to_string())
    }
    
    /// Synthesize balanced response
    async fn synthesize_balanced_response(
        &self,
        _query: &str,
        weighted_contributions: &[(CognitivePatternType, PatternResult, f32)],
        total_weight: f32,
    ) -> Result<String> {
        // Use the highest weighted response as primary, with key insights from others
        let primary_answer = &weighted_contributions[0].1.answer;
        
        if weighted_contributions.len() == 1 || weighted_contributions[0].2 / total_weight > 0.8 {
            return Ok(primary_answer.clone());
        }
        
        let mut synthesis = primary_answer.clone();
        
        // Add supporting insights from other patterns if they're significant
        let mut additional_insights = Vec::new();
        for (_, result, weight) in &weighted_contributions[1..] {
            if *weight / total_weight > 0.2 && !result.answer.is_empty() {
                additional_insights.push(result.answer.trim());
            }
        }
        
        if !additional_insights.is_empty() {
            synthesis.push_str("\n\nAdditional considerations: ");
            synthesis.push_str(&additional_insights.join("; "));
        }
        
        Ok(synthesis)
    }
    
    /// Calculate integrated quality metrics
    fn calculate_integrated_quality_metrics(
        &self,
        weighted_contributions: &[(CognitivePatternType, PatternResult, f32)],
        _analysis: &QueryAnalysis,
    ) -> QualityMetrics {
        if weighted_contributions.is_empty() {
            return QualityMetrics::default();
        }
        
        let total_weight: f32 = weighted_contributions.iter().map(|(_, _, w)| w).sum();
        
        if total_weight == 0.0 {
            return QualityMetrics::default();
        }
        
        // Weighted average of quality metrics
        let mut overall_confidence = 0.0;
        let mut consistency_score = 0.0;
        let mut completeness_score = 0.0;
        let mut novelty_score = 0.0;
        let mut efficiency_score = 0.0;
        
        for (_, result, weight) in weighted_contributions {
            let normalized_weight = weight / total_weight;
            overall_confidence += result.confidence * normalized_weight;
            
            // Use reasonable defaults for pattern-specific metrics
            consistency_score += 0.8 * normalized_weight; // Patterns are generally consistent
            completeness_score += (result.reasoning_trace.len() as f32 / 10.0).min(1.0) * normalized_weight;
            
            // Novelty based on pattern type
            let pattern_novelty = match result.pattern_type {
                CognitivePatternType::Lateral => 0.9,
                CognitivePatternType::Divergent => 0.7,
                CognitivePatternType::Abstract => 0.6,
                _ => 0.4,
            };
            novelty_score += pattern_novelty * normalized_weight;
            
            efficiency_score += self.calculate_efficiency_score(&result.metadata) * normalized_weight;
        }
        
        // Consistency bonus for multiple agreeing patterns
        if weighted_contributions.len() > 1 {
            let agreement_bonus = 0.1;
            consistency_score = (consistency_score + agreement_bonus).min(1.0);
        }
        
        QualityMetrics {
            overall_confidence,
            consistency_score,
            completeness_score,
            novelty_score,
            efficiency_score,
            coherence_score: 0.85, // Default coherence score
        }
    }
    
    /// Create a new cognitive orchestrator with all patterns
    pub async fn new(
        brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
        config: CognitiveOrchestratorConfig,
    ) -> Result<Self> {
        let performance_monitor = Arc::new(PerformanceMonitor::new_with_defaults().await?);
        
        // Initialize all cognitive patterns
        let convergent = Arc::new(ConvergentThinking::new(
            brain_graph.clone(),
        ));
        
        let divergent = Arc::new(DivergentThinking::new(
            brain_graph.clone(),
        ));
        
        let lateral = Arc::new(LateralThinking::new(
            brain_graph.clone(),
        ));
        
        let systems = Arc::new(SystemsThinking::new(
            brain_graph.clone(),
        ));
        
        let critical = Arc::new(CriticalThinking::new(
            brain_graph.clone(),
        ));
        
        let abstract_pattern = Arc::new(AbstractThinking::new(
            brain_graph.clone(),
        ));
        
        let adaptive = Arc::new(AdaptiveThinking::new(
            brain_graph.clone(),
        ));
        
        // Create pattern registry
        let mut patterns: AHashMap<CognitivePatternType, Arc<dyn CognitivePattern>> = AHashMap::new();
        patterns.insert(CognitivePatternType::Convergent, convergent);
        patterns.insert(CognitivePatternType::Divergent, divergent);
        patterns.insert(CognitivePatternType::Lateral, lateral);
        patterns.insert(CognitivePatternType::Systems, systems);
        patterns.insert(CognitivePatternType::Critical, critical);
        patterns.insert(CognitivePatternType::Abstract, abstract_pattern);
        patterns.insert(CognitivePatternType::Adaptive, adaptive.clone());
        
        Ok(Self {
            patterns,
            adaptive_selector: adaptive,
            performance_monitor,
            brain_graph,
            config,
            runtime_profiler: None,
        })
    }
    
    /// Set runtime profiler for function tracing
    pub fn set_runtime_profiler(&mut self, profiler: Arc<RuntimeProfiler>) {
        self.runtime_profiler = Some(profiler);
    }
    
    /// Main reasoning entry point
    pub async fn reason(
        &self,
        query: &str,
        context: Option<&str>,
        strategy: ReasoningStrategy,
    ) -> Result<ReasoningResult> {
        let _trace = if let Some(profiler) = &self.runtime_profiler {
            Some(trace_function!(profiler, "cognitive_reason", query.len(), context.map(|c| c.len()).unwrap_or(0)))
        } else {
            None
        };
        
        let start_time = std::time::Instant::now();
        
        let result = match strategy.clone() {
            ReasoningStrategy::Automatic => {
                if self.config.enable_adaptive_selection {
                    self.execute_adaptive_reasoning(query, context).await?
                } else {
                    // Fallback to convergent thinking
                    self.execute_specific_pattern(query, context, CognitivePatternType::Convergent).await?
                }
            }
            ReasoningStrategy::Specific(pattern_type) => {
                self.execute_specific_pattern(query, context, pattern_type).await?
            }
            ReasoningStrategy::Ensemble(pattern_types) => {
                if self.config.enable_ensemble_methods {
                    self.execute_ensemble_reasoning(query, context, pattern_types).await?
                } else {
                    return Err(GraphError::UnsupportedOperation("Ensemble methods disabled".to_string()));
                }
            }
        };
        
        let execution_time = start_time.elapsed();
        
        // Record performance metrics
        if self.config.performance_tracking {
            let operation = Operation {
                name: format!("cognitive_reasoning_{:?}", strategy),
                operation_type: "reasoning".to_string(),
                custom_metrics: {
                    let mut metrics = std::collections::HashMap::new();
                    metrics.insert("confidence".to_string(), result.quality_metrics.overall_confidence as f64);
                    metrics.insert("patterns_used".to_string(), result.execution_metadata.patterns_executed.len() as f64);
                    metrics
                },
            };
            
            let _ = self.performance_monitor.end_operation_tracking(
                &format!("reasoning_{}", query.chars().take(20).collect::<String>()),
                &operation,
                execution_time,
                true,
            ).await;
        }
        
        Ok(result)
    }
    
    /// Execute adaptive reasoning (automatic pattern selection)
    async fn execute_adaptive_reasoning(
        &self,
        query: &str,
        context: Option<&str>,
    ) -> Result<ReasoningResult> {
        let start_time = std::time::Instant::now();
        
        // 1. Analyze query to determine optimal cognitive approach
        let query_characteristics = self.analyze_query_for_patterns(query, context).await?;
        
        // 2. Select appropriate patterns based on query analysis
        let selected_patterns = self.select_optimal_patterns(&query_characteristics).await?;
        
        // 3. Execute selected patterns with intelligent orchestration
        let pattern_results = self.execute_patterns_intelligently(
            query,
            context, 
            &selected_patterns,
            &query_characteristics
        ).await?;
        
        // 4. Integrate results using competitive inhibition and ensemble methods
        let integrated_result = self.integrate_pattern_results(
            query,
            pattern_results,
            &query_characteristics
        ).await?;
        
        let execution_time = start_time.elapsed();
        
        Ok(ReasoningResult {
            query: query.to_string(),
            final_answer: integrated_result.final_answer,
            strategy_used: ReasoningStrategy::Automatic,
            patterns_executed: selected_patterns,
            execution_metadata: ExecutionMetadata {
                total_time_ms: execution_time.as_millis() as u64,
                patterns_executed: vec![], // Duplicate patterns_executed field - will use outer one
                nodes_activated: integrated_result.total_nodes_activated,
                energy_consumed: integrated_result.total_energy_consumed,
                cache_hits: integrated_result.cache_hits,
                cache_misses: integrated_result.cache_misses,
            },
            quality_metrics: integrated_result.quality_metrics,
        })
    }
    
    /// Execute a specific cognitive pattern
    async fn execute_specific_pattern(
        &self,
        query: &str,
        context: Option<&str>,
        pattern_type: CognitivePatternType,
    ) -> Result<ReasoningResult> {
        let pattern = self.patterns.get(&pattern_type)
            .ok_or(GraphError::PatternNotFound(format!("{:?}", pattern_type)))?;
        
        let pattern_result = pattern.execute(
            query,
            context,
            PatternParameters::default(),
        ).await?;
        
        // Extract values before moving
        let answer = pattern_result.answer.clone();
        let confidence = pattern_result.confidence;
        let execution_time = pattern_result.metadata.execution_time_ms;
        let nodes_activated = pattern_result.metadata.nodes_activated;
        let energy_consumed = pattern_result.metadata.total_energy;
        let completeness_score = self.estimate_completeness(&pattern_result);
        let novelty_score = self.estimate_novelty(&pattern_result);
        let efficiency_score = self.calculate_efficiency_score(&pattern_result.metadata);
        
        Ok(ReasoningResult {
            query: query.to_string(),
            final_answer: answer,
            strategy_used: ReasoningStrategy::Specific(pattern_type),
            patterns_executed: vec![pattern_type],
            execution_metadata: ExecutionMetadata {
                total_time_ms: execution_time,
                patterns_executed: vec![], // Duplicate field - will use outer one
                nodes_activated,
                energy_consumed,
                cache_hits: 0,
                cache_misses: 0,
            },
            quality_metrics: QualityMetrics {
                overall_confidence: confidence,
                consistency_score: 1.0, // Single pattern is always consistent with itself
                completeness_score,
                novelty_score,
                efficiency_score,
                coherence_score: 0.8, // Default coherence for single pattern
            },
        })
    }
    
    /// Execute ensemble reasoning with multiple patterns
    async fn execute_ensemble_reasoning(
        &self,
        query: &str,
        context: Option<&str>,
        pattern_types: Vec<CognitivePatternType>,
    ) -> Result<ReasoningResult> {
        let mut pattern_results = Vec::new();
        let mut _tasks: Vec<tokio::task::JoinHandle<Result<PatternResult>>> = Vec::new();
        
        // Execute patterns in parallel (up to max_parallel_patterns)
        for chunk in pattern_types.chunks(self.config.max_parallel_patterns) {
            let mut chunk_tasks = Vec::new();
            
            for &pattern_type in chunk {
                if let Some(pattern) = self.patterns.get(&pattern_type) {
                    let pattern_clone = pattern.clone();
                    let query_clone = query.to_string();
                    let context_clone = context.map(|s| s.to_string());
                    
                    let task = tokio::spawn(async move {
                        pattern_clone.execute(
                            &query_clone,
                            context_clone.as_deref(),
                            PatternParameters::default(),
                        ).await
                    });
                    
                    chunk_tasks.push((pattern_type, task));
                }
            }
            
            // Wait for chunk to complete
            for (pattern_type, task) in chunk_tasks {
                match task.await {
                    Ok(Ok(result)) => pattern_results.push((pattern_type, result)),
                    Ok(Err(e)) => {
                        log::warn!("Pattern {:?} failed: {}", pattern_type, e);
                        // Continue with other patterns
                    }
                    Err(e) => {
                        log::warn!("Pattern {:?} task failed: {}", pattern_type, e);
                    }
                }
            }
        }
        
        if pattern_results.is_empty() {
            return Err(GraphError::ProcessingError("All patterns failed".to_string()));
        }
        
        // Merge results using ensemble methods
        let ensemble_result = self.merge_pattern_results(pattern_results).await?;
        
        Ok(ensemble_result)
    }
    
    /// Merge results from multiple patterns using ensemble methods
    async fn merge_pattern_results(
        &self,
        pattern_results: Vec<(CognitivePatternType, PatternResult)>,
    ) -> Result<ReasoningResult> {
        if pattern_results.is_empty() {
            return Err(GraphError::ProcessingError("No results to merge".to_string()));
        }
        
        // Simple confidence-weighted averaging for now
        // In a full implementation, this would use sophisticated ensemble methods
        let mut total_weighted_confidence = 0.0;
        let mut total_weight = 0.0;
        let mut best_answer = String::new();
        let mut best_confidence = 0.0;
        
        let executed_patterns: Vec<CognitivePatternType> = pattern_results.iter()
            .map(|(pattern_type, _)| *pattern_type)
            .collect();
        
        let mut total_execution_time = 0;
        let mut total_nodes_activated = 0;
        let mut total_energy = 0.0;
        
        // Collect all non-empty answers for potential combination
        let mut all_answers: Vec<(CognitivePatternType, String, f32)> = Vec::new();
        
        for (pattern_type, result) in &pattern_results {
            let weight = self.get_pattern_weight(*pattern_type);
            total_weighted_confidence += result.confidence * weight;
            total_weight += weight;
            
            if !result.answer.trim().is_empty() {
                all_answers.push((*pattern_type, result.answer.clone(), result.confidence));
            }
            
            if result.confidence > best_confidence && !result.answer.trim().is_empty() {
                best_confidence = result.confidence;
                best_answer = result.answer.clone();
            }
            
            total_execution_time += result.metadata.execution_time_ms;
            total_nodes_activated += result.metadata.nodes_activated;
            total_energy += result.metadata.total_energy;
        }
        
        // If no single best answer, combine results from multiple patterns
        if best_answer.is_empty() && !all_answers.is_empty() {
            best_answer = self.combine_pattern_answers(&all_answers);
        } else if best_answer.is_empty() {
            // Fallback: provide a summary of what was attempted
            best_answer = format!(
                "Ensemble reasoning attempted with {} patterns but found limited results. Patterns used: {:?}",
                pattern_results.len(),
                executed_patterns
            );
        }
        
        let ensemble_confidence = if total_weight > 0.0 {
            total_weighted_confidence / total_weight
        } else {
            best_confidence
        };
        
        Ok(ReasoningResult {
            query: pattern_results[0].1.metadata.additional_info.get("query")
                .unwrap_or(&"Unknown".to_string()).clone(),
            final_answer: best_answer,
            strategy_used: ReasoningStrategy::Ensemble(executed_patterns.clone()),
            patterns_executed: executed_patterns.clone(),
            execution_metadata: ExecutionMetadata {
                total_time_ms: total_execution_time,
                patterns_executed: vec![], // Duplicate field - will use outer one
                nodes_activated: total_nodes_activated,
                energy_consumed: total_energy,
                cache_hits: 0,
                cache_misses: 0,
            },
            quality_metrics: QualityMetrics {
                overall_confidence: ensemble_confidence,
                consistency_score: self.calculate_ensemble_consistency(&pattern_results),
                completeness_score: self.calculate_ensemble_completeness(&pattern_results),
                novelty_score: self.calculate_ensemble_novelty(&pattern_results),
                efficiency_score: self.calculate_ensemble_efficiency(&pattern_results),
                coherence_score: self.calculate_ensemble_coherence(&pattern_results),
            },
        })
    }
    
    /// Get available cognitive patterns
    fn get_available_patterns(&self) -> Vec<CognitivePatternType> {
        self.patterns.keys().cloned().collect()
    }
    
    /// Combine answers from multiple patterns into a coherent response
    fn combine_pattern_answers(&self, answers: &[(CognitivePatternType, String, f32)]) -> String {
        if answers.is_empty() {
            return String::new();
        }
        
        if answers.len() == 1 {
            return answers[0].1.clone();
        }
        
        // Group answers by pattern type for better organization
        let mut combined = String::new();
        combined.push_str("Combined insights from multiple cognitive patterns:\n\n");
        
        for (pattern_type, answer, confidence) in answers {
            if confidence > &0.1 {  // Only include answers with minimal confidence
                let pattern_name = match pattern_type {
                    CognitivePatternType::Convergent => "Factual Analysis",
                    CognitivePatternType::Divergent => "Exploratory Findings",
                    CognitivePatternType::Lateral => "Creative Connections",
                    CognitivePatternType::Systems => "Systems Perspective",
                    CognitivePatternType::Critical => "Critical Analysis",
                    CognitivePatternType::Abstract => "Pattern Recognition",
                    CognitivePatternType::Adaptive => "Adaptive Insights",
                    CognitivePatternType::ChainOfThought => "Chain of Thought",
                    CognitivePatternType::TreeOfThoughts => "Tree of Thoughts",
                    CognitivePatternType::Analytical => "Analytical Thinking",
                    CognitivePatternType::PatternRecognition => "Pattern Recognition",
                    CognitivePatternType::Linguistic => "Linguistic Analysis",
                    CognitivePatternType::Creative => "Creative Thinking",
                    CognitivePatternType::Ensemble => "Ensemble Method",
                    CognitivePatternType::Unknown => "Unknown Pattern",
                };
                combined.push_str(&format!("{}: {}\n", pattern_name, answer.trim()));
            }
        }
        
        combined.trim().to_string()
    }
    
    /// Get weight for a specific pattern in ensemble methods
    fn get_pattern_weight(&self, pattern_type: CognitivePatternType) -> f32 {
        // Default weights - could be learned over time
        match pattern_type {
            CognitivePatternType::Convergent => 1.0,
            CognitivePatternType::Divergent => 0.8,
            CognitivePatternType::Lateral => 0.6,
            CognitivePatternType::Systems => 0.9,
            CognitivePatternType::Critical => 1.1,
            CognitivePatternType::Abstract => 0.7,
            CognitivePatternType::Adaptive => 1.2,
            CognitivePatternType::ChainOfThought => 1.0,
            CognitivePatternType::TreeOfThoughts => 0.9,
            CognitivePatternType::Analytical => 1.0,
            CognitivePatternType::PatternRecognition => 0.8,
            CognitivePatternType::Linguistic => 0.7,
            CognitivePatternType::Creative => 0.9,
            CognitivePatternType::Ensemble => 1.0,
            CognitivePatternType::Unknown => 0.5,
        }
    }
    
    /// Calculate consistency score for pattern contributions
    fn calculate_consistency_score(&self, contributions: &[PatternContribution]) -> f32 {
        if contributions.len() < 2 {
            return 1.0;
        }
        
        // Simple consistency measure based on confidence variance
        let confidences: Vec<f32> = contributions.iter().map(|c| c.confidence).collect();
        let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let variance = confidences.iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f32>() / confidences.len() as f32;
        
        // Convert variance to consistency score (lower variance = higher consistency)
        1.0 / (1.0 + variance)
    }
    
    /// Calculate completeness score for pattern contributions
    fn calculate_completeness_score(&self, contributions: &[PatternContribution]) -> f32 {
        // Completeness based on number of patterns contributing and their weights
        let total_weight: f32 = contributions.iter().map(|c| c.contribution_weight).sum();
        let normalized_completeness = total_weight / contributions.len() as f32;
        normalized_completeness.min(1.0)
    }
    
    /// Calculate novelty score for pattern contributions
    fn calculate_novelty_score(&self, contributions: &[PatternContribution]) -> f32 {
        // Higher novelty if lateral or divergent thinking contributed significantly
        let mut novelty = 0.0;
        for contribution in contributions {
            let pattern_novelty = match contribution.pattern_type {
                CognitivePatternType::Lateral => 0.9,
                CognitivePatternType::Divergent => 0.7,
                CognitivePatternType::Abstract => 0.6,
                CognitivePatternType::Adaptive => 0.5,
                _ => 0.3,
            };
            novelty += pattern_novelty * contribution.contribution_weight;
        }
        novelty / contributions.len() as f32
    }
    
    /// Estimate completeness for single pattern result
    fn estimate_completeness(&self, result: &PatternResult) -> f32 {
        // Based on reasoning trace length and confidence
        let trace_completeness = (result.reasoning_trace.len() as f32 / 10.0).min(1.0);
        (trace_completeness + result.confidence) / 2.0
    }
    
    /// Estimate novelty for single pattern result
    fn estimate_novelty(&self, result: &PatternResult) -> f32 {
        match result.pattern_type {
            CognitivePatternType::Lateral => 0.9,
            CognitivePatternType::Divergent => 0.7,
            CognitivePatternType::Abstract => 0.6,
            _ => 0.3,
        }
    }
    
    /// Calculate efficiency score from execution metadata
    fn calculate_efficiency_score(&self, metadata: &ResultMetadata) -> f32 {
        // Efficiency based on time, energy, and convergence
        let time_efficiency = 1.0 / (1.0 + metadata.execution_time_ms as f32 / 1000.0);
        let energy_efficiency = 1.0 / (1.0 + metadata.total_energy);
        let convergence_bonus = if metadata.converged { 0.2 } else { 0.0 };
        
        (time_efficiency + energy_efficiency) / 2.0 + convergence_bonus
    }
    
    /// Calculate consistency for ensemble results
    fn calculate_ensemble_consistency(&self, results: &[(CognitivePatternType, PatternResult)]) -> f32 {
        if results.len() < 2 {
            return 1.0;
        }
        
        let confidences: Vec<f32> = results.iter().map(|(_, r)| r.confidence).collect();
        let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let variance = confidences.iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f32>() / confidences.len() as f32;
        
        1.0 / (1.0 + variance)
    }
    
    /// Calculate completeness for ensemble results
    fn calculate_ensemble_completeness(&self, results: &[(CognitivePatternType, PatternResult)]) -> f32 {
        let avg_completeness: f32 = results.iter()
            .map(|(_, r)| self.estimate_completeness(r))
            .sum::<f32>() / results.len() as f32;
        
        // Bonus for having multiple perspectives
        let diversity_bonus = 0.1 * (results.len() - 1) as f32;
        (avg_completeness + diversity_bonus).min(1.0)
    }
    
    /// Calculate novelty for ensemble results
    fn calculate_ensemble_novelty(&self, results: &[(CognitivePatternType, PatternResult)]) -> f32 {
        results.iter()
            .map(|(_, r)| self.estimate_novelty(r))
            .fold(0.0, f32::max) // Take the maximum novelty
    }
    
    /// Calculate efficiency for ensemble results
    fn calculate_ensemble_efficiency(&self, results: &[(CognitivePatternType, PatternResult)]) -> f32 {
        let avg_efficiency: f32 = results.iter()
            .map(|(_, r)| self.calculate_efficiency_score(&r.metadata))
            .sum::<f32>() / results.len() as f32;
        
        // Penalty for using multiple patterns (more resources)
        let resource_penalty = 0.1 * (results.len() - 1) as f32;
        (avg_efficiency - resource_penalty).max(0.0)
    }

    /// Calculate coherence score for ensemble results
    fn calculate_ensemble_coherence(&self, results: &[(CognitivePatternType, PatternResult)]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }
        
        // Average coherence across all pattern results
        let avg_coherence: f32 = results.iter()
            .map(|(_, r)| r.quality_scores.coherence_score)
            .sum::<f32>() / results.len() as f32;
        
        // Apply consistency bonus (more consistent results are more coherent)
        let consistency_bonus = self.calculate_ensemble_consistency(results) * 0.2;
        
        (avg_coherence + consistency_bonus).min(1.0)
    }
    
    /// Analyze query to determine optimal cognitive patterns
    async fn analyze_query_for_patterns(
        &self,
        query: &str,
        context: Option<&str>,
    ) -> Result<QueryAnalysis> {
        let query_lower = query.to_lowercase();
        let context_lower = context.map(|c| c.to_lowercase());
        
        // Analyze query characteristics for pattern selection
        let mut analysis = QueryAnalysis {
            complexity_level: self.assess_complexity(query),
            ambiguity_level: self.assess_ambiguity(query),
            creativity_requirement: self.assess_creativity_need(&query_lower),
            factual_focus: self.assess_factual_focus(&query_lower),
            systems_thinking_need: self.assess_systems_need(&query_lower),
            critical_analysis_need: self.assess_critical_need(&query_lower, &context_lower),
            abstraction_level: self.assess_abstraction_need(&query_lower),
            lateral_thinking_cues: self.detect_lateral_cues(&query_lower),
            convergent_indicators: self.detect_convergent_indicators(&query_lower),
            divergent_indicators: self.detect_divergent_indicators(&query_lower),
        };
        
        // Adjust analysis based on context
        if let Some(ctx) = &context_lower {
            analysis.adjust_for_context(ctx);
        }
        
        Ok(analysis)
    }
    
    /// Select optimal cognitive patterns based on query analysis
    async fn select_optimal_patterns(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<CognitivePatternType>> {
        let mut selected_patterns = Vec::new();
        let mut pattern_scores = AHashMap::new();
        
        // Score each pattern based on query characteristics
        for pattern_type in self.get_available_patterns() {
            let score = self.calculate_pattern_fitness(pattern_type, analysis);
            pattern_scores.insert(pattern_type, score);
        }
        
        // Select patterns based on scores and interaction effects
        let mut sorted_patterns: Vec<_> = pattern_scores.iter().collect();
        sorted_patterns.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Always include the best pattern
        if let Some((best_pattern, score)) = sorted_patterns.first() {
            if **score > 0.3 {
                selected_patterns.push(**best_pattern);
            }
        }
        
        // Add complementary patterns based on query needs
        self.add_complementary_patterns(&mut selected_patterns, analysis, &pattern_scores).await?;
        
        // Ensure we have at least one pattern
        if selected_patterns.is_empty() {
            selected_patterns.push(CognitivePatternType::Convergent); // Safe fallback
        }
        
        Ok(selected_patterns)
    }
    
    /// Execute patterns with intelligent coordination
    async fn execute_patterns_intelligently(
        &self,
        query: &str,
        context: Option<&str>,
        patterns: &[CognitivePatternType],
        analysis: &QueryAnalysis,
    ) -> Result<Vec<(CognitivePatternType, PatternResult)>> {
        let mut results = Vec::new();
        
        // Determine execution strategy
        let execution_strategy = self.determine_execution_strategy(patterns, analysis);
        
        match execution_strategy {
            ExecutionStrategy::Sequential => {
                // Execute patterns sequentially, using results from previous patterns
                let mut accumulated_context = context.map(|s| s.to_string());
                
                for &pattern_type in patterns {
                    let pattern = self.patterns.get(&pattern_type)
                        .ok_or(GraphError::PatternNotFound(format!("{:?}", pattern_type)))?;
                    
                    let parameters = self.create_pattern_parameters(pattern_type, analysis);
                    let result = pattern.execute(
                        query,
                        accumulated_context.as_deref(),
                        parameters,
                    ).await?;
                    
                    // Update context with results for next pattern
                    if !result.answer.is_empty() {
                        accumulated_context = Some(
                            format!(
                                "{} Previous analysis: {}",
                                accumulated_context.unwrap_or_default(),
                                result.answer
                            )
                        );
                    }
                    
                    results.push((pattern_type, result));
                }
            },
            ExecutionStrategy::Parallel => {
                // Execute compatible patterns in parallel
                let mut tasks = Vec::new();
                
                for &pattern_type in patterns {
                    if let Some(pattern) = self.patterns.get(&pattern_type) {
                        let pattern_clone = pattern.clone();
                        let query_clone = query.to_string();
                        let context_clone = context.map(|s| s.to_string());
                        let parameters = self.create_pattern_parameters(pattern_type, analysis);
                        
                        let task = tokio::spawn(async move {
                            pattern_clone.execute(
                                &query_clone,
                                context_clone.as_deref(),
                                parameters,
                            ).await.map(|result| (pattern_type, result))
                        });
                        
                        tasks.push(task);
                    }
                }
                
                // Collect results as they complete
                for task in tasks {
                    match task.await {
                        Ok(Ok(result)) => results.push(result),
                        Ok(Err(e)) => log::warn!("Pattern execution failed: {}", e),
                        Err(e) => log::warn!("Pattern task failed: {}", e),
                    }
                }
            },
            ExecutionStrategy::Hybrid => {
                // Execute primary pattern first, then supporting patterns in parallel
                if let Some(&primary_pattern) = patterns.first() {
                    let pattern = self.patterns.get(&primary_pattern)
                        .ok_or(GraphError::PatternNotFound(format!("{:?}", primary_pattern)))?;
                    
                    let parameters = self.create_pattern_parameters(primary_pattern, analysis);
                    let primary_result = pattern.execute(query, context, parameters).await?;
                    results.push((primary_pattern, primary_result.clone()));
                    
                    // Execute remaining patterns in parallel with primary result as context
                    let enhanced_context = Some(
                        format!(
                            "{} Primary analysis: {}",
                            context.unwrap_or(""),
                            primary_result.answer
                        )
                    );
                    
                    let mut tasks = Vec::new();
                    for &pattern_type in &patterns[1..] {
                        if let Some(pattern) = self.patterns.get(&pattern_type) {
                            let pattern_clone = pattern.clone();
                            let query_clone = query.to_string();
                            let context_clone = enhanced_context.clone();
                            let parameters = self.create_pattern_parameters(pattern_type, analysis);
                            
                            let task = tokio::spawn(async move {
                                pattern_clone.execute(
                                    &query_clone,
                                    context_clone.as_deref(),
                                    parameters,
                                ).await.map(|result| (pattern_type, result))
                            });
                            
                            tasks.push(task);
                        }
                    }
                    
                    for task in tasks {
                        match task.await {
                            Ok(Ok(result)) => results.push(result),
                            Ok(Err(e)) => log::warn!("Supporting pattern failed: {}", e),
                            Err(e) => log::warn!("Supporting task failed: {}", e),
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Integrate results from multiple patterns using advanced ensemble methods
    async fn integrate_pattern_results(
        &self,
        query: &str,
        pattern_results: Vec<(CognitivePatternType, PatternResult)>,
        analysis: &QueryAnalysis,
    ) -> Result<IntegratedResult> {
        if pattern_results.is_empty() {
            return Err(GraphError::ProcessingError("No pattern results to integrate".to_string()));
        }
        
        // Competitive inhibition - patterns compete based on confidence and relevance
        let mut weighted_contributions = Vec::new();
        let mut total_nodes_activated = 0;
        let mut total_energy_consumed = 0.0;
        let mut cache_hits = 0;
        let mut cache_misses = 0;
        
        for (pattern_type, result) in &pattern_results {
            let pattern_weight = self.calculate_pattern_weight_for_integration(
                *pattern_type,
                &result,
                analysis
            );
            
            weighted_contributions.push((*pattern_type, result.clone(), pattern_weight));
            total_nodes_activated += result.metadata.nodes_activated;
            total_energy_consumed += result.metadata.total_energy;
        }
        
        // Apply competitive inhibition
        self.apply_competitive_inhibition(&mut weighted_contributions);
        
        // Generate final integrated answer
        let final_answer = self.synthesize_final_answer(
            query,
            &weighted_contributions,
            analysis
        ).await?;
        
        // Calculate quality metrics
        let quality_metrics = self.calculate_integrated_quality_metrics(
            &weighted_contributions,
            analysis
        );
        
        Ok(IntegratedResult {
            final_answer,
            total_nodes_activated,
            total_energy_consumed,
            cache_hits,
            cache_misses,
            quality_metrics,
        })
    }
    
    /// Get orchestrator statistics
    pub async fn get_statistics(&self) -> Result<OrchestratorStatistics> {
        Ok(OrchestratorStatistics {
            total_patterns: self.patterns.len(),
            available_patterns: self.get_available_patterns(),
            performance_metrics: AHashMap::new(), // Placeholder until monitoring is available
            config: self.config.clone(),
        })
    }
    
    /// Get performance metrics for the orchestrator
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics> {
        let metrics = self.performance_monitor.get_metrics().await?;
        
        Ok(PerformanceMetrics {
            total_queries_processed: metrics.get("total_queries").unwrap_or(&0.0).clone() as u64,
            average_response_time_ms: metrics.get("avg_response_time").unwrap_or(&0.0).clone() as f64,
            pattern_usage_stats: {
                let mut usage_stats = AHashMap::new();
                for pattern_type in &self.get_available_patterns() {
                    let usage_key = format!("pattern_{:?}_usage", pattern_type);
                    let usage_count = metrics.get(&usage_key).unwrap_or(&0.0).clone() as u64;
                    usage_stats.insert(*pattern_type, usage_count);
                }
                usage_stats
            },
            success_rate: metrics.get("success_rate").unwrap_or(&0.95).clone() as f64,
            cache_hit_rate: metrics.get("cache_hit_rate").unwrap_or(&0.0).clone() as f64,
            memory_usage_mb: metrics.get("memory_usage_mb").unwrap_or(&0.0).clone() as f64,
            active_entities: metrics.get("active_entities").unwrap_or(&0.0).clone() as u64,
        })
    }
}

/// Query analysis structure for pattern selection
#[derive(Debug, Clone)]
struct QueryAnalysis {
    complexity_level: f32,
    ambiguity_level: f32,
    creativity_requirement: f32,
    factual_focus: f32,
    systems_thinking_need: f32,
    critical_analysis_need: f32,
    abstraction_level: f32,
    lateral_thinking_cues: Vec<String>,
    convergent_indicators: Vec<String>,
    divergent_indicators: Vec<String>,
}

impl QueryAnalysis {
    fn adjust_for_context(&mut self, context: &str) {
        // Adjust analysis based on contextual clues
        if context.contains("creative") || context.contains("innovative") {
            self.creativity_requirement = (self.creativity_requirement + 0.3).min(1.0);
        }
        if context.contains("fact") || context.contains("evidence") {
            self.factual_focus = (self.factual_focus + 0.3).min(1.0);
        }
        if context.contains("system") || context.contains("hierarchy") {
            self.systems_thinking_need = (self.systems_thinking_need + 0.3).min(1.0);
        }
    }
}

/// Execution strategy for cognitive patterns
#[derive(Debug, Clone)]
enum ExecutionStrategy {
    Sequential,  // Execute patterns one after another
    Parallel,    // Execute patterns simultaneously
    Hybrid,      // Primary pattern first, then supporting patterns in parallel
}

/// Integrated result from multiple patterns
#[derive(Debug, Clone)]
struct IntegratedResult {
    final_answer: String,
    total_nodes_activated: usize,
    total_energy_consumed: f32,
    cache_hits: usize,
    cache_misses: usize,
    quality_metrics: QualityMetrics,
}

/// Calculate pattern weight based on confidence, complexity, and priority
/// This is a private function used for pattern weighting in orchestration
pub(crate) fn calculate_pattern_weight(confidence: f32, complexity: u32, is_priority: bool) -> f32 {
    let mut weight = confidence;
    
    // Adjust for complexity (higher complexity reduces weight)
    weight *= 1.0 / (1.0 + complexity as f32 * 0.1);
    
    // Boost priority patterns
    if is_priority {
        weight *= 1.2;
    }
    
    weight
}

/// Statistics for the cognitive orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorStatistics {
    pub total_patterns: usize,
    pub available_patterns: Vec<CognitivePatternType>,
    pub performance_metrics: AHashMap<String, f32>,
    pub config: CognitiveOrchestratorConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_pattern_weight() {
        // Direct test of private function
        assert!(calculate_pattern_weight(0.9, 5, true) > calculate_pattern_weight(0.9, 5, false));
        assert!(calculate_pattern_weight(0.9, 5, true) > calculate_pattern_weight(0.7, 5, true));
        assert!(calculate_pattern_weight(0.9, 3, true) > calculate_pattern_weight(0.9, 10, true));
        
        // Additional test cases for edge conditions
        assert!(calculate_pattern_weight(1.0, 0, true) > calculate_pattern_weight(1.0, 0, false));
        assert!(calculate_pattern_weight(0.5, 1, false) > calculate_pattern_weight(0.3, 1, false));
        
        // Test that weight decreases with complexity
        assert!(calculate_pattern_weight(0.8, 1, false) > calculate_pattern_weight(0.8, 5, false));
        assert!(calculate_pattern_weight(0.8, 5, false) > calculate_pattern_weight(0.8, 10, false));
    }
}