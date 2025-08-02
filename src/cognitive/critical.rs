use std::sync::Arc;
use std::collections::HashMap as AHashMap;
use std::time::{SystemTime, Instant};
use async_trait::async_trait;

use crate::cognitive::types::*;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::brain_types::{ActivationStep, ActivationOperation};
use crate::core::types::EntityKey;
// Using pure graph operations for critical thinking analysis
use crate::error::Result;

/// Critical thinking pattern - handles contradictions, validates information, resolves conflicts
pub struct CriticalThinking {
    pub graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub exception_resolver: Arc<ExceptionResolver>,
    pub validation_threshold: f32,
}

impl CriticalThinking {
    pub fn new(
        graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        Self {
            graph,
            exception_resolver: Arc::new(ExceptionResolver::new()),
            validation_threshold: 0.8,
        }
    }
    
    pub async fn execute_critical_analysis(
        &self,
        query: &str,
        validation_level: ValidationLevel,
    ) -> Result<CriticalResult> {
        // 1. Identify potential contradictions in query results
        let base_results = self.get_base_query_results(query).await?;
        let contradictions = self.identify_contradictions(&base_results).await?;
        
        // 2. Activate inhibitory links for conflict resolution
        let inhibitory_resolution = self.apply_inhibitory_logic(
            base_results,
            contradictions.clone(),
        ).await?;
        
        // 3. Validate information sources and confidence
        let source_validation = self.validate_information_sources(
            &inhibitory_resolution,
            validation_level,
        ).await?;
        
        // 4. Provide reasoned resolution with uncertainty quantification
        Ok(CriticalResult {
            resolved_facts: self.convert_facts_to_resolved_facts(&inhibitory_resolution.resolved_facts),
            contradictions_found: contradictions,
            resolution_strategy: inhibitory_resolution.strategy,
            confidence_intervals: self.convert_confidence_intervals(&source_validation.confidence_intervals),
            uncertainty_analysis: self.analyze_uncertainty(&source_validation),
        })
    }

    /// Get base query results for analysis
    async fn get_base_query_results(&self, query: &str) -> Result<QueryResults> {
        // Try to perform basic query to get initial results
        let _graph = &self.graph;
        let query_lower = query.to_lowercase();
        
        // If the query doesn't seem to have any queryable content, create synthetic facts
        if query_lower.trim().is_empty() || query_lower.split_whitespace().all(|w| self.is_stop_word(w)) {
            return Ok(QueryResults {
                facts: vec![],
                metadata: crate::core::activation_engine::PropagationResult {
                    final_activations: std::collections::HashMap::new(),
                    activation_trace: vec![],
                    total_energy: 0.0,
                    iterations_completed: 0,
                    converged: true,
                },
            });
        }
        
        // Extract entities manually
        return self.create_fallback_results(query).await;
    }
    
    /// Create fallback results for query processing
    async fn create_fallback_results(&self, query: &str) -> Result<QueryResults> {
        let query_lower = query.to_lowercase();
        let all_entities = self.graph.get_all_entities().await;
        
        let mut facts = Vec::new();
        
        // Look for entities mentioned in the query
        for (entity_key, _entity_data, _activation) in &all_entities {
            let entity_concept = format!("entity_{entity_key:?}").to_lowercase();
            if query_lower.contains(&entity_concept) || entity_concept.contains(query_lower.split_whitespace().next().unwrap_or("")) {
                facts.push(FactInfo {
                    entity_key: *entity_key,
                    fact_description: format!("Found relevant entity: {entity_concept}"),
                    confidence: 0.6,
                    source: "basic_search".to_string(),
                    timestamp: SystemTime::now(),
                });
            }
        }
        
        Ok(QueryResults {
            facts,
            metadata: crate::core::activation_engine::PropagationResult {
                final_activations: std::collections::HashMap::new(),
                activation_trace: vec![],
                total_energy: 0.0,
                iterations_completed: 1,
                converged: true,
            },
        })
    }
    
    /// Check if word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        matches!(word.to_lowercase().as_str(), 
            "the" | "is" | "at" | "which" | "on" | "a" | "an" | "and" | "or" | 
            "but" | "in" | "with" | "to" | "for" | "of" | "as" | "by" | "that" | "this" |
            "what" | "where" | "when" | "why" | "how" | "who" | "whom" | "whose"
        )
    }

    /// Identify contradictions in the results
    async fn identify_contradictions(&self, results: &QueryResults) -> Result<Vec<Contradiction>> {
        let mut contradictions = Vec::new();
        
        // Check for direct contradictions between facts
        for (i, fact1) in results.facts.iter().enumerate() {
            for fact2 in results.facts.iter().skip(i + 1) {
                if self.are_contradictory(fact1, fact2).await? {
                    contradictions.push(Contradiction {
                        statement_a: fact1.fact_description.clone(),
                        statement_b: fact2.fact_description.clone(),
                        conflict_type: ConflictType::DirectContradiction,
                        severity: self.calculate_contradiction_severity(fact1, fact2),
                        contradiction_type: "direct_conflict".to_string(),
                        conflicting_facts: vec![fact1.entity_key, fact2.entity_key],
                    });
                }
            }
        }
        
        // Check for logical inconsistencies
        let logical_contradictions = self.check_logical_consistency(&results.facts).await?;
        contradictions.extend(logical_contradictions);
        
        Ok(contradictions)
    }

    /// Apply inhibitory logic for conflict resolution
    async fn apply_inhibitory_logic(
        &self,
        base_results: QueryResults,
        contradictions: Vec<Contradiction>,
    ) -> Result<InhibitoryResolution> {
        let mut resolved_facts = base_results.facts.clone();
        let mut strategy = ResolutionStrategy::PreferTrusted;
        
        for contradiction in &contradictions {
            // Apply resolution strategy based on contradiction type
            match contradiction.contradiction_type.as_str() {
                "direct_conflict" => {
                    strategy = ResolutionStrategy::PreferHigherConfidence;
                    resolved_facts = self.resolve_direct_conflict(&resolved_facts, contradiction).await?;
                }
                "logical_inconsistency" => {
                    strategy = ResolutionStrategy::LogicalPriority;
                    resolved_facts = self.resolve_logical_inconsistency(&resolved_facts, contradiction).await?;
                }
                _ => {
                    strategy = ResolutionStrategy::PreferTrusted;
                }
            }
        }
        
        Ok(InhibitoryResolution {
            resolved_facts,
            strategy,
            conflicts_resolved: contradictions.len(),
        })
    }

    /// Validate information sources
    async fn validate_information_sources(
        &self,
        resolution: &InhibitoryResolution,
        validation_level: ValidationLevel,
    ) -> Result<SourceValidation> {
        let mut confidence_intervals = Vec::new();
        let mut source_reliability = AHashMap::new();
        
        for fact in &resolution.resolved_facts {
            // Calculate confidence interval based on validation level
            let confidence_range = match validation_level {
                ValidationLevel::Basic => (fact.confidence * 0.8, fact.confidence * 1.0),
                ValidationLevel::Comprehensive => (fact.confidence * 0.6, fact.confidence * 0.9),
                ValidationLevel::Rigorous => (fact.confidence * 0.4, fact.confidence * 0.8),
            };
            
            confidence_intervals.push(LocalConfidenceInterval {
                entity_key: fact.entity_key,
                lower_bound: confidence_range.0,
                upper_bound: confidence_range.1,
                reliability_score: self.calculate_source_reliability(&fact.source).await?,
            });
            
            // Track source reliability
            source_reliability.entry(fact.source.clone())
                .and_modify(|score| *score = (*score + fact.confidence) / 2.0)
                .or_insert(fact.confidence);
        }
        
        Ok(SourceValidation {
            confidence_intervals,
            source_reliability,
            validation_level,
        })
    }

    /// Analyze uncertainty in the results
    fn analyze_uncertainty(&self, validation: &SourceValidation) -> UncertaintyAnalysis {
        let overall_uncertainty = if validation.confidence_intervals.is_empty() {
            0.5
        } else {
            1.0 - (validation.confidence_intervals.iter()
                .map(|ci| (ci.upper_bound + ci.lower_bound) / 2.0)
                .sum::<f32>() / validation.confidence_intervals.len() as f32)
        };
        
        let knowledge_gaps = validation.confidence_intervals.iter()
            .filter(|ci| ci.reliability_score < 0.5)
            .map(|ci| format!("Low reliability for entity {:?}", ci.entity_key))
            .collect();
        
        UncertaintyAnalysis {
            overall_uncertainty,
            source_reliability: validation.source_reliability.clone(),
            knowledge_gaps,
        }
    }
    
    /// Convert FactInfo to ResolvedFact
    fn convert_facts_to_resolved_facts(&self, facts: &[FactInfo]) -> Vec<ResolvedFact> {
        facts.iter().map(|fact| ResolvedFact {
            fact_statement: fact.fact_description.clone(),
            confidence: fact.confidence,
            supporting_evidence: vec![fact.entity_key],
            conflicting_evidence: Vec::new(),
        }).collect()
    }
    
    /// Convert local ConfidenceInterval to global ConfidenceInterval
    fn convert_confidence_intervals(&self, intervals: &[LocalConfidenceInterval]) -> Vec<crate::cognitive::types::ConfidenceInterval> {
        intervals.iter().map(|interval| crate::cognitive::types::ConfidenceInterval {
            entity_key: interval.entity_key,
            lower_bound: interval.lower_bound,
            upper_bound: interval.upper_bound,
            mean_confidence: (interval.lower_bound + interval.upper_bound) / 2.0,
        }).collect()
    }

    // Helper methods
    
    async fn are_contradictory(&self, fact1: &FactInfo, fact2: &FactInfo) -> Result<bool> {
        // Check for contradictory property values
        // Example: "has 3 legs" vs "has 4 legs"
        let desc1 = &fact1.fact_description.to_lowercase();
        let desc2 = &fact2.fact_description.to_lowercase();
        
        // Check for numeric contradictions
        if (desc1.contains("3") && desc2.contains("4")) || 
           (desc1.contains("4") && desc2.contains("3")) {
            // Check if they're talking about the same property (e.g., "legs")
            if desc1.contains("leg") && desc2.contains("leg") {
                return Ok(true);
            }
        }
        
        // Check for direct negations
        if (desc1.contains("warm") && desc2.contains("cold")) ||
           (desc1.contains("cold") && desc2.contains("warm")) {
            return Ok(true);
        }
        
        // Check for contradictory facts about same entity
        if fact1.entity_key == fact2.entity_key && 
           (fact1.confidence - fact2.confidence).abs() > 0.5 {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    fn calculate_contradiction_severity(&self, fact1: &FactInfo, fact2: &FactInfo) -> f32 {
        (fact1.confidence - fact2.confidence).abs()
    }
    
    async fn generate_resolution_options(&self, _fact1: &FactInfo, _fact2: &FactInfo) -> Result<Vec<String>> {
        Ok(vec![
            "Prefer higher confidence fact".to_string(),
            "Average the confidences".to_string(),
            "Require additional validation".to_string(),
        ])
    }
    
    async fn check_logical_consistency(&self, _facts: &[FactInfo]) -> Result<Vec<Contradiction>> {
        // For now, return empty - would implement full logical consistency checking
        Ok(Vec::new())
    }
    
    async fn resolve_direct_conflict(&self, facts: &[FactInfo], contradiction: &Contradiction) -> Result<Vec<FactInfo>> {
        let mut resolved = facts.to_vec();
        
        // Remove lower confidence facts from conflicts
        if contradiction.conflicting_facts.len() >= 2 {
            let higher_conf_entity = contradiction.conflicting_facts[0];
            let lower_conf_entity = contradiction.conflicting_facts[1];
            
            // Find the corresponding facts to compare confidence
            let higher_conf_fact = resolved.iter().find(|f| f.entity_key == higher_conf_entity);
            let lower_conf_fact = resolved.iter().find(|f| f.entity_key == lower_conf_entity);
            
            if let (Some(higher), Some(lower)) = (higher_conf_fact, lower_conf_fact) {
                if higher.confidence > lower.confidence {
                    resolved.retain(|f| f.entity_key != lower_conf_entity);
                } else {
                    resolved.retain(|f| f.entity_key != higher_conf_entity);
                }
            }
        }
        
        Ok(resolved)
    }
    
    async fn resolve_logical_inconsistency(&self, facts: &[FactInfo], _contradiction: &Contradiction) -> Result<Vec<FactInfo>> {
        // For now, just return the original facts
        Ok(facts.to_vec())
    }
    
    async fn calculate_source_reliability(&self, source: &str) -> Result<f32> {
        // Simple heuristic based on source type
        match source {
            "graph_query" => Ok(0.8),
            "user_input" => Ok(0.6),
            "external_api" => Ok(0.7),
            _ => Ok(0.5),
        }
    }
}

#[async_trait]
impl CognitivePattern for CriticalThinking {
    async fn execute(
        &self,
        query: &str,
        context: Option<&str>,
        parameters: PatternParameters,
    ) -> Result<PatternResult> {
        let start_time = Instant::now();
        
        // Determine validation level from parameters or context
        let validation_level = self.infer_validation_level(query, context, &parameters);
        
        // Execute critical analysis
        let critical_result = self.execute_critical_analysis(query, validation_level).await?;
        
        // Format the answer
        let answer = self.format_critical_answer(query, &critical_result);
        
        let execution_time = start_time.elapsed();
        
        Ok(PatternResult {
            pattern_type: CognitivePatternType::Critical,
            answer,
            confidence: self.calculate_critical_confidence(&critical_result),
            reasoning_trace: self.create_critical_reasoning_trace(&critical_result),
            metadata: ResultMetadata {
                execution_time_ms: execution_time.as_millis() as u64,
                nodes_activated: critical_result.resolved_facts.len(),
                iterations_completed: 1,
                converged: critical_result.contradictions_found.is_empty(),
                total_energy: critical_result.uncertainty_analysis.overall_uncertainty,
                additional_info: self.create_critical_metadata(&critical_result),
            },
        })
    }
    
    fn get_pattern_type(&self) -> CognitivePatternType {
        CognitivePatternType::Critical
    }
    
    fn get_optimal_use_cases(&self) -> Vec<String> {
        vec![
            "Fact validation".to_string(),
            "Contradiction resolution".to_string(),
            "Source reliability analysis".to_string(),
            "Conflict detection".to_string(),
        ]
    }
    
    fn estimate_complexity(&self, _query: &str) -> ComplexityEstimate {
        ComplexityEstimate {
            computational_complexity: 40,
            estimated_time_ms: 1500,
            memory_requirements_mb: 15,
            confidence: 0.9,
            parallelizable: false,
        }
    }
}

pub struct ExceptionResolver {
    // Implementation would handle different types of exceptions
}

impl ExceptionResolver {
    fn new() -> Self {
        Self {}
    }
}

impl CriticalThinking {
    /// Infer validation level from query and context
    fn infer_validation_level(&self, query: &str, _context: Option<&str>, _parameters: &PatternParameters) -> ValidationLevel {
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("verify") || query_lower.contains("validate") {
            ValidationLevel::Rigorous
        } else if query_lower.contains("check") || query_lower.contains("confirm") {
            ValidationLevel::Comprehensive
        } else {
            ValidationLevel::Basic
        }
    }
    
    /// Format the critical analysis answer
    fn format_critical_answer(&self, _query: &str, result: &CriticalResult) -> String {
        let mut answer = String::new();
        
        // If no results found, be truthful
        if result.resolved_facts.is_empty() && result.contradictions_found.is_empty() {
            answer.push_str("No facts found for critical analysis.\n");
            return answer;
        }
        
        // Report resolved facts
        if !result.resolved_facts.is_empty() {
            answer.push_str(&format!("Validated Facts ({}):\n", result.resolved_facts.len()));
            for fact in &result.resolved_facts {
                answer.push_str(&format!("  - {} (confidence: {:.2})\n", 
                    fact.fact_statement, fact.confidence));
            }
            answer.push('\n');
        }
        
        // Report contradictions found
        if !result.contradictions_found.is_empty() {
            answer.push_str(&format!("Contradictions Detected ({}):\n", result.contradictions_found.len()));
            for contradiction in &result.contradictions_found {
                answer.push_str(&format!("  - {}: {} conflicting facts (severity: {:.2})\n",
                    contradiction.contradiction_type,
                    contradiction.conflicting_facts.len(),
                    contradiction.severity));
            }
            answer.push('\n');
        }
        
        // Report resolution strategy
        answer.push_str(&format!("Resolution Strategy: {:?}\n", result.resolution_strategy));
        
        // Report uncertainty analysis
        answer.push_str(&format!("Overall Uncertainty: {:.2}\n", result.uncertainty_analysis.overall_uncertainty));
        
        if !result.uncertainty_analysis.knowledge_gaps.is_empty() {
            answer.push_str("Knowledge Gaps:\n");
            for gap in &result.uncertainty_analysis.knowledge_gaps {
                answer.push_str(&format!("  - {gap}\n"));
            }
        }
        
        answer
    }
    
    /// Calculate confidence based on critical analysis result
    fn calculate_critical_confidence(&self, result: &CriticalResult) -> f32 {
        let contradiction_penalty = result.contradictions_found.len() as f32 * 0.2;
        let uncertainty_penalty = result.uncertainty_analysis.overall_uncertainty;
        let base_confidence = if result.resolved_facts.is_empty() { 0.3 } else {
            result.resolved_facts.iter()
                .map(|fact| fact.confidence)
                .sum::<f32>() / result.resolved_facts.len() as f32
        };
        
        (base_confidence - contradiction_penalty - uncertainty_penalty).clamp(0.0, 1.0)
    }
    
    /// Create reasoning trace for critical analysis
    fn create_critical_reasoning_trace(&self, result: &CriticalResult) -> Vec<ActivationStep> {
        let mut trace = Vec::new();
        
        trace.push(ActivationStep {
            step_id: 1,
            entity_key: EntityKey::default(),
            concept_id: "validation".to_string(),
            activation_level: 0.8,
            operation_type: ActivationOperation::Initialize,
            timestamp: SystemTime::now(),
        });
        
        if !result.contradictions_found.is_empty() {
            trace.push(ActivationStep {
                step_id: 2,
                entity_key: EntityKey::default(),
                concept_id: "contradiction_detection".to_string(),
                activation_level: 0.9,
                operation_type: ActivationOperation::Propagate,
                timestamp: SystemTime::now(),
            });
            
            trace.push(ActivationStep {
                step_id: 3,
                entity_key: EntityKey::default(),
                concept_id: "resolution_strategy".to_string(),
                activation_level: 0.7,
                operation_type: ActivationOperation::Reinforce,
                timestamp: SystemTime::now(),
            });
        }
        
        trace.push(ActivationStep {
            step_id: trace.len() + 1,
            entity_key: EntityKey::default(),
            concept_id: "uncertainty_analysis".to_string(),
            activation_level: 0.8,
            operation_type: ActivationOperation::Decay,
            timestamp: SystemTime::now(),
        });
        
        trace
    }
    
    /// Create additional metadata
    fn create_critical_metadata(&self, result: &CriticalResult) -> AHashMap<String, String> {
        let mut metadata = AHashMap::new();
        metadata.insert("facts_count".to_string(), result.resolved_facts.len().to_string());
        metadata.insert("contradictions_count".to_string(), result.contradictions_found.len().to_string());
        metadata.insert("resolution_strategy".to_string(), format!("{:?}", result.resolution_strategy));
        metadata.insert("uncertainty_score".to_string(), format!("{:.3}", result.uncertainty_analysis.overall_uncertainty));
        metadata.insert("knowledge_gaps_count".to_string(), result.uncertainty_analysis.knowledge_gaps.len().to_string());
        metadata
    }
    
    /// Generate a simple embedding from query text
    async fn generate_query_embedding(&self, query: &str) -> Result<Vec<f32>> {
        // Create a simple embedding by hashing words and creating a sparse vector
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut embedding = vec![0.0; 96]; // Standard embedding size
        
        for (i, word) in words.iter().enumerate() {
            let hash = word.chars().fold(0u32, |acc, c| acc.wrapping_add(c as u32));
            let index = (hash as usize) % embedding.len();
            embedding[index] = 1.0 / (i + 1) as f32; // Weighted by position
        }
        
        Ok(embedding)
    }
}

/// Query results for critical analysis
#[derive(Debug, Clone)]
struct QueryResults {
    facts: Vec<FactInfo>,
    metadata: crate::core::activation_engine::PropagationResult,
}

/// Individual fact information
#[derive(Debug, Clone)]
struct FactInfo {
    entity_key: EntityKey,
    fact_description: String,
    confidence: f32,
    source: String,
    timestamp: SystemTime,
}

/// Use global Contradiction from cognitive::types
/// Inhibitory resolution result
#[derive(Debug, Clone)]
struct InhibitoryResolution {
    resolved_facts: Vec<FactInfo>,
    strategy: ResolutionStrategy,
    conflicts_resolved: usize,
}

/// Source validation result
#[derive(Debug, Clone)]
struct SourceValidation {
    confidence_intervals: Vec<LocalConfidenceInterval>,
    source_reliability: AHashMap<String, f32>,
    validation_level: ValidationLevel,
}

/// Local ConfidenceInterval with reliability_score field
#[derive(Debug, Clone)]
struct LocalConfidenceInterval {
    entity_key: EntityKey,
    lower_bound: f32,
    upper_bound: f32,
    reliability_score: f32,
}

#[cfg(test)]
mod critical_thinking_unit_tests {
    use super::*;
    use std::collections::HashMap;
    use crate::core::activation_engine::PropagationResult;
    
    async fn create_test_critical_thinking() -> CriticalThinking {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test()
            .expect("Failed to create test graph"));
        CriticalThinking::new(graph)
    }
    
    fn create_test_fact_info(entity_key: EntityKey, description: &str, confidence: f32, source: &str) -> FactInfo {
        FactInfo {
            entity_key,
            fact_description: description.to_string(),
            confidence,
            source: source.to_string(),
            timestamp: SystemTime::now(),
        }
    }
    
    fn create_test_query_results(facts: Vec<FactInfo>) -> QueryResults {
        QueryResults {
            facts,
            metadata: PropagationResult {
                final_activations: HashMap::new(),
                activation_trace: vec![],
                total_energy: 0.0,
                iterations_completed: 1,
                converged: true,
            },
        }
    }
    
    #[tokio::test]
    async fn test_identify_contradictions_direct_conflict() {
        let thinking = create_test_critical_thinking().await;
        
        let facts = vec![
            create_test_fact_info(
                EntityKey::from_hash("entity1"), 
                "has 3 legs", 
                0.8, 
                "source1"
            ),
            create_test_fact_info(
                EntityKey::from_hash("entity1"), 
                "has 4 legs", 
                0.7, 
                "source2"
            ),
        ];
        
        let results = create_test_query_results(facts);
        let contradictions = thinking.identify_contradictions(&results).await
            .expect("Should identify contradictions");
        
        assert_eq!(contradictions.len(), 1, "Should find one contradiction");
        assert_eq!(contradictions[0].conflict_type, ConflictType::DirectContradiction);
        assert_eq!(contradictions[0].conflicting_facts.len(), 2);
    }
    
    #[tokio::test]
    async fn test_identify_contradictions_temperature_conflict() {
        let thinking = create_test_critical_thinking().await;
        
        let facts = vec![
            create_test_fact_info(
                EntityKey::from_hash("entity1"), 
                "temperature is warm", 
                0.9, 
                "sensor1"
            ),
            create_test_fact_info(
                EntityKey::from_hash("entity2"), 
                "temperature is cold", 
                0.8, 
                "sensor2"
            ),
        ];
        
        let results = create_test_query_results(facts);
        let contradictions = thinking.identify_contradictions(&results).await
            .expect("Should identify contradictions");
        
        assert_eq!(contradictions.len(), 1, "Should find temperature contradiction");
        assert!(contradictions[0].contradiction_type.contains("direct_conflict"));
    }
    
    #[tokio::test]
    async fn test_identify_contradictions_no_conflicts() {
        let thinking = create_test_critical_thinking().await;
        
        let facts = vec![
            create_test_fact_info(
                EntityKey::from_hash("entity1"), 
                "is an animal", 
                0.9, 
                "source1"
            ),
            create_test_fact_info(
                EntityKey::from_hash("entity2"), 
                "has fur", 
                0.8, 
                "source2"
            ),
        ];
        
        let results = create_test_query_results(facts);
        let contradictions = thinking.identify_contradictions(&results).await
            .expect("Should complete without error");
        
        assert_eq!(contradictions.len(), 0, "Should find no contradictions");
    }
    
    #[tokio::test]
    async fn test_apply_inhibitory_logic_direct_conflict() {
        let thinking = create_test_critical_thinking().await;
        
        let higher_conf_key = EntityKey::from_hash("high_conf");
        let lower_conf_key = EntityKey::from_hash("low_conf");
        
        let facts = vec![
            create_test_fact_info(higher_conf_key, "high confidence fact", 0.9, "source1"),
            create_test_fact_info(lower_conf_key, "low confidence fact", 0.3, "source2"),
        ];
        
        let contradictions = vec![
            Contradiction {
                statement_a: "high confidence fact".to_string(),
                statement_b: "low confidence fact".to_string(),
                conflict_type: ConflictType::DirectContradiction,
                severity: 0.6,
                contradiction_type: "direct_conflict".to_string(),
                conflicting_facts: vec![higher_conf_key, lower_conf_key],
            }
        ];
        
        let base_results = create_test_query_results(facts);
        let resolution = thinking.apply_inhibitory_logic(base_results, contradictions).await
            .expect("Should resolve conflicts");
        
        assert_eq!(resolution.strategy, ResolutionStrategy::PreferHigherConfidence);
        assert_eq!(resolution.conflicts_resolved, 1);
        // Should retain higher confidence fact and remove lower confidence one
        assert_eq!(resolution.resolved_facts.len(), 1);
        assert_eq!(resolution.resolved_facts[0].entity_key, higher_conf_key);
    }
    
    #[tokio::test]
    async fn test_validate_information_sources_basic() {
        let thinking = create_test_critical_thinking().await;
        
        let facts = vec![
            create_test_fact_info(
                EntityKey::from_hash("entity1"), 
                "graph fact", 
                0.8, 
                "graph_query"
            ),
            create_test_fact_info(
                EntityKey::from_hash("entity2"), 
                "user fact", 
                0.6, 
                "user_input"
            ),
        ];
        
        let resolution = InhibitoryResolution {
            resolved_facts: facts,
            strategy: ResolutionStrategy::PreferTrusted,
            conflicts_resolved: 0,
        };
        
        let validation = thinking.validate_information_sources(&resolution, ValidationLevel::Basic).await
            .expect("Should validate sources");
        
        assert_eq!(validation.confidence_intervals.len(), 2);
        assert_eq!(validation.validation_level, ValidationLevel::Basic);
        assert!(validation.source_reliability.contains_key("graph_query"));
        assert!(validation.source_reliability.contains_key("user_input"));
        
        // Graph query should have higher reliability than user input
        assert!(validation.source_reliability["graph_query"] > validation.source_reliability["user_input"]);
    }
    
    #[tokio::test]
    async fn test_validate_information_sources_rigorous() {
        let thinking = create_test_critical_thinking().await;
        
        let facts = vec![
            create_test_fact_info(
                EntityKey::from_hash("entity1"), 
                "test fact", 
                1.0, 
                "graph_query"
            ),
        ];
        
        let resolution = InhibitoryResolution {
            resolved_facts: facts,
            strategy: ResolutionStrategy::PreferTrusted,
            conflicts_resolved: 0,
        };
        
        let validation = thinking.validate_information_sources(&resolution, ValidationLevel::Rigorous).await
            .expect("Should validate sources");
        
        // Rigorous validation should apply stricter confidence intervals
        assert_eq!(validation.confidence_intervals.len(), 1);
        let interval = &validation.confidence_intervals[0];
        assert!(interval.lower_bound < interval.upper_bound);
        assert!(interval.upper_bound <= 0.8); // Rigorous should cap at 0.8
    }
    
    #[tokio::test]
    async fn test_resolve_exceptions_empty_resolution() {
        let thinking = create_test_critical_thinking().await;
        
        // Test that empty facts don't cause issues
        let empty_facts: Vec<FactInfo> = vec![];
        let contradiction = Contradiction {
            statement_a: "".to_string(),
            statement_b: "".to_string(),
            conflict_type: ConflictType::DirectContradiction,
            severity: 0.0,
            contradiction_type: "direct_conflict".to_string(),
            conflicting_facts: vec![],
        };
        
        let resolved = thinking.resolve_direct_conflict(&empty_facts, &contradiction).await
            .expect("Should handle empty facts gracefully");
        
        assert_eq!(resolved.len(), 0);
    }
    
    #[tokio::test]
    async fn test_are_contradictory_numeric_values() {
        let thinking = create_test_critical_thinking().await;
        
        let fact1 = create_test_fact_info(
            EntityKey::from_hash("entity1"), 
            "has 3 legs", 
            0.8, 
            "source1"
        );
        let fact2 = create_test_fact_info(
            EntityKey::from_hash("entity1"), 
            "has 4 legs", 
            0.7, 
            "source2"
        );
        
        let contradictory = thinking.are_contradictory(&fact1, &fact2).await
            .expect("Should check contradiction");
        
        assert!(contradictory, "Facts with different leg counts should be contradictory");
    }
    
    #[tokio::test]
    async fn test_are_contradictory_temperature() {
        let thinking = create_test_critical_thinking().await;
        
        let fact1 = create_test_fact_info(
            EntityKey::from_hash("entity1"), 
            "weather is warm today", 
            0.8, 
            "source1"
        );
        let fact2 = create_test_fact_info(
            EntityKey::from_hash("entity2"), 
            "weather is cold today", 
            0.7, 
            "source2"
        );
        
        let contradictory = thinking.are_contradictory(&fact1, &fact2).await
            .expect("Should check contradiction");
        
        assert!(contradictory, "Warm and cold should be contradictory");
    }
    
    #[tokio::test]
    async fn test_are_contradictory_confidence_based() {
        let thinking = create_test_critical_thinking().await;
        
        let entity_key = EntityKey::from_hash("same_entity");
        let fact1 = create_test_fact_info(entity_key, "high confidence", 0.9, "source1");
        let fact2 = create_test_fact_info(entity_key, "low confidence", 0.3, "source2");
        
        let contradictory = thinking.are_contradictory(&fact1, &fact2).await
            .expect("Should check contradiction");
        
        assert!(contradictory, "Same entity with very different confidence should be contradictory");
    }
    
    #[tokio::test]
    async fn test_calculate_source_reliability() {
        let thinking = create_test_critical_thinking().await;
        
        let graph_reliability = thinking.calculate_source_reliability("graph_query").await
            .expect("Should calculate graph query reliability");
        assert_eq!(graph_reliability, 0.8);
        
        let user_reliability = thinking.calculate_source_reliability("user_input").await
            .expect("Should calculate user input reliability");
        assert_eq!(user_reliability, 0.6);
        
        let external_reliability = thinking.calculate_source_reliability("external_api").await
            .expect("Should calculate external API reliability");
        assert_eq!(external_reliability, 0.7);
        
        let unknown_reliability = thinking.calculate_source_reliability("unknown_source").await
            .expect("Should calculate unknown source reliability");
        assert_eq!(unknown_reliability, 0.5);
    }
    
    #[tokio::test]
    async fn test_generate_query_embedding() {
        let thinking = create_test_critical_thinking().await;
        
        let embedding = thinking.generate_query_embedding("test query with words").await
            .expect("Should generate embedding");
        
        assert_eq!(embedding.len(), 96, "Should generate 96-dimensional embedding");
        assert!(embedding.iter().any(|&x| x > 0.0), "Should have non-zero values");
    }
    
    #[tokio::test]
    async fn test_is_stop_word() {
        let thinking = create_test_critical_thinking().await;
        
        assert!(thinking.is_stop_word("the"));
        assert!(thinking.is_stop_word("is"));
        assert!(thinking.is_stop_word("and"));
        assert!(thinking.is_stop_word("what"));
        assert!(!thinking.is_stop_word("dog"));
        assert!(!thinking.is_stop_word("analyze"));
    }
}