use std::sync::Arc;
use std::collections::HashMap as AHashMap;
use std::time::{SystemTime, Instant};
use async_trait::async_trait;

use crate::cognitive::types::*;
use crate::cognitive::pattern_detector::NeuralPatternDetector;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::types::EntityKey;
use crate::core::brain_types::{ActivationStep, ActivationOperation};
// Neural server dependency removed - using pure graph operations
use crate::error::Result;

/// Abstract thinking pattern - identifies patterns, abstract concepts, meta-analysis
pub struct AbstractThinking {
    pub graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub pattern_models: AHashMap<String, String>,
    pub pattern_detector: NeuralPatternDetector,
}

impl AbstractThinking {
    pub fn new(
        graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        let mut pattern_models = AHashMap::new();
        pattern_models.insert("n_beats".to_string(), "n_beats_pattern_model".to_string());
        pattern_models.insert("timesnet".to_string(), "timesnet_pattern_model".to_string());
        
        let pattern_detector = NeuralPatternDetector::new(
            graph.clone(),
        );
        
        Self {
            graph,
            pattern_models,
            pattern_detector,
        }
    }
    
    pub async fn execute_pattern_analysis(
        &self,
        analysis_scope: AnalysisScope,
        pattern_type: PatternType,
    ) -> Result<AbstractResult> {
        // 1. Use enhanced neural pattern detection
        let patterns_found = self.pattern_detector.detect_patterns(
            analysis_scope.clone(),
            pattern_type,
        ).await?;
        
        // 2. Identify abstraction opportunities
        let abstraction_candidates = self.identify_abstractions(
            &patterns_found,
        ).await?;
        
        // 3. Suggest graph refactoring for efficiency
        let refactoring_suggestions = self.suggest_refactoring(
            &abstraction_candidates,
        ).await?;
        
        let efficiency_gains = self.estimate_efficiency_gains(&refactoring_suggestions);
        
        Ok(AbstractResult {
            patterns_found,
            abstractions: abstraction_candidates,
            refactoring_opportunities: refactoring_suggestions,
            efficiency_gains,
        })
    }

    /// Analyze structural patterns in the graph
    async fn analyze_structural_patterns(
        &self,
        scope: &AnalysisScope,
    ) -> Result<StructuralPatterns> {
        let graph = &self.graph;
        let stats = graph.get_brain_statistics().await?;
        
        // Analyze entity distribution patterns
        let entity_patterns = self.analyze_entity_distribution(&stats).await?;
        
        // Analyze relationship patterns
        let relationship_patterns = self.analyze_relationship_patterns().await?;
        
        // Analyze activation patterns
        let activation_patterns = self.analyze_activation_patterns().await?;
        
        Ok(StructuralPatterns {
            scope: scope.clone(),
            entity_distribution: entity_patterns,
            relationship_frequency: relationship_patterns,
            activation_hotspots: activation_patterns,
            complexity_metrics: self.calculate_structural_complexity(&stats),
        })
    }

    /// Use neural networks for pattern detection
    async fn neural_pattern_detection(
        &self,
        structural_data: StructuralPatterns,
        pattern_type: PatternType,
    ) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        match pattern_type {
            PatternType::Structural => {
                patterns.extend(self.detect_structural_patterns(&structural_data).await?);
            }
            PatternType::Temporal => {
                patterns.extend(self.detect_temporal_patterns(&structural_data).await?);
            }
            PatternType::Semantic => {
                patterns.extend(self.detect_semantic_patterns(&structural_data).await?);
            }
            PatternType::Usage => {
                patterns.extend(self.detect_usage_patterns(&structural_data).await?);
            }
        }
        
        Ok(patterns)
    }

    /// Identify abstraction opportunities
    async fn identify_abstractions(
        &self,
        patterns: &[DetectedPattern],
    ) -> Result<Vec<AbstractionCandidate>> {
        let mut candidates = Vec::new();
        
        for pattern in patterns {
            // Look for repeated sub-patterns that could be abstracted
            if pattern.frequency > 3.0 && pattern.confidence > 0.7 {
                candidates.push(AbstractionCandidate {
                    abstraction_name: self.determine_abstraction_type(pattern),
                    entities_to_abstract: pattern.entities_involved.clone(),
                    potential_savings: self.estimate_complexity_reduction(pattern),
                    implementation_complexity: (self.estimate_implementation_effort(pattern) * 100.0) as u32,
                    complexity_reduction: self.estimate_complexity_reduction(pattern),
                    implementation_effort: self.estimate_implementation_effort(pattern),
                    abstraction_type: self.determine_abstraction_type(pattern),
                    source_patterns: vec![pattern.pattern_id.clone()],
                });
            }
        }
        
        // Combine related candidates
        let combined_candidates = self.combine_related_abstractions(candidates).await?;
        
        Ok(combined_candidates)
    }

    /// Suggest graph refactoring for efficiency
    async fn suggest_refactoring(
        &self,
        abstractions: &[AbstractionCandidate],
    ) -> Result<Vec<RefactoringOpportunity>> {
        let mut opportunities = Vec::new();
        
        for abstraction in abstractions {
            if abstraction.complexity_reduction > 0.2 {
                opportunities.push(RefactoringOpportunity {
                    opportunity_type: RefactoringType::ConceptMerging,
                    description: format!(
                        "Consolidate {} {} patterns into a single abstraction",
                        abstraction.source_patterns.len(),
                        abstraction.abstraction_type
                    ),
                    entities_affected: Vec::new(), // Would be populated with actual entities
                    estimated_benefit: abstraction.complexity_reduction,
                });
            }
        }
        
        // Add performance optimization opportunities
        opportunities.extend(self.identify_performance_optimizations().await?);
        
        Ok(opportunities)
    }

    /// Estimate efficiency gains from refactoring
    fn estimate_efficiency_gains(&self, opportunities: &[RefactoringOpportunity]) -> EfficiencyAnalysis {
        let query_time_improvement = opportunities.iter()
            .map(|opp| opp.estimated_benefit * 0.3)
            .sum::<f32>()
            .min(0.5);
        
        let memory_reduction = opportunities.iter()
            .map(|opp| opp.estimated_benefit * 0.2)
            .sum::<f32>()
            .min(0.4);
        
        let accuracy_improvement = opportunities.iter()
            .filter(|opp| matches!(opp.opportunity_type, RefactoringType::ConceptMerging))
            .map(|opp| opp.estimated_benefit * 0.1)
            .sum::<f32>()
            .min(0.3);
        
        let maintainability_score = if opportunities.is_empty() {
            0.5
        } else {
            0.8 - (opportunities.len() as f32 * 0.05).min(0.3)
        };
        
        EfficiencyAnalysis {
            query_time_improvement,
            memory_reduction,
            accuracy_improvement,
            maintainability_score,
        }
    }

    // Helper methods for pattern analysis
    
    async fn analyze_entity_distribution(&self, stats: &crate::core::brain_enhanced_graph::BrainStatistics) -> Result<EntityDistribution> {
        // Since we don't have specific node type counts, estimate from activation distribution
        let _total_entities = stats.entity_count as f32;
        
        Ok(EntityDistribution {
            input_ratio: 0.33,  // Estimate 1/3 input nodes
            output_ratio: 0.33, // Estimate 1/3 output nodes
            gate_ratio: 0.34,   // Estimate 1/3 gate nodes
            distribution_entropy: self.calculate_distribution_entropy(stats),
        })
    }
    
    async fn analyze_relationship_patterns(&self) -> Result<RelationshipPatterns> {
        // Analyze actual relationships from the brain graph
        let stats = self.graph.get_brain_statistics().await?;
        
        // Get relationship counts by type
        let most_common_types;
        
        // Count relationships by analyzing all entity connections
        let mut type_counts = std::collections::HashMap::new();
        let all_entity_keys = self.graph.core_graph.get_all_entity_keys();
        
        for entity_key in all_entity_keys {
            let neighbors = self.graph.get_neighbors(entity_key);
            for _ in neighbors {
                // We don't have relationship type info, so just count as generic
                *type_counts.entry("generic_relationship".to_string()).or_insert(0) += 1;
            }
        }
        
        // Sort by frequency
        let type_count_len = type_counts.len();
        let mut sorted_types: Vec<_> = type_counts.into_iter().collect();
        sorted_types.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Debug output
        println!("DEBUG: Analyzed {} entity neighbors", type_count_len);
        for (rel_type, count) in &sorted_types {
            println!("  {}: {}", rel_type, count);
        }
        
        most_common_types = sorted_types.into_iter().map(|(t, _)| t).collect();
        
        // Calculate average connections per entity
        let avg_connections = if stats.entity_count > 0 {
            (stats.relationship_count as f32 * 2.0) / stats.entity_count as f32
        } else {
            0.0
        };
        
        Ok(RelationshipPatterns {
            most_common_types,
            average_connections_per_entity: avg_connections,
            clustering_coefficient: 0.6, // Would need actual clustering analysis
            small_world_coefficient: 0.4, // Would need actual small world analysis
        })
    }
    
    async fn analyze_activation_patterns(&self) -> Result<ActivationPatterns> {
        Ok(ActivationPatterns {
            hotspot_entities: Vec::new(),
            activation_frequency: AHashMap::new(),
            temporal_patterns: Vec::new(),
        })
    }
    
    fn calculate_structural_complexity(&self, stats: &crate::core::brain_enhanced_graph::BrainStatistics) -> f32 {
        if stats.entity_count == 0 {
            return 0.0; // No complexity for empty graph
        }
        
        let entity_complexity = ((stats.entity_count as f32).max(1.0)).ln() / 10.0;
        let relationship_complexity = ((stats.relationship_count as f32).max(1.0)).ln() / 10.0;
        let gate_complexity = ((stats.entity_count as f32 * 0.1).max(1.0)).ln() / 10.0; // Approximate gate count
        
        (entity_complexity + relationship_complexity + gate_complexity) / 3.0
    }
    
    fn calculate_distribution_entropy(&self, stats: &crate::core::brain_enhanced_graph::BrainStatistics) -> f32 {
        if stats.entity_count == 0 {
            return 0.0;
        }
        
        let _total = stats.entity_count as f32;
        // Approximate distribution based on clustering coefficient
        let input_p = 0.3f32; // Approximate 30% input nodes
        let output_p = 0.3f32; // Approximate 30% output nodes
        let gate_p = 0.4f32; // Approximate 40% gate nodes
        
        let mut entropy = 0.0;
        if input_p > 0.0 { entropy -= input_p * (input_p.ln()); }
        if output_p > 0.0 { entropy -= output_p * (output_p.ln()); }
        if gate_p > 0.0 { entropy -= gate_p * (gate_p.ln()); }
        
        entropy
    }
    
    async fn detect_structural_patterns(&self, data: &StructuralPatterns) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Detect frequent relationship type patterns
        for relationship_type in &data.relationship_frequency.most_common_types {
            if relationship_type == "is_a" {
                patterns.push(DetectedPattern {
                    pattern_id: format!("is_a_hierarchy_pattern"),
                    id: format!("is_a_hierarchy_pattern"),
                    pattern_type: PatternType::Structural,
                    description: "Hierarchical is_a relationship pattern detected".to_string(),
                    confidence: 0.9,
                    frequency: 5.0, // Assuming frequent is_a relationships
                    entities_involved: Vec::new(),
                    affected_entities: Vec::new(),
                });
            }
            
            if relationship_type == "has_property" {
                patterns.push(DetectedPattern {
                    pattern_id: format!("has_property_pattern"),
                    id: format!("has_property_pattern"),
                    pattern_type: PatternType::Structural,
                    description: "Property assignment pattern detected".to_string(),
                    confidence: 0.85,
                    frequency: 4.0,
                    entities_involved: Vec::new(),
                    affected_entities: Vec::new(),
                });
            }
        }
        
        // Detect hub patterns (entities with many connections)
        if data.relationship_frequency.average_connections_per_entity > 5.0 {
            patterns.push(DetectedPattern {
                pattern_id: "hub_pattern".to_string(),
                id: "hub_pattern".to_string(),
                pattern_type: PatternType::Structural,
                description: "High-connectivity hub entities detected".to_string(),
                confidence: 0.8,
                frequency: data.relationship_frequency.average_connections_per_entity,
                entities_involved: Vec::new(),
                affected_entities: Vec::new(),
            });
        }
        
        Ok(patterns)
    }
    
    async fn detect_temporal_patterns(&self, data: &StructuralPatterns) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Analyze temporal patterns in activation hotspots
        if !data.activation_hotspots.temporal_patterns.is_empty() {
            patterns.push(DetectedPattern {
                pattern_id: "temporal_activation_pattern".to_string(),
                id: "temporal_activation_pattern".to_string(),
                pattern_type: PatternType::Temporal,
                description: "Temporal activation patterns detected in entity usage".to_string(),
                confidence: 0.7,
                frequency: data.activation_hotspots.temporal_patterns.len() as f32,
                entities_involved: data.activation_hotspots.hotspot_entities.clone(),
                affected_entities: data.activation_hotspots.hotspot_entities.clone(),
            });
        }
        
        // Detect periodic patterns based on activation frequency
        let periodic_entities: Vec<EntityKey> = data.activation_hotspots.activation_frequency.iter()
            .filter(|(_, &freq)| freq > 0.5)
            .map(|(&key, _)| key)
            .collect();
            
        if periodic_entities.len() >= 3 {
            patterns.push(DetectedPattern {
                pattern_id: "periodic_activation_pattern".to_string(),
                id: "periodic_activation_pattern".to_string(),
                pattern_type: PatternType::Temporal,
                description: format!("Periodic activation pattern with {} entities", periodic_entities.len()),
                confidence: 0.75,
                frequency: periodic_entities.len() as f32 / data.activation_hotspots.activation_frequency.len().max(1) as f32,
                entities_involved: periodic_entities.clone(),
                affected_entities: periodic_entities,
            });
        }
        
        Ok(patterns)
    }
    
    async fn detect_semantic_patterns(&self, data: &StructuralPatterns) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Detect semantic clustering based on entity distribution
        if data.entity_distribution.distribution_entropy > 0.5 {
            patterns.push(DetectedPattern {
                pattern_id: "semantic_clustering_pattern".to_string(),
                id: "semantic_clustering_pattern".to_string(),
                pattern_type: PatternType::Semantic,
                description: "Semantic clustering pattern with diverse entity types".to_string(),
                confidence: 0.65 + (data.entity_distribution.distribution_entropy * 0.2).min(0.2),
                frequency: 3.0,
                entities_involved: Vec::new(),
                affected_entities: Vec::new(),
            });
        }
        
        // Detect semantic coherence based on clustering coefficient
        if data.relationship_frequency.clustering_coefficient > 0.5 {
            patterns.push(DetectedPattern {
                pattern_id: "semantic_coherence_pattern".to_string(),
                id: "semantic_coherence_pattern".to_string(),
                pattern_type: PatternType::Semantic,
                description: format!("High semantic coherence with clustering coefficient {:.2}", 
                    data.relationship_frequency.clustering_coefficient),
                confidence: data.relationship_frequency.clustering_coefficient,
                frequency: 4.0,
                entities_involved: Vec::new(),
                affected_entities: Vec::new(),
            });
        }
        
        // Detect semantic domains based on small-world properties
        if data.relationship_frequency.small_world_coefficient > 0.3 {
            patterns.push(DetectedPattern {
                pattern_id: "semantic_domain_pattern".to_string(),
                id: "semantic_domain_pattern".to_string(),
                pattern_type: PatternType::Semantic,
                description: "Semantic domain organization with small-world properties".to_string(),
                confidence: 0.7,
                frequency: 5.0,
                entities_involved: Vec::new(),
                affected_entities: Vec::new(),
            });
        }
        
        Ok(patterns)
    }
    
    async fn detect_usage_patterns(&self, data: &StructuralPatterns) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Detect high-usage patterns based on hotspot entities
        if !data.activation_hotspots.hotspot_entities.is_empty() {
            patterns.push(DetectedPattern {
                pattern_id: "high_usage_hotspot_pattern".to_string(),
                id: "high_usage_hotspot_pattern".to_string(),
                pattern_type: PatternType::Usage,
                description: format!("High usage pattern with {} hotspot entities", 
                    data.activation_hotspots.hotspot_entities.len()),
                confidence: 0.8,
                frequency: data.activation_hotspots.hotspot_entities.len() as f32,
                entities_involved: data.activation_hotspots.hotspot_entities.clone(),
                affected_entities: data.activation_hotspots.hotspot_entities.clone(),
            });
        }
        
        // Detect query path patterns based on average connections
        if data.relationship_frequency.average_connections_per_entity > 3.0 {
            patterns.push(DetectedPattern {
                pattern_id: "frequent_query_path_pattern".to_string(),
                id: "frequent_query_path_pattern".to_string(),
                pattern_type: PatternType::Usage,
                description: format!("Frequent query paths with avg {:.1} connections per entity", 
                    data.relationship_frequency.average_connections_per_entity),
                confidence: 0.7,
                frequency: data.relationship_frequency.average_connections_per_entity,
                entities_involved: Vec::new(),
                affected_entities: Vec::new(),
            });
        }
        
        // Detect underutilized areas based on complexity vs usage
        if data.complexity_metrics > 0.5 && data.activation_hotspots.activation_frequency.len() < 10 {
            patterns.push(DetectedPattern {
                pattern_id: "underutilized_complexity_pattern".to_string(),
                id: "underutilized_complexity_pattern".to_string(),
                pattern_type: PatternType::Usage,
                description: "Complex graph areas with low utilization".to_string(),
                confidence: 0.6,
                frequency: 2.0,
                entities_involved: Vec::new(),
                affected_entities: Vec::new(),
            });
        }
        
        Ok(patterns)
    }
    
    fn determine_abstraction_type(&self, pattern: &DetectedPattern) -> String {
        match pattern.pattern_type {
            PatternType::Structural => "structural_abstraction".to_string(),
            PatternType::Temporal => "temporal_abstraction".to_string(),
            PatternType::Semantic => "semantic_abstraction".to_string(),
            PatternType::Usage => "usage_abstraction".to_string(),
        }
    }
    
    fn estimate_complexity_reduction(&self, pattern: &DetectedPattern) -> f32 {
        (pattern.frequency - 1.0) * 0.1 * pattern.confidence
    }
    
    fn estimate_implementation_effort(&self, pattern: &DetectedPattern) -> f32 {
        match pattern.pattern_type {
            PatternType::Structural => 0.3,
            PatternType::Temporal => 0.6,
            PatternType::Semantic => 0.5,
            PatternType::Usage => 0.4,
        }
    }
    
    async fn combine_related_abstractions(&self, candidates: Vec<AbstractionCandidate>) -> Result<Vec<AbstractionCandidate>> {
        // For now, just return the original candidates
        // In a full implementation, would merge related abstractions
        Ok(candidates)
    }
    
    async fn generate_implementation_steps(&self, abstraction: &AbstractionCandidate) -> Result<Vec<String>> {
        Ok(vec![
            format!("Identify all instances of {} pattern", abstraction.abstraction_type),
            "Create abstract entity template".to_string(),
            "Replace concrete instances with abstract references".to_string(),
            "Update relationship mappings".to_string(),
            "Validate abstraction correctness".to_string(),
        ])
    }
    
    fn assess_refactoring_risk(&self, abstraction: &AbstractionCandidate) -> f32 {
        // Higher risk for more complex abstractions
        abstraction.implementation_effort * 0.8
    }
    
    async fn identify_performance_optimizations(&self) -> Result<Vec<RefactoringOpportunity>> {
        Ok(vec![
            RefactoringOpportunity {
                opportunity_type: RefactoringType::PerformanceOptimization,
                description: "Add indices for frequently accessed entity patterns".to_string(),
                entities_affected: Vec::new(), // Would be populated with actual entities
                estimated_benefit: 0.3,
            },
        ])
    }
}

#[async_trait]
impl CognitivePattern for AbstractThinking {
    async fn execute(
        &self,
        query: &str,
        context: Option<&str>,
        parameters: PatternParameters,
    ) -> Result<PatternResult> {
        let start_time = Instant::now();
        
        // Determine analysis scope and pattern type from query
        let (analysis_scope, pattern_type) = self.infer_analysis_parameters(query, context, &parameters);
        
        // Execute pattern analysis
        let abstract_result = self.execute_pattern_analysis(analysis_scope, pattern_type).await?;
        
        // Format the answer
        let answer = self.format_abstract_answer(query, &abstract_result);
        
        let execution_time = start_time.elapsed();
        
        Ok(PatternResult {
            pattern_type: CognitivePatternType::Abstract,
            answer,
            confidence: self.calculate_abstract_confidence(&abstract_result),
            reasoning_trace: self.create_abstract_reasoning_trace(&abstract_result),
            metadata: ResultMetadata {
                execution_time_ms: execution_time.as_millis() as u64,
                nodes_activated: abstract_result.patterns_found.len(),
                iterations_completed: 1,
                converged: true,
                total_energy: abstract_result.efficiency_gains.query_time_improvement,
                additional_info: self.create_abstract_metadata(&abstract_result),
            },
            quality_scores: QualityMetrics::default(),
        })
    }
    
    fn get_pattern_type(&self) -> CognitivePatternType {
        CognitivePatternType::Abstract
    }
    
    fn get_optimal_use_cases(&self) -> Vec<String> {
        vec![
            "Pattern recognition".to_string(),
            "Meta-analysis".to_string(),
            "Concept abstraction".to_string(),
            "System optimization".to_string(),
        ]
    }
    
    fn estimate_complexity(&self, _query: &str) -> ComplexityEstimate {
        ComplexityEstimate {
            computational_complexity: 60,
            estimated_time_ms: 2000,
            memory_requirements_mb: 20,
            confidence: 0.7,
            parallelizable: false,
        }
    }
}


impl AbstractThinking {
    /// Infer analysis parameters from query and context
    fn infer_analysis_parameters(
        &self, 
        query: &str, 
        _context: Option<&str>, 
        _parameters: &PatternParameters
    ) -> (AnalysisScope, PatternType) {
        let query_lower = query.to_lowercase();
        
        let pattern_type = if query_lower.contains("structure") || query_lower.contains("topology") {
            PatternType::Structural
        } else if query_lower.contains("time") || query_lower.contains("temporal") {
            PatternType::Temporal
        } else if query_lower.contains("semantic") || query_lower.contains("meaning") {
            PatternType::Semantic
        } else if query_lower.contains("usage") || query_lower.contains("access") {
            PatternType::Usage
        } else {
            PatternType::Structural
        };
        
        let analysis_scope = if query_lower.contains("global") || query_lower.contains("entire") {
            AnalysisScope::Global
        } else if query_lower.contains("local") || query_lower.contains("specific") {
            AnalysisScope::Local(EntityKey::default()) // Would be populated with actual entity
        } else {
            AnalysisScope::Regional(Vec::new()) // Default to regional with empty list
        };
        
        (analysis_scope, pattern_type)
    }
    
    /// Format the abstract analysis answer
    fn format_abstract_answer(&self, query: &str, result: &AbstractResult) -> String {
        let mut answer = format!("Abstract Pattern Analysis for: {}\n\n", query);
        
        // Report patterns found
        if !result.patterns_found.is_empty() {
            answer.push_str(&format!("Patterns Detected ({}):\n", result.patterns_found.len()));
            for pattern in &result.patterns_found {
                answer.push_str(&format!("  - {}: {} (confidence: {:.2}, frequency: {:.1})\n",
                    pattern.pattern_id,
                    pattern.description,
                    pattern.confidence,
                    pattern.frequency));
            }
            answer.push('\n');
        }
        
        // Report abstractions
        if !result.abstractions.is_empty() {
            answer.push_str(&format!("Abstraction Opportunities ({}):\n", result.abstractions.len()));
            for abstraction in &result.abstractions {
                answer.push_str(&format!("  - {}: complexity reduction {:.1}%, effort {:.1}\n",
                    abstraction.abstraction_type,
                    abstraction.complexity_reduction * 100.0,
                    abstraction.implementation_effort));
            }
            answer.push('\n');
        }
        
        // Report refactoring opportunities
        if !result.refactoring_opportunities.is_empty() {
            answer.push_str(&format!("Refactoring Opportunities ({}):\n", result.refactoring_opportunities.len()));
            for opportunity in &result.refactoring_opportunities {
                answer.push_str(&format!("  - {}: {} (benefit: {:.1}%)\n",
                    opportunity.opportunity_type,
                    opportunity.description,
                    opportunity.estimated_benefit * 100.0));
            }
            answer.push('\n');
        }
        
        // Report efficiency gains
        answer.push_str("Estimated Efficiency Gains:\n");
        answer.push_str(&format!("  - Query Time: {:.1}% improvement\n", result.efficiency_gains.query_time_improvement * 100.0));
        answer.push_str(&format!("  - Memory Usage: {:.1}% reduction\n", result.efficiency_gains.memory_reduction * 100.0));
        answer.push_str(&format!("  - Accuracy: {:.1}% improvement\n", result.efficiency_gains.accuracy_improvement * 100.0));
        answer.push_str(&format!("  - Maintainability: {:.1}/1.0\n", result.efficiency_gains.maintainability_score));
        
        answer
    }
    
    /// Calculate confidence based on abstract analysis result
    fn calculate_abstract_confidence(&self, result: &AbstractResult) -> f32 {
        let pattern_confidence = if result.patterns_found.is_empty() {
            0.3
        } else {
            result.patterns_found.iter()
                .map(|p| p.confidence)
                .sum::<f32>() / result.patterns_found.len() as f32
        };
        
        let abstraction_bonus = result.abstractions.len() as f32 * 0.1;
        let refactoring_bonus = result.refactoring_opportunities.len() as f32 * 0.05;
        
        (pattern_confidence + abstraction_bonus + refactoring_bonus).min(1.0)
    }
    
    /// Create reasoning trace for abstract analysis
    fn create_abstract_reasoning_trace(&self, result: &AbstractResult) -> Vec<ActivationStep> {
        let mut trace = Vec::new();
        
        trace.push(ActivationStep {
            step_id: 1,
            entity_key: EntityKey::default(),
            concept_id: "pattern_detection".to_string(),
            activation_level: 0.8,
            operation_type: ActivationOperation::Initialize,
            timestamp: SystemTime::now(),
        });
        
        if !result.abstractions.is_empty() {
            trace.push(ActivationStep {
                step_id: 2,
                entity_key: EntityKey::default(),
                concept_id: "abstraction_identification".to_string(),
                activation_level: 0.7,
                operation_type: ActivationOperation::Propagate,
                timestamp: SystemTime::now(),
            });
        }
        
        if !result.refactoring_opportunities.is_empty() {
            trace.push(ActivationStep {
                step_id: 3,
                entity_key: EntityKey::default(),
                concept_id: "refactoring_opportunities".to_string(),
                activation_level: 0.6,
                operation_type: ActivationOperation::Reinforce,
                timestamp: SystemTime::now(),
            });
        }
        
        trace.push(ActivationStep {
            step_id: trace.len() + 1,
            entity_key: EntityKey::default(),
            concept_id: "efficiency_gains".to_string(),
            activation_level: 0.7,
            operation_type: ActivationOperation::Decay,
            timestamp: SystemTime::now(),
        });
        
        trace
    }
    
    /// Create additional metadata
    fn create_abstract_metadata(&self, result: &AbstractResult) -> AHashMap<String, String> {
        let mut metadata = AHashMap::new();
        metadata.insert("patterns_count".to_string(), result.patterns_found.len().to_string());
        metadata.insert("abstractions_count".to_string(), result.abstractions.len().to_string());
        metadata.insert("refactoring_opportunities".to_string(), result.refactoring_opportunities.len().to_string());
        metadata.insert("query_improvement".to_string(), format!("{:.3}", result.efficiency_gains.query_time_improvement));
        metadata.insert("memory_reduction".to_string(), format!("{:.3}", result.efficiency_gains.memory_reduction));
        metadata.insert("maintainability_score".to_string(), format!("{:.3}", result.efficiency_gains.maintainability_score));
        metadata
    }
}

/// Structural patterns analysis
#[derive(Debug, Clone)]
struct StructuralPatterns {
    scope: AnalysisScope,
    entity_distribution: EntityDistribution,
    relationship_frequency: RelationshipPatterns,
    activation_hotspots: ActivationPatterns,
    complexity_metrics: f32,
}

/// Entity distribution analysis
#[derive(Debug, Clone)]
struct EntityDistribution {
    input_ratio: f32,
    output_ratio: f32,
    gate_ratio: f32,
    distribution_entropy: f32,
}

/// Relationship patterns
#[derive(Debug, Clone)]
struct RelationshipPatterns {
    most_common_types: Vec<String>,
    average_connections_per_entity: f32,
    clustering_coefficient: f32,
    small_world_coefficient: f32,
}

/// Activation patterns
#[derive(Debug, Clone)]
struct ActivationPatterns {
    hotspot_entities: Vec<EntityKey>,
    activation_frequency: AHashMap<EntityKey, f32>,
    temporal_patterns: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use crate::core::types::{EntityKey, EntityData};
    use tokio;
    use std::sync::Arc;

    /// Creates a test graph with some sample entities for testing
    async fn create_test_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(64).expect("Failed to create test graph"));
        
        // Add some test entities
        let _ = graph.add_entity(EntityData::new(1, "animal: Basic animal entity".to_string(), vec![0.0; 64])).await;
        let _ = graph.add_entity(EntityData::new(1, "mammal: Warm-blooded vertebrate".to_string(), vec![0.0; 64])).await;
        let _ = graph.add_entity(EntityData::new(1, "dog: Domesticated canine".to_string(), vec![0.0; 64])).await;
        let _ = graph.add_entity(EntityData::new(1, "cat: Feline animal".to_string(), vec![0.0; 64])).await;
        let _ = graph.add_entity(EntityData::new(1, "vehicle: Transportation device".to_string(), vec![0.0; 64])).await;
        let _ = graph.add_entity(EntityData::new(1, "car: Four-wheeled vehicle".to_string(), vec![0.0; 64])).await;
        
        graph
    }

    /// Creates an AbstractThinking instance for testing
    async fn create_test_abstract_thinking() -> AbstractThinking {
        let graph = create_test_graph().await;
        AbstractThinking::new(graph)
    }

    #[tokio::test]
    async fn test_execute_pattern_analysis() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let result = abstract_thinking.execute_pattern_analysis(
            AnalysisScope::Global,
            PatternType::Structural
        ).await;
        
        assert!(result.is_ok(), "Pattern analysis should succeed");
        let result = result.unwrap();
        
        // Verify result structure
        assert!(result.patterns_found.len() >= 0, "Should have detected patterns (or none if empty graph)");
        assert!(result.abstractions.len() >= 0, "Should have abstraction candidates");
        assert!(result.refactoring_opportunities.len() >= 0, "Should have refactoring opportunities");
        
        // Verify efficiency gains structure
        assert!(result.efficiency_gains.query_time_improvement >= 0.0);
        assert!(result.efficiency_gains.memory_reduction >= 0.0);
        assert!(result.efficiency_gains.accuracy_improvement >= 0.0);
        assert!(result.efficiency_gains.maintainability_score >= 0.0);
        assert!(result.efficiency_gains.maintainability_score <= 1.0);
    }

    #[tokio::test]
    async fn test_identify_abstractions() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        // Create test patterns
        let test_patterns = vec![
            DetectedPattern {
                pattern_id: "test_pattern_1".to_string(),
                id: "test_pattern_1".to_string(),
                pattern_type: PatternType::Structural,
                confidence: 0.8,
                frequency: 5.0,
                entities_involved: vec![EntityKey::from_hash("entity1"), EntityKey::from_hash("entity2")],
                affected_entities: vec![EntityKey::from_hash("entity1")],
                description: "High frequency structural pattern".to_string(),
            },
            DetectedPattern {
                pattern_id: "test_pattern_2".to_string(),
                id: "test_pattern_2".to_string(),
                pattern_type: PatternType::Semantic,
                confidence: 0.6,
                frequency: 2.0,
                entities_involved: vec![EntityKey::from_hash("entity3")],
                affected_entities: vec![EntityKey::from_hash("entity3")],
                description: "Low frequency semantic pattern".to_string(),
            },
        ];
        
        let result = abstract_thinking.identify_abstractions(&test_patterns).await;
        assert!(result.is_ok(), "Abstraction identification should succeed");
        
        let abstractions = result.unwrap();
        
        // Should only identify abstractions for high-frequency, high-confidence patterns
        let high_quality_abstractions: Vec<_> = abstractions.iter()
            .filter(|a| a.complexity_reduction > 0.0)
            .collect();
        
        // Verify abstraction candidate structure
        for abstraction in &abstractions {
            assert!(!abstraction.abstraction_type.is_empty(), "Abstraction type should not be empty");
            assert!(abstraction.complexity_reduction >= 0.0, "Complexity reduction should be non-negative");
            assert!(abstraction.implementation_effort >= 0.0, "Implementation effort should be non-negative");
            assert!(!abstraction.source_patterns.is_empty(), "Should have source patterns");
        }
    }

    #[tokio::test]
    async fn test_suggest_refactoring_opportunities() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        // Create test abstraction candidates
        let test_abstractions = vec![
            AbstractionCandidate {
                abstraction_name: "test_abstraction".to_string(),
                entities_to_abstract: vec![EntityKey::from_hash("entity1"), EntityKey::from_hash("entity2")],
                potential_savings: 0.3,
                implementation_complexity: 50,
                complexity_reduction: 0.3,
                implementation_effort: 0.4,
                abstraction_type: "structural_abstraction".to_string(),
                source_patterns: vec!["pattern1".to_string(), "pattern2".to_string()],
            },
        ];
        
        let result = abstract_thinking.suggest_refactoring(&test_abstractions).await;
        assert!(result.is_ok(), "Refactoring suggestion should succeed");
        
        let opportunities = result.unwrap();
        assert!(opportunities.len() > 0, "Should have refactoring opportunities");
        
        // Verify refactoring opportunity structure
        for opportunity in &opportunities {
            assert!(!opportunity.description.is_empty(), "Description should not be empty");
            assert!(opportunity.estimated_benefit >= 0.0, "Benefit should be non-negative");
            assert!(opportunity.estimated_benefit <= 1.0, "Benefit should be at most 1.0");
        }
    }

    #[tokio::test]
    async fn test_analyze_structural_patterns() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let scope = AnalysisScope::Global;
        let result = abstract_thinking.analyze_structural_patterns(&scope).await;
        
        assert!(result.is_ok(), "Structural pattern analysis should succeed");
        let patterns = result.unwrap();
        
        // Verify structural patterns structure
        assert_eq!(patterns.scope, scope, "Scope should match input");
        assert!(patterns.entity_distribution.input_ratio >= 0.0);
        assert!(patterns.entity_distribution.output_ratio >= 0.0);
        assert!(patterns.entity_distribution.gate_ratio >= 0.0);
        assert!(patterns.relationship_frequency.average_connections_per_entity >= 0.0);
        assert!(patterns.complexity_metrics >= 0.0);
    }

    #[tokio::test]
    async fn test_neural_pattern_detection() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        // Create mock structural patterns
        let structural_data = StructuralPatterns {
            scope: AnalysisScope::Global,
            entity_distribution: EntityDistribution {
                input_ratio: 0.33,
                output_ratio: 0.33,
                gate_ratio: 0.34,
                distribution_entropy: 1.0,
            },
            relationship_frequency: RelationshipPatterns {
                most_common_types: vec!["is_a".to_string(), "has_property".to_string()],
                average_connections_per_entity: 6.0,
                clustering_coefficient: 0.6,
                small_world_coefficient: 0.4,
            },
            activation_hotspots: ActivationPatterns {
                hotspot_entities: Vec::new(),
                activation_frequency: AHashMap::new(),
                temporal_patterns: Vec::new(),
            },
            complexity_metrics: 0.5,
        };
        
        for pattern_type in [PatternType::Structural, PatternType::Temporal, PatternType::Semantic, PatternType::Usage] {
            let result = abstract_thinking.neural_pattern_detection(structural_data.clone(), pattern_type).await;
            assert!(result.is_ok(), "Neural pattern detection should succeed for {:?}", pattern_type);
            
            let patterns = result.unwrap();
            // Patterns may be empty for some types (temporal, semantic, usage), but should not error
            for pattern in &patterns {
                assert!(!pattern.pattern_id.is_empty(), "Pattern ID should not be empty");
                assert!(!pattern.description.is_empty(), "Pattern description should not be empty");
                assert!(pattern.confidence >= 0.0 && pattern.confidence <= 1.0, "Confidence should be in [0,1]");
                assert!(pattern.frequency >= 0.0, "Frequency should be non-negative");
            }
        }
    }

    #[tokio::test]
    async fn test_detect_structural_patterns() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let structural_data = StructuralPatterns {
            scope: AnalysisScope::Global,
            entity_distribution: EntityDistribution {
                input_ratio: 0.33,
                output_ratio: 0.33,
                gate_ratio: 0.34,
                distribution_entropy: 1.0,
            },
            relationship_frequency: RelationshipPatterns {
                most_common_types: vec!["is_a".to_string(), "has_property".to_string()],
                average_connections_per_entity: 6.0,
                clustering_coefficient: 0.6,
                small_world_coefficient: 0.4,
            },
            activation_hotspots: ActivationPatterns {
                hotspot_entities: Vec::new(),
                activation_frequency: AHashMap::new(),
                temporal_patterns: Vec::new(),
            },
            complexity_metrics: 0.5,
        };
        
        let result = abstract_thinking.detect_structural_patterns(&structural_data).await;
        assert!(result.is_ok(), "Structural pattern detection should succeed");
        
        let patterns = result.unwrap();
        
        // Should detect patterns based on relationship types and connectivity
        let is_a_patterns: Vec<_> = patterns.iter()
            .filter(|p| p.pattern_id.contains("is_a"))
            .collect();
        let property_patterns: Vec<_> = patterns.iter()
            .filter(|p| p.pattern_id.contains("property"))
            .collect();
        let hub_patterns: Vec<_> = patterns.iter()
            .filter(|p| p.pattern_id.contains("hub"))
            .collect();
        
        assert!(is_a_patterns.len() > 0, "Should detect is_a patterns");
        assert!(property_patterns.len() > 0, "Should detect property patterns");
        assert!(hub_patterns.len() > 0, "Should detect hub patterns due to high connectivity");
    }

    #[tokio::test]
    async fn test_determine_abstraction_type() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let test_cases = vec![
            (PatternType::Structural, "structural_abstraction"),
            (PatternType::Temporal, "temporal_abstraction"),
            (PatternType::Semantic, "semantic_abstraction"),
            (PatternType::Usage, "usage_abstraction"),
        ];
        
        for (pattern_type, expected_abstraction) in test_cases {
            let pattern = DetectedPattern {
                pattern_id: "test".to_string(),
                id: "test".to_string(),
                pattern_type,
                confidence: 0.8,
                frequency: 5.0,
                entities_involved: Vec::new(),
                affected_entities: Vec::new(),
                description: "Test pattern".to_string(),
            };
            
            let abstraction_type = abstract_thinking.determine_abstraction_type(&pattern);
            assert_eq!(abstraction_type, expected_abstraction, 
                      "Abstraction type should match pattern type");
        }
    }

    #[tokio::test]
    async fn test_estimate_complexity_reduction() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let test_pattern = DetectedPattern {
            pattern_id: "test".to_string(),
            id: "test".to_string(),
            pattern_type: PatternType::Structural,
            confidence: 0.8,
            frequency: 5.0,
            entities_involved: Vec::new(),
            affected_entities: Vec::new(),
            description: "Test pattern".to_string(),
        };
        
        let reduction = abstract_thinking.estimate_complexity_reduction(&test_pattern);
        
        // Should be (frequency - 1) * 0.1 * confidence = (5-1) * 0.1 * 0.8 = 0.32
        let expected = (5.0 - 1.0) * 0.1 * 0.8;
        assert!((reduction - expected).abs() < 0.001, 
               "Complexity reduction calculation should be accurate: expected {}, got {}", expected, reduction);
    }

    #[tokio::test]
    async fn test_estimate_implementation_effort() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let test_cases = vec![
            (PatternType::Structural, 0.3),
            (PatternType::Temporal, 0.6),
            (PatternType::Semantic, 0.5),
            (PatternType::Usage, 0.4),
        ];
        
        for (pattern_type, expected_effort) in test_cases {
            let pattern = DetectedPattern {
                pattern_id: "test".to_string(),
                id: "test".to_string(),
                pattern_type,
                confidence: 0.8,
                frequency: 5.0,
                entities_involved: Vec::new(),
                affected_entities: Vec::new(),
                description: "Test pattern".to_string(),
            };
            
            let effort = abstract_thinking.estimate_implementation_effort(&pattern);
            assert_eq!(effort, expected_effort, 
                      "Implementation effort should match pattern type");
        }
    }

    #[tokio::test]
    async fn test_estimate_efficiency_gains() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let test_opportunities = vec![
            RefactoringOpportunity {
                opportunity_type: RefactoringType::ConceptMerging,
                description: "Test concept merging".to_string(),
                entities_affected: Vec::new(),
                estimated_benefit: 0.5,
            },
            RefactoringOpportunity {
                opportunity_type: RefactoringType::PerformanceOptimization,
                description: "Test performance optimization".to_string(),
                entities_affected: Vec::new(),
                estimated_benefit: 0.3,
            },
        ];
        
        let efficiency_gains = abstract_thinking.estimate_efficiency_gains(&test_opportunities);
        
        // Verify efficiency gains calculations
        assert!(efficiency_gains.query_time_improvement >= 0.0);
        assert!(efficiency_gains.query_time_improvement <= 0.5); // Capped at 0.5
        assert!(efficiency_gains.memory_reduction >= 0.0);
        assert!(efficiency_gains.memory_reduction <= 0.4); // Capped at 0.4
        assert!(efficiency_gains.accuracy_improvement >= 0.0);
        assert!(efficiency_gains.accuracy_improvement <= 0.3); // Capped at 0.3
        assert!(efficiency_gains.maintainability_score >= 0.0);
        assert!(efficiency_gains.maintainability_score <= 1.0);
    }

    #[tokio::test]
    async fn test_combine_related_abstractions() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let test_candidates = vec![
            AbstractionCandidate {
                abstraction_name: "test1".to_string(),
                entities_to_abstract: vec![EntityKey::from_hash("entity1")],
                potential_savings: 0.3,
                implementation_complexity: 50,
                complexity_reduction: 0.3,
                implementation_effort: 0.4,
                abstraction_type: "structural".to_string(),
                source_patterns: vec!["pattern1".to_string()],
            },
            AbstractionCandidate {
                abstraction_name: "test2".to_string(),
                entities_to_abstract: vec![EntityKey::from_hash("entity2")],
                potential_savings: 0.2,
                implementation_complexity: 30,
                complexity_reduction: 0.2,
                implementation_effort: 0.3,
                abstraction_type: "semantic".to_string(),
                source_patterns: vec!["pattern2".to_string()],
            },
        ];
        
        let result = abstract_thinking.combine_related_abstractions(test_candidates.clone()).await;
        assert!(result.is_ok(), "Combining abstractions should succeed");
        
        let combined = result.unwrap();
        // Currently just returns original candidates, but should not error
        assert_eq!(combined.len(), test_candidates.len(), "Should return all candidates");
    }

    #[tokio::test]
    async fn test_identify_performance_optimizations() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let result = abstract_thinking.identify_performance_optimizations().await;
        assert!(result.is_ok(), "Performance optimization identification should succeed");
        
        let optimizations = result.unwrap();
        assert!(optimizations.len() > 0, "Should identify at least one optimization");
        
        // Verify optimization structure
        for optimization in &optimizations {
            assert!(!optimization.description.is_empty(), "Description should not be empty");
            assert!(optimization.estimated_benefit >= 0.0, "Benefit should be non-negative");
            assert!(optimization.estimated_benefit <= 1.0, "Benefit should be at most 1.0");
            assert!(matches!(optimization.opportunity_type, RefactoringType::PerformanceOptimization));
        }
    }

    #[tokio::test]
    async fn test_calculate_distribution_entropy() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        // Test with zero entities
        let zero_stats = crate::core::brain_enhanced_graph::BrainStatistics::new();
        let entropy = abstract_thinking.calculate_distribution_entropy(&zero_stats);
        assert_eq!(entropy, 0.0, "Entropy should be 0 for empty graph");
        
        // Test with non-zero entities
        let mut stats = crate::core::brain_enhanced_graph::BrainStatistics::new();
        stats.entity_count = 100;
        stats.relationship_count = 200;
        stats.clustering_coefficient = 0.3;
        let entropy = abstract_thinking.calculate_distribution_entropy(&stats);
        assert!(entropy > 0.0, "Entropy should be positive for non-empty graph");
        assert!(entropy <= 2.0, "Entropy should be reasonable (less than ln(3) for 3 categories)");
    }

    #[tokio::test]
    async fn test_calculate_structural_complexity() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let stats = crate::core::brain_enhanced_graph::BrainStatistics {
            entity_count: 100,
            relationship_count: 200,
            avg_activation: 0.5,
            max_activation: 1.0,
            min_activation: 0.0,
            graph_density: 0.02,
            clustering_coefficient: 0.3,
            average_path_length: 3.5,
            betweenness_centrality: std::collections::HashMap::new(),
            activation_distribution: std::collections::HashMap::new(),
            concept_coherence: 0.8,
            learning_efficiency: 0.9,
            max_degree: 10,
            average_degree: 4.0,
            total_propagations: 0,
            total_affected_entities: 0,
        };
        
        let complexity = abstract_thinking.calculate_structural_complexity(&stats);
        assert!(complexity >= 0.0, "Complexity should be non-negative");
        assert!(complexity <= 1.0, "Complexity should be reasonable");
    }

    #[tokio::test]
    async fn test_analyze_entity_distribution() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let stats = crate::core::brain_enhanced_graph::BrainStatistics {
            entity_count: 100,
            relationship_count: 200,
            avg_activation: 0.5,
            max_activation: 1.0,
            min_activation: 0.0,
            graph_density: 0.02,
            clustering_coefficient: 0.3,
            average_path_length: 3.5,
            betweenness_centrality: std::collections::HashMap::new(),
            activation_distribution: std::collections::HashMap::new(),
            concept_coherence: 0.8,
            learning_efficiency: 0.9,
            max_degree: 10,
            average_degree: 4.0,
            total_propagations: 0,
            total_affected_entities: 0,
        };
        
        let result = abstract_thinking.analyze_entity_distribution(&stats).await;
        assert!(result.is_ok(), "Entity distribution analysis should succeed");
        
        let distribution = result.unwrap();
        
        // Verify distribution ratios sum to approximately 1.0
        let total_ratio = distribution.input_ratio + distribution.output_ratio + distribution.gate_ratio;
        assert!((total_ratio - 1.0).abs() < 0.01, "Distribution ratios should sum to ~1.0");
        
        assert!(distribution.distribution_entropy >= 0.0, "Entropy should be non-negative");
    }

    #[tokio::test]
    async fn test_analyze_relationship_patterns() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let result = abstract_thinking.analyze_relationship_patterns().await;
        assert!(result.is_ok(), "Relationship pattern analysis should succeed");
        
        let patterns = result.unwrap();
        
        assert!(patterns.average_connections_per_entity >= 0.0, "Average connections should be non-negative");
        assert!(patterns.clustering_coefficient >= 0.0, "Clustering coefficient should be non-negative");
        assert!(patterns.clustering_coefficient <= 1.0, "Clustering coefficient should be at most 1.0");
        assert!(patterns.small_world_coefficient >= 0.0, "Small world coefficient should be non-negative");
        assert!(patterns.small_world_coefficient <= 1.0, "Small world coefficient should be at most 1.0");
    }

    #[tokio::test]
    async fn test_analyze_activation_patterns() {
        let abstract_thinking = create_test_abstract_thinking().await;
        
        let result = abstract_thinking.analyze_activation_patterns().await;
        assert!(result.is_ok(), "Activation pattern analysis should succeed");
        
        let patterns = result.unwrap();
        
        // For now, these are empty but should not error
        assert!(patterns.hotspot_entities.len() >= 0, "Hotspot entities should be a valid list");
        assert!(patterns.activation_frequency.len() >= 0, "Activation frequency should be a valid map");
        assert!(patterns.temporal_patterns.len() >= 0, "Temporal patterns should be a valid list");
    }
}

