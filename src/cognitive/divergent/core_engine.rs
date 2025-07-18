//! Core divergent thinking engine

use super::constants::*;
use crate::cognitive::types::{
    CognitivePattern, CognitivePatternType, ExplorationPath, ExplorationMap,
    DivergentResult, PatternComplexity, PatternUseCase, ExplorationContext,
    ExplorationResult, ExplorationNode, ExplorationEdge, ExplorationParameters,
    ExplorationAnalysis, ExplorationQuality, ExplorationFeedback,
    ExplorationResultMap, ExplorationTracker, ExplorationHistory,
    ExplorationMetrics, ExplorationState, ExplorationStatistics,
    ExplorationDirection, ExplorationNodeType, ExplorationEdgeType,
    ExplorationBreadth, ExplorationDepth, ExplorationStrategy,
    ExplorationValue, ExplorationWeight, ExplorationScore,
    ExplorationSummary, ExplorationTrace, ExplorationInsight,
    ExplorationCreativity, ExplorationNovelty, ExplorationRelevance,
    ExplorationReasoningTrace, ExplorationOutcome, ExplorationConclusion,
    ExplorationRecommendation, ExplorationNextStep, ExplorationAction,
    ExplorationOption, ExplorationAlternative, ExplorationVariation,
    ExplorationExtension, ExplorationRefinement, ExplorationOptimization,
    ExplorationValidation, ExplorationVerification, ExplorationComparison,
    ExplorationEvaluation, ExplorationAssessment, ExplorationJudgment,
    ExplorationCritique, ExplorationReview, ExplorationRating,
    ExplorationRanking, ExplorationSelection, ExplorationDecision,
    ExplorationChoice, ExplorationPreference, ExplorationPriority,
    ExplorationImportance, ExplorationUrgency, ExplorationImpact,
    ExplorationEffectiveness, ExplorationEfficiency, ExplorationFeasibility,
    ExplorationViability, ExplorationSuitability, ExplorationApplicability,
    ExplorationReliability, ExplorationConsistency, ExplorationStability,
    ExplorationRobustness, ExplorationFlexibility, ExplorationAdaptability,
    ExplorationScalability, ExplorationSustainability, ExplorationMaintainability,
    ExplorationUsability, ExplorationAccessibility, ExplorationCompatibility,
    ExplorationInteroperability, ExplorationIntegration, ExplorationSynergy,
    ExplorationCollaboration, ExplorationCoordination, ExplorationAlignment,
    ExplorationHarmony, ExplorationBalance, ExplorationOptimum,
    ExplorationEquilibrium, ExplorationStability as ExplorationStabilityType,
    ExplorationDynamics, ExplorationEvolution, ExplorationAdaptation,
    ExplorationLearning, ExplorationDevelopment, ExplorationGrowth,
    ExplorationProgress, ExplorationAdvancement, ExplorationImprovement,
    ExplorationEnhancement, ExplorationUpgrade, ExplorationTransformation,
    ExplorationInnovation, ExplorationCreativity as ExplorationCreativityType,
    ExplorationOriginality, ExplorationUniqueness, ExplorationDistinctiveness,
    ExplorationSpecialization, ExplorationGeneralization, ExplorationAbstraction,
    ExplorationConcretization, ExplorationSpecification, ExplorationImplementation,
    ExplorationExecution, ExplorationDeployment, ExplorationOperation,
    ExplorationMaintenance, ExplorationSupport, ExplorationMonitoring,
    ExplorationObservation, ExplorationAnalysis as ExplorationAnalysisType,
    ExplorationDiagnosis, ExplorationTroubleshooting, ExplorationDebugging,
    ExplorationTesting, ExplorationValidation as ExplorationValidationType,
    ExplorationVerification as ExplorationVerificationType,
    ExplorationQualityAssurance, ExplorationQualityControl
};
use crate::core::types::EntityKey;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Divergent thinking processor with configurable parameters
pub struct DivergentThinking {
    pub exploration_breadth: usize,
    pub creativity_threshold: f32,
    pub max_path_length: usize,
    pub novelty_weight: f32,
    pub relevance_weight: f32,
    pub activation_decay: f32,
    pub min_activation: f32,
    pub max_results: usize,
}

impl DivergentThinking {
    /// Create new divergent thinking processor with default parameters
    pub fn new() -> Self {
        Self {
            exploration_breadth: DEFAULT_EXPLORATION_BREADTH,
            creativity_threshold: DEFAULT_CREATIVITY_THRESHOLD,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
            novelty_weight: DEFAULT_NOVELTY_WEIGHT,
            relevance_weight: DEFAULT_RELEVANCE_WEIGHT,
            activation_decay: DEFAULT_ACTIVATION_DECAY,
            min_activation: DEFAULT_MIN_ACTIVATION,
            max_results: DEFAULT_MAX_RESULTS,
        }
    }

    /// Create new divergent thinking processor with custom parameters
    pub fn new_with_params(
        exploration_breadth: usize,
        creativity_threshold: f32,
        max_path_length: usize,
        novelty_weight: f32,
        relevance_weight: f32,
    ) -> Self {
        Self {
            exploration_breadth,
            creativity_threshold,
            max_path_length,
            novelty_weight,
            relevance_weight,
            activation_decay: DEFAULT_ACTIVATION_DECAY,
            min_activation: DEFAULT_MIN_ACTIVATION,
            max_results: DEFAULT_MAX_RESULTS,
        }
    }

    /// Execute divergent exploration
    pub async fn execute_divergent_exploration(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        context: &ExplorationContext,
    ) -> Result<ExplorationMap> {
        let mut exploration_map = ExplorationMap::new();
        
        // Extract query components
        let query = &context.query;
        let exploration_type = self.infer_exploration_type(query);
        let seed_concept = self.extract_seed_concept(query);
        
        // Initialize exploration state
        let mut exploration_state = ExplorationState {
            current_depth: 0,
            visited_entities: HashSet::new(),
            activation_levels: HashMap::new(),
            path_scores: HashMap::new(),
            exploration_direction: ExplorationDirection::Bidirectional,
            exploration_strategy: ExplorationStrategy::BreadthFirst,
            exploration_breadth: ExplorationBreadth::Wide,
            exploration_depth: ExplorationDepth::Medium,
        };
        
        // Activate seed concept
        let activated_entities = self.activate_seed_concept(graph, &seed_concept, &mut exploration_state).await?;
        
        // Spread activation through the graph
        let activated_paths = self.spread_activation(graph, &activated_entities, &mut exploration_state).await?;
        
        // Perform neural path exploration
        let enhanced_paths = self.neural_path_exploration(graph, &activated_paths, &exploration_type, &mut exploration_state).await?;
        
        // Find typed connections
        let typed_connections = self.find_typed_connections(&enhanced_paths, &exploration_type);
        
        // Build exploration map
        exploration_map.paths = enhanced_paths;
        exploration_map.connections = typed_connections;
        exploration_map.exploration_type = exploration_type;
        exploration_map.seed_concept = seed_concept;
        exploration_map.exploration_statistics = self.calculate_exploration_statistics(&exploration_map, &exploration_state);
        
        Ok(exploration_map)
    }

    /// Activate seed concept and find initial entities
    async fn activate_seed_concept(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        seed_concept: &str,
        exploration_state: &mut ExplorationState,
    ) -> Result<Vec<EntityKey>> {
        let mut activated_entities = Vec::new();
        
        // Find entities matching the seed concept
        let matching_entities = self.find_concept_entities(graph, seed_concept).await?;
        
        // Activate matching entities
        for entity_key in matching_entities {
            let initial_activation = 1.0; // Full activation for seed concepts
            exploration_state.activation_levels.insert(entity_key, initial_activation);
            exploration_state.visited_entities.insert(entity_key);
            activated_entities.push(entity_key);
        }
        
        Ok(activated_entities)
    }

    /// Spread activation through the graph
    async fn spread_activation(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        initial_entities: &[EntityKey],
        exploration_state: &mut ExplorationState,
    ) -> Result<Vec<ExplorationPath>> {
        let mut paths = Vec::new();
        let mut current_entities = initial_entities.to_vec();
        
        for depth in 0..self.max_path_length {
            if current_entities.is_empty() {
                break;
            }
            
            let mut next_entities = Vec::new();
            
            for &entity_key in &current_entities {
                let current_activation = exploration_state.activation_levels.get(&entity_key).copied().unwrap_or(0.0);
                
                if current_activation < self.min_activation {
                    continue;
                }
                
                // Get neighbors
                let neighbors = graph.get_neighbors(entity_key).await;
                
                for (neighbor_key, connection_weight) in neighbors {
                    if exploration_state.visited_entities.contains(&neighbor_key) {
                        continue;
                    }
                    
                    // Calculate activation for neighbor
                    let neighbor_activation = current_activation * connection_weight * self.activation_decay;
                    
                    if neighbor_activation >= self.min_activation {
                        exploration_state.activation_levels.insert(neighbor_key, neighbor_activation);
                        exploration_state.visited_entities.insert(neighbor_key);
                        next_entities.push(neighbor_key);
                        
                        // Create exploration path
                        let path = ExplorationPath {
                            entities: vec![entity_key, neighbor_key],
                            weights: vec![connection_weight],
                            total_weight: connection_weight,
                            depth: depth + 1,
                            activation_level: neighbor_activation,
                            creativity_score: 0.0, // Will be calculated later
                            novelty_score: 0.0,    // Will be calculated later
                            relevance_score: 0.0,  // Will be calculated later
                            path_type: ExplorationNodeType::Intermediate,
                            exploration_value: ExplorationValue::Medium,
                        };
                        
                        paths.push(path);
                    }
                }
            }
            
            current_entities = next_entities;
            exploration_state.current_depth = depth + 1;
            
            // Limit breadth
            if current_entities.len() > self.exploration_breadth {
                current_entities.truncate(self.exploration_breadth);
            }
        }
        
        Ok(paths)
    }

    /// Perform neural path exploration with enhancement
    async fn neural_path_exploration(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        paths: &[ExplorationPath],
        exploration_type: &str,
        exploration_state: &mut ExplorationState,
    ) -> Result<Vec<ExplorationPath>> {
        let mut enhanced_paths = Vec::new();
        
        for path in paths {
            // Calculate creativity score
            let creativity_score = self.calculate_path_creativity(graph, path, exploration_type).await?;
            
            // Calculate novelty score
            let novelty_score = self.calculate_path_novelty(graph, path).await?;
            
            // Calculate relevance score
            let relevance_score = self.calculate_path_relevance(graph, path, exploration_type).await?;
            
            // Create enhanced path
            let mut enhanced_path = path.clone();
            enhanced_path.creativity_score = creativity_score;
            enhanced_path.novelty_score = novelty_score;
            enhanced_path.relevance_score = relevance_score;
            
            // Calculate combined score
            let combined_score = (novelty_score * self.novelty_weight) + 
                               (relevance_score * self.relevance_weight);
            
            // Store path score
            exploration_state.path_scores.insert(enhanced_path.entities.clone(), combined_score);
            
            // Only keep paths above creativity threshold
            if creativity_score >= self.creativity_threshold {
                enhanced_paths.push(enhanced_path);
            }
        }
        
        // Sort by combined score
        enhanced_paths.sort_by(|a, b| {
            let score_a = (a.novelty_score * self.novelty_weight) + (a.relevance_score * self.relevance_weight);
            let score_b = (b.novelty_score * self.novelty_weight) + (b.relevance_score * self.relevance_weight);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Limit results
        if enhanced_paths.len() > self.max_results {
            enhanced_paths.truncate(self.max_results);
        }
        
        Ok(enhanced_paths)
    }

    /// Find typed connections based on exploration type
    fn find_typed_connections(&self, paths: &[ExplorationPath], exploration_type: &str) -> Vec<ExplorationEdge> {
        let mut connections = Vec::new();
        
        for path in paths {
            for i in 0..path.entities.len() - 1 {
                let source = path.entities[i];
                let target = path.entities[i + 1];
                let weight = if i < path.weights.len() { path.weights[i] } else { 1.0 };
                
                let edge_type = match exploration_type {
                    "creative" => ExplorationEdgeType::Creative,
                    "analytical" => ExplorationEdgeType::Analytical,
                    "associative" => ExplorationEdgeType::Associative,
                    _ => ExplorationEdgeType::Exploratory,
                };
                
                let connection = ExplorationEdge {
                    source,
                    target,
                    weight,
                    edge_type,
                    exploration_value: ExplorationValue::Medium,
                };
                
                connections.push(connection);
            }
        }
        
        connections
    }

    /// Calculate exploration statistics
    fn calculate_exploration_statistics(
        &self,
        exploration_map: &ExplorationMap,
        exploration_state: &ExplorationState,
    ) -> ExplorationStatistics {
        let total_paths = exploration_map.paths.len();
        let total_entities = exploration_state.visited_entities.len();
        let max_depth = exploration_state.current_depth;
        
        let avg_creativity = if total_paths > 0 {
            exploration_map.paths.iter().map(|p| p.creativity_score).sum::<f32>() / total_paths as f32
        } else {
            0.0
        };
        
        let avg_novelty = if total_paths > 0 {
            exploration_map.paths.iter().map(|p| p.novelty_score).sum::<f32>() / total_paths as f32
        } else {
            0.0
        };
        
        let avg_relevance = if total_paths > 0 {
            exploration_map.paths.iter().map(|p| p.relevance_score).sum::<f32>() / total_paths as f32
        } else {
            0.0
        };
        
        ExplorationStatistics {
            total_paths,
            total_entities,
            max_depth,
            avg_creativity,
            avg_novelty,
            avg_relevance,
            exploration_coverage: total_entities as f32 / self.exploration_breadth as f32,
            exploration_efficiency: total_paths as f32 / total_entities as f32,
        }
    }

    /// Find entities matching a concept
    async fn find_concept_entities(&self, graph: &BrainEnhancedKnowledgeGraph, concept: &str) -> Result<Vec<EntityKey>> {
        // Generate embedding for concept
        let concept_embedding = self.generate_concept_embedding(concept);
        
        // Search for similar entities
        let similar_entities = graph.core_graph.similarity_search(&concept_embedding, 10)?;
        
        // Extract entity keys
        let entity_keys: Vec<EntityKey> = similar_entities.into_iter().map(|(key, _)| key).collect();
        
        Ok(entity_keys)
    }

    /// Generate embedding for concept (simplified)
    fn generate_concept_embedding(&self, concept: &str) -> Vec<f32> {
        // This is a simplified embedding generation
        // In practice, you'd use a proper embedding model
        let mut embedding = vec![0.0; 96]; // Standard embedding dimension
        
        // Hash-based embedding generation
        let hash = self.simple_hash(concept);
        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((hash as u64).wrapping_mul(i as u64 + 1) % 1000) as f32 / 1000.0;
        }
        
        // Normalize
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in embedding.iter_mut() {
                *val /= magnitude;
            }
        }
        
        embedding
    }

    /// Simple hash function for concept
    fn simple_hash(&self, s: &str) -> u32 {
        let mut hash = 0u32;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    /// Calculate path creativity
    async fn calculate_path_creativity(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        path: &ExplorationPath,
        exploration_type: &str,
    ) -> Result<f32> {
        let mut creativity_score = 0.0;
        
        // Base creativity from path length
        creativity_score += (path.depth as f32) * 0.1;
        
        // Creativity from activation level
        creativity_score += path.activation_level * 0.3;
        
        // Creativity from connection weights
        let avg_weight = if path.weights.is_empty() {
            0.0
        } else {
            path.weights.iter().sum::<f32>() / path.weights.len() as f32
        };
        creativity_score += avg_weight * 0.2;
        
        // Boost for creative exploration type
        if exploration_type == "creative" {
            creativity_score *= 1.2;
        }
        
        Ok(creativity_score.clamp(0.0, 1.0))
    }

    /// Calculate path novelty
    async fn calculate_path_novelty(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        path: &ExplorationPath,
    ) -> Result<f32> {
        let mut novelty_score = 0.0;
        
        // Novelty from path length (longer paths are more novel)
        novelty_score += (path.depth as f32 / self.max_path_length as f32) * 0.4;
        
        // Novelty from weak connections (unusual connections are more novel)
        let avg_weight = if path.weights.is_empty() {
            0.0
        } else {
            path.weights.iter().sum::<f32>() / path.weights.len() as f32
        };
        novelty_score += (1.0 - avg_weight) * 0.3;
        
        // Novelty from activation level (lower activation can be more novel)
        novelty_score += (1.0 - path.activation_level) * 0.3;
        
        Ok(novelty_score.clamp(0.0, 1.0))
    }

    /// Calculate path relevance
    async fn calculate_path_relevance(
        &self,
        graph: &BrainEnhancedKnowledgeGraph,
        path: &ExplorationPath,
        exploration_type: &str,
    ) -> Result<f32> {
        let mut relevance_score = 0.0;
        
        // Relevance from activation level
        relevance_score += path.activation_level * 0.4;
        
        // Relevance from connection strength
        let avg_weight = if path.weights.is_empty() {
            0.0
        } else {
            path.weights.iter().sum::<f32>() / path.weights.len() as f32
        };
        relevance_score += avg_weight * 0.3;
        
        // Relevance penalty for very long paths
        if path.depth > 3 {
            relevance_score *= 0.8;
        }
        
        // Boost for analytical exploration type
        if exploration_type == "analytical" {
            relevance_score *= 1.1;
        }
        
        Ok(relevance_score.clamp(0.0, 1.0))
    }

    /// Infer exploration type from query
    fn infer_exploration_type(&self, query: &str) -> String {
        let query_lower = query.to_lowercase();
        let keywords = get_exploration_type_keywords();
        
        let mut type_scores = HashMap::new();
        
        for (exploration_type, type_keywords) in keywords {
            let mut score = 0;
            for keyword in type_keywords {
                if query_lower.contains(&keyword) {
                    score += 1;
                }
            }
            type_scores.insert(exploration_type, score);
        }
        
        // Find the type with highest score
        let best_type = type_scores.iter()
            .max_by_key(|(_, &score)| score)
            .map(|(type_name, _)| type_name.clone())
            .unwrap_or_else(|| "exploratory".to_string());
        
        best_type
    }

    /// Extract seed concept from query
    fn extract_seed_concept(&self, query: &str) -> String {
        let stop_words = get_stop_words();
        let words: Vec<&str> = query.split_whitespace().collect();
        
        // Remove stop words and find meaningful concept
        let meaningful_words: Vec<&str> = words.into_iter()
            .filter(|&word| !stop_words.contains(&word.to_lowercase()))
            .collect();
        
        if meaningful_words.is_empty() {
            query.to_string()
        } else {
            meaningful_words[0].to_string()
        }
    }
}

impl Default for DivergentThinking {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitivePattern for DivergentThinking {
    fn execute(&self, context: &ExplorationContext) -> Result<ExplorationResult> {
        // This is a simplified synchronous wrapper
        // In practice, you'd use async runtime
        let result = ExplorationResult {
            insights: vec![],
            connections: vec![],
            patterns: vec![],
            confidence: 0.8,
            reasoning_trace: vec![],
            next_actions: vec![],
        };
        
        Ok(result)
    }

    fn get_pattern_type(&self) -> CognitivePatternType {
        CognitivePatternType::Divergent
    }

    fn get_optimal_use_cases(&self) -> Vec<PatternUseCase> {
        vec![
            PatternUseCase::CreativeIdeation,
            PatternUseCase::ConceptExploration,
            PatternUseCase::AlternativeGeneration,
            PatternUseCase::BrainstormingSupport,
            PatternUseCase::InnovationDiscovery,
        ]
    }

    fn estimate_complexity(&self, context: &ExplorationContext) -> PatternComplexity {
        let query_length = context.query.len();
        let word_count = context.query.split_whitespace().count();
        
        if query_length < 50 && word_count < 10 {
            PatternComplexity::Low
        } else if query_length < 200 && word_count < 30 {
            PatternComplexity::Medium
        } else {
            PatternComplexity::High
        }
    }
}