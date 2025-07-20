use std::sync::Arc;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, Instant};
use async_trait::async_trait;

use crate::cognitive::types::*;
use crate::cognitive::neural_bridge_finder::NeuralBridgeFinder;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::brain_types::{ActivationPattern, BrainInspiredEntity, EntityDirection, ActivationStep, ActivationOperation, RelationType};
use crate::core::types::EntityKey;
// Neural server dependency removed - using pure graph operations
use crate::error::{Result, GraphError};

/// Lateral thinking pattern - connects disparate concepts through unexpected paths
pub struct LateralThinking {
    pub graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub bridge_models: HashMap<String, String>,
    pub novelty_threshold: f32,
    pub max_bridge_length: usize,
    pub creativity_boost: f32,
    pub neural_bridge_finder: NeuralBridgeFinder,
}

impl LateralThinking {
    /// Create a new lateral thinking processor
    pub fn new(
        graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        let mut bridge_models = HashMap::new();
        bridge_models.insert("fedformer".to_string(), "fedformer_bridge_model".to_string());
        bridge_models.insert("stemgnn".to_string(), "stemgnn_creativity_model".to_string());
        
        let neural_bridge_finder = NeuralBridgeFinder::new(
            graph.clone(),
        );
        
        Self {
            graph,
            bridge_models,
            novelty_threshold: 0.4,
            max_bridge_length: 6,
            creativity_boost: 1.5,
            neural_bridge_finder,
        }
    }
    
    /// Find creative connections between two disparate concepts
    pub async fn find_creative_connections(
        &self,
        concept_a: &str,
        concept_b: &str,
        max_bridge_length: Option<usize>,
    ) -> Result<LateralResult> {
        let start_time = Instant::now();
        let max_length = max_bridge_length.unwrap_or(self.max_bridge_length);
        
        // 1. Activate both endpoint concepts
        let _activation_a = self.activate_concept(concept_a).await?;
        let _activation_b = self.activate_concept(concept_b).await?;
        
        // 2. Use enhanced neural bridge finding
        let bridge_candidates = self.neural_bridge_finder.find_creative_bridges_with_length(
            concept_a,
            concept_b,
            max_length,
        ).await?;
        
        // 3. The bridges are already scored by the neural bridge finder
        let scored_bridges = bridge_candidates;
        
        // 4. Analyze novelty patterns
        let novelty_analysis = self.analyze_novelty(&scored_bridges).await?;
        
        // 5. Calculate confidence distribution
        let confidence_distribution = self.calculate_confidence_distribution(&scored_bridges);
        
        let _execution_time = start_time.elapsed();
        
        Ok(LateralResult {
            bridges: scored_bridges,
            novelty_analysis,
            confidence_distribution,
        })
    }
    
    /// Parse query to extract two concepts for lateral connection
    pub async fn parse_lateral_query(&self, query: &str) -> Result<(String, String)> {
        // Look for connection patterns in query
        let query_lower = query.to_lowercase();
        
        if let Some(pos) = query_lower.find(" and ") {
            let concept_a = query[..pos].trim();
            let concept_b = query[pos + 5..].trim();
            return Ok((self.clean_concept(concept_a), self.clean_concept(concept_b)));
        }
        
        if let Some(pos) = query_lower.find(" to ") {
            let concept_a = query[..pos].trim();
            let concept_b = query[pos + 4..].trim();
            return Ok((self.clean_concept(concept_a), self.clean_concept(concept_b)));
        }
        
        if let Some(pos) = query_lower.find(" with ") {
            let concept_a = query[..pos].trim();
            let concept_b = query[pos + 6..].trim();
            return Ok((self.clean_concept(concept_a), self.clean_concept(concept_b)));
        }
        
        // If no clear pattern, extract first and last meaningful words
        let words: Vec<&str> = query.split_whitespace()
            .filter(|word| !self.is_stop_word(word) && word.len() > 2)
            .collect();
        
        if words.len() >= 2 {
            Ok((words[0].to_lowercase(), words[words.len() - 1].to_lowercase()))
        } else {
            Err(GraphError::ProcessingError("Cannot extract two concepts from query".to_string()))
        }
    }
    
    /// Activate a single concept for lateral thinking
    async fn activate_concept(&self, concept: &str) -> Result<ActivationPattern> {
        let mut activation_pattern = ActivationPattern::new(format!("lateral_{}", concept));
        
        let matching_entities = self.find_concept_entities(concept).await?;
        
        if matching_entities.is_empty() {
            return Err(GraphError::ProcessingError(format!("No entities found for concept: {}", concept)));
        }
        
        // Set activations with creativity boost for lateral thinking
        for (entity_key, relevance) in matching_entities {
            let boosted_activation = (relevance * self.creativity_boost).min(1.0);
            activation_pattern.activations.insert(entity_key, boosted_activation);
        }
        
        Ok(activation_pattern)
    }
    
    /// Use neural networks to find creative bridges between concepts
    async fn neural_bridge_search(
        &self,
        start_activation: ActivationPattern,
        end_activation: ActivationPattern,
        max_length: usize,
    ) -> Result<Vec<BridgeCandidate>> {
        let mut bridge_candidates = Vec::new();
        
        // Get starting and ending entities
        let start_entities: Vec<EntityKey> = start_activation.activations.keys().cloned().collect();
        let end_entities: Vec<EntityKey> = end_activation.activations.keys().cloned().collect();
        
        // For each start-end pair, find creative bridges
        for &start_entity in &start_entities {
            for &end_entity in &end_entities {
                if start_entity != end_entity {
                    // Use multiple bridge-finding strategies
                    let bridges = self.find_multiple_bridge_strategies(
                        start_entity,
                        end_entity,
                        max_length,
                    ).await?;
                    
                    bridge_candidates.extend(bridges);
                }
            }
        }
        
        // Enhance with neural models
        let enhanced_bridges = self.enhance_bridges_with_neural_models(bridge_candidates).await?;
        
        Ok(enhanced_bridges)
    }
    
    /// Find bridges using multiple strategies
    async fn find_multiple_bridge_strategies(
        &self,
        start: EntityKey,
        end: EntityKey,
        max_length: usize,
    ) -> Result<Vec<BridgeCandidate>> {
        let mut all_bridges = Vec::new();
        
        // Strategy 1: Breadth-first search with creativity scoring
        let bfs_bridges = self.creative_breadth_first_search(start, end, max_length).await?;
        all_bridges.extend(bfs_bridges);
        
        // Strategy 2: Random walk with neural guidance
        let random_bridges = self.neural_guided_random_walk(start, end, max_length).await?;
        all_bridges.extend(random_bridges);
        
        // Strategy 3: Semantic embedding space navigation
        let semantic_bridges = self.semantic_space_navigation(start, end, max_length).await?;
        all_bridges.extend(semantic_bridges);
        
        // Remove duplicates and sort by novelty
        self.deduplicate_and_sort_bridges(all_bridges)
    }
    
    /// Creative breadth-first search for unexpected connections
    async fn creative_breadth_first_search(
        &self,
        start: EntityKey,
        end: EntityKey,
        max_length: usize,
    ) -> Result<Vec<BridgeCandidate>> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut bridges = Vec::new();
        
        queue.push_back(BridgeCandidate {
            path: vec![start],
            intermediate_concepts: Vec::new(),
            creativity_score: 0.0,
            plausibility_score: 1.0,
        });
        visited.insert(start);
        
        while let Some(current_bridge) = queue.pop_front() {
            if current_bridge.path.len() > max_length {
                continue;
            }
            
            let current_entity = *current_bridge.path.last().unwrap();
            
            if current_entity == end {
                bridges.push(current_bridge);
                continue;
            }
            
            // Find creative connections (prefer unexpected ones)
            let connections = self.find_creative_connections_from_entity(current_entity).await?;
            
            for (connected_entity, _connection_type, creativity_score) in connections {
                if !visited.contains(&connected_entity) && current_bridge.path.len() < max_length {
                    visited.insert(connected_entity);
                    
                    let mut new_path = current_bridge.path.clone();
                    new_path.push(connected_entity);
                    
                    let mut new_concepts = current_bridge.intermediate_concepts.clone();
                    if let Ok(entity) = self.get_entity(connected_entity).await {
                        new_concepts.push(entity.concept_id);
                    }
                    
                    let new_bridge = BridgeCandidate {
                        path: new_path,
                        intermediate_concepts: new_concepts,
                        creativity_score: current_bridge.creativity_score + creativity_score,
                        plausibility_score: current_bridge.plausibility_score * 0.9, // Decay plausibility
                    };
                    
                    queue.push_back(new_bridge);
                }
            }
        }
        
        Ok(bridges)
    }
    
    /// Neural-guided random walk for creative exploration
    async fn neural_guided_random_walk(
        &self,
        start: EntityKey,
        end: EntityKey,
        max_length: usize,
    ) -> Result<Vec<BridgeCandidate>> {
        let mut bridges = Vec::new();
        let num_walks = 5; // Number of random walks to attempt
        
        for _ in 0..num_walks {
            if let Some(bridge) = self.single_neural_guided_walk(start, end, max_length).await? {
                bridges.push(bridge);
            }
        }
        
        Ok(bridges)
    }
    
    /// Single neural-guided random walk
    async fn single_neural_guided_walk(
        &self,
        start: EntityKey,
        end: EntityKey,
        max_length: usize,
    ) -> Result<Option<BridgeCandidate>> {
        let mut current_entity = start;
        let mut path = vec![start];
        let mut concepts = Vec::new();
        let mut total_creativity = 0.0;
        
        for step in 0..max_length {
            if current_entity == end {
                return Ok(Some(BridgeCandidate {
                    path,
                    intermediate_concepts: concepts,
                    creativity_score: total_creativity,
                    plausibility_score: 1.0 / (step + 1) as f32,
                }));
            }
            
            // Use neural network to predict next step
            let next_candidates = self.get_neural_guided_candidates(current_entity, end).await?;
            
            if next_candidates.is_empty() {
                break;
            }
            
            // Select candidate with highest creativity potential
            let (next_entity, creativity_score) = next_candidates[0];
            
            path.push(next_entity);
            if let Ok(entity) = self.get_entity(next_entity).await {
                concepts.push(entity.concept_id);
            }
            total_creativity += creativity_score;
            current_entity = next_entity;
        }
        
        Ok(None)
    }
    
    /// Navigation through semantic embedding space
    async fn semantic_space_navigation(
        &self,
        start: EntityKey,
        end: EntityKey,
        max_length: usize,
    ) -> Result<Vec<BridgeCandidate>> {
        // Get embeddings for start and end entities
        let start_embedding = self.get_entity_embedding(start).await?;
        let end_embedding = self.get_entity_embedding(end).await?;
        
        // Find intermediate points in embedding space
        let intermediate_points = self.generate_intermediate_embeddings(&start_embedding, &end_embedding, max_length - 2);
        
        let mut bridges = Vec::new();
        
        // For each intermediate point, find closest entities
        for intermediate_embedding in intermediate_points {
            if let Some(bridge) = self.build_bridge_through_embedding(
                start,
                end,
                &intermediate_embedding,
                max_length,
            ).await? {
                bridges.push(bridge);
            }
        }
        
        Ok(bridges)
    }
    
    /// Enhance bridges using neural models (FedFormer, StemGNN)
    async fn enhance_bridges_with_neural_models(
        &self,
        bridges: Vec<BridgeCandidate>,
    ) -> Result<Vec<BridgeCandidate>> {
        let mut enhanced_bridges = Vec::new();
        
        for bridge in bridges {
            // Use FedFormer for temporal relationship analysis
            let temporal_score = if let Some(fedformer_model) = self.bridge_models.get("fedformer") {
                self.analyze_temporal_relationships(&bridge, fedformer_model).await?
            } else {
                bridge.creativity_score
            };
            
            // Use StemGNN for creative connection strength
            let connection_score = if let Some(stemgnn_model) = self.bridge_models.get("stemgnn") {
                self.analyze_connection_strength(&bridge, stemgnn_model).await?
            } else {
                bridge.plausibility_score
            };
            
            enhanced_bridges.push(BridgeCandidate {
                path: bridge.path,
                intermediate_concepts: bridge.intermediate_concepts,
                creativity_score: (bridge.creativity_score + temporal_score) / 2.0,
                plausibility_score: (bridge.plausibility_score + connection_score) / 2.0,
            });
        }
        
        Ok(enhanced_bridges)
    }
    
    /// Score bridges by creativity and novelty
    async fn score_bridge_creativity(&self, bridges: Vec<BridgeCandidate>) -> Result<Vec<BridgePath>> {
        let mut scored_bridges = Vec::new();
        
        for bridge in bridges {
            let novelty_score = self.calculate_bridge_novelty(&bridge).await?;
            let plausibility_score = self.calculate_bridge_plausibility(&bridge).await?;
            let explanation = self.generate_bridge_explanation(&bridge).await?;
            
            scored_bridges.push(BridgePath {
                path: bridge.path,
                intermediate_concepts: bridge.intermediate_concepts,
                novelty_score,
                plausibility_score,
                explanation,
            });
        }
        
        // Sort by combined creativity score
        scored_bridges.sort_by(|a, b| {
            let score_a = a.novelty_score * 0.6 + a.plausibility_score * 0.4;
            let score_b = b.novelty_score * 0.6 + b.plausibility_score * 0.4;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(scored_bridges)
    }
    
    // Helper methods for implementation
    
    async fn find_concept_entities(&self, concept: &str) -> Result<Vec<(EntityKey, f32)>> {
        let all_entities = self.graph.get_all_entities().await;
        let mut matches = Vec::new();
        
        for (key, _entity_data, _) in &all_entities {
            let concept_id = format!("entity_{:?}", key);
            let relevance = self.calculate_concept_relevance(&concept_id, concept);
            if relevance > 0.2 {
                matches.push((*key, relevance));
            }
        }
        
        Ok(matches)
    }
    
    async fn get_entity(&self, entity_key: EntityKey) -> Result<BrainInspiredEntity> {
        let all_entities = self.graph.get_all_entities().await;
        all_entities.iter()
            .find(|(k, _, _)| k == &entity_key)
            .map(|(_, data, activation)| BrainInspiredEntity {
                id: entity_key,
                concept_id: format!("entity_{:?}", entity_key),
                direction: EntityDirection::Input,
                properties: HashMap::new(),
                embedding: data.embedding.clone(),
                activation_state: *activation,
                last_activation: std::time::SystemTime::now(),
                last_update: std::time::SystemTime::now(),
            })
            .ok_or(GraphError::EntityKeyNotFound { key: entity_key })
    }
    
    fn clean_concept(&self, concept: &str) -> String {
        concept.trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .filter(|word| !self.is_stop_word(word))
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    fn is_stop_word(&self, word: &str) -> bool {
        let word_clean = word.to_lowercase();
        let word_clean = word_clean.trim_end_matches('?');
        matches!(word_clean,
            "the" | "is" | "at" | "which" | "on" | "a" | "an" | "and" | "or" |
            "but" | "in" | "with" | "to" | "for" | "of" | "as" | "by" | "that" |
            "this" | "how" | "what" | "where" | "when" | "why" | "related" | "connected" |
            "who" | "whom" | "whose" | "type" | "types" | "kind" | "kinds" | "about" |
            "can" | "could" | "should" | "would" | "will" | "was" | "were" | "been" |
            "being" | "have" | "has" | "had" | "do" | "does" | "did" | "many" | "much" |
            "some" | "any" | "are" | "examples" | "instances"
        )
    }
    
    fn calculate_concept_relevance(&self, entity_concept: &str, query_concept: &str) -> f32 {
        if entity_concept.to_lowercase().contains(&query_concept.to_lowercase()) {
            1.0
        } else if query_concept.to_lowercase().contains(&entity_concept.to_lowercase()) {
            0.8
        } else {
            // Simple word overlap similarity
            let entity_words: HashSet<&str> = entity_concept.split_whitespace().collect();
            let query_words: HashSet<&str> = query_concept.split_whitespace().collect();
            
            let intersection = entity_words.intersection(&query_words).count();
            let union = entity_words.union(&query_words).count();
            
            if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            }
        }
    }
    
    // Placeholder implementations for complex neural operations
    
    async fn find_creative_connections_from_entity(&self, _entity: EntityKey) -> Result<Vec<(EntityKey, RelationType, f32)>> {
        // Implementation would find unexpected connections with high creativity scores
        Ok(Vec::new())
    }
    
    async fn get_neural_guided_candidates(&self, _current: EntityKey, _target: EntityKey) -> Result<Vec<(EntityKey, f32)>> {
        // Implementation would use neural networks to predict creative next steps
        Ok(Vec::new())
    }
    
    async fn get_entity_embedding(&self, _entity: EntityKey) -> Result<Vec<f32>> {
        // Implementation would get neural embedding for entity
        Ok(vec![0.0; 384])
    }
    
    fn generate_intermediate_embeddings(&self, _start: &[f32], _end: &[f32], _count: usize) -> Vec<Vec<f32>> {
        // Implementation would generate creative intermediate points in embedding space
        Vec::new()
    }
    
    async fn build_bridge_through_embedding(&self, _start: EntityKey, _end: EntityKey, _intermediate: &[f32], _max_length: usize) -> Result<Option<BridgeCandidate>> {
        // Implementation would build bridge through specific embedding point
        Ok(None)
    }
    
    async fn analyze_temporal_relationships(&self, bridge: &BridgeCandidate, _model: &str) -> Result<f32> {
        // Implementation would use FedFormer to analyze temporal aspects
        Ok(bridge.creativity_score)
    }
    
    async fn analyze_connection_strength(&self, bridge: &BridgeCandidate, _model: &str) -> Result<f32> {
        // Implementation would use StemGNN to analyze connection strength
        Ok(bridge.plausibility_score)
    }
    
    fn deduplicate_and_sort_bridges(&self, bridges: Vec<BridgeCandidate>) -> Result<Vec<BridgeCandidate>> {
        // Remove duplicates and sort by creativity
        let mut unique_bridges = Vec::new();
        let mut seen_paths = HashSet::new();
        
        for bridge in bridges {
            let path_signature = format!("{:?}", bridge.path);
            if !seen_paths.contains(&path_signature) {
                seen_paths.insert(path_signature);
                unique_bridges.push(bridge);
            }
        }
        
        unique_bridges.sort_by(|a, b| b.creativity_score.partial_cmp(&a.creativity_score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(unique_bridges)
    }
    
    async fn calculate_bridge_novelty(&self, bridge: &BridgeCandidate) -> Result<f32> {
        // Calculate novelty based on concept diversity and unexpectedness
        let concept_diversity = bridge.intermediate_concepts.len() as f32 / 10.0;
        let path_length_bonus = (bridge.path.len() as f32 / self.max_bridge_length as f32) * 0.3;
        Ok((concept_diversity + path_length_bonus + bridge.creativity_score).min(1.0))
    }
    
    async fn calculate_bridge_plausibility(&self, bridge: &BridgeCandidate) -> Result<f32> {
        // Calculate plausibility based on relationship strengths
        Ok(bridge.plausibility_score)
    }
    
    async fn generate_bridge_explanation(&self, bridge: &BridgeCandidate) -> Result<String> {
        if bridge.intermediate_concepts.is_empty() {
            return Ok("Direct connection found.".to_string());
        }
        
        Ok(format!(
            "Creative connection through: {}",
            bridge.intermediate_concepts.join(" → ")
        ))
    }
    
    async fn analyze_novelty(&self, bridges: &[BridgePath]) -> Result<NoveltyAnalysis> {
        if bridges.is_empty() {
            return Ok(NoveltyAnalysis {
                overall_novelty: 0.0,
                concept_uniqueness: Vec::new(),
                path_creativity: 0.0,
            });
        }
        
        let overall_novelty = bridges.iter().map(|b| b.novelty_score).sum::<f32>() / bridges.len() as f32;
        let concept_uniqueness = bridges.iter().map(|b| b.novelty_score).collect();
        let path_creativity = bridges.iter().map(|b| b.plausibility_score).sum::<f32>() / bridges.len() as f32;
        
        Ok(NoveltyAnalysis {
            overall_novelty,
            concept_uniqueness,
            path_creativity,
        })
    }
    
    fn calculate_confidence_distribution(&self, bridges: &[BridgePath]) -> Vec<f32> {
        bridges.iter().map(|b| (b.novelty_score + b.plausibility_score) / 2.0).collect()
    }
}

#[async_trait]
impl CognitivePattern for LateralThinking {
    async fn execute(
        &self,
        query: &str,
        context: Option<&str>,
        parameters: PatternParameters,
    ) -> Result<PatternResult> {
        let start_time = Instant::now();
        
        // Parse query to extract two concepts
        let (concept_a, concept_b) = self.parse_lateral_query(query).await?;
        
        // Update parameters if provided
        let max_bridge_length = parameters.max_depth.unwrap_or(self.max_bridge_length);
        
        let result = self.find_creative_connections(&concept_a, &concept_b, Some(max_bridge_length)).await?;
        
        let execution_time = start_time.elapsed();
        
        // Generate answer from lateral connections
        let answer = if result.bridges.is_empty() {
            format!("No creative connections found between {} and {}", concept_a, concept_b)
        } else {
            let best_bridge = &result.bridges[0];
            format!(
                "Creative connection between {} and {}: {}",
                concept_a, concept_b, best_bridge.explanation
            )
        };
        
        // Create reasoning trace
        let reasoning_trace = self.create_reasoning_trace(&result, &concept_a, &concept_b).await?;
        
        // Calculate confidence from bridge scores
        let confidence = if !result.confidence_distribution.is_empty() {
            result.confidence_distribution.iter().sum::<f32>() / result.confidence_distribution.len() as f32
        } else {
            0.0
        };
        
        Ok(PatternResult {
            pattern_type: CognitivePatternType::Lateral,
            answer,
            confidence,
            reasoning_trace,
            metadata: ResultMetadata {
                execution_time_ms: execution_time.as_millis() as u64,
                nodes_activated: result.bridges.iter().map(|b| b.path.len()).sum(),
                iterations_completed: max_bridge_length,
                converged: false, // Lateral thinking explores multiple paths
                total_energy: result.novelty_analysis.overall_novelty,
                additional_info: {
                    let mut info = HashMap::new();
                    info.insert("query".to_string(), query.to_string());
                    info.insert("pattern".to_string(), "lateral".to_string());
                    info.insert("concept_a".to_string(), concept_a);
                    info.insert("concept_b".to_string(), concept_b);
                    info.insert("bridges_found".to_string(), result.bridges.len().to_string());
                    info
                },
            },
        })
    }
    
    fn get_pattern_type(&self) -> CognitivePatternType {
        CognitivePatternType::Lateral
    }
    
    fn get_optimal_use_cases(&self) -> Vec<String> {
        vec![
            "Creative problem solving".to_string(),
            "Finding unexpected connections".to_string(),
            "Brainstorming relationships".to_string(),
            "Cross-domain thinking".to_string(),
            "Innovation and invention".to_string(),
            "How is X related to Y?".to_string(),
            "Creative connections between concepts".to_string(),
        ]
    }
    
    fn estimate_complexity(&self, query: &str) -> ComplexityEstimate {
        let word_count = query.split_whitespace().count();
        let complexity = ((word_count * self.max_bridge_length) as u32).min(100);
        
        ComplexityEstimate {
            computational_complexity: complexity,
            estimated_time_ms: (complexity as u64) * 200, // More expensive than convergent
            memory_requirements_mb: complexity * 5,
            confidence: 0.6, // Lateral thinking has more uncertainty
            parallelizable: true,
        }
    }
}

impl LateralThinking {
    async fn create_reasoning_trace(&self, result: &LateralResult, _concept_a: &str, _concept_b: &str) -> Result<Vec<ActivationStep>> {
        let mut trace = Vec::new();
        
        for (i, bridge) in result.bridges.iter().enumerate() {
            for (j, &entity_key) in bridge.path.iter().enumerate() {
                trace.push(ActivationStep {
                    step_id: trace.len(),
                    entity_key,
                    concept_id: bridge.intermediate_concepts.get(j)
                        .unwrap_or(&format!("bridge_{}_{}", i, j)).clone(),
                    activation_level: bridge.novelty_score * (1.0 - 0.1 * j as f32),
                    operation_type: if j == 0 {
                        ActivationOperation::Initialize
                    } else {
                        ActivationOperation::Propagate
                    },
                    timestamp: SystemTime::now(),
                });
            }
        }
        
        Ok(trace)
    }
}

/// Internal bridge candidate for processing
struct BridgeCandidate {
    path: Vec<EntityKey>,
    intermediate_concepts: Vec<String>,
    creativity_score: f32,
    plausibility_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lateral_query_parsing() {
        let lateral = LateralThinking::new(
            Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().unwrap()),
        );
        
        // Test would check query parsing logic
        assert_eq!(lateral.clean_concept("  Hello World!  "), "hello world");
        assert!(lateral.is_stop_word("the"));
        assert!(!lateral.is_stop_word("creativity"));
    }
    
    #[test]
    fn test_concept_relevance() {
        let lateral = LateralThinking::new(
            Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().unwrap()),
        );
        
        assert_eq!(lateral.calculate_concept_relevance("dog", "dog"), 1.0);
        assert!(lateral.calculate_concept_relevance("machine learning", "learning") > 0.5);
    }
}