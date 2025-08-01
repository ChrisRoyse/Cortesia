//! Multi-Hop Reasoner
//! 
//! Implements multi-hop reasoning capabilities for complex queries that
//! require following chains of relationships and making inferences.

use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tracing::{info, error, debug, instrument};
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    hierarchical_storage::types::*,
    retrieval_system::types::*,
    logging::LogContext,
};

/// Multi-hop reasoning engine
pub struct MultiHopReasoner {
    model_manager: Arc<ModelResourceManager>,
    config: RetrievalConfig,
}

impl MultiHopReasoner {
    /// Create new multi-hop reasoner
    pub fn new(
        model_manager: Arc<ModelResourceManager>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Perform multi-hop reasoning
    #[instrument(
        skip(self, query, initial_results, graph_context),
        fields(
            max_hops = max_hops,
            initial_results_count = initial_results.len(),
            graph_nodes = graph_context.layers.len(),
            graph_connections = graph_context.connections.len()
        )
    )]
    pub async fn perform_reasoning(
        &self,
        query: &ProcessedQuery,
        initial_results: &[RetrievedItem],
        graph_context: &GraphContext,
        max_hops: u32,
    ) -> RetrievalResult2<ReasoningChain> {
        let start_time = Instant::now();
        
        let log_context = LogContext::new("multi_hop_reasoning", "multi_hop_reasoner");
        
        info!(
            context = ?log_context,
            max_hops = max_hops,
            initial_results_count = initial_results.len(),
            graph_nodes = graph_context.layers.len(),
            graph_connections = graph_context.connections.len(),
            query_complexity = ?query.understanding.complexity_level,
            "Starting multi-hop reasoning"
        );
        
        // Step 1: Identify reasoning type needed
        info!("Step 1/3: Identifying reasoning type");
        let reasoning_start = Instant::now();
        let reasoning_type = self.identify_reasoning_type(query, initial_results).await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    error = %e,
                    "Failed to identify reasoning type"
                );
                e
            })?;
        debug!(
            reasoning_type_identification_ms = reasoning_start.elapsed().as_millis(),
            reasoning_type = ?reasoning_type,
            "Reasoning type identification completed"
        );
        
        // Step 2: Generate initial hypotheses
        info!("Step 2/3: Generating initial hypotheses");
        let hypothesis_start = Instant::now();
        let initial_hypotheses = self.generate_hypotheses(query, initial_results).await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    error = %e,
                    "Failed to generate initial hypotheses"
                );
                e
            })?;
        debug!(
            hypothesis_generation_ms = hypothesis_start.elapsed().as_millis(),
            hypotheses_generated = initial_hypotheses.len(),
            "Initial hypothesis generation completed"
        );
        
        // Step 3: Execute reasoning steps
        info!("Step 3/3: Executing reasoning steps (max {} hops)", max_hops);
        let mut reasoning_steps = Vec::new();
        let mut current_evidence = initial_results.to_vec();
        let mut visited_layers = HashSet::new();
        
        let loop_start = Instant::now();
        for hop in 0..max_hops {
            if initial_hypotheses.is_empty() {
                debug!(hop = hop, "No more hypotheses to explore, stopping reasoning");
                break;
            }
            
            let hop_start = Instant::now();
            debug!(hop = hop + 1, max_hops = max_hops, "Starting reasoning hop");
            
            // Select next hypothesis to explore
            let hypothesis = if (hop as usize) < initial_hypotheses.len() {
                &initial_hypotheses[hop as usize]
            } else {
                // Generate new hypothesis based on current evidence
                debug!("Generating new hypothesis from current evidence");
                let new_hypothesis = self.generate_next_hypothesis(
                    query,
                    &current_evidence,
                    &reasoning_steps,
                ).await
                .map_err(|e| {
                    error!(
                        context = ?log_context,
                        hop = hop,
                        error = %e,
                        "Failed to generate next hypothesis"
                    );
                    e
                })?;
                
                if new_hypothesis.is_none() {
                    debug!(hop = hop, "No new hypothesis generated, stopping reasoning");
                    break;
                }
                
                initial_hypotheses.first().unwrap() // Fallback
            };
            
            debug!(
                hop = hop + 1,
                hypothesis = %hypothesis,
                "Selected hypothesis for exploration"
            );
            
            // Find supporting evidence through graph traversal
            let evidence_start = Instant::now();
            let (new_evidence, inference) = self.find_supporting_evidence(
                hypothesis,
                &current_evidence,
                graph_context,
                &mut visited_layers,
            ).await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    hop = hop,
                    error = %e,
                    "Failed to find supporting evidence"
                );
                e
            })?;
            
            debug!(
                evidence_search_ms = evidence_start.elapsed().as_millis(),
                new_evidence_count = new_evidence.len(),
                visited_layers_count = visited_layers.len(),
                "Evidence search completed for hop"
            );
            
            // Create reasoning step
            let step_confidence = self.calculate_step_confidence(&new_evidence);
            let step_type = self.determine_step_type(&inference);
            
            let step = ReasoningStep {
                step_number: hop + 1,
                hypothesis: hypothesis.clone(),
                supporting_evidence: new_evidence
                    .iter()
                    .map(|item| item.content.clone())
                    .take(3)
                    .collect(),
                layer_ids: new_evidence
                    .iter()
                    .map(|item| item.layer_id.clone())
                    .collect(),
                inference: inference.clone(),
                confidence: step_confidence,
                step_type: step_type.clone(),
            };
            
            reasoning_steps.push(step);
            
            debug!(
                hop = hop + 1,
                step_confidence = step_confidence,
                step_type = ?step_type,
                inference_length = inference.len(),
                hop_time_ms = hop_start.elapsed().as_millis(),
                "Reasoning step completed"
            );
            
            // Update current evidence
            let pre_evidence_count = current_evidence.len();
            current_evidence.extend(new_evidence);
            debug!(
                evidence_before = pre_evidence_count,
                evidence_after = current_evidence.len(),
                "Updated current evidence pool"
            );
            
            // Check if we have enough evidence to conclude
            if self.has_sufficient_evidence(&current_evidence, query) {
                info!(
                    hop = hop + 1,
                    total_evidence = current_evidence.len(),
                    "Sufficient evidence found, stopping reasoning early"
                );
                break;
            }
        }
        
        debug!(
            reasoning_loop_ms = loop_start.elapsed().as_millis(),
            total_steps = reasoning_steps.len(),
            "Reasoning loop completed"
        );
        
        // Step 4: Generate final conclusion
        info!("Generating final conclusion from {} reasoning steps", reasoning_steps.len());
        let conclusion_start = Instant::now();
        let (final_conclusion, confidence) = self.generate_conclusion(
            query,
            &reasoning_steps,
            &current_evidence,
        ).await
        .map_err(|e| {
            error!(
                context = ?log_context,
                error = %e,
                "Failed to generate final conclusion"
            );
            e
        })?;
        debug!(
            conclusion_generation_ms = conclusion_start.elapsed().as_millis(),
            conclusion_length = final_conclusion.len(),
            final_confidence = confidence,
            "Final conclusion generation completed"
        );
        
        // Step 5: Calculate evidence strength
        let strength_start = Instant::now();
        let evidence_strength = self.calculate_evidence_strength(&reasoning_steps);
        debug!(
            strength_calculation_ms = strength_start.elapsed().as_millis(),
            evidence_strength = evidence_strength,
            "Evidence strength calculation completed"
        );
        
        let total_time = start_time.elapsed();
        let chain = ReasoningChain {
            reasoning_steps: reasoning_steps.clone(),
            final_conclusion: final_conclusion.clone(),
            confidence,
            evidence_strength,
            reasoning_type,
        };
        
        info!(
            context = ?log_context,
            total_reasoning_time_ms = total_time.as_millis(),
            reasoning_steps_count = chain.reasoning_steps.len(),
            final_confidence = chain.confidence,
            evidence_strength = chain.evidence_strength,
            reasoning_type = ?chain.reasoning_type,
            conclusion_length = chain.final_conclusion.len(),
            "Multi-hop reasoning completed successfully"
        );
        
        Ok(chain)
    }
    
    /// Identify the type of reasoning needed
    async fn identify_reasoning_type(
        &self,
        query: &ProcessedQuery,
        _initial_results: &[RetrievedItem],
    ) -> RetrievalResult2<ReasoningType> {
        // Based on query intent and initial results
        match query.understanding.intent {
            QueryIntent::CausalAnalysis => Ok(ReasoningType::Causal),
            QueryIntent::TemporalSequence => Ok(ReasoningType::Temporal),
            QueryIntent::Comparison => Ok(ReasoningType::Analogical),
            _ => {
                // Analyze query for reasoning indicators
                let query_lower = query.original_query.natural_language_query.to_lowercase();
                
                if query_lower.contains("why") || query_lower.contains("because") {
                    Ok(ReasoningType::Abductive)
                } else if query_lower.contains("if") || query_lower.contains("then") {
                    Ok(ReasoningType::Deductive)
                } else {
                    Ok(ReasoningType::Inductive)
                }
            }
        }
    }
    
    /// Generate initial hypotheses based on query
    async fn generate_hypotheses(
        &self,
        query: &ProcessedQuery,
        initial_results: &[RetrievedItem],
    ) -> RetrievalResult2<Vec<String>> {
        let prompt = format!(
            r#"Generate hypotheses for multi-hop reasoning:

Query: "{}"
Intent: {:?}
Initial evidence: {} items found

Key entities: {:?}
Key concepts: {:?}

Generate 3-5 hypotheses that could help answer the query through multi-hop reasoning.
Each hypothesis should suggest a path to explore through the knowledge graph.

Return as JSON array:
["hypothesis1", "hypothesis2", "hypothesis3"]

JSON Response:"#,
            query.original_query.natural_language_query,
            query.understanding.intent,
            initial_results.len(),
            query.understanding.extracted_entities,
            query.understanding.extracted_concepts
        );
        
        let task = ProcessingTask::new(ComplexityLevel::High, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| RetrievalError::ReasoningError(e.to_string()))?;
        
        self.parse_hypotheses_response(&result.output)
    }
    
    /// Parse hypotheses from model response
    fn parse_hypotheses_response(&self, response: &str) -> RetrievalResult2<Vec<String>> {
        let json_start = response.find('[').unwrap_or(0);
        let json_end = response.rfind(']').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];
        
        let parsed: Vec<String> = serde_json::from_str(json_str)
            .map_err(|e| RetrievalError::ReasoningError(format!("JSON parse error: {e}")))?;
        
        Ok(parsed)
    }
    
    /// Generate next hypothesis based on current evidence
    async fn generate_next_hypothesis(
        &self,
        query: &ProcessedQuery,
        current_evidence: &[RetrievedItem],
        reasoning_steps: &[ReasoningStep],
    ) -> RetrievalResult2<Option<String>> {
        if current_evidence.is_empty() {
            return Ok(None);
        }
        
        let recent_evidence = current_evidence
            .iter()
            .rev()
            .take(3)
            .map(|item| &item.content)
            .collect::<Vec<_>>();
        
        let prompt = format!(
            r#"Generate next hypothesis for reasoning:

Original query: "{}"
Previous steps: {} completed

Recent evidence:
{}

What should we explore next to answer the query?

Return a single hypothesis as a string, or "COMPLETE" if we have enough information.

Response:"#,
            query.original_query.natural_language_query,
            reasoning_steps.len(),
            recent_evidence.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n---\n")
        );
        
        let task = ProcessingTask::new(ComplexityLevel::Medium, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| RetrievalError::ReasoningError(e.to_string()))?;
        
        let response = result.output.trim();
        if response == "COMPLETE" || response.is_empty() {
            Ok(None)
        } else {
            Ok(Some(response.to_string()))
        }
    }
    
    /// Find supporting evidence through graph traversal
    async fn find_supporting_evidence(
        &self,
        hypothesis: &str,
        current_evidence: &[RetrievedItem],
        graph_context: &GraphContext,
        visited_layers: &mut HashSet<String>,
    ) -> RetrievalResult2<(Vec<RetrievedItem>, String)> {
        let mut new_evidence = Vec::new();
        let inference;
        
        // Extract key terms from hypothesis
        let hypothesis_terms = self.extract_key_terms(hypothesis);
        
        // Find connected layers through semantic links
        for item in current_evidence.iter().rev().take(5) {
            // Skip if already visited
            if visited_layers.contains(&item.layer_id) {
                continue;
            }
            visited_layers.insert(item.layer_id.clone());
            
            // Get connected layers from graph
            let connected = graph_context.get_connected_layers(&item.layer_id);
            
            for (connected_id, link_types) in connected {
                if visited_layers.contains(&connected_id) {
                    continue;
                }
                
                // Check if connected layer is relevant to hypothesis
                if let Some(layer_content) = graph_context.get_layer_content(&connected_id) {
                    let relevance = self.calculate_relevance_to_hypothesis(
                        &layer_content,
                        hypothesis,
                        &hypothesis_terms,
                    );
                    
                    if relevance > 0.6 {
                        let retrieved_item = RetrievedItem {
                            layer_id: connected_id.clone(),
                            document_id: item.document_id.clone(),
                            content: layer_content.clone(),
                            relevance_score: relevance,
                            match_explanation: MatchExplanation {
                                matched_keywords: hypothesis_terms.clone(),
                                matched_entities: Vec::new(),
                                matched_concepts: Vec::new(),
                                semantic_similarity: Some(relevance),
                                reasoning_steps: vec![format!("Connected via {:?}", link_types)],
                                match_type: MatchType::MultiHopInference,
                            },
                            context_before: None,
                            context_after: None,
                            layer_type: LayerType::Paragraph, // Would get actual type
                            importance_score: 0.7,
                            semantic_links: Vec::new(),
                        };
                        
                        new_evidence.push(retrieved_item);
                        visited_layers.insert(connected_id);
                    }
                }
            }
        }
        
        // Generate inference from found evidence
        if !new_evidence.is_empty() {
            inference = self.generate_inference(hypothesis, &new_evidence).await?;
        } else {
            inference = format!("No direct evidence found for: {hypothesis}");
        }
        
        Ok((new_evidence, inference))
    }
    
    /// Generate inference from evidence
    async fn generate_inference(
        &self,
        hypothesis: &str,
        evidence: &[RetrievedItem],
    ) -> RetrievalResult2<String> {
        let evidence_summary = evidence
            .iter()
            .take(3)
            .map(|item| item.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        let prompt = format!(
            r#"Generate an inference based on evidence:

Hypothesis: "{hypothesis}"

Evidence found:
{evidence_summary}

What can we infer from this evidence regarding the hypothesis?

Inference:"#
        );
        
        let task = ProcessingTask::new(ComplexityLevel::Medium, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| RetrievalError::ReasoningError(e.to_string()))?;
        
        Ok(result.output.trim().to_string())
    }
    
    /// Generate final conclusion from reasoning chain
    async fn generate_conclusion(
        &self,
        query: &ProcessedQuery,
        reasoning_steps: &[ReasoningStep],
        all_evidence: &[RetrievedItem],
    ) -> RetrievalResult2<(String, f32)> {
        let steps_summary = reasoning_steps
            .iter()
            .map(|step| format!("Step {}: {} -> {}", step.step_number, step.hypothesis, step.inference))
            .collect::<Vec<_>>()
            .join("\n");
        
        let prompt = format!(
            r#"Generate a conclusion from multi-hop reasoning:

Original query: "{}"

Reasoning chain:
{}

Total evidence pieces: {}

Synthesize a clear, concise conclusion that answers the original query.
Also provide a confidence score (0.0-1.0) for this conclusion.

Format:
Conclusion: [your conclusion]
Confidence: [0.0-1.0]

Response:"#,
            query.original_query.natural_language_query,
            steps_summary,
            all_evidence.len()
        );
        
        let task = ProcessingTask::new(ComplexityLevel::High, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| RetrievalError::ReasoningError(e.to_string()))?;
        
        self.parse_conclusion_response(&result.output)
    }
    
    /// Parse conclusion response
    fn parse_conclusion_response(&self, response: &str) -> RetrievalResult2<(String, f32)> {
        let lines: Vec<&str> = response.lines().collect();
        let mut conclusion = String::new();
        let mut confidence = 0.5;
        
        for line in lines {
            if line.starts_with("Conclusion:") {
                conclusion = line.strip_prefix("Conclusion:").unwrap_or("").trim().to_string();
            } else if line.starts_with("Confidence:") {
                let conf_str = line.strip_prefix("Confidence:").unwrap_or("0.5").trim();
                confidence = conf_str.parse::<f32>().unwrap_or(0.5).clamp(0.0, 1.0);
            }
        }
        
        if conclusion.is_empty() {
            conclusion = "Unable to reach a definitive conclusion based on available evidence.".to_string();
        }
        
        Ok((conclusion, confidence))
    }
    
    // Helper methods
    
    /// Extract key terms from hypothesis
    fn extract_key_terms(&self, hypothesis: &str) -> Vec<String> {
        hypothesis
            .split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|word| word.len() > 3 && !self.is_stop_word(word))
            .collect()
    }
    
    /// Check if word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "been", "be",
        ];
        
        STOP_WORDS.contains(&word)
    }
    
    /// Calculate relevance of content to hypothesis
    fn calculate_relevance_to_hypothesis(
        &self,
        content: &str,
        _hypothesis: &str,
        hypothesis_terms: &[String],
    ) -> f32 {
        let content_lower = content.to_lowercase();
        let mut relevance: f32 = 0.0;
        let mut matches = 0;
        
        // Check for exact hypothesis terms
        for term in hypothesis_terms {
            if content_lower.contains(term) {
                matches += 1;
                relevance += 0.2;
            }
        }
        
        // Boost if multiple terms appear
        if matches > 1 {
            relevance += 0.2;
        }
        
        // Check for semantic indicators
        if content_lower.contains("because") || content_lower.contains("therefore") {
            relevance += 0.1;
        }
        
        relevance.min(1.0)
    }
    
    /// Calculate confidence for a reasoning step
    fn calculate_step_confidence(&self, evidence: &[RetrievedItem]) -> f32 {
        if evidence.is_empty() {
            return 0.0;
        }
        
        let avg_relevance: f32 = evidence
            .iter()
            .map(|item| item.relevance_score)
            .sum::<f32>() / evidence.len() as f32;
        
        // Boost confidence if multiple pieces of evidence
        let evidence_boost = (evidence.len() as f32 / 5.0).min(0.3);
        
        (avg_relevance + evidence_boost).min(1.0)
    }
    
    /// Determine the type of reasoning step
    fn determine_step_type(&self, inference: &str) -> StepType {
        let inference_lower = inference.to_lowercase();
        
        if inference_lower.contains("directly") || inference_lower.contains("shows") {
            StepType::DirectEvidence
        } else if inference_lower.contains("implies") || inference_lower.contains("suggests") {
            StepType::InferredConnection
        } else if inference_lower.contains("through") || inference_lower.contains("via") {
            StepType::TransitiveRelation
        } else if inference_lower.contains("causes") || inference_lower.contains("results") {
            StepType::CausalLink
        } else if inference_lower.contains("before") || inference_lower.contains("after") {
            StepType::TemporalSequence
        } else {
            StepType::ConceptualBridge
        }
    }
    
    /// Check if we have sufficient evidence
    fn has_sufficient_evidence(&self, evidence: &[RetrievedItem], query: &ProcessedQuery) -> bool {
        // Check if we have evidence for all key entities and concepts
        let mut entities_found = 0;
        let mut concepts_found = 0;
        
        for item in evidence {
            for entity in &query.understanding.extracted_entities {
                if item.content.to_lowercase().contains(&entity.to_lowercase()) {
                    entities_found += 1;
                }
            }
            
            for concept in &query.understanding.extracted_concepts {
                if item.content.to_lowercase().contains(&concept.to_lowercase()) {
                    concepts_found += 1;
                }
            }
        }
        
        let entity_coverage = if query.understanding.extracted_entities.is_empty() {
            1.0
        } else {
            entities_found as f32 / query.understanding.extracted_entities.len() as f32
        };
        
        let concept_coverage = if query.understanding.extracted_concepts.is_empty() {
            1.0
        } else {
            concepts_found as f32 / query.understanding.extracted_concepts.len() as f32
        };
        
        // Sufficient if we have good coverage and enough evidence pieces
        evidence.len() >= 5 && entity_coverage > 0.7 && concept_coverage > 0.6
    }
    
    /// Calculate overall evidence strength
    fn calculate_evidence_strength(&self, reasoning_steps: &[ReasoningStep]) -> f32 {
        if reasoning_steps.is_empty() {
            return 0.0;
        }
        
        let total_confidence: f32 = reasoning_steps
            .iter()
            .map(|step| step.confidence)
            .sum();
        
        let avg_confidence = total_confidence / reasoning_steps.len() as f32;
        
        // Boost for longer chains that maintain high confidence
        let chain_bonus = if reasoning_steps.len() >= 3 && avg_confidence > 0.7 {
            0.1
        } else {
            0.0
        };
        
        (avg_confidence + chain_bonus).min(1.0)
    }
}

/// Graph context for multi-hop reasoning
pub struct GraphContext {
    layers: HashMap<String, LayerInfo>,
    connections: HashMap<String, Vec<(String, Vec<SemanticLinkType>)>>,
}

impl Default for GraphContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphContext {
    /// Create new graph context
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
            connections: HashMap::new(),
        }
    }
    
    /// Add layer information
    pub fn add_layer(&mut self, layer_id: String, content: String, layer_type: LayerType) {
        self.layers.insert(
            layer_id.clone(),
            LayerInfo {
                content,
                layer_type,
            }
        );
    }
    
    /// Add connection between layers
    pub fn add_connection(&mut self, source: String, target: String, link_types: Vec<SemanticLinkType>) {
        self.connections
            .entry(source)
            .or_default()
            .push((target, link_types));
    }
    
    /// Get connected layers
    pub fn get_connected_layers(&self, layer_id: &str) -> Vec<(String, Vec<SemanticLinkType>)> {
        self.connections
            .get(layer_id)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get layer content
    pub fn get_layer_content(&self, layer_id: &str) -> Option<String> {
        self.layers
            .get(layer_id)
            .map(|info| info.content.clone())
    }
}

/// Layer information for graph context
#[derive(Debug, Clone)]
struct LayerInfo {
    content: String,
    layer_type: LayerType,
}

// Re-export ProcessedQuery from query_processor
pub use crate::enhanced_knowledge_storage::retrieval_system::query_processor::ProcessedQuery;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    
    #[tokio::test]
    async fn test_multi_hop_reasoner_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let retrieval_config = RetrievalConfig::default();
        
        let reasoner = MultiHopReasoner::new(model_manager, retrieval_config);
        
        assert_eq!(reasoner.config.reasoning_model_id, "smollm2_360m");
    }
    
    #[tokio::test]
    async fn test_key_term_extraction() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let retrieval_config = RetrievalConfig::default();
        
        let reasoner = MultiHopReasoner::new(model_manager, retrieval_config);
        
        let hypothesis = "Einstein's theory of relativity influenced quantum mechanics";
        let terms = reasoner.extract_key_terms(hypothesis);
        
        // "Einstein's" becomes "einstein's" after lowercasing (apostrophe is preserved)
        assert!(terms.contains(&"einstein's".to_string()));
        assert!(terms.contains(&"theory".to_string()));
        assert!(terms.contains(&"relativity".to_string()));
        assert!(terms.contains(&"influenced".to_string()));
        assert!(terms.contains(&"quantum".to_string()));
        assert!(terms.contains(&"mechanics".to_string()));
    }
    
    #[tokio::test]
    async fn test_relevance_calculation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let retrieval_config = RetrievalConfig::default();
        
        let reasoner = MultiHopReasoner::new(model_manager, retrieval_config);
        
        let content = "Einstein developed the theory of relativity which revolutionized physics";
        let hypothesis = "Einstein's relativity theory changed physics";
        let terms = vec!["einstein".to_string(), "relativity".to_string(), "theory".to_string(), "changed".to_string(), "physics".to_string()];
        
        let relevance = reasoner.calculate_relevance_to_hypothesis(content, hypothesis, &terms);
        
        assert!(relevance > 0.5); // Should have high relevance
    }
    
    #[tokio::test]
    async fn test_step_type_determination() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let retrieval_config = RetrievalConfig::default();
        
        let reasoner = MultiHopReasoner::new(model_manager, retrieval_config);
        
        assert_eq!(
            reasoner.determine_step_type("This directly shows that..."),
            StepType::DirectEvidence
        );
        
        assert_eq!(
            reasoner.determine_step_type("This implies a connection..."),
            StepType::InferredConnection
        );
        
        assert_eq!(
            reasoner.determine_step_type("A causes B to happen"),
            StepType::CausalLink
        );
        
        assert_eq!(
            reasoner.determine_step_type("This happened before that"),
            StepType::TemporalSequence
        );
    }
    
    #[test]
    fn test_graph_context() {
        let mut context = GraphContext::new();
        
        context.add_layer("layer1".to_string(), "Content 1".to_string(), LayerType::Paragraph);
        context.add_layer("layer2".to_string(), "Content 2".to_string(), LayerType::Paragraph);
        
        context.add_connection(
            "layer1".to_string(),
            "layer2".to_string(),
            vec![SemanticLinkType::Semantic]
        );
        
        let connected = context.get_connected_layers("layer1");
        assert_eq!(connected.len(), 1);
        assert_eq!(connected[0].0, "layer2");
        
        let content = context.get_layer_content("layer1");
        assert_eq!(content, Some("Content 1".to_string()));
    }
}