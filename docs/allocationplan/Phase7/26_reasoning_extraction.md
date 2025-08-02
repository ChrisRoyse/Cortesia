# Micro Task 26: Reasoning Extraction

**Priority**: CRITICAL  
**Estimated Time**: 45 minutes  
**Dependencies**: 25_explanation_templates.md completed  
**Skills Required**: Graph analysis, pathway interpretation, reasoning chain construction

## Objective

Implement reasoning path extraction from activation traces and pathways to construct logical reasoning chains that can be used for human-interpretable explanations of how the brain-inspired system reached its conclusions.

## Context

Understanding how the system reached a particular conclusion is crucial for transparency and trust. This task analyzes activation pathways, node relationships, and processing steps to extract and reconstruct the logical reasoning sequence that led to the final answer.

## Specifications

### Core Reasoning Components

1. **ReasoningExtractor struct**
   - Pathway analysis engine
   - Logical step identification
   - Causal relationship detection
   - Reasoning chain construction

2. **ReasoningStep struct**
   - Individual reasoning element
   - Evidence and premises
   - Logical connections
   - Confidence scoring

3. **ReasoningChain struct**
   - Ordered sequence of steps
   - Branch point handling
   - Alternative path tracking
   - Chain validation

4. **LogicalConnection struct**
   - Relationship types
   - Strength measurements
   - Bidirectional linking
   - Context preservation

### Performance Requirements

- Reasoning extraction < 10ms per query
- Support for complex multi-hop reasoning
- Memory efficient for long reasoning chains
- Concurrent reasoning analysis
- Real-time pathway interpretation

## Implementation Guide

### Step 1: Core Reasoning Types

```rust
// File: src/cognitive/explanation/reasoning_extraction.rs

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use serde::{Deserialize, Serialize};
use crate::core::types::{NodeId, EntityId, ActivationLevel};
use crate::cognitive::learning::pathway_tracing::{ActivationPathway, PathwaySegment};
use crate::cognitive::explanation::templates::Evidence;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_id: StepId,
    pub step_type: StepType,
    pub premise: String,
    pub conclusion: String,
    pub evidence: Vec<Evidence>,
    pub confidence: f32,
    pub activation_nodes: Vec<NodeId>,
    pub logical_operation: LogicalOperation,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    InitialActivation,
    EntityRecognition,
    RelationshipInference,
    ConceptualBridge,
    LogicalDeduction,
    FactualLookup,
    AnalogicalReasoning,
    SynthesisStep,
    ValidationStep,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperation {
    DirectReference,
    Implication,
    Conjunction,
    Disjunction,
    Causation,
    Association,
    Similarity,
    Contradiction,
    Generalization,
    Specialization,
}

#[derive(Debug, Clone)]
pub struct ReasoningChain {
    pub chain_id: ChainId,
    pub steps: Vec<ReasoningStep>,
    pub connections: Vec<LogicalConnection>,
    pub source_pathways: Vec<PathwayId>,
    pub confidence_score: f32,
    pub completeness_score: f32,
    pub coherence_score: f32,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ChainId(pub u64);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PathwayId(pub u64);

#[derive(Debug, Clone)]
pub struct LogicalConnection {
    pub connection_id: ConnectionId,
    pub source_step: StepId,
    pub target_step: StepId,
    pub connection_type: ConnectionType,
    pub strength: f32,
    pub evidence_support: f32,
    pub logical_validity: f32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ConnectionId(pub u64);

#[derive(Debug, Clone)]
pub enum ConnectionType {
    Sequential,
    Causal,
    SupportiveEvidence,
    Contradictory,
    Parallel,
    Alternative,
    Refinement,
    Elaboration,
}

#[derive(Debug)]
pub struct ReasoningExtractor {
    active_chains: HashMap<ChainId, ReasoningChain>,
    completed_chains: VecDeque<ReasoningChain>,
    next_chain_id: u64,
    next_step_id: u64,
    next_connection_id: u64,
    max_chain_length: usize,
    min_confidence_threshold: f32,
    history_capacity: usize,
}

#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    pub max_chain_length: usize,
    pub min_confidence_threshold: f32,
    pub history_capacity: usize,
    pub enable_alternative_paths: bool,
    pub enable_contradiction_detection: bool,
    pub enable_analogical_reasoning: bool,
}

#[derive(Debug, Clone)]
pub struct ReasoningAnalysis {
    pub primary_chain: Option<ReasoningChain>,
    pub alternative_chains: Vec<ReasoningChain>,
    pub confidence_distribution: HashMap<StepType, f32>,
    pub logical_gaps: Vec<LogicalGap>,
    pub contradiction_points: Vec<ContradictionPoint>,
    pub reasoning_quality: ReasoningQuality,
}

#[derive(Debug, Clone)]
pub struct LogicalGap {
    pub gap_id: u64,
    pub between_steps: (StepId, StepId),
    pub gap_type: GapType,
    pub severity: f32,
    pub suggested_bridge: Option<String>,
}

#[derive(Debug, Clone)]
pub enum GapType {
    MissingPremise,
    LogicalJump,
    EvidenceGap,
    ConceptualGap,
    CausalGap,
}

#[derive(Debug, Clone)]
pub struct ContradictionPoint {
    pub contradiction_id: u64,
    pub conflicting_steps: Vec<StepId>,
    pub contradiction_type: ContradictionType,
    pub severity: f32,
    pub resolution_suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ContradictionType {
    DirectContradiction,
    ImplicitContradiction,
    EvidenceConflict,
    TemporalInconsistency,
}

#[derive(Debug, Clone)]
pub struct ReasoningQuality {
    pub logical_coherence: f32,
    pub evidence_support: f32,
    pub step_completeness: f32,
    pub chain_connectivity: f32,
    pub overall_quality: f32,
}
```

### Step 2: Reasoning Extractor Implementation

```rust
impl ReasoningExtractor {
    pub fn new() -> Self {
        Self {
            active_chains: HashMap::new(),
            completed_chains: VecDeque::new(),
            next_chain_id: 1,
            next_step_id: 1,
            next_connection_id: 1,
            max_chain_length: 20,
            min_confidence_threshold: 0.1,
            history_capacity: 100,
        }
    }
    
    pub fn with_config(config: ExtractionConfig) -> Self {
        Self {
            active_chains: HashMap::new(),
            completed_chains: VecDeque::new(),
            next_chain_id: 1,
            next_step_id: 1,
            next_connection_id: 1,
            max_chain_length: config.max_chain_length,
            min_confidence_threshold: config.min_confidence_threshold,
            history_capacity: config.history_capacity,
        }
    }
    
    pub fn extract_reasoning_from_pathways(
        &mut self,
        pathways: &[ActivationPathway],
        query: &str,
        activation_data: &HashMap<NodeId, ActivationLevel>,
    ) -> Result<ReasoningAnalysis, ExtractionError> {
        // Start new reasoning chain
        let chain_id = self.start_reasoning_chain();
        
        // Extract initial activation step
        self.extract_initial_activation(chain_id, query, activation_data)?;
        
        // Process each pathway to extract reasoning steps
        for pathway in pathways {
            self.extract_pathway_reasoning(chain_id, pathway)?;
        }
        
        // Finalize and analyze the chain
        let completed_chain = self.finalize_reasoning_chain(chain_id)?;
        
        // Perform reasoning analysis
        let analysis = self.analyze_reasoning_chain(&completed_chain)?;
        
        Ok(analysis)
    }
    
    fn start_reasoning_chain(&mut self) -> ChainId {
        let chain_id = ChainId(self.next_chain_id);
        self.next_chain_id += 1;
        
        let chain = ReasoningChain {
            chain_id,
            steps: Vec::new(),
            connections: Vec::new(),
            source_pathways: Vec::new(),
            confidence_score: 0.0,
            completeness_score: 0.0,
            coherence_score: 0.0,
            start_time: Instant::now(),
            end_time: None,
        };
        
        self.active_chains.insert(chain_id, chain);
        chain_id
    }
    
    fn extract_initial_activation(
        &mut self,
        chain_id: ChainId,
        query: &str,
        activation_data: &HashMap<NodeId, ActivationLevel>,
    ) -> Result<(), ExtractionError> {
        let step = ReasoningStep {
            step_id: StepId(self.next_step_id),
            step_type: StepType::InitialActivation,
            premise: format!("Query: {}", query),
            conclusion: format!("Initial activation of {} nodes", activation_data.len()),
            evidence: vec![Evidence {
                source: "activation_system".to_string(),
                content: format!("Activated nodes: {:?}", activation_data.keys().collect::<Vec<_>>()),
                confidence: 1.0,
                relevance: 1.0,
                timestamp: Instant::now(),
            }],
            confidence: 1.0,
            activation_nodes: activation_data.keys().copied().collect(),
            logical_operation: LogicalOperation::DirectReference,
            timestamp: Instant::now(),
        };
        
        self.next_step_id += 1;
        
        if let Some(chain) = self.active_chains.get_mut(&chain_id) {
            chain.steps.push(step);
        }
        
        Ok(())
    }
    
    fn extract_pathway_reasoning(
        &mut self,
        chain_id: ChainId,
        pathway: &ActivationPathway,
    ) -> Result<(), ExtractionError> {
        // Analyze pathway segments to identify reasoning steps
        let mut previous_step_id = None;
        
        for (i, segment) in pathway.segments.iter().enumerate() {
            let step_type = self.classify_segment_reasoning(&segment, i, &pathway.segments)?;
            let step = self.create_reasoning_step_from_segment(segment, step_type, pathway)?;
            
            let step_id = step.step_id;
            
            // Add step to chain
            if let Some(chain) = self.active_chains.get_mut(&chain_id) {
                chain.steps.push(step);
                
                // Create connection to previous step
                if let Some(prev_id) = previous_step_id {
                    let connection = LogicalConnection {
                        connection_id: ConnectionId(self.next_connection_id),
                        source_step: prev_id,
                        target_step: step_id,
                        connection_type: ConnectionType::Sequential,
                        strength: segment.activation_transfer,
                        evidence_support: self.calculate_evidence_support(segment),
                        logical_validity: self.calculate_logical_validity(segment, pathway),
                    };
                    
                    self.next_connection_id += 1;
                    chain.connections.push(connection);
                }
            }
            
            previous_step_id = Some(step_id);
        }
        
        Ok(())
    }
    
    fn classify_segment_reasoning(
        &self,
        segment: &PathwaySegment,
        index: usize,
        all_segments: &[PathwaySegment],
    ) -> Result<StepType, ExtractionError> {
        // Analyze segment characteristics to determine reasoning type
        let activation_strength = segment.activation_transfer;
        let propagation_delay = segment.propagation_delay.as_secs_f32();
        
        // Strong activation with low delay suggests direct factual lookup
        if activation_strength > 0.8 && propagation_delay < 0.001 {
            return Ok(StepType::FactualLookup);
        }
        
        // Medium activation with moderate delay suggests inference
        if activation_strength > 0.5 && propagation_delay < 0.01 {
            return Ok(StepType::LogicalDeduction);
        }
        
        // Analyze position in chain for context
        match index {
            0 => Ok(StepType::EntityRecognition),
            1..=2 => Ok(StepType::RelationshipInference),
            _ => {
                // Later steps might be conceptual bridges or synthesis
                if self.is_bridging_step(segment, all_segments) {
                    Ok(StepType::ConceptualBridge)
                } else if index == all_segments.len() - 1 {
                    Ok(StepType::SynthesisStep)
                } else {
                    Ok(StepType::LogicalDeduction)
                }
            }
        }
    }
    
    fn is_bridging_step(&self, segment: &PathwaySegment, all_segments: &[PathwaySegment]) -> bool {
        // Heuristic: bridging steps often have moderate activation and connect distant concepts
        let avg_activation: f32 = all_segments.iter()
            .map(|s| s.activation_transfer)
            .sum::<f32>() / all_segments.len() as f32;
        
        segment.activation_transfer > avg_activation * 0.7 && 
        segment.activation_transfer < avg_activation * 1.3
    }
    
    fn create_reasoning_step_from_segment(
        &mut self,
        segment: &PathwaySegment,
        step_type: StepType,
        pathway: &ActivationPathway,
    ) -> Result<ReasoningStep, ExtractionError> {
        let step_id = StepId(self.next_step_id);
        self.next_step_id += 1;
        
        let (premise, conclusion, logical_op) = self.generate_step_description(segment, &step_type);
        
        let step = ReasoningStep {
            step_id,
            step_type,
            premise,
            conclusion,
            evidence: vec![Evidence {
                source: "pathway_analysis".to_string(),
                content: format!("Activation transfer: {:.3}, Edge weight: {:.3}", 
                                segment.activation_transfer, segment.edge_weight),
                confidence: segment.activation_transfer,
                relevance: segment.edge_weight,
                timestamp: segment.timestamp,
            }],
            confidence: segment.activation_transfer,
            activation_nodes: vec![segment.source_node, segment.target_node],
            logical_operation: logical_op,
            timestamp: segment.timestamp,
        };
        
        Ok(step)
    }
    
    fn generate_step_description(
        &self,
        segment: &PathwaySegment,
        step_type: &StepType,
    ) -> (String, String, LogicalOperation) {
        match step_type {
            StepType::EntityRecognition => (
                format!("Recognize entity at node {:?}", segment.source_node),
                format!("Entity identified with confidence {:.2}", segment.activation_transfer),
                LogicalOperation::DirectReference,
            ),
            StepType::RelationshipInference => (
                format!("Analyze relationship between {:?} and {:?}", segment.source_node, segment.target_node),
                format!("Relationship strength: {:.2}", segment.activation_transfer),
                LogicalOperation::Association,
            ),
            StepType::LogicalDeduction => (
                format!("Apply logical reasoning from {:?}", segment.source_node),
                format!("Derived conclusion at {:?}", segment.target_node),
                LogicalOperation::Implication,
            ),
            StepType::ConceptualBridge => (
                format!("Bridge concepts between {:?} and {:?}", segment.source_node, segment.target_node),
                format!("Conceptual connection established"),
                LogicalOperation::Similarity,
            ),
            StepType::FactualLookup => (
                format!("Retrieve factual information from {:?}", segment.source_node),
                format!("Fact retrieved with confidence {:.2}", segment.activation_transfer),
                LogicalOperation::DirectReference,
            ),
            StepType::SynthesisStep => (
                format!("Synthesize information at {:?}", segment.target_node),
                format!("Final synthesis complete"),
                LogicalOperation::Conjunction,
            ),
            _ => (
                format!("Process activation from {:?} to {:?}", segment.source_node, segment.target_node),
                format!("Activation processed"),
                LogicalOperation::DirectReference,
            ),
        }
    }
    
    fn calculate_evidence_support(&self, segment: &PathwaySegment) -> f32 {
        // Evidence support based on activation strength and edge weight
        (segment.activation_transfer * segment.edge_weight).min(1.0)
    }
    
    fn calculate_logical_validity(&self, segment: &PathwaySegment, pathway: &ActivationPathway) -> f32 {
        // Logical validity based on consistency with pathway efficiency
        let efficiency_factor = pathway.path_efficiency.unwrap_or(0.5);
        let strength_factor = segment.activation_transfer;
        let delay_factor = 1.0 - (segment.propagation_delay.as_secs_f32().min(1.0));
        
        (efficiency_factor * 0.4 + strength_factor * 0.4 + delay_factor * 0.2).min(1.0)
    }
    
    fn finalize_reasoning_chain(&mut self, chain_id: ChainId) -> Result<ReasoningChain, ExtractionError> {
        let mut chain = self.active_chains.remove(&chain_id)
            .ok_or(ExtractionError::ChainNotFound)?;
        
        chain.end_time = Some(Instant::now());
        
        // Calculate chain scores
        chain.confidence_score = self.calculate_chain_confidence(&chain);
        chain.completeness_score = self.calculate_chain_completeness(&chain);
        chain.coherence_score = self.calculate_chain_coherence(&chain);
        
        // Add to history
        self.completed_chains.push_back(chain.clone());
        if self.completed_chains.len() > self.history_capacity {
            self.completed_chains.pop_front();
        }
        
        Ok(chain)
    }
    
    fn calculate_chain_confidence(&self, chain: &ReasoningChain) -> f32 {
        if chain.steps.is_empty() {
            return 0.0;
        }
        
        let total_confidence: f32 = chain.steps.iter()
            .map(|step| step.confidence)
            .sum();
        
        total_confidence / chain.steps.len() as f32
    }
    
    fn calculate_chain_completeness(&self, chain: &ReasoningChain) -> f32 {
        if chain.steps.is_empty() {
            return 0.0;
        }
        
        // Completeness based on step coverage and connection density
        let step_coverage = (chain.steps.len() as f32 / self.max_chain_length as f32).min(1.0);
        
        let expected_connections = if chain.steps.len() > 1 {
            chain.steps.len() - 1
        } else {
            0
        };
        
        let connection_density = if expected_connections > 0 {
            (chain.connections.len() as f32 / expected_connections as f32).min(1.0)
        } else {
            1.0
        };
        
        (step_coverage * 0.6 + connection_density * 0.4).min(1.0)
    }
    
    fn calculate_chain_coherence(&self, chain: &ReasoningChain) -> f32 {
        if chain.connections.is_empty() {
            return 0.5; // Neutral coherence for single step
        }
        
        let avg_logical_validity: f32 = chain.connections.iter()
            .map(|conn| conn.logical_validity)
            .sum::<f32>() / chain.connections.len() as f32;
        
        let avg_strength: f32 = chain.connections.iter()
            .map(|conn| conn.strength)
            .sum::<f32>() / chain.connections.len() as f32;
        
        (avg_logical_validity * 0.7 + avg_strength * 0.3).min(1.0)
    }
    
    fn analyze_reasoning_chain(&self, chain: &ReasoningChain) -> Result<ReasoningAnalysis, ExtractionError> {
        let confidence_distribution = self.calculate_confidence_distribution(chain);
        let logical_gaps = self.detect_logical_gaps(chain);
        let contradiction_points = self.detect_contradictions(chain);
        let reasoning_quality = self.assess_reasoning_quality(chain);
        
        let analysis = ReasoningAnalysis {
            primary_chain: Some(chain.clone()),
            alternative_chains: vec![], // Would be populated with alternative reasoning paths
            confidence_distribution,
            logical_gaps,
            contradiction_points,
            reasoning_quality,
        };
        
        Ok(analysis)
    }
    
    fn calculate_confidence_distribution(&self, chain: &ReasoningChain) -> HashMap<StepType, f32> {
        let mut distribution = HashMap::new();
        let mut type_counts = HashMap::new();
        
        for step in &chain.steps {
            let current_conf = distribution.entry(step.step_type.clone()).or_insert(0.0);
            let current_count = type_counts.entry(step.step_type.clone()).or_insert(0);
            
            *current_conf += step.confidence;
            *current_count += 1;
        }
        
        // Average confidence per step type
        for (step_type, total_conf) in &mut distribution {
            if let Some(&count) = type_counts.get(step_type) {
                *total_conf /= count as f32;
            }
        }
        
        distribution
    }
    
    fn detect_logical_gaps(&self, chain: &ReasoningChain) -> Vec<LogicalGap> {
        let mut gaps = Vec::new();
        
        for i in 0..chain.steps.len().saturating_sub(1) {
            let current_step = &chain.steps[i];
            let next_step = &chain.steps[i + 1];
            
            // Check for large confidence drops
            if current_step.confidence - next_step.confidence > 0.4 {
                gaps.push(LogicalGap {
                    gap_id: gaps.len() as u64,
                    between_steps: (current_step.step_id, next_step.step_id),
                    gap_type: GapType::LogicalJump,
                    severity: current_step.confidence - next_step.confidence,
                    suggested_bridge: Some(format!("Need intermediate reasoning step")),
                });
            }
            
            // Check for missing evidence
            if next_step.evidence.is_empty() && matches!(next_step.step_type, StepType::LogicalDeduction) {
                gaps.push(LogicalGap {
                    gap_id: gaps.len() as u64,
                    between_steps: (current_step.step_id, next_step.step_id),
                    gap_type: GapType::EvidenceGap,
                    severity: 0.6,
                    suggested_bridge: Some(format!("Need supporting evidence")),
                });
            }
        }
        
        gaps
    }
    
    fn detect_contradictions(&self, chain: &ReasoningChain) -> Vec<ContradictionPoint> {
        let mut contradictions = Vec::new();
        
        // Simple contradiction detection: conflicting conclusions
        for i in 0..chain.steps.len() {
            for j in i + 1..chain.steps.len() {
                let step1 = &chain.steps[i];
                let step2 = &chain.steps[j];
                
                // Check for directly contradictory statements
                if self.are_contradictory(&step1.conclusion, &step2.conclusion) {
                    contradictions.push(ContradictionPoint {
                        contradiction_id: contradictions.len() as u64,
                        conflicting_steps: vec![step1.step_id, step2.step_id],
                        contradiction_type: ContradictionType::DirectContradiction,
                        severity: (step1.confidence + step2.confidence) / 2.0,
                        resolution_suggestion: Some(format!("Resolve conflicting conclusions")),
                    });
                }
            }
        }
        
        contradictions
    }
    
    fn are_contradictory(&self, conclusion1: &str, conclusion2: &str) -> bool {
        // Simple heuristic for contradiction detection
        let negation_words = ["not", "no", "never", "false", "incorrect"];
        
        for word in negation_words {
            if conclusion1.contains(word) != conclusion2.contains(word) {
                // One contains negation, other doesn't - potential contradiction
                let base1 = conclusion1.replace(word, "").trim();
                let base2 = conclusion2.replace(word, "").trim();
                if base1.contains(base2) || base2.contains(base1) {
                    return true;
                }
            }
        }
        
        false
    }
    
    fn assess_reasoning_quality(&self, chain: &ReasoningChain) -> ReasoningQuality {
        let logical_coherence = chain.coherence_score;
        
        let evidence_support = if chain.steps.is_empty() {
            0.0
        } else {
            let total_evidence_quality: f32 = chain.steps.iter()
                .map(|step| {
                    if step.evidence.is_empty() {
                        0.0
                    } else {
                        step.evidence.iter()
                            .map(|ev| ev.confidence * ev.relevance)
                            .sum::<f32>() / step.evidence.len() as f32
                    }
                })
                .sum();
            
            total_evidence_quality / chain.steps.len() as f32
        };
        
        let step_completeness = chain.completeness_score;
        
        let chain_connectivity = if chain.connections.is_empty() {
            0.5
        } else {
            chain.connections.iter()
                .map(|conn| conn.strength * conn.logical_validity)
                .sum::<f32>() / chain.connections.len() as f32
        };
        
        let overall_quality = (logical_coherence * 0.3 + 
                              evidence_support * 0.3 + 
                              step_completeness * 0.2 + 
                              chain_connectivity * 0.2).min(1.0);
        
        ReasoningQuality {
            logical_coherence,
            evidence_support,
            step_completeness,
            chain_connectivity,
            overall_quality,
        }
    }
    
    pub fn get_reasoning_statistics(&self) -> ReasoningStatistics {
        let completed_count = self.completed_chains.len();
        let active_count = self.active_chains.len();
        
        let avg_chain_length = if completed_count > 0 {
            self.completed_chains.iter()
                .map(|chain| chain.steps.len())
                .sum::<usize>() as f32 / completed_count as f32
        } else {
            0.0
        };
        
        let avg_confidence = if completed_count > 0 {
            self.completed_chains.iter()
                .map(|chain| chain.confidence_score)
                .sum::<f32>() / completed_count as f32
        } else {
            0.0
        };
        
        ReasoningStatistics {
            active_chains: active_count,
            completed_chains: completed_count,
            average_chain_length: avg_chain_length,
            average_confidence: avg_confidence,
        }
    }
    
    pub fn clear_history(&mut self) {
        self.completed_chains.clear();
    }
}

#[derive(Debug, Clone)]
pub struct ReasoningStatistics {
    pub active_chains: usize,
    pub completed_chains: usize,
    pub average_chain_length: f32,
    pub average_confidence: f32,
}

#[derive(Debug, Clone)]
pub enum ExtractionError {
    ChainNotFound,
    InvalidPathway,
    InsufficientData,
    ProcessingError(String),
}

impl std::fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractionError::ChainNotFound => write!(f, "Reasoning chain not found"),
            ExtractionError::InvalidPathway => write!(f, "Invalid pathway data"),
            ExtractionError::InsufficientData => write!(f, "Insufficient data for reasoning extraction"),
            ExtractionError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl std::error::Error for ExtractionError {}
```

### Step 3: Integration with Template System

```rust
// File: src/cognitive/explanation/mod.rs

pub mod templates;
pub mod reasoning_extraction;

pub use templates::*;
pub use reasoning_extraction::*;

use crate::cognitive::learning::pathway_tracing::ActivationPathway;
use crate::core::types::{NodeId, ActivationLevel};

pub struct ReasoningExplanationGenerator {
    template_renderer: TemplateRenderer,
    reasoning_extractor: ReasoningExtractor,
}

impl ReasoningExplanationGenerator {
    pub fn new() -> Self {
        Self {
            template_renderer: TemplateRenderer::new(),
            reasoning_extractor: ReasoningExtractor::new(),
        }
    }
    
    pub fn generate_reasoning_explanation(
        &mut self,
        query: &str,
        pathways: &[ActivationPathway],
        activation_data: &HashMap<NodeId, ActivationLevel>,
    ) -> Result<String, ExtractionError> {
        // Extract reasoning chain
        let reasoning_analysis = self.reasoning_extractor
            .extract_reasoning_from_pathways(pathways, query, activation_data)?;
        
        if let Some(primary_chain) = &reasoning_analysis.primary_chain {
            // Convert reasoning chain to explanation context
            let context = self.create_explanation_context(
                query,
                primary_chain,
                &reasoning_analysis,
                activation_data,
            );
            
            // Generate explanation using reasoning chain template
            match self.template_renderer.render_explanation(&TemplateCategory::ReasoningChain, &context) {
                Ok(explanation) => Ok(explanation),
                Err(e) => {
                    // Fallback to simpler explanation
                    Ok(format!("Reasoning chain extracted with {} steps and quality score: {:.2}", 
                              primary_chain.steps.len(),
                              reasoning_analysis.reasoning_quality.overall_quality))
                }
            }
        } else {
            Ok("Unable to extract clear reasoning chain from the activation pathways.".to_string())
        }
    }
    
    fn create_explanation_context(
        &self,
        query: &str,
        chain: &ReasoningChain,
        analysis: &ReasoningAnalysis,
        activation_data: &HashMap<NodeId, ActivationLevel>,
    ) -> ExplanationContext {
        let mut metadata = HashMap::new();
        metadata.insert("chain_length".to_string(), chain.steps.len().to_string());
        metadata.insert("coherence_score".to_string(), format!("{:.2}", chain.coherence_score));
        metadata.insert("overall_quality".to_string(), format!("{:.2}", analysis.reasoning_quality.overall_quality));
        
        // Convert reasoning steps to evidence
        let evidence: Vec<Evidence> = chain.steps.iter()
            .flat_map(|step| step.evidence.clone())
            .collect();
        
        ExplanationContext {
            query: query.to_string(),
            query_type: "reasoning".to_string(),
            activation_data: activation_data.clone(),
            pathways: vec![], // Would include original pathways
            entities: chain.steps.iter()
                .flat_map(|step| step.activation_nodes.iter())
                .map(|&node_id| EntityId(node_id.0 as u64))
                .collect(),
            evidence,
            confidence: chain.confidence_score,
            processing_time: 0.0,
            metadata,
        }
    }
    
    pub fn get_reasoning_quality_report(&self, chain: &ReasoningChain) -> String {
        let analysis = self.reasoning_extractor.analyze_reasoning_chain(chain)
            .unwrap_or_else(|_| ReasoningAnalysis {
                primary_chain: None,
                alternative_chains: vec![],
                confidence_distribution: HashMap::new(),
                logical_gaps: vec![],
                contradiction_points: vec![],
                reasoning_quality: ReasoningQuality {
                    logical_coherence: 0.0,
                    evidence_support: 0.0,
                    step_completeness: 0.0,
                    chain_connectivity: 0.0,
                    overall_quality: 0.0,
                },
            });
        
        format!(
            "Reasoning Quality Report:\n\
             - Logical Coherence: {:.2}\n\
             - Evidence Support: {:.2}\n\
             - Step Completeness: {:.2}\n\
             - Chain Connectivity: {:.2}\n\
             - Overall Quality: {:.2}\n\
             - Logical Gaps: {}\n\
             - Contradictions: {}",
            analysis.reasoning_quality.logical_coherence,
            analysis.reasoning_quality.evidence_support,
            analysis.reasoning_quality.step_completeness,
            analysis.reasoning_quality.chain_connectivity,
            analysis.reasoning_quality.overall_quality,
            analysis.logical_gaps.len(),
            analysis.contradiction_points.len()
        )
    }
}
```

### Step 4: Reasoning Visualization Support

```rust
// File: src/cognitive/explanation/reasoning_visualization.rs

use serde::{Deserialize, Serialize};
use crate::cognitive::explanation::reasoning_extraction::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningGraph {
    pub nodes: Vec<ReasoningNode>,
    pub edges: Vec<ReasoningEdge>,
    pub layout_hints: LayoutHints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningNode {
    pub id: String,
    pub step_id: StepId,
    pub label: String,
    pub node_type: String,
    pub confidence: f32,
    pub position: Option<Position>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningEdge {
    pub source: String,
    pub target: String,
    pub connection_type: String,
    pub strength: f32,
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutHints {
    pub flow_direction: FlowDirection,
    pub clustering_enabled: bool,
    pub confidence_coloring: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowDirection {
    TopToBottom,
    LeftToRight,
    Circular,
}

pub fn create_reasoning_graph(chain: &ReasoningChain) -> ReasoningGraph {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    
    // Create nodes for each reasoning step
    for (i, step) in chain.steps.iter().enumerate() {
        let mut metadata = HashMap::new();
        metadata.insert("step_type".to_string(), format!("{:?}", step.step_type));
        metadata.insert("logical_operation".to_string(), format!("{:?}", step.logical_operation));
        
        let node = ReasoningNode {
            id: format!("step_{}", step.step_id.0),
            step_id: step.step_id,
            label: format!("{}: {}", step.premise, step.conclusion),
            node_type: format!("{:?}", step.step_type),
            confidence: step.confidence,
            position: Some(Position {
                x: i as f32 * 100.0,
                y: 0.0,
            }),
            metadata,
        };
        
        nodes.push(node);
    }
    
    // Create edges for logical connections
    for connection in &chain.connections {
        let edge = ReasoningEdge {
            source: format!("step_{}", connection.source_step.0),
            target: format!("step_{}", connection.target_step.0),
            connection_type: format!("{:?}", connection.connection_type),
            strength: connection.strength,
            label: Some(format!("{:.2}", connection.logical_validity)),
        };
        
        edges.push(edge);
    }
    
    ReasoningGraph {
        nodes,
        edges,
        layout_hints: LayoutHints {
            flow_direction: FlowDirection::LeftToRight,
            clustering_enabled: true,
            confidence_coloring: true,
        },
    }
}
```

## File Locations

- `src/cognitive/explanation/reasoning_extraction.rs` - Main implementation
- `src/cognitive/explanation/reasoning_visualization.rs` - Visualization support
- `src/cognitive/explanation/mod.rs` - Module exports and integration
- `tests/cognitive/explanation/reasoning_extraction_tests.rs` - Test implementation

## Success Criteria

- [ ] ReasoningExtractor analyzes pathways correctly
- [ ] Logical reasoning steps extracted accurately
- [ ] Reasoning chains constructed properly
- [ ] Contradiction and gap detection functional
- [ ] Quality assessment provides meaningful scores
- [ ] Integration with template system works
- [ ] All tests pass:
  - Pathway to reasoning conversion
  - Chain construction and analysis
  - Quality assessment accuracy
  - Error handling and edge cases

## Test Requirements

```rust
#[test]
fn test_basic_reasoning_extraction() {
    let mut extractor = ReasoningExtractor::new();
    
    // Create mock pathway
    let pathway = ActivationPathway {
        pathway_id: PathwayId(1),
        segments: vec![
            PathwaySegment {
                source_node: NodeId(1),
                target_node: NodeId(2),
                activation_transfer: 0.8,
                timestamp: Instant::now(),
                propagation_delay: Duration::from_micros(100),
                edge_weight: 1.0,
            },
            PathwaySegment {
                source_node: NodeId(2),
                target_node: NodeId(3),
                activation_transfer: 0.6,
                timestamp: Instant::now(),
                propagation_delay: Duration::from_micros(150),
                edge_weight: 0.9,
            },
        ],
        source_query: "test query".to_string(),
        start_time: Instant::now(),
        end_time: Some(Instant::now()),
        total_activation: 1.4,
        path_efficiency: Some(0.75),
        significance_score: 0.8,
    };
    
    let activation_data = HashMap::from([
        (NodeId(1), 0.9),
        (NodeId(2), 0.7),
        (NodeId(3), 0.5),
    ]);
    
    let analysis = extractor.extract_reasoning_from_pathways(
        &[pathway],
        "test query",
        &activation_data,
    ).unwrap();
    
    assert!(analysis.primary_chain.is_some());
    let chain = analysis.primary_chain.unwrap();
    assert!(chain.steps.len() >= 2); // At least initial + pathway steps
    assert!(!chain.connections.is_empty());
    assert!(chain.confidence_score > 0.0);
}

#[test]
fn test_step_type_classification() {
    let extractor = ReasoningExtractor::new();
    
    // High activation, low delay -> FactualLookup
    let factual_segment = PathwaySegment {
        source_node: NodeId(1),
        target_node: NodeId(2),
        activation_transfer: 0.9,
        timestamp: Instant::now(),
        propagation_delay: Duration::from_nanos(500),
        edge_weight: 1.0,
    };
    
    let step_type = extractor.classify_segment_reasoning(&factual_segment, 0, &[factual_segment.clone()]).unwrap();
    assert!(matches!(step_type, StepType::EntityRecognition | StepType::FactualLookup));
    
    // Medium activation, moderate delay -> LogicalDeduction
    let deduction_segment = PathwaySegment {
        source_node: NodeId(2),
        target_node: NodeId(3),
        activation_transfer: 0.6,
        timestamp: Instant::now(),
        propagation_delay: Duration::from_micros(5000),
        edge_weight: 0.8,
    };
    
    let step_type = extractor.classify_segment_reasoning(&deduction_segment, 1, &[factual_segment, deduction_segment.clone()]).unwrap();
    assert!(matches!(step_type, StepType::RelationshipInference | StepType::LogicalDeduction));
}

#[test]
fn test_logical_gap_detection() {
    let mut extractor = ReasoningExtractor::new();
    
    // Create chain with confidence drop
    let chain_id = extractor.start_reasoning_chain();
    
    let high_conf_step = ReasoningStep {
        step_id: StepId(1),
        step_type: StepType::FactualLookup,
        premise: "High confidence premise".to_string(),
        conclusion: "Strong conclusion".to_string(),
        evidence: vec![],
        confidence: 0.9,
        activation_nodes: vec![NodeId(1)],
        logical_operation: LogicalOperation::DirectReference,
        timestamp: Instant::now(),
    };
    
    let low_conf_step = ReasoningStep {
        step_id: StepId(2),
        step_type: StepType::LogicalDeduction,
        premise: "Weak premise".to_string(),
        conclusion: "Uncertain conclusion".to_string(),
        evidence: vec![],
        confidence: 0.3,
        activation_nodes: vec![NodeId(2)],
        logical_operation: LogicalOperation::Implication,
        timestamp: Instant::now(),
    };
    
    if let Some(chain) = extractor.active_chains.get_mut(&chain_id) {
        chain.steps = vec![high_conf_step, low_conf_step];
    }
    
    let completed_chain = extractor.finalize_reasoning_chain(chain_id).unwrap();
    let gaps = extractor.detect_logical_gaps(&completed_chain);
    
    assert!(!gaps.is_empty());
    assert!(matches!(gaps[0].gap_type, GapType::LogicalJump));
    assert!(gaps[0].severity > 0.4);
}

#[test]
fn test_reasoning_quality_assessment() {
    let extractor = ReasoningExtractor::new();
    
    let chain = ReasoningChain {
        chain_id: ChainId(1),
        steps: vec![
            ReasoningStep {
                step_id: StepId(1),
                step_type: StepType::EntityRecognition,
                premise: "premise".to_string(),
                conclusion: "conclusion".to_string(),
                evidence: vec![Evidence {
                    source: "test".to_string(),
                    content: "content".to_string(),
                    confidence: 0.8,
                    relevance: 0.9,
                    timestamp: Instant::now(),
                }],
                confidence: 0.8,
                activation_nodes: vec![NodeId(1)],
                logical_operation: LogicalOperation::DirectReference,
                timestamp: Instant::now(),
            },
        ],
        connections: vec![],
        source_pathways: vec![],
        confidence_score: 0.8,
        completeness_score: 0.7,
        coherence_score: 0.75,
        start_time: Instant::now(),
        end_time: Some(Instant::now()),
    };
    
    let quality = extractor.assess_reasoning_quality(&chain);
    
    assert!(quality.overall_quality > 0.0);
    assert!(quality.evidence_support > 0.0);
    assert!(quality.logical_coherence == chain.coherence_score);
}

#[test]
fn test_contradiction_detection() {
    let extractor = ReasoningExtractor::new();
    
    let chain = ReasoningChain {
        chain_id: ChainId(1),
        steps: vec![
            ReasoningStep {
                step_id: StepId(1),
                step_type: StepType::FactualLookup,
                premise: "X is true".to_string(),
                conclusion: "X is correct".to_string(),
                evidence: vec![],
                confidence: 0.8,
                activation_nodes: vec![NodeId(1)],
                logical_operation: LogicalOperation::DirectReference,
                timestamp: Instant::now(),
            },
            ReasoningStep {
                step_id: StepId(2),
                step_type: StepType::LogicalDeduction,
                premise: "Y analysis".to_string(),
                conclusion: "X is not correct".to_string(),
                evidence: vec![],
                confidence: 0.7,
                activation_nodes: vec![NodeId(2)],
                logical_operation: LogicalOperation::Implication,
                timestamp: Instant::now(),
            },
        ],
        connections: vec![],
        source_pathways: vec![],
        confidence_score: 0.75,
        completeness_score: 1.0,
        coherence_score: 0.5,
        start_time: Instant::now(),
        end_time: Some(Instant::now()),
    };
    
    let contradictions = extractor.detect_contradictions(&chain);
    
    assert!(!contradictions.is_empty());
    assert!(matches!(contradictions[0].contradiction_type, ContradictionType::DirectContradiction));
}
```

## Quality Gates

- [ ] Reasoning extraction < 10ms per query
- [ ] Memory usage < 20MB for complex reasoning chains
- [ ] Accurate step classification (>80% accuracy)
- [ ] Meaningful quality scores correlation with human judgment
- [ ] Gap and contradiction detection sensitivity
- [ ] Thread-safe concurrent reasoning analysis

## Next Task

Upon completion, proceed to **27_llm_explanation.md**