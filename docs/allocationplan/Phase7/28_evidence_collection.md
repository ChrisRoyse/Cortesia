# Micro Task 28: Evidence Collection

**Priority**: CRITICAL  
**Estimated Time**: 35 minutes  
**Dependencies**: 27_llm_explanation.md completed  
**Skills Required**: Knowledge graph traversal, evidence scoring, information retrieval

## Objective

Implement evidence collection system that gathers supporting information, facts, and contextual data during the reasoning process to provide substantiated explanations with verifiable evidence trails.

## Context

Effective explanations require supporting evidence that users can verify and understand. This task creates a system to collect, score, and organize evidence from the knowledge graph during activation spreading and reasoning extraction phases.

## Specifications

### Core Evidence Components

1. **EvidenceCollector struct**
   - Real-time evidence gathering
   - Multi-source evidence integration
   - Evidence quality assessment
   - Contextual relevance scoring

2. **Evidence struct**
   - Source identification
   - Content and metadata
   - Quality metrics
   - Relevance assessment

3. **EvidenceIndex struct**
   - Fast evidence lookup
   - Evidence categorization
   - Source reliability tracking
   - Update and versioning

4. **EvidenceValidator struct**
   - Quality assessment
   - Consistency checking
   - Contradiction detection
   - Confidence calculation

### Performance Requirements

- Evidence collection < 20ms per query
- Real-time evidence scoring
- Memory efficient evidence storage
- Concurrent evidence gathering
- Incremental evidence updates

## Implementation Guide

### Step 1: Core Evidence Types

```rust
// File: src/cognitive/explanation/evidence_collection.rs

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant, SystemTime};
use serde::{Deserialize, Serialize};
use crate::core::types::{NodeId, EntityId, ActivationLevel};
use crate::cognitive::learning::pathway_tracing::{ActivationPathway, PathwaySegment};
use crate::cognitive::explanation::reasoning_extraction::{ReasoningStep, ReasoningChain};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub evidence_id: EvidenceId,
    pub source: EvidenceSource,
    pub content: EvidenceContent,
    pub evidence_type: EvidenceType,
    pub quality_metrics: EvidenceQuality,
    pub relevance_score: f32,
    pub confidence: f32,
    pub timestamp: Instant,
    pub context: EvidenceContext,
    pub relationships: Vec<EvidenceRelationship>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSource {
    pub source_type: SourceType,
    pub source_identifier: String,
    pub source_reliability: f32,
    pub last_updated: SystemTime,
    pub access_count: u32,
    pub verification_status: VerificationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceType {
    KnowledgeGraph,
    ActivationTrace,
    PathwayAnalysis,
    ReasoningStep,
    ExternalDatabase,
    UserProvided,
    Inferred,
    Cached,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationStatus {
    Verified,
    Unverified,
    Disputed,
    Outdated,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceContent {
    pub primary_text: String,
    pub supporting_data: HashMap<String, String>,
    pub structured_data: Option<StructuredData>,
    pub multimedia_refs: Vec<MultimediaReference>,
    pub citations: Vec<Citation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredData {
    pub data_type: String,
    pub schema_version: String,
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimediaReference {
    pub media_type: MediaType,
    pub reference: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediaType {
    Image,
    Video,
    Audio,
    Document,
    Chart,
    Diagram,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub citation_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub publication_date: Option<SystemTime>,
    pub source_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    FactualClaim,
    StatisticalData,
    ExpertOpinion,
    HistoricalRecord,
    ScientificStudy,
    LogicalInference,
    Observational,
    Experimental,
    Testimonial,
    Documentation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceQuality {
    pub accuracy: f32,
    pub completeness: f32,
    pub timeliness: f32,
    pub objectivity: f32,
    pub source_credibility: f32,
    pub verification_level: f32,
    pub overall_quality: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceContext {
    pub query_context: String,
    pub reasoning_step_id: Option<StepId>,
    pub pathway_segment: Option<PathwaySegmentRef>,
    pub activation_nodes: Vec<NodeId>,
    pub related_entities: Vec<EntityId>,
    pub domain_context: String,
    pub temporal_context: Option<TemporalContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwaySegmentRef {
    pub pathway_id: u64,
    pub segment_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub time_range: Option<(SystemTime, SystemTime)>,
    pub temporal_relevance: f32,
    pub temporal_specificity: TemporalSpecificity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalSpecificity {
    Exact,
    Approximate,
    Range,
    Ongoing,
    Historical,
    Projected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRelationship {
    pub relationship_type: RelationshipType,
    pub target_evidence: EvidenceId,
    pub strength: f32,
    pub relationship_context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    Supports,
    Contradicts,
    Elaborates,
    Exemplifies,
    Contextualizes,
    Updates,
    Supersedes,
    References,
}

use crate::cognitive::explanation::reasoning_extraction::StepId;

#[derive(Debug)]
pub struct EvidenceCollector {
    evidence_index: EvidenceIndex,
    active_collections: HashMap<CollectionId, ActiveCollection>,
    validator: EvidenceValidator,
    next_evidence_id: u64,
    next_collection_id: u64,
    config: CollectionConfig,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct CollectionId(pub u64);

#[derive(Debug)]
pub struct ActiveCollection {
    pub collection_id: CollectionId,
    pub query: String,
    pub evidence_items: Vec<EvidenceId>,
    pub collection_strategy: CollectionStrategy,
    pub start_time: Instant,
    pub quality_threshold: f32,
    pub max_evidence_count: usize,
}

#[derive(Debug, Clone)]
pub enum CollectionStrategy {
    Breadth,   // Collect diverse evidence
    Depth,     // Collect detailed evidence
    Quality,   // Prioritize high-quality evidence
    Relevance, // Prioritize most relevant evidence
    Balanced,  // Balance multiple factors
}

#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub max_evidence_per_query: usize,
    pub quality_threshold: f32,
    pub relevance_threshold: f32,
    pub collection_timeout: Duration,
    pub enable_contradiction_detection: bool,
    pub enable_source_verification: bool,
    pub max_depth: usize,
}

#[derive(Debug)]
pub struct EvidenceIndex {
    by_source: HashMap<String, Vec<EvidenceId>>,
    by_type: HashMap<EvidenceType, Vec<EvidenceId>>,
    by_entity: HashMap<EntityId, Vec<EvidenceId>>,
    by_quality: Vec<(f32, EvidenceId)>, // Sorted by quality score
    evidence_store: HashMap<EvidenceId, Evidence>,
    source_reliability: HashMap<String, SourceReliability>,
}

#[derive(Debug, Clone)]
pub struct SourceReliability {
    pub overall_score: f32,
    pub accuracy_history: VecDeque<f32>,
    pub verification_count: u32,
    pub dispute_count: u32,
    pub last_assessment: Instant,
}

#[derive(Debug)]
pub struct EvidenceValidator {
    validation_rules: Vec<ValidationRule>,
    consistency_checker: ConsistencyChecker,
    quality_assessor: QualityAssessor,
}

#[derive(Debug)]
pub struct ValidationRule {
    pub rule_id: String,
    pub rule_type: RuleType,
    pub condition: ValidationCondition,
    pub action: ValidationAction,
    pub severity: Severity,
}

#[derive(Debug)]
pub enum RuleType {
    Quality,
    Consistency,
    Relevance,
    Timeliness,
    Source,
}

#[derive(Debug)]
pub enum ValidationCondition {
    QualityBelow(f32),
    RelevanceBelow(f32),
    SourceUnreliable,
    ContentTooShort(usize),
    ContentTooLong(usize),
    TemporalMismatch,
    ConsistencyViolation,
}

#[derive(Debug)]
pub enum ValidationAction {
    Reject,
    Downgrade,
    Flag,
    RequireVerification,
    AddWarning,
}

#[derive(Debug)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug)]
pub struct ConsistencyChecker {
    contradiction_rules: Vec<ContradictionRule>,
    logical_constraints: Vec<LogicalConstraint>,
}

#[derive(Debug)]
pub struct ContradictionRule {
    pub pattern: String,
    pub contradiction_type: ContradictionType,
    pub confidence_threshold: f32,
}

#[derive(Debug)]
pub enum ContradictionType {
    Direct,
    Implicit,
    Temporal,
    Logical,
    Quantitative,
}

#[derive(Debug)]
pub struct LogicalConstraint {
    pub constraint_type: ConstraintType,
    pub entities: Vec<EntityId>,
    pub rule_expression: String,
}

#[derive(Debug)]
pub enum ConstraintType {
    Mutual Exclusion,
    RequiredCoexistence,
    TemporalOrdering,
    CausalRelation,
    QuantitativeRelation,
}

#[derive(Debug)]
pub struct QualityAssessor {
    quality_metrics: Vec<QualityMetric>,
    scoring_weights: HashMap<String, f32>,
}

#[derive(Debug)]
pub struct QualityMetric {
    pub metric_name: String,
    pub metric_type: MetricType,
    pub weight: f32,
    pub assessment_function: fn(&Evidence) -> f32,
}

#[derive(Debug)]
pub enum MetricType {
    Intrinsic,    // Based on content itself
    Extrinsic,    // Based on external factors
    Contextual,   // Based on usage context
    Temporal,     // Based on time factors
}
```

### Step 2: Evidence Collector Implementation

```rust
impl EvidenceCollector {
    pub fn new() -> Self {
        let config = CollectionConfig {
            max_evidence_per_query: 50,
            quality_threshold: 0.3,
            relevance_threshold: 0.2,
            collection_timeout: Duration::from_millis(100),
            enable_contradiction_detection: true,
            enable_source_verification: true,
            max_depth: 5,
        };
        
        Self {
            evidence_index: EvidenceIndex::new(),
            active_collections: HashMap::new(),
            validator: EvidenceValidator::new(),
            next_evidence_id: 1,
            next_collection_id: 1,
            config,
        }
    }
    
    pub fn with_config(config: CollectionConfig) -> Self {
        Self {
            evidence_index: EvidenceIndex::new(),
            active_collections: HashMap::new(),
            validator: EvidenceValidator::new(),
            next_evidence_id: 1,
            next_collection_id: 1,
            config,
        }
    }
    
    pub fn start_collection(
        &mut self,
        query: &str,
        strategy: CollectionStrategy,
    ) -> CollectionId {
        let collection_id = CollectionId(self.next_collection_id);
        self.next_collection_id += 1;
        
        let collection = ActiveCollection {
            collection_id,
            query: query.to_string(),
            evidence_items: Vec::new(),
            collection_strategy: strategy,
            start_time: Instant::now(),
            quality_threshold: self.config.quality_threshold,
            max_evidence_count: self.config.max_evidence_per_query,
        };
        
        self.active_collections.insert(collection_id, collection);
        collection_id
    }
    
    pub fn collect_from_activation_pathway(
        &mut self,
        collection_id: CollectionId,
        pathway: &ActivationPathway,
    ) -> Result<Vec<EvidenceId>, CollectionError> {
        let mut collected_evidence = Vec::new();
        
        for (index, segment) in pathway.segments.iter().enumerate() {
            // Extract evidence from each pathway segment
            if let Ok(evidence) = self.extract_pathway_evidence(segment, index, pathway, collection_id) {
                let evidence_id = self.add_evidence(evidence)?;
                collected_evidence.push(evidence_id);
                
                // Update active collection
                if let Some(collection) = self.active_collections.get_mut(&collection_id) {
                    collection.evidence_items.push(evidence_id);
                    
                    // Check collection limits
                    if collection.evidence_items.len() >= collection.max_evidence_count {
                        break;
                    }
                }
            }
        }
        
        Ok(collected_evidence)
    }
    
    fn extract_pathway_evidence(
        &self,
        segment: &PathwaySegment,
        index: usize,
        pathway: &ActivationPathway,
        collection_id: CollectionId,
    ) -> Result<Evidence, CollectionError> {
        let evidence_id = EvidenceId(self.next_evidence_id);
        
        let source = EvidenceSource {
            source_type: SourceType::PathwayAnalysis,
            source_identifier: format!("pathway_{}_segment_{}", pathway.pathway_id.0, index),
            source_reliability: self.calculate_pathway_reliability(pathway),
            last_updated: SystemTime::now(),
            access_count: 1,
            verification_status: VerificationStatus::Verified,
        };
        
        let content = EvidenceContent {
            primary_text: format!(
                "Activation propagated from node {:?} to node {:?} with strength {:.3}",
                segment.source_node,
                segment.target_node,
                segment.activation_transfer
            ),
            supporting_data: HashMap::from([
                ("edge_weight".to_string(), segment.edge_weight.to_string()),
                ("propagation_delay".to_string(), format!("{:?}", segment.propagation_delay)),
                ("pathway_efficiency".to_string(), pathway.path_efficiency.unwrap_or(0.0).to_string()),
            ]),
            structured_data: Some(StructuredData {
                data_type: "pathway_segment".to_string(),
                schema_version: "1.0".to_string(),
                fields: HashMap::from([
                    ("source_node".to_string(), serde_json::Value::Number(serde_json::Number::from(segment.source_node.0))),
                    ("target_node".to_string(), serde_json::Value::Number(serde_json::Number::from(segment.target_node.0))),
                    ("activation_transfer".to_string(), serde_json::Value::Number(
                        serde_json::Number::from_f64(segment.activation_transfer as f64).unwrap()
                    )),
                ]),
            }),
            multimedia_refs: vec![],
            citations: vec![],
        };
        
        let quality_metrics = EvidenceQuality {
            accuracy: segment.activation_transfer, // Higher activation = more accurate
            completeness: if segment.edge_weight > 0.5 { 0.9 } else { 0.6 },
            timeliness: 1.0, // Real-time data
            objectivity: 1.0, // System-generated data
            source_credibility: self.calculate_pathway_reliability(pathway),
            verification_level: 0.9, // System-verified
            overall_quality: 0.0, // Will be calculated
        };
        
        let context = EvidenceContext {
            query_context: self.active_collections.get(&collection_id)
                .map(|c| c.query.clone())
                .unwrap_or_default(),
            reasoning_step_id: None,
            pathway_segment: Some(PathwaySegmentRef {
                pathway_id: pathway.pathway_id.0,
                segment_index: index,
            }),
            activation_nodes: vec![segment.source_node, segment.target_node],
            related_entities: vec![], // Would be populated from graph structure
            domain_context: "neural_activation".to_string(),
            temporal_context: Some(TemporalContext {
                time_range: None,
                temporal_relevance: 1.0,
                temporal_specificity: TemporalSpecificity::Exact,
            }),
        };
        
        let mut evidence = Evidence {
            evidence_id,
            source,
            content,
            evidence_type: EvidenceType::Observational,
            quality_metrics,
            relevance_score: self.calculate_pathway_relevance(segment, pathway),
            confidence: segment.activation_transfer,
            timestamp: segment.timestamp,
            context,
            relationships: vec![],
        };
        
        // Calculate overall quality
        evidence.quality_metrics.overall_quality = self.calculate_overall_quality(&evidence.quality_metrics);
        
        Ok(evidence)
    }
    
    pub fn collect_from_reasoning_step(
        &mut self,
        collection_id: CollectionId,
        step: &ReasoningStep,
    ) -> Result<Vec<EvidenceId>, CollectionError> {
        let mut collected_evidence = Vec::new();
        
        // Convert reasoning step to evidence
        let evidence = self.extract_reasoning_evidence(step, collection_id)?;
        let evidence_id = self.add_evidence(evidence)?;
        collected_evidence.push(evidence_id);
        
        // Extract evidence from step's existing evidence
        for existing_evidence in &step.evidence {
            let enhanced_evidence = self.enhance_step_evidence(existing_evidence, step, collection_id)?;
            let evidence_id = self.add_evidence(enhanced_evidence)?;
            collected_evidence.push(evidence_id);
        }
        
        // Update active collection
        if let Some(collection) = self.active_collections.get_mut(&collection_id) {
            collection.evidence_items.extend(&collected_evidence);
        }
        
        Ok(collected_evidence)
    }
    
    fn extract_reasoning_evidence(
        &self,
        step: &ReasoningStep,
        collection_id: CollectionId,
    ) -> Result<Evidence, CollectionError> {
        let evidence_id = EvidenceId(self.next_evidence_id);
        
        let source = EvidenceSource {
            source_type: SourceType::ReasoningStep,
            source_identifier: format!("reasoning_step_{}", step.step_id.0),
            source_reliability: step.confidence,
            last_updated: SystemTime::now(),
            access_count: 1,
            verification_status: VerificationStatus::Verified,
        };
        
        let content = EvidenceContent {
            primary_text: format!("{} → {}", step.premise, step.conclusion),
            supporting_data: HashMap::from([
                ("step_type".to_string(), format!("{:?}", step.step_type)),
                ("logical_operation".to_string(), format!("{:?}", step.logical_operation)),
                ("confidence".to_string(), step.confidence.to_string()),
            ]),
            structured_data: Some(StructuredData {
                data_type: "reasoning_step".to_string(),
                schema_version: "1.0".to_string(),
                fields: HashMap::from([
                    ("step_id".to_string(), serde_json::Value::Number(serde_json::Number::from(step.step_id.0))),
                    ("premise".to_string(), serde_json::Value::String(step.premise.clone())),
                    ("conclusion".to_string(), serde_json::Value::String(step.conclusion.clone())),
                ]),
            }),
            multimedia_refs: vec![],
            citations: vec![],
        };
        
        let quality_metrics = EvidenceQuality {
            accuracy: step.confidence,
            completeness: if step.evidence.is_empty() { 0.5 } else { 0.9 },
            timeliness: 1.0,
            objectivity: self.assess_reasoning_objectivity(step),
            source_credibility: 0.9, // System-generated
            verification_level: step.confidence,
            overall_quality: 0.0,
        };
        
        let context = EvidenceContext {
            query_context: self.active_collections.get(&collection_id)
                .map(|c| c.query.clone())
                .unwrap_or_default(),
            reasoning_step_id: Some(step.step_id),
            pathway_segment: None,
            activation_nodes: step.activation_nodes.clone(),
            related_entities: step.activation_nodes.iter()
                .map(|&node_id| EntityId(node_id.0 as u64))
                .collect(),
            domain_context: "logical_reasoning".to_string(),
            temporal_context: Some(TemporalContext {
                time_range: None,
                temporal_relevance: 1.0,
                temporal_specificity: TemporalSpecificity::Exact,
            }),
        };
        
        let mut evidence = Evidence {
            evidence_id,
            source,
            content,
            evidence_type: EvidenceType::LogicalInference,
            quality_metrics,
            relevance_score: self.calculate_reasoning_relevance(step),
            confidence: step.confidence,
            timestamp: step.timestamp,
            context,
            relationships: vec![],
        };
        
        evidence.quality_metrics.overall_quality = self.calculate_overall_quality(&evidence.quality_metrics);
        
        Ok(evidence)
    }
    
    fn enhance_step_evidence(
        &self,
        original_evidence: &crate::cognitive::explanation::templates::Evidence,
        step: &ReasoningStep,
        collection_id: CollectionId,
    ) -> Result<Evidence, CollectionError> {
        let evidence_id = EvidenceId(self.next_evidence_id);
        
        let source = EvidenceSource {
            source_type: SourceType::KnowledgeGraph,
            source_identifier: original_evidence.source.clone(),
            source_reliability: original_evidence.confidence,
            last_updated: SystemTime::now(),
            access_count: 1,
            verification_status: VerificationStatus::Unverified,
        };
        
        let content = EvidenceContent {
            primary_text: original_evidence.content.clone(),
            supporting_data: HashMap::new(),
            structured_data: None,
            multimedia_refs: vec![],
            citations: vec![],
        };
        
        let quality_metrics = EvidenceQuality {
            accuracy: original_evidence.confidence,
            completeness: if original_evidence.content.len() > 50 { 0.8 } else { 0.5 },
            timeliness: self.assess_evidence_timeliness(original_evidence),
            objectivity: 0.7, // Default for external evidence
            source_credibility: original_evidence.relevance,
            verification_level: 0.5, // Needs verification
            overall_quality: 0.0,
        };
        
        let context = EvidenceContext {
            query_context: self.active_collections.get(&collection_id)
                .map(|c| c.query.clone())
                .unwrap_or_default(),
            reasoning_step_id: Some(step.step_id),
            pathway_segment: None,
            activation_nodes: step.activation_nodes.clone(),
            related_entities: vec![],
            domain_context: "knowledge_base".to_string(),
            temporal_context: None,
        };
        
        let mut evidence = Evidence {
            evidence_id,
            source,
            content,
            evidence_type: EvidenceType::FactualClaim,
            quality_metrics,
            relevance_score: original_evidence.relevance,
            confidence: original_evidence.confidence,
            timestamp: original_evidence.timestamp,
            context,
            relationships: vec![],
        };
        
        evidence.quality_metrics.overall_quality = self.calculate_overall_quality(&evidence.quality_metrics);
        
        Ok(evidence)
    }
    
    fn add_evidence(&mut self, evidence: Evidence) -> Result<EvidenceId, CollectionError> {
        // Validate evidence before adding
        let validation_result = self.validator.validate_evidence(&evidence);
        
        if !validation_result.is_valid {
            return Err(CollectionError::ValidationFailed(validation_result.issues));
        }
        
        let evidence_id = evidence.evidence_id;
        
        // Update next ID
        if evidence_id.0 >= self.next_evidence_id {
            self.next_evidence_id = evidence_id.0 + 1;
        }
        
        // Add to index
        self.evidence_index.add_evidence(evidence)?;
        
        Ok(evidence_id)
    }
    
    pub fn finalize_collection(
        &mut self,
        collection_id: CollectionId,
    ) -> Result<EvidenceCollection, CollectionError> {
        let collection = self.active_collections.remove(&collection_id)
            .ok_or(CollectionError::CollectionNotFound)?;
        
        // Retrieve all evidence for this collection
        let evidence_items: Result<Vec<Evidence>, CollectionError> = collection.evidence_items
            .iter()
            .map(|&evidence_id| {
                self.evidence_index.get_evidence(evidence_id)
                    .ok_or(CollectionError::EvidenceNotFound)
                    .map(|e| e.clone())
            })
            .collect();
        
        let evidence_items = evidence_items?;
        
        // Sort evidence by quality and relevance
        let mut sorted_evidence = evidence_items;
        sorted_evidence.sort_by(|a, b| {
            let score_a = a.quality_metrics.overall_quality * a.relevance_score;
            let score_b = b.quality_metrics.overall_quality * b.relevance_score;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Detect relationships between evidence items
        let relationships = self.detect_evidence_relationships(&sorted_evidence);
        
        let evidence_collection = EvidenceCollection {
            collection_id,
            query: collection.query,
            evidence_items: sorted_evidence,
            relationships,
            collection_strategy: collection.collection_strategy,
            quality_summary: self.calculate_collection_quality_summary(&collection.evidence_items),
            collection_time: collection.start_time.elapsed(),
            timestamp: Instant::now(),
        };
        
        Ok(evidence_collection)
    }
    
    fn detect_evidence_relationships(&self, evidence_items: &[Evidence]) -> Vec<EvidenceRelationship> {
        let mut relationships = Vec::new();
        
        for i in 0..evidence_items.len() {
            for j in i + 1..evidence_items.len() {
                let evidence_a = &evidence_items[i];
                let evidence_b = &evidence_items[j];
                
                // Check for supporting relationships
                if self.is_supporting_evidence(evidence_a, evidence_b) {
                    relationships.push(EvidenceRelationship {
                        relationship_type: RelationshipType::Supports,
                        target_evidence: evidence_b.evidence_id,
                        strength: self.calculate_support_strength(evidence_a, evidence_b),
                        relationship_context: "Provides supporting information".to_string(),
                    });
                }
                
                // Check for contradictions
                if self.is_contradictory_evidence(evidence_a, evidence_b) {
                    relationships.push(EvidenceRelationship {
                        relationship_type: RelationshipType::Contradicts,
                        target_evidence: evidence_b.evidence_id,
                        strength: self.calculate_contradiction_strength(evidence_a, evidence_b),
                        relationship_context: "Contains contradictory information".to_string(),
                    });
                }
            }
        }
        
        relationships
    }
    
    fn is_supporting_evidence(&self, evidence_a: &Evidence, evidence_b: &Evidence) -> bool {
        // Simple heuristic: evidence from same source or similar content
        evidence_a.source.source_identifier == evidence_b.source.source_identifier ||
        self.calculate_content_similarity(&evidence_a.content.primary_text, &evidence_b.content.primary_text) > 0.7
    }
    
    fn is_contradictory_evidence(&self, evidence_a: &Evidence, evidence_b: &Evidence) -> bool {
        // Simple contradiction detection
        let text_a = &evidence_a.content.primary_text;
        let text_b = &evidence_b.content.primary_text;
        
        // Look for negation patterns
        (text_a.contains("not") && text_b.contains(text_a.replace("not", "").trim())) ||
        (text_b.contains("not") && text_a.contains(text_b.replace("not", "").trim()))
    }
    
    fn calculate_content_similarity(&self, text_a: &str, text_b: &str) -> f32 {
        // Simple word overlap similarity
        let words_a: HashSet<&str> = text_a.split_whitespace().collect();
        let words_b: HashSet<&str> = text_b.split_whitespace().collect();
        
        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();
        
        if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        }
    }
    
    fn calculate_support_strength(&self, _evidence_a: &Evidence, _evidence_b: &Evidence) -> f32 {
        // Simplified support strength calculation
        0.7
    }
    
    fn calculate_contradiction_strength(&self, _evidence_a: &Evidence, _evidence_b: &Evidence) -> f32 {
        // Simplified contradiction strength calculation
        0.8
    }
    
    fn calculate_collection_quality_summary(&self, evidence_ids: &[EvidenceId]) -> QualitySummary {
        let evidence_items: Vec<_> = evidence_ids.iter()
            .filter_map(|&id| self.evidence_index.get_evidence(id))
            .collect();
        
        if evidence_items.is_empty() {
            return QualitySummary::default();
        }
        
        let avg_quality = evidence_items.iter()
            .map(|e| e.quality_metrics.overall_quality)
            .sum::<f32>() / evidence_items.len() as f32;
        
        let avg_relevance = evidence_items.iter()
            .map(|e| e.relevance_score)
            .sum::<f32>() / evidence_items.len() as f32;
        
        let avg_confidence = evidence_items.iter()
            .map(|e| e.confidence)
            .sum::<f32>() / evidence_items.len() as f32;
        
        QualitySummary {
            average_quality: avg_quality,
            average_relevance: avg_relevance,
            average_confidence: avg_confidence,
            total_evidence_count: evidence_items.len(),
            high_quality_count: evidence_items.iter()
                .filter(|e| e.quality_metrics.overall_quality > 0.7)
                .count(),
            verified_count: evidence_items.iter()
                .filter(|e| matches!(e.source.verification_status, VerificationStatus::Verified))
                .count(),
        }
    }
    
    // Helper calculation methods
    fn calculate_pathway_reliability(&self, pathway: &ActivationPathway) -> f32 {
        // Higher efficiency and significance = higher reliability
        let efficiency_factor = pathway.path_efficiency.unwrap_or(0.5);
        let significance_factor = pathway.significance_score;
        (efficiency_factor + significance_factor) / 2.0
    }
    
    fn calculate_pathway_relevance(&self, segment: &PathwaySegment, pathway: &ActivationPathway) -> f32 {
        // Relevance based on activation strength and pathway significance
        (segment.activation_transfer + pathway.significance_score) / 2.0
    }
    
    fn calculate_reasoning_relevance(&self, step: &ReasoningStep) -> f32 {
        // Higher confidence and more evidence = higher relevance
        let evidence_factor = if step.evidence.is_empty() { 0.3 } else { 0.9 };
        (step.confidence + evidence_factor) / 2.0
    }
    
    fn assess_reasoning_objectivity(&self, step: &ReasoningStep) -> f32 {
        // System-generated reasoning is generally objective
        match step.step_type {
            StepType::FactualLookup => 0.95,
            StepType::LogicalDeduction => 0.9,
            StepType::EntityRecognition => 0.85,
            _ => 0.8,
        }
    }
    
    fn assess_evidence_timeliness(&self, _evidence: &crate::cognitive::explanation::templates::Evidence) -> f32 {
        // Real-time evidence is timely
        let age = _evidence.timestamp.elapsed().as_secs();
        if age < 60 { 1.0 } else if age < 300 { 0.8 } else { 0.6 }
    }
    
    fn calculate_overall_quality(&self, metrics: &EvidenceQuality) -> f32 {
        // Weighted combination of quality metrics
        (metrics.accuracy * 0.25 +
         metrics.completeness * 0.2 +
         metrics.timeliness * 0.15 +
         metrics.objectivity * 0.15 +
         metrics.source_credibility * 0.15 +
         metrics.verification_level * 0.1).min(1.0)
    }
    
    pub fn get_collection_statistics(&self) -> CollectionStatistics {
        let total_evidence = self.evidence_index.evidence_store.len();
        let active_collections = self.active_collections.len();
        
        let avg_quality = if total_evidence > 0 {
            self.evidence_index.evidence_store.values()
                .map(|e| e.quality_metrics.overall_quality)
                .sum::<f32>() / total_evidence as f32
        } else {
            0.0
        };
        
        CollectionStatistics {
            total_evidence_items: total_evidence,
            active_collections,
            average_quality: avg_quality,
            verified_evidence_count: self.evidence_index.evidence_store.values()
                .filter(|e| matches!(e.source.verification_status, VerificationStatus::Verified))
                .count(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvidenceCollection {
    pub collection_id: CollectionId,
    pub query: String,
    pub evidence_items: Vec<Evidence>,
    pub relationships: Vec<EvidenceRelationship>,
    pub collection_strategy: CollectionStrategy,
    pub quality_summary: QualitySummary,
    pub collection_time: Duration,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Default)]
pub struct QualitySummary {
    pub average_quality: f32,
    pub average_relevance: f32,
    pub average_confidence: f32,
    pub total_evidence_count: usize,
    pub high_quality_count: usize,
    pub verified_count: usize,
}

#[derive(Debug, Clone)]
pub struct CollectionStatistics {
    pub total_evidence_items: usize,
    pub active_collections: usize,
    pub average_quality: f32,
    pub verified_evidence_count: usize,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub quality_score: f32,
    pub issues: Vec<ValidationIssue>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub issue_type: IssueType,
    pub severity: Severity,
    pub description: String,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone)]
pub enum IssueType {
    LowQuality,
    LowRelevance,
    UnreliableSource,
    IncompleteContent,
    TemporalIssue,
    ConsistencyViolation,
}

#[derive(Debug, Clone)]
pub enum CollectionError {
    CollectionNotFound,
    EvidenceNotFound,
    ValidationFailed(Vec<ValidationIssue>),
    IndexError(String),
    ProcessingTimeout,
    InsufficientQuality,
}

impl std::fmt::Display for CollectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollectionError::CollectionNotFound => write!(f, "Evidence collection not found"),
            CollectionError::EvidenceNotFound => write!(f, "Evidence item not found"),
            CollectionError::ValidationFailed(issues) => write!(f, "Validation failed: {} issues", issues.len()),
            CollectionError::IndexError(msg) => write!(f, "Index error: {}", msg),
            CollectionError::ProcessingTimeout => write!(f, "Evidence collection timeout"),
            CollectionError::InsufficientQuality => write!(f, "Evidence quality below threshold"),
        }
    }
}

impl std::error::Error for CollectionError {}
```

### Step 3: Evidence Index and Validator Implementation

```rust
impl EvidenceIndex {
    pub fn new() -> Self {
        Self {
            by_source: HashMap::new(),
            by_type: HashMap::new(),
            by_entity: HashMap::new(),
            by_quality: Vec::new(),
            evidence_store: HashMap::new(),
            source_reliability: HashMap::new(),
        }
    }
    
    pub fn add_evidence(&mut self, evidence: Evidence) -> Result<(), CollectionError> {
        let evidence_id = evidence.evidence_id;
        
        // Update source index
        self.by_source
            .entry(evidence.source.source_identifier.clone())
            .or_insert_with(Vec::new)
            .push(evidence_id);
        
        // Update type index
        self.by_type
            .entry(evidence.evidence_type.clone())
            .or_insert_with(Vec::new)
            .push(evidence_id);
        
        // Update entity index
        for entity_id in &evidence.context.related_entities {
            self.by_entity
                .entry(*entity_id)
                .or_insert_with(Vec::new)
                .push(evidence_id);
        }
        
        // Update quality index (maintain sorted order)
        let quality_score = evidence.quality_metrics.overall_quality;
        let insert_pos = self.by_quality.binary_search_by(|&(score, _)| {
            score.partial_cmp(&quality_score).unwrap_or(std::cmp::Ordering::Equal).reverse()
        }).unwrap_or_else(|pos| pos);
        
        self.by_quality.insert(insert_pos, (quality_score, evidence_id));
        
        // Update source reliability
        self.update_source_reliability(&evidence.source.source_identifier, quality_score);
        
        // Store evidence
        self.evidence_store.insert(evidence_id, evidence);
        
        Ok(())
    }
    
    pub fn get_evidence(&self, evidence_id: EvidenceId) -> Option<&Evidence> {
        self.evidence_store.get(&evidence_id)
    }
    
    pub fn find_evidence_by_quality(&self, min_quality: f32) -> Vec<&Evidence> {
        self.by_quality.iter()
            .take_while(|&&(quality, _)| quality >= min_quality)
            .filter_map(|&(_, evidence_id)| self.evidence_store.get(&evidence_id))
            .collect()
    }
    
    pub fn find_evidence_by_entity(&self, entity_id: EntityId) -> Vec<&Evidence> {
        self.by_entity.get(&entity_id)
            .map(|evidence_ids| {
                evidence_ids.iter()
                    .filter_map(|&id| self.evidence_store.get(&id))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    pub fn find_evidence_by_type(&self, evidence_type: &EvidenceType) -> Vec<&Evidence> {
        self.by_type.get(evidence_type)
            .map(|evidence_ids| {
                evidence_ids.iter()
                    .filter_map(|&id| self.evidence_store.get(&id))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    fn update_source_reliability(&mut self, source_id: &str, quality_score: f32) {
        let reliability = self.source_reliability
            .entry(source_id.to_string())
            .or_insert_with(|| SourceReliability {
                overall_score: quality_score,
                accuracy_history: VecDeque::new(),
                verification_count: 0,
                dispute_count: 0,
                last_assessment: Instant::now(),
            });
        
        reliability.accuracy_history.push_back(quality_score);
        if reliability.accuracy_history.len() > 100 {
            reliability.accuracy_history.pop_front();
        }
        
        // Recalculate overall score
        reliability.overall_score = reliability.accuracy_history.iter().sum::<f32>() / 
                                   reliability.accuracy_history.len() as f32;
        reliability.last_assessment = Instant::now();
    }
}

impl EvidenceValidator {
    pub fn new() -> Self {
        Self {
            validation_rules: Self::create_default_rules(),
            consistency_checker: ConsistencyChecker::new(),
            quality_assessor: QualityAssessor::new(),
        }
    }
    
    pub fn validate_evidence(&self, evidence: &Evidence) -> ValidationResult {
        let mut issues = Vec::new();
        let mut is_valid = true;
        
        // Apply validation rules
        for rule in &self.validation_rules {
            if self.check_rule_condition(&rule.condition, evidence) {
                let issue = self.create_validation_issue(rule, evidence);
                
                if matches!(rule.severity, Severity::Critical | Severity::High) {
                    is_valid = false;
                }
                
                issues.push(issue);
            }
        }
        
        // Quality assessment
        let quality_score = self.quality_assessor.assess_quality(evidence);
        
        ValidationResult {
            is_valid,
            quality_score,
            issues,
            recommendations: self.generate_recommendations(evidence, &issues),
        }
    }
    
    fn check_rule_condition(&self, condition: &ValidationCondition, evidence: &Evidence) -> bool {
        match condition {
            ValidationCondition::QualityBelow(threshold) => {
                evidence.quality_metrics.overall_quality < *threshold
            },
            ValidationCondition::RelevanceBelow(threshold) => {
                evidence.relevance_score < *threshold
            },
            ValidationCondition::SourceUnreliable => {
                evidence.source.source_reliability < 0.3
            },
            ValidationCondition::ContentTooShort(min_length) => {
                evidence.content.primary_text.len() < *min_length
            },
            ValidationCondition::ContentTooLong(max_length) => {
                evidence.content.primary_text.len() > *max_length
            },
            ValidationCondition::TemporalMismatch => {
                evidence.timestamp.elapsed() > Duration::from_secs(3600) // 1 hour old
            },
            ValidationCondition::ConsistencyViolation => {
                // Would check against other evidence
                false
            },
        }
    }
    
    fn create_validation_issue(&self, rule: &ValidationRule, evidence: &Evidence) -> ValidationIssue {
        ValidationIssue {
            issue_type: match rule.rule_type {
                RuleType::Quality => IssueType::LowQuality,
                RuleType::Consistency => IssueType::ConsistencyViolation,
                RuleType::Relevance => IssueType::LowRelevance,
                RuleType::Source => IssueType::UnreliableSource,
                RuleType::Timeliness => IssueType::TemporalIssue,
            },
            severity: rule.severity.clone(),
            description: format!("Rule '{}' triggered for evidence {}", rule.rule_id, evidence.evidence_id.0),
            suggested_fix: Some("Consider improving evidence quality or finding alternative sources".to_string()),
        }
    }
    
    fn generate_recommendations(&self, evidence: &Evidence, issues: &[ValidationIssue]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if evidence.quality_metrics.overall_quality < 0.5 {
            recommendations.push("Consider finding higher quality evidence sources".to_string());
        }
        
        if evidence.relevance_score < 0.3 {
            recommendations.push("Evidence may not be directly relevant to the query".to_string());
        }
        
        if !issues.is_empty() {
            recommendations.push(format!("Address {} validation issues before using this evidence", issues.len()));
        }
        
        recommendations
    }
    
    fn create_default_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                rule_id: "minimum_quality".to_string(),
                rule_type: RuleType::Quality,
                condition: ValidationCondition::QualityBelow(0.2),
                action: ValidationAction::Reject,
                severity: Severity::High,
            },
            ValidationRule {
                rule_id: "minimum_relevance".to_string(),
                rule_type: RuleType::Relevance,
                condition: ValidationCondition::RelevanceBelow(0.1),
                action: ValidationAction::Downgrade,
                severity: Severity::Medium,
            },
            ValidationRule {
                rule_id: "unreliable_source".to_string(),
                rule_type: RuleType::Source,
                condition: ValidationCondition::SourceUnreliable,
                action: ValidationAction::Flag,
                severity: Severity::Medium,
            },
            ValidationRule {
                rule_id: "content_too_short".to_string(),
                rule_type: RuleType::Quality,
                condition: ValidationCondition::ContentTooShort(10),
                action: ValidationAction::AddWarning,
                severity: Severity::Low,
            },
        ]
    }
}

impl ConsistencyChecker {
    pub fn new() -> Self {
        Self {
            contradiction_rules: vec![],
            logical_constraints: vec![],
        }
    }
}

impl QualityAssessor {
    pub fn new() -> Self {
        Self {
            quality_metrics: vec![],
            scoring_weights: HashMap::from([
                ("accuracy".to_string(), 0.25),
                ("completeness".to_string(), 0.20),
                ("timeliness".to_string(), 0.15),
                ("objectivity".to_string(), 0.15),
                ("credibility".to_string(), 0.15),
                ("verification".to_string(), 0.10),
            ]),
        }
    }
    
    pub fn assess_quality(&self, evidence: &Evidence) -> f32 {
        // Use the already calculated overall quality
        evidence.quality_metrics.overall_quality
    }
}
```

## File Locations

- `src/cognitive/explanation/evidence_collection.rs` - Main implementation
- `src/cognitive/explanation/mod.rs` - Module exports and integration
- `tests/cognitive/explanation/evidence_collection_tests.rs` - Test implementation

## Success Criteria

- [ ] Evidence collection from pathways and reasoning functional
- [ ] Evidence quality assessment accurate
- [ ] Evidence validation and filtering working
- [ ] Evidence indexing and retrieval efficient
- [ ] Relationship detection between evidence items
- [ ] Real-time evidence gathering performance meets targets
- [ ] All tests pass:
  - Basic evidence collection
  - Quality assessment accuracy
  - Validation rule enforcement
  - Evidence relationship detection
  - Performance benchmarks

## Test Requirements

```rust
#[test]
fn test_evidence_collection_from_pathway() {
    let mut collector = EvidenceCollector::new();
    
    let collection_id = collector.start_collection("test query", CollectionStrategy::Quality);
    
    let pathway = create_test_pathway();
    let evidence_ids = collector.collect_from_activation_pathway(collection_id, &pathway).unwrap();
    
    assert!(!evidence_ids.is_empty());
    assert_eq!(evidence_ids.len(), pathway.segments.len());
    
    let collection = collector.finalize_collection(collection_id).unwrap();
    assert_eq!(collection.evidence_items.len(), evidence_ids.len());
    assert!(!collection.query.is_empty());
}

#[test]
fn test_evidence_quality_assessment() {
    let collector = EvidenceCollector::new();
    
    // High quality evidence
    let high_quality_evidence = Evidence {
        evidence_id: EvidenceId(1),
        source: EvidenceSource {
            source_type: SourceType::KnowledgeGraph,
            source_identifier: "reliable_source".to_string(),
            source_reliability: 0.9,
            last_updated: SystemTime::now(),
            access_count: 1,
            verification_status: VerificationStatus::Verified,
        },
        content: EvidenceContent {
            primary_text: "Well-documented fact with supporting data".to_string(),
            supporting_data: HashMap::new(),
            structured_data: None,
            multimedia_refs: vec![],
            citations: vec![],
        },
        evidence_type: EvidenceType::FactualClaim,
        quality_metrics: EvidenceQuality {
            accuracy: 0.9,
            completeness: 0.8,
            timeliness: 1.0,
            objectivity: 0.9,
            source_credibility: 0.9,
            verification_level: 1.0,
            overall_quality: 0.0, // Will be calculated
        },
        relevance_score: 0.8,
        confidence: 0.9,
        timestamp: Instant::now(),
        context: EvidenceContext {
            query_context: "test".to_string(),
            reasoning_step_id: None,
            pathway_segment: None,
            activation_nodes: vec![],
            related_entities: vec![],
            domain_context: "test".to_string(),
            temporal_context: None,
        },
        relationships: vec![],
    };
    
    let overall_quality = collector.calculate_overall_quality(&high_quality_evidence.quality_metrics);
    assert!(overall_quality > 0.8);
}

#[test]
fn test_evidence_validation() {
    let validator = EvidenceValidator::new();
    
    // Valid evidence
    let valid_evidence = create_valid_test_evidence();
    let result = validator.validate_evidence(&valid_evidence);
    assert!(result.is_valid);
    assert!(result.issues.is_empty());
    
    // Invalid evidence (low quality)
    let mut invalid_evidence = valid_evidence.clone();
    invalid_evidence.quality_metrics.overall_quality = 0.1;
    
    let result = validator.validate_evidence(&invalid_evidence);
    assert!(!result.is_valid);
    assert!(!result.issues.is_empty());
}

#[test]
fn test_evidence_relationship_detection() {
    let mut collector = EvidenceCollector::new();
    
    let supporting_evidence = vec![
        create_evidence_with_content("AI is a field of computer science"),
        create_evidence_with_content("Computer science studies computational systems"),
    ];
    
    let relationships = collector.detect_evidence_relationships(&supporting_evidence);
    
    // Should detect supporting relationship
    assert!(!relationships.is_empty());
    assert!(relationships.iter().any(|r| matches!(r.relationship_type, RelationshipType::Supports)));
}

#[test]
fn test_evidence_indexing() {
    let mut index = EvidenceIndex::new();
    
    let evidence = create_valid_test_evidence();
    let evidence_id = evidence.evidence_id;
    
    index.add_evidence(evidence).unwrap();
    
    // Test retrieval
    let retrieved = index.get_evidence(evidence_id);
    assert!(retrieved.is_some());
    
    // Test quality-based search
    let high_quality = index.find_evidence_by_quality(0.7);
    assert!(!high_quality.is_empty());
}

#[test]
fn test_collection_performance() {
    let mut collector = EvidenceCollector::new();
    
    let start_time = Instant::now();
    let collection_id = collector.start_collection("performance test", CollectionStrategy::Balanced);
    
    // Collect evidence from multiple pathways
    for i in 0..10 {
        let pathway = create_test_pathway_with_id(i);
        collector.collect_from_activation_pathway(collection_id, &pathway).unwrap();
    }
    
    let _collection = collector.finalize_collection(collection_id).unwrap();
    let elapsed = start_time.elapsed();
    
    // Should complete within performance target
    assert!(elapsed < Duration::from_millis(100));
}

fn create_test_pathway() -> ActivationPathway {
    ActivationPathway {
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
    }
}

fn create_valid_test_evidence() -> Evidence {
    Evidence {
        evidence_id: EvidenceId(1),
        source: EvidenceSource {
            source_type: SourceType::KnowledgeGraph,
            source_identifier: "test_source".to_string(),
            source_reliability: 0.8,
            last_updated: SystemTime::now(),
            access_count: 1,
            verification_status: VerificationStatus::Verified,
        },
        content: EvidenceContent {
            primary_text: "Test evidence content".to_string(),
            supporting_data: HashMap::new(),
            structured_data: None,
            multimedia_refs: vec![],
            citations: vec![],
        },
        evidence_type: EvidenceType::FactualClaim,
        quality_metrics: EvidenceQuality {
            accuracy: 0.8,
            completeness: 0.7,
            timeliness: 1.0,
            objectivity: 0.8,
            source_credibility: 0.8,
            verification_level: 0.9,
            overall_quality: 0.82,
        },
        relevance_score: 0.7,
        confidence: 0.8,
        timestamp: Instant::now(),
        context: EvidenceContext {
            query_context: "test query".to_string(),
            reasoning_step_id: None,
            pathway_segment: None,
            activation_nodes: vec![],
            related_entities: vec![],
            domain_context: "test".to_string(),
            temporal_context: None,
        },
        relationships: vec![],
    }
}
```

## Quality Gates

- [ ] Evidence collection < 20ms per query
- [ ] Memory usage < 30MB for large evidence collections
- [ ] Evidence quality assessment accuracy > 85%
- [ ] Validation rule effectiveness in filtering low-quality evidence
- [ ] Evidence relationship detection precision > 70%
- [ ] Thread-safe concurrent evidence collection

## Next Task

Upon completion, proceed to **29_explanation_quality.md**