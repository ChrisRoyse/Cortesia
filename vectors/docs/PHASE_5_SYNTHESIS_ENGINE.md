# Phase 5: Synthesis Engine - Intelligent Signal Fusion for 95-97% Accuracy

## Objective
Combine all search signals (text, semantic, temporal, AST) into accurate, ranked answers with confidence scoring, contradiction resolution, and explainable results. Achieve 95-97% accuracy through deterministic synthesis algorithms following London School TDD methodology.

## Prerequisites
- **Completed Phases**: 0-4 (Foundation, Boolean Logic, Advanced Search, Scale & Performance)
- **Dependencies**: All search engines from previous phases
- **Test Data**: Comprehensive ground truth dataset with known correct answers
- **Performance Baseline**: Sub-200ms synthesis for complex queries

## Duration
1 Week (5 days, 40 hours) - Mock-first development with progressive real implementation

## Why Synthesis Engine is Critical
The Synthesis Engine is the intelligence layer that transforms raw search results into accurate answers:

- ✅ **Signal Fusion**: Combines exact match, semantic, temporal, and AST signals
- ✅ **Confidence Scoring**: Provides accuracy estimates for each result
- ✅ **Contradiction Resolution**: Handles conflicting information across sources
- ✅ **Weighted Voting**: Balances different search method strengths
- ✅ **Answer Assembly**: Constructs coherent responses from fragmented data
- ✅ **Explanation Generation**: Provides reasoning chains for answers

## SPARC Framework Application

### Specification
- **Input**: Results from multiple search engines with different confidence levels
- **Output**: Ranked, synthesized answers with confidence scores and explanations
- **Constraints**: Deterministic algorithms only, no LLM dependencies, Windows-compatible
- **Performance**: < 200ms synthesis latency, > 95% accuracy on well-defined queries

### Pseudocode
```
FOR each query:
    1. Collect results from all search engines
    2. Normalize scores across different engines
    3. Apply weighted voting based on query type
    4. Resolve contradictions using evidence strength
    5. Calculate confidence scores
    6. Assemble final answer with explanation
    7. Rank results by combined confidence
```

### Architecture
```rust
pub struct SynthesisEngine {
    signal_processors: Vec<Box<dyn SignalProcessor>>,
    voting_strategy: Box<dyn VotingStrategy>,
    contradiction_resolver: ContradictionResolver,
    confidence_calculator: ConfidenceCalculator,
    answer_assembler: AnswerAssembler,
    explanation_generator: ExplanationGenerator,
}
```

### Refinement
- Progressive implementation from mock to real components
- Each component fully tested before integration
- Performance optimization after correctness validation

### Completion
- All synthesis components working with real data
- 95%+ accuracy on validation dataset
- Full explanation generation for all answers

## Technical Approach

### 1. Mock Synthesis Engine Interface (London TDD - Mock First)
```rust
use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait SynthesisEngine {
    async fn synthesize(&self, query: &str, raw_results: MultiSearchResults) -> anyhow::Result<SynthesizedAnswer>;
    async fn calculate_confidence(&self, evidence: &Evidence) -> f32;
    async fn resolve_contradictions(&self, conflicting_results: Vec<SearchResult>) -> Vec<SearchResult>;
    async fn generate_explanation(&self, answer: &SynthesizedAnswer) -> String;
}

// Mock implementation for TDD
pub struct MockSynthesisEngine {
    expected_queries: HashMap<String, SynthesizedAnswer>,
    confidence_scores: HashMap<String, f32>,
}

#[async_trait]
impl SynthesisEngine for MockSynthesisEngine {
    async fn synthesize(&self, query: &str, _raw_results: MultiSearchResults) -> anyhow::Result<SynthesizedAnswer> {
        self.expected_queries
            .get(query)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Query not found in mock: {}", query))
    }
    
    async fn calculate_confidence(&self, evidence: &Evidence) -> f32 {
        self.confidence_scores
            .get(&evidence.id)
            .copied()
            .unwrap_or(0.5)
    }
    
    async fn resolve_contradictions(&self, conflicting_results: Vec<SearchResult>) -> Vec<SearchResult> {
        // Mock: return first result for predictable testing
        conflicting_results.into_iter().take(1).collect()
    }
    
    async fn generate_explanation(&self, answer: &SynthesizedAnswer) -> String {
        format!("Mock explanation for answer with confidence {:.2}", answer.confidence)
    }
}

impl MockSynthesisEngine {
    pub fn new() -> Self {
        Self {
            expected_queries: HashMap::new(),
            confidence_scores: HashMap::new(),
        }
    }
    
    pub fn expect_query(&mut self, query: &str, answer: SynthesizedAnswer) {
        self.expected_queries.insert(query.to_string(), answer);
    }
    
    pub fn set_confidence(&mut self, evidence_id: &str, confidence: f32) {
        self.confidence_scores.insert(evidence_id.to_string(), confidence);
    }
}
```

### 2. Signal Processing and Normalization
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSearchResults {
    pub text_results: Vec<TextSearchResult>,
    pub vector_results: Vec<VectorSearchResult>,
    pub ast_results: Vec<ASTSearchResult>,
    pub temporal_results: Vec<TemporalResult>,
    pub boolean_results: Vec<BooleanSearchResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedSignal {
    pub source: SignalSource,
    pub confidence: f32,          // 0.0 to 1.0
    pub relevance: f32,          // 0.0 to 1.0
    pub evidence: Vec<Evidence>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalSource {
    ExactMatch,
    SemanticVector,
    ASTStructural,
    TemporalGit,
    BooleanLogic,
    FuzzyMatch,
    ProximitySearch,
}

pub struct SignalNormalizer {
    source_weights: HashMap<SignalSource, f32>,
    query_type_modifiers: HashMap<QueryType, HashMap<SignalSource, f32>>,
}

impl SignalNormalizer {
    pub fn new() -> Self {
        let mut source_weights = HashMap::new();
        source_weights.insert(SignalSource::ExactMatch, 1.0);      // Highest confidence
        source_weights.insert(SignalSource::BooleanLogic, 0.95);   // Very reliable
        source_weights.insert(SignalSource::ASTStructural, 0.9);   // Code structure reliable
        source_weights.insert(SignalSource::SemanticVector, 0.8);  // Good semantic understanding
        source_weights.insert(SignalSource::TemporalGit, 0.7);     // Context dependent
        source_weights.insert(SignalSource::FuzzyMatch, 0.6);      // Typo handling
        source_weights.insert(SignalSource::ProximitySearch, 0.75); // Moderate reliability
        
        Self {
            source_weights,
            query_type_modifiers: HashMap::new(),
        }
    }
    
    pub fn normalize_signals(&self, raw_results: MultiSearchResults, query_type: QueryType) -> Vec<NormalizedSignal> {
        let mut normalized = Vec::new();
        
        // Normalize text search results
        for result in raw_results.text_results {
            let base_confidence = self.source_weights.get(&SignalSource::ExactMatch).unwrap_or(&0.5);
            let modifier = self.get_query_type_modifier(&query_type, &SignalSource::ExactMatch);
            
            normalized.push(NormalizedSignal {
                source: SignalSource::ExactMatch,
                confidence: (base_confidence * modifier).min(1.0),
                relevance: result.score,
                evidence: vec![Evidence {
                    id: result.id,
                    content: result.content,
                    location: result.file_path,
                    score: result.score,
                    metadata: result.metadata,
                }],
                metadata: HashMap::new(),
            });
        }
        
        // Normalize vector search results
        for result in raw_results.vector_results {
            let base_confidence = self.source_weights.get(&SignalSource::SemanticVector).unwrap_or(&0.5);
            let modifier = self.get_query_type_modifier(&query_type, &SignalSource::SemanticVector);
            
            normalized.push(NormalizedSignal {
                source: SignalSource::SemanticVector,
                confidence: (base_confidence * modifier).min(1.0),
                relevance: result.similarity_score,
                evidence: vec![Evidence {
                    id: result.id,
                    content: result.content,
                    location: result.file_path,
                    score: result.similarity_score,
                    metadata: HashMap::new(),
                }],
                metadata: HashMap::new(),
            });
        }
        
        // Normalize AST search results
        for result in raw_results.ast_results {
            let base_confidence = self.source_weights.get(&SignalSource::ASTStructural).unwrap_or(&0.5);
            let modifier = self.get_query_type_modifier(&query_type, &SignalSource::ASTStructural);
            
            normalized.push(NormalizedSignal {
                source: SignalSource::ASTStructural,
                confidence: (base_confidence * modifier).min(1.0),
                relevance: result.structural_match_score,
                evidence: vec![Evidence {
                    id: result.id,
                    content: result.matched_code,
                    location: result.file_path,
                    score: result.structural_match_score,
                    metadata: result.ast_metadata,
                }],
                metadata: HashMap::new(),
            });
        }
        
        normalized
    }
    
    fn get_query_type_modifier(&self, query_type: &QueryType, source: &SignalSource) -> f32 {
        self.query_type_modifiers
            .get(query_type)
            .and_then(|modifiers| modifiers.get(source))
            .copied()
            .unwrap_or(1.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: String,
    pub content: String,
    pub location: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum QueryType {
    SpecialCharacters,
    BooleanQuery,
    SemanticSearch,
    CodeStructure,
    Debugging,
    APIUsage,
    PerformanceAnalysis,
    SecurityAudit,
}
```

### 3. Weighted Voting System
```rust
pub struct WeightedVotingSystem {
    voting_strategies: HashMap<QueryType, Box<dyn VotingStrategy>>,
    default_strategy: Box<dyn VotingStrategy>,
}

#[async_trait]
pub trait VotingStrategy: Send + Sync {
    async fn vote(&self, signals: Vec<NormalizedSignal>) -> VotingResult;
    fn get_strategy_name(&self) -> &'static str;
}

#[derive(Debug, Clone)]
pub struct VotingResult {
    pub ranked_evidence: Vec<RankedEvidence>,
    pub combined_confidence: f32,
    pub voting_breakdown: HashMap<SignalSource, f32>,
}

#[derive(Debug, Clone)]
pub struct RankedEvidence {
    pub evidence: Evidence,
    pub final_score: f32,
    pub contributing_signals: Vec<SignalSource>,
    pub vote_weight: f32,
}

// Reciprocal Rank Fusion (RRF) Strategy
pub struct ReciprocalRankFusionStrategy {
    k_parameter: f32, // Typically 60.0
}

#[async_trait]
impl VotingStrategy for ReciprocalRankFusionStrategy {
    async fn vote(&self, signals: Vec<NormalizedSignal>) -> VotingResult {
        let mut evidence_scores: HashMap<String, f32> = HashMap::new();
        let mut evidence_map: HashMap<String, Evidence> = HashMap::new();
        let mut signal_contributions: HashMap<String, Vec<SignalSource>> = HashMap::new();
        
        // Group signals by evidence ID
        for signal in signals {
            for evidence in signal.evidence {
                evidence_map.insert(evidence.id.clone(), evidence.clone());
                signal_contributions
                    .entry(evidence.id.clone())
                    .or_insert_with(Vec::new)
                    .push(signal.source.clone());
            }
        }
        
        // Apply RRF scoring
        for signal in &signals {
            let ranked_evidence: Vec<_> = signal.evidence
                .iter()
                .enumerate()
                .collect();
                
            for (rank, evidence) in ranked_evidence {
                let rrf_score = 1.0 / (self.k_parameter + rank as f32 + 1.0);
                let weighted_score = rrf_score * signal.confidence * signal.relevance;
                
                *evidence_scores.entry(evidence.id.clone()).or_insert(0.0) += weighted_score;
            }
        }
        
        // Create ranked results
        let mut ranked_evidence: Vec<_> = evidence_scores
            .into_iter()
            .filter_map(|(id, score)| {
                evidence_map.get(&id).map(|evidence| RankedEvidence {
                    evidence: evidence.clone(),
                    final_score: score,
                    contributing_signals: signal_contributions.get(&id).cloned().unwrap_or_default(),
                    vote_weight: score,
                })
            })
            .collect();
        
        ranked_evidence.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
        
        let combined_confidence = if ranked_evidence.is_empty() {
            0.0
        } else {
            ranked_evidence.iter().map(|e| e.final_score).sum::<f32>() / ranked_evidence.len() as f32
        };
        
        VotingResult {
            ranked_evidence,
            combined_confidence,
            voting_breakdown: HashMap::new(), // Simplified for mock
        }
    }
    
    fn get_strategy_name(&self) -> &'static str {
        "ReciprocalRankFusion"
    }
}

// Borda Count Strategy for balanced voting
pub struct BordaCountStrategy {
    signal_weights: HashMap<SignalSource, f32>,
}

#[async_trait]
impl VotingStrategy for BordaCountStrategy {
    async fn vote(&self, signals: Vec<NormalizedSignal>) -> VotingResult {
        let mut evidence_scores: HashMap<String, f32> = HashMap::new();
        let mut evidence_map: HashMap<String, Evidence> = HashMap::new();
        
        for signal in &signals {
            let signal_weight = self.signal_weights
                .get(&signal.source)
                .copied()
                .unwrap_or(1.0);
                
            let evidence_count = signal.evidence.len() as f32;
            
            for (i, evidence) in signal.evidence.iter().enumerate() {
                evidence_map.insert(evidence.id.clone(), evidence.clone());
                
                // Borda count: higher positions get more points
                let position_score = (evidence_count - i as f32) / evidence_count;
                let weighted_score = position_score * signal_weight * signal.confidence;
                
                *evidence_scores.entry(evidence.id.clone()).or_insert(0.0) += weighted_score;
            }
        }
        
        let mut ranked_evidence: Vec<_> = evidence_scores
            .into_iter()
            .filter_map(|(id, score)| {
                evidence_map.get(&id).map(|evidence| RankedEvidence {
                    evidence: evidence.clone(),
                    final_score: score,
                    contributing_signals: Vec::new(), // Simplified
                    vote_weight: score,
                })
            })
            .collect();
        
        ranked_evidence.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
        
        let combined_confidence = if ranked_evidence.is_empty() {
            0.0
        } else {
            ranked_evidence.iter().map(|e| e.final_score).sum::<f32>() / ranked_evidence.len() as f32
        };
        
        VotingResult {
            ranked_evidence,
            combined_confidence,
            voting_breakdown: HashMap::new(),
        }
    }
    
    fn get_strategy_name(&self) -> &'static str {
        "BordaCount"
    }
}
```

### 4. Contradiction Resolution
```rust
pub struct ContradictionResolver {
    resolution_strategies: HashMap<ContradictionType, Box<dyn ResolutionStrategy>>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ContradictionType {
    VersionMismatch,      // Different versions of same API
    LocationConflict,     // Same function in multiple files
    ImplementationDiff,   // Different implementations
    MetadataConflict,     // Conflicting timestamps/authors
    ScoreDiscrepancy,     // Vastly different confidence scores
}

#[async_trait]
pub trait ResolutionStrategy: Send + Sync {
    async fn resolve(&self, contradictions: Vec<Evidence>) -> Evidence;
    fn can_handle(&self, contradiction_type: &ContradictionType) -> bool;
}

pub struct EvidenceStrengthResolver;

#[async_trait]
impl ResolutionStrategy for EvidenceStrengthResolver {
    async fn resolve(&self, contradictions: Vec<Evidence>) -> Evidence {
        // Choose evidence with highest combined score and metadata quality
        contradictions
            .into_iter()
            .max_by(|a, b| {
                let a_strength = self.calculate_evidence_strength(a);
                let b_strength = self.calculate_evidence_strength(b);
                a_strength.partial_cmp(&b_strength).unwrap()
            })
            .unwrap_or_else(|| Evidence {
                id: "unresolved".to_string(),
                content: "Unable to resolve contradiction".to_string(),
                location: "unknown".to_string(),
                score: 0.0,
                metadata: HashMap::new(),
            })
    }
    
    fn can_handle(&self, _contradiction_type: &ContradictionType) -> bool {
        true // Universal fallback
    }
}

impl EvidenceStrengthResolver {
    fn calculate_evidence_strength(&self, evidence: &Evidence) -> f32 {
        let mut strength = evidence.score;
        
        // Bonus for recency (if timestamp available)
        if let Some(timestamp) = evidence.metadata.get("timestamp") {
            if let Ok(ts) = timestamp.parse::<u64>() {
                let current_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let age_days = (current_time - ts) / (24 * 60 * 60);
                let recency_bonus = if age_days < 30 { 0.1 } else { 0.0 };
                strength += recency_bonus;
            }
        }
        
        // Bonus for more detailed content
        let content_quality = (evidence.content.len() as f32 / 1000.0).min(0.2);
        strength += content_quality;
        
        // Bonus for having rich metadata
        let metadata_bonus = (evidence.metadata.len() as f32 / 10.0).min(0.1);
        strength += metadata_bonus;
        
        strength
    }
}

pub struct TemporalResolver;

#[async_trait]
impl ResolutionStrategy for TemporalResolver {
    async fn resolve(&self, contradictions: Vec<Evidence>) -> Evidence {
        // For version conflicts, choose the most recent
        contradictions
            .into_iter()
            .max_by(|a, b| {
                let a_time = self.extract_timestamp(a).unwrap_or(0);
                let b_time = self.extract_timestamp(b).unwrap_or(0);
                a_time.cmp(&b_time)
            })
            .unwrap_or_else(|| Evidence {
                id: "temporal_unresolved".to_string(),
                content: "No temporal data available".to_string(),
                location: "unknown".to_string(),
                score: 0.0,
                metadata: HashMap::new(),
            })
    }
    
    fn can_handle(&self, contradiction_type: &ContradictionType) -> bool {
        matches!(contradiction_type, ContradictionType::VersionMismatch | ContradictionType::MetadataConflict)
    }
}

impl TemporalResolver {
    fn extract_timestamp(&self, evidence: &Evidence) -> Option<u64> {
        evidence.metadata
            .get("timestamp")
            .and_then(|ts| ts.parse().ok())
    }
}

impl ContradictionResolver {
    pub fn new() -> Self {
        let mut resolution_strategies: HashMap<ContradictionType, Box<dyn ResolutionStrategy>> = HashMap::new();
        
        resolution_strategies.insert(ContradictionType::VersionMismatch, Box::new(TemporalResolver));
        resolution_strategies.insert(ContradictionType::MetadataConflict, Box::new(TemporalResolver));
        resolution_strategies.insert(ContradictionType::LocationConflict, Box::new(EvidenceStrengthResolver));
        resolution_strategies.insert(ContradictionType::ImplementationDiff, Box::new(EvidenceStrengthResolver));
        resolution_strategies.insert(ContradictionType::ScoreDiscrepancy, Box::new(EvidenceStrengthResolver));
        
        Self {
            resolution_strategies,
        }
    }
    
    pub async fn resolve_contradictions(&self, contradictions: HashMap<ContradictionType, Vec<Evidence>>) -> HashMap<ContradictionType, Evidence> {
        let mut resolved = HashMap::new();
        
        for (contradiction_type, evidence_list) in contradictions {
            if let Some(resolver) = self.resolution_strategies.get(&contradiction_type) {
                let resolution = resolver.resolve(evidence_list).await;
                resolved.insert(contradiction_type, resolution);
            }
        }
        
        resolved
    }
    
    pub fn detect_contradictions(&self, evidence: Vec<Evidence>) -> HashMap<ContradictionType, Vec<Evidence>> {
        let mut contradictions: HashMap<ContradictionType, Vec<Evidence>> = HashMap::new();
        
        // Group evidence by location to detect conflicts
        let mut by_location: HashMap<String, Vec<Evidence>> = HashMap::new();
        for ev in evidence {
            by_location.entry(ev.location.clone()).or_insert_with(Vec::new).push(ev);
        }
        
        // Check for location conflicts (same content, different locations)
        for evidence_group in by_location.values() {
            if evidence_group.len() > 1 {
                // Check if they have significantly different scores
                let scores: Vec<f32> = evidence_group.iter().map(|e| e.score).collect();
                let max_score = scores.iter().fold(0.0f32, |a, b| a.max(*b));
                let min_score = scores.iter().fold(1.0f32, |a, b| a.min(*b));
                
                if max_score - min_score > 0.3 {
                    contradictions
                        .entry(ContradictionType::ScoreDiscrepancy)
                        .or_insert_with(Vec::new)
                        .extend(evidence_group.clone());
                }
            }
        }
        
        contradictions
    }
}
```

### 5. Confidence Calculation
```rust
pub struct ConfidenceCalculator {
    signal_reliability: HashMap<SignalSource, f32>,
    query_type_modifiers: HashMap<QueryType, f32>,
}

impl ConfidenceCalculator {
    pub fn new() -> Self {
        let mut signal_reliability = HashMap::new();
        signal_reliability.insert(SignalSource::ExactMatch, 0.98);
        signal_reliability.insert(SignalSource::BooleanLogic, 0.95);
        signal_reliability.insert(SignalSource::ASTStructural, 0.92);
        signal_reliability.insert(SignalSource::SemanticVector, 0.85);
        signal_reliability.insert(SignalSource::TemporalGit, 0.78);
        signal_reliability.insert(SignalSource::FuzzyMatch, 0.65);
        signal_reliability.insert(SignalSource::ProximitySearch, 0.8);
        
        let mut query_type_modifiers = HashMap::new();
        query_type_modifiers.insert(QueryType::SpecialCharacters, 0.95); // Text search excels
        query_type_modifiers.insert(QueryType::BooleanQuery, 0.98);      // Boolean logic is deterministic
        query_type_modifiers.insert(QueryType::SemanticSearch, 0.87);    // Vector search good but not perfect
        query_type_modifiers.insert(QueryType::CodeStructure, 0.93);     // AST parsing reliable
        query_type_modifiers.insert(QueryType::Debugging, 0.89);         // Multiple signals help
        query_type_modifiers.insert(QueryType::APIUsage, 0.91);          // Code patterns clear
        query_type_modifiers.insert(QueryType::PerformanceAnalysis, 0.85); // Complex analysis
        query_type_modifiers.insert(QueryType::SecurityAudit, 0.88);     // Pattern-based detection
        
        Self {
            signal_reliability,
            query_type_modifiers,
        }
    }
    
    pub fn calculate_confidence(&self, voting_result: &VotingResult, query_type: &QueryType) -> f32 {
        if voting_result.ranked_evidence.is_empty() {
            return 0.0;
        }
        
        // Base confidence from voting
        let base_confidence = voting_result.combined_confidence;
        
        // Signal diversity bonus (more signals = higher confidence)
        let unique_signals: std::collections::HashSet<_> = voting_result
            .ranked_evidence
            .iter()
            .flat_map(|e| &e.contributing_signals)
            .collect();
        let diversity_bonus = (unique_signals.len() as f32 / 7.0).min(0.15); // Max 15% bonus
        
        // Query type modifier
        let query_modifier = self.query_type_modifiers
            .get(query_type)
            .copied()
            .unwrap_or(0.85);
        
        // Evidence quality assessment
        let evidence_quality = self.assess_evidence_quality(&voting_result.ranked_evidence);
        
        // Final confidence calculation
        let raw_confidence = base_confidence * query_modifier + diversity_bonus;
        let quality_adjusted = raw_confidence * evidence_quality;
        
        quality_adjusted.min(1.0).max(0.0)
    }
    
    fn assess_evidence_quality(&self, evidence: &[RankedEvidence]) -> f32 {
        if evidence.is_empty() {
            return 0.0;
        }
        
        let mut quality_score = 0.0;
        let mut total_weight = 0.0;
        
        for ranked_evidence in evidence.iter().take(5) { // Top 5 results
            let weight = ranked_evidence.vote_weight;
            total_weight += weight;
            
            // Content length indicates thoroughness
            let content_quality = (ranked_evidence.evidence.content.len() as f32 / 500.0).min(1.0);
            
            // Metadata richness
            let metadata_quality = (ranked_evidence.evidence.metadata.len() as f32 / 5.0).min(1.0);
            
            // Signal consensus (how many signals agree)
            let consensus_quality = (ranked_evidence.contributing_signals.len() as f32 / 3.0).min(1.0);
            
            let evidence_quality = (content_quality + metadata_quality + consensus_quality) / 3.0;
            quality_score += evidence_quality * weight;
        }
        
        if total_weight > 0.0 {
            quality_score / total_weight
        } else {
            0.0
        }
    }
    
    pub fn calculate_result_confidence(&self, evidence: &Evidence, contributing_signals: &[SignalSource]) -> f32 {
        if contributing_signals.is_empty() {
            return evidence.score * 0.5; // Penalty for no signal information
        }
        
        // Average reliability of contributing signals
        let signal_reliability = contributing_signals
            .iter()
            .map(|source| self.signal_reliability.get(source).copied().unwrap_or(0.5))
            .sum::<f32>() / contributing_signals.len() as f32;
        
        // Combine with evidence score
        let combined = (evidence.score * 0.7) + (signal_reliability * 0.3);
        
        combined.min(1.0).max(0.0)
    }
}
```

### 6. Answer Assembly Pipeline
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedAnswer {
    pub primary_result: Evidence,
    pub supporting_evidence: Vec<Evidence>,
    pub confidence: f32,
    pub explanation: String,
    pub query_type: QueryType,
    pub synthesis_metadata: SynthesisMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisMetadata {
    pub signals_used: Vec<SignalSource>,
    pub voting_strategy: String,
    pub contradictions_resolved: usize,
    pub synthesis_time_ms: u64,
    pub result_count: usize,
}

pub struct AnswerAssembler {
    max_supporting_evidence: usize,
    min_confidence_threshold: f32,
    explanation_generator: ExplanationGenerator,
}

impl AnswerAssembler {
    pub fn new() -> Self {
        Self {
            max_supporting_evidence: 5,
            min_confidence_threshold: 0.1,
            explanation_generator: ExplanationGenerator::new(),
        }
    }
    
    pub async fn assemble_answer(
        &self,
        query: &str,
        query_type: QueryType,
        voting_result: VotingResult,
        confidence: f32,
        synthesis_start: std::time::Instant,
    ) -> anyhow::Result<SynthesizedAnswer> {
        
        if voting_result.ranked_evidence.is_empty() {
            return Ok(SynthesizedAnswer {
                primary_result: Evidence {
                    id: "no_results".to_string(),
                    content: "No relevant results found".to_string(),
                    location: "unknown".to_string(),
                    score: 0.0,
                    metadata: HashMap::new(),
                },
                supporting_evidence: Vec::new(),
                confidence: 0.0,
                explanation: "No matching content found for the query".to_string(),
                query_type,
                synthesis_metadata: SynthesisMetadata {
                    signals_used: Vec::new(),
                    voting_strategy: "none".to_string(),
                    contradictions_resolved: 0,
                    synthesis_time_ms: synthesis_start.elapsed().as_millis() as u64,
                    result_count: 0,
                },
            });
        }
        
        // Primary result is the highest ranked
        let primary_result = voting_result.ranked_evidence[0].evidence.clone();
        
        // Supporting evidence from remaining results
        let supporting_evidence: Vec<Evidence> = voting_result.ranked_evidence
            .iter()
            .skip(1)
            .take(self.max_supporting_evidence)
            .filter(|re| re.final_score >= self.min_confidence_threshold)
            .map(|re| re.evidence.clone())
            .collect();
        
        // Collect all unique signals used
        let signals_used: std::collections::HashSet<SignalSource> = voting_result.ranked_evidence
            .iter()
            .flat_map(|re| &re.contributing_signals)
            .cloned()
            .collect();
        
        // Generate explanation
        let explanation = self.explanation_generator.generate_explanation(
            query,
            &primary_result,
            &supporting_evidence,
            &signals_used.iter().cloned().collect::<Vec<_>>(),
            confidence,
        ).await;
        
        Ok(SynthesizedAnswer {
            primary_result,
            supporting_evidence,
            confidence,
            explanation,
            query_type,
            synthesis_metadata: SynthesisMetadata {
                signals_used: signals_used.into_iter().collect(),
                voting_strategy: "weighted_voting".to_string(), // TODO: Make dynamic
                contradictions_resolved: 0, // TODO: Track from resolver
                synthesis_time_ms: synthesis_start.elapsed().as_millis() as u64,
                result_count: voting_result.ranked_evidence.len(),
            },
        })
    }
}
```

### 7. Explanation Generation
```rust
pub struct ExplanationGenerator {
    templates: HashMap<QueryType, ExplanationTemplate>,
}

#[derive(Debug, Clone)]
pub struct ExplanationTemplate {
    pub intro_pattern: String,
    pub evidence_pattern: String,
    pub confidence_pattern: String,
    pub signal_descriptions: HashMap<SignalSource, String>,
}

impl ExplanationGenerator {
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        
        // Special Characters query template
        templates.insert(QueryType::SpecialCharacters, ExplanationTemplate {
            intro_pattern: "Found exact matches for special characters '{query}' using text search.".to_string(),
            evidence_pattern: "Located in {location} with {signal_count} supporting signals.".to_string(),
            confidence_pattern: "Confidence: {confidence:.1%} (exact text matching is highly reliable).".to_string(),
            signal_descriptions: [
                (SignalSource::ExactMatch, "exact text matching".to_string()),
                (SignalSource::BooleanLogic, "boolean query logic".to_string()),
            ].iter().cloned().collect(),
        });
        
        // Semantic Search query template
        templates.insert(QueryType::SemanticSearch, ExplanationTemplate {
            intro_pattern: "Found semantically similar content for '{query}' using vector embeddings.".to_string(),
            evidence_pattern: "Best match in {location} with semantic similarity score of {score:.2}.".to_string(),
            confidence_pattern: "Confidence: {confidence:.1%} (semantic matching with {signal_count} confirming signals).".to_string(),
            signal_descriptions: [
                (SignalSource::SemanticVector, "vector similarity matching".to_string()),
                (SignalSource::ExactMatch, "exact keyword matching".to_string()),
                (SignalSource::FuzzyMatch, "fuzzy text matching".to_string()),
            ].iter().cloned().collect(),
        });
        
        // Code Structure query template
        templates.insert(QueryType::CodeStructure, ExplanationTemplate {
            intro_pattern: "Analyzed code structure for '{query}' using AST parsing.".to_string(),
            evidence_pattern: "Found structural match in {location} with AST confidence {score:.2}.".to_string(),
            confidence_pattern: "Confidence: {confidence:.1%} (structural analysis with {signal_count} supporting methods).".to_string(),
            signal_descriptions: [
                (SignalSource::ASTStructural, "abstract syntax tree analysis".to_string()),
                (SignalSource::ExactMatch, "exact code matching".to_string()),
                (SignalSource::SemanticVector, "semantic code understanding".to_string()),
            ].iter().cloned().collect(),
        });
        
        Self { templates }
    }
    
    pub async fn generate_explanation(
        &self,
        query: &str,
        primary_result: &Evidence,
        supporting_evidence: &[Evidence],
        signals_used: &[SignalSource],
        confidence: f32,
    ) -> String {
        
        // Determine query type from signals (simplified heuristic)
        let query_type = self.infer_query_type(signals_used);
        
        let template = self.templates
            .get(&query_type)
            .cloned()
            .unwrap_or_else(|| self.default_template());
        
        let mut explanation = String::new();
        
        // Introduction
        explanation.push_str(&template.intro_pattern
            .replace("{query}", query));
        explanation.push_str(" ");
        
        // Primary evidence
        explanation.push_str(&template.evidence_pattern
            .replace("{location}", &primary_result.location)
            .replace("{score}", &primary_result.score.to_string())
            .replace("{signal_count}", &signals_used.len().to_string()));
        explanation.push_str(" ");
        
        // Supporting evidence summary
        if !supporting_evidence.is_empty() {
            explanation.push_str(&format!(
                "Additional supporting evidence found in {} other locations. ",
                supporting_evidence.len()
            ));
        }
        
        // Confidence explanation
        explanation.push_str(&template.confidence_pattern
            .replace("{confidence}", &confidence.to_string())
            .replace("{signal_count}", &signals_used.len().to_string()));
        explanation.push_str(" ");
        
        // Signal breakdown
        if signals_used.len() > 1 {
            explanation.push_str("Search methods used: ");
            let signal_descriptions: Vec<String> = signals_used
                .iter()
                .filter_map(|signal| template.signal_descriptions.get(signal))
                .cloned()
                .collect();
            explanation.push_str(&signal_descriptions.join(", "));
            explanation.push('.');
        }
        
        explanation
    }
    
    fn infer_query_type(&self, signals_used: &[SignalSource]) -> QueryType {
        // Simple heuristic based on dominant signal type
        if signals_used.contains(&SignalSource::ASTStructural) {
            QueryType::CodeStructure
        } else if signals_used.contains(&SignalSource::SemanticVector) {
            QueryType::SemanticSearch
        } else if signals_used.contains(&SignalSource::ExactMatch) {
            QueryType::SpecialCharacters
        } else if signals_used.contains(&SignalSource::BooleanLogic) {
            QueryType::BooleanQuery
        } else {
            QueryType::SemanticSearch // Default
        }
    }
    
    fn default_template(&self) -> ExplanationTemplate {
        ExplanationTemplate {
            intro_pattern: "Found results for '{query}' using multiple search methods.".to_string(),
            evidence_pattern: "Best match located in {location}.".to_string(),
            confidence_pattern: "Confidence: {confidence:.1%} based on {signal_count} search methods.".to_string(),
            signal_descriptions: [
                (SignalSource::ExactMatch, "exact matching".to_string()),
                (SignalSource::SemanticVector, "semantic search".to_string()),
                (SignalSource::ASTStructural, "code analysis".to_string()),
                (SignalSource::BooleanLogic, "boolean queries".to_string()),
                (SignalSource::FuzzyMatch, "fuzzy matching".to_string()),
                (SignalSource::ProximitySearch, "proximity search".to_string()),
                (SignalSource::TemporalGit, "git history analysis".to_string()),
            ].iter().cloned().collect(),
        }
    }
}
```

## Implementation Tasks (500-599)

### Task 500: Mock Synthesis Engine Foundation (1 day)
Create complete mock infrastructure for TDD development:
- MockSynthesisEngine with configurable responses
- MockSignalProcessor for each signal type
- MockVotingStrategy with predictable outcomes
- MockContradictionResolver for testing conflict scenarios
- Complete test harness for synthesis pipeline validation

### Task 501: Signal Normalization Implementation (1 day)
Replace mocks with real signal processing:
- Implement SignalNormalizer with configurable weights
- Create signal-specific processors for each source type
- Add query type detection and modifier application
- Build comprehensive test suite for normalization accuracy
- Performance optimization for large result sets

### Task 502: Weighted Voting System (1 day)
Implement multiple voting strategies:
- ReciprocalRankFusionStrategy with tunable parameters
- BordaCountStrategy for balanced democratic voting
- WeightedAverageStrategy for simple confidence weighting
- Strategy selection based on query characteristics
- Comprehensive voting accuracy validation

### Task 503: Contradiction Resolution Engine (1 day)
Build sophisticated conflict resolution:
- Implement EvidenceStrengthResolver with multi-factor scoring
- Create TemporalResolver for version conflicts
- Add LocationConflictResolver for duplicate detection
- Build ContradictionDetector with pattern recognition
- Validate resolution accuracy on synthetic conflicts

### Task 504: Confidence Calculation System (1 day)
Develop accurate confidence estimation:
- Implement multi-factor confidence calculation
- Create query-type-specific confidence modifiers
- Add evidence quality assessment algorithms
- Build signal diversity bonus calculations
- Calibrate confidence scores against ground truth

### Task 505: Answer Assembly Pipeline (Half day)
Create final answer construction:
- Implement AnswerAssembler with configurable parameters
- Add support for multiple evidence types
- Create answer ranking and filtering logic
- Build metadata collection and reporting
- Optimize assembly performance for real-time use

### Task 506: Explanation Generation (Half day)
Build interpretable answer explanations:
- Implement template-based explanation system
- Create query-type-specific explanation patterns
- Add signal contribution descriptions
- Build confidence reasoning explanations
- Validate explanation quality and clarity

### Task 507: Real Synthesis Engine Integration (1 day)
Replace all mocks with real implementations:
- Integrate with actual search engines from Phases 1-4
- Build complete synthesis pipeline
- Add error handling and graceful degradation
- Implement performance monitoring and logging
- Comprehensive integration testing

### Task 508: Performance Optimization (Half day)
Optimize synthesis for production use:
- Profile synthesis pipeline bottlenecks
- Implement result caching strategies
- Add parallel processing where beneficial
- Optimize memory usage for large result sets
- Achieve < 200ms synthesis latency target

### Task 509: Synthesis Validation Suite (Half day)
Create comprehensive testing framework:
- Build ground truth validation dataset
- Implement accuracy measurement tools
- Create contradiction resolution test cases
- Add confidence calibration validation
- Performance benchmark suite for synthesis engine

## Deliverables

### Rust Source Files
1. `src/synthesis/mod.rs` - Main synthesis engine interface
2. `src/synthesis/signal_processing.rs` - Signal normalization and processing
3. `src/synthesis/voting.rs` - Weighted voting strategies
4. `src/synthesis/contradiction.rs` - Contradiction detection and resolution
5. `src/synthesis/confidence.rs` - Confidence calculation algorithms
6. `src/synthesis/assembly.rs` - Answer assembly pipeline
7. `src/synthesis/explanation.rs` - Explanation generation system
8. `src/synthesis/mocks.rs` - Mock implementations for TDD

### Test Suites
1. Unit tests for each synthesis component
2. Integration tests for end-to-end synthesis
3. Performance benchmarks for synthesis latency
4. Accuracy validation against ground truth
5. Contradiction resolution test scenarios

### Configuration Files
1. Signal weight configurations for different query types
2. Voting strategy parameters and thresholds
3. Confidence calculation calibration data
4. Explanation templates for all query types

## Success Metrics

### Functional Requirements ✅ DESIGN COMPLETE
- [x] Multi-signal fusion with weighted voting designed
- [x] Contradiction resolution for conflicting results designed
- [x] Confidence scoring with query-type awareness designed
- [x] Answer assembly with supporting evidence designed
- [x] Explanation generation for all answer types designed
- [x] Mock-first TDD development approach designed

### Performance Targets ✅ DESIGN TARGETS SET
- [x] Synthesis latency target: < 200ms for complex queries
- [x] Accuracy target: 95-97% on well-defined queries
- [x] Confidence calibration target: ±5% of actual accuracy
- [x] Memory usage target: < 100MB for synthesis pipeline
- [x] Throughput target: > 50 syntheses per second

### Quality Gates ✅ DESIGN COMPLETE
- [x] Mock coverage designed: 100% before implementation
- [x] Unit test coverage designed: > 95% for all components
- [x] Integration test coverage designed: All synthesis paths
- [x] Performance validation designed: Latency and accuracy
- [x] Explanation quality designed: Human-readable and accurate

## Risk Mitigation

### Synthesis Accuracy Risks
- **Risk**: Conflicting signals produce incorrect answers
- **Mitigation**: Sophisticated contradiction resolution and confidence thresholds

### Performance Risks
- **Risk**: Complex synthesis exceeds latency targets
- **Mitigation**: Profiling, caching, and parallel processing optimization

### Complexity Risks
- **Risk**: Synthesis logic becomes too complex to maintain
- **Mitigation**: Mock-first development ensures testable, modular components

## Next Phase
With intelligent synthesis engine complete, proceed to Phase 6: Tiered Execution for cost-optimized query routing.

---

*Phase 5 creates the intelligence layer that transforms raw search results into accurate, explainable answers through deterministic synthesis algorithms.*