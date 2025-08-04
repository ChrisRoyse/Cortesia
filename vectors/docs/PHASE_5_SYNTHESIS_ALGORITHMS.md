# Phase 5: Synthesis Algorithms - Mathematical Foundations for Signal Fusion

## Overview
This document provides the concrete mathematical formulas, decision trees, and implementable algorithms that form the foundation of the Phase 5 Synthesis Engine. These algorithms transform raw search results into accurate, ranked answers with confidence scoring.

## Critical Gap Addresses

### 1. Weighted Voting Algorithm - Exact Formulas and Weights

#### Reciprocal Rank Fusion (RRF) Formula
```
RRF_score(d) = Σ(1 / (k + rank_i(d)))
```
Where:
- `d` = document/result
- `k` = constant (typically 60)
- `rank_i(d)` = rank of document d in result list i
- `i` = each search method/signal

#### Signal Weight Matrix
```rust
const SIGNAL_WEIGHTS: [(SignalSource, f32); 7] = [
    (SignalSource::ExactMatch, 1.00),       // Perfect precision for exact queries
    (SignalSource::BooleanLogic, 0.95),     // Deterministic boolean operators
    (SignalSource::ASTStructural, 0.90),    // Code structure highly reliable
    (SignalSource::SemanticVector, 0.80),   // Good semantic understanding
    (SignalSource::ProximitySearch, 0.75),  // Moderate proximity reliability
    (SignalSource::TemporalGit, 0.70),      // Context-dependent temporal data
    (SignalSource::FuzzyMatch, 0.60),       // Typo handling with noise
];
```

#### Query Type Modifiers
```rust
const QUERY_TYPE_MODIFIERS: [(QueryType, [(SignalSource, f32); 7]); 8] = [
    (QueryType::SpecialCharacters, [
        (SignalSource::ExactMatch, 1.20),      // Boost exact match for special chars
        (SignalSource::BooleanLogic, 1.10),    // Boolean handles special chars well
        (SignalSource::SemanticVector, 0.70),  // Vectors struggle with special chars
        (SignalSource::ASTStructural, 0.80),   // AST partially handles special chars
        (SignalSource::ProximitySearch, 0.90), // Proximity maintains relevance
        (SignalSource::TemporalGit, 0.60),     // Temporal less relevant
        (SignalSource::FuzzyMatch, 0.50),      // Fuzzy conflicts with exact special chars
    ]),
    (QueryType::BooleanQuery, [
        (SignalSource::BooleanLogic, 1.30),    // Major boost for boolean queries
        (SignalSource::ExactMatch, 1.10),      // Exact match supports boolean
        (SignalSource::ASTStructural, 0.95),   // AST understands boolean structure
        (SignalSource::SemanticVector, 0.85),  // Vectors less precise for boolean
        (SignalSource::ProximitySearch, 0.80), // Proximity secondary to boolean logic
        (SignalSource::TemporalGit, 0.70),     // Temporal not relevant to boolean
        (SignalSource::FuzzyMatch, 0.40),      // Fuzzy contradicts boolean precision
    ]),
    (QueryType::SemanticSearch, [
        (SignalSource::SemanticVector, 1.25),  // Major boost for semantic queries
        (SignalSource::FuzzyMatch, 1.10),      // Fuzzy supports semantic understanding
        (SignalSource::ProximitySearch, 1.05), // Proximity enhances semantic relevance
        (SignalSource::ExactMatch, 0.85),      // Exact match less flexible for semantic
        (SignalSource::ASTStructural, 0.80),   // AST limited for pure semantic queries
        (SignalSource::BooleanLogic, 0.75),    // Boolean too rigid for semantic
        (SignalSource::TemporalGit, 0.90),     // Temporal adds context
    ]),
    (QueryType::CodeStructure, [
        (SignalSource::ASTStructural, 1.40),   // Massive boost for code structure
        (SignalSource::ExactMatch, 1.15),      // Exact match good for code patterns
        (SignalSource::BooleanLogic, 1.05),    // Boolean supports code logic
        (SignalSource::SemanticVector, 0.90),  // Vectors good for code semantics
        (SignalSource::ProximitySearch, 0.85), // Proximity less critical for structure
        (SignalSource::FuzzyMatch, 0.70),      // Fuzzy risky for precise code structure
        (SignalSource::TemporalGit, 1.10),     // Git history valuable for code evolution
    ]),
    // ... Additional query types with specific modifier matrices
];
```

#### Composite Scoring Formula
```
final_score(d) = Σ(signal_weight_i × query_modifier_i × RRF_score_i(d) × confidence_i) / Σ(signal_weight_i × query_modifier_i × confidence_i)
```

### 2. Contradiction Resolution - Specific Resolution Strategies

#### Evidence Strength Calculation
```
evidence_strength(e) = base_score × recency_bonus × content_quality × metadata_richness × signal_consensus
```

Where:
- `base_score = e.score` (0.0 to 1.0)
- `recency_bonus = 1.0 + (0.1 × max(0, (30 - age_days) / 30))` (10% bonus for <30 days)
- `content_quality = min(0.2, content_length / 1000)` (up to 20% bonus for detailed content)
- `metadata_richness = min(0.1, metadata_count / 10)` (up to 10% bonus for rich metadata)
- `signal_consensus = min(0.15, contributing_signals / 7 × 0.15)` (up to 15% bonus for multi-signal agreement)

#### Contradiction Detection Algorithm
```rust
fn detect_contradictions(evidence: Vec<Evidence>) -> HashMap<ContradictionType, Vec<Evidence>> {
    let mut contradictions = HashMap::new();
    
    // Group by location for duplicate detection
    let by_location = group_by_location(evidence);
    
    for (location, evidence_group) in by_location {
        if evidence_group.len() > 1 {
            // Score discrepancy detection
            let scores: Vec<f32> = evidence_group.iter().map(|e| e.score).collect();
            let score_range = scores.iter().max() - scores.iter().min();
            
            if score_range > 0.3 {
                contradictions.entry(ContradictionType::ScoreDiscrepancy)
                    .or_insert_with(Vec::new)
                    .extend(evidence_group.clone());
            }
            
            // Temporal conflict detection
            let timestamps: Vec<u64> = evidence_group.iter()
                .filter_map(|e| e.metadata.get("timestamp")?.parse().ok())
                .collect();
            
            if timestamps.len() > 1 {
                let time_range = timestamps.iter().max().unwrap() - timestamps.iter().min().unwrap();
                if time_range > 86400 * 30 { // More than 30 days apart
                    contradictions.entry(ContradictionType::VersionMismatch)
                        .or_insert_with(Vec::new)
                        .extend(evidence_group.clone());
                }
            }
        }
    }
    
    contradictions
}
```

#### Resolution Strategy Decision Tree
```
If ContradictionType::VersionMismatch:
    └── Use TemporalResolver (select most recent)
        └── If no timestamps available:
            └── Fallback to EvidenceStrengthResolver

If ContradictionType::LocationConflict:
    └── Use EvidenceStrengthResolver
        └── If strength_difference < 0.1:
            └── Merge evidences with averaged scores

If ContradictionType::ScoreDiscrepancy:
    └── If score_range > 0.5:
        └── Use ConfidenceWeightedResolver
    └── Else:
        └── Use EvidenceStrengthResolver

If ContradictionType::ImplementationDiff:
    └── Use CodeVersionResolver (prioritize latest commit)
        └── Fallback to EvidenceStrengthResolver

If ContradictionType::MetadataConflict:
    └── Use MetadataMergeResolver
        └── If metadata_types_incompatible:
            └── Use TemporalResolver
```

### 3. Confidence Calculation - Mathematical Formulas

#### Base Confidence Formula
```
base_confidence = voting_result.combined_confidence × query_type_modifier
```

#### Signal Diversity Bonus
```
diversity_bonus = min(0.15, unique_signals_count / 7 × 0.15)
```

#### Evidence Quality Assessment
```
evidence_quality = Σ(evidence_quality_i × weight_i) / Σ(weight_i)

Where evidence_quality_i = (content_quality + metadata_quality + consensus_quality) / 3

content_quality = min(1.0, content_length / 500)
metadata_quality = min(1.0, metadata_count / 5)  
consensus_quality = min(1.0, contributing_signals_count / 3)
```

#### Final Confidence Calculation
```
final_confidence = min(1.0, max(0.0, 
    (base_confidence + diversity_bonus) × evidence_quality × calibration_factor
))

calibration_factor = query_type_calibration[query_type] × complexity_calibration[complexity]
```

#### Query Type Calibration Factors
```rust
const QUERY_TYPE_CALIBRATION: [(QueryType, f32); 8] = [
    (QueryType::SpecialCharacters, 1.05),    // Text search highly reliable
    (QueryType::BooleanQuery, 1.08),         // Boolean logic deterministic
    (QueryType::SemanticSearch, 0.92),       // Semantic inherently less precise
    (QueryType::CodeStructure, 1.02),        // AST parsing quite reliable
    (QueryType::Debugging, 0.88),            // Debugging queries complex
    (QueryType::APIUsage, 0.95),             // API patterns fairly clear
    (QueryType::PerformanceAnalysis, 0.85),  // Performance analysis complex
    (QueryType::SecurityAudit, 0.83),        // Security requires high precision
];
```

#### Complexity Calibration Factors
```rust
const COMPLEXITY_CALIBRATION: [(QueryComplexity, f32); 3] = [
    (QueryComplexity::Simple, 1.10),    // Simple queries more reliable
    (QueryComplexity::Medium, 1.00),    // Medium queries baseline
    (QueryComplexity::Complex, 0.85),   // Complex queries inherently uncertain
];
```

### 4. Empty Result Handling - Fallback Strategies

#### Empty Result Detection
```rust
fn is_empty_result(voting_result: &VotingResult) -> bool {
    voting_result.ranked_evidence.is_empty() || 
    voting_result.combined_confidence < 0.1 ||
    voting_result.ranked_evidence.iter().all(|e| e.final_score < 0.2)
}
```

#### Fallback Strategy Decision Tree
```
If empty_results && query_has_special_chars:
    └── Retry with escaped special characters
        └── If still empty:
            └── Retry with fuzzy matching (increased tolerance)
                └── If still empty:
                    └── Return "No exact matches found" with suggestions

If empty_results && query_type == SemanticSearch:
    └── Retry with expanded semantic query (synonyms)
        └── If still empty:
            └── Retry with reduced semantic threshold
                └── If still empty:
                    └── Fall back to keyword-based search

If empty_results && query_type == CodeStructure:
    └── Retry with relaxed AST matching
        └── If still empty:
            └── Fall back to text-based code search
                └── If still empty:
                    └── Return code pattern suggestions

If empty_results && query_type == BooleanQuery:
    └── Analyze boolean expression for syntax errors
        └── If syntax_error:
            └── Return corrected boolean syntax suggestions
        └── Else:
            └── Simplify boolean expression (remove complex clauses)
                └── If still empty:
                    └── Fall back to individual term searches
```

#### Suggestion Generation Algorithm
```rust
fn generate_fallback_suggestions(query: &str, query_type: QueryType) -> Vec<String> {
    let mut suggestions = Vec::new();
    
    match query_type {
        QueryType::SpecialCharacters => {
            suggestions.push(format!("Try escaping special characters: {}", escape_special_chars(query)));
            suggestions.push(format!("Try broader search: {}", remove_special_chars(query)));
            suggestions.push("Consider using quotes for exact phrase matching".to_string());
        },
        
        QueryType::SemanticSearch => {
            let synonyms = generate_synonyms(query);
            suggestions.extend(synonyms.into_iter().map(|s| format!("Try: {}", s)));
            suggestions.push("Try using more specific terms".to_string());
            suggestions.push("Try using different terminology".to_string());
        },
        
        QueryType::CodeStructure => {
            suggestions.push("Try searching for similar code patterns".to_string());
            suggestions.push(format!("Try searching for: {}", extract_code_keywords(query)));
            suggestions.push("Consider searching in different file types".to_string());
        },
        
        QueryType::BooleanQuery => {
            if let Some(corrected) = suggest_boolean_correction(query) {
                suggestions.push(format!("Did you mean: {}", corrected));
            }
            suggestions.push("Try simplifying the boolean expression".to_string());
            suggestions.push("Try searching individual terms separately".to_string());
        },
        
        _ => {
            suggestions.push("Try using different keywords".to_string());
            suggestions.push("Try broadening your search terms".to_string());
            suggestions.push("Try using synonyms or related terms".to_string());
        }
    }
    
    suggestions
}
```

#### Confidence Penalty for Fallback Results
```rust
fn apply_fallback_penalty(confidence: f32, fallback_level: u8) -> f32 {
    let penalty_factor = match fallback_level {
        0 => 1.0,    // No fallback
        1 => 0.8,    // First fallback (20% penalty)
        2 => 0.6,    // Second fallback (40% penalty)  
        3 => 0.4,    // Third fallback (60% penalty)
        _ => 0.2,    // Deep fallback (80% penalty)
    };
    
    confidence * penalty_factor
}
```

## Implementation Priority

### High Priority (Phase 5 Core)
1. **Weighted Voting Implementation**: RRF and Borda Count with signal weights
2. **Confidence Calculation**: Mathematical formulas with calibration factors
3. **Basic Contradiction Resolution**: Evidence strength and temporal resolvers
4. **Empty Result Handling**: Fallback strategies and suggestion generation

### Medium Priority (Phase 5 Enhancement)
1. **Advanced Contradiction Detection**: Multi-dimensional conflict analysis
2. **Dynamic Weight Adjustment**: Learning-based weight optimization
3. **Context-Aware Calibration**: User feedback integration
4. **Sophisticated Fallback**: Multi-level retry strategies

### Low Priority (Phase 5 Optimization)
1. **Machine Learning Integration**: Pattern-based weight learning
2. **User Preference Learning**: Personalized confidence calibration
3. **Advanced Suggestion Engine**: AI-powered query refinement
4. **Real-time Weight Tuning**: Performance-based weight adjustment

## Validation Metrics

### Weighted Voting Accuracy
- **Target**: 95%+ correct ranking of top 5 results
- **Measurement**: Ground truth comparison against manual ranking
- **Threshold**: RRF parameter k=60 ±10 for optimal performance

### Confidence Calibration
- **Target**: Confidence scores within ±5% of actual accuracy
- **Measurement**: Confidence vs. ground truth accuracy correlation
- **Threshold**: Pearson correlation > 0.85

### Contradiction Resolution Success
- **Target**: 90%+ correct resolution of detected contradictions
- **Measurement**: Manual evaluation of resolution decisions
- **Threshold**: Evidence strength calculation accuracy > 90%

### Fallback Strategy Effectiveness
- **Target**: 70%+ of fallback attempts produce useful results
- **Measurement**: User acceptance rate of fallback suggestions
- **Threshold**: Suggestion relevance score > 0.7

## Mathematical Constants and Thresholds

```rust
// Core algorithm parameters
pub const RRF_K_PARAMETER: f32 = 60.0;
pub const MIN_CONFIDENCE_THRESHOLD: f32 = 0.1;
pub const MAX_EVIDENCE_STRENGTH_BONUS: f32 = 0.45; // 45% max bonus
pub const CONTRADICTION_SCORE_THRESHOLD: f32 = 0.3;
pub const DIVERSITY_BONUS_MAX: f32 = 0.15;
pub const CONTENT_QUALITY_MAX_BONUS: f32 = 0.2;
pub const METADATA_RICHNESS_MAX_BONUS: f32 = 0.1;
pub const SIGNAL_CONSENSUS_MAX_BONUS: f32 = 0.15;

// Fallback strategy parameters
pub const EMPTY_RESULT_CONFIDENCE_THRESHOLD: f32 = 0.1;
pub const EMPTY_RESULT_SCORE_THRESHOLD: f32 = 0.2;
pub const MAX_FALLBACK_LEVELS: u8 = 4;
pub const FALLBACK_PENALTIES: [f32; 5] = [1.0, 0.8, 0.6, 0.4, 0.2];

// Temporal resolution parameters
pub const RECENCY_BONUS_DAYS: u64 = 30;
pub const VERSION_CONFLICT_THRESHOLD_DAYS: u64 = 30;
pub const MAX_RECENCY_BONUS: f32 = 0.1;
```

This comprehensive algorithmic foundation enables the Phase 5 Synthesis Engine to make deterministic, mathematically-grounded decisions while handling the full spectrum of search scenarios from perfect matches to complex contradictions and empty results.