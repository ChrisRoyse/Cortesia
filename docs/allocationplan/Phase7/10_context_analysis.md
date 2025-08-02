# Micro Task 10: Context Analysis

**Priority**: CRITICAL  
**Estimated Time**: 40 minutes  
**Dependencies**: 09_entity_extraction.md  
**Skills Required**: Context inference, domain classification

## Objective

Implement comprehensive context analysis for queries to understand domain, temporal/spatial constraints, and confidence requirements for intelligent query processing.

## Context

Context analysis provides critical information about the query environment, domain expertise level, and constraints that guide the activation spreading strategy and result filtering. This enables more precise and relevant responses.

## Specifications

### Context Classification System

1. **Domain Analysis**
   - Domain identification (biology, physics, general knowledge)
   - Expertise level detection (novice, intermediate, expert)
   - Subdomain classification
   - Cross-domain query detection

2. **Constraint Detection**
   - Temporal constraints (time periods, durations)
   - Spatial constraints (geographic, scale limitations)
   - Confidence requirements (precision vs. completeness)
   - Result scope preferences

3. **Query Characteristics**
   - Complexity assessment
   - Ambiguity detection
   - Intent clarity scoring
   - Processing priority classification

## Implementation Guide

### Step 1: Core Context Types
```rust
// File: src/query/context_analysis.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    pub domain: DomainContext,
    pub temporal: TemporalContext,
    pub spatial: SpatialContext,
    pub confidence: ConfidenceContext,
    pub complexity: ComplexityContext,
    pub constraints: Vec<QueryConstraint>,
    pub metadata: ContextMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainContext {
    pub primary_domain: Domain,
    pub subdomains: Vec<String>,
    pub expertise_level: ExpertiseLevel,
    pub interdisciplinary: bool,
    pub domain_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Domain {
    Biology,
    Physics,
    Chemistry,
    Medicine,
    Psychology,
    Technology,
    History,
    Geography,
    GeneralKnowledge,
    Mixed,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Novice,      // Basic explanations needed
    Intermediate, // Moderate technical detail
    Expert,      // Full technical complexity
    Academic,    // Research-level detail
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub time_period: Option<TimePeriod>,
    pub duration_constraint: Option<Duration>,
    pub temporal_relevance: TemporalRelevance,
    pub chronological_ordering: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePeriod {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
    pub era: Option<String>,
    pub relative_time: Option<RelativeTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelativeTime {
    Recent,      // Last few years
    Current,     // Present day
    Historical,  // Past decades/centuries
    Ancient,     // Very old
    Future,      // Predictions/projections
    Timeless,    // No time relevance
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalRelevance {
    Critical,    // Time is essential to answer
    Important,   // Time affects answer quality
    Relevant,    // Time provides useful context
    Optional,    // Time not important
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialContext {
    pub geographic_scope: GeographicScope,
    pub scale_level: ScaleLevel,
    pub location_specificity: LocationSpecificity,
    pub spatial_relationships: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeographicScope {
    Global,
    Continental,
    National,
    Regional,
    Local,
    Microscopic,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleLevel {
    Molecular,
    Cellular,
    Organism,
    Population,
    Ecosystem,
    Planetary,
    Universal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LocationSpecificity {
    Exact,       // Specific coordinates/place
    Regional,    // General area
    Categorical, // Type of environment
    Abstract,    // Conceptual space
}
```

### Step 2: Confidence and Complexity Analysis
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceContext {
    pub required_confidence: f32,
    pub prefer_completeness: bool,
    pub acceptable_uncertainty: f32,
    pub verification_level: VerificationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationLevel {
    None,        // Accept any reasonable answer
    Basic,       // Simple fact checking
    Moderate,    // Cross-reference sources
    Rigorous,    // Multiple verification methods
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityContext {
    pub syntactic_complexity: f32,
    pub semantic_complexity: f32,
    pub conceptual_depth: ConceptualDepth,
    pub processing_priority: ProcessingPriority,
    pub decomposition_needed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConceptualDepth {
    Surface,     // Basic facts
    Shallow,     // Simple relationships
    Moderate,    // Multi-step reasoning
    Deep,        // Complex inference
    Expert,      // Advanced analysis
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingPriority {
    Low,         // Can wait for optimal processing
    Normal,      // Standard priority
    High,        // Process quickly
    Urgent,      // Immediate processing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConstraint {
    pub constraint_type: ConstraintType,
    pub value: String,
    pub strictness: ConstraintStrictness,
    pub impact_on_results: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    MaxResults,
    MinConfidence,
    TimeLimit,
    LocationBound,
    SourceType,
    DataRecency,
    Language,
    Format,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintStrictness {
    Flexible,    // Can be relaxed if needed
    Preferred,   // Strong preference
    Required,    // Must be satisfied
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetadata {
    pub analysis_confidence: f32,
    pub ambiguity_score: f32,
    pub processing_hints: Vec<String>,
    pub suggested_strategies: Vec<String>,
    pub analysis_time_ms: u64,
}
```

### Step 3: Context Analyzer Implementation
```rust
pub struct ContextAnalyzer {
    domain_classifier: DomainClassifier,
    temporal_analyzer: TemporalAnalyzer,
    spatial_analyzer: SpatialAnalyzer,
    complexity_assessor: ComplexityAssessor,
    constraint_detector: ConstraintDetector,
    config: ContextAnalysisConfig,
}

#[derive(Debug, Clone)]
pub struct ContextAnalysisConfig {
    pub default_expertise_level: ExpertiseLevel,
    pub confidence_threshold: f32,
    pub max_processing_time_ms: u64,
    pub enable_deep_analysis: bool,
}

impl Default for ContextAnalysisConfig {
    fn default() -> Self {
        Self {
            default_expertise_level: ExpertiseLevel::Intermediate,
            confidence_threshold: 0.7,
            max_processing_time_ms: 100,
            enable_deep_analysis: true,
        }
    }
}

impl ContextAnalyzer {
    pub fn new() -> Self {
        Self {
            domain_classifier: DomainClassifier::new(),
            temporal_analyzer: TemporalAnalyzer::new(),
            spatial_analyzer: SpatialAnalyzer::new(),
            complexity_assessor: ComplexityAssessor::new(),
            constraint_detector: ConstraintDetector::new(),
            config: ContextAnalysisConfig::default(),
        }
    }
    
    pub fn analyze_context(&self, query: &str, entities: &[ExtractedEntity]) -> Result<QueryContext> {
        let start_time = std::time::Instant::now();
        
        // Parallel analysis of different context aspects
        let domain = self.domain_classifier.classify_domain(query, entities)?;
        let temporal = self.temporal_analyzer.analyze_temporal_aspects(query)?;
        let spatial = self.spatial_analyzer.analyze_spatial_aspects(query, entities)?;
        let complexity = self.complexity_assessor.assess_complexity(query, entities)?;
        let constraints = self.constraint_detector.detect_constraints(query)?;
        
        // Determine confidence requirements based on domain and complexity
        let confidence = self.infer_confidence_requirements(&domain, &complexity, query)?;
        
        let analysis_time = start_time.elapsed().as_millis() as u64;
        
        // Create metadata
        let metadata = ContextMetadata {
            analysis_confidence: self.calculate_overall_confidence(&domain, &temporal, &spatial)?,
            ambiguity_score: self.calculate_ambiguity_score(query, entities)?,
            processing_hints: self.generate_processing_hints(&domain, &complexity)?,
            suggested_strategies: self.suggest_processing_strategies(&domain, &complexity, &constraints)?,
            analysis_time_ms: analysis_time,
        };
        
        Ok(QueryContext {
            domain,
            temporal,
            spatial,
            confidence,
            complexity,
            constraints,
            metadata,
        })
    }
}
```

### Step 4: Domain Classification
```rust
pub struct DomainClassifier {
    domain_keywords: HashMap<Domain, Vec<String>>,
    subdomain_patterns: HashMap<Domain, Vec<String>>,
    expertise_indicators: HashMap<ExpertiseLevel, Vec<String>>,
}

impl DomainClassifier {
    pub fn new() -> Self {
        Self {
            domain_keywords: Self::create_domain_keywords(),
            subdomain_patterns: Self::create_subdomain_patterns(),
            expertise_indicators: Self::create_expertise_indicators(),
        }
    }
    
    fn create_domain_keywords() -> HashMap<Domain, Vec<String>> {
        let mut keywords = HashMap::new();
        
        keywords.insert(Domain::Biology, vec![
            "animal".into(), "plant".into(), "organism".into(), "species".into(),
            "evolution".into(), "genetics".into(), "ecology".into(), "cell".into(),
            "DNA".into(), "protein".into(), "ecosystem".into(), "habitat".into(),
        ]);
        
        keywords.insert(Domain::Physics, vec![
            "force".into(), "energy".into(), "particle".into(), "wave".into(),
            "quantum".into(), "relativity".into(), "electromagnetic".into(),
            "gravity".into(), "momentum".into(), "thermodynamics".into(),
        ]);
        
        keywords.insert(Domain::Chemistry, vec![
            "molecule".into(), "atom".into(), "reaction".into(), "compound".into(),
            "element".into(), "bond".into(), "catalyst".into(), "acid".into(),
            "base".into(), "oxidation".into(), "organic".into(),
        ]);
        
        keywords.insert(Domain::Medicine, vec![
            "disease".into(), "treatment".into(), "diagnosis".into(), "symptom".into(),
            "therapy".into(), "medication".into(), "surgery".into(), "patient".into(),
            "clinical".into(), "medical".into(), "health".into(),
        ]);
        
        keywords
    }
    
    fn create_expertise_indicators() -> HashMap<ExpertiseLevel, Vec<String>> {
        let mut indicators = HashMap::new();
        
        indicators.insert(ExpertiseLevel::Novice, vec![
            "what is".into(), "explain".into(), "simple".into(), "basic".into(),
            "introduction".into(), "beginner".into(), "easy".into(),
        ]);
        
        indicators.insert(ExpertiseLevel::Expert, vec![
            "molecular mechanism".into(), "statistical significance".into(),
            "methodology".into(), "quantitative".into(), "empirical".into(),
            "peer-reviewed".into(), "research".into(),
        ]);
        
        indicators.insert(ExpertiseLevel::Academic, vec![
            "meta-analysis".into(), "systematic review".into(), "hypothesis".into(),
            "correlation coefficient".into(), "p-value".into(), "confidence interval".into(),
        ]);
        
        indicators
    }
    
    pub fn classify_domain(&self, query: &str, entities: &[ExtractedEntity]) -> Result<DomainContext> {
        let query_lower = query.to_lowercase();
        let mut domain_scores = HashMap::new();
        
        // Score based on keywords
        for (domain, keywords) in &self.domain_keywords {
            let score = keywords.iter()
                .filter(|keyword| query_lower.contains(keyword.as_str()))
                .count() as f32;
            domain_scores.insert(domain.clone(), score);
        }
        
        // Boost scores based on entity types
        for entity in entities {
            match entity.entity_type {
                EntityType::Organism => {
                    *domain_scores.entry(Domain::Biology).or_insert(0.0) += 1.0;
                }
                EntityType::Concept => {
                    // Concepts can belong to multiple domains
                    for (domain, _) in &domain_scores {
                        *domain_scores.get_mut(domain).unwrap() += 0.2;
                    }
                }
                EntityType::Location => {
                    *domain_scores.entry(Domain::Geography).or_insert(0.0) += 0.5;
                }
                _ => {}
            }
        }
        
        // Find the highest scoring domain
        let primary_domain = domain_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(domain, _)| domain.clone())
            .unwrap_or(Domain::GeneralKnowledge);
        
        // Determine expertise level
        let expertise_level = self.classify_expertise_level(&query_lower)?;
        
        // Check for interdisciplinary queries
        let interdisciplinary = domain_scores.values()
            .filter(|&&score| score > 0.0)
            .count() > 2;
        
        // Calculate domain confidence
        let max_score = domain_scores.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let total_score: f32 = domain_scores.values().sum();
        let domain_confidence = if total_score > 0.0 {
            max_score / total_score
        } else {
            0.5
        };
        
        Ok(DomainContext {
            primary_domain,
            subdomains: self.identify_subdomains(&query_lower)?,
            expertise_level,
            interdisciplinary,
            domain_confidence,
        })
    }
    
    fn classify_expertise_level(&self, query: &str) -> Result<ExpertiseLevel> {
        for (level, indicators) in &self.expertise_indicators {
            for indicator in indicators {
                if query.contains(indicator) {
                    return Ok(level.clone());
                }
            }
        }
        
        // Default based on query complexity
        if query.split_whitespace().count() > 15 {
            Ok(ExpertiseLevel::Expert)
        } else if query.contains('?') && query.split_whitespace().count() < 8 {
            Ok(ExpertiseLevel::Novice)
        } else {
            Ok(ExpertiseLevel::Intermediate)
        }
    }
    
    fn identify_subdomains(&self, query: &str) -> Result<Vec<String>> {
        let mut subdomains = Vec::new();
        
        // Biology subdomains
        if query.contains("molecular") || query.contains("DNA") || query.contains("protein") {
            subdomains.push("molecular_biology".into());
        }
        if query.contains("ecosystem") || query.contains("habitat") || query.contains("environment") {
            subdomains.push("ecology".into());
        }
        if query.contains("behavior") || query.contains("psychology") {
            subdomains.push("behavioral_biology".into());
        }
        
        // Physics subdomains
        if query.contains("quantum") {
            subdomains.push("quantum_physics".into());
        }
        if query.contains("relativity") || query.contains("spacetime") {
            subdomains.push("relativity".into());
        }
        
        Ok(subdomains)
    }
}
```

### Step 5: Temporal and Spatial Analysis
```rust
pub struct TemporalAnalyzer {
    time_indicators: Vec<(String, RelativeTime)>,
    temporal_patterns: Vec<regex::Regex>,
}

impl TemporalAnalyzer {
    pub fn new() -> Self {
        Self {
            time_indicators: Self::create_time_indicators(),
            temporal_patterns: Self::create_temporal_patterns(),
        }
    }
    
    fn create_time_indicators() -> Vec<(String, RelativeTime)> {
        vec![
            ("recent".into(), RelativeTime::Recent),
            ("currently".into(), RelativeTime::Current),
            ("today".into(), RelativeTime::Current),
            ("historical".into(), RelativeTime::Historical),
            ("ancient".into(), RelativeTime::Ancient),
            ("prehistoric".into(), RelativeTime::Ancient),
            ("future".into(), RelativeTime::Future),
            ("will".into(), RelativeTime::Future),
        ]
    }
    
    pub fn analyze_temporal_aspects(&self, query: &str) -> Result<TemporalContext> {
        let query_lower = query.to_lowercase();
        
        // Detect relative time references
        let mut relative_time = None;
        let mut temporal_relevance = TemporalRelevance::Optional;
        
        for (indicator, time_type) in &self.time_indicators {
            if query_lower.contains(indicator) {
                relative_time = Some(time_type.clone());
                temporal_relevance = TemporalRelevance::Important;
                break;
            }
        }
        
        // Check for chronological ordering requirements
        let chronological_ordering = query_lower.contains("sequence") ||
            query_lower.contains("order") ||
            query_lower.contains("timeline") ||
            query_lower.contains("evolution");
        
        // Determine temporal relevance based on query content
        if query_lower.contains("when") || query_lower.contains("date") {
            temporal_relevance = TemporalRelevance::Critical;
        } else if chronological_ordering {
            temporal_relevance = TemporalRelevance::Important;
        }
        
        Ok(TemporalContext {
            time_period: None, // Would be populated by more sophisticated parsing
            duration_constraint: None,
            temporal_relevance,
            chronological_ordering,
        })
    }
}

pub struct SpatialAnalyzer {
    location_patterns: Vec<regex::Regex>,
    scale_indicators: HashMap<String, ScaleLevel>,
}

impl SpatialAnalyzer {
    pub fn new() -> Self {
        Self {
            location_patterns: Self::create_location_patterns(),
            scale_indicators: Self::create_scale_indicators(),
        }
    }
    
    fn create_scale_indicators() -> HashMap<String, ScaleLevel> {
        let mut indicators = HashMap::new();
        
        indicators.insert("molecular".into(), ScaleLevel::Molecular);
        indicators.insert("cellular".into(), ScaleLevel::Cellular);
        indicators.insert("organism".into(), ScaleLevel::Organism);
        indicators.insert("population".into(), ScaleLevel::Population);
        indicators.insert("ecosystem".into(), ScaleLevel::Ecosystem);
        indicators.insert("global".into(), ScaleLevel::Planetary);
        indicators.insert("worldwide".into(), ScaleLevel::Planetary);
        
        indicators
    }
    
    pub fn analyze_spatial_aspects(&self, query: &str, entities: &[ExtractedEntity]) -> Result<SpatialContext> {
        let query_lower = query.to_lowercase();
        
        // Determine geographic scope based on location entities
        let geographic_scope = if entities.iter().any(|e| matches!(e.entity_type, EntityType::Location)) {
            // More sophisticated analysis would parse specific locations
            GeographicScope::Regional
        } else if query_lower.contains("global") || query_lower.contains("worldwide") {
            GeographicScope::Global
        } else {
            GeographicScope::None
        };
        
        // Determine scale level
        let scale_level = self.scale_indicators.iter()
            .find(|(indicator, _)| query_lower.contains(indicator.as_str()))
            .map(|(_, scale)| scale.clone())
            .unwrap_or(ScaleLevel::Organism);
        
        // Check for spatial relationships
        let spatial_relationships = query_lower.contains("where") ||
            query_lower.contains("location") ||
            query_lower.contains("distance") ||
            query_lower.contains("near") ||
            query_lower.contains("between");
        
        Ok(SpatialContext {
            geographic_scope,
            scale_level,
            location_specificity: LocationSpecificity::Abstract,
            spatial_relationships,
        })
    }
}
```

## File Locations

- `src/query/context_analysis.rs` - Main implementation
- `src/query/domain_classifier.rs` - Domain classification
- `src/query/temporal_analyzer.rs` - Temporal context analysis
- `src/query/spatial_analyzer.rs` - Spatial context analysis
- `src/query/complexity_assessor.rs` - Complexity assessment
- `tests/query/context_analysis_tests.rs` - Test implementation

## Success Criteria

- [ ] Domain classification > 85% accuracy
- [ ] Expertise level detection working
- [ ] Temporal constraints identified correctly
- [ ] Spatial scope determined accurately
- [ ] Confidence requirements inferred properly
- [ ] Context analysis under 100ms
- [ ] All tests pass

## Test Requirements

```rust
#[test]
fn test_domain_classification() {
    let analyzer = ContextAnalyzer::new();
    
    let test_cases = vec![
        ("What animals live in the ocean?", Domain::Biology),
        ("How does quantum mechanics work?", Domain::Physics),
        ("What causes diabetes?", Domain::Medicine),
        ("Explain photosynthesis", Domain::Biology),
    ];
    
    for (query, expected_domain) in test_cases {
        let entities = vec![]; // Simplified for test
        let context = analyzer.analyze_context(query, &entities).unwrap();
        assert_eq!(context.domain.primary_domain, expected_domain);
    }
}

#[test]
fn test_expertise_level_detection() {
    let classifier = DomainClassifier::new();
    
    let test_cases = vec![
        ("What is DNA?", ExpertiseLevel::Novice),
        ("How do transcription factors regulate gene expression?", ExpertiseLevel::Expert),
        ("Explain the molecular mechanism of CRISPR-Cas9", ExpertiseLevel::Academic),
    ];
    
    for (query, expected_level) in test_cases {
        let level = classifier.classify_expertise_level(&query.to_lowercase()).unwrap();
        assert_eq!(level, expected_level);
    }
}

#[test]
fn test_temporal_analysis() {
    let analyzer = TemporalAnalyzer::new();
    
    let test_cases = vec![
        ("Recent discoveries in genetics", RelativeTime::Recent),
        ("Ancient civilizations", RelativeTime::Ancient),
        ("Current climate change effects", RelativeTime::Current),
        ("Future space exploration", RelativeTime::Future),
    ];
    
    for (query, expected_time) in test_cases {
        let context = analyzer.analyze_temporal_aspects(query).unwrap();
        assert_eq!(context.time_period.unwrap().relative_time.unwrap(), expected_time);
    }
}

#[test]
fn test_spatial_scope_detection() {
    let analyzer = SpatialAnalyzer::new();
    
    let entities = vec![
        ExtractedEntity {
            text: "Africa".to_string(),
            entity_type: EntityType::Location,
            start_pos: 0,
            end_pos: 6,
            confidence: 0.9,
            aliases: vec![],
            context_clues: vec![],
            modifiers: vec![],
        }
    ];
    
    let context = analyzer.analyze_spatial_aspects("Animals in Africa", &entities).unwrap();
    assert_eq!(context.geographic_scope, GeographicScope::Regional);
}

#[test]
fn test_complexity_assessment() {
    let assessor = ComplexityAssessor::new();
    
    let simple_query = "What is a cat?";
    let complex_query = "How do epigenetic modifications influence gene expression patterns during embryonic development?";
    
    let simple_entities = vec![];
    let complex_entities = vec![];
    
    let simple_complexity = assessor.assess_complexity(simple_query, &simple_entities).unwrap();
    let complex_complexity = assessor.assess_complexity(complex_query, &complex_entities).unwrap();
    
    assert!(matches!(simple_complexity.conceptual_depth, ConceptualDepth::Surface));
    assert!(matches!(complex_complexity.conceptual_depth, ConceptualDepth::Deep | ConceptualDepth::Expert));
}

#[test]
fn test_constraint_detection() {
    let detector = ConstraintDetector::new();
    
    let queries = vec![
        ("Show me 5 examples", ConstraintType::MaxResults),
        ("I need high confidence results", ConstraintType::MinConfidence),
        ("Recent research only", ConstraintType::DataRecency),
    ];
    
    for (query, expected_constraint) in queries {
        let constraints = detector.detect_constraints(query).unwrap();
        assert!(constraints.iter().any(|c| matches!(c.constraint_type, expected_constraint)));
    }
}
```

## Quality Gates

- [ ] Context analysis is deterministic for identical inputs
- [ ] Performance stable across different query types
- [ ] No memory leaks during continuous analysis
- [ ] Domain confidence scores are meaningful
- [ ] Ambiguity detection helps downstream processing

## Next Task

Upon completion, proceed to **11_query_decomposition.md**