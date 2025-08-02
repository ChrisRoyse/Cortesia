# Micro Task 07: Query Intent Types

**Priority**: CRITICAL  
**Estimated Time**: 45 minutes  
**Dependencies**: 06_activation_tests.md completed  
**Skills Required**: Natural language processing, type system design

## Objective

Define a comprehensive type system for classifying different query intents to enable intelligent query processing.

## Context

Query intent classification is the foundation of intelligent query processing. By understanding what the user wants to achieve, we can apply appropriate activation patterns and processing strategies.

## Specifications

### Intent Type Hierarchy

1. **Core Intent Categories**
   - Filter queries (find entities matching criteria)
   - Relationship queries (explore connections)
   - Hierarchy queries (navigate structures)
   - Comparison queries (contrast entities)
   - Causal queries (understand cause-effect)
   - Definition queries (describe entities)

2. **Intent Complexity Levels**
   - Simple (single-step)
   - Compound (multi-step)
   - Complex (requires decomposition)

3. **Context Information**
   - Domain specification
   - Temporal constraints
   - Spatial limitations
   - Confidence requirements

## Implementation Guide

### Step 1: Define Core Types
```rust
// File: src/query/intent_types.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryIntent {
    Filter {
        entity_type: String,
        property: String,
        value: String,
        operator: FilterOperator,
    },
    Relationship {
        entity1: String,
        entity2: String,
        relation_type: RelationType,
        direction: RelationDirection,
    },
    Hierarchy {
        root_entity: String,
        direction: HierarchyDirection,
        depth_limit: Option<usize>,
    },
    Comparison {
        entities: Vec<String>,
        aspect: String,
        comparison_type: ComparisonType,
    },
    Causal {
        cause: String,
        effect: String,
        mechanism: Option<String>,
    },
    Definition {
        entity: String,
        detail_level: DetailLevel,
    },
    Aggregation {
        entities: String,
        operation: AggregateOperation,
        grouping: Option<String>,
    },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    Contains,
    GreaterThan,
    LessThan,
    IsA,
    HasProperty,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationType {
    Similarity,
    Dependency,
    Inheritance,
    Association,
    Composition,
    Temporal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HierarchyDirection {
    Ancestors,
    Descendants,
    Siblings,
    Both,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonType {
    Similarities,
    Differences,
    Ranking,
    Classification,
}
```

### Step 2: Add Context and Metadata
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    pub domain: Option<String>,
    pub temporal_range: Option<TimeRange>,
    pub spatial_bounds: Option<SpatialBounds>,
    pub confidence_threshold: f32,
    pub max_results: Option<usize>,
    pub language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedQuery {
    pub original_query: String,
    pub intent_type: QueryIntent,
    pub entities: Vec<ExtractedEntity>,
    pub context: QueryContext,
    pub confidence: f32,
    pub sub_queries: Vec<ParsedQuery>,
    pub complexity: ComplexityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: Option<String>,
    pub aliases: Vec<String>,
    pub confidence: f32,
    pub span: TextSpan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Simple,    // Single intent, clear entities
    Compound,  // Multiple related intents
    Complex,   // Requires decomposition
}
```

### Step 3: Intent Classification Logic
```rust
impl QueryIntent {
    pub fn requires_spreading_activation(&self) -> bool {
        match self {
            QueryIntent::Filter { .. } => true,
            QueryIntent::Relationship { .. } => true,
            QueryIntent::Hierarchy { .. } => true,
            QueryIntent::Comparison { .. } => true,
            QueryIntent::Causal { .. } => true,
            QueryIntent::Definition { .. } => false, // Direct lookup
            QueryIntent::Aggregation { .. } => true,
            QueryIntent::Unknown => false,
        }
    }
    
    pub fn suggested_activation_strategy(&self) -> ActivationStrategy {
        match self {
            QueryIntent::Filter { .. } => ActivationStrategy::Focused,
            QueryIntent::Relationship { .. } => ActivationStrategy::Bidirectional,
            QueryIntent::Hierarchy { .. } => ActivationStrategy::Directional,
            QueryIntent::Comparison { .. } => ActivationStrategy::Parallel,
            QueryIntent::Causal { .. } => ActivationStrategy::ChainTracing,
            QueryIntent::Definition { .. } => ActivationStrategy::Local,
            QueryIntent::Aggregation { .. } => ActivationStrategy::BreadthFirst,
            QueryIntent::Unknown => ActivationStrategy::Conservative,
        }
    }
    
    pub fn expected_result_count(&self) -> ResultCountHint {
        match self {
            QueryIntent::Definition { .. } => ResultCountHint::Single,
            QueryIntent::Hierarchy { depth_limit, .. } => {
                match depth_limit {
                    Some(d) if *d <= 2 => ResultCountHint::Few,
                    _ => ResultCountHint::Many,
                }
            }
            QueryIntent::Filter { .. } => ResultCountHint::Variable,
            QueryIntent::Comparison { entities, .. } => {
                ResultCountHint::Fixed(entities.len())
            }
            _ => ResultCountHint::Many,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ActivationStrategy {
    Focused,      // High initial activation, low spread
    Bidirectional, // Spread from both entities
    Directional,   // Follow hierarchy direction
    Parallel,      // Multiple independent spreads
    ChainTracing,  // Follow causal chains
    Local,         // Minimal spreading
    BreadthFirst,  // Wide shallow spreading
    Conservative,  // Default safe strategy
}

#[derive(Debug, Clone)]
pub enum ResultCountHint {
    Single,
    Few,
    Many,
    Variable,
    Fixed(usize),
}
```

### Step 4: Intent Pattern Matching
```rust
pub struct IntentPatternMatcher {
    patterns: Vec<IntentPattern>,
}

#[derive(Debug, Clone)]
pub struct IntentPattern {
    pub keywords: Vec<String>,
    pub structure_pattern: String,
    pub intent_type: QueryIntent,
    pub confidence_weight: f32,
}

impl IntentPatternMatcher {
    pub fn new() -> Self {
        Self {
            patterns: Self::create_default_patterns(),
        }
    }
    
    fn create_default_patterns() -> Vec<IntentPattern> {
        vec![
            IntentPattern {
                keywords: vec!["what".into(), "which".into(), "find".into()],
                structure_pattern: "WH + VERB + ENTITY + PROPERTY".into(),
                intent_type: QueryIntent::Filter {
                    entity_type: "".into(),
                    property: "".into(),
                    value: "".into(),
                    operator: FilterOperator::HasProperty,
                },
                confidence_weight: 0.8,
            },
            IntentPattern {
                keywords: vec!["how".into(), "related".into(), "connected".into()],
                structure_pattern: "HOW + ENTITY + RELATED + ENTITY".into(),
                intent_type: QueryIntent::Relationship {
                    entity1: "".into(),
                    entity2: "".into(),
                    relation_type: RelationType::Association,
                    direction: RelationDirection::Bidirectional,
                },
                confidence_weight: 0.9,
            },
            IntentPattern {
                keywords: vec!["compare".into(), "difference".into(), "versus".into()],
                structure_pattern: "COMPARE + ENTITY + AND + ENTITY".into(),
                intent_type: QueryIntent::Comparison {
                    entities: vec![],
                    aspect: "".into(),
                    comparison_type: ComparisonType::Differences,
                },
                confidence_weight: 0.85,
            },
        ]
    }
    
    pub fn match_patterns(&self, query: &str) -> Vec<(QueryIntent, f32)> {
        let query_lower = query.to_lowercase();
        let mut matches = Vec::new();
        
        for pattern in &self.patterns {
            let keyword_matches = pattern.keywords.iter()
                .filter(|keyword| query_lower.contains(keyword.as_str()))
                .count();
            
            if keyword_matches > 0 {
                let confidence = (keyword_matches as f32 / pattern.keywords.len() as f32) 
                    * pattern.confidence_weight;
                
                matches.push((pattern.intent_type.clone(), confidence));
            }
        }
        
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        matches
    }
}
```

## File Locations

- `src/query/intent_types.rs` - Core type definitions
- `src/query/patterns.rs` - Pattern matching logic
- `src/query/mod.rs` - Module exports
- `tests/query/intent_types_tests.rs` - Test implementation

## Success Criteria

- [ ] All intent types defined correctly
- [ ] Pattern matching works for common queries
- [ ] Type system is extensible
- [ ] Context information captured properly
- [ ] Performance acceptable (< 1ms classification)
- [ ] All tests pass

## Test Requirements

```rust
#[test]
fn test_filter_intent_classification() {
    let matcher = IntentPatternMatcher::new();
    
    let queries = vec![
        "What animals can fly?",
        "Find all red cars",
        "Which movies are comedies?",
    ];
    
    for query in queries {
        let matches = matcher.match_patterns(query);
        assert!(!matches.is_empty());
        
        let (intent, confidence) = &matches[0];
        assert!(matches!(intent, QueryIntent::Filter { .. }));
        assert!(*confidence > 0.5);
    }
}

#[test]
fn test_relationship_intent_classification() {
    let matcher = IntentPatternMatcher::new();
    
    let queries = vec![
        "How are dogs related to wolves?",
        "What's the connection between A and B?",
        "Show relationships between humans and primates",
    ];
    
    for query in queries {
        let matches = matcher.match_patterns(query);
        let (intent, _) = &matches[0];
        assert!(matches!(intent, QueryIntent::Relationship { .. }));
    }
}

#[test]
fn test_activation_strategy_selection() {
    let filter_intent = QueryIntent::Filter {
        entity_type: "animals".into(),
        property: "can_fly".into(),
        value: "true".into(),
        operator: FilterOperator::Equals,
    };
    
    let strategy = filter_intent.suggested_activation_strategy();
    assert!(matches!(strategy, ActivationStrategy::Focused));
    
    let relationship_intent = QueryIntent::Relationship {
        entity1: "dogs".into(),
        entity2: "wolves".into(),
        relation_type: RelationType::Similarity,
        direction: RelationDirection::Bidirectional,
    };
    
    let strategy = relationship_intent.suggested_activation_strategy();
    assert!(matches!(strategy, ActivationStrategy::Bidirectional));
}

#[test]
fn test_context_extraction() {
    let context = QueryContext {
        domain: Some("biology".into()),
        temporal_range: None,
        spatial_bounds: None,
        confidence_threshold: 0.8,
        max_results: Some(10),
        language: "en".into(),
    };
    
    assert_eq!(context.domain.unwrap(), "biology");
    assert_eq!(context.confidence_threshold, 0.8);
}
```

## Quality Gates

- [ ] Type system compiles without warnings
- [ ] Serialization/deserialization works correctly
- [ ] Pattern matching is case-insensitive
- [ ] No false positive intent classifications
- [ ] Extensible for new intent types

## Next Task

Upon completion, proceed to **08_intent_parser_llm.md**