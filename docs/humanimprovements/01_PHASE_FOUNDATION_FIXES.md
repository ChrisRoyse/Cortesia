# Phase 1: Foundation Fixes

## Overview
**Duration**: 4 weeks  
**Goal**: Fix critical issues preventing basic functionality  
**Priority**: CRITICAL  

## Week 1: Entity Extraction Overhaul

### Task 1.1: Implement Multi-Word Entity Recognition
**File**: `src/core/entity_extractor.rs` (new file)
```rust
// Implementation details:
- Use rust-bert or similar NLP library
- Implement Named Entity Recognition (NER)
- Support compound entities (e.g., "Albert Einstein", "Theory of Relativity")
- Add entity type detection (Person, Place, Organization, Concept)
```

**Acceptance Criteria**:
- Correctly extracts "Albert Einstein" as single entity
- Identifies entity types with 90%+ accuracy
- Handles edge cases (possessives, abbreviations)

### Task 1.2: Refactor Knowledge Storage
**File**: `src/core/knowledge_types.rs`
```rust
pub struct Entity {
    pub id: Uuid,
    pub name: String,
    pub entity_type: EntityType,
    pub aliases: Vec<String>,
    pub context: Option<String>,
}

pub enum EntityType {
    Person,
    Place,
    Organization,
    Concept,
    Event,
    Time,
    Quantity,
    Unknown,
}
```

### Task 1.3: Update Entity Extraction Pipeline
**Files**: 
- `src/mcp/llm_friendly_server/handlers/storage.rs`
- `src/core/knowledge_engine.rs`

**Changes**:
- Replace simple word tokenization with NER
- Add entity disambiguation
- Implement coreference resolution

## Week 2: Relationship Extraction Enhancement

### Task 2.1: Implement Semantic Relationship Extraction
**File**: `src/core/relationship_extractor.rs` (new file)
```rust
pub struct RelationshipExtractor {
    patterns: Vec<RelationshipPattern>,
    dependency_parser: DependencyParser,
}

impl RelationshipExtractor {
    pub fn extract_relationships(&self, text: &str) -> Vec<Relationship> {
        // Use dependency parsing to find subject-verb-object
        // Match against known relationship patterns
        // Extract context window around relationship
    }
}
```

### Task 2.2: Create Relationship Ontology
**File**: `src/core/relationship_types.rs`
```rust
pub enum RelationshipType {
    // Causal
    Causes,
    CausedBy,
    Prevents,
    EnableS,
    
    // Temporal
    Before,
    After,
    During,
    SimultaneousWith,
    
    // Hierarchical
    IsA,
    PartOf,
    Contains,
    BelongsTo,
    
    // Action
    Created,
    Discovered,
    Invented,
    Developed,
    
    // Association
    RelatedTo,
    SimilarTo,
    OppositeTo,
    WorksWith,
}
```

### Task 2.3: Integrate Relationship Extraction
**Files**: 
- `src/mcp/llm_friendly_server/handlers/storage.rs`
- Update `store_knowledge` to use new extraction

## Week 3: Question Answering Implementation

### Task 3.1: Implement Question Parser
**File**: `src/core/question_parser.rs` (new file)
```rust
pub struct QuestionParser {
    pub fn parse(question: &str) -> QuestionIntent {
        // Identify question type (What, Who, When, Where, Why, How)
        // Extract key entities
        // Determine expected answer type
    }
}

pub struct QuestionIntent {
    question_type: QuestionType,
    entities: Vec<String>,
    expected_answer_type: AnswerType,
    temporal_context: Option<TimeRange>,
}
```

### Task 3.2: Build Answer Generation
**File**: `src/core/answer_generator.rs` (new file)
```rust
pub struct AnswerGenerator {
    pub fn generate_answer(
        facts: Vec<Fact>,
        intent: QuestionIntent,
    ) -> Answer {
        // Group related facts
        // Order by relevance
        // Generate natural language response
        // Include confidence score
    }
}
```

### Task 3.3: Fix ask_question Handler
**File**: `src/mcp/llm_friendly_server/handlers/storage.rs`
```rust
pub async fn handle_ask_question(params: Value) -> Result<Value> {
    let question = params["question"].as_str().unwrap_or("");
    let context = params.get("context").and_then(|v| v.as_str());
    
    // Parse question
    let intent = QuestionParser::parse(question);
    
    // Search for relevant facts
    let facts = search_facts_by_intent(&intent);
    
    // Generate answer
    let answer = AnswerGenerator::generate_answer(facts, intent);
    
    Ok(json!({
        "question": question,
        "answer": answer.text,
        "confidence": answer.confidence,
        "supporting_facts": answer.facts,
        "relevant_entities": answer.entities,
    }))
}
```

## Week 4: Testing and Stabilization

### Task 4.1: Comprehensive Test Suite
**File**: `tests/foundation_tests.rs`
```rust
#[cfg(test)]
mod entity_extraction_tests {
    #[test]
    fn test_person_entity_extraction() {
        let text = "Albert Einstein developed the Theory of Relativity";
        let entities = extract_entities(text);
        assert!(entities.contains("Albert Einstein"));
        assert!(entities.contains("Theory of Relativity"));
    }
    
    #[test]
    fn test_complex_entities() {
        // Test cases for various entity types
        // Edge cases and error handling
    }
}

#[cfg(test)]
mod relationship_extraction_tests {
    #[test]
    fn test_verb_relationships() {
        let text = "Einstein invented E=mc²";
        let rels = extract_relationships(text);
        assert_eq!(rels[0].predicate, "invented");
    }
}

#[cfg(test)]
mod question_answering_tests {
    #[test]
    fn test_what_questions() {
        store_fact("Einstein", "developed", "Relativity");
        let answer = ask_question("What did Einstein develop?");
        assert!(answer.contains("Relativity"));
    }
}
```

### Task 4.2: Performance Benchmarking
**File**: `benches/foundation_bench.rs`
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn entity_extraction_benchmark(c: &mut Criterion) {
    c.bench_function("extract entities 1k chars", |b| {
        b.iter(|| extract_entities(black_box(SAMPLE_TEXT_1K)))
    });
}

fn question_answering_benchmark(c: &mut Criterion) {
    c.bench_function("answer simple question", |b| {
        b.iter(|| ask_question(black_box("What is quantum mechanics?")))
    });
}
```

### Task 4.3: Documentation Update
**Files**:
- `README.md` - Update with new capabilities
- `docs/API.md` - Document enhanced endpoints
- `docs/EXAMPLES.md` - Provide usage examples

### Task 4.4: Migration Tools
**File**: `src/tools/migration.rs`
```rust
pub fn migrate_v1_to_v2() {
    // Re-process existing entities with new extractor
    // Update relationship types
    // Rebuild indices
}
```

## Deliverables
1. **Working entity extraction** with multi-word support
2. **Semantic relationship extraction** with typed relationships
3. **Functional question answering** returning relevant answers
4. **Comprehensive test coverage** > 90%
5. **Performance benchmarks** showing < 100ms response times
6. **Migration guide** for existing data

## Success Criteria
- [ ] Entity extraction accuracy > 95% on test corpus
- [ ] Relationship extraction identifies 20+ relationship types
- [ ] Question answering returns relevant results 85%+ of the time
- [ ] All existing tests pass
- [ ] No performance regression
- [ ] Documentation complete

## Dependencies
- rust-bert or similar NLP library
- Enhanced test corpus with ground truth
- Development environment with GPU support

## Risks & Mitigations
1. **NLP library integration complexity**
   - Mitigation: Start with simple patterns, gradually add ML
2. **Performance impact of NLP**
   - Mitigation: Implement caching, batch processing
3. **Breaking API changes**
   - Mitigation: Version endpoints, provide compatibility layer