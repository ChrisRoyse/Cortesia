# Directory Overview: extraction

## 1. High-Level Summary

The `extraction` directory implements a comprehensive Natural Language Processing (NLP) system for extracting entities and relationships from text. It provides both simple regex-based extraction for basic use cases and advanced NLP processing with multiple specialized models. The system is designed to extract structured knowledge (entities and relations) from unstructured text and convert it into triples for knowledge graph storage.

## 2. Tech Stack

*   **Language:** Rust
*   **Async Framework:** async-trait for trait implementations
*   **Text Processing:** regex crate for pattern matching
*   **Serialization:** serde for data structure serialization/deserialization
*   **Internal Dependencies:** 
    *   `crate::core::triple::Triple` - Core triple data structure
    *   `crate::error::{GraphError, Result}` - Error handling system

## 3. Directory Structure

*   `mod.rs` - Module definition and trait interfaces
*   `advanced_nlp.rs` - Advanced NLP processing implementations

## 4. File Breakdown

### `mod.rs`

*   **Purpose:** Module entry point that defines core traits and provides simple implementations
*   **Key Exports:** Re-exports all major types from `advanced_nlp` module
*   **Traits:**
    *   `EntityExtractor`
        *   **Description:** Async trait for entity extraction functionality
        *   **Methods:**
            *   `extract_entities(text: &str) -> Result<Vec<Entity>>`: Extract entities from text
            *   `extract_entities_with_confidence(text: &str) -> Result<Vec<(Entity, f32)>>`: Extract entities with confidence scores
*   **Implementations:**
    *   **For `&str`**: Simple regex-based extraction using capitalized words heuristic
    *   **For `String`**: Delegates to `&str` implementation
    *   **For `AdvancedEntityExtractor`**: Delegates to the advanced implementation

### `advanced_nlp.rs`

*   **Purpose:** Advanced NLP processing system with multiple specialized extraction models
*   **Classes:**

#### `AdvancedEntityExtractor`
*   **Description:** Main orchestrator for advanced entity and relation extraction
*   **Key Fields:**
    *   `ner_models: HashMap<String, Arc<dyn NERModel>>` - Collection of NER models by type
    *   `entity_linker: EntityLinker` - Links entities to existing knowledge graph
    *   `coreference_resolver: CoreferenceResolver` - Resolves pronouns to entities
    *   `relation_extractor: RelationExtractor` - Extracts relationships between entities
*   **Methods:**
    *   `new() -> Self`: Creates instance with built-in NER models (person, location, organization, misc)
    *   `extract_entities(text: &str) -> Result<Vec<Entity>>`: Full entity extraction pipeline
    *   `extract_relations(text: &str, entities: &[Entity]) -> Result<Vec<Relation>>`: Extract relations between entities
    *   `extract_triples(text: &str) -> Result<Vec<Triple>>`: Complete extraction to knowledge graph triples
    *   `merge_entities(entities: Vec<Entity>) -> Result<Vec<Entity>>`: Deduplication and merging logic
    *   `merge_entity_group(entities: Vec<Entity>) -> Result<Entity>`: Merges overlapping entities

#### Named Entity Recognition Models

**`PersonNERModel`**
*   **Description:** Extracts person names using regex patterns
*   **Patterns:** First Last, First M. Last, Dr. Name, Prof. Name
*   **Confidence:** 0.8

**`LocationNERModel`**
*   **Description:** Extracts location names and places
*   **Patterns:** City, State format, City Name, Universities
*   **Confidence:** 0.7

**`OrganizationNERModel`**
*   **Description:** Extracts organization and company names
*   **Patterns:** Inc/Corp/LLC/Ltd suffixes, Company/Corporation, Institute/Foundation
*   **Confidence:** 0.75

**`MiscNERModel`**
*   **Description:** Extracts miscellaneous entities (dates, concepts, awards)
*   **Patterns:** 4-digit years, Scientific theories/principles/laws, Prizes/Awards/Medals
*   **Confidence:** 0.6
*   **Dynamic Typing:** Classifies as DATE, CONCEPT, or MISC based on content

#### Supporting Components

**`EntityLinker`**
*   **Description:** Links extracted entities to existing knowledge graph nodes
*   **Functionality:** Normalizes entity names, removes prefixes (Dr., Prof.)
*   **Threshold:** 0.8 similarity threshold for linking

**`CoreferenceResolver`**
*   **Description:** Resolves pronouns to their referent entities
*   **Approach:** Simple rule-based resolution using gender/type heuristics
*   **Pronouns Handled:** he/she/it/they/this/that

**`RelationExtractor`**
*   **Description:** Extracts semantic relationships between entities
*   **Components:**
    *   `relation_models: Vec<Arc<dyn RelationModel>>` - Multiple extraction models
    *   `predicate_normalizer: PredicateNormalizer` - Normalizes relation predicates
    *   `confidence_scorer: ConfidenceScorer` - Scores relation confidence
*   **Methods:**
    *   `extract_relations(text: &str, entities: &[Entity]) -> Result<Vec<Relation>>`: Main extraction method
    *   `filter_relations(relations: Vec<Relation>) -> Vec<Relation>`: Deduplication and filtering

#### Relation Models

**`PatternBasedRelationModel`**
*   **Description:** Extracts relations using regex patterns
*   **Patterns:**
    *   "X is/was a Y" → `is` relation
    *   "X invented/created/developed Y" → `invented` relation
    *   "X works at/employed by Y" → `works_at` relation
    *   "X born in/from Y" → `born_in` relation

**`DependencyRelationModel`**
*   **Description:** Placeholder for dependency parsing-based extraction
*   **Status:** Currently returns empty results (future implementation)

#### Utility Components

**`PredicateNormalizer`**
*   **Description:** Normalizes relation predicates to canonical forms
*   **Normalizations:**
    *   "is a", "was a", "are", "were" → "is"
    *   "works at", "employed by" → "works_at"
    *   "born in", "from" → "born_in"

**`ConfidenceScorer`**
*   **Description:** Calculates confidence scores for extracted relations
*   **Scoring Factors:**
    *   Base score: 0.5
    *   Evidence length bonus: +0.1 if > 20 chars
    *   Common predicate bonus: +0.2 for is/works_at/born_in/invented
    *   Long predicate penalty: -0.2 if > 20 chars

## 5. Data Structures

### `Entity`
*   **Fields:**
    *   `id: String` - Unique identifier
    *   `text: String` - Original text span
    *   `canonical_name: String` - Normalized name
    *   `entity_type: String` - Entity category (PERSON, LOCATION, etc.)
    *   `start_pos: usize` - Start position in text
    *   `end_pos: usize` - End position in text
    *   `confidence: f32` - Extraction confidence (0.0-1.0)
    *   `source_model: String` - Model that extracted this entity
    *   `linked_id: Option<String>` - Link to knowledge graph node
    *   `properties: HashMap<String, String>` - Additional properties

### `Relation`
*   **Fields:**
    *   `subject_id: String` - Subject entity ID
    *   `predicate: String` - Relation type/predicate
    *   `object_id: String` - Object entity ID
    *   `confidence: f32` - Extraction confidence
    *   `evidence: String` - Text evidence for the relation
    *   `extraction_model: String` - Model that extracted this relation

### `RelationPattern`
*   **Fields:**
    *   `pattern: Regex` - Regex pattern for matching
    *   `predicate: String` - Relation predicate to assign
    *   `subject_group: usize` - Regex group for subject
    *   `object_group: usize` - Regex group for object

## 6. Key Traits

### `NERModel`
*   **Purpose:** Interface for Named Entity Recognition models
*   **Methods:**
    *   `extract_entities(text: &str) -> Result<Vec<Entity>>`: Extract entities from text
    *   `get_model_name() -> &str`: Return model identifier
    *   `get_supported_types() -> Vec<&str>`: Return supported entity types

### `RelationModel`
*   **Purpose:** Interface for relation extraction models
*   **Methods:**
    *   `extract_relations(text: &str, entities: &[Entity]) -> Result<Vec<Relation>>`: Extract relations

### `EntityExtractor`
*   **Purpose:** Main interface for entity extraction functionality
*   **Implementations:** Available for `&str`, `String`, and `AdvancedEntityExtractor`

## 7. Processing Pipeline

1. **Coreference Resolution:** Resolve pronouns to entity references
2. **Entity Extraction:** Apply multiple NER models in parallel
3. **Entity Merging:** Deduplicate and merge overlapping entities
4. **Entity Linking:** Link entities to existing knowledge graph nodes
5. **Relation Extraction:** Extract relationships using multiple models
6. **Predicate Normalization:** Normalize relation predicates
7. **Confidence Scoring:** Score relation confidence
8. **Filtering:** Remove duplicates and low-confidence relations
9. **Triple Generation:** Convert to knowledge graph triples

## 8. Dependencies

*   **Internal:**
    *   `crate::core::triple::Triple` - Knowledge graph triple structure
    *   `crate::error::{GraphError, Result}` - Error handling system
*   **External:**
    *   `regex` - Regular expression processing
    *   `async_trait` - Async trait support
    *   `serde` - Serialization/deserialization
*   **Standard Library:**
    *   `std::collections::{HashMap, HashSet}` - Collection types
    *   `std::sync::Arc` - Atomic reference counting

## 9. Usage Examples

### Basic Entity Extraction
```rust
let extractor = AdvancedEntityExtractor::new();
let entities = extractor.extract_entities("Dr. John Smith works at Microsoft Corporation.").await?;
```

### Full Knowledge Extraction
```rust
let extractor = AdvancedEntityExtractor::new();
let triples = extractor.extract_triples("Einstein invented the theory of relativity.").await?;
```

### Simple Extraction
```rust
let text = "Simple text";
let entities = text.extract_entities("John works in Boston").await?;
```

## 10. Configuration and Extensibility

*   **NER Models:** New models can be added by implementing the `NERModel` trait
*   **Relation Models:** New relation extraction approaches via `RelationModel` trait
*   **Confidence Thresholds:** Entity linking threshold (0.8), relation filtering threshold (0.3)
*   **Pattern Extension:** New regex patterns can be added to existing models
*   **Normalization Rules:** Predicate normalizations can be extended in `PredicateNormalizer`