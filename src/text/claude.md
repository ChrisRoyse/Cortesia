# Directory Overview: src/text

## 1. High-Level Summary

The `src/text` directory contains a comprehensive text processing module for the LLMKG (Large Language Model Knowledge Graph) system. This module provides ultra-fast text analysis, normalization, chunking, and structure prediction capabilities designed to prevent data bloat in the knowledge graph while maintaining sub-millisecond performance for text operations.

The module serves as the core text processing layer that transforms raw text into structured, normalized data suitable for knowledge graph operations. It emphasizes performance optimization with heuristic-based approaches rather than heavy ML models to ensure scalability.

## 2. Tech Stack

- **Language:** Rust
- **Frameworks:** Standard Rust library
- **External Dependencies:** 
  - `crate::error::Result` (internal error handling)
- **Performance Focus:** Sub-millisecond text processing
- **Architecture Pattern:** Trait-based design with multiple implementation strategies

## 3. Directory Structure

```
src/text/
├── mod.rs                  # Module definition with TextCompressor and utility functions
├── chunkers.rs            # Text chunking strategies (sliding window, semantic, adaptive)
├── importance.rs          # Graph-based importance scoring with heuristic analysis
├── normalizer.rs          # Fast string normalization and canonicalization
└── structure_predictor.rs # Graph structure prediction using pattern matching
```

## 4. File Breakdown

### `mod.rs`

**Purpose:** Main module file providing text compression to prevent data bloat and utility functions.

**Key Constants:**
- `MAX_NODE_WORDS: usize = 400` - Maximum words allowed per node
- `TARGET_SUMMARY_WORDS: usize = 75` - Target compression size (50-100 words)

**Main Struct:**
- `TextCompressor`
  - **Description:** Ultra-fast text summarizer for preventing knowledge graph data bloat
  - **Fields:**
    - `stop_words: Vec<&'static str>` - Common English stop words for filtering
  - **Methods:**
    - `new() -> Self` - Creates new compressor with predefined stop words
    - `compress(&self, text: &str) -> String` - Compresses text using TF-IDF-like scoring (MUST be sub-millisecond)
    - `validate_text_size(text: &str) -> Result<()>` - Validates text doesn't exceed MAX_NODE_WORDS limit
    - `calculate_word_scores(&self, words: &[&str]) -> HashMap<String, f32>` - Simplified TF-IDF scoring
    - `extract_sentences<'a>(&self, text: &'a str) -> Vec<&'a str>` - Fast sentence extraction
    - `select_top_sentences<'a>(&self, sentences: &[&'a str], word_scores: &HashMap<String, f32>) -> Vec<&'a str>` - Selects sentences by score
    - `build_summary(&self, sentences: &[&str], max_words: usize) -> String` - Constructs final summary

**Utility Functions:**
- `utils::word_count(text: &str) -> usize` - Counts words in text
- `utils::truncate_to_words(text: &str, max_words: usize) -> String` - Truncates text to word limit

### `chunkers.rs`

**Purpose:** Provides various text chunking strategies for breaking down large documents into manageable pieces.

**Core Types:**
- `TextChunk`
  - **Description:** Represents a text chunk with metadata
  - **Fields:**
    - `text: String` - The actual text content
    - `metadata: HashMap<String, String>` - Chunk metadata (offsets, size, type, etc.)

**Trait:**
- `Chunker`
  - **Method:** `chunk(&self, text: &str) -> Result<Vec<TextChunk>>` - Main chunking interface

**Implementations:**

#### `SlidingWindowChunker`
- **Description:** Creates overlapping chunks using sliding window approach
- **Fields:**
  - `window_size: usize` - Size of each chunk window
  - `overlap_size: usize` - Number of overlapping characters
- **Constructor:** `new(window_size: usize, overlap_size: usize) -> Self`
- **Metadata:** Includes start_offset, end_offset, chunk_size

#### `SemanticChunker`
- **Description:** Splits text at natural sentence boundaries
- **Fields:**
  - `max_chunk_size: usize` - Maximum size per chunk
  - `similarity_threshold: f32` - Semantic similarity threshold (currently unused in heuristic version)
- **Constructor:** `new(max_chunk_size: usize, similarity_threshold: f32) -> Self`
- **Methods:**
  - `find_sentence_boundaries(&self, text: &str) -> Vec<usize>` - Locates sentence endings (.!?)
- **Metadata:** Includes start_offset, end_offset, sentence_count

#### `AdaptiveChunker`
- **Description:** Adjusts chunk size based on content structure (paragraphs, chapters)
- **Fields:**
  - `min_chunk_size: usize` - Minimum acceptable chunk size
  - `max_chunk_size: usize` - Maximum chunk size limit
- **Constructor:** `new(min_chunk_size: usize, max_chunk_size: usize) -> Self`
- **Methods:**
  - `find_natural_breaks(&self, text: &str) -> Vec<usize>` - Identifies paragraph/chapter breaks
- **Metadata:** Includes chunk_index, chunk_type ("chapter", "paragraph", "single"), start/end offsets

### `importance.rs`

**Purpose:** Provides graph-based importance scoring for knowledge entities using heuristic methods and graph metrics.

**Core Types:**

#### `GraphMetrics`
- **Description:** Container for graph centrality metrics
- **Fields:**
  - `degree_centrality: f32` - Node degree centrality score
  - `betweenness_centrality: f32` - Betweenness centrality score
  - `closeness_centrality: f32` - Closeness centrality score
  - `pagerank_score: f32` - PageRank score (default: 0.1)
  - `clustering_coefficient: f32` - Local clustering coefficient
  - `connection_count: usize` - Total number of connections

#### `HeuristicImportanceScorer`
- **Description:** Main importance scoring engine using weighted heuristics
- **Fields:**
  - `keyword_weights: HashMap<String, f32>` - Predefined keyword importance weights
  - `graph_weight: f32 = 0.4` - Weight for graph-based scores
  - `keyword_weight: f32 = 0.3` - Weight for keyword-based scores
  - `frequency_weight: f32 = 0.3` - Weight for frequency-based scores

**Key Methods:**
- `calculate_importance(&self, text: &str, graph_metrics: Option<GraphMetrics>) -> f32` - Main scoring function
- `calculate_importance_async(&self, text: &str, graph_metrics: Option<GraphMetrics>) -> Result<f32>` - Async wrapper
- `calculate_batch_importance(&self, texts: &[String], metrics: &[Option<GraphMetrics>]) -> Result<Vec<f32>>` - Batch processing
- `calculate_contextual_importance(&self, text: &str, context: &str, graph_metrics: Option<GraphMetrics>) -> f32` - Context-aware scoring
- `calculate_relative_importance(&self, texts: &[String]) -> Vec<f32>` - Relative importance between texts
- `calculate_keyword_score(&self, text: &str) -> f32` - Keyword-based importance
- `calculate_frequency_score(&self, text: &str) -> f32` - TF-IDF-like frequency scoring
- `calculate_graph_score(&self, metrics: Option<GraphMetrics>) -> f32` - Graph centrality scoring
- `calculate_context_bonus(&self, text: &str, context: &str) -> f32` - Context overlap bonus

**Keyword Categories:** Technical terms (algorithm, system, framework), Knowledge graph terms (knowledge, graph, node, edge), AI/ML terms (machine learning, model, network), Data terms (database, storage, indexing), Business terms (strategy, solution, implementation)

### `normalizer.rs`

**Purpose:** Fast string normalization and canonicalization for consistent text processing.

#### `StringNormalizer`
- **Description:** Heuristic-based text normalizer for sub-millisecond performance
- **Fields:**
  - `stop_words: Vec<&'static str>` - Common English stop words
  - `synonyms: HashMap<String, String>` - Basic synonym mappings

**Key Methods:**
- `normalize(&self, text: &str) -> Result<String>` - Main normalization interface
- `canonicalize(&self, text: &str) -> Result<String>` - Alias for normalize (compatibility)
- `normalize_sync(&self, text: &str) -> String` - Synchronous normalization core
- `normalize_batch(&self, texts: &[String]) -> Result<Vec<String>>` - Batch processing
- `are_equivalent(&self, text1: &str, text2: &str) -> bool` - Canonical equivalence check
- `extract_key_terms(&self, text: &str) -> Vec<String>` - Extract significant terms (>3 chars)

**Normalization Process:**
1. Convert to lowercase
2. Remove punctuation (replace with spaces)
3. Normalize whitespace
4. Apply synonym substitutions
5. Filter stop words and short words (<3 chars)

**Synonym Categories:** Technical abbreviations (db→database, api→interface, ui→interface), Common contractions (can't→cannot, won't→will not)

### `structure_predictor.rs`

**Purpose:** Graph structure prediction using heuristic pattern matching and linguistic analysis.

**Core Types:**

#### `GraphOperation` (Enum)
- **Description:** Represents different graph construction operations
- **Variants:**
  - `CreateNode { id: String, node_type: String, properties: HashMap<String, String> }` - Node creation
  - `CreateEdge { from: String, to: String, relationship: String, weight: f32 }` - Edge creation
  - `MergeNodes { nodes: Vec<String>, target_id: String }` - Node merging
  - `InferRelationship { from: String, to: String, confidence: f32 }` - Relationship inference

#### `GraphStructurePredictor`
- **Description:** Heuristic-based graph structure prediction engine
- **Fields:**
  - `model_name: String` - Model identifier
  - `relationship_patterns: HashMap<String, Vec<String>>` - Relationship detection patterns
  - `entity_types: HashMap<String, f32>` - Entity type classification weights

**Key Methods:**
- `predict_structure(&self, text: &str) -> Result<Vec<GraphOperation>>` - Async structure prediction
- `predict_structure_sync(&self, text: &str) -> Vec<GraphOperation>` - Synchronous prediction core
- `predict_relationships(&self, entity1: &str, entity2: &str, context: &str) -> Result<Vec<(String, f32)>>` - Relationship prediction
- `predict_entity_types(&self, text: &str) -> Result<HashMap<String, String>>` - Entity type classification
- `predict_query_structure(&self, query: &str) -> Result<Vec<GraphOperation>>` - Query-optimized structure

**Entity Extraction Methods:**
- `extract_entities(&self, text: &str) -> HashMap<String, String>` - Pattern-based entity extraction
- `extract_relationships(&self, text: &str, entities: &HashMap<String, String>) -> Vec<(String, String, String, f32)>` - Relationship extraction
- `classify_entity_type(&self, word: &str) -> String` - Entity type classification

**Entity Types:** technical_concept, data_entity, process_entity, named_entity, concept, compound_entity

**Relationship Patterns:** has/contains/includes, is/was/are, uses/utilizes/employs, creates/generates/produces, connects/links/relates, depends/requires/needs, influences/affects/impacts, implements/realizes/executes

## 5. Key Variables and Logic

### Performance Constraints
- **Sub-millisecond Requirement:** All text operations must complete in under 1ms
- **Memory Efficiency:** Uses heuristic approaches instead of heavy ML models
- **Batch Processing:** Supports batch operations for improved throughput

### Scoring Algorithms
- **TF-IDF Simplified:** Uses `tf * ln(total_words / (freq + 1))` for frequency scoring
- **Graph Score Weighting:** 30% degree centrality + 20% betweenness + 20% closeness + 20% PageRank + 10% clustering
- **Importance Combination:** 40% graph metrics + 30% keywords + 30% frequency

### Text Processing Pipeline
1. **Input Validation:** Check word limits (MAX_NODE_WORDS = 400)
2. **Normalization:** Lowercase, punctuation removal, whitespace normalization
3. **Chunking:** Apply appropriate chunking strategy based on content
4. **Scoring:** Calculate importance using combined heuristics
5. **Structure Prediction:** Extract entities and relationships using pattern matching

## 6. Dependencies

### Internal Dependencies
- `crate::error::Result` - Error handling system
- `crate::error::GraphError` - Specific graph-related errors

### External Dependencies
- `std::collections::HashMap` - Key-value storage for mappings and metadata
- `std::time::Instant` - Performance measurement in tests

### Module Exports
```rust
pub use chunkers::{TextChunk, Chunker, SlidingWindowChunker, SemanticChunker, AdaptiveChunker};
pub use normalizer::StringNormalizer;
pub use importance::{HeuristicImportanceScorer, GraphMetrics};
pub use structure_predictor::{GraphStructurePredictor, GraphOperation};
```

## 7. Performance Characteristics

### Benchmarks (from tests)
- **Text Compression:** 1000 compressions in <5 seconds
- **Importance Scoring:** 1000 calculations in <100ms  
- **Normalization:** 1000 normalizations in <100ms
- **Structure Prediction:** 100 predictions in <1 second

### Memory Usage
- Minimal heap allocation through string reuse
- HashMap-based caching for repeated lookups
- No persistent model loading (heuristic-based)

### Scalability Features
- Batch processing APIs for improved throughput
- Stateless design for parallel processing
- Configurable thresholds and limits

## 8. Design Patterns

### Strategy Pattern
- `Chunker` trait with multiple implementations (SlidingWindow, Semantic, Adaptive)
- Each strategy optimized for different content types

### Builder Pattern
- Consistent constructor patterns across all main structs
- Default implementations provided for common use cases

### Pipeline Pattern
- Text processing flows through normalization → chunking → scoring → structure prediction
- Each stage can be used independently or in combination

### Heuristic-First Approach
- Prioritizes speed and determinism over ML accuracy
- Uses pattern matching and rule-based classification
- Fallback to baseline scores when patterns don't match

This text processing module forms the foundation for efficient knowledge graph construction by providing fast, reliable text analysis capabilities while maintaining strict performance requirements.