# Directory Overview: Hierarchical Storage System

## 1. High-Level Summary

The hierarchical storage system is a core component of the LLMKG (Large Language Model Knowledge Graph) that implements layered knowledge storage. It organizes information in a hierarchical structure: Document → Section → Paragraph → Sentence → Entity/Relationship/Concept. This system provides intelligent storage, semantic linking, efficient indexing, and retrieval capabilities for processed knowledge documents.

The system transforms flat document content into a rich, interconnected knowledge graph with multiple abstraction layers, enabling sophisticated querying, semantic search, and knowledge discovery through graph traversal.

## 2. Tech Stack

- **Language:** Rust
- **Frameworks:** 
  - Tokio (async runtime)
  - Tracing (structured logging)
- **Libraries:** 
  - Serde (serialization/deserialization)
  - std::collections (HashMap, HashSet)
  - std::sync (Arc, RwLock for thread-safe concurrent access)
- **Architecture:** Multi-layered async system with in-memory storage and semantic graph processing

## 3. Directory Structure

```
hierarchical_storage/
├── mod.rs              # Module entry point and public exports
├── types.rs            # Core type definitions and data structures
├── knowledge_layers.rs # Layer creation and management logic
├── semantic_links.rs   # Semantic link graph construction and analysis
├── hierarchical_index.rs # Indexing structures for efficient retrieval
└── storage_engine.rs   # Main storage engine coordination
```

## 4. File Breakdown

### `mod.rs`

- **Purpose:** Module entry point that re-exports all public interfaces
- **Exports:** All major components (`KnowledgeLayerManager`, `SemanticLinkManager`, `HierarchicalIndexManager`, `HierarchicalStorageEngine`, and all types)

### `types.rs`

- **Purpose:** Comprehensive type definitions for the hierarchical storage system
- **Key Structures:**
  - `HierarchicalKnowledge`: Top-level container with document metadata, layers, semantic links, and retrieval index
  - `KnowledgeLayer`: Individual layer with content, entities, relationships, and position information
  - `SemanticLinkGraph`: Graph structure with nodes and edges representing semantic relationships
  - `HierarchicalIndex`: Multi-dimensional indexing structure for efficient retrieval

- **Key Enums:**
  - `LayerType`: Document, Section, Paragraph, Sentence, Entity, Relationship, Concept
  - `SemanticLinkType`: Hierarchical, Sequential, Referential, Semantic, Causal, Temporal, Categorical, Comparative, Definitional, Explanatory
  - `SemanticNodeType`: LayerNode, EntityNode, ConceptNode, RelationshipNode, TopicNode

- **Key Methods:**
  - `HierarchicalKnowledge::new(document_id, global_context)`: Create new hierarchical knowledge structure
  - `LayerType::can_contain(child_type)`: Check hierarchical containment rules
  - `SemanticLinkType::is_directional()`: Determine if link type is directional
  - `SemanticLinkType::strength_range()`: Get typical strength range for link type

### `knowledge_layers.rs`

- **Purpose:** Manages creation and organization of hierarchical knowledge layers from processed document content
- **Classes:**
  - `KnowledgeLayerManager`
    - **Description:** Core manager for creating layered knowledge structures
    - **Methods:**
      - `new(model_manager, config)`: Create new layer manager
      - `create_hierarchical_layers(processing_result)`: Main entry point - creates complete layer hierarchy
      - `create_document_layer(processing_result)`: Creates top-level document layer
      - `create_section_layers(processing_result)`: Creates section-level layers from document structure
      - `create_paragraph_layers(processing_result)`: Creates paragraph layers from semantic chunks
      - `create_sentence_layers(paragraph_layers)`: Splits paragraphs into sentence layers
      - `create_entity_layers(processing_result)`: Creates entity-focused layers with all contexts
      - `create_relationship_layers(processing_result)`: Creates relationship-focused layers
      - `create_concept_layers(processing_result)`: Creates thematic concept groupings
      - `enhance_layers_with_analysis(layers)`: Adds semantic embeddings and importance scores
      - `establish_layer_hierarchy(layers)`: Links parent-child relationships

- **Key Logic:**
  - Progressive decomposition from document → sections → paragraphs → sentences
  - Cross-cutting layers for entities, relationships, and concepts
  - Importance scoring based on layer type, content length, and entity/relationship density
  - Semantic embedding generation using model manager
  - Coherence scoring based on entity and relationship confidence

### `semantic_links.rs`

- **Purpose:** Creates and manages semantic links between knowledge layers to preserve context and enable intelligent graph traversal
- **Classes:**
  - `SemanticLinkManager`
    - **Description:** Builds semantic relationship graphs between knowledge layers
    - **Methods:**
      - `new(model_manager, config)`: Create new semantic link manager
      - `build_semantic_link_graph(layers)`: Main entry point - builds complete semantic graph
      - `create_semantic_nodes(graph, layers)`: Creates nodes for each layer
      - `create_hierarchical_links(graph, layers)`: Links parent-child relationships
      - `create_sequential_links(graph, layers)`: Links sequential document flow
      - `create_referential_links(graph, layers)`: Links layers sharing entities
      - `create_semantic_similarity_links(graph, layers)`: Links semantically similar content
      - `create_causal_temporal_links(graph, layers)`: Identifies causal and temporal relationships
      - `create_categorical_links(graph, layers)`: Links similar layer types and concepts
      - `calculate_centrality_scores(graph)`: Computes graph centrality metrics
      - `optimize_graph(graph)`: Prunes weak links and consolidates edges

- **Key Logic:**
  - Multi-type link creation (10 different semantic link types)
  - Graph analysis with centrality scoring (degree, betweenness, closeness, eigenvector)
  - Embedding-based similarity computation using cosine similarity
  - Linguistic analysis for causal/temporal relationship detection
  - Graph optimization with configurable thresholds

### `hierarchical_index.rs`

- **Purpose:** Manages indexing structures for efficient retrieval across hierarchical knowledge layers
- **Classes:**
  - `HierarchicalIndexManager`
    - **Description:** Creates and maintains multi-dimensional indexes for fast retrieval
    - **Methods:**
      - `new(model_manager, config)`: Create new index manager
      - `build_hierarchical_index(layers)`: Main entry point - builds complete index
      - `build_layer_index(index, layers)`: Fast layer lookup by ID
      - `build_entity_index(index, layers)`: Maps entity names to layer IDs
      - `build_concept_index(index, layers)`: Maps concepts to layer IDs
      - `build_relationship_index(index, layers)`: Maps relationship types to layer IDs
      - `build_full_text_index(index, layers)`: Tokenized full-text search index
      - `build_semantic_index(index, layers)`: Semantic clustering for similarity search
      - `search_index(index, query)`: Multi-criteria search with scoring
      - `find_similar_layers(index, target_embedding, max_results)`: Semantic similarity search
      - `update_index_incremental(index, new_layers)`: Incremental index updates

  - `IndexQuery`
    - **Description:** Query structure for multi-criteria search
    - **Properties:** keywords, entities, concepts, relationship_types, layer_types, scoring weights
  
  - `SearchResult`
    - **Description:** Search result with layer ID, score, type, and matched terms

- **Key Logic:**
  - Six index types: layer, entity, concept, relationship, full-text, semantic
  - K-means clustering for semantic similarity grouping
  - Stop word filtering and term relevance scoring
  - Incremental index updates for new content
  - Configurable search weights and thresholds

### `storage_engine.rs`

- **Purpose:** Main storage engine that coordinates all hierarchical storage components and provides the primary interface
- **Classes:**
  - `HierarchicalStorageEngine`
    - **Description:** Primary interface coordinating all storage operations
    - **Methods:**
      - `new(model_manager, config)`: Create new storage engine
      - `store_knowledge(processing_result, global_context)`: Store complete hierarchical knowledge
      - `retrieve_document(document_id)`: Retrieve complete document with all layers
      - `retrieve_layer(layer_id)`: Retrieve specific layer by ID
      - `search(query)`: Multi-document search with IndexQuery
      - `find_similar(target_embedding, max_results)`: Semantic similarity search
      - `get_connected_layers(layer_id, link_types, max_hops)`: Graph traversal for connected content
      - `update_document(document_id, new_layers)`: Update existing document
      - `delete_document(document_id)`: Remove document and associated data
      - `get_storage_stats()`: Comprehensive system statistics

  - `InMemoryStorage`
    - **Description:** Internal storage container with caching and access logging
    - **Properties:** documents HashMap, layer_cache HashMap, access_log Vec

- **Key Logic:**
  - Five-step storage process: layers → semantic links → index → structure → memory
  - Comprehensive instrumentation and logging with tracing
  - LRU cache management with configurable size limits
  - Access logging for usage analytics
  - Performance metrics and timing analysis
  - Configurable thresholds and limits

## 5. Key Data Structures

### Core Storage Types
- **`HierarchicalKnowledge`**: Complete document representation with layers, semantic graph, and index
- **`KnowledgeLayer`**: Individual layer with content, metadata, entities, relationships, position, and scores
- **`LayerContent`**: Text content with processing metadata, key phrases, and summary
- **`SemanticLinkGraph`**: Graph with nodes (layers/entities/concepts) and edges (semantic relationships)

### Index Types
- **`HierarchicalIndex`**: Multi-dimensional index with layer, entity, concept, relationship, full-text, and semantic indexes
- **`SemanticCluster`**: Clustered content groups with center embeddings and coherence scores
- **`IndexMatch`**: Full-text search result with positions and relevance scoring

## 6. External Dependencies

### Internal Dependencies
- **`model_management::ModelResourceManager`**: AI model coordination for semantic analysis
- **`knowledge_processing::types::KnowledgeProcessingResult`**: Input from document processing pipeline
- **`logging::LogContext`**: Structured logging context management
- **`types::ComplexityLevel`**: Task complexity classification

### Key External Libraries
- **`tokio`**: Async runtime for concurrent operations
- **`tracing`**: Structured logging and instrumentation
- **`serde`**: JSON serialization for data structures
- **`std::sync`**: Thread-safe data structures (Arc, RwLock)

## 7. Configuration and Tuning

### `HierarchicalStorageConfig`
- **`max_layers_per_document`**: 1000 (prevents excessive layer creation)
- **`max_nodes_per_layer`**: 50 (limits node complexity)
- **`semantic_similarity_threshold`**: 0.7 (minimum similarity for linking)
- **`importance_score_threshold`**: 0.3 (minimum importance for inclusion)
- **`enable_semantic_clustering`**: true (enables semantic index)
- **`cache_size_limit`**: 10000 (maximum cached layers)

## 8. Key Algorithms

### Layer Creation Algorithm
1. Document analysis and global context extraction
2. Progressive decomposition: document → sections → paragraphs → sentences
3. Cross-cutting layer creation: entities → relationships → concepts
4. Semantic enhancement with embeddings and importance scoring
5. Hierarchy establishment with parent-child linking

### Semantic Link Construction
1. Node creation for each layer with type classification
2. Hierarchical link creation from explicit parent-child relationships
3. Sequential link creation based on document flow
4. Referential link creation from shared entities
5. Similarity link creation using embedding cosine similarity
6. Causal/temporal link detection using linguistic analysis
7. Categorical link creation for similar content types
8. Graph optimization with centrality analysis and weak link pruning

### Indexing Strategy
1. Multi-dimensional index construction: 6 parallel index types
2. Stop word filtering and term relevance scoring
3. K-means clustering for semantic similarity grouping
4. Incremental updates for new content integration
5. Cache management with LRU eviction strategy

## 9. Performance Characteristics

### Storage Complexity
- **Time**: O(n log n) for n layers (due to sorting and indexing)
- **Space**: O(n²) worst case for semantic links (dense graph)
- **Retrieval**: O(1) for direct layer access, O(log n) for search

### Scalability Considerations
- Configurable limits prevent excessive resource usage
- LRU caching reduces memory pressure
- Incremental index updates avoid full rebuilds
- Async operations enable concurrent processing

### Optimization Features
- Weak link pruning reduces graph complexity
- Semantic clustering improves similarity search
- Access logging enables usage-based optimization
- Comprehensive metrics for performance monitoring

## 10. Error Handling

### `HierarchicalStorageError` Types
- **`LayerNotFound`**: Missing layer or document
- **`InvalidLayerStructure`**: Malformed hierarchy or excessive layers
- **`IndexingError`**: Index construction or update failures
- **`SemanticAnalysisError`**: Embedding or similarity computation errors
- **`StorageError`**: Memory or persistence issues
- **`RetrievalError`**: Search or access failures
- **`GraphError`**: Semantic link construction errors
- **`ConfigurationError`**: Invalid configuration parameters

## 11. Usage Patterns

### Typical Storage Flow
```rust
// 1. Create storage engine
let engine = HierarchicalStorageEngine::new(model_manager, config);

// 2. Store processed knowledge
let doc_id = engine.store_knowledge(processing_result, global_context).await?;

// 3. Search for relevant content
let query = IndexQuery { keywords: Some(vec!["AI".to_string()]), ..Default::default() };
let results = engine.search(query).await?;

// 4. Retrieve connected layers
let connected = engine.get_connected_layers("layer_1", None, 2).await?;
```

### Typical Search Patterns
- **Keyword Search**: Full-text search with relevance scoring
- **Entity Search**: Find all layers mentioning specific entities
- **Concept Search**: Thematic content discovery
- **Semantic Search**: Embedding-based similarity matching
- **Graph Traversal**: Multi-hop relationship exploration

This hierarchical storage system provides a sophisticated foundation for knowledge management with semantic understanding, efficient retrieval, and scalable architecture suitable for large-scale document processing and knowledge discovery applications.