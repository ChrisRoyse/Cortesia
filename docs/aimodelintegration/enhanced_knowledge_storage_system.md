# Enhanced Knowledge Storage System: Solving RAG Context Problems

## Executive Summary

The current LLMKG `store_knowledge` implementation suffers from critical RAG (Retrieval Augmented Generation) problems that severely limit knowledge retrieval effectiveness. This document outlines a comprehensive solution using small language models for enhanced knowledge processing while preserving anti-bloat optimizations for the main LLM interface.

## Current Problems Analysis

### 1. Context Fragmentation Crisis
- **Hard 2KB chunk limit** breaks sentences mid-way, destroying semantic coherence
- **Simple pattern matching** (capitalized words only) misses 70%+ of actual entities
- **Primitive relationship extraction** (only "is"/"has" patterns) ignores complex connections
- **No hierarchical organization** treats all knowledge as flat, isolated chunks
- **Boundary information loss** - critical context spanning chunks gets severed

### 2. Retrieval Failures
- **Semantic coherence breakdown** due to forced fragmentation
- **Context loss at boundaries** prevents understanding of complex topics
- **Global context destruction** eliminates document-wide understanding
- **Over-chunking computational waste** with fragmented, low-value pieces
- **Under-contextualized responses** due to isolated information retrieval

## Solution Architecture: Hierarchical Semantic Knowledge Graph

### Core Philosophy
**"Let small models handle complexity, deliver simplicity to main LLM"**

- Use SmolLM/MiniLM models for deep knowledge processing and organization
- Present clean, contextually-rich, structured information to main LLM
- Maintain anti-bloat for tool responses while enriching internal storage

### Key Components

#### 1. **Intelligent Knowledge Processor** 
*Using SmolLM-360M-Instruct or SmolLM2-360M*

```rust
pub struct IntelligentKnowledgeProcessor {
    model: SmolLMVariant,
    entity_extractor: AdvancedEntityExtractor,
    relationship_mapper: RelationshipMapper,
    context_analyzer: ContextAnalyzer,
    semantic_chunker: SemanticChunker,
}
```

**Capabilities:**
- **Advanced Entity Recognition**: Using instruction-tuned SmolLM for NLP tasks
- **Complex Relationship Extraction**: Multi-pattern recognition beyond "is/has"
- **Semantic Boundary Detection**: Intelligent chunking that preserves meaning
- **Context Preservation**: Maintains cross-reference links between related content

#### 2. **Hierarchical Knowledge Structure**

```rust
pub struct HierarchicalKnowledge {
    pub document_id: String,
    pub global_context: GlobalContext,
    pub knowledge_layers: Vec<KnowledgeLayer>,
    pub semantic_links: SemanticLinkGraph,
    pub retrieval_index: HierarchicalIndex,
}

pub struct KnowledgeLayer {
    pub level: LayerType,  // Document, Section, Paragraph, Sentence, Entity
    pub content: LayerContent,
    pub parent_refs: Vec<String>,
    pub child_refs: Vec<String>,
    pub semantic_embedding: Vec<f32>,
    pub cross_references: Vec<CrossReference>,
}

pub enum LayerType {
    Document,      // Full document context and summary
    Section,       // Major sections with titles and themes  
    Paragraph,     // Coherent thought units
    Sentence,      // Individual statements
    Entity,        // Extracted entities with rich context
    Relationship,  // Complex multi-entity relationships
}
```

#### 3. **Semantic Link Graph**

```rust
pub struct SemanticLinkGraph {
    pub entity_relationships: HashMap<String, Vec<EntityRelation>>,
    pub conceptual_connections: HashMap<String, Vec<ConceptualLink>>,
    pub document_references: HashMap<String, Vec<DocumentReference>>,
    pub temporal_links: HashMap<String, Vec<TemporalConnection>>,
}

pub struct EntityRelation {
    pub source: String,
    pub predicate: String,
    pub target: String,
    pub context: String,           // Original sentence/paragraph context
    pub confidence: f32,
    pub supporting_evidence: Vec<String>,
}

pub struct ConceptualLink {
    pub concept: String,
    pub related_concepts: Vec<String>,
    pub similarity_score: f32,
    pub shared_context: String,
}
```

## Implementation Strategy

### Phase 1: Enhanced Knowledge Processing Pipeline

#### A. Intelligent Text Analysis
```rust
impl IntelligentKnowledgeProcessor {
    pub async fn process_knowledge(&self, content: &str, title: &str) -> Result<HierarchicalKnowledge> {
        // 1. Global Context Analysis
        let global_context = self.analyze_global_context(content, title).await?;
        
        // 2. Semantic Structure Detection  
        let document_structure = self.detect_document_structure(content).await?;
        
        // 3. Intelligent Chunking
        let semantic_chunks = self.create_semantic_chunks(content, &document_structure).await?;
        
        // 4. Advanced Entity Extraction
        let entities = self.extract_entities_with_context(content, &semantic_chunks).await?;
        
        // 5. Complex Relationship Mapping
        let relationships = self.extract_complex_relationships(content, &entities).await?;
        
        // 6. Hierarchical Structure Creation
        let knowledge_layers = self.create_knowledge_layers(
            semantic_chunks, entities, relationships, global_context
        ).await?;
        
        // 7. Semantic Link Graph Construction
        let semantic_links = self.build_semantic_links(&knowledge_layers).await?;
        
        Ok(HierarchicalKnowledge {
            document_id: generate_document_id(title),
            global_context,
            knowledge_layers,
            semantic_links,
            retrieval_index: self.build_hierarchical_index(&knowledge_layers)?,
        })
    }
}
```

#### B. SmolLM-Powered Entity & Relationship Extraction

**Instead of simple pattern matching, use SmolLM for:**

```rust
impl AdvancedEntityExtractor {
    pub async fn extract_entities_with_context(&self, text: &str) -> Result<Vec<ContextualEntity>> {
        let prompt = format!(
            "Extract all important entities from this text. For each entity, provide:
            1. Entity name
            2. Entity type (Person, Organization, Concept, Event, etc.)
            3. Surrounding context (the sentence or phrase it appears in)
            4. Confidence level (0.0-1.0)
            
            Text: {}
            
            Return as JSON array with format:
            [{{\"name\": \"EntityName\", \"type\": \"EntityType\", \"context\": \"surrounding text\", \"confidence\": 0.95}}]",
            text
        );
        
        let response = self.model.generate_text(&prompt, Some(1000)).await?;
        self.parse_entity_response(&response)
    }
}
```

#### C. Complex Relationship Extraction

```rust
impl RelationshipMapper {
    pub async fn extract_complex_relationships(&self, text: &str, entities: &[ContextualEntity]) -> Result<Vec<ComplexRelationship>> {
        let prompt = format!(
            "Given this text and these entities: {}, find all relationships between entities.
            Include:
            - Direct relationships (X created Y, A works for B)
            - Implicit relationships (A and B both mentioned in same context)
            - Temporal relationships (X happened before Y)
            - Causal relationships (X caused Y)
            - Hierarchical relationships (X is part of Y)
            
            Text: {}
            
            Return detailed relationships with confidence scores.",
            serde_json::to_string(entities)?,
            text
        );
        
        let response = self.model.generate_text(&prompt, Some(1500)).await?;
        self.parse_relationship_response(&response)
    }
}
```

### Phase 2: Context-Preserving Storage

#### A. Semantic Chunking Strategy

```rust
pub struct SemanticChunker {
    model: SmolLMVariant,
    max_chunk_size: usize,    // Configurable: 1KB-8KB based on content complexity
    min_chunk_size: usize,    // Prevent over-fragmentation
    overlap_strategy: OverlapStrategy,
}

pub enum OverlapStrategy {
    SentenceBoundary { sentences: usize },
    SemanticOverlap { tokens: usize },
    ConceptualBridging { concepts: Vec<String> },
}

impl SemanticChunker {
    pub async fn create_semantic_chunks(&self, text: &str) -> Result<Vec<SemanticChunk>> {
        // 1. Analyze document structure
        let structure = self.analyze_document_structure(text).await?;
        
        // 2. Identify semantic boundaries
        let boundaries = self.find_semantic_boundaries(text, &structure).await?;
        
        // 3. Create overlapping chunks with context preservation
        let chunks = self.create_overlapping_chunks(text, &boundaries).await?;
        
        // 4. Ensure semantic coherence
        let validated_chunks = self.validate_chunk_coherence(chunks).await?;
        
        Ok(validated_chunks)
    }
    
    async fn find_semantic_boundaries(&self, text: &str, structure: &DocumentStructure) -> Result<Vec<SemanticBoundary>> {
        let prompt = format!(
            "Analyze this text and identify optimal boundaries for chunking that preserve semantic meaning.
            Consider:
            - Sentence completeness
            - Paragraph coherence  
            - Topic transitions
            - Entity relationships
            - Logical flow
            
            Text: {}
            
            Return boundary positions with reasons.",
            text
        );
        
        let response = self.model.generate_text(&prompt, Some(1000)).await?;
        self.parse_boundary_response(&response, text.len())
    }
}
```

#### B. Knowledge Layer Architecture

```rust
pub struct KnowledgeLayer {
    pub id: String,
    pub layer_type: LayerType,
    pub content: LayerContent,
    pub metadata: LayerMetadata,
    pub relationships: LayerRelationships,
    pub embedding: Vec<f32>,
}

pub struct LayerContent {
    pub primary_text: String,
    pub summary: String,           // SmolLM-generated summary
    pub key_concepts: Vec<String>, // Main ideas/themes
    pub extracted_facts: Vec<Triple>,
    pub context_snippets: Vec<String>, // Important surrounding context
}

pub struct LayerMetadata {
    pub confidence_score: f32,
    pub importance_score: f32,     // Based on entity centrality and semantic richness
    pub complexity_level: ComplexityLevel,
    pub topics: Vec<String>,
    pub word_count: usize,
    pub created_at: u64,
}

pub struct LayerRelationships {
    pub parent_layers: Vec<String>,
    pub child_layers: Vec<String>,
    pub sibling_layers: Vec<String>,
    pub semantic_neighbors: Vec<SemanticNeighbor>,
    pub cross_references: Vec<CrossReference>,
}

pub struct SemanticNeighbor {
    pub layer_id: String,
    pub similarity_score: f32,
    pub shared_entities: Vec<String>,
    pub connection_type: ConnectionType,
}

pub enum ConnectionType {
    EntityOverlap,
    ConceptualSimilarity,
    TemporalSequence,
    CausalRelation,
    HierarchicalLink,
}
```

### Phase 3: Enhanced Retrieval System

#### A. Multi-Layer Query Strategy

```rust
pub struct HierarchicalRetriever {
    pub layers: Arc<RwLock<HashMap<String, KnowledgeLayer>>>,
    pub semantic_index: SemanticIndex,
    pub graph_traverser: GraphTraverser,
    pub context_assembler: ContextAssembler,
}

impl HierarchicalRetriever {
    pub async fn retrieve_knowledge(&self, query: &str, max_context: usize) -> Result<EnrichedKnowledgeResult> {
        // 1. Multi-level semantic search
        let layer_matches = self.search_all_layers(query).await?;
        
        // 2. Graph traversal for related context
        let expanded_context = self.expand_context_via_graph(&layer_matches).await?;
        
        // 3. Hierarchical context assembly
        let assembled_context = self.assemble_hierarchical_context(
            &expanded_context, 
            max_context
        ).await?;
        
        // 4. Deliver optimized response to main LLM
        Ok(EnrichedKnowledgeResult {
            direct_matches: layer_matches,
            contextual_information: assembled_context,
            entity_relationships: self.get_relevant_relationships(&layer_matches).await?,
            confidence_scores: self.calculate_confidence_scores(&layer_matches),
            retrieval_path: self.document_retrieval_reasoning(&layer_matches),
        })
    }
    
    async fn assemble_hierarchical_context(&self, matches: &[LayerMatch], max_size: usize) -> Result<HierarchicalContext> {
        let mut context = HierarchicalContext::new();
        let mut current_size = 0;
        
        // Priority order: Direct matches -> Parent context -> Related entities -> Sibling context
        for layer_match in matches.iter().take_while(|_| current_size < max_size) {
            // Add the direct match
            context.add_primary_content(&layer_match.content);
            current_size += layer_match.content.len();
            
            // Add parent context for broader understanding
            if let Some(parent) = self.get_parent_layer(&layer_match.layer_id).await? {
                if current_size + parent.summary.len() < max_size {
                    context.add_parent_context(&parent.summary);
                    current_size += parent.summary.len();
                }
            }
            
            // Add related entity information
            for entity_relation in &layer_match.related_entities {
                if current_size + entity_relation.context.len() < max_size {
                    context.add_entity_context(entity_relation);
                    current_size += entity_relation.context.len();
                }
            }
        }
        
        Ok(context)
    }
}
```

#### B. Optimized Response Generation

```rust
pub struct EnrichedKnowledgeResult {
    pub summary: String,                    // Concise summary for main LLM
    pub primary_facts: Vec<Triple>,         // Key triples extracted
    pub contextual_entities: Vec<ContextualEntity>,
    pub relationship_map: HashMap<String, Vec<EntityRelation>>,
    pub confidence_assessment: ConfidenceAssessment,
    pub retrieval_metadata: RetrievalMetadata,
}

impl EnrichedKnowledgeResult {
    pub fn to_llm_optimized_response(&self) -> LLMOptimizedResponse {
        LLMOptimizedResponse {
            // Deliver clean, structured information to main LLM
            main_content: self.summary.clone(),
            supporting_facts: self.primary_facts.clone(),
            entity_context: self.contextual_entities.iter()
                .map(|e| format!("{}: {}", e.name, e.context))
                .collect(),
            confidence: self.confidence_assessment.overall_confidence,
            // Hide complexity from main LLM
            complexity_handled_internally: true,
        }
    }
}
```

## Model Selection and Resource Allocation

### Recommended Model Configuration

#### Primary Knowledge Processor: **SmolLM2-360M**
- **Parameters**: 360M (optimal balance of capability vs resource usage)
- **Context Length**: 2048 tokens
- **Capabilities**: Text generation, reasoning, instruction following
- **Use Cases**: Entity extraction, relationship mapping, semantic analysis
- **Memory Footprint**: ~700MB loaded

#### Backup/Specialized Processor: **MiniLM-L12-H384**
- **Parameters**: 33M (ultra-fast for specific tasks)
- **Specialization**: Sentence transformers, embeddings
- **Use Cases**: Semantic similarity, clustering, boundary detection
- **Memory Footprint**: ~130MB loaded

### Resource Management Strategy

```rust
pub struct ModelResourceManager {
    primary_model: Option<Model>,
    specialized_models: HashMap<TaskType, Model>,
    model_pool: ModelPool,
    resource_limits: ResourceLimits,
}

pub struct ResourceLimits {
    max_concurrent_models: usize,      // Usually 2-3
    max_memory_usage: usize,           // e.g., 2GB total
    model_idle_timeout: Duration,       // Unload after 5 minutes idle
    processing_timeout: Duration,       // 30 seconds per knowledge chunk
}

impl ModelResourceManager {
    pub async fn process_with_optimal_model(&self, task: ProcessingTask) -> Result<TaskResult> {
        match task.complexity {
            ComplexityLevel::High => {
                let model = self.ensure_model_loaded(ModelType::SmolLM360M).await?;
                model.process_complex_task(task).await
            },
            ComplexityLevel::Medium => {
                let model = self.ensure_model_loaded(ModelType::SmolLM135M).await?;
                model.process_medium_task(task).await
            },
            ComplexityLevel::Low => {
                let model = self.ensure_model_loaded(ModelType::MiniLM).await?;
                model.process_simple_task(task).await
            }
        }
    }
}
```

## Advanced Features

### 1. Contextual Bridging

```rust
pub struct ContextualBridge {
    pub source_layer: String,
    pub target_layer: String,
    pub bridge_content: String,       // Generated connecting context
    pub bridge_entities: Vec<String>, // Shared entities
    pub semantic_similarity: f32,
}

impl ContextAssembler {
    async fn create_contextual_bridges(&self, layers: &[KnowledgeLayer]) -> Result<Vec<ContextualBridge>> {
        let mut bridges = Vec::new();
        
        for (i, layer1) in layers.iter().enumerate() {
            for layer2 in layers.iter().skip(i + 1) {
                if let Some(bridge) = self.analyze_connection_potential(layer1, layer2).await? {
                    bridges.push(bridge);
                }
            }
        }
        
        Ok(bridges)
    }
}
```

### 2. Dynamic Context Expansion

```rust
pub struct DynamicContextExpander {
    expansion_strategies: Vec<ExpansionStrategy>,
    context_budget: usize,
    relevance_threshold: f32,
}

pub enum ExpansionStrategy {
    EntityChasing,      // Follow entity mentions to related content
    ConceptualSimilarity, // Find semantically similar content
    TemporalSequence,   // Follow chronological connections  
    CausalChaining,     // Follow cause-effect relationships
    HierarchicalClimbing, // Move up/down knowledge hierarchy
}
```

### 3. Intelligent Caching and Optimization

```rust
pub struct IntelligentCache {
    processed_knowledge: LRUCache<String, HierarchicalKnowledge>,
    semantic_clusters: HashMap<String, Vec<String>>,
    access_patterns: AccessPatternAnalyzer,
    precomputed_contexts: ContextCache,
}

impl IntelligentCache {
    pub async fn precompute_likely_contexts(&self) -> Result<()> {
        // Analyze access patterns
        let patterns = self.access_patterns.analyze_query_patterns().await?;
        
        // Pre-compute context for frequently accessed entity combinations
        for pattern in patterns.high_frequency_patterns {
            let context = self.assemble_context_for_pattern(&pattern).await?;
            self.precomputed_contexts.insert(pattern.signature(), context);
        }
        
        Ok(())
    }
}
```

## Integration with Existing System

### Modified `store_knowledge` Handler

```rust
pub async fn handle_store_knowledge_enhanced(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    intelligent_processor: &Arc<IntelligentKnowledgeProcessor>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let content = params.get("content").and_then(|v| v.as_str())
        .ok_or("Missing required field: content")?;
    let title = params.get("title").and_then(|v| v.as_str())
        .ok_or("Missing required field: title")?;
    let category = params.get("category").and_then(|v| v.as_str())
        .unwrap_or("general");
    let source = params.get("source").and_then(|v| v.as_str());
    
    // Enhanced processing with intelligent models
    let hierarchical_knowledge = intelligent_processor
        .process_knowledge(content, title)
        .await
        .map_err(|e| format!("Enhanced processing failed: {}", e))?;
    
    // Store in hierarchical structure
    let document_id = knowledge_engine.write().await
        .store_hierarchical_knowledge(hierarchical_knowledge.clone())
        .await
        .map_err(|e| format!("Failed to store hierarchical knowledge: {}", e))?;
    
    // Update temporal tracking
    TEMPORAL_INDEX.record_hierarchical_operation(
        hierarchical_knowledge.clone(),
        TemporalOperation::Create,
        None
    );
    
    // Return optimized response for main LLM
    let data = json!({
        "stored": true,
        "document_id": document_id,
        "title": title,
        "category": category,
        "layers_created": hierarchical_knowledge.knowledge_layers.len(),
        "entities_extracted": hierarchical_knowledge.count_entities(),
        "relationships_mapped": hierarchical_knowledge.count_relationships(),
        "semantic_links": hierarchical_knowledge.semantic_links.total_links(),
        "processing_model": "SmolLM2-360M",
        "context_preservation_score": hierarchical_knowledge.calculate_context_preservation_score()
    });
    
    let message = format!(
        "✓ Enhanced knowledge processing completed for '{}': {} layers, {} entities, {} relationships with preserved context",
        title,
        hierarchical_knowledge.knowledge_layers.len(),
        hierarchical_knowledge.count_entities(),
        hierarchical_knowledge.count_relationships()
    );
    
    let suggestions = vec![
        "Use enhanced_knowledge_query for complex multi-hop questions".to_string(),
        "Try hierarchical_search for broad-to-specific exploration".to_string(),
        format!("Explore entity relationships with: explore_entity_graph(entity=\"{}\")", 
            hierarchical_knowledge.get_primary_entities().first().unwrap_or(&"concept".to_string())),
    ];
    
    Ok((data, message, suggestions))
}
```

### Enhanced Query Tools

```rust
pub async fn handle_enhanced_knowledge_query(
    hierarchical_retriever: &Arc<HierarchicalRetriever>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let question = params.get("question").and_then(|v| v.as_str())
        .ok_or("Missing required field: question")?;
    let context_budget = params.get("context_budget")
        .and_then(|v| v.as_u64())
        .unwrap_or(2000) as usize; // Allow larger context for complex queries
    let include_reasoning = params.get("include_reasoning")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    let enriched_result = hierarchical_retriever
        .retrieve_knowledge(question, context_budget)
        .await
        .map_err(|e| format!("Enhanced retrieval failed: {}", e))?;
    
    let optimized_response = enriched_result.to_llm_optimized_response();
    
    let data = json!({
        "answer_context": optimized_response.main_content,
        "supporting_facts": optimized_response.supporting_facts,
        "entity_context": optimized_response.entity_context,
        "confidence": optimized_response.confidence,
        "retrieval_reasoning": if include_reasoning { 
            Some(enriched_result.retrieval_metadata.reasoning_path) 
        } else { 
            None 
        },
        "context_layers_used": enriched_result.retrieval_metadata.layers_accessed,
        "processing_efficiency": optimized_response.complexity_handled_internally
    });
    
    let message = format!(
        "📚 Enhanced knowledge query processed: {} context layers accessed, confidence: {:.2}",
        enriched_result.retrieval_metadata.layers_accessed,
        optimized_response.confidence
    );
    
    Ok((data, message, vec![]))
}
```

## Performance Characteristics

### Storage Efficiency
- **Primary Storage**: Still optimalized for main LLM (~60 bytes per triple)
- **Enhanced Storage**: Additional ~200-500 bytes per knowledge layer
- **Context Links**: ~50 bytes per semantic connection
- **Total Overhead**: ~3-5x current storage (justified by retrieval quality improvement)

### Retrieval Performance  
- **Simple Queries**: <10ms (cached responses)
- **Complex Multi-hop**: 50-200ms (model-assisted assembly)
- **Context Assembly**: 20-100ms depending on hierarchy depth
- **Memory Usage**: 700MB-2GB for loaded models (vs. current ~100MB)

### Quality Improvements
- **Context Preservation**: 85%+ vs. current ~40%
- **Entity Recognition**: 90%+ vs. current ~30% 
- **Relationship Accuracy**: 80%+ vs. current ~20%
- **Multi-hop Queries**: Newly supported (0% -> 70%+ success rate)

## Migration Strategy

### Phase 1: Infrastructure (Weeks 1-2)
1. Implement model loading and resource management
2. Create hierarchical knowledge structures
3. Build semantic chunking system
4. Add enhanced entity/relationship extraction

### Phase 2: Storage Integration (Weeks 3-4)  
1. Modify knowledge engine for hierarchical storage
2. Implement semantic link graph
3. Add contextual bridging system
4. Create migration tools for existing data

### Phase 3: Retrieval Enhancement (Weeks 5-6)
1. Build hierarchical retriever
2. Implement context assembly system
3. Add dynamic context expansion
4. Create optimized response formatting

### Phase 4: Performance Optimization (Weeks 7-8)
1. Implement intelligent caching
2. Add model resource optimization
3. Build access pattern analysis
4. Fine-tune performance characteristics

## Success Metrics

### Technical Metrics
- **Context Preservation Score**: Target 85%+ (vs. current ~40%)
- **Entity Recognition Accuracy**: Target 90%+ (vs. current ~30%)
- **Relationship Extraction Quality**: Target 80%+ (vs. current ~20%)
- **Multi-hop Query Success**: Target 70%+ (currently 0%)
- **Retrieval Latency**: <200ms for 95% of complex queries

### User Experience Metrics
- **Answer Completeness**: Measured by human evaluation
- **Context Coherence**: Measured by semantic coherence scores
- **Factual Accuracy**: Measured against ground truth datasets
- **Response Relevance**: Measured by user satisfaction scores

## Risk Mitigation

### Resource Management Risks
- **Memory Usage**: Implement intelligent model loading/unloading
- **Processing Time**: Set timeout limits and fallback to simpler methods
- **Storage Growth**: Implement retention policies and archival strategies

### Quality Assurance Risks  
- **Model Hallucination**: Cross-validate extractions across multiple methods
- **Context Pollution**: Implement relevance filtering and confidence thresholds
- **Performance Degradation**: Monitor metrics and implement circuit breakers

# Test-Driven Development Implementation Plan

## London School TDD Methodology

This implementation follows **London School TDD** (also known as Outside-In TDD) which emphasizes:

- **Outside-In Development**: Start with acceptance tests and work inward to implementation details
- **Heavy Use of Test Doubles**: Mock external dependencies and collaborators for isolation
- **Behavior Verification**: Focus on testing interactions and behaviors rather than state
- **Rapid Feedback Loops**: Quick red-green-refactor cycles with fast-running tests
- **Collaboration-Based Design**: Design emerges from understanding how objects collaborate

### TDD Cycle Structure

Each feature follows the **Double-Loop TDD** approach:

```
OUTER LOOP (Acceptance Test):
   RED → Write failing acceptance test
   ↓
   INNER LOOP (Unit Tests):
      RED → Write failing unit test
      GREEN → Make unit test pass (minimal implementation)
      REFACTOR → Clean up code while keeping tests green
      ↑
   GREEN → All unit tests pass, acceptance test passes
   ↓
   REFACTOR → Clean up overall design
```

## Test Architecture and Organization

### Test Structure

```
tests/
├── acceptance/                    # End-to-end behavior tests
│   ├── knowledge_storage_flow.rs
│   ├── knowledge_retrieval_flow.rs
│   └── enhanced_query_flow.rs
├── integration/                   # Component integration tests
│   ├── intelligent_processor_integration.rs
│   ├── hierarchical_storage_integration.rs
│   └── retrieval_system_integration.rs
├── unit/                         # Fast, isolated unit tests
│   ├── model_management/
│   ├── knowledge_processing/
│   ├── storage_layers/
│   └── retrieval_systems/
└── test_support/                 # Test utilities and fixtures
    ├── mocks/
    ├── builders/
    └── fixtures/
```

### Test Support Infrastructure

```rust
// tests/test_support/mocks/mod.rs
pub mod model_mocks;
pub mod storage_mocks;
pub mod processing_mocks;

// tests/test_support/builders/mod.rs
pub mod knowledge_layer_builder;
pub mod hierarchical_knowledge_builder;
pub mod test_data_builder;

// tests/test_support/fixtures/mod.rs
pub mod sample_documents;
pub mod expected_extractions;
pub mod performance_benchmarks;
```

## Phase-by-Phase TDD Implementation

## Phase 1: Model Management Infrastructure (Weeks 1-2)

### Week 1: Model Loading and Resource Management

#### Day 1-2: Acceptance Test - Model Loading Flow

**Acceptance Test**: `tests/acceptance/model_loading_flow.rs`

```rust
#[tokio::test]
async fn should_load_and_manage_models_efficiently() {
    // GIVEN: A model resource manager with resource limits
    let config = ModelResourceConfig {
        max_memory_usage: 2_000_000_000, // 2GB
        max_concurrent_models: 3,
        idle_timeout: Duration::from_secs(300),
    };
    let manager = ModelResourceManager::new(config);
    
    // WHEN: Multiple processing tasks with different complexity levels are requested
    let simple_task = ProcessingTask::new(ComplexityLevel::Low, "simple text");
    let complex_task = ProcessingTask::new(ComplexityLevel::High, "complex document");
    
    // THEN: Appropriate models are loaded and tasks are processed efficiently
    let simple_result = manager.process_with_optimal_model(simple_task).await.unwrap();
    let complex_result = manager.process_with_optimal_model(complex_task).await.unwrap();
    
    assert!(simple_result.processing_time < Duration::from_millis(100));
    assert!(complex_result.quality_score > 0.8);
    assert!(manager.current_memory_usage() <= config.max_memory_usage);
}
```

#### Unit Tests Development (TDD Cycles)

**Cycle 1**: Model Registry Behavior

```rust
// tests/unit/model_management/model_registry_test.rs

#[cfg(test)]
mod model_registry_tests {
    use super::*;
    use crate::test_support::mocks::model_mocks::MockModel;

    #[test]
    fn should_register_model_with_metadata() {
        // RED: Write failing test
        let mut registry = ModelRegistry::new();
        let model_metadata = ModelMetadata {
            name: "SmolLM2-360M".to_string(),
            parameters: 360_000_000,
            memory_footprint: 700_000_000,
        };
        
        registry.register_model("smollm2_360m", model_metadata.clone());
        
        let retrieved = registry.get_model_metadata("smollm2_360m").unwrap();
        assert_eq!(retrieved.name, "SmolLM2-360M");
        assert_eq!(retrieved.parameters, 360_000_000);
    }

    #[test]
    fn should_suggest_optimal_model_for_task() {
        // RED: Write failing test
        let registry = ModelRegistry::with_default_models();
        
        let simple_task = TaskComplexity::Low;
        let complex_task = TaskComplexity::High;
        
        let simple_model = registry.suggest_optimal_model(simple_task).unwrap();
        let complex_model = registry.suggest_optimal_model(complex_task).unwrap();
        
        assert!(simple_model.parameters < complex_model.parameters);
        assert!(simple_model.memory_footprint < complex_model.memory_footprint);
    }
}
```

**Cycle 2**: Model Loader with Mocks

```rust
// tests/unit/model_management/model_loader_test.rs

#[cfg(test)]
mod model_loader_tests {
    use mockall::*;
    use super::*;

    // Define mock traits for external dependencies
    mock! {
        ModelBackend {}
        
        #[async_trait]
        impl ModelBackend for ModelBackend {
            async fn load_model(&self, model_id: &str) -> Result<Box<dyn Model>>;
            async fn unload_model(&self, model_id: &str) -> Result<()>;
            fn get_memory_usage(&self) -> usize;
        }
    }

    #[tokio::test]
    async fn should_load_model_when_not_in_cache() {
        // RED: Write failing test
        let mut mock_backend = MockModelBackend::new();
        mock_backend
            .expect_load_model()
            .with(eq("smollm2_360m"))
            .times(1)
            .return_once(|_| Ok(Box::new(MockModel::new())));
        
        let loader = ModelLoader::new(Box::new(mock_backend));
        let model = loader.ensure_model_loaded("smollm2_360m").await.unwrap();
        
        assert!(model.is_loaded());
    }

    #[tokio::test]
    async fn should_not_reload_already_loaded_model() {
        // RED: Write failing test
        let mut mock_backend = MockModelBackend::new();
        mock_backend
            .expect_load_model()
            .times(1) // Should only be called once
            .return_once(|_| Ok(Box::new(MockModel::new())));
        
        let loader = ModelLoader::new(Box::new(mock_backend));
        
        // Load model twice
        let _model1 = loader.ensure_model_loaded("smollm2_360m").await.unwrap();
        let _model2 = loader.ensure_model_loaded("smollm2_360m").await.unwrap();
        
        // Mock expectation verifies it was only called once
    }
}
```

**Implementation**: Start with minimal implementation to make tests pass

```rust
// src/models/resource_manager.rs
pub struct ModelResourceManager {
    loaded_models: HashMap<String, Box<dyn Model>>,
    model_loader: Box<dyn ModelLoader>,
    resource_monitor: ResourceMonitor,
    config: ModelResourceConfig,
}

impl ModelResourceManager {
    pub fn new(config: ModelResourceConfig) -> Self {
        Self {
            loaded_models: HashMap::new(),
            model_loader: Box::new(DefaultModelLoader::new()),
            resource_monitor: ResourceMonitor::new(),
            config,
        }
    }

    pub async fn ensure_model_loaded(&mut self, model_id: &str) -> Result<&dyn Model> {
        if !self.loaded_models.contains_key(model_id) {
            self.load_model_with_resource_check(model_id).await?;
        }
        Ok(self.loaded_models.get(model_id).unwrap().as_ref())
    }

    async fn load_model_with_resource_check(&mut self, model_id: &str) -> Result<()> {
        // Check if we need to evict models to stay within limits
        if self.would_exceed_memory_limit(model_id) {
            self.evict_least_recently_used_model().await?;
        }
        
        let model = self.model_loader.load_model(model_id).await?;
        self.loaded_models.insert(model_id.to_string(), model);
        Ok(())
    }
}
```

#### Day 3-4: Resource Management and Limits

**Unit Tests for Resource Monitoring**

```rust
// tests/unit/model_management/resource_monitor_test.rs

#[cfg(test)]
mod resource_monitor_tests {
    use super::*;

    #[test]
    fn should_track_memory_usage_accurately() {
        // RED: Write failing test
        let mut monitor = ResourceMonitor::new();
        
        monitor.record_model_loaded("model1", 500_000_000); // 500MB
        monitor.record_model_loaded("model2", 300_000_000); // 300MB
        
        assert_eq!(monitor.total_memory_usage(), 800_000_000);
        
        monitor.record_model_unloaded("model1");
        assert_eq!(monitor.total_memory_usage(), 300_000_000);
    }

    #[test]
    fn should_identify_models_exceeding_idle_timeout() {
        // RED: Write failing test
        let mut monitor = ResourceMonitor::new();
        let old_time = SystemTime::now() - Duration::from_secs(400);
        let recent_time = SystemTime::now() - Duration::from_secs(100);
        
        monitor.record_model_access("old_model", old_time);
        monitor.record_model_access("recent_model", recent_time);
        
        let idle_models = monitor.get_models_exceeding_idle_timeout(Duration::from_secs(300));
        
        assert_eq!(idle_models.len(), 1);
        assert_eq!(idle_models[0], "old_model");
    }
}
```

#### Day 5: Model Pool and Lifecycle Management

**Unit Tests for Model Pool**

```rust
// tests/unit/model_management/model_pool_test.rs

#[cfg(test)]
mod model_pool_tests {
    use super::*;
    use crate::test_support::mocks::model_mocks::*;

    #[tokio::test]
    async fn should_evict_lru_model_when_at_capacity() {
        // RED: Write failing test
        let config = ModelPoolConfig {
            max_models: 2,
            eviction_strategy: EvictionStrategy::LeastRecentlyUsed,
        };
        let mut pool = ModelPool::new(config);
        
        // Fill pool to capacity
        pool.add_model("model1", MockModel::new()).await;
        pool.add_model("model2", MockModel::new()).await;
        
        // Access model1 to make it more recently used
        pool.access_model("model1").await;
        
        // Add third model, should evict model2 (LRU)
        pool.add_model("model3", MockModel::new()).await;
        
        assert!(pool.has_model("model1"));
        assert!(!pool.has_model("model2"));
        assert!(pool.has_model("model3"));
    }

    #[tokio::test]
    async fn should_preload_frequently_used_models() {
        // RED: Write failing test
        let mut mock_loader = MockModelLoader::new();
        mock_loader
            .expect_load_model()
            .with(eq("frequently_used_model"))
            .times(1)
            .return_once(|_| Ok(Box::new(MockModel::new())));
        
        let mut pool = ModelPool::with_loader(Box::new(mock_loader));
        
        // Simulate usage pattern analysis indicating frequent use
        let usage_pattern = UsagePattern {
            model_id: "frequently_used_model".to_string(),
            access_frequency: 0.8,
            recent_accesses: 50,
        };
        
        pool.preload_based_on_usage_patterns(&[usage_pattern]).await.unwrap();
        
        assert!(pool.has_model("frequently_used_model"));
    }
}
```

### Week 2: Model Integration and Processing Pipeline

#### Day 6-7: SmolLM Integration

**Acceptance Test**: `tests/acceptance/smollm_integration_flow.rs`

```rust
#[tokio::test]
async fn should_process_knowledge_using_smollm() {
    // GIVEN: A knowledge processor with SmolLM integration
    let processor = IntelligentKnowledgeProcessor::new(
        SmolLMVariant::SmolLM2_360M,
        ProcessingConfig::default()
    ).await.unwrap();
    
    let sample_text = "Albert Einstein developed the theory of relativity, \
                       which revolutionized physics in the early 20th century. \
                       He received the Nobel Prize in Physics in 1921.";
    
    // WHEN: Processing knowledge with the intelligent processor
    let result = processor.process_knowledge(sample_text, "Einstein Biography").await.unwrap();
    
    // THEN: Advanced extraction and hierarchical organization occurs
    assert!(result.knowledge_layers.len() > 1);
    assert!(result.count_entities() >= 3); // Einstein, relativity, Nobel Prize
    assert!(result.count_relationships() >= 2);
    assert!(result.global_context.summary.len() > 0);
    assert!(result.semantic_links.total_links() > 0);
}
```

**Unit Tests with Model Mocks**

```rust
// tests/unit/knowledge_processing/intelligent_processor_test.rs

#[cfg(test)]
mod intelligent_processor_tests {
    use super::*;
    use crate::test_support::mocks::model_mocks::*;

    mock! {
        SmolLMModel {}
        
        #[async_trait]
        impl Model for SmolLMModel {
            async fn generate_text(&self, prompt: &str, max_tokens: Option<u32>) -> Result<String>;
            fn is_loaded(&self) -> bool;
            fn parameter_count(&self) -> u64;
        }
    }

    #[tokio::test]
    async fn should_extract_entities_using_model() {
        // RED: Write failing test
        let mut mock_model = MockSmolLMModel::new();
        mock_model
            .expect_generate_text()
            .with(predicate::str::contains("Extract all important entities"))
            .times(1)
            .return_once(|_, _| Ok(r#"[
                {"name": "Einstein", "type": "Person", "context": "Albert Einstein developed", "confidence": 0.95},
                {"name": "relativity", "type": "Concept", "context": "theory of relativity", "confidence": 0.90}
            ]"#.to_string()));

        let extractor = AdvancedEntityExtractor::new(Box::new(mock_model));
        let entities = extractor.extract_entities_with_context("Albert Einstein developed the theory of relativity").await.unwrap();
        
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].name, "Einstein");
        assert_eq!(entities[0].entity_type, "Person");
        assert_eq!(entities[1].name, "relativity");
    }

    #[tokio::test]
    async fn should_handle_model_errors_gracefully() {
        // RED: Write failing test
        let mut mock_model = MockSmolLMModel::new();
        mock_model
            .expect_generate_text()
            .times(1)
            .return_once(|_, _| Err(GraphError::ModelError("Model timeout".to_string())));

        let extractor = AdvancedEntityExtractor::new(Box::new(mock_model));
        let result = extractor.extract_entities_with_context("test text").await;
        
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::ModelError(msg) => assert_eq!(msg, "Model timeout"),
            _ => panic!("Expected ModelError"),
        }
    }
}
```

#### Day 8-9: Semantic Chunking with Model Integration

**Unit Tests for Semantic Chunker**

```rust
// tests/unit/knowledge_processing/semantic_chunker_test.rs

#[cfg(test)]
mod semantic_chunker_tests {
    use super::*;

    #[tokio::test]
    async fn should_identify_semantic_boundaries() {
        // RED: Write failing test
        let mut mock_model = MockSmolLMModel::new();
        mock_model
            .expect_generate_text()
            .with(predicate::str::contains("identify optimal boundaries"))
            .times(1)
            .return_once(|_, _| Ok(r#"{
                "boundaries": [
                    {"position": 156, "reason": "Topic transition from biography to achievements"},
                    {"position": 298, "reason": "Shift from scientific work to recognition"}
                ]
            }"#.to_string()));

        let chunker = SemanticChunker::new(
            Box::new(mock_model),
            ChunkingConfig {
                max_chunk_size: 2000,
                min_chunk_size: 200,
                overlap_strategy: OverlapStrategy::SentenceBoundary { sentences: 2 },
            }
        );

        let text = "Albert Einstein was born in Germany. He developed relativity theory. \
                   This revolutionized physics. He won the Nobel Prize in 1921.";
        
        let chunks = chunker.create_semantic_chunks(text).await.unwrap();
        
        assert!(chunks.len() >= 2);
        assert!(chunks[0].content.len() > 0);
        assert!(chunks[0].semantic_coherence_score > 0.7);
    }

    #[tokio::test]
    async fn should_preserve_context_across_chunk_boundaries() {
        // RED: Write failing test
        let chunker = SemanticChunker::new(
            Box::new(MockSmolLMModel::new()),
            ChunkingConfig::default()
        );

        let text = "Einstein developed relativity. This theory changed physics forever.";
        let chunks = chunker.create_semantic_chunks(text).await.unwrap();
        
        // Check that overlapping content preserves key entities
        if chunks.len() > 1 {
            let overlap = chunker.calculate_semantic_overlap(&chunks[0], &chunks[1]);
            assert!(overlap.shared_entities.contains(&"Einstein".to_string()));
            assert!(overlap.shared_entities.contains(&"relativity".to_string()));
        }
    }
}
```

#### Day 10: Relationship Extraction Pipeline

**Unit Tests for Relationship Mapper**

```rust
// tests/unit/knowledge_processing/relationship_mapper_test.rs

#[cfg(test)]
mod relationship_mapper_tests {
    use super::*;

    #[tokio::test]
    async fn should_extract_complex_relationships() {
        // RED: Write failing test
        let mut mock_model = MockSmolLMModel::new();
        mock_model
            .expect_generate_text()
            .with(predicate::str::contains("find all relationships"))
            .times(1)
            .return_once(|_, _| Ok(r#"{
                "relationships": [
                    {
                        "source": "Einstein",
                        "predicate": "developed",
                        "target": "relativity",
                        "type": "created",
                        "confidence": 0.95,
                        "context": "Einstein developed the theory of relativity"
                    },
                    {
                        "source": "Einstein",
                        "predicate": "received",
                        "target": "Nobel Prize",
                        "type": "awarded",
                        "confidence": 0.90,
                        "context": "He received the Nobel Prize in Physics"
                    }
                ]
            }"#.to_string()));

        let entities = vec![
            ContextualEntity::new("Einstein", "Person"),
            ContextualEntity::new("relativity", "Concept"),
            ContextualEntity::new("Nobel Prize", "Award"),
        ];

        let mapper = RelationshipMapper::new(Box::new(mock_model));
        let relationships = mapper.extract_complex_relationships(
            "Einstein developed the theory of relativity. He received the Nobel Prize in Physics.",
            &entities
        ).await.unwrap();
        
        assert_eq!(relationships.len(), 2);
        assert_eq!(relationships[0].source, "Einstein");
        assert_eq!(relationships[0].predicate, "developed");
        assert_eq!(relationships[0].target, "relativity");
    }

    #[tokio::test]
    async fn should_detect_implicit_relationships() {
        // RED: Write failing test
        let mapper = RelationshipMapper::new(Box::new(MockSmolLMModel::new()));
        
        let entities = vec![
            ContextualEntity::new("Einstein", "Person"),
            ContextualEntity::new("Physics", "Field"),
        ];

        // Text doesn't explicitly state relationship but implies it
        let relationships = mapper.extract_complex_relationships(
            "Einstein's work in theoretical physics was groundbreaking.",
            &entities
        ).await.unwrap();
        
        // Should detect implicit "works_in" relationship
        assert!(relationships.iter().any(|r| 
            r.source == "Einstein" && r.target == "Physics" && r.relationship_type == RelationshipType::Professional
        ));
    }
}
```

## Phase 2: Hierarchical Storage System (Weeks 3-4)

### Week 3: Knowledge Layer Architecture

#### Day 11-12: Knowledge Layer Structure

**Acceptance Test**: `tests/acceptance/hierarchical_storage_flow.rs`

```rust
#[tokio::test]
async fn should_store_knowledge_in_hierarchical_layers() {
    // GIVEN: A hierarchical knowledge storage system
    let storage = HierarchicalKnowledgeStorage::new().await;
    let processor = IntelligentKnowledgeProcessor::new_with_mocks().await;
    
    let document = "Einstein's Biography\n\n\
                   Albert Einstein was born in 1879 in Germany. \
                   He developed the theory of relativity which revolutionized physics. \
                   In 1921, he received the Nobel Prize in Physics for his work on the photoelectric effect.";
    
    // WHEN: Processing and storing hierarchical knowledge
    let hierarchical_knowledge = processor.process_knowledge(document, "Einstein Biography").await.unwrap();
    let document_id = storage.store_hierarchical_knowledge(hierarchical_knowledge).await.unwrap();
    
    // THEN: Knowledge is organized in proper hierarchy
    let retrieved = storage.get_hierarchical_knowledge(&document_id).await.unwrap();
    
    // Verify layer hierarchy: Document -> Section -> Paragraph -> Sentence -> Entity
    assert!(retrieved.has_layer_type(LayerType::Document));
    assert!(retrieved.has_layer_type(LayerType::Paragraph));
    assert!(retrieved.has_layer_type(LayerType::Sentence));
    assert!(retrieved.has_layer_type(LayerType::Entity));
    
    // Verify parent-child relationships are established
    let document_layer = retrieved.get_layers_by_type(LayerType::Document)[0].clone();
    let paragraph_layers = retrieved.get_layers_by_type(LayerType::Paragraph);
    
    for paragraph in &paragraph_layers {
        assert!(paragraph.relationships.parent_layers.contains(&document_layer.id));
        assert!(document_layer.relationships.child_layers.contains(&paragraph.id));
    }
}
```

**Unit Tests for Knowledge Layer**

```rust
// tests/unit/storage_layers/knowledge_layer_test.rs

#[cfg(test)]
mod knowledge_layer_tests {
    use super::*;
    use crate::test_support::builders::knowledge_layer_builder::KnowledgeLayerBuilder;

    #[test]
    fn should_create_layer_with_proper_metadata() {
        // RED: Write failing test
        let layer = KnowledgeLayerBuilder::new()
            .with_type(LayerType::Paragraph)
            .with_content("Einstein developed relativity theory.")
            .with_confidence(0.95)
            .with_importance(0.8)
            .build();
        
        assert_eq!(layer.layer_type, LayerType::Paragraph);
        assert_eq!(layer.metadata.confidence_score, 0.95);
        assert_eq!(layer.metadata.importance_score, 0.8);
        assert!(layer.content.primary_text.contains("Einstein"));
    }

    #[test]
    fn should_establish_parent_child_relationships() {
        // RED: Write failing test
        let parent = KnowledgeLayerBuilder::new()
            .with_type(LayerType::Document)
            .with_id("doc_1")
            .build();
        
        let mut child = KnowledgeLayerBuilder::new()
            .with_type(LayerType::Paragraph)
            .with_id("para_1")
            .build();
        
        child.add_parent_relationship("doc_1");
        
        assert!(child.relationships.parent_layers.contains(&"doc_1".to_string()));
        assert_eq!(child.get_primary_parent().unwrap(), "doc_1");
    }

    #[test]
    fn should_calculate_semantic_similarity_between_layers() {
        // RED: Write failing test
        let layer1 = KnowledgeLayerBuilder::new()
            .with_content("Einstein developed relativity theory")
            .with_embedding(vec![0.1, 0.2, 0.3, 0.4])
            .build();
        
        let layer2 = KnowledgeLayerBuilder::new()
            .with_content("Newton formulated gravity laws")
            .with_embedding(vec![0.2, 0.1, 0.4, 0.3])
            .build();
        
        let similarity = layer1.calculate_semantic_similarity(&layer2);
        assert!(similarity > 0.0 && similarity <= 1.0);
    }
}
```

#### Day 13-14: Semantic Link Graph

**Unit Tests for Semantic Link Graph**

```rust
// tests/unit/storage_layers/semantic_link_graph_test.rs

#[cfg(test)]
mod semantic_link_graph_tests {
    use super::*;

    mock! {
        EmbeddingSimilarityCalculator {}
        
        impl EmbeddingSimilarityCalculator for EmbeddingSimilarityCalculator {
            fn calculate_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32;
        }
    }

    #[test]
    fn should_create_entity_relationships() {
        // RED: Write failing test
        let mut graph = SemanticLinkGraph::new();
        
        let relation = EntityRelation {
            source: "Einstein".to_string(),
            predicate: "developed".to_string(),
            target: "relativity".to_string(),
            context: "Einstein developed the theory of relativity".to_string(),
            confidence: 0.95,
            supporting_evidence: vec!["Nobel Prize citation".to_string()],
        };
        
        graph.add_entity_relationship(relation.clone());
        
        let retrieved = graph.get_entity_relationships("Einstein");
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].target, "relativity");
    }

    #[test]
    fn should_discover_conceptual_connections() {
        // RED: Write failing test
        let mut mock_similarity = MockEmbeddingSimilarityCalculator::new();
        mock_similarity
            .expect_calculate_similarity()
            .return_const(0.85f32);
        
        let mut graph = SemanticLinkGraph::with_similarity_calculator(Box::new(mock_similarity));
        
        let concept1 = ConceptualNode {
            concept: "relativity".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            context: "Einstein's theory".to_string(),
        };
        
        let concept2 = ConceptualNode {
            concept: "spacetime".to_string(),
            embedding: vec![0.15, 0.25, 0.35],
            context: "curved by mass".to_string(),
        };
        
        graph.add_conceptual_node(concept1);
        graph.add_conceptual_node(concept2);
        
        let connections = graph.discover_conceptual_connections(0.8).unwrap();
        assert!(connections.len() > 0);
        assert!(connections[0].similarity_score > 0.8);
    }

    #[test]
    fn should_maintain_temporal_links() {
        // RED: Write failing test
        let mut graph = SemanticLinkGraph::new();
        
        let temporal_link = TemporalConnection {
            earlier_entity: "Einstein_birth".to_string(),
            later_entity: "relativity_theory".to_string(),
            temporal_relation: TemporalRelationType::Before,
            time_span: Some(Duration::from_secs(365 * 24 * 3600 * 26)), // 26 years
            confidence: 0.9,
        };
        
        graph.add_temporal_link(temporal_link.clone());
        
        let timeline = graph.get_temporal_sequence("Einstein_birth");
        assert!(timeline.contains(&temporal_link));
    }
}
```

#### Day 15: Cross-Reference System

**Unit Tests for Cross-Reference Management**

```rust
// tests/unit/storage_layers/cross_reference_system_test.rs

#[cfg(test)]
mod cross_reference_tests {
    use super::*;

    #[test]
    fn should_create_bidirectional_references() {
        // RED: Write failing test
        let mut ref_system = CrossReferenceSystem::new();
        
        let reference = CrossReference {
            source_layer: "layer_1".to_string(),
            target_layer: "layer_2".to_string(),
            reference_type: ReferenceType::Elaboration,
            strength: 0.8,
            context: "Provides additional detail".to_string(),
        };
        
        ref_system.add_cross_reference(reference.clone());
        
        // Should create bidirectional links
        let forward_refs = ref_system.get_references_from("layer_1");
        let backward_refs = ref_system.get_references_to("layer_2");
        
        assert_eq!(forward_refs.len(), 1);
        assert_eq!(backward_refs.len(), 1);
        assert_eq!(forward_refs[0].target_layer, "layer_2");
        assert_eq!(backward_refs[0].source_layer, "layer_1");
    }

    #[test]
    fn should_resolve_reference_chains() {
        // RED: Write failing test
        let mut ref_system = CrossReferenceSystem::new();
        
        // Create chain: A -> B -> C
        ref_system.add_cross_reference(CrossReference {
            source_layer: "A".to_string(),
            target_layer: "B".to_string(),
            reference_type: ReferenceType::Continuation,
            strength: 0.9,
            context: "A continues in B".to_string(),
        });
        
        ref_system.add_cross_reference(CrossReference {
            source_layer: "B".to_string(),
            target_layer: "C".to_string(),
            reference_type: ReferenceType::Continuation,
            strength: 0.8,
            context: "B continues in C".to_string(),
        });
        
        let chain = ref_system.resolve_reference_chain("A", ReferenceType::Continuation, 3);
        assert_eq!(chain.len(), 3); // A -> B -> C
        assert_eq!(chain[0], "A");
        assert_eq!(chain[1], "B");
        assert_eq!(chain[2], "C");
    }
}
```

### Week 4: Storage Integration and Indexing

#### Day 16-17: Hierarchical Index System

**Unit Tests for Hierarchical Index**

```rust
// tests/unit/storage_layers/hierarchical_index_test.rs

#[cfg(test)]
mod hierarchical_index_tests {
    use super::*;

    mock! {
        EmbeddingIndex {}
        
        impl EmbeddingIndex for EmbeddingIndex {
            fn add_embedding(&mut self, id: &str, embedding: &[f32]) -> Result<()>;
            fn search_similar(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(String, f32)>>;
            fn remove_embedding(&mut self, id: &str) -> Result<()>;
        }
    }

    #[test]
    fn should_index_layers_by_type_and_hierarchy() {
        // RED: Write failing test
        let mut index = HierarchicalIndex::new();
        
        let document_layer = KnowledgeLayerBuilder::new()
            .with_type(LayerType::Document)
            .with_id("doc_1")
            .build();
        
        let paragraph_layer = KnowledgeLayerBuilder::new()
            .with_type(LayerType::Paragraph)
            .with_id("para_1")
            .with_parent("doc_1")
            .build();
        
        index.add_layer(document_layer);
        index.add_layer(paragraph_layer);
        
        let document_layers = index.get_layers_by_type(LayerType::Document);
        let paragraph_layers = index.get_layers_by_type(LayerType::Paragraph);
        
        assert_eq!(document_layers.len(), 1);
        assert_eq!(paragraph_layers.len(), 1);
        
        let children = index.get_child_layers("doc_1");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, "para_1");
    }

    #[tokio::test]
    async fn should_perform_multi_level_search() {
        // RED: Write failing test
        let mut mock_embedding_index = MockEmbeddingIndex::new();
        mock_embedding_index
            .expect_search_similar()
            .times(1)
            .return_once(|_, _| Ok(vec![
                ("doc_1".to_string(), 0.95),
                ("para_1".to_string(), 0.87),
                ("sent_1".to_string(), 0.82),
            ]));
        
        let index = HierarchicalIndex::with_embedding_index(Box::new(mock_embedding_index));
        
        let query_embedding = vec![0.1, 0.2, 0.3, 0.4];
        let results = index.search_multi_level(&query_embedding, 5).await.unwrap();
        
        assert_eq!(results.len(), 3);
        assert!(results[0].layer_id == "doc_1");
        assert!(results[0].similarity_score > 0.9);
    }

    #[test]
    fn should_maintain_layer_hierarchy_during_updates() {
        // RED: Write failing test
        let mut index = HierarchicalIndex::new();
        
        // Add initial hierarchy
        let doc = KnowledgeLayerBuilder::new()
            .with_type(LayerType::Document)
            .with_id("doc_1")
            .build();
        let para = KnowledgeLayerBuilder::new()
            .with_type(LayerType::Paragraph)
            .with_id("para_1")
            .with_parent("doc_1")
            .build();
        
        index.add_layer(doc);
        index.add_layer(para);
        
        // Update paragraph content
        let updated_para = KnowledgeLayerBuilder::new()
            .with_type(LayerType::Paragraph)
            .with_id("para_1")
            .with_parent("doc_1")
            .with_content("Updated content")
            .build();
        
        index.update_layer(updated_para);
        
        // Verify hierarchy is maintained
        let children = index.get_child_layers("doc_1");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, "para_1");
        assert!(children[0].content.primary_text.contains("Updated content"));
    }
}
```

#### Day 18-19: Storage Backend Integration

**Integration Tests for Storage Backend**

```rust
// tests/integration/hierarchical_storage_integration.rs

#[cfg(test)]
mod storage_integration_tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn should_persist_and_retrieve_hierarchical_knowledge() {
        // Integration test with real storage backend
        let temp_dir = TempDir::new().unwrap();
        let storage_config = StorageConfig {
            data_directory: temp_dir.path().to_path_buf(),
            index_type: IndexType::HNSW,
            compression_enabled: true,
        };
        
        let storage = HierarchicalKnowledgeStorage::new(storage_config).await.unwrap();
        
        // Create test hierarchical knowledge
        let hierarchical_knowledge = HierarchicalKnowledgeBuilder::new()
            .with_document_layer("Test Document", "This is a test document about Einstein.")
            .with_paragraph_layer("Para 1", "Einstein was a physicist.")
            .with_entity_layer("Einstein", "Person", "Famous physicist")
            .build();
        
        // Store and retrieve
        let document_id = storage.store_hierarchical_knowledge(hierarchical_knowledge.clone()).await.unwrap();
        let retrieved = storage.get_hierarchical_knowledge(&document_id).await.unwrap();
        
        // Verify integrity
        assert_eq!(retrieved.document_id, document_id);
        assert_eq!(retrieved.knowledge_layers.len(), hierarchical_knowledge.knowledge_layers.len());
        assert_eq!(retrieved.semantic_links.total_links(), hierarchical_knowledge.semantic_links.total_links());
    }

    #[tokio::test] 
    async fn should_handle_concurrent_storage_operations() {
        // Integration test for concurrent access
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(HierarchicalKnowledgeStorage::new(
            StorageConfig::with_directory(temp_dir.path())
        ).await.unwrap());
        
        let mut handles = vec![];
        
        // Spawn multiple concurrent storage operations
        for i in 0..10 {
            let storage_clone = storage.clone();
            let handle = tokio::spawn(async move {
                let knowledge = HierarchicalKnowledgeBuilder::new()
                    .with_document_layer(&format!("Doc {}", i), &format!("Content {}", i))
                    .build();
                
                storage_clone.store_hierarchical_knowledge(knowledge).await
            });
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        let results: Result<Vec<_>, _> = futures::future::try_join_all(handles).await;
        let document_ids = results.unwrap();
        
        // Verify all documents were stored successfully
        assert_eq!(document_ids.len(), 10);
        for doc_id in document_ids {
            let retrieved = storage.get_hierarchical_knowledge(&doc_id?).await;
            assert!(retrieved.is_ok());
        }
    }
}
```

#### Day 20: Migration and Compatibility

**Unit Tests for Data Migration**

```rust
// tests/unit/storage_layers/migration_test.rs

#[cfg(test)]
mod migration_tests {
    use super::*;

    mock! {
        LegacyStorage {}
        
        impl LegacyStorage for LegacyStorage {
            fn get_all_triples(&self) -> Result<Vec<Triple>>;
            fn get_all_chunks(&self) -> Result<Vec<KnowledgeChunk>>;
        }
    }

    #[tokio::test]
    async fn should_migrate_existing_triples_to_hierarchical_structure() {
        // RED: Write failing test
        let mut mock_legacy = MockLegacyStorage::new();
        mock_legacy
            .expect_get_all_triples()
            .times(1)
            .return_once(|| Ok(vec![
                Triple::new("Einstein".to_string(), "is".to_string(), "physicist".to_string()).unwrap(),
                Triple::new("Einstein".to_string(), "developed".to_string(), "relativity".to_string()).unwrap(),
            ]));
        
        mock_legacy
            .expect_get_all_chunks()
            .times(1)
            .return_once(|| Ok(vec![
                KnowledgeChunk {
                    id: "chunk_1".to_string(),
                    content: "Einstein was a physicist who developed relativity.".to_string(),
                    title: "Einstein Info".to_string(),
                }
            ]));
        
        let migrator = DataMigrator::new(Box::new(mock_legacy));
        let hierarchical_storage = HierarchicalKnowledgeStorage::new_in_memory().await.unwrap();
        
        let migration_result = migrator.migrate_to_hierarchical(&hierarchical_storage).await.unwrap();
        
        assert_eq!(migration_result.triples_migrated, 2);
        assert_eq!(migration_result.chunks_migrated, 1);
        assert_eq!(migration_result.layers_created, 3); // Document, Entity layers for Einstein, relativity
    }

    #[test]
    fn should_preserve_temporal_information_during_migration() {
        // RED: Write failing test
        let legacy_triple = Triple::with_metadata(
            "Einstein".to_string(),
            "born_in".to_string(),
            "1879".to_string(),
            0.95,
            Some("birth_record".to_string())
        ).unwrap();
        
        let migrator = DataMigrator::new(Box::new(MockLegacyStorage::new()));
        let migrated_layer = migrator.convert_triple_to_layer(&legacy_triple).unwrap();
        
        assert_eq!(migrated_layer.layer_type, LayerType::Entity);
        assert_eq!(migrated_layer.metadata.confidence_score, 0.95);
        assert!(migrated_layer.content.extracted_facts.contains(&legacy_triple));
    }
}
```

## Phase 3: Enhanced Retrieval System (Weeks 5-6)

### Week 5: Multi-Layer Query Strategy

#### Day 21-22: Hierarchical Retriever Core

**Acceptance Test**: `tests/acceptance/enhanced_retrieval_flow.rs`

```rust
#[tokio::test]
async fn should_perform_intelligent_multi_layer_retrieval() {
    // GIVEN: A populated hierarchical knowledge storage with rich content
    let storage = setup_test_storage_with_einstein_knowledge().await;
    let retriever = HierarchicalRetriever::new(storage.clone()).await;
    
    // WHEN: Querying for complex, multi-hop information
    let query = "What scientific achievements led to Einstein receiving recognition?";
    let result = retriever.retrieve_knowledge(query, 2000).await.unwrap();
    
    // THEN: Retrieval provides comprehensive, contextually rich information
    assert!(result.direct_matches.len() > 0);
    assert!(result.contextual_information.parent_context.len() > 0);
    assert!(result.entity_relationships.len() > 0);
    
    // Verify multi-hop reasoning worked
    assert!(result.retrieval_path.reasoning_steps.len() >= 2);
    assert!(result.confidence_scores.overall_confidence > 0.7);
    
    // Verify context preservation
    let optimized_response = result.to_llm_optimized_response();
    assert!(optimized_response.main_content.contains("Einstein"));
    assert!(optimized_response.main_content.contains("relativity"));
    assert!(optimized_response.main_content.contains("Nobel Prize"));
}
```

**Unit Tests for Query Processing**

```rust
// tests/unit/retrieval_systems/query_processor_test.rs

#[cfg(test)]
mod query_processor_tests {
    use super::*;

    mock! {
        QueryAnalyzer {}
        
        impl QueryAnalyzer for QueryAnalyzer {
            fn analyze_query_intent(&self, query: &str) -> QueryIntent;
            fn extract_query_entities(&self, query: &str) -> Vec<String>;
            fn determine_complexity_level(&self, query: &str) -> QueryComplexity;
        }
    }

    #[test]
    fn should_analyze_query_intent_correctly() {
        // RED: Write failing test
        let mut mock_analyzer = MockQueryAnalyzer::new();
        mock_analyzer
            .expect_analyze_query_intent()
            .with(eq("What did Einstein discover?"))
            .times(1)
            .return_once(|_| QueryIntent {
                intent_type: IntentType::Discovery,
                focus_entities: vec!["Einstein".to_string()],
                required_relationships: vec!["discovered".to_string(), "invented".to_string()],
                temporal_context: None,
            });
        
        let processor = QueryProcessor::with_analyzer(Box::new(mock_analyzer));
        let intent = processor.analyze_query("What did Einstein discover?");
        
        assert_eq!(intent.intent_type, IntentType::Discovery);
        assert!(intent.focus_entities.contains(&"Einstein".to_string()));
        assert!(intent.required_relationships.contains(&"discovered".to_string()));
    }

    #[test]
    fn should_determine_query_complexity_appropriately() {
        // RED: Write failing test
        let mut mock_analyzer = MockQueryAnalyzer::new();
        mock_analyzer
            .expect_determine_complexity_level()
            .with(predicate::str::contains("What were the implications"))
            .times(1)
            .return_once(|_| QueryComplexity::High);
        
        mock_analyzer
            .expect_determine_complexity_level()
            .with(eq("Who is Einstein?"))
            .times(1)
            .return_once(|_| QueryComplexity::Low);
        
        let processor = QueryProcessor::with_analyzer(Box::new(mock_analyzer));
        
        let complex_query = "What were the implications of Einstein's theory for modern physics?";
        let simple_query = "Who is Einstein?";
        
        assert_eq!(processor.determine_complexity(complex_query), QueryComplexity::High);
        assert_eq!(processor.determine_complexity(simple_query), QueryComplexity::Low);
    }
}
```

#### Day 23-24: Graph Traversal and Context Expansion

**Unit Tests for Graph Traverser**

```rust
// tests/unit/retrieval_systems/graph_traverser_test.rs

#[cfg(test)]
mod graph_traverser_tests {
    use super::*;

    mock! {
        LayerGraph {}
        
        impl LayerGraph for LayerGraph {
            fn get_neighbors(&self, layer_id: &str) -> Vec<String>;
            fn get_relationship_strength(&self, from: &str, to: &str) -> f32;
            fn find_path(&self, from: &str, to: &str, max_depth: usize) -> Option<Vec<String>>;
        }
    }

    #[test]
    fn should_expand_context_via_semantic_neighbors() {
        // RED: Write failing test
        let mut mock_graph = MockLayerGraph::new();
        mock_graph
            .expect_get_neighbors()
            .with(eq("einstein_layer"))
            .times(1)
            .return_once(|_| vec![
                "relativity_layer".to_string(),
                "physics_layer".to_string(),
                "nobel_prize_layer".to_string(),
            ]);
        
        mock_graph
            .expect_get_relationship_strength()
            .returning(|_, _| 0.8);
        
        let traverser = GraphTraverser::with_graph(Box::new(mock_graph));
        let initial_matches = vec![LayerMatch {
            layer_id: "einstein_layer".to_string(),
            similarity_score: 0.95,
            content: "Einstein content".to_string(),
            related_entities: vec![],
        }];
        
        let expanded = traverser.expand_context_via_graph(&initial_matches, 0.7).unwrap();
        
        assert!(expanded.len() > initial_matches.len());
        assert!(expanded.iter().any(|m| m.layer_id == "relativity_layer"));
        assert!(expanded.iter().any(|m| m.layer_id == "physics_layer"));
    }

    #[test]
    fn should_find_multi_hop_connections() {
        // RED: Write failing test
        let mut mock_graph = MockLayerGraph::new();
        mock_graph
            .expect_find_path()
            .with(eq("einstein_layer"), eq("quantum_mechanics_layer"), eq(3))
            .times(1)
            .return_once(|_, _, _| Some(vec![
                "einstein_layer".to_string(),
                "photoelectric_effect_layer".to_string(),
                "quantum_mechanics_layer".to_string(),
            ]));
        
        let traverser = GraphTraverser::with_graph(Box::new(mock_graph));
        let path = traverser.find_knowledge_path("einstein_layer", "quantum_mechanics_layer", 3).unwrap();
        
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "einstein_layer");
        assert_eq!(path[2], "quantum_mechanics_layer");
        assert!(path[1].contains("photoelectric"));
    }

    #[tokio::test]
    async fn should_respect_traversal_depth_limits() {
        // RED: Write failing test
        let traverser = GraphTraverser::new();
        let config = TraversalConfig {
            max_depth: 2,
            min_relationship_strength: 0.5,
            max_expanded_nodes: 10,
        };
        
        let initial_layer = LayerMatch {
            layer_id: "start_layer".to_string(),
            similarity_score: 1.0,
            content: "Starting content".to_string(),
            related_entities: vec![],
        };
        
        let expanded = traverser.expand_with_config(&[initial_layer], config).await.unwrap();
        
        // Should not exceed max_depth or max_expanded_nodes
        assert!(expanded.len() <= 11); // initial + max_expanded_nodes
        
        // Verify no layers beyond max_depth are included
        for layer_match in &expanded {
            let depth = traverser.calculate_depth_from_initial(&layer_match.layer_id);
            assert!(depth <= 2);
        }
    }
}
```

#### Day 25: Context Assembly System

**Unit Tests for Context Assembler**

```rust
// tests/unit/retrieval_systems/context_assembler_test.rs

#[cfg(test)]
mod context_assembler_tests {
    use super::*;

    mock! {
        ContextPrioritizer {}
        
        impl ContextPrioritizer for ContextPrioritizer {
            fn prioritize_contexts(&self, contexts: &[ContextCandidate]) -> Vec<PrioritizedContext>;
            fn calculate_context_value(&self, context: &ContextCandidate) -> f32;
        }
    }

    #[tokio::test]
    async fn should_assemble_hierarchical_context_efficiently() {
        // RED: Write failing test
        let mut mock_prioritizer = MockContextPrioritizer::new();
        mock_prioritizer
            .expect_prioritize_contexts()
            .times(1)
            .return_once(|contexts| {
                contexts.iter().enumerate().map(|(i, ctx)| PrioritizedContext {
                    context: ctx.clone(),
                    priority_score: 1.0 - (i as f32 * 0.1),
                    inclusion_reason: format!("Priority rank {}", i + 1),
                }).collect()
            });
        
        let assembler = ContextAssembler::with_prioritizer(Box::new(mock_prioritizer));
        
        let layer_matches = vec![
            LayerMatch::with_content("einstein_layer", "Einstein was a physicist"),
            LayerMatch::with_content("relativity_layer", "Relativity theory revolutionized physics"),
        ];
        
        let assembled = assembler.assemble_hierarchical_context(&layer_matches, 500).await.unwrap();
        
        assert!(assembled.total_size() <= 500);
        assert!(assembled.primary_content.len() > 0);
        assert!(assembled.parent_context.len() > 0);
        assert!(assembled.entity_contexts.len() > 0);
    }

    #[test]
    fn should_maintain_context_coherence() {
        // RED: Write failing test
        let assembler = ContextAssembler::new();
        
        let contexts = vec![
            ContextCandidate {
                content: "Einstein developed relativity".to_string(),
                context_type: ContextType::Primary,
                relevance_score: 0.95,
                layer_id: "layer_1".to_string(),
            },
            ContextCandidate {
                content: "Quantum mechanics is different".to_string(),
                context_type: ContextType::Supporting,
                relevance_score: 0.3,
                layer_id: "layer_2".to_string(),
            },
            ContextCandidate {
                content: "Relativity has implications for spacetime".to_string(),
                context_type: ContextType::Elaboration,
                relevance_score: 0.8,
                layer_id: "layer_3".to_string(),
            },
        ];
        
        let coherent_context = assembler.ensure_context_coherence(&contexts, 0.7).unwrap();
        
        // Should exclude low-relevance, incoherent content
        assert_eq!(coherent_context.len(), 2);
        assert!(coherent_context[0].content.contains("Einstein"));
        assert!(coherent_context[1].content.contains("Relativity"));
    }

    #[test]
    fn should_create_contextual_bridges_between_disconnected_content() {
        // RED: Write failing test
        let assembler = ContextAssembler::new();
        
        let disconnected_content = vec![
            LayerMatch::with_content("layer_1", "Einstein won Nobel Prize"),
            LayerMatch::with_content("layer_2", "Photoelectric effect explained"),
        ];
        
        let bridged_context = assembler.create_contextual_bridges(&disconnected_content).await.unwrap();
        
        // Should have identified connection between Nobel Prize and photoelectric effect
        assert!(bridged_context.bridges.len() > 0);
        assert!(bridged_context.bridges[0].bridge_content.contains("photoelectric"));
        assert!(bridged_context.bridges[0].bridge_content.contains("Nobel"));
    }
}
```

### Week 6: Response Optimization and Caching

#### Day 26-27: Response Generation and Optimization

**Unit Tests for Response Optimizer**

```rust
// tests/unit/retrieval_systems/response_optimizer_test.rs

#[cfg(test)]
mod response_optimizer_tests {
    use super::*;

    mock! {
        ResponseFormatter {}
        
        impl ResponseFormatter for ResponseFormatter {
            fn format_for_llm(&self, content: &EnrichedKnowledgeResult) -> LLMOptimizedResponse;
            fn estimate_token_count(&self, content: &str) -> usize;
            fn compress_content(&self, content: &str, target_size: usize) -> String;
        }
    }

    #[test]
    fn should_optimize_response_for_main_llm() {
        // RED: Write failing test
        let mut mock_formatter = MockResponseFormatter::new();
        mock_formatter
            .expect_format_for_llm()
            .times(1)
            .return_once(|result| LLMOptimizedResponse {
                main_content: format!("Optimized: {}", result.summary),
                supporting_facts: result.primary_facts.clone(),
                entity_context: vec!["Einstein: physicist".to_string()],
                confidence: result.confidence_assessment.overall_confidence,
                complexity_handled_internally: true,
            });
        
        let optimizer = ResponseOptimizer::with_formatter(Box::new(mock_formatter));
        
        let enriched_result = EnrichedKnowledgeResult {
            summary: "Einstein developed relativity".to_string(),
            primary_facts: vec![Triple::new("Einstein".to_string(), "developed".to_string(), "relativity".to_string()).unwrap()],
            contextual_entities: vec![],
            relationship_map: HashMap::new(),
            confidence_assessment: ConfidenceAssessment { overall_confidence: 0.9 },
            retrieval_metadata: RetrievalMetadata::default(),
        };
        
        let optimized = optimizer.optimize_for_llm(enriched_result);
        
        assert!(optimized.main_content.starts_with("Optimized:"));
        assert_eq!(optimized.confidence, 0.9);
        assert!(optimized.complexity_handled_internally);
    }

    #[test]
    fn should_compress_content_while_preserving_key_information() {
        // RED: Write failing test
        let mut mock_formatter = MockResponseFormatter::new();
        mock_formatter
            .expect_estimate_token_count()
            .return_const(1500usize);
        
        mock_formatter
            .expect_compress_content()
            .with(predicate::str::contains("Einstein"), eq(1000))
            .times(1)
            .return_once(|content, _| {
                // Simulate intelligent compression
                content.split_whitespace()
                    .filter(|word| !["the", "a", "an"].contains(word))
                    .collect::<Vec<_>>()
                    .join(" ")
            });
        
        let optimizer = ResponseOptimizer::with_formatter(Box::new(mock_formatter));
        
        let long_content = "The famous physicist Albert Einstein developed the revolutionary theory of relativity";
        let compressed = optimizer.compress_while_preserving_entities(long_content, 1000);
        
        // Should preserve key entities while reducing size
        assert!(compressed.contains("Einstein"));
        assert!(compressed.contains("relativity"));
        assert!(!compressed.contains("the"));
    }
}
```

#### Day 28-29: Intelligent Caching System

**Unit Tests for Intelligent Cache**

```rust
// tests/unit/retrieval_systems/intelligent_cache_test.rs

#[cfg(test)]
mod intelligent_cache_tests {
    use super::*;

    mock! {
        AccessPatternAnalyzer {}
        
        impl AccessPatternAnalyzer for AccessPatternAnalyzer {
            fn analyze_query_patterns(&self) -> Result<QueryPatternAnalysis>;
            fn predict_likely_queries(&self) -> Vec<PredictedQuery>;
            fn should_precompute(&self, query_signature: &str) -> bool;
        }
    }

    #[tokio::test]
    async fn should_cache_frequently_accessed_contexts() {
        // RED: Write failing test
        let cache = IntelligentCache::new(CacheConfig {
            max_entries: 1000,
            ttl: Duration::from_secs(3600),
            precompute_threshold: 0.7,
        });
        
        let context = HierarchicalContext {
            primary_content: "Einstein information".to_string(),
            parent_context: "Physics context".to_string(),
            entity_contexts: vec!["Einstein: physicist".to_string()],
        };
        
        cache.store("einstein_query_sig", context.clone()).await;
        
        let retrieved = cache.get("einstein_query_sig").await.unwrap();
        assert_eq!(retrieved.primary_content, context.primary_content);
        
        // Verify access tracking
        let access_count = cache.get_access_count("einstein_query_sig");
        assert_eq!(access_count, 1);
    }

    #[tokio::test]
    async fn should_precompute_likely_contexts_based_on_patterns() {
        // RED: Write failing test
        let mut mock_analyzer = MockAccessPatternAnalyzer::new();
        mock_analyzer
            .expect_analyze_query_patterns()
            .times(1)
            .return_once(|| Ok(QueryPatternAnalysis {
                high_frequency_patterns: vec![
                    QueryPattern {
                        signature: "einstein_relativity".to_string(),
                        frequency: 0.8,
                        entities: vec!["Einstein".to_string(), "relativity".to_string()],
                    }
                ]
            }));
        
        let cache = IntelligentCache::with_analyzer(Box::new(mock_analyzer));
        
        cache.precompute_likely_contexts().await.unwrap();
        
        // Should have precomputed context for high-frequency pattern
        let precomputed = cache.get("einstein_relativity").await;
        assert!(precomputed.is_some());
    }

    #[test]
    fn should_evict_least_valuable_entries_when_at_capacity() {
        // RED: Write failing test
        let cache = IntelligentCache::new(CacheConfig {
            max_entries: 2,
            ttl: Duration::from_secs(3600),
            precompute_threshold: 0.7,
        });
        
        let context1 = HierarchicalContext::with_content("Content 1");
        let context2 = HierarchicalContext::with_content("Content 2");
        let context3 = HierarchicalContext::with_content("Content 3");
        
        // Fill cache to capacity
        cache.store_sync("query1", context1.clone());
        cache.store_sync("query2", context2.clone());
        
        // Access query1 to make it more valuable
        cache.get_sync("query1");
        cache.get_sync("query1");
        
        // Adding third entry should evict query2 (less valuable)
        cache.store_sync("query3", context3.clone());
        
        assert!(cache.contains("query1"));
        assert!(!cache.contains("query2"));
        assert!(cache.contains("query3"));
    }
}
```

#### Day 30: Performance Optimization

**Performance and Load Tests**

```rust
// tests/integration/performance_integration.rs

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn should_meet_retrieval_latency_requirements() {
        // Integration test for performance requirements
        let storage = setup_large_test_knowledge_base(10000).await; // 10k documents
        let retriever = HierarchicalRetriever::new(storage).await;
        
        let queries = vec![
            "What did Einstein discover?",
            "How does relativity relate to spacetime?",
            "What are the implications of quantum mechanics?",
        ];
        
        for query in queries {
            let start = Instant::now();
            let result = retriever.retrieve_knowledge(query, 2000).await.unwrap();
            let duration = start.elapsed();
            
            // Performance requirement: <200ms for 95% of complex queries
            assert!(duration < Duration::from_millis(200), 
                "Query '{}' took {:?}, expected <200ms", query, duration);
            
            assert!(result.confidence_scores.overall_confidence > 0.7);
            assert!(result.direct_matches.len() > 0);
        }
    }

    #[tokio::test]
    async fn should_handle_concurrent_retrieval_requests() {
        // Load test for concurrent access
        let storage = setup_test_knowledge_base().await;
        let retriever = Arc::new(HierarchicalRetriever::new(storage).await);
        
        let mut handles = vec![];
        
        // Spawn 50 concurrent retrieval requests
        for i in 0..50 {
            let retriever_clone = retriever.clone();
            let handle = tokio::spawn(async move {
                let query = format!("Query number {}: What is relativity?", i);
                retriever_clone.retrieve_knowledge(&query, 1000).await
            });
            handles.push(handle);
        }
        
        let start = Instant::now();
        let results: Result<Vec<_>, _> = futures::future::try_join_all(handles).await;
        let duration = start.elapsed();
        
        let retrieval_results = results.unwrap();
        
        // All requests should complete successfully
        assert_eq!(retrieval_results.len(), 50);
        for result in retrieval_results {
            let result = result.unwrap();
            assert!(result.confidence_scores.overall_confidence > 0.0);
        }
        
        // Total time should be reasonable for concurrent processing
        assert!(duration < Duration::from_secs(10));
    }

    #[tokio::test]
    async fn should_maintain_memory_usage_within_limits() {
        // Memory usage test
        let initial_memory = get_process_memory_usage();
        
        let storage = setup_large_test_knowledge_base(1000).await;
        let retriever = HierarchicalRetriever::new(storage).await;
        
        // Process many queries to test memory stability
        for i in 0..100 {
            let query = format!("Test query {}", i);
            let _result = retriever.retrieve_knowledge(&query, 1000).await.unwrap();
            
            if i % 10 == 0 {
                // Check memory every 10 queries
                let current_memory = get_process_memory_usage();
                let memory_growth = current_memory - initial_memory;
                
                // Memory growth should be bounded
                assert!(memory_growth < 1_000_000_000, // 1GB growth limit
                    "Memory grew by {} bytes after {} queries", memory_growth, i);
            }
        }
    }
}
```

## Phase 4: Full System Integration (Weeks 7-8)

### Week 7: End-to-End Integration

#### Day 31-32: Full Pipeline Integration

**End-to-End Acceptance Tests**

```rust
// tests/acceptance/full_system_integration.rs

#[cfg(test)]
mod full_system_integration_tests {
    use super::*;

    #[tokio::test]
    async fn should_complete_full_knowledge_lifecycle() {
        // GIVEN: A complete enhanced knowledge storage system
        let system = EnhancedKnowledgeSystem::new().await.unwrap();
        
        let sample_document = r#"
            # Albert Einstein: A Revolutionary Physicist
            
            Albert Einstein (1879-1955) was a German-born theoretical physicist who 
            developed the theory of relativity, one of the two pillars of modern physics.
            
            ## Scientific Contributions
            
            Einstein's most famous equation, E=mc², shows the mass-energy equivalence.
            His work on the photoelectric effect earned him the Nobel Prize in Physics in 1921.
            
            ## Legacy
            
            Einstein's theories revolutionized our understanding of space, time, and gravity.
            His work laid the foundation for quantum mechanics and modern cosmology.
        "#;
        
        // WHEN: Storing knowledge through the enhanced system
        let store_result = system.store_knowledge_enhanced(
            sample_document,
            "Einstein Biography",
            "biography"
        ).await.unwrap();
        
        // THEN: Knowledge is processed and stored hierarchically
        assert!(store_result.layers_created >= 4); // Document, Section, Paragraph, Entity layers
        assert!(store_result.entities_extracted >= 5); // Einstein, relativity, physics, Nobel Prize, etc.
        assert!(store_result.relationships_mapped >= 3);
        assert!(store_result.context_preservation_score > 0.8);
        
        // AND WHEN: Querying the stored knowledge
        let queries = vec![
            "What did Einstein discover?",
            "Why did Einstein win the Nobel Prize?",
            "How did Einstein's work impact modern physics?",
        ];
        
        for query in queries {
            let query_result = system.query_knowledge_enhanced(query, 2000).await.unwrap();
            
            // THEN: Queries return comprehensive, contextual answers
            assert!(query_result.confidence >= 0.7);
            assert!(query_result.answer_context.len() > 100);
            assert!(query_result.supporting_facts.len() > 0);
            assert!(query_result.entity_context.len() > 0);
            
            // Verify multi-hop reasoning capabilities
            if query.contains("impact") {
                assert!(query_result.context_layers_used >= 3);
                assert!(query_result.answer_context.contains("relativity"));
                assert!(query_result.answer_context.contains("quantum"));
            }
        }
    }

    #[tokio::test]
    async fn should_handle_complex_multi_document_scenarios() {
        // GIVEN: Multiple related documents
        let system = EnhancedKnowledgeSystem::new().await.unwrap();
        
        let documents = vec![
            ("Einstein Biography", "Albert Einstein was born in 1879 and developed relativity theory."),
            ("Relativity Theory", "Special relativity deals with objects moving at constant speeds."),
            ("Quantum Physics", "Einstein contributed to quantum theory through the photoelectric effect."),
            ("Nobel Prizes", "The Nobel Prize in Physics 1921 was awarded to Einstein for photoelectric effect."),
        ];
        
        // WHEN: Storing multiple documents
        let mut document_ids = vec![];
        for (title, content) in documents {
            let result = system.store_knowledge_enhanced(content, title, "physics").await.unwrap();
            document_ids.push(result.document_id);
        }
        
        // AND WHEN: Querying across documents
        let cross_doc_query = "How does Einstein's work connect relativity theory to quantum physics and his Nobel Prize?";
        let result = system.query_knowledge_enhanced(cross_doc_query, 3000).await.unwrap();
        
        // THEN: System provides comprehensive cross-document analysis
        assert!(result.confidence >= 0.8);
        assert!(result.context_layers_used >= 6); // Multiple documents involved
        assert!(result.answer_context.contains("relativity"));
        assert!(result.answer_context.contains("quantum"));
        assert!(result.answer_context.contains("Nobel"));
        assert!(result.answer_context.contains("photoelectric"));
        
        // Verify cross-document relationship detection
        assert!(result.supporting_facts.iter().any(|fact| 
            (fact.subject.contains("Einstein") && fact.object.contains("relativity")) ||
            (fact.subject.contains("Einstein") && fact.object.contains("quantum"))
        ));
    }

    #[tokio::test]
    async fn should_maintain_performance_under_realistic_load() {
        // GIVEN: A system with realistic data load
        let system = EnhancedKnowledgeSystem::new().await.unwrap();
        
        // Load 100 documents of varying complexity
        for i in 0..100 {
            let content = generate_realistic_document_content(i);
            let title = format!("Document {}", i);
            system.store_knowledge_enhanced(&content, &title, "general").await.unwrap();
        }
        
        // WHEN: Performing realistic query workload
        let queries = generate_realistic_query_workload(50);
        let start_time = Instant::now();
        
        for query in queries {
            let result = system.query_knowledge_enhanced(&query, 2000).await;
            assert!(result.is_ok());
            
            let result = result.unwrap();
            assert!(result.confidence > 0.5); // Reasonable confidence threshold
        }
        
        let total_time = start_time.elapsed();
        
        // THEN: Performance meets requirements
        assert!(total_time < Duration::from_secs(25)); // 50 queries in <25 seconds
        
        // Verify memory usage is stable
        let memory_usage = get_system_memory_usage();
        assert!(memory_usage < 3_000_000_000); // <3GB total memory usage
    }
}
```

#### Day 33-34: MCP Tool Integration

**Integration Tests for Enhanced MCP Tools**

```rust
// tests/integration/mcp_tools_integration.rs

#[cfg(test)]
mod mcp_tools_integration_tests {
    use super::*;

    #[tokio::test]
    async fn should_integrate_enhanced_store_knowledge_with_existing_mcp_framework() {
        // GIVEN: MCP server with enhanced knowledge storage
        let mcp_server = create_test_mcp_server_with_enhanced_storage().await;
        
        // WHEN: Calling enhanced store_knowledge through MCP
        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "store_knowledge",
                "arguments": {
                    "content": "Einstein developed the theory of relativity, which revolutionized physics. He received the Nobel Prize in 1921 for his work on the photoelectric effect.",
                    "title": "Einstein's Scientific Contributions",
                    "category": "physics",
                    "source": "physics_textbook"
                }
            }
        });
        
        let response = mcp_server.handle_request(request).await.unwrap();
        
        // THEN: Response indicates successful enhanced processing
        let result = response["result"].as_object().unwrap();
        assert_eq!(result["stored"], true);
        assert!(result["layers_created"].as_u64().unwrap() >= 3);
        assert!(result["entities_extracted"].as_u64().unwrap() >= 4);
        assert!(result["relationships_mapped"].as_u64().unwrap() >= 2);
        assert!(result["context_preservation_score"].as_f64().unwrap() > 0.8);
    }

    #[tokio::test]
    async fn should_provide_enhanced_query_capabilities_through_mcp() {
        // GIVEN: MCP server with stored enhanced knowledge
        let mcp_server = create_test_mcp_server_with_stored_knowledge().await;
        
        // WHEN: Using enhanced knowledge query through MCP
        let request = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "enhanced_knowledge_query",
                "arguments": {
                    "question": "What scientific achievements led to Einstein's recognition?",
                    "context_budget": 2000,
                    "include_reasoning": true
                }
            }
        });
        
        let response = mcp_server.handle_request(request).await.unwrap();
        
        // THEN: Response provides rich, contextual information
        let result = response["result"].as_object().unwrap();
        assert!(result["answer_context"].as_str().unwrap().len() > 200);
        assert!(result["supporting_facts"].as_array().unwrap().len() > 0);
        assert!(result["entity_context"].as_array().unwrap().len() > 0);
        assert!(result["confidence"].as_f64().unwrap() > 0.7);
        assert!(result["retrieval_reasoning"].is_object());
        assert!(result["context_layers_used"].as_u64().unwrap() >= 2);
    }

    #[tokio::test]
    async fn should_maintain_backward_compatibility_with_existing_tools() {
        // GIVEN: MCP server with both old and new tools
        let mcp_server = create_test_mcp_server_with_both_systems().await;
        
        // WHEN: Using traditional store_fact tool
        let fact_request = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "store_fact",
                "arguments": {
                    "subject": "Einstein",
                    "predicate": "won",
                    "object": "Nobel_Prize",
                    "confidence": 0.95
                }
            }
        });
        
        let fact_response = mcp_server.handle_request(fact_request).await.unwrap();
        assert!(fact_response["result"]["success"].as_bool().unwrap());
        
        // AND WHEN: Using traditional find_facts tool
        let find_request = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "find_facts",
                "arguments": {
                    "query": {
                        "subject": "Einstein"
                    },
                    "limit": 10
                }
            }
        });
        
        let find_response = mcp_server.handle_request(find_request).await.unwrap();
        
        // THEN: Traditional tools work alongside enhanced system
        let facts = find_response["result"]["facts"].as_array().unwrap();
        assert!(facts.len() > 0);
        
        // Should find both traditionally stored facts and enhanced knowledge
        assert!(facts.iter().any(|fact| 
            fact["subject"].as_str().unwrap() == "Einstein" &&
            fact["object"].as_str().unwrap().contains("Nobel")
        ));
    }
}
```

#### Day 35: Error Handling and Edge Cases

**Integration Tests for Error Scenarios**

```rust
// tests/integration/error_handling_integration.rs

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn should_handle_model_loading_failures_gracefully() {
        // GIVEN: System configured with unavailable model
        let config = EnhancedKnowledgeConfig {
            primary_model: ModelType::UnavailableModel,
            fallback_enabled: true,
            fallback_model: ModelType::SimpleExtractor,
        };
        
        let system = EnhancedKnowledgeSystem::with_config(config).await;
        
        // WHEN: Attempting to store knowledge with failed model loading
        let result = system.store_knowledge_enhanced(
            "Test content for model failure scenario",
            "Test Title",
            "test"
        ).await;
        
        // THEN: System gracefully falls back to simpler processing
        match result {
            Ok(store_result) => {
                // Fallback succeeded - verify reduced capability
                assert!(store_result.processing_model.contains("Simple"));
                assert!(store_result.entities_extracted > 0); // Still extracts entities
                assert!(store_result.context_preservation_score > 0.4); // Reduced but functional
            },
            Err(error) => {
                // If fallback also fails, error should be informative
                assert!(error.to_string().contains("model"));
                assert!(error.to_string().contains("fallback"));
            }
        }
    }

    #[tokio::test]
    async fn should_handle_malformed_content_robustly() {
        // GIVEN: System with various malformed inputs
        let system = EnhancedKnowledgeSystem::new().await.unwrap();
        
        let malformed_inputs = vec![
            ("", "Empty content"),
            ("a", "Single character"),
            ("a".repeat(100000), "Extremely long content"),
            ("🚀🔬🧬🌟💡🎭🎨", "Only emojis"),
            ("\n\n\n\t\t\t   ", "Only whitespace"),
            ("Ṽëṝÿ śṗëċïäḷ ċḧäṝäċẗëṝṡ", "Special unicode characters"),
        ];
        
        for (content, description) in malformed_inputs {
            // WHEN: Processing malformed content
            let result = system.store_knowledge_enhanced(content, description, "test").await;
            
            // THEN: System handles gracefully without crashing
            match result {
                Ok(store_result) => {
                    // Some content was processed
                    assert!(store_result.layers_created >= 1);
                },
                Err(error) => {
                    // Error is informative and specific
                    assert!(!error.to_string().is_empty());
                    assert!(!error.to_string().contains("panic"));
                }
            }
        }
    }

    #[tokio::test]
    async fn should_handle_resource_exhaustion_scenarios() {
        // GIVEN: System with limited resources
        let limited_config = EnhancedKnowledgeConfig {
            max_memory_usage: 100_000_000, // 100MB limit
            max_processing_time: Duration::from_secs(5),
            max_concurrent_operations: 2,
        };
        
        let system = EnhancedKnowledgeSystem::with_config(limited_config).await.unwrap();
        
        // WHEN: Overwhelming system with concurrent requests
        let mut handles = vec![];
        for i in 0..20 {
            let system_clone = system.clone();
            let handle = tokio::spawn(async move {
                let content = format!("Large document content {}: {}", i, "text ".repeat(1000));
                system_clone.store_knowledge_enhanced(&content, &format!("Doc {}", i), "test").await
            });
            handles.push(handle);
        }
        
        let results = futures::future::join_all(handles).await;
        
        // THEN: System handles resource limits gracefully
        let mut successes = 0;
        let mut resource_errors = 0;
        let mut timeouts = 0;
        
        for result in results {
            match result {
                Ok(Ok(_)) => successes += 1,
                Ok(Err(error)) => {
                    let error_str = error.to_string();
                    if error_str.contains("memory") || error_str.contains("resource") {
                        resource_errors += 1;
                    } else if error_str.contains("timeout") {
                        timeouts += 1;
                    }
                },
                Err(_) => {} // Task panicked - this shouldn't happen
            }
        }
        
        // Some operations should succeed
        assert!(successes > 0);
        // Resource limits should be enforced
        assert!(resource_errors + timeouts > 0);
        // System should remain stable
        assert!(system.health_check().await.is_ok());
    }
}
```

### Week 8: Performance Tuning and Final Integration

#### Day 36-37: Performance Optimization

**Performance Benchmark Tests**

```rust
// tests/integration/performance_benchmarks.rs

#[cfg(test)]
mod performance_benchmarks {
    use super::*;
    use criterion::*;

    #[tokio::test]
    async fn benchmark_knowledge_storage_performance() {
        let system = EnhancedKnowledgeSystem::new().await.unwrap();
        
        // Benchmark different document sizes
        let test_cases = vec![
            (100, "Small document"),
            (1000, "Medium document"),
            (5000, "Large document"),
        ];
        
        for (word_count, description) in test_cases {
            let content = generate_test_content(word_count);
            
            let start = Instant::now();
            let result = system.store_knowledge_enhanced(&content, description, "benchmark").await.unwrap();
            let duration = start.elapsed();
            
            println!("{}: {} words processed in {:?}", description, word_count, duration);
            println!("  - Layers created: {}", result.layers_created);
            println!("  - Entities extracted: {}", result.entities_extracted);
            println!("  - Relationships mapped: {}", result.relationships_mapped);
            println!("  - Context preservation: {:.2}", result.context_preservation_score);
            
            // Performance assertions
            match word_count {
                100 => assert!(duration < Duration::from_millis(500)),
                1000 => assert!(duration < Duration::from_secs(2)),
                5000 => assert!(duration < Duration::from_secs(8)),
                _ => {}
            }
        }
    }

    #[tokio::test]
    async fn benchmark_retrieval_performance() {
        let system = setup_benchmark_knowledge_base(1000).await; // 1000 documents
        
        let query_types = vec![
            ("Simple entity query", "Who is Einstein?"),
            ("Relationship query", "What did Einstein discover?"),
            ("Complex multi-hop", "How did Einstein's theories influence modern quantum physics?"),
            ("Cross-document synthesis", "Compare Einstein's and Newton's contributions to physics"),
        ];
        
        for (query_type, query) in query_types {
            let start = Instant::now();
            let result = system.query_knowledge_enhanced(query, 2000).await.unwrap();
            let duration = start.elapsed();
            
            println!("{}: {:?}", query_type, duration);
            println!("  - Confidence: {:.2}", result.confidence);
            println!("  - Context layers used: {}", result.context_layers_used);
            println!("  - Answer length: {} chars", result.answer_context.len());
            
            // Performance requirements based on query complexity
            match query_type {
                "Simple entity query" => assert!(duration < Duration::from_millis(50)),
                "Relationship query" => assert!(duration < Duration::from_millis(100)),
                "Complex multi-hop" => assert!(duration < Duration::from_millis(200)),
                "Cross-document synthesis" => assert!(duration < Duration::from_millis(300)),
                _ => {}
            }
            
            // Quality requirements
            assert!(result.confidence > 0.6);
            assert!(result.answer_context.len() > 50);
        }
    }

    #[tokio::test]
    async fn benchmark_memory_efficiency() {
        let initial_memory = get_process_memory_usage();
        
        let system = EnhancedKnowledgeSystem::new().await.unwrap();
        
        // Store progressively more documents and track memory
        for batch in 0..10 {
            for i in 0..100 {
                let doc_id = batch * 100 + i;
                let content = generate_realistic_document_content(doc_id);
                system.store_knowledge_enhanced(&content, &format!("Doc {}", doc_id), "memory_test").await.unwrap();
            }
            
            let current_memory = get_process_memory_usage();
            let memory_per_doc = (current_memory - initial_memory) / ((batch + 1) * 100) as u64;
            
            println!("After {} documents: {} MB total, {} KB per document", 
                (batch + 1) * 100,
                (current_memory - initial_memory) / 1_000_000,
                memory_per_doc / 1000
            );
            
            // Memory efficiency requirement: <50KB per document average
            assert!(memory_per_doc < 50_000, 
                "Memory per document {} exceeds 50KB limit", memory_per_doc);
        }
    }
}
```

#### Day 38-39: Integration Testing and Validation

**Comprehensive System Validation**

```rust
// tests/integration/system_validation.rs

#[cfg(test)]
mod system_validation_tests {
    use super::*;

    #[tokio::test]
    async fn validate_complete_system_against_requirements() {
        // GIVEN: Fully configured enhanced knowledge system
        let system = EnhancedKnowledgeSystem::new().await.unwrap();
        
        // Load comprehensive test dataset
        let test_documents = load_comprehensive_test_dataset().await;
        
        // WHEN: Processing entire dataset
        let mut processing_results = vec![];
        for document in test_documents {
            let result = system.store_knowledge_enhanced(
                &document.content, 
                &document.title, 
                &document.category
            ).await.unwrap();
            processing_results.push(result);
        }
        
        // THEN: Validate against success metrics
        
        // Technical Metrics Validation
        let avg_context_preservation: f32 = processing_results.iter()
            .map(|r| r.context_preservation_score)
            .sum::<f32>() / processing_results.len() as f32;
        assert!(avg_context_preservation >= 0.85, 
            "Context preservation {} below target 85%", avg_context_preservation);
        
        let avg_entities_per_doc: f32 = processing_results.iter()
            .map(|r| r.entities_extracted)
            .sum::<u32>() as f32 / processing_results.len() as f32;
        assert!(avg_entities_per_doc >= 5.0,
            "Entity extraction {} below reasonable threshold", avg_entities_per_doc);
        
        let avg_relationships_per_doc: f32 = processing_results.iter()
            .map(|r| r.relationships_mapped)
            .sum::<u32>() as f32 / processing_results.len() as f32;
        assert!(avg_relationships_per_doc >= 3.0,
            "Relationship extraction {} below reasonable threshold", avg_relationships_per_doc);
        
        // Query Performance Validation
        let test_queries = generate_comprehensive_test_queries();
        let mut query_results = vec![];
        
        for query in test_queries {
            let start = Instant::now();
            let result = system.query_knowledge_enhanced(&query.text, 2000).await.unwrap();
            let duration = start.elapsed();
            
            query_results.push((query, result, duration));
        }
        
        // Validate retrieval latency requirement: <200ms for 95% of queries
        let mut latencies: Vec<_> = query_results.iter().map(|(_, _, duration)| *duration).collect();
        latencies.sort();
        let p95_latency = latencies[(latencies.len() * 95) / 100];
        assert!(p95_latency < Duration::from_millis(200),
            "95th percentile latency {:?} exceeds 200ms requirement", p95_latency);
        
        // Validate multi-hop query success rate: >70%
        let multi_hop_queries: Vec<_> = query_results.iter()
            .filter(|(query, _, _)| query.complexity == QueryComplexity::MultiHop)
            .collect();
        let successful_multi_hop = multi_hop_queries.iter()
            .filter(|(_, result, _)| result.confidence >= 0.7 && result.context_layers_used >= 2)
            .count();
        let multi_hop_success_rate = successful_multi_hop as f32 / multi_hop_queries.len() as f32;
        assert!(multi_hop_success_rate >= 0.7,
            "Multi-hop query success rate {} below 70% target", multi_hop_success_rate);
    }

    #[tokio::test]
    async fn validate_system_stability_under_extended_operation() {
        // GIVEN: System running extended operation simulation
        let system = EnhancedKnowledgeSystem::new().await.unwrap();
        
        let start_time = Instant::now();
        let mut operation_count = 0;
        let mut error_count = 0;
        
        // RUN: 2-hour continuous operation simulation
        while start_time.elapsed() < Duration::from_secs(60) { // Reduced for testing
            // Alternate between storage and retrieval operations
            if operation_count % 2 == 0 {
                // Storage operation
                let content = generate_random_realistic_content();
                let result = system.store_knowledge_enhanced(&content, "Test Doc", "test").await;
                if result.is_err() {
                    error_count += 1;
                }
            } else {
                // Retrieval operation
                let query = generate_random_realistic_query();
                let result = system.query_knowledge_enhanced(&query, 1000).await;
                if result.is_err() {
                    error_count += 1;
                }
            }
            
            operation_count += 1;
            
            // Check system health every 100 operations
            if operation_count % 100 == 0 {
                let health = system.health_check().await.unwrap();
                assert!(health.memory_usage_ratio < 0.9); // <90% memory usage
                assert!(health.average_response_time < Duration::from_millis(500));
                assert!(health.error_rate < 0.05); // <5% error rate
            }
        }
        
        // THEN: System maintains stability
        let overall_error_rate = error_count as f32 / operation_count as f32;
        assert!(overall_error_rate < 0.02, 
            "Error rate {} exceeds 2% stability requirement", overall_error_rate);
        
        let final_health = system.health_check().await.unwrap();
        assert!(final_health.is_healthy());
        
        println!("Stability test completed: {} operations, {:.2}% error rate", 
            operation_count, overall_error_rate * 100.0);
    }

    #[tokio::test]
    async fn validate_backward_compatibility_with_existing_system() {
        // GIVEN: System with both old and new interfaces
        let enhanced_system = EnhancedKnowledgeSystem::new().await.unwrap();
        let legacy_interface = enhanced_system.get_legacy_interface();
        
        // WHEN: Using legacy interface for storage
        legacy_interface.store_triple(Triple::new(
            "Einstein".to_string(),
            "is".to_string(),
            "physicist".to_string()
        ).unwrap()).await.unwrap();
        
        // AND: Using enhanced interface for storage
        enhanced_system.store_knowledge_enhanced(
            "Einstein developed the theory of relativity",
            "Einstein Theory",
            "physics"
        ).await.unwrap();
        
        // THEN: Both types of data are queryable through both interfaces
        
        // Legacy query finding enhanced data
        let legacy_query_result = legacy_interface.query_triples(TripleQuery {
            subject: Some("Einstein".to_string()),
            predicate: None,
            object: None,
            limit: 10,
            min_confidence: 0.0,
            include_chunks: true,
        }).await.unwrap();
        
        assert!(legacy_query_result.triples.len() >= 2); // Original triple + extracted from enhanced
        
        // Enhanced query finding legacy data
        let enhanced_query_result = enhanced_system.query_knowledge_enhanced(
            "What is Einstein?", 
            1000
        ).await.unwrap();
        
        assert!(enhanced_query_result.answer_context.contains("physicist"));
        assert!(enhanced_query_result.answer_context.contains("relativity"));
    }
}
```

#### Day 40: Final Testing and Documentation

**Production Readiness Tests**

```rust
// tests/integration/production_readiness.rs

#[cfg(test)]
mod production_readiness_tests {
    use super::*;

    #[tokio::test]
    async fn validate_production_deployment_readiness() {
        // GIVEN: Production configuration
        let prod_config = EnhancedKnowledgeConfig {
            environment: Environment::Production,
            model_config: ModelConfig {
                primary_model: ModelType::SmolLM2_360M,
                fallback_model: Some(ModelType::MiniLM_L12),
                max_memory_usage: 2_000_000_000, // 2GB
                model_timeout: Duration::from_secs(30),
            },
            storage_config: StorageConfig {
                persistence_enabled: true,
                backup_enabled: true,
                compression_enabled: true,
                encryption_enabled: true,
            },
            monitoring_config: MonitoringConfig {
                metrics_enabled: true,
                logging_level: LogLevel::Info,
                performance_tracking: true,
            },
        };
        
        let system = EnhancedKnowledgeSystem::with_config(prod_config).await.unwrap();
        
        // WHEN: Running production readiness checks
        let readiness_check = system.production_readiness_check().await.unwrap();
        
        // THEN: All production requirements are met
        assert!(readiness_check.models_loaded);
        assert!(readiness_check.storage_accessible);
        assert!(readiness_check.monitoring_active);
        assert!(readiness_check.backup_configured);
        assert!(readiness_check.security_enabled);
        assert!(readiness_check.performance_acceptable);
        
        // Validate specific metrics
        assert!(readiness_check.average_storage_latency < Duration::from_millis(100));
        assert!(readiness_check.average_query_latency < Duration::from_millis(200));
        assert!(readiness_check.memory_usage_ratio < 0.8);
        assert!(readiness_check.error_rate < 0.01);
    }

    #[tokio::test]
    async fn validate_graceful_degradation_capabilities() {
        // GIVEN: System with degradation policies
        let system = EnhancedKnowledgeSystem::with_degradation_policies().await.unwrap();
        
        // WHEN: Simulating various failure scenarios
        let scenarios = vec![
            DegradationScenario::ModelUnavailable,
            DegradationScenario::HighMemoryUsage,
            DegradationScenario::SlowResponseTimes,
            DegradationScenario::StorageLatency,
        ];
        
        for scenario in scenarios {
            system.simulate_degradation_scenario(scenario.clone()).await;
            
            // Verify system still functions with reduced capability
            let result = system.store_knowledge_enhanced(
                "Test content during degradation",
                "Degradation Test",
                "test"
            ).await;
            
            match scenario {
                DegradationScenario::ModelUnavailable => {
                    // Should fall back to simpler extraction
                    assert!(result.is_ok());
                    let result = result.unwrap();
                    assert!(result.processing_model.contains("fallback"));
                    assert!(result.context_preservation_score > 0.3); // Reduced but functional
                },
                DegradationScenario::HighMemoryUsage => {
                    // Should reduce complexity or queue operations
                    assert!(result.is_ok() || result.is_err()); // May reject or process
                    if let Ok(result) = result {
                        assert!(result.layers_created <= 3); // Simplified processing
                    }
                },
                _ => {
                    // Other scenarios should still allow basic functionality
                    assert!(result.is_ok());
                }
            }
            
            system.clear_degradation_scenario().await;
        }
    }

    #[tokio::test]
    async fn validate_monitoring_and_observability() {
        // GIVEN: System with full monitoring enabled
        let system = EnhancedKnowledgeSystem::with_monitoring().await.unwrap();
        
        // WHEN: Performing various operations
        for i in 0..50 {
            let content = format!("Monitoring test document {}", i);
            system.store_knowledge_enhanced(&content, &format!("Doc {}", i), "monitoring").await.unwrap();
            
            if i % 10 == 0 {
                let query = format!("What is in document {}?", i);
                system.query_knowledge_enhanced(&query, 1000).await.unwrap();
            }
        }
        
        // THEN: Monitoring data is captured accurately
        let metrics = system.get_monitoring_metrics().await.unwrap();
        
        assert_eq!(metrics.total_storage_operations, 50);
        assert_eq!(metrics.total_query_operations, 5);
        assert!(metrics.average_storage_latency > Duration::from_millis(0));
        assert!(metrics.average_query_latency > Duration::from_millis(0));
        assert!(metrics.memory_usage_over_time.len() > 0);
        assert!(metrics.error_count == 0);
        
        // Validate alerting thresholds
        let alerts = system.get_active_alerts().await.unwrap();
        for alert in alerts {
            match alert.severity {
                AlertSeverity::Critical => {
                    // Critical alerts should be addressed
                    panic!("Critical alert active: {}", alert.message);
                },
                AlertSeverity::Warning => {
                    // Warnings logged but acceptable for test
                    println!("Warning alert: {}", alert.message);
                },
                _ => {}
            }
        }
    }
}
```

## Test Execution Strategy

### Test Execution Order

1. **Unit Tests First** (London School TDD principle)
   - Run all unit tests in isolation with mocks
   - Ensure 100% unit test coverage before integration
   - Fix any failing unit tests immediately

2. **Integration Tests Second**
   - Run component integration tests
   - Test real component interactions
   - Validate integration contracts

3. **Acceptance Tests Last**
   - Run end-to-end acceptance tests
   - Validate complete user scenarios
   - Performance and load testing

### Continuous Integration Pipeline

```rust
// .github/workflows/tdd_pipeline.yml equivalent in Rust

#[cfg(test)]
mod ci_pipeline_tests {
    use super::*;

    #[tokio::test]
    async fn ci_unit_tests_phase() {
        // Run all unit tests in parallel
        let unit_test_results = run_all_unit_tests().await;
        assert!(unit_test_results.all_passed());
        assert!(unit_test_results.coverage_percentage >= 90.0);
    }

    #[tokio::test]
    async fn ci_integration_tests_phase() {
        // Run integration tests after unit tests pass
        let integration_results = run_integration_tests().await;
        assert!(integration_results.all_passed());
    }

    #[tokio::test]
    async fn ci_acceptance_tests_phase() {
        // Run acceptance tests last
        let acceptance_results = run_acceptance_tests().await;
        assert!(acceptance_results.all_passed());
        assert!(acceptance_results.performance_metrics_met());
    }
}
```

### Test Quality Gates

Each phase must pass these gates before proceeding:

1. **Unit Test Gate**
   - 100% unit tests passing
   - >90% code coverage
   - No critical bugs detected
   - All mocks properly isolated

2. **Integration Test Gate**
   - All integration tests passing
   - Component contracts validated
   - Cross-component communication working
   - Resource management functioning

3. **Acceptance Test Gate**
   - All acceptance scenarios passing
   - Performance requirements met
   - Memory usage within limits
   - Error handling robust

## Implementation Timeline with TDD

### Detailed Week-by-Week Schedule

**Week 1-2: Model Infrastructure + TDD Setup**
- Day 1: Write acceptance tests for model loading
- Day 2-3: Write unit tests, implement ModelRegistry, ModelLoader
- Day 4-5: Write integration tests, implement ModelPool, ResourceManager
- Day 6-7: Write acceptance tests for SmolLM integration
- Day 8-10: Write unit tests for processing pipeline, implement with mocks

**Week 3-4: Storage System + TDD**
- Day 11-12: Write acceptance tests for hierarchical storage
- Day 13-15: Write unit tests for knowledge layers, implement with test doubles
- Day 16-17: Write integration tests for storage backend
- Day 18-20: Write unit tests for migration system, implement with mocks

**Week 5-6: Retrieval System + TDD**
- Day 21-22: Write acceptance tests for enhanced retrieval
- Day 23-25: Write unit tests for graph traversal, context assembly
- Day 26-27: Write integration tests for query processing
- Day 28-30: Write unit tests for caching, response optimization

**Week 7-8: Full Integration + TDD**
- Day 31-33: Write acceptance tests for complete system
- Day 34-35: Write integration tests for MCP tool integration
- Day 36-37: Write performance tests, optimize
- Day 38-40: Write production readiness tests, final validation

### Success Criteria for Each Phase

**Phase 1 Success**: 
- All model management unit tests passing (100%)
- Model loading integration tests passing (100%)
- Acceptance test for basic model operations passing
- Resource management within defined limits

**Phase 2 Success**:
- All storage unit tests passing (100%)
- Storage integration tests passing (100%)
- Hierarchical knowledge acceptance tests passing
- Data migration tests passing

**Phase 3 Success**:
- All retrieval unit tests passing (100%)
- Retrieval integration tests passing (100%)
- Enhanced query acceptance tests passing
- Performance benchmarks met

**Phase 4 Success**:
- All system integration tests passing (100%)
- Production readiness tests passing (100%)
- Full system acceptance tests passing
- Backward compatibility maintained

## Conclusion

This enhanced knowledge storage system addresses the core RAG problems of context fragmentation, semantic coherence loss, and poor retrieval quality while maintaining the system's anti-bloat philosophy. By leveraging small language models for complex processing and delivering optimized responses to the main LLM, we achieve the best of both worlds: sophisticated knowledge understanding with efficient interface design.

The comprehensive TDD approach ensures:
- **High Code Quality**: London School TDD with extensive mocking and behavior verification
- **Robust Error Handling**: Edge cases and failure scenarios thoroughly tested
- **Performance Assurance**: Benchmarking and load testing integrated throughout
- **Production Readiness**: Monitoring, observability, and degradation testing
- **Maintainable Codebase**: Clear test structure and comprehensive coverage

The system transforms LLMKG from a simple triple store into a sophisticated knowledge graph with hierarchical organization, semantic linking, and context-aware retrieval - positioning it as a state-of-the-art solution for LLM-powered knowledge management.