# LLMKG - Lightning-Fast Knowledge Graph for LLM Integration

## Code Organization Rules

### File Size Limitation
**CRITICAL RULE**: No file should ever be created with more than 500 lines of code. If any file is found to be over 500 lines of code, it MUST be modularized immediately. The only acceptable exception is documentation files (*.md files) which may exceed this limit.

This rule ensures:
- Maintainable codebase
- Reduced data bloat
- Optimal performance for world's fastest knowledge graph
- Easy navigation and understanding

## System Overview

LLMKG is the world's fastest knowledge graph system designed specifically as an AI memory system. It implements a sophisticated brain-inspired approach to knowledge representation and processing, combining traditional knowledge graph concepts with neural network principles to create a highly efficient, biologically-inspired cognitive architecture. The system is also designed to limit databloat as to not overwhelm context windows.

### Core Purpose
The system serves as an intelligent memory system for AI models, designed to:
- Reduce data bloat through efficient knowledge representation
- Provide lightning-fast knowledge retrieval and processing
- Enable sophisticated cognitive reasoning patterns
- Function as a Model Context Protocol (MCP) server

## Architecture Components

### 1. Brain-Enhanced Graph (`core/brain_enhanced_graph.rs`)
- Brain-inspired entities with activation states and temporal decay
- Biological learning mechanisms (Hebbian learning, synaptic plasticity)
- High-performance optimization with SIMD, compression, and memory mapping

### 2. Cognitive Algorithms (`src/cognitive/`)
The system implements 7 distinct cognitive patterns:

#### Core Cognitive Patterns:
- **Convergent Thinking**: Finds specific answers through activation propagation
- **Divergent Thinking**: Explores multiple creative possibilities
- **Lateral Thinking**: Finds unexpected connections between disparate concepts
- **Systems Thinking**: Analyzes hierarchical relationships and inheritance
- **Critical Thinking**: Resolves contradictions and validates facts
- **Abstract Thinking**: Detects patterns and identifies abstraction opportunities
- **Adaptive Thinking**: Dynamically selects optimal cognitive strategies

#### Advanced Systems (Phase 3-4):
- **Working Memory System**: Multi-buffer architecture with capacity constraints
- **Attention Manager**: 5 attention types with executive control
- **Competitive Inhibition**: Multiple competition types for conflict resolution
- **Learning Systems**: Hebbian learning, synaptic homeostasis, graph optimization

### 3. Phase 4 Learning Systems (`src/learning/`)
- **Hebbian Learning Engine**: Synaptic strengthening based on coactivation
- **Synaptic Homeostasis**: Maintains network stability
- **Adaptive Learning System**: Meta-learning and parameter tuning

## MCP Server Integration - Hybrid Cognitive Tool Architecture

The system functions as a comprehensive MCP (Model Context Protocol) server that provides AI models with access to sophisticated cognitive tools through a **3-tier hybrid approach**:

### Tier 1: Individual Cognitive Pattern Tools (7 tools)
Each cognitive pattern is exposed as a dedicated MCP tool for precise control:

1. **convergent_thinking**: Fast, focused answers through activation propagation (~100ms)
   - Parameters: query, context, activation_threshold (0.5), max_depth (5), beam_width (3)
   - Use cases: Factual queries, direct relationship lookup, simple Q&A

2. **divergent_thinking**: Creative exploration of multiple possibilities (~200ms)
   - Parameters: seed_concept, exploration_type, breadth (20), creativity_threshold (0.3), max_depth (4)
   - Use cases: Brainstorming, finding examples/types, creative generation

3. **lateral_thinking**: Unexpected connections between disparate concepts (~400ms)
   - Parameters: concept_a, concept_b, novelty_threshold (0.4), max_bridge_length (6), creativity_boost (1.5)
   - Use cases: Cross-domain thinking, innovation, "How is X related to Y?"

4. **systems_thinking**: Hierarchical analysis and attribute inheritance (~300ms)
   - Parameters: query, reasoning_type, system_boundaries, inheritance_depth
   - Use cases: System analysis, hierarchy understanding, inheritance patterns

5. **critical_thinking**: Fact validation and contradiction resolution (~500ms)
   - Parameters: query, validation_level, evidence_requirements, bias_detection
   - Use cases: Fact-checking, argument analysis, evidence evaluation

6. **abstract_thinking**: Pattern detection and abstraction opportunities (~350ms)
   - Parameters: analysis_scope, pattern_type, abstraction_level
   - Use cases: Pattern analysis, conceptual understanding, abstraction

7. **adaptive_thinking**: Meta-reasoning with automatic pattern selection (~3000ms)
   - Parameters: query_characteristics, available_patterns, performance_history
   - Use cases: Complex queries, automatic optimization, multi-perspective analysis

### Tier 2: Orchestrated Reasoning Tool (1 tool)
8. **intelligent_reasoning**: Automatic pattern selection and ensemble reasoning
   - Automatically selects optimal cognitive pattern(s) based on query analysis
   - Combines multiple patterns for comprehensive answers
   - Includes confidence-weighted result merging and conflict resolution
   - Meta-learning improves strategy selection over time

### Tier 3: Specialized Composite Tools (4 tools)
Pre-configured combinations of patterns optimized for specific use cases:

9. **creative_brainstorm**: Divergent + Lateral + Abstract patterns
   - Optimized for creative ideation and innovation
   - Parallel execution for maximum creative output

10. **fact_checker**: Critical + Convergent + Systems patterns
    - Comprehensive fact validation with supporting evidence
    - Cross-references multiple sources and reasoning paths

11. **problem_solver**: Adaptive + Systems + Critical patterns
    - Multi-faceted problem analysis and solution generation
    - Validates solutions through critical evaluation

12. **pattern_analyzer**: Abstract + Systems + Convergent patterns
    - Deep pattern recognition across hierarchical structures
    - Identifies recurring themes and structural relationships

### Hybrid Approach Benefits:

#### Performance Optimization:
- **Granular Control**: AI can select fast tools (100ms) for simple queries
- **Automatic Escalation**: Complex queries automatically use ensemble methods
- **Parallel Execution**: Multiple patterns can run simultaneously
- **Caching**: Frequent patterns cached to reduce execution time

#### Flexibility and Transparency:
- **Precise Control**: AI chooses exactly the right cognitive strategy
- **Transparent Reasoning**: Clear reasoning traces from individual patterns
- **Automatic Optimization**: Intelligent tool handles complex orchestration
- **Learning Integration**: System improves through usage patterns

#### Resource Efficiency:
- **Selective Execution**: Only run patterns needed for specific queries
- **Early Stopping**: Ensemble methods stop when confidence threshold met
- **Result Streaming**: Long-running patterns stream intermediate results
- **Memory Management**: Working memory integration prevents resource bloat

### MCP Tool Parameters:
All tools support standardized parameters through `PatternParameters`:
```json
{
  "max_depth": "optional int (5)",
  "activation_threshold": "optional float (0.5)",
  "exploration_breadth": "optional int (10)",
  "creativity_threshold": "optional float (0.3)",
  "validation_level": "optional enum (Basic|Standard|Rigorous)"
}
```

### MCP Features:
- **Structured Knowledge Retrieval**: Reduces LLM hallucinations through validated responses
- **Context-Aware Responses**: Metadata includes confidence scores and reasoning traces
- **Performance Monitoring**: Real-time execution metrics and cache statistics
- **Error Handling**: Detailed error messages with recovery suggestions
- **Adaptive Learning**: System learns from usage patterns to improve tool selection

## Testing Strategy

### Dual Testing Approach for Hybrid Architecture
The system requires TWO types of tests for every cognitive algorithm and MCP tool across all three tiers:

#### 1. Mock LLM Tests
- Test cognitive algorithms with synthetic data
- Verify algorithm correctness without external dependencies
- Fast execution for development workflow
- Located in `tests/` directory with comprehensive coverage

#### 2. Real LLM Integration Tests
- Test WITH actual LLM integration (DeepSeek API)
- Validate real-world performance and behavior
- API configuration in `.env` file
- Continuous learning loops with LLM evaluation

### Hybrid Tool Testing Structure:

#### Tier 1 Testing (Individual Patterns):
- **Unit Tests**: Each cognitive pattern tested independently
- **Performance Tests**: Execution time benchmarks (100ms-3000ms targets)
- **Parameter Tests**: Validation of all pattern-specific parameters
- **Error Handling**: Robustness testing with invalid inputs

#### Tier 2 Testing (Orchestrated Reasoning):
- **Integration Tests**: Pattern selection and ensemble coordination
- **Meta-Learning Tests**: Strategy improvement over time
- **Confidence Tests**: Result merging and conflict resolution
- **Adaptive Tests**: Performance optimization and learning

#### Tier 3 Testing (Composite Tools):
- **Combination Tests**: Multi-pattern coordination and results
- **Use Case Tests**: Real-world scenario validation
- **Performance Tests**: Parallel execution and optimization
- **Consistency Tests**: Reproducible results across runs

### Test Structure:
- **Phase 1**: Basic knowledge graph functionality
- **Phase 2**: Individual cognitive pattern testing
- **Phase 3**: Orchestrated reasoning and composite tool validation
- **Phase 4**: Self-organization, learning, and hybrid tool optimization

### Testing Requirements:
- **Performance Benchmarks**: Each tool must meet execution time targets
- **Accuracy Thresholds**: Measurable confidence and correctness metrics
- **Stress Testing**: 5000+ activation events across all tool tiers
- **Integration Scenarios**: Real-world usage patterns and tool combinations
- **Memory Management**: Working memory integration and resource efficiency
- **Cache Validation**: Result caching and performance optimization

## Development Conventions

### Architecture Patterns:
- **Async/await throughout** with tokio runtime
- **Arc<RwLock<>>** for thread-safe shared state
- **Result<T>** error handling with custom error types
- **Trait-based design** for cognitive patterns and extensibility

### Performance Optimizations:
- **SIMD operations** for vector computations
- **Memory-mapped storage** for large graphs
- **Bloom filters** for fast existence checks
- **Compression** with zstd for storage efficiency
- **Concurrent processing** with rayon and crossbeam

### Data Structures:
- **SlotMap** for efficient entity storage
- **Sparse representations** for memory efficiency
- **Temporal tracking** with SystemTime
- **Hierarchical caching** for performance

## Configuration

### Environment Variables (.env):
- DeepSeek API configuration for LLM integration
- Test environment settings
- Performance tuning parameters

### Feature Flags:
- WASM compatibility for web deployment
- Native optimizations
- SIMD acceleration
- CUDA support (future)

## Development Goals

### Primary Objectives:
1. **World's Fastest Knowledge Graph**: Optimize for speed and efficiency
2. **AI Memory System**: Serve as intelligent memory for AI models
3. **Data Bloat Reduction**: Maintain minimal resource footprint
4. **MCP Server Excellence**: Provide superior tools for LLM integration

### Quality Standards:
- All files must be under 500 lines (except documentation)
- Comprehensive testing with both mocked and real LLM integration
- Performance benchmarks must be maintained
- Code must be modular and maintainable

## Current Status

The system is in **Phase 4** development with hybrid MCP tool architecture:

### Completed:
- ✅ Complete Phase 1-3 implementation
- ✅ All 7 cognitive patterns working independently
- ✅ Advanced reasoning systems integrated (working memory, attention, inhibition)
- ✅ Orchestration system with pattern selection and ensemble methods
- ✅ Basic MCP server foundation

### In Active Development:
- 🔄 **Tier 1 Tools**: Individual cognitive pattern MCP tool implementation
- 🔄 **Tier 2 Tools**: Intelligent reasoning orchestrator tool
- 🔄 **Tier 3 Tools**: Specialized composite tool combinations
- 🔄 Phase 4 learning systems integration with MCP tools
- 🔄 Performance optimization and caching for hybrid architecture

### Planned:
- 🔜 **Comprehensive Testing**: Mock and real LLM tests for all tool tiers
- 🔜 **Parameter Standardization**: Unified parameter interface across tools
- 🔜 **Result Streaming**: Long-running pattern results streaming
- 🔜 **Tool Analytics**: Usage pattern analysis and optimization
- 🔜 **Production Deployment**: Scalable MCP server with hybrid tools

## Contributing Guidelines

1. **File Size**: Never exceed 500 lines per file
2. **Testing**: Implement both mock and real LLM tests for all three tool tiers
3. **Performance**: Maintain execution time benchmarks for each cognitive pattern
4. **Tool Design**: Follow hybrid architecture with individual, orchestrated, and composite tools
5. **Parameter Standardization**: Use unified parameter interface across all tools
6. **Documentation**: Update this file when adding new cognitive patterns or tools
7. **Modularity**: Design for easy extension and maintenance of hybrid tool system

## How the System Stores Information

### Overview of Storage Pipeline
LLMKG uses a sophisticated multi-stage neural-enhanced pipeline to transform raw text into structured, queryable knowledge. The storage process is designed to be both intelligent and automatic - LLMs simply provide text through MCP tools, and the system handles all complexity internally.

### Storage Entry Points
The system provides two primary MCP tools for storing information:

#### 1. **store_knowledge** Tool (Recommended)
```json
{
  "tool": "store_knowledge",
  "arguments": {
    "text": "Eagles are large birds of prey with excellent vision and powerful talons",
    "context": "Bird characteristics",
    "use_neural_construction": true
  }
}
```
- Handles complex text with multiple facts
- Automatically extracts entities and relationships
- Uses neural structure prediction
- Creates brain-inspired graph structure

#### 2. **store_fact** Tool (Legacy/Simple)
```json
{
  "tool": "store_fact",
  "arguments": {
    "subject": "Einstein",
    "predicate": "invented",
    "object": "relativity",
    "confidence": 0.95
  }
}
```
- Direct Subject-Predicate-Object triples
- Simple and fast for basic facts
- Limited to 128 chars for entities, 64 for predicates

### The Complete Storage Pipeline

#### Stage 1: Neural Canonicalization
When text arrives, the system first canonicalizes entities to ensure consistency and prevent duplicates:

1. **Entity Normalization**:
   - "Dr. Einstein" → "Albert Einstein"
   - "einstein" → "Albert Einstein"
   - Removes titles (Dr., Prof., Mr.)
   - Removes suffixes (Jr., Sr., III)
   - Applies title casing

2. **Context-Aware Disambiguation**:
   - "Einstein" in physics context → "Albert Einstein"
   - "Einstein" in pet context → Different entity
   - Uses neural embeddings for similarity matching

3. **Predicate Standardization**:
   - "is" → "is_a"
   - "works for" → "works_at"
   - "created by" → "created"

#### Stage 2: Neural Structure Prediction
The `GraphStructurePredictor` analyzes text to determine optimal graph structure:

1. **Text Analysis**:
   - Tokenizes input text
   - Generates embeddings
   - Identifies key concepts

2. **Operation Prediction**:
   - `CreateNode`: For entities (Input/Output directions)
   - `CreateLogicGate`: For logical relationships
   - `CreateRelationship`: For typed connections

3. **Pattern Recognition**:
   - "X is Y" → IsA relationship
   - "X has Y" → HasProperty relationship
   - "X created Y" → Created relationship
   - Complex sentences → Multiple operations

#### Stage 3: Brain-Inspired Entity Creation
Each entity is created with neural properties:

```rust
BrainInspiredEntity {
    id: EntityKey::new(),
    concept_id: "Eagles",              // Canonical form
    direction: EntityDirection::Input,  // Input/Output/Gate
    properties: HashMap::new(),         // Metadata
    embedding: vec![...],              // 384-dimensional vector
    activation_state: 0.0,             // Neural activation level
    last_activation: SystemTime::now(), // Temporal tracking
}
```

#### Stage 4: Relationship and Logic Gate Creation
The system creates sophisticated connections:

1. **Direct Relationships**:
   ```rust
   BrainInspiredRelationship {
       source: "Eagles",
       target: "birds of prey",
       relation_type: RelationType::IsA,
       weight: 1.0,               // Connection strength
       is_inhibitory: false,      // Positive relationship
       temporal_decay: 0.9,       // Decay over time
       activation_count: 0,       // Usage tracking
   }
   ```

2. **Logic Gates** (Brain-inspired computation):
   - **AND Gate**: All inputs must be true
   - **OR Gate**: Any input can be true
   - **NOT Gate**: Negation
   - **Inhibitory Gate**: Suppresses activation
   - **Weighted Gate**: Variable strength inputs

#### Stage 5: Temporal Storage
All information is stored with bi-temporal metadata for versioning:

```rust
TemporalEntity {
    entity: brain_inspired_entity,
    valid_time: TimeRange {         // When fact was true
        start: "2024-01-01",
        end: None,                  // Still valid
    },
    transaction_time: TimeRange {   // When stored in system
        start: "2024-01-15",
        end: None,
    },
    version_id: 1,
    supersedes: None,              // Previous version if updated
}
```

### Automatic Relationship Extraction

The system automatically identifies and creates relationships through:

1. **Pattern-Based Extraction**:
   - Subject-Verb-Object patterns
   - Prepositional phrases
   - Attribute assignments
   - Temporal expressions

2. **Neural Network Prediction**:
   - Trained on examples of text → graph structure
   - Learns domain-specific patterns
   - Improves with usage

3. **Hierarchical Understanding**:
   - Inheritance relationships
   - Part-whole relationships
   - Category membership

### Storage Examples

#### Example 1: Scientific Fact
**Input**: "Marie Curie discovered radium in 1898"

**Processing**:
1. Canonicalization:
   - "Marie Curie" → "Marie Curie" (already canonical)
   - "discovered" → "discovered"
   - "radium" → "Radium"

2. Structure Prediction:
   - CreateNode("Marie Curie", Input)
   - CreateNode("Radium", Output)
   - CreateRelationship("Marie Curie", "Radium", "discovered")
   - Add temporal property: year = 1898

3. Storage:
   - Entities with embeddings
   - Relationship with temporal metadata
   - Logic gate linking discovery

#### Example 2: Complex Knowledge
**Input**: "Quantum mechanics describes the behavior of matter and energy at the atomic scale"

**Processing**:
1. Entity Extraction:
   - "Quantum mechanics"
   - "matter"
   - "energy"
   - "atomic scale"

2. Relationship Creation:
   - (Quantum mechanics, describes, behavior)
   - (matter, at_scale, atomic scale)
   - (energy, at_scale, atomic scale)

3. Logic Gate:
   - AND gate linking quantum mechanics to both matter and energy

### Anti-Bloat Measures

1. **Size Limits**:
   - Entity names: ≤ 128 characters
   - Predicates: ≤ 64 characters
   - Knowledge chunks: ≤ 2048 bytes

2. **Deduplication**:
   - Neural canonicalization prevents duplicates
   - Embedding similarity matching
   - Automatic entity merging

3. **Pruning**:
   - Weak relationships decay over time
   - Unused entities can be archived
   - Automatic optimization

### How LLMs Know What to Store

1. **Clear Tool Schemas**: Each MCP tool provides detailed parameter descriptions

2. **Intelligent Defaults**: 
   - `use_neural_construction: true` by default
   - Automatic pattern detection
   - Context-aware processing

3. **Flexible Input**:
   - Accept natural language text
   - Handle structured data
   - Support confidence scores

4. **Error Recovery**:
   - Fallback to simpler storage if neural fails
   - Clear error messages
   - Suggested corrections

### Performance Characteristics

- **Entity Insertion**: < 10ms average
- **Neural Prediction**: < 50ms for structure
- **Canonicalization**: < 5ms with caching
- **Total Storage**: < 100ms for most inputs
- **Parallel Processing**: Multiple facts stored concurrently

### Storage Benefits

1. **Consistency**: All variations map to canonical forms
2. **Intelligence**: Automatic relationship extraction
3. **Queryability**: Brain-inspired structure enables sophisticated retrieval
4. **Versioning**: Full temporal history maintained
5. **Efficiency**: Minimal storage with maximum information

The storage system ensures that information is not just saved but is transformed into a rich, interconnected knowledge structure that supports the advanced cognitive reasoning capabilities of the LLMKG system.

## Implementation Priority

When implementing the hybrid MCP tool architecture, follow this priority order:

1. **Tier 1 Implementation**: Individual cognitive pattern tools (7 tools)
2. **Tier 2 Implementation**: Intelligent reasoning orchestrator (1 tool)
3. **Tier 3 Implementation**: Specialized composite tools (4 tools)
4. **Testing Implementation**: Comprehensive testing suite for all tiers
5. **Performance Optimization**: Caching, streaming, and parallel execution
6. **Analytics and Learning**: Usage pattern analysis and system optimization

This system represents the cutting edge of knowledge graph technology, specifically designed to enhance AI model capabilities through sophisticated cognitive reasoning and efficient knowledge management via a comprehensive hybrid MCP tool architecture.