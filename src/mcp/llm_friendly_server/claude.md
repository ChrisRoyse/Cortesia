# Directory Overview: LLM-Friendly MCP Server

## 1. High-Level Summary

This directory contains a sophisticated LLM-friendly Model Context Protocol (MCP) server implementation for the LLMKG knowledge graph system. The server provides a high-level, intuitive API designed specifically for Large Language Model consumption and generation. It acts as a bridge between LLMs and the underlying knowledge graph engine, offering simplified operations for storing, querying, and reasoning over knowledge.

The architecture includes advanced features like temporal tracking, database branching (git-like versioning), cognitive reasoning engines, enhanced search capabilities, and comprehensive validation systems. The server is designed to be fault-tolerant with fallback mechanisms when enhanced AI processing fails.

## 2. Tech Stack

*   **Languages:** Rust
*   **Frameworks:** Tokio (async runtime), Serde (serialization)
*   **Libraries:** 
    *   `uuid` - Unique identifier generation
    *   `chrono` - Date/time handling
    *   `regex` - Pattern matching for validation
    *   `lazy_static` - Global static variables
    *   `serde_json` - JSON manipulation
*   **Database:** Custom knowledge graph engine (in-memory with persistence)
*   **AI Integration:** Model management system (temporarily disabled)
*   **Protocols:** Model Context Protocol (MCP)

## 3. Directory Structure

### Core Files
- `mod.rs` - Main module definition and server implementation
- `types.rs` - Type definitions and data structures
- `tools.rs` - Tool definitions for MCP protocol
- `validation.rs` - Knowledge validation logic

### Processing Modules
- `reasoning_engine.rs` - Pure algorithmic reasoning (deductive, inductive, abductive, analogical)
- `query_generation.rs` - Query generation utilities
- `query_generation_enhanced.rs` - Enhanced query generation with advanced entity extraction
- `query_generation_native.rs` - Native query generation
- `search_fusion.rs` - Search result fusion algorithms
- `utils.rs` - Utility functions and statistics management

### Advanced Features
- `database_branching.rs` - Git-like branching for knowledge databases
- `temporal_tracking.rs` - Time-based change tracking
- `divergent_graph_traversal.rs` - Creative exploration algorithms
- `migration.rs` - Tool migration and deprecation handling

### Handlers Directory
Contains specialized request handlers organized by functionality:
- `storage.rs` - Knowledge storage operations
- `query.rs` - Knowledge querying operations
- `advanced.rs` - Advanced search and validation
- `cognitive.rs` - Cognitive reasoning handlers
- `stats.rs` - Statistics and metrics
- `enhanced_search.rs` - Enhanced search capabilities
- `graph_analysis.rs` - Graph analysis tools
- `temporal.rs` - Temporal operations and branching
- `exploration.rs` - Knowledge exploration tools

## 4. Core Architecture

### Main Server: `LLMFriendlyMCPServer`

**Purpose:** Central server that orchestrates all MCP operations for LLM interaction.

**Key Components:**
- `knowledge_engine: Arc<RwLock<KnowledgeEngine>>` - Core knowledge graph storage
- `usage_stats: Arc<RwLock<UsageStats>>` - Performance and usage tracking
- `version_manager: Arc<MultiDatabaseVersionManager>` - Database versioning
- `enhanced_config: EnhancedStorageConfig` - Configuration for AI features

**Key Methods:**
- `new(knowledge_engine)` - Creates server with default configuration
- `new_with_enhanced_config(engine, config)` - Creates server with custom AI configuration
- `handle_request(request: LLMMCPRequest)` - Main request router with timeout protection
- `get_available_tools()` - Returns all available MCP tools
- `get_health()` - Health check with model manager status

### Enhanced Storage Configuration

**Purpose:** Controls AI-powered processing features.

**Configuration Options:**
- `enable_intelligent_processing: bool` - Toggle for AI features
- `enable_multi_hop_reasoning: bool` - Advanced reasoning capabilities
- `model_memory_limit: u64` - Memory limit for AI models (default: 2GB)
- `max_processing_time_seconds: u64` - Timeout for AI operations
- `fallback_on_failure: bool` - Graceful degradation to basic processing
- `cache_enhanced_results: bool` - Result caching

## 5. File Breakdown

### `mod.rs` - Main Server Implementation

**Classes:**
- `LLMFriendlyMCPServer`
  - **Description:** Main MCP server handling all LLM requests
  - **Methods:**
    - `handle_request(request)` - Routes requests to appropriate handlers with timeout
    - `get_usage_stats()` - Returns performance metrics
    - `reset_stats()` - Clears usage statistics
    - `get_health()` - Health status including model manager info

**Functions:**
- Request routing logic for all 20+ MCP tools
- Timeout protection (5-second default)
- Migration handling for deprecated tools
- Performance tracking and statistics

### `types.rs` - Type Definitions

**Structures:**
- `UsageStats` - Performance metrics tracking
  - `total_operations: u64` - Total requests processed
  - `avg_response_time_ms: f64` - Average response time
  - `memory_efficiency: f64` - Memory usage efficiency
  - `cache_hits/misses: u64` - Cache performance
  - `uptime: Instant` - Server uptime tracking

- `ValidationResult` - Knowledge validation results
  - `is_valid: bool` - Validation status
  - `confidence: f64` - Confidence score (0.0-1.0)
  - `conflicts: Vec<String>` - List of validation issues
  - `validation_notes: Vec<String>` - Additional notes

### `tools.rs` - MCP Tool Definitions

**Purpose:** Defines all 20+ tools available to LLMs with schemas, examples, and tips.

**Core Storage Tools:**
- `store_fact` - Store simple subject-predicate-object triples
- `store_knowledge` - Store complex text with AI-powered extraction
- `find_facts` - Query triples with enhanced retrieval
- `ask_question` - Natural language question answering

**Advanced Tools:**
- `hybrid_search` - Multi-mode search (semantic, structural, keyword)
- `analyze_graph` - Graph analysis (centrality, clustering, prediction)
- `validate_knowledge` - Comprehensive validation with quality metrics
- `divergent_thinking_engine` - Creative exploration and ideation

**Specialized Tools:**
- `time_travel_query` - Temporal database queries
- `cognitive_reasoning_chains` - Logical reasoning (deductive, inductive, etc.)
- `create_branch`/`merge_branches` - Git-like database versioning

### `handlers/storage.rs` - Storage Operations

**Functions:**
- `handle_store_fact(engine, stats, params)` - Stores individual triples
  - Validates input parameters (length limits, empty fields)
  - Creates Triple with metadata
  - Records temporal changes
  - Updates usage statistics

- `handle_store_knowledge(engine, stats, params)` - Stores complex knowledge
  - Falls back to basic processing when AI features unavailable
  - Extracts entities and relationships from text
  - Creates knowledge chunks with UUID identifiers
  - Links entities to chunks via relationships

**Helper Functions:**
- `extract_entities_from_text(text)` - Simple NLP entity extraction
- `extract_relationships_from_text(text, entities)` - Basic relationship extraction
- `is_common_word(word)` - Stop word filtering

### `handlers/query.rs` - Query Operations

**Functions:**
- `handle_find_facts(engine, stats, params)` - Triple pattern matching
  - Supports subject/predicate/object filters
  - Falls back to basic search when enhanced retrieval fails
  - Returns formatted results with relevance scoring

- `handle_ask_question(engine, stats, params)` - Natural language Q&A
  - Extracts key terms from questions
  - Searches across multiple triple patterns
  - Generates contextual answers
  - Provides relevance scoring

**Helper Functions:**
- `extract_key_terms(question)` - Extracts entities and quoted phrases
- `calculate_relevance(triple, question)` - Relevance scoring
- `generate_answer(facts, question)` - Simple answer generation

### `reasoning_engine.rs` - Cognitive Reasoning

**Purpose:** Pure algorithmic reasoning without AI models.

**Reasoning Types:**
- `execute_deductive_reasoning()` - General to specific reasoning
- `execute_inductive_reasoning()` - Pattern recognition from examples  
- `execute_abductive_reasoning()` - Best explanation finding
- `execute_analogical_reasoning()` - Similarity-based inference

**Structures:**
- `ReasoningChain` - Represents a chain of logical steps
- `ReasoningStep` - Individual inference step
- `ReasoningResult` - Complete reasoning analysis

**Features:**
- Cycle detection to prevent infinite loops
- Contradiction identification
- Confidence scoring for each reasoning step
- Alternative reasoning path generation

### `database_branching.rs` - Version Control

**Purpose:** Git-like branching for knowledge graphs.

**Classes:**
- `DatabaseBranchManager` - Manages multiple database branches
  - **Methods:**
    - `create_branch(source, name, description)` - Creates new branch
    - `compare_branches(branch1, branch2)` - Shows differences
    - `merge_branches(source, target, strategy)` - Merges changes
    - `copy_knowledge_engine(source)` - Deep copy of knowledge

**Structures:**
- `BranchInfo` - Branch metadata (creation time, parent, description)
- `BranchComparison` - Comparison results between branches
- `MergeResult` - Results of merge operations

**Merge Strategies:**
- `AcceptSource` - Take all changes from source branch
- `AcceptTarget` - Keep target unchanged
- `Manual` - Manual conflict resolution (not implemented)

### `validation.rs` - Knowledge Validation

**Functions:**
- `validate_triple(triple)` - Single triple validation
  - Checks for empty fields and length limits
  - Validates predicate naming conventions
  - Identifies problematic characters

- `validate_consistency(new_triples, existing_triples)` - Cross-validation
  - Detects conflicts in single-valued predicates
  - Identifies circular relationships
  - Builds fact indices for efficient checking

- `validate_with_llm(triple, context)` - AI-assisted validation
  - Logical contradiction detection
  - Temporal consistency checking
  - Common sense violation detection
  - Context consistency analysis

- `validate_completeness(entity, triples)` - Missing information detection
  - Checks for expected predicates based on entity type
  - Identifies gaps in entity descriptions

### `temporal_tracking.rs` - Time-Based Tracking

**Purpose:** Track all changes to knowledge over time.

**Operations Tracked:**
- `Create` - New knowledge creation
- `Update` - Modification of existing knowledge
- `Delete` - Knowledge removal

**Features:**
- Global temporal index using lazy_static
- ISO timestamp recording
- Change history preservation
- Evolution tracking for entities

## 6. Key Dependencies

### Internal Dependencies
- `crate::core::knowledge_engine::KnowledgeEngine` - Core graph storage
- `crate::core::triple::Triple` - Basic knowledge representation
- `crate::core::knowledge_types::TripleQuery` - Query structures
- `crate::versioning::MultiDatabaseVersionManager` - Version control
- `crate::mcp::shared_types::*` - MCP protocol types

### External Dependencies
- `tokio::sync::RwLock` - Async read-write locks
- `serde_json` - JSON serialization/deserialization
- `chrono` - Date and time handling
- `uuid` - Unique identifier generation
- `regex` - Pattern matching for validation

## 7. API Endpoints (MCP Tools)

### Core Storage Operations
- **`store_fact`** - Store simple facts as triples
  - **Parameters:** subject, predicate, object, confidence
  - **Returns:** Success status and node ID

- **`store_knowledge`** - Store complex text with AI processing
  - **Parameters:** content, title, category, source
  - **Returns:** Extracted entities and relationships

### Query Operations
- **`find_facts`** - Find triples matching patterns
  - **Parameters:** query (subject/predicate/object filters), limit
  - **Returns:** Matching triples with confidence scores

- **`ask_question`** - Natural language question answering
  - **Parameters:** question, context, max_results
  - **Returns:** Answer with supporting evidence

### Advanced Operations
- **`hybrid_search`** - Multi-modal search
  - **Parameters:** query, search_type, performance_mode, filters
  - **Returns:** Ranked results with similarity scores

- **`analyze_graph`** - Graph analysis suite
  - **Parameters:** analysis_type, config
  - **Returns:** Analysis results (centrality, clustering, etc.)

### Reasoning Operations
- **`cognitive_reasoning_chains`** - Logical reasoning
  - **Parameters:** reasoning_type, premise, max_chain_length
  - **Returns:** Reasoning chains with confidence scores

- **`divergent_thinking_engine`** - Creative exploration
  - **Parameters:** seed_concept, creativity_level, exploration_depth
  - **Returns:** Novel connections and insights

### Temporal Operations
- **`time_travel_query`** - Historical knowledge queries
  - **Parameters:** query_type, timestamp, entity, time_range
  - **Returns:** Historical knowledge states

### Branching Operations
- **`create_branch`** - Create new database branch
  - **Parameters:** source_db_id, branch_name, description
  - **Returns:** New branch database ID

- **`merge_branches`** - Merge branch changes
  - **Parameters:** source_branch, target_branch, merge_strategy
  - **Returns:** Merge results and statistics

## 8. Key Features and Design Patterns

### Fault Tolerance
- All enhanced AI features have fallback mechanisms
- Basic processing continues when advanced features fail
- Timeout protection on all operations (5-second default)
- Graceful error handling with descriptive messages

### Performance Optimization
- Async/await throughout for non-blocking operations
- Read-write locks for concurrent access
- Result caching for expensive operations
- Memory-efficient data structures

### Extensibility
- Plugin-like handler architecture
- Easy addition of new MCP tools
- Configurable AI processing pipeline
- Migration system for tool evolution

### Data Integrity
- Comprehensive validation at multiple levels
- Temporal change tracking
- Database versioning and branching
- Conflict detection and resolution

### LLM-Friendly Design
- Natural language interfaces
- Rich error messages with suggestions
- Extensive documentation and examples
- Confidence scoring throughout

## 9. Usage Patterns

### Basic Knowledge Storage
```rust
// Store a simple fact
let request = LLMMCPRequest {
    method: "store_fact".to_string(),
    params: json!({
        "subject": "Einstein",
        "predicate": "is",
        "object": "scientist",
        "confidence": 1.0
    })
};
```

### Natural Language Querying
```rust
// Ask a question
let request = LLMMCPRequest {
    method: "ask_question".to_string(),
    params: json!({
        "question": "What did Einstein discover?",
        "max_results": 5
    })
};
```

### Advanced Reasoning
```rust
// Perform deductive reasoning
let request = LLMMCPRequest {
    method: "cognitive_reasoning_chains".to_string(),
    params: json!({
        "reasoning_type": "deductive",
        "premise": "Einstein developed special relativity",
        "max_chain_length": 4
    })
};
```

### Database Branching
```rust
// Create experimental branch
let request = LLMMCPRequest {
    method: "create_branch".to_string(),
    params: json!({
        "source_db_id": "main",
        "branch_name": "quantum-experiment",
        "description": "Testing quantum physics relationships"
    })
};
```

## 10. Error Handling and Validation

### Input Validation
- Parameter type checking
- Length limits enforcement
- Required field validation
- Range checking for numeric parameters

### Knowledge Validation
- Triple consistency checking
- Logical contradiction detection
- Temporal consistency validation
- Source credibility assessment

### Error Recovery
- Automatic fallback to basic processing
- Partial result return when possible
- Detailed error messages with suggestions
- Operation statistics tracking

## 11. Performance Considerations

### Scalability
- Configurable memory limits
- Result pagination
- Timeout protection
- Resource monitoring

### Optimization
- Lazy loading of AI models
- Result caching
- Batch processing support
- Memory-efficient data structures

This LLM-friendly MCP server represents a sophisticated knowledge graph interface designed specifically for AI interaction, with comprehensive features for storage, querying, reasoning, and validation while maintaining high performance and reliability.