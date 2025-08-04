# Phase 1: Mock Infrastructure

## Executive Summary
Create comprehensive mock infrastructure following London School TDD methodology. Build ALL mocks and interface contracts BEFORE any real implementation. Establish test harness for progressive mock replacement in subsequent phases.

## Duration
1 Week (40 hours) - Mock-first development with comprehensive test coverage

## London School TDD Philosophy

### Mock-First Principles
1. **Create mocks BEFORE implementation**
2. **Define interfaces through test behavior**
3. **Isolate units under test completely**
4. **Focus on interaction testing**
5. **Progressive mock replacement in later phases**

### Outside-In Development
- Start with acceptance tests
- Work from user-facing features inward
- Define collaborator interfaces through tests
- Defer implementation decisions

## SPARC Framework Application

### Specification
**Objective**: Create complete mock ecosystem for multi-method search system
**Requirements**:
- Mock implementations for all search engines
- Mock embedding services with multiple models
- Mock data access layers
- Mock external integrations (Git, file system)
- Interface contracts defined through tests
- Comprehensive test harness

**Success Criteria**:
- All mocks implement defined interfaces
- 100% test coverage for mock interactions
- Progressive replacement strategy documented
- Interface contracts validated

### Pseudocode
```
MockInfrastructure {
    1. Search Engine Mocks
       - MockRipgrepEngine
       - MockTantivyEngine
       - MockFuzzySearch
       - MockASTParser
       - MockBM25Engine
    
    2. Embedding Service Mocks
       - MockVoyageCode2
       - MockE5Mistral
       - MockBGEM3
       - MockCodeBERT
       - MockSQLCoder
       - MockBERTConfig
       - MockStackTraceBERT
    
    3. Data Layer Mocks
       - MockFileSystem
       - MockGitAnalyzer
       - MockVectorDatabase
       - MockCacheLayer
    
    4. Integration Mocks
       - MockSynthesisEngine
       - MockQueryRouter
       - MockResultRanker
}
```

### Architecture
```
Mock Infrastructure Architecture:
├── Search Engine Mocks
│   ├── MockRipgrepEngine (Exact Match)
│   ├── MockTantivyEngine (Token Search)
│   ├── MockFuzzySearch (Typo Tolerance)
│   ├── MockASTParser (Structural)
│   └── MockBM25Engine (Statistical)
├── Embedding Service Mocks
│   ├── Code Specialists
│   │   ├── MockVoyageCode2 (93% accuracy target)
│   │   ├── MockCodeBERT (89% accuracy target)
│   │   └── MockSQLCoder (91% accuracy target)
│   ├── Documentation Specialists
│   │   ├── MockE5Mistral (92% accuracy target)
│   │   └── MockBERTConfig (88% accuracy target)
│   └── Local Processing
│       ├── MockBGEM3 (86% accuracy target)
│       └── MockStackTraceBERT (90% accuracy target)
├── Data Access Mocks
│   ├── MockFileSystem
│   ├── MockGitAnalyzer
│   ├── MockVectorDatabase
│   └── MockCacheLayer
└── Integration Mocks
    ├── MockSynthesisEngine
    ├── MockQueryRouter
    └── MockResultRanker
```

### Refinement
- Interface-driven design with clear contracts
- Configurable mock behaviors for different test scenarios
- State tracking for interaction verification
- Performance simulation for realistic testing
- Error injection capabilities for robustness testing

### Completion
- All 99 mock tasks completed (task_100 through task_199)
- Interface contracts documented and validated
- Test harness ready for progressive replacement
- Mock replacement strategy defined
- Ready for Phase 2: Multi-Method Search Implementation

## Mock Interface Contracts

### Search Engine Interface
```rust
pub trait SearchEngine: Send + Sync {
    async fn search(&self, query: &SearchQuery) -> Result<SearchResults, SearchError>;
    async fn index_document(&self, document: &Document) -> Result<(), IndexError>;
    fn get_engine_type(&self) -> EngineType;
    fn get_accuracy_rating(&self) -> f32;
    fn supports_special_characters(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub text: String,
    pub query_type: QueryType,
    pub filters: Vec<SearchFilter>,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct SearchResults {
    pub results: Vec<SearchResult>,
    pub total_found: usize,
    pub search_time_ms: u64,
    pub engine_used: EngineType,
}
```

### Embedding Service Interface
```rust
pub trait EmbeddingService: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    fn get_dimension_count(&self) -> usize;
    fn get_model_name(&self) -> &str;
    fn get_accuracy_rating(&self) -> f32;
    fn get_content_specialization(&self) -> ContentType;
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContentType {
    Code,
    Documentation,
    Comments,
    Identifiers,
    SQL,
    Configuration,
    ErrorMessages,
}
```

## Atomic Task Breakdown (100-199)

### Mock Search Engines (100-119)
- **task_100**: Design SearchEngine trait interface
- **task_101**: Create MockRipgrepEngine structure
- **task_102**: Implement MockRipgrepEngine exact matching behavior
- **task_103**: Add special character support to MockRipgrepEngine
- **task_104**: Create MockTantivyEngine structure
- **task_105**: Implement MockTantivyEngine token search behavior
- **task_106**: Add boolean query support to MockTantivyEngine
- **task_107**: Create MockFuzzySearch structure
- **task_108**: Implement MockFuzzySearch typo tolerance behavior
- **task_109**: Add edit distance configuration to MockFuzzySearch
- **task_110**: Create MockASTParser structure
- **task_111**: Implement MockASTParser structural matching behavior
- **task_112**: Add language-specific AST support to MockASTParser
- **task_113**: Create MockBM25Engine structure
- **task_114**: Implement MockBM25Engine statistical scoring behavior
- **task_115**: Add TF-IDF calculations to MockBM25Engine
- **task_116**: Create search engine factory pattern
- **task_117**: Implement search engine registry
- **task_118**: Add search engine performance simulation
- **task_119**: Create comprehensive search engine tests

### Mock Embedding Services (120-139)
- **task_120**: Design EmbeddingService trait interface
- **task_121**: Create MockVoyageCode2 structure
- **task_122**: Implement MockVoyageCode2 code embedding behavior
- **task_123**: Add 93% accuracy simulation to MockVoyageCode2
- **task_124**: Create MockE5Mistral structure
- **task_125**: Implement MockE5Mistral documentation embedding behavior
- **task_126**: Add 92% accuracy simulation to MockE5Mistral
- **task_127**: Create MockBGEM3 structure (local processing)
- **task_128**: Implement MockBGEM3 comment embedding behavior
- **task_129**: Add 86% accuracy simulation to MockBGEM3
- **task_130**: Create MockCodeBERT structure
- **task_131**: Implement MockCodeBERT identifier embedding behavior
- **task_132**: Add 89% accuracy simulation to MockCodeBERT
- **task_133**: Create MockSQLCoder structure
- **task_134**: Implement MockSQLCoder SQL embedding behavior
- **task_135**: Add 91% accuracy simulation to MockSQLCoder
- **task_136**: Create MockBERTConfig structure
- **task_137**: Implement MockBERTConfig configuration embedding behavior
- **task_138**: Create MockStackTraceBERT structure
- **task_139**: Implement MockStackTraceBERT error embedding behavior

### Mock Data Access Layer (140-159)
- **task_140**: Design FileSystem interface
- **task_141**: Create MockFileSystem structure
- **task_142**: Implement MockFileSystem read operations
- **task_143**: Implement MockFileSystem write operations
- **task_144**: Add Windows path handling to MockFileSystem
- **task_145**: Design GitAnalyzer interface
- **task_146**: Create MockGitAnalyzer structure
- **task_147**: Implement MockGitAnalyzer commit history analysis
- **task_148**: Implement MockGitAnalyzer author expertise mapping
- **task_149**: Add regression detection to MockGitAnalyzer
- **task_150**: Design VectorDatabase interface
- **task_151**: Create MockVectorDatabase structure
- **task_152**: Implement MockVectorDatabase CRUD operations
- **task_153**: Add transaction support to MockVectorDatabase
- **task_154**: Implement MockVectorDatabase similarity search
- **task_155**: Design CacheLayer interface
- **task_156**: Create MockCacheLayer structure
- **task_157**: Implement MockCacheLayer storage operations
- **task_158**: Add cache eviction policies to MockCacheLayer
- **task_159**: Create data access integration tests

### Mock Integration Layer (160-179)
- **task_160**: Design SynthesisEngine interface
- **task_161**: Create MockSynthesisEngine structure
- **task_162**: Implement MockSynthesisEngine result combination
- **task_163**: Add confidence scoring to MockSynthesisEngine
- **task_164**: Implement contradiction resolution in MockSynthesisEngine
- **task_165**: Design QueryRouter interface
- **task_166**: Create MockQueryRouter structure
- **task_167**: Implement MockQueryRouter tier selection
- **task_168**: Add cost optimization to MockQueryRouter
- **task_169**: Implement query complexity analysis in MockQueryRouter
- **task_170**: Design ResultRanker interface
- **task_171**: Create MockResultRanker structure
- **task_172**: Implement MockResultRanker scoring algorithms
- **task_173**: Add relevance boosting to MockResultRanker
- **task_174**: Implement diversity ranking in MockResultRanker
- **task_175**: Create integration orchestration layer
- **task_176**: Implement mock service discovery
- **task_177**: Add service health monitoring mocks
- **task_178**: Create integration failure simulation
- **task_179**: Create integration layer tests

### Mock Test Harness (180-199)
- **task_180**: Design comprehensive test harness
- **task_181**: Create mock behavior configuration system
- **task_182**: Implement mock state tracking
- **task_183**: Add interaction verification framework
- **task_184**: Create mock performance simulation
- **task_185**: Implement error injection capabilities
- **task_186**: Add concurrent access testing
- **task_187**: Create mock replacement validation
- **task_188**: Implement integration test scenarios
- **task_189**: Add stress testing framework
- **task_190**: Create mock behavior documentation
- **task_191**: Implement mock metrics collection
- **task_192**: Add mock debugging capabilities
- **task_193**: Create acceptance test scenarios
- **task_194**: Implement end-to-end mock testing
- **task_195**: Add performance regression testing
- **task_196**: Create mock replacement roadmap
- **task_197**: Implement mock retirement strategy
- **task_198**: Add production readiness validation
- **task_199**: Final mock infrastructure validation

## TDD Implementation Pattern for Mocks

### Mock Creation Cycle (RED-GREEN-REFACTOR)

#### RED Phase: Define Expected Behavior
```rust
#[test]
fn test_mock_ripgrep_exact_search() {
    // Arrange: Create mock with expected behavior
    let mut mock_ripgrep = MockRipgrepEngine::new();
    mock_ripgrep
        .expect_search()
        .with(eq(SearchQuery::exact("println!")))
        .returning(|_| Ok(SearchResults {
            results: vec![SearchResult {
                file_path: "src/main.rs".to_string(),
                line_number: 42,
                content: "    println!(\"Hello, world!\");".to_string(),
                score: 1.0,
            }],
            total_found: 1,
            search_time_ms: 5,
            engine_used: EngineType::Ripgrep,
        }));

    // Act: Use mock in system under test
    let system = SearchSystem::new(Box::new(mock_ripgrep));
    let results = system.search_exact("println!").await.unwrap();

    // Assert: Verify expected interactions
    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].file_path, "src/main.rs");
}
```

#### GREEN Phase: Minimal Mock Implementation
```rust
pub struct MockRipgrepEngine {
    expectations: Vec<SearchExpectation>,
}

impl MockRipgrepEngine {
    pub fn new() -> Self {
        Self {
            expectations: Vec::new(),
        }
    }

    pub fn expect_search(&mut self) -> &mut SearchExpectation {
        let expectation = SearchExpectation::new();
        self.expectations.push(expectation);
        self.expectations.last_mut().unwrap()
    }
}

impl SearchEngine for MockRipgrepEngine {
    async fn search(&self, query: &SearchQuery) -> Result<SearchResults, SearchError> {
        // Find matching expectation and return configured result
        for expectation in &self.expectations {
            if expectation.matches(query) {
                return expectation.get_result();
            }
        }
        Err(SearchError::NoExpectationMatched)
    }
}
```

#### REFACTOR Phase: Rich Mock Capabilities
```rust
impl MockRipgrepEngine {
    pub fn with_performance_simulation(mut self, latency_ms: u64) -> Self {
        self.simulated_latency = Some(latency_ms);
        self
    }

    pub fn with_error_injection(mut self, error_rate: f32) -> Self {
        self.error_injection_rate = error_rate;
        self
    }

    pub fn with_accuracy_simulation(mut self, accuracy: f32) -> Self {
        self.simulated_accuracy = accuracy;
        self
    }
}
```

## Mock Behavior Specifications

### Search Engine Mock Behaviors

#### MockRipgrepEngine (100% Exact Match)
```rust
pub struct MockRipgrepEngine {
    // Behavior: Perfect literal string matching
    // Accuracy: 100% for exact strings
    // Performance: <5ms simulated latency
    // Special chars: Full support
}
```

#### MockTantivyEngine (Token Search)
```rust
pub struct MockTantivyEngine {
    // Behavior: Token-based search with boolean logic
    // Accuracy: 95% for token queries
    // Performance: <10ms simulated latency
    // Special chars: Configurable tokenization
}
```

#### MockFuzzySearch (Typo Tolerance)
```rust
pub struct MockFuzzySearch {
    // Behavior: Edit distance-based matching
    // Accuracy: 85% with configurable distance
    // Performance: <20ms simulated latency
    // Special chars: Limited support
}
```

### Embedding Service Mock Behaviors

#### MockVoyageCode2 (Code Specialist)
```rust
pub struct MockVoyageCode2 {
    // Specialization: Code structures, syntax
    // Accuracy: 93% simulated on code
    // Dimensions: 3072
    // API cost: $0.01 per 1K tokens simulated
}
```

#### MockE5Mistral (Documentation Specialist)
```rust
pub struct MockE5Mistral {
    // Specialization: Technical documentation
    // Accuracy: 92% simulated on docs
    // Dimensions: 4096
    // API cost: $0.005 per 1K tokens simulated
}
```

## Mock Replacement Strategy

### Phase 2: Replace Search Engine Mocks
- Replace MockRipgrepEngine with real ripgrep integration
- Replace MockTantivyEngine with real Tantivy implementation
- Keep embedding mocks until Phase 3
- Validate interface compatibility

### Phase 3: Replace Embedding Mocks
- Replace MockVoyageCode2 with real API integration
- Replace MockE5Mistral with real API integration
- Keep local MockBGEM3 for fast processing
- Validate embedding quality

### Phase 4: Replace Data Access Mocks
- Replace MockFileSystem with real file operations
- Replace MockGitAnalyzer with real Git integration
- Replace MockVectorDatabase with real LanceDB
- Validate performance characteristics

### Phase 5: Replace Integration Mocks
- Replace MockSynthesisEngine with real synthesis logic
- Replace MockQueryRouter with real routing implementation
- Replace MockResultRanker with real ranking algorithms
- Validate end-to-end system behavior

## Success Criteria

### Mock Infrastructure Complete When:
- [ ] All 99 mock tasks completed (100-199)
- [ ] Interface contracts defined and validated
- [ ] Mock behaviors accurately simulate expected real behavior
- [ ] Test harness supports all testing scenarios
- [ ] Mock replacement strategy documented
- [ ] Performance simulation realistic
- [ ] Error injection comprehensive
- [ ] Documentation complete

### Quality Gates
- **Interface Compliance**: 100% adherence to defined contracts
- **Test Coverage**: 100% mock interaction coverage
- **Behavioral Accuracy**: Mock behavior matches expected real behavior
- **Performance Simulation**: Realistic latency and throughput simulation
- **Error Handling**: Comprehensive error scenario coverage

## Handoff to Phase 2

### Deliverables
1. **Complete Mock Ecosystem**: All components mocked and tested
2. **Interface Contracts**: Formal definitions for all system boundaries
3. **Test Harness**: Comprehensive testing infrastructure
4. **Behavioral Specifications**: Documented expected behaviors
5. **Replacement Strategy**: Clear roadmap for progressive mock replacement

### Phase 2 Prerequisites Met
- ✅ All search engines mocked and tested
- ✅ All embedding services mocked and tested
- ✅ All data access layers mocked and tested
- ✅ Test harness ready for real implementation validation
- ✅ Interface contracts established for implementation guidance

---

**Next Phase**: Phase 2: Multi-Method Search Implementation (Progressive mock replacement)