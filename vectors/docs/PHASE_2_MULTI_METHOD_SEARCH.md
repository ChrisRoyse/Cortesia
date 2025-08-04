# Phase 2: Multi-Method Search Implementation

## Executive Summary
Replace search engine mocks progressively with real implementations. Build comprehensive multi-method search foundation with exact match (ripgrep), token search (Tantivy), fuzzy search, AST search (tree-sitter), and BM25/TF-IDF statistical scoring. Each method achieves specific accuracy targets for different query types.

## Duration
1 Week (40 hours) - Progressive mock replacement with validation

## Progressive Mock Replacement Strategy

### London TDD Progression
1. **Replace one mock at a time**
2. **Validate interface compatibility**
3. **Maintain existing test suite**
4. **Add real implementation tests**
5. **Keep other mocks until their replacement phase**

### Mock Replacement Order
- **First**: MockRipgrepEngine → Real ripgrep integration
- **Second**: MockTantivyEngine → Real Tantivy implementation
- **Third**: MockFuzzySearch → Real fuzzy search algorithm
- **Fourth**: MockASTParser → Real tree-sitter integration
- **Fifth**: MockBM25Engine → Real statistical scoring

## SPARC Framework Application

### Specification
**Objective**: Create production-ready multi-method search system
**Requirements**:
- Exact match search: 100% accuracy for literal strings
- Token search: 95% accuracy for structured queries with special character support
- Fuzzy search: 85% accuracy with configurable edit distance
- AST search: 90% accuracy for structural code queries
- Statistical search: 88% accuracy using BM25/TF-IDF scoring
- Performance targets: <50ms per search method
- Windows compatibility maintained

**Success Criteria**:
- All mock replacements complete with interface compatibility
- Performance targets met or exceeded
- Accuracy targets validated with test data
- Integration tests passing
- Real-world query scenarios working

### Pseudocode
```
MultiMethodSearch {
    1. Exact Match Implementation (ripgrep)
       - Replace MockRipgrepEngine
       - Integrate ripgrep as library
       - Handle special characters perfectly
       - Achieve <5ms search latency
    
    2. Token Search Implementation (Tantivy)
       - Replace MockTantivyEngine
       - Build Tantivy index with custom schema
       - Support boolean queries
       - Handle special character tokenization
    
    3. Fuzzy Search Implementation
       - Replace MockFuzzySearch
       - Implement edit distance algorithms
       - Support configurable thresholds
       - Handle Unicode properly
    
    4. AST Search Implementation (tree-sitter)
       - Replace MockASTParser
       - Parse multiple programming languages
       - Support structural queries
       - Cache parsed ASTs for performance
    
    5. Statistical Search Implementation
       - Replace MockBM25Engine
       - Implement BM25 scoring
       - Add TF-IDF calculations
       - Support relevance ranking
}
```

### Architecture
```
Multi-Method Search Architecture:
├── Search Coordinator
│   ├── Method Selection Logic
│   ├── Parallel Execution Engine
│   └── Result Fusion System
├── Exact Match Layer (ripgrep)
│   ├── ripgrep Integration
│   ├── Pattern Compilation
│   └── Result Processing
├── Token Search Layer (Tantivy)
│   ├── Index Management
│   ├── Query Parsing
│   ├── Boolean Logic
│   └── Special Character Handling
├── Fuzzy Search Layer
│   ├── Edit Distance Algorithms
│   ├── Threshold Configuration
│   └── Unicode Normalization
├── AST Search Layer (tree-sitter)
│   ├── Language Grammar Loading
│   ├── AST Parsing and Caching
│   ├── Structural Query Processing
│   └── Multi-Language Support
└── Statistical Search Layer
    ├── BM25 Implementation
    ├── TF-IDF Calculations
    ├── Document Frequency Stats
    └── Score Normalization
```

### Refinement
- Incremental mock replacement with validation at each step
- Performance optimization only after correctness established
- Comprehensive error handling for each search method
- Caching strategies for expensive operations (AST parsing)
- Configurable parameters for fine-tuning accuracy vs speed

### Completion
- All 99 implementation tasks completed (task_200 through task_299)
- All search engine mocks replaced with real implementations
- Performance targets met
- Accuracy targets validated
- Ready for Phase 3: Multi-Embedding System

## Search Method Specifications

### Method 1: Exact Match (ripgrep) - 100% Accuracy
```rust
pub struct RipgrepEngine {
    // Integrates ripgrep as library for exact string matching
    // Accuracy: 100% for literal strings
    // Performance: <5ms average latency
    // Special characters: Full support including regex metacharacters
    // Use cases: API names, exact code snippets, configuration values
}

impl SearchEngine for RipgrepEngine {
    async fn search(&self, query: &SearchQuery) -> Result<SearchResults, SearchError> {
        // Direct integration with ripgrep library
        // Handle special characters by escaping
        // Return exact matches with context
    }
}
```

### Method 2: Token Search (Tantivy) - 95% Accuracy
```rust
pub struct TantivyEngine {
    // Tantivy-based indexing with custom tokenization
    // Accuracy: 95% for structured queries
    // Performance: <10ms average latency
    // Special characters: Custom tokenizer preserves important chars
    // Use cases: Multi-word searches, boolean queries, filtered searches
}

impl SearchEngine for TantivyEngine {
    async fn search(&self, query: &SearchQuery) -> Result<SearchResults, SearchError> {
        // Parse query using Tantivy query parser
        // Handle boolean logic (AND, OR, NOT)
        // Support phrase queries and wildcards
        // Return ranked results with BM25 scoring
    }
}
```

### Method 3: Fuzzy Search - 85% Accuracy
```rust
pub struct FuzzySearchEngine {
    // Edit distance-based fuzzy matching
    // Accuracy: 85% with configurable edit distance
    // Performance: <20ms average latency
    // Special characters: Unicode-aware normalization
    // Use cases: Typo tolerance, similar function names, approximate matching
}

impl SearchEngine for FuzzySearchEngine {
    async fn search(&self, query: &SearchQuery) -> Result<SearchResults, SearchError> {
        // Calculate edit distance using Wagner-Fischer algorithm
        // Support configurable distance thresholds
        // Handle Unicode normalization
        // Return results sorted by similarity score
    }
}
```

### Method 4: AST Search (tree-sitter) - 90% Accuracy
```rust
pub struct ASTSearchEngine {
    // Structural code search using AST parsing
    // Accuracy: 90% for structural queries
    // Performance: <30ms average latency (with caching)
    // Special characters: Language-specific handling
    // Use cases: Function signatures, class hierarchies, import statements
}

impl SearchEngine for ASTSearchEngine {
    async fn search(&self, query: &SearchQuery) -> Result<SearchResults, SearchError> {
        // Parse query as structural pattern
        // Match against cached ASTs
        // Support multiple programming languages
        // Return semantically relevant results
    }
}
```

### Method 5: Statistical Search (BM25) - 88% Accuracy
```rust
pub struct BM25Engine {
    // BM25 and TF-IDF statistical ranking
    // Accuracy: 88% for relevance ranking
    // Performance: <25ms average latency
    // Special characters: Statistical treatment
    // Use cases: Document ranking, relevance scoring, similarity search
}

impl SearchEngine for BM25Engine {
    async fn search(&self, query: &SearchQuery) -> Result<SearchResults, SearchError> {
        // Calculate BM25 scores for query terms
        // Apply TF-IDF weighting
        // Normalize scores across documents
        // Return statistically ranked results
    }
}
```

## Atomic Task Breakdown (200-299)

### Exact Match Implementation (200-219)
- **task_200**: Replace MockRipgrepEngine with RipgrepEngine structure
- **task_201**: Integrate ripgrep library programmatically
- **task_202**: Implement exact string matching with ripgrep
- **task_203**: Add special character escaping for ripgrep patterns
- **task_204**: Handle Windows path separators in ripgrep
- **task_205**: Implement result parsing from ripgrep output
- **task_206**: Add context line extraction around matches
- **task_207**: Implement performance optimization for large files
- **task_208**: Add file type filtering to ripgrep searches
- **task_209**: Implement recursive directory searching
- **task_210**: Add ignore file support (.gitignore, etc.)
- **task_211**: Implement parallel file processing with ripgrep
- **task_212**: Add binary file detection and handling
- **task_213**: Implement match highlighting and context
- **task_214**: Add search statistics and timing
- **task_215**: Implement error handling for ripgrep failures
- **task_216**: Add Unicode support validation
- **task_217**: Implement ripgrep configuration management
- **task_218**: Add comprehensive ripgrep integration tests
- **task_219**: Validate ripgrep performance targets (<5ms)

### Token Search Implementation (220-239)
- **task_220**: Replace MockTantivyEngine with TantivyEngine structure
- **task_221**: Design Tantivy schema for code search
- **task_222**: Implement custom tokenizer for special characters
- **task_223**: Build Tantivy index with document chunking
- **task_224**: Implement boolean query parsing (AND, OR, NOT)
- **task_225**: Add phrase query support to Tantivy
- **task_226**: Implement wildcard query processing
- **task_227**: Add field-specific searching (file, content, path)
- **task_228**: Implement BM25 scoring in Tantivy
- **task_229**: Add query result highlighting
- **task_230**: Implement index incremental updates
- **task_231**: Add faceted search capabilities
- **task_232**: Implement search result pagination
- **task_233**: Add query performance monitoring
- **task_234**: Implement index optimization and merging
- **task_235**: Add search suggestion/correction
- **task_236**: Implement multi-field query boosting
- **task_237**: Add comprehensive Tantivy integration tests
- **task_238**: Validate Tantivy performance targets (<10ms)
- **task_239**: Test special character indexing and searching

### Fuzzy Search Implementation (240-259)
- **task_240**: Replace MockFuzzySearch with FuzzySearchEngine structure
- **task_241**: Implement Wagner-Fischer edit distance algorithm
- **task_242**: Add Levenshtein distance calculation
- **task_243**: Implement Damerau-Levenshtein distance
- **task_244**: Add configurable edit distance thresholds
- **task_245**: Implement Unicode normalization for fuzzy matching
- **task_246**: Add phonetic matching algorithms (Soundex, Metaphone)
- **task_247**: Implement fuzzy matching with character weights
- **task_248**: Add fuzzy search result ranking by similarity
- **task_249**: Implement fuzzy prefix matching
- **task_250**: Add fuzzy substring matching
- **task_251**: Implement n-gram based fuzzy search
- **task_252**: Add fuzzy search performance optimization
- **task_253**: Implement fuzzy search caching
- **task_254**: Add fuzzy search configuration management
- **task_255**: Implement fuzzy search error handling
- **task_256**: Add comprehensive fuzzy search tests
- **task_257**: Validate fuzzy search accuracy targets (85%)
- **task_258**: Test fuzzy search performance targets (<20ms)
- **task_259**: Validate Unicode fuzzy matching

### AST Search Implementation (260-279)
- **task_260**: Replace MockASTParser with ASTSearchEngine structure
- **task_261**: Integrate tree-sitter library for AST parsing
- **task_262**: Load and configure programming language grammars
- **task_263**: Implement AST parsing with caching
- **task_264**: Add structural query pattern matching
- **task_265**: Implement function signature matching
- **task_266**: Add class and struct definition matching
- **task_267**: Implement import/include statement matching
- **task_268**: Add variable declaration matching
- **task_269**: Implement method call matching
- **task_270**: Add control flow structure matching
- **task_271**: Implement AST node relationship queries
- **task_272**: Add AST-based refactoring support
- **task_273**: Implement multi-language AST support
- **task_274**: Add AST parsing error handling
- **task_275**: Implement AST cache management
- **task_276**: Add comprehensive AST search tests
- **task_277**: Validate AST search accuracy targets (90%)
- **task_278**: Test AST search performance targets (<30ms)
- **task_279**: Validate multi-language AST support

### Statistical Search Implementation (280-299)
- **task_280**: Replace MockBM25Engine with BM25Engine structure
- **task_281**: Implement BM25 scoring algorithm
- **task_282**: Add TF-IDF calculation implementation
- **task_283**: Implement document frequency statistics
- **task_284**: Add term frequency analysis
- **task_285**: Implement inverse document frequency calculations
- **task_286**: Add score normalization algorithms
- **task_287**: Implement query term weighting
- **task_288**: Add document length normalization
- **task_289**: Implement relevance feedback mechanisms
- **task_290**: Add statistical model training
- **task_291**: Implement statistical search result ranking
- **task_292**: Add statistical search performance optimization
- **task_293**: Implement statistical model persistence
- **task_294**: Add statistical search configuration
- **task_295**: Implement comprehensive statistical search tests
- **task_296**: Validate BM25 accuracy targets (88%)
- **task_297**: Test statistical search performance targets (<25ms)
- **task_298**: Validate statistical ranking effectiveness
- **task_299**: Final multi-method search integration validation

## TDD Implementation Pattern

### Mock Replacement Cycle (RED-GREEN-REFACTOR)

#### RED Phase: Test Real Implementation Against Mock Interface
```rust
#[test]
fn test_ripgrep_engine_replaces_mock() {
    // Arrange: Create real RipgrepEngine
    let ripgrep_engine = RipgrepEngine::new(test_config());
    
    // Act: Use same interface as mock
    let query = SearchQuery::exact("println!");
    let result = ripgrep_engine.search(&query).await;
    
    // Assert: Should behave like mock but with real results
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.engine_used, EngineType::Ripgrep);
    // Real implementation should find actual matches
    assert!(!results.results.is_empty());
}
```

#### GREEN Phase: Minimal Real Implementation
```rust
pub struct RipgrepEngine {
    config: RipgrepConfig,
}

impl RipgrepEngine {
    pub fn new(config: RipgrepConfig) -> Self {
        Self { config }
    }
}

impl SearchEngine for RipgrepEngine {
    async fn search(&self, query: &SearchQuery) -> Result<SearchResults, SearchError> {
        // Minimal implementation using ripgrep
        let output = std::process::Command::new("rg")
            .arg(&query.text)
            .arg("--json")
            .output()?;
        
        // Parse JSON output into SearchResults
        let results = parse_ripgrep_output(&output.stdout)?;
        Ok(results)
    }
}
```

#### REFACTOR Phase: Optimized Real Implementation
```rust
impl SearchEngine for RipgrepEngine {
    async fn search(&self, query: &SearchQuery) -> Result<SearchResults, SearchError> {
        // Optimized implementation with proper error handling
        let mut grep = self.build_ripgrep_searcher()?;
        let matcher = self.build_pattern_matcher(&query.text)?;
        
        let mut results = Vec::new();
        let mut total_matches = 0;
        let start_time = Instant::now();
        
        grep.search_reader(
            &matcher,
            &mut std::io::stdin(),
            UTF8(|lnum, line| {
                results.push(SearchResult {
                    file_path: self.current_file.clone(),
                    line_number: lnum,
                    content: line.to_string(),
                    score: 1.0, // Exact match
                });
                total_matches += 1;
                Ok(true)
            })
        )?;
        
        Ok(SearchResults {
            results,
            total_found: total_matches,
            search_time_ms: start_time.elapsed().as_millis() as u64,
            engine_used: EngineType::Ripgrep,
        })
    }
}
```

## Performance and Accuracy Targets

### Method Performance Targets
| Search Method | Target Latency | Accuracy Target | Use Case |
|---------------|----------------|-----------------|----------|
| Exact Match (ripgrep) | <5ms | 100% | Literal strings, API names |
| Token Search (Tantivy) | <10ms | 95% | Boolean queries, multi-word |
| Fuzzy Search | <20ms | 85% | Typo tolerance, similar names |
| AST Search (tree-sitter) | <30ms | 90% | Structural code queries |
| Statistical (BM25) | <25ms | 88% | Relevance ranking |

### Accuracy Validation Strategy
- **Ground Truth Dataset**: Use Phase 0 generated test data
- **Cross-Validation**: Compare results between methods
- **Human Evaluation**: Manual verification of complex queries
- **Regression Testing**: Continuous accuracy monitoring

## Integration Strategy

### Parallel Execution Model
```rust
pub struct MultiMethodSearchCoordinator {
    ripgrep_engine: RipgrepEngine,
    tantivy_engine: TantivyEngine,
    fuzzy_engine: FuzzySearchEngine,
    ast_engine: ASTSearchEngine,
    bm25_engine: BM25Engine,
}

impl MultiMethodSearchCoordinator {
    pub async fn search_all_methods(&self, query: &SearchQuery) -> SearchResults {
        // Execute all search methods in parallel
        let (ripgrep_results, tantivy_results, fuzzy_results, ast_results, bm25_results) = 
            tokio::join!(
                self.ripgrep_engine.search(query),
                self.tantivy_engine.search(query),
                self.fuzzy_engine.search(query),
                self.ast_engine.search(query),
                self.bm25_engine.search(query),
            );
        
        // Combine and rank results
        self.fusion_engine.combine_results(vec![
            ripgrep_results?,
            tantivy_results?,
            fuzzy_results?,
            ast_results?,
            bm25_results?,
        ])
    }
}
```

## Success Criteria

### Phase Completion Requirements
- [ ] All 99 implementation tasks completed (200-299)
- [ ] All search engine mocks replaced with real implementations
- [ ] Interface compatibility maintained with existing tests
- [ ] Performance targets met for each search method
- [ ] Accuracy targets validated with test data
- [ ] Integration tests passing for multi-method coordination
- [ ] Windows compatibility verified for all implementations
- [ ] Error handling comprehensive and tested
- [ ] Documentation updated for real implementations

### Quality Gates
- **Functionality**: All search methods working correctly
- **Performance**: Latency targets met under load
- **Accuracy**: Accuracy targets validated with ground truth
- **Reliability**: Error handling and recovery tested
- **Maintainability**: Code quality and documentation standards met

## Handoff to Phase 3

### Deliverables  
1. **Multi-Method Search System**: All five search methods implemented and tested
2. **Performance Benchmarks**: Validated performance characteristics
3. **Accuracy Validation**: Proven accuracy against test datasets
4. **Integration Framework**: Coordinated execution of multiple search methods
5. **Windows Compatibility**: Full Windows support validated

### Phase 3 Prerequisites Met
- ✅ Search foundation solid for embedding integration
- ✅ Performance baselines established for optimization
- ✅ Test infrastructure ready for embedding validation
- ✅ Multi-method coordination ready for semantic enhancement
- ✅ Interface contracts proven with real implementations

---

**Next Phase**: Phase 3: Multi-Embedding System (Replace embedding service mocks)