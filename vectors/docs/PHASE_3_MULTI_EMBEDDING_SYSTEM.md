# Phase 3: Multi-Embedding System - Specialized Content Recognition

## Executive Summary
Implement a sophisticated multi-embedding system that achieves **95-97% accuracy** through content-type-aware embedding selection. Following London School TDD, we mock ALL embedding services first, then progressively replace mocks with real implementations, ensuring proper testing at each integration step.

## Duration
1.5 weeks (12 days) - Mock-first development with incremental real integration

## Objective
Build a multi-embedding pipeline that automatically detects content types and applies specialized embeddings for maximum accuracy:
- **MockVoyageCode2**: 93% accuracy on code structures
- **MockE5Mistral**: 92% accuracy on documentation 
- **MockBGE_M3**: 86% accuracy on comments (local processing)
- **MockCodeBERT**: 89% accuracy on identifiers
- **MockSQLCoder**: 91% accuracy on SQL queries
- **MockBERTConfig**: 88% accuracy on configuration files
- **MockStackTraceBERT**: 90% accuracy on error traces

## SPARC Framework Application

### Specification

#### Content Type Detection Requirements
```rust
pub enum ContentType {
    RustCode,        // .rs files with code structures
    Documentation,   // .md, .txt, comments
    Comments,        // // /* */ #
    Identifiers,     // function names, variables
    SqlQueries,      // SQL statements
    ConfigFiles,     // .toml, .json, .yaml
    ErrorTraces,     // stack traces, error logs
    Unknown,         // fallback
}

pub trait ContentTypeDetector {
    fn detect_content_type(&self, content: &str, file_path: &Path) -> ContentType;
    fn confidence_score(&self) -> f32;
}
```

#### Embedding Service Specifications
```rust
pub trait EmbeddingService {
    type Error;
    
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, Self::Error>;
    fn dimension_count(&self) -> usize;
    fn max_token_length(&self) -> usize;
    fn service_name(&self) -> &'static str;
}

// Accuracy targets from Master Plan
pub struct EmbeddingAccuracySpec {
    pub code_accuracy: f32,      // 93% for VoyageCode2
    pub doc_accuracy: f32,       // 92% for E5Mistral
    pub comment_accuracy: f32,   // 86% for BGE_M3
    pub identifier_accuracy: f32, // 89% for CodeBERT
    pub sql_accuracy: f32,       // 91% for SQLCoder
    pub config_accuracy: f32,    // 88% for BERTConfig
    pub error_accuracy: f32,     // 90% for StackTraceBERT
}
```

#### Pipeline Specifications
```rust
pub struct EmbeddingPipeline {
    content_detector: Box<dyn ContentTypeDetector>,
    embedding_services: HashMap<ContentType, Box<dyn EmbeddingService>>,
    cache: EmbeddingCache,
    fallback_service: Box<dyn EmbeddingService>,
}

// Success criteria
pub struct PipelineMetrics {
    pub accuracy_by_type: HashMap<ContentType, f32>,
    pub cache_hit_rate: f32,
    pub avg_embedding_time: Duration,
    pub error_rate: f32,
}
```

### Pseudocode

#### Content Detection Algorithm
```
ALGORITHM detect_content_type(content: str, file_path: Path) -> ContentType:
    // Extension-based initial detection
    IF file_path.extension() IN [".rs", ".py", ".js"]:
        IF contains_code_patterns(content):
            RETURN RustCode/PythonCode/JavaScript
    
    // Content-based detection
    IF contains_sql_keywords(content):
        RETURN SqlQueries
    
    IF is_configuration_format(content):
        RETURN ConfigFiles
    
    IF contains_stack_trace_patterns(content):
        RETURN ErrorTraces
    
    IF is_comment_heavy(content):
        RETURN Comments
    
    IF is_documentation_format(content):
        RETURN Documentation
    
    RETURN Unknown
END

ALGORITHM generate_specialized_embedding(content: str, content_type: ContentType) -> Vec<f32>:
    service = embedding_services[content_type]
    
    // Check cache first
    cache_key = hash(service.name(), content)
    IF cache.contains(cache_key):
        RETURN cache.get(cache_key)
    
    // Generate embedding
    TRY:
        embedding = service.generate_embedding(content)
        cache.put(cache_key, embedding)
        RETURN embedding
    CATCH error:
        LOG error
        RETURN fallback_service.generate_embedding(content)
END
```

#### Pipeline Processing Flow
```
ALGORITHM process_document_embeddings(document: Document) -> EmbeddedDocument:
    chunks = chunk_document(document)
    embedded_chunks = []
    
    FOR chunk IN chunks:
        content_type = detect_content_type(chunk.content, document.path)
        embedding = generate_specialized_embedding(chunk.content, content_type)
        
        embedded_chunk = EmbeddedChunk {
            content: chunk.content,
            embedding: embedding,
            content_type: content_type,
            chunk_index: chunk.index
        }
        
        embedded_chunks.append(embedded_chunk)
    END
    
    RETURN EmbeddedDocument {
        document: document,
        chunks: embedded_chunks,
        total_embeddings: embedded_chunks.length
    }
END
```

### Architecture

#### Component Relationships
```rust
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Embedding System                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐    ┌─────────────────────────────┐   │
│  │  Content Type    │    │     Embedding Services     │   │
│  │    Detector      │    │                             │   │
│  │                  │    │  ┌─────────────────────────┐ │   │
│  │ • File Extension │    │  │    MockVoyageCode2      │ │   │
│  │ • Pattern Match  │────┼──┤     (Code: 93%)        │ │   │
│  │ • Heuristics     │    │  └─────────────────────────┘ │   │
│  └──────────────────┘    │                             │   │
│                          │  ┌─────────────────────────┐ │   │
│  ┌──────────────────┐    │  │     MockE5Mistral       │ │   │
│  │  Embedding       │    │  │     (Docs: 92%)        │ │   │
│  │    Cache         │────┼──┤                         │ │   │
│  │                  │    │  └─────────────────────────┘ │   │
│  │ • LRU Eviction   │    │                             │   │
│  │ • Memory Limits  │    │  ┌─────────────────────────┐ │   │
│  │ • Persistence    │    │  │      MockBGE_M3         │ │   │
│  └──────────────────┘    │  │   (Comments: 86%)      │ │   │
│                          │  │    (Local Model)        │ │   │
│  ┌──────────────────┐    │  └─────────────────────────┘ │   │
│  │   Pipeline       │    │                             │   │
│  │  Coordinator     │    │  ┌─────────────────────────┐ │   │
│  │                  │    │  │    MockCodeBERT         │ │   │
│  │ • Batch Process  │────┼──┤  (Identifiers: 89%)     │ │   │
│  │ • Error Handling │    │  └─────────────────────────┘ │   │
│  │ • Metrics        │    │                             │   │
│  └──────────────────┘    │  ┌─────────────────────────┐ │   │
│                          │  │    MockSQLCoder         │ │   │
│                          │  │     (SQL: 91%)          │ │   │
│                          │  └─────────────────────────┘ │   │
│                          │                             │   │
│                          │  ┌─────────────────────────┐ │   │
│                          │  │   MockBERTConfig        │ │   │
│                          │  │   (Config: 88%)         │ │   │
│                          │  └─────────────────────────┘ │   │
│                          │                             │   │
│                          │  ┌─────────────────────────┐ │   │
│                          │  │ MockStackTraceBERT      │ │   │
│                          │  │   (Errors: 90%)         │ │   │
│                          │  └─────────────────────────┘ │   │
│                          └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### Interface Contracts
```rust
// Primary interfaces that all mocks must implement
pub trait EmbeddingService: Send + Sync {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    fn dimension_count(&self) -> usize;
    fn max_token_length(&self) -> usize;
    fn service_name(&self) -> &'static str;
    fn expected_accuracy(&self) -> f32;
}

pub trait ContentTypeDetector: Send + Sync {
    fn detect_content_type(&self, content: &str, file_path: &Path) -> ContentType;
    fn confidence_score(&self, content: &str, file_path: &Path) -> f32;
}

pub trait EmbeddingCache: Send + Sync {
    fn get(&self, key: &str) -> Option<Vec<f32>>;
    fn put(&self, key: String, embedding: Vec<f32>);
    fn evict_expired(&mut self);
    fn cache_stats(&self) -> CacheStats;
}
```

### Refinement

#### Mock Implementation Details
```rust
// All mocks return deterministic embeddings for testing
pub struct MockVoyageCode2 {
    dimension: usize, // 1024
    accuracy: f32,    // 0.93
}

impl EmbeddingService for MockVoyageCode2 {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Deterministic mock embedding based on content hash
        let hash = calculate_content_hash(text);
        let embedding = generate_deterministic_embedding(hash, self.dimension);
        
        // Simulate API latency
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        Ok(embedding)
    }
    
    fn dimension_count(&self) -> usize { self.dimension }
    fn service_name(&self) -> &'static str { "MockVoyageCode2" }
    fn expected_accuracy(&self) -> f32 { self.accuracy }
}

// Similar implementations for all other mock services...
```

#### Error Handling Strategy
```rust
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Service unavailable: {service}")]
    ServiceUnavailable { service: String },
    
    #[error("Content too long: {length} > {max_length}")]
    ContentTooLong { length: usize, max_length: usize },
    
    #[error("Invalid content type: {content_type:?}")]
    InvalidContentType { content_type: ContentType },
    
    #[error("Cache error: {source}")]
    CacheError { #[from] source: CacheError },
    
    #[error("Network error: {source}")]
    NetworkError { #[from] source: reqwest::Error },
}

pub struct ErrorRecoveryStrategy {
    max_retries: usize,
    backoff_strategy: ExponentialBackoff,
    fallback_service: Box<dyn EmbeddingService>,
}
```

### Completion

#### Integration Testing Strategy
```rust
pub struct EmbeddingSystemValidator {
    test_cases: Vec<EmbeddingTestCase>,
    accuracy_thresholds: HashMap<ContentType, f32>,
    performance_targets: PerformanceTargets,
}

pub struct EmbeddingTestCase {
    content: String,
    file_path: PathBuf,
    expected_content_type: ContentType,
    expected_embedding_service: &'static str,
    ground_truth_embedding: Option<Vec<f32>>,
}
```

## Implementation Tasks (300-399)

### Phase 3.1: Mock Infrastructure (Days 1-4)

#### Task 300: Content Type Detection Foundation
**Type**: Mock Creation  
**Duration**: 30 minutes  
**Dependencies**: None

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[test]
   fn test_content_type_detector_fails_without_implementation() {
       let detector = MockContentTypeDetector::new();
       let content = "pub fn test() {}";
       let path = Path::new("test.rs");
       
       // Should fail - no implementation yet
       assert!(detector.detect_content_type(content, path).is_err());
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MockContentTypeDetector;
   
   impl ContentTypeDetector for MockContentTypeDetector {
       fn detect_content_type(&self, content: &str, file_path: &Path) -> ContentType {
           // Minimal implementation to pass test
           match file_path.extension().and_then(|s| s.to_str()) {
               Some("rs") => ContentType::RustCode,
               _ => ContentType::Unknown,
           }
       }
   }
   ```

3. **REFACTOR Phase**
   - Add comprehensive pattern matching
   - Implement confidence scoring
   - Add validation for edge cases

##### Verification
- [ ] Mock returns correct content types for basic cases
- [ ] Pattern matching works for code structures
- [ ] File extension detection is accurate
- [ ] Unknown content falls back properly

#### Task 301: MockVoyageCode2 Service
**Type**: Mock Creation  
**Duration**: 45 minutes  
**Dependencies**: Task 300

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_voyage_code2_embedding_fails_initially() {
       let service = MockVoyageCode2::new();
       let result = service.generate_embedding("pub fn test() {}").await;
       
       // Should fail until implemented
       assert!(result.is_err());
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MockVoyageCode2 {
       dimension: usize,
       accuracy: f32,
   }
   
   impl EmbeddingService for MockVoyageCode2 {
       async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
           // Deterministic mock embedding
           let hash = self.hash_content(text);
           Ok(self.generate_mock_embedding(hash))
       }
   }
   ```

3. **REFACTOR Phase**
   - Add proper error handling
   - Implement cache key generation
   - Add latency simulation
   - Validate dimension consistency

##### Verification
- [ ] Returns 1024-dimensional embeddings
- [ ] Embedding generation is deterministic
- [ ] Service name and accuracy are correct
- [ ] Async interface works properly

#### Task 302: MockE5Mistral Documentation Service
**Type**: Mock Creation  
**Duration**: 45 minutes  
**Dependencies**: Task 301

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_e5_mistral_documentation_embedding() {
       let service = MockE5Mistral::new();
       let doc_content = "# API Documentation\nThis function processes data...";
       
       let result = service.generate_embedding(doc_content).await;
       assert!(result.is_err()); // Fails until implemented
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MockE5Mistral {
       dimension: usize, // 4096 for E5-Mistral
       accuracy: f32,    // 0.92
   }
   
   impl EmbeddingService for MockE5Mistral {
       async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
           // Mock implementation optimized for documentation
           let processed_text = self.preprocess_documentation(text);
           let embedding = self.generate_doc_optimized_embedding(&processed_text);
           Ok(embedding)
       }
   }
   ```

3. **REFACTOR Phase**
   - Add documentation-specific preprocessing
   - Implement markdown parsing awareness
   - Add section importance weighting
   - Optimize for technical documentation

##### Verification
- [ ] Returns 4096-dimensional embeddings
- [ ] Handles markdown formatting correctly
- [ ] Documentation sections are weighted properly
- [ ] Technical terms are emphasized

#### Task 303: MockBGE_M3 Local Comments Service
**Type**: Mock Creation  
**Duration**: 60 minutes  
**Dependencies**: Task 302

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_bge_m3_comment_processing() {
       let service = MockBGE_M3::new();
       let comment = "// This function calculates the similarity score between vectors";
       
       let result = service.generate_embedding(comment).await;
       assert!(result.is_err()); // No implementation yet
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MockBGE_M3 {
       dimension: usize, // 1024
       accuracy: f32,    // 0.86
       local_model: bool, // true - runs locally
   }
   
   impl EmbeddingService for MockBGE_M3 {
       async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
           // Simulate local processing (faster, no network)
           let cleaned_comment = self.clean_comment_text(text);
           let embedding = self.generate_local_embedding(&cleaned_comment);
           Ok(embedding)
       }
   }
   ```

3. **REFACTOR Phase**
   - Add comment syntax cleaning
   - Implement local processing simulation
   - Add multi-language comment support
   - Optimize for short text snippets

##### Verification
- [ ] Processes comment syntax correctly
- [ ] Local processing is faster than API calls
- [ ] Handles multiple comment formats
- [ ] Works with inline and block comments

#### Task 304: MockCodeBERT Identifier Service
**Type**: Mock Creation  
**Duration**: 45 minutes  
**Dependencies**: Task 303

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_codebert_identifier_embedding() {
       let service = MockCodeBERT::new();
       let identifiers = "calculate_similarity_score process_data_async";
       
       let result = service.generate_embedding(identifiers).await;
       assert!(result.is_err());
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MockCodeBERT {
       dimension: usize, // 768
       accuracy: f32,    // 0.89
   }
   
   impl EmbeddingService for MockCodeBERT {
       async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
           let identifiers = self.extract_identifiers(text);
           let embedding = self.generate_identifier_embedding(&identifiers);
           Ok(embedding)
       }
   }
   ```

3. **REFACTOR Phase**
   - Add identifier extraction logic
   - Implement camelCase/snake_case normalization
   - Add semantic meaning inference
   - Handle abbreviated identifiers

##### Verification
- [ ] Extracts identifiers correctly
- [ ] Normalizes naming conventions
- [ ] Handles function and variable names
- [ ] Works with abbreviated names

#### Task 305: MockSQLCoder Query Service
**Type**: Mock Creation  
**Duration**: 45 minutes  
**Dependencies**: Task 304

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_sqlcoder_query_embedding() {
       let service = MockSQLCoder::new();
       let sql = "SELECT users.name, COUNT(orders.id) FROM users JOIN orders ON users.id = orders.user_id GROUP BY users.id";
       
       let result = service.generate_embedding(sql).await;
       assert!(result.is_err());
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MockSQLCoder {
       dimension: usize, // 1536
       accuracy: f32,    // 0.91
   }
   
   impl EmbeddingService for MockSQLCoder {
       async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
           let parsed_sql = self.parse_sql_structure(text);
           let embedding = self.generate_sql_embedding(&parsed_sql);
           Ok(embedding)
       }
   }
   ```

3. **REFACTOR Phase**
   - Add SQL parsing and structure analysis
   - Implement table and column recognition
   - Add query pattern classification
   - Handle complex joins and subqueries

##### Verification
- [ ] Parses SQL syntax correctly
- [ ] Recognizes table and column references
- [ ] Handles different SQL dialects
- [ ] Works with complex query structures

#### Task 306: MockBERTConfig File Service
**Type**: Mock Creation  
**Duration**: 45 minutes  
**Dependencies**: Task 305

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_bert_config_file_embedding() {
       let service = MockBERTConfig::new();
       let config = r#"
       [package]
       name = "vector-search"
       version = "0.1.0"
       
       [dependencies]
       tantivy = "0.21"
       "#;
       
       let result = service.generate_embedding(config).await;
       assert!(result.is_err());
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MockBERTConfig {
       dimension: usize, // 768
       accuracy: f32,    // 0.88
   }
   
   impl EmbeddingService for MockBERTConfig {
       async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
           let config_structure = self.parse_config_format(text);
           let embedding = self.generate_config_embedding(&config_structure);
           Ok(embedding)
       }
   }
   ```

3. **REFACTOR Phase**
   - Add multi-format config parsing (TOML, JSON, YAML)
   - Implement key-value pair importance weighting
   - Add configuration section recognition
   - Handle nested configuration structures

##### Verification
- [ ] Parses TOML, JSON, and YAML formats
- [ ] Recognizes configuration sections
- [ ] Weights important configuration keys
- [ ] Handles nested structures properly

#### Task 307: MockStackTraceBERT Error Service
**Type**: Mock Creation  
**Duration**: 45 minutes  
**Dependencies**: Task 306

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_stacktrace_bert_error_embedding() {
       let service = MockStackTraceBERT::new();
       let error = r#"
       thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 5'
       src/main.rs:42:9
       stack backtrace:
         0: rust_begin_unwind
         1: core::panicking::panic_fmt
         2: main::process_data
       "#;
       
       let result = service.generate_embedding(error).await;
       assert!(result.is_err());
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MockStackTraceBERT {
       dimension: usize, // 1024
       accuracy: f32,    // 0.90
   }
   
   impl EmbeddingService for MockStackTraceBERT {
       async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
           let parsed_trace = self.parse_stack_trace(text);
           let embedding = self.generate_error_embedding(&parsed_trace);
           Ok(embedding)
       }
   }
   ```

3. **REFACTOR Phase**
   - Add stack trace parsing for multiple languages
   - Implement error type classification
   - Add source location importance weighting
   - Handle different stack trace formats

##### Verification
- [ ] Parses stack traces from multiple languages
- [ ] Extracts error types and messages
- [ ] Identifies source file locations
- [ ] Handles truncated or partial traces

#### Task 308: Embedding Cache Implementation
**Type**: Mock Creation  
**Duration**: 60 minutes  
**Dependencies**: Task 307

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[test]
   fn test_embedding_cache_stores_and_retrieves() {
       let mut cache = EmbeddingCache::new(100, Duration::from_secs(3600));
       let embedding = vec![1.0, 2.0, 3.0];
       
       cache.put("test_key".to_string(), embedding.clone());
       let retrieved = cache.get("test_key");
       
       assert!(retrieved.is_none()); // No implementation yet
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct EmbeddingCache {
       cache: HashMap<String, CacheEntry>,
       max_entries: usize,
       ttl: Duration,
   }
   
   impl EmbeddingCache {
       pub fn get(&self, key: &str) -> Option<Vec<f32>> {
           self.cache.get(key)
               .filter(|entry| !entry.is_expired())
               .map(|entry| entry.embedding.clone())
       }
       
       pub fn put(&mut self, key: String, embedding: Vec<f32>) {
           if self.cache.len() >= self.max_entries {
               self.evict_oldest();
           }
           
           self.cache.insert(key, CacheEntry::new(embedding));
       }
   }
   ```

3. **REFACTOR Phase**
   - Add LRU eviction policy
   - Implement memory usage tracking
   - Add cache hit/miss statistics
   - Handle concurrent access safely

##### Verification
- [ ] Stores and retrieves embeddings correctly
- [ ] Expires entries based on TTL
- [ ] Evicts oldest entries when full
- [ ] Tracks cache statistics accurately

### Phase 3.2: Pipeline Integration (Days 5-8)

#### Task 309: Multi-Embedding Pipeline Coordinator
**Type**: Integration  
**Duration**: 90 minutes  
**Dependencies**: Tasks 300-308

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_pipeline_coordinates_multiple_services() {
       let pipeline = EmbeddingPipeline::new();
       let document = Document {
           path: PathBuf::from("test.rs"),
           content: "pub fn test() {} // This is a comment".to_string(),
       };
       
       let result = pipeline.process_document(document).await;
       assert!(result.is_err()); // No coordination logic yet
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct EmbeddingPipeline {
       content_detector: Box<dyn ContentTypeDetector>,
       embedding_services: HashMap<ContentType, Box<dyn EmbeddingService>>,
       cache: EmbeddingCache,
   }
   
   impl EmbeddingPipeline {
       pub async fn process_document(&self, document: Document) -> Result<EmbeddedDocument, EmbeddingError> {
           let chunks = self.chunk_document(&document);
           let mut embedded_chunks = Vec::new();
           
           for chunk in chunks {
               let content_type = self.content_detector.detect_content_type(&chunk.content, &document.path);
               let service = self.get_service_for_type(&content_type)?;
               let embedding = service.generate_embedding(&chunk.content).await?;
               
               embedded_chunks.push(EmbeddedChunk {
                   content: chunk.content,
                   embedding,
                   content_type,
                   chunk_index: chunk.index,
               });
           }
           
           Ok(EmbeddedDocument {
               document,
               chunks: embedded_chunks,
           })
       }
   }
   ```

3. **REFACTOR Phase**
   - Add parallel processing for chunks
   - Implement batch processing optimization
   - Add error recovery for failed chunks
   - Include progress tracking for large documents

##### Verification
- [ ] Processes documents with multiple content types
- [ ] Routes chunks to appropriate embedding services
- [ ] Handles processing failures gracefully
- [ ] Returns complete embedded documents

#### Task 310: Content Type Detection Enhancement
**Type**: Implementation  
**Duration**: 75 minutes  
**Dependencies**: Task 309

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[test]
   fn test_advanced_content_detection() {
       let detector = ContentTypeDetector::new();
       
       // SQL in Rust file should be detected as SQL
       let content = r#"
       pub fn query_users() -> String {
           "SELECT * FROM users WHERE active = true".to_string()
       }
       "#;
       
       let detected = detector.detect_content_type(content, Path::new("database.rs"));
       assert_eq!(detected, ContentType::SqlQueries); // Will fail initially
   }
   ```

2. **GREEN Phase**
   ```rust
   impl ContentTypeDetector {
       fn detect_content_type(&self, content: &str, file_path: &Path) -> ContentType {
           // Check for SQL patterns first (highest priority)
           if self.contains_sql_patterns(content) {
               return ContentType::SqlQueries;
           }
           
           // Check for error traces
           if self.contains_stack_trace_patterns(content) {
               return ContentType::ErrorTraces;
           }
           
           // File extension based detection
           match file_path.extension().and_then(|s| s.to_str()) {
               Some("rs") | Some("py") | Some("js") => ContentType::RustCode,
               Some("md") | Some("txt") => ContentType::Documentation,
               Some("toml") | Some("json") | Some("yaml") => ContentType::ConfigFiles,
               _ => ContentType::Unknown,
           }
       }
   }
   ```

3. **REFACTOR Phase**
   - Add confidence scoring for detection
   - Implement multi-pattern detection for mixed content
   - Add support for embedded content types
   - Include language-specific pattern recognition

##### Verification
- [ ] Detects SQL patterns in code files
- [ ] Recognizes stack traces in logs
- [ ] Handles mixed content appropriately
- [ ] Provides confidence scores for detection

#### Task 311: Batch Processing Optimization
**Type**: Performance  
**Duration**: 90 minutes  
**Dependencies**: Task 310

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_batch_processing_efficiency() {
       let pipeline = EmbeddingPipeline::new();
       let documents = create_test_documents(100);
       
       let start = Instant::now();
       let results = pipeline.process_documents_batch(documents).await;
       let duration = start.elapsed();
       
       assert!(results.is_err()); // No batch processing yet
   }
   ```

2. **GREEN Phase**
   ```rust
   impl EmbeddingPipeline {
       pub async fn process_documents_batch(&self, documents: Vec<Document>) -> Result<Vec<EmbeddedDocument>, EmbeddingError> {
           let batch_size = 10;
           let mut results = Vec::new();
           
           for batch in documents.chunks(batch_size) {
               let batch_futures: Vec<_> = batch.iter()
                   .map(|doc| self.process_document(doc.clone()))
                   .collect();
               
               let batch_results = futures::future::try_join_all(batch_futures).await?;
               results.extend(batch_results);
           }
           
           Ok(results)
       }
   }
   ```

3. **REFACTOR Phase**
   - Add dynamic batch sizing based on content
   - Implement memory-aware processing
   - Add progress reporting for large batches
   - Include rate limiting for API services

##### Verification
- [ ] Processes documents in parallel batches
- [ ] Respects memory constraints
- [ ] Provides progress updates
- [ ] Handles batch failures appropriately

#### Task 312: Error Recovery and Fallback System
**Type**: Robustness  
**Duration**: 60 minutes  
**Dependencies**: Task 311

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_fallback_on_service_failure() {
       let mut pipeline = EmbeddingPipeline::new();
       
       // Configure a service to fail
       pipeline.configure_service_failure("MockVoyageCode2");
       
       let document = Document {
           path: PathBuf::from("test.rs"),
           content: "pub fn test() {}".to_string(),
       };
       
       let result = pipeline.process_document(document).await;
       assert!(result.is_ok()); // Should succeed with fallback
       assert!(result.is_err()); // But it will fail without fallback logic
   }
   ```

2. **GREEN Phase**
   ```rust
   impl EmbeddingPipeline {
       async fn generate_embedding_with_fallback(&self, content: &str, content_type: ContentType) -> Result<Vec<f32>, EmbeddingError> {
           let primary_service = self.get_service_for_type(&content_type)?;
           
           match primary_service.generate_embedding(content).await {
               Ok(embedding) => Ok(embedding),
               Err(e) => {
                   warn!("Primary service failed: {}, using fallback", e);
                   self.fallback_service.generate_embedding(content).await
               }
           }
       }
   }
   ```

3. **REFACTOR Phase**
   - Add retry logic with exponential backoff
   - Implement circuit breaker pattern
   - Add service health monitoring
   - Include detailed error reporting

##### Verification
- [ ] Falls back to alternative services on failure
- [ ] Implements retry logic for transient failures
- [ ] Maintains service health status
- [ ] Reports detailed error information

### Phase 3.3: Similarity Search Infrastructure (Days 9-12)

#### Task 313: Vector Similarity Search Engine
**Type**: Core Implementation  
**Duration**: 120 minutes  
**Dependencies**: Task 312

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_similarity_search_with_multiple_embeddings() {
       let search_engine = SimilaritySearchEngine::new();
       
       // Index documents with different embedding types
       let documents = vec![
           EmbeddedDocument::with_code_embedding("test.rs", "pub fn test() {}"),
           EmbeddedDocument::with_doc_embedding("readme.md", "# Test Documentation"),
       ];
       
       search_engine.index_documents(documents).await?;
       
       let results = search_engine.search_similar("function definition", 5).await;
       assert!(results.is_err()); // No search implementation yet
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct SimilaritySearchEngine {
       vector_store: TransactionalVectorStore,
       embedding_pipeline: EmbeddingPipeline,
   }
   
   impl SimilaritySearchEngine {
       pub async fn search_similar(&self, query: &str, limit: usize) -> Result<Vec<SimilarityResult>, SearchError> {
           // Generate query embedding using appropriate service
           let query_content_type = self.detect_query_type(query);
           let query_embedding = self.embedding_pipeline.generate_embedding(query, query_content_type).await?;
           
           // Search in vector store
           let results = self.vector_store.similarity_search(&query_embedding, limit).await?;
           
           // Convert to similarity results with scores
           Ok(results.into_iter().map(|r| SimilarityResult {
               document_id: r.id,
               file_path: r.file_path,
               content: r.content,
               similarity_score: r.score,
               content_type: r.content_type,
           }).collect())
       }
   }
   ```

3. **REFACTOR Phase**
   - Add multi-embedding search fusion
   - Implement result ranking optimization
   - Add similarity threshold filtering
   - Include content type weighting

##### Verification
- [ ] Searches across multiple embedding types
- [ ] Returns relevance-ranked results
- [ ] Handles different query types appropriately
- [ ] Provides accurate similarity scores

#### Task 314: Multi-Embedding Search Fusion
**Type**: Advanced Search  
**Duration**: 90 minutes  
**Dependencies**: Task 313

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_fusion_search_combines_results() {
       let fusion_engine = MultiEmbeddingFusionEngine::new();
       
       let query = "calculate similarity between code functions";
       
       // Should search with multiple embedding services and fuse results
       let results = fusion_engine.search_with_fusion(query, 10).await;
       assert!(results.is_err()); // No fusion logic yet
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MultiEmbeddingFusionEngine {
       similarity_engines: HashMap<ContentType, SimilaritySearchEngine>,
   }
   
   impl MultiEmbeddingFusionEngine {
       pub async fn search_with_fusion(&self, query: &str, limit: usize) -> Result<Vec<FusedResult>, SearchError> {
           let mut all_results = Vec::new();
           
           // Search with each embedding type
           for (content_type, engine) in &self.similarity_engines {
               let type_results = engine.search_similar(query, limit).await?;
               for result in type_results {
                   all_results.push(FusedResult {
                       result,
                       source_embedding: *content_type,
                       fusion_score: self.calculate_fusion_score(&result, content_type),
                   });
               }
           }
           
           // Apply reciprocal rank fusion
           self.apply_reciprocal_rank_fusion(&mut all_results);
           all_results.truncate(limit);
           
           Ok(all_results)
       }
   }
   ```

3. **REFACTOR Phase**
   - Implement sophisticated fusion algorithms
   - Add query-type-specific weighting
   - Include confidence-based result merging
   - Add deduplication across embedding types

##### Verification
- [ ] Combines results from multiple embedding types
- [ ] Applies reciprocal rank fusion correctly
- [ ] Weights results based on query relevance
- [ ] Removes duplicate results across embeddings

#### Task 315: Caching and Performance Optimization
**Type**: Performance  
**Duration**: 75 minutes  
**Dependencies**: Task 314

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_search_result_caching() {
       let mut engine = CachedSimilarityEngine::new();
       
       let query = "test function implementation";
       
       // First search - should be slow
       let start = Instant::now();
       let results1 = engine.search_with_cache(query, 5).await?;
       let first_duration = start.elapsed();
       
       // Second search - should be fast (cached)
       let start = Instant::now();
       let results2 = engine.search_with_cache(query, 5).await?;
       let cached_duration = start.elapsed();
       
       assert!(cached_duration < first_duration / 2); // Will fail without caching
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct CachedSimilarityEngine {
       inner_engine: MultiEmbeddingFusionEngine,
       result_cache: HashMap<String, (Vec<FusedResult>, Instant)>,
       cache_ttl: Duration,
   }
   
   impl CachedSimilarityEngine {
       pub async fn search_with_cache(&mut self, query: &str, limit: usize) -> Result<Vec<FusedResult>, SearchError> {
           let cache_key = format!("{}:{}", query, limit);
           
           // Check cache first
           if let Some((cached_results, timestamp)) = self.result_cache.get(&cache_key) {
               if timestamp.elapsed() < self.cache_ttl {
                   return Ok(cached_results.clone());
               }
           }
           
           // Perform actual search
           let results = self.inner_engine.search_with_fusion(query, limit).await?;
           
           // Cache results
           self.result_cache.insert(cache_key, (results.clone(), Instant::now()));
           
           Ok(results)
       }
   }
   ```

3. **REFACTOR Phase**
   - Add LRU cache eviction
   - Implement memory-aware caching
   - Add cache warming strategies
   - Include cache hit/miss metrics

##### Verification
- [ ] Caches search results effectively
- [ ] Evicts old entries appropriately
- [ ] Provides significant performance improvement
- [ ] Maintains result accuracy with caching

#### Task 316: Integration Testing and Validation
**Type**: Validation  
**Duration**: 120 minutes  
**Dependencies**: Task 315

##### TDD Cycle
1. **RED Phase**
   ```rust
   #[tokio::test]
   async fn test_end_to_end_multi_embedding_system() {
       let system = MultiEmbeddingSystem::new().await?;
       
       // Index various content types
       let documents = vec![
           Document::rust_code("src/lib.rs", "pub fn calculate_similarity() -> f32 {}"),
           Document::documentation("README.md", "# Similarity Calculator\nThis library provides..."),
           Document::config("Cargo.toml", "[package]\nname = \"similarity\""),
           Document::sql("queries.sql", "SELECT * FROM embeddings WHERE similarity > 0.8"),
       ];
       
       system.index_documents(documents).await?;
       
       // Test multi-type search
       let results = system.search("similarity calculation function", 10).await;
       
       // Should find results from multiple content types with appropriate ranking
       assert!(results.len() > 0);
       assert!(results.iter().any(|r| r.content_type == ContentType::RustCode));
       assert!(results.iter().any(|r| r.content_type == ContentType::Documentation));
       
       // This will fail until full integration is complete
       assert!(false, "Integration not complete");
   }
   ```

2. **GREEN Phase**
   ```rust
   pub struct MultiEmbeddingSystem {
       embedding_pipeline: EmbeddingPipeline,
       search_engine: CachedSimilarityEngine,
       vector_store: TransactionalVectorStore,
   }
   
   impl MultiEmbeddingSystem {
       pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<UnifiedSearchResult>, SystemError> {
           // Use the cached similarity engine for search
           let fusion_results = self.search_engine.search_with_cache(query, limit).await?;
           
           // Convert to unified results
           Ok(fusion_results.into_iter().map(|r| UnifiedSearchResult {
               file_path: r.result.file_path,
               content: r.result.content,
               similarity_score: r.fusion_score,
               content_type: r.result.content_type,
               embedding_source: r.source_embedding,
           }).collect())
       }
   }
   ```

3. **REFACTOR Phase**
   - Add comprehensive error handling
   - Implement system health monitoring
   - Add performance metrics collection
   - Include detailed logging and tracing

##### Verification
- [ ] Indexes documents of all supported types
- [ ] Searches across all embedding services
- [ ] Returns properly ranked and fused results
- [ ] Handles system errors gracefully
- [ ] Achieves target accuracy for each content type

## Success Metrics

### Functional Requirements
- [ ] Content type detection accuracy > 95%
- [ ] All 7 embedding services implemented as mocks
- [ ] Pipeline processes mixed content documents
- [ ] Similarity search works across all embedding types
- [ ] Result fusion improves overall accuracy
- [ ] Caching provides > 50% performance improvement

### Performance Targets
- [ ] Content type detection < 1ms per document
- [ ] Embedding generation < 100ms per chunk (mocked)
- [ ] Similarity search < 200ms for 10 results
- [ ] Batch processing > 100 documents/minute
- [ ] Cache hit rate > 80% for repeated queries
- [ ] Memory usage < 2GB for 100K embeddings

### Quality Gates
- [ ] 100% test coverage for all components
- [ ] All mocks implement identical interfaces
- [ ] Error handling covers all failure modes
- [ ] Documentation complete for all APIs
- [ ] Integration tests pass for all scenarios
- [ ] Performance benchmarks meet targets

### Accuracy Validation
- [ ] Code embedding accuracy: 93% (VoyageCode2)
- [ ] Documentation embedding accuracy: 92% (E5Mistral)
- [ ] Comment embedding accuracy: 86% (BGE_M3)
- [ ] Identifier embedding accuracy: 89% (CodeBERT)
- [ ] SQL embedding accuracy: 91% (SQLCoder)
- [ ] Config embedding accuracy: 88% (BERTConfig)
- [ ] Error embedding accuracy: 90% (StackTraceBERT)

## Risk Mitigation

### Technical Risks
- **Mock-to-Real Replacement**: Each mock implements exact interface for seamless replacement
- **Embedding Dimension Mismatches**: Strict dimension validation in all interfaces
- **Performance Degradation**: Comprehensive benchmarking at each integration step
- **Memory Constraints**: Memory-aware processing and cache management

### Integration Risks
- **Service Compatibility**: Mock services validate against real API specifications
- **Content Type Errors**: Fallback to generic embedding on detection failures
- **Batch Processing Failures**: Graceful degradation with partial results
- **Cache Consistency**: TTL-based invalidation and health checks

## Next Phase

With the multi-embedding system complete and all mocks validated, proceed to **Phase 4: Temporal Analysis** for Git history integration and regression detection capabilities.

---

*Phase 3 establishes the foundation for specialized content understanding through mock-first development, ensuring each embedding service can be tested and validated independently before real API integration.*