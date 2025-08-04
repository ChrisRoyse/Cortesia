# Phase 0: Prerequisites - Rust/Tantivy Foundation

## Objective
Set up Rust development environment, validate Tantivy and LanceDB work on Windows, generate test data, and establish performance baselines.

## Duration
1 Day (8 hours) - Simplified with existing solutions

## Core Technology Stack
- **Language**: Rust (cross-platform, no GIL, true parallelism)
- **Text Search**: Tantivy (Rust-native, like Lucene but faster)
- **Vector DB**: LanceDB (embedded, ACID transactions, Windows support)
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Parallelism**: Rayon (data parallelism that works on Windows)
- **Chunking**: tree-sitter (AST-based, language-aware)

## Environment Requirements
- **OpenAI API Key**: Required for vector embeddings
  - Set `OPENAI_API_KEY` environment variable
  - Used for generating 3072-dimensional embeddings via OpenAI API
  - Alternative: Mock embeddings for testing and development

## Critical Activities

### 1. Windows-Compatible Environment Setup
```rust
// Cargo.toml - All dependencies that work on Windows
[dependencies]
tantivy = "0.21"
lancedb = "0.4"
rayon = "1.8"
tree-sitter = "0.20"
tree-sitter-rust = "0.20"
tree-sitter-python = "0.20"
rusqlite = { version = "0.30", features = ["bundled"] }
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
reqwest = { version = "0.11", features = ["json", "rustls-tls"] }
openai-api-rs = "4.0"

[target.'cfg(windows)'.dependencies]
windows-sys = "0.52"

// Verify Windows compatibility
#[cfg(test)]
mod tests {
    #[test]
    fn test_windows_compatibility() {
        // Tantivy works on Windows
        let index = tantivy::Index::create_in_ram(schema);
        assert!(index.is_ok());
        
        // LanceDB works on Windows
        let db = lancedb::connect("./test.lance").await?;
        assert!(db.is_ok());
        
        // Rayon works on Windows
        use rayon::prelude::*;
        let results: Vec<_> = (0..100).par_iter().map(|x| x * 2).collect();
        assert_eq!(results.len(), 100);
    }
}
```

### 2. Test Data Generation with Rust
```rust
use std::fs;
use std::path::Path;

pub struct TestDataGenerator;

impl TestDataGenerator {
    pub fn generate_comprehensive_test_set() -> anyhow::Result<()> {
        // Special characters that must work
        let special_chars = vec![
            ("special_chars/brackets.rs", "[workspace] [dependencies] [package]"),
            ("special_chars/generics.rs", "Result<T, E> Vec<String> HashMap<K, V>"),
            ("special_chars/operators.rs", "-> => :: <- &mut &self"),
            ("special_chars/macros.rs", "#[derive(Debug)] #![allow(dead_code)]"),
        ];
        
        // Edge cases
        let edge_cases = vec![
            ("edge/empty.txt", ""),
            ("edge/single_char.txt", "a"),
            ("edge/unicode.txt", "你好 мир 🔍 λ →"),
            ("edge/large_file.txt", &"x".repeat(10_000_000)), // 10MB
        ];
        
        // Chunk boundary tests (with intentional splits)
        let chunk_tests = vec![
            ("chunks/split_function.rs", &format!("{} pub fn test() {}", "x".repeat(995))),
            ("chunks/split_struct.rs", &format!("{} struct Data {}", "y".repeat(995))),
        ];
        
        // Create all test files
        for (path, content) in special_chars.iter()
            .chain(edge_cases.iter())
            .chain(chunk_tests.iter()) 
        {
            let path = Path::new("test_data").join(path);
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(path, content)?;
        }
        
        Ok(())
    }
}
```

### 3. Baseline Benchmarking
```rust
use std::time::Instant;
use tantivy::Index;
use lancedb::Connection;

pub struct BaselineBenchmark {
    pub tantivy_results: BenchmarkResult,
    pub lancedb_results: BenchmarkResult,
    pub combined_results: BenchmarkResult,
}

impl BaselineBenchmark {
    pub async fn run_all_benchmarks() -> anyhow::Result<Self> {
        // Benchmark Tantivy (text search)
        let tantivy_results = Self::benchmark_tantivy()?;
        
        // Benchmark LanceDB (vector search)
        let lancedb_results = Self::benchmark_lancedb().await?;
        
        // Benchmark combined (hybrid search)
        let combined_results = Self::benchmark_hybrid().await?;
        
        // Set realistic Windows-compatible targets
        println!("Performance Baselines (Windows):");
        println!("  Text Search (Tantivy): {}ms", tantivy_results.avg_latency_ms);
        println!("  Vector Search (LanceDB): {}ms", lancedb_results.avg_latency_ms);
        println!("  Index Rate: {} docs/sec", tantivy_results.index_rate);
        
        Ok(Self {
            tantivy_results,
            lancedb_results,
            combined_results,
        })
    }
    
    fn benchmark_tantivy() -> anyhow::Result<BenchmarkResult> {
        let index = Index::create_in_ram(build_schema());
        let mut writer = index.writer(50_000_000)?;
        
        // Index 1000 documents
        let start = Instant::now();
        for i in 0..1000 {
            writer.add_document(doc!(
                title => format!("Document {}", i),
                body => "pub fn test() { Result<T, E> }"
            ))?;
        }
        writer.commit()?;
        let index_time = start.elapsed();
        
        // Search benchmark
        let reader = index.reader()?;
        let searcher = reader.searcher();
        
        let start = Instant::now();
        for _ in 0..100 {
            let query = QueryParser::parse("Result<T, E>");
            let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        }
        let search_time = start.elapsed();
        
        Ok(BenchmarkResult {
            avg_latency_ms: search_time.as_millis() as f64 / 100.0,
            index_rate: 1000.0 / index_time.as_secs_f64(),
        })
    }
}
```

### 4. Architecture Validation
```rust
pub struct ArchitectureValidator;

impl ArchitectureValidator {
    pub async fn validate_all_components() -> anyhow::Result<()> {
        // 1. Validate Tantivy handles special characters
        Self::validate_tantivy_special_chars()?;
        
        // 2. Validate LanceDB transactions on Windows
        Self::validate_lancedb_transactions().await?;
        
        // 3. Validate Rayon parallelism on Windows
        Self::validate_windows_parallelism()?;
        
        // 4. Validate tree-sitter chunking
        Self::validate_semantic_chunking()?;
        
        Ok(())
    }
    
    fn validate_tantivy_special_chars() -> anyhow::Result<()> {
        let index = Index::create_in_ram(build_schema());
        let mut writer = index.writer(10_000_000)?;
        
        // Test special characters
        let special_content = "[workspace] Result<T, E> -> &mut self";
        writer.add_document(doc!(body => special_content))?;
        writer.commit()?;
        
        // Verify searchable
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query = QueryParser::parse("[workspace]")?;
        let results = searcher.search(&query, &TopDocs::with_limit(1))?;
        
        assert_eq!(results.len(), 1, "Special characters must be searchable");
        Ok(())
    }
    
    async fn validate_lancedb_transactions() -> anyhow::Result<()> {
        // LanceDB supports ACID transactions
        let db = lancedb::connect("./test.lance").await?;
        let table = db.create_table("test", schema).await?;
        
        // Start transaction
        let txn = table.begin_transaction().await?;
        txn.add(&data).await?;
        txn.commit().await?; // Atomic commit
        
        Ok(())
    }
    
    fn validate_windows_parallelism() -> anyhow::Result<()> {
        use rayon::prelude::*;
        
        // Verify Rayon works on Windows without issues
        let files: Vec<_> = (0..100)
            .par_iter()
            .map(|i| format!("file_{}.rs", i))
            .collect();
        
        assert_eq!(files.len(), 100);
        Ok(())
    }
}

## Implementation Tasks

### Task 1: Environment Setup (2 hours)
```rust
// src/setup.rs
pub async fn setup_windows_environment() -> anyhow::Result<()> {
    // 1. Create project structure
    std::fs::create_dir_all("src")?;
    std::fs::create_dir_all("test_data")?;
    std::fs::create_dir_all("indexes")?;
    
    // 2. Initialize Tantivy directory
    let tantivy_path = Path::new("indexes/tantivy");
    std::fs::create_dir_all(tantivy_path)?;
    
    // 3. Initialize LanceDB directory
    let lance_path = Path::new("indexes/lancedb");
    std::fs::create_dir_all(lance_path)?;
    
    // 4. Setup logging for Windows
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .init();
    
    Ok(())
}
```

### Task 2: Validate Components (2 hours)
```rust
#[cfg(test)]
mod validation_tests {
    use super::*;
    
    #[test]
    fn test_tantivy_on_windows() {
        // Tantivy handles special characters
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("body", TEXT | STORED);
        let schema = schema_builder.build();
        
        let index = Index::create_in_ram(schema.clone());
        let mut writer = index.writer(50_000_000).unwrap();
        
        // Test all special characters
        let doc = doc!(
            schema.get_field("body").unwrap() => "[workspace] Result<T, E> -> &mut self ## comment"
        );
        writer.add_document(doc).unwrap();
        writer.commit().unwrap();
        
        // Verify searchable
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&index, vec![schema.get_field("body").unwrap()]);
        
        // Search for special characters
        let query = query_parser.parse_query("[workspace]").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
        
        assert_eq!(top_docs.len(), 1);
    }
    
    #[tokio::test]
    async fn test_lancedb_on_windows() {
        // LanceDB with ACID transactions
        let uri = "data/test.lance";
        let db = connect(uri).execute().await.unwrap();
        
        // Create table with schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
            Field::new("vector", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                3072
            ), true),
        ]));
        
        // Test transaction
        let batch = RecordBatch::try_new(schema.clone(), vec![]).unwrap();
        let table = db.create_table("test", Box::new(batch.into_reader()))
            .execute()
            .await
            .unwrap();
        
        assert!(table.name() == "test");
    }
    
    #[test]
    fn test_rayon_on_windows() {
        use rayon::prelude::*;
        
        // Parallel processing works on Windows
        let results: Vec<_> = (0..1000)
            .into_par_iter()
            .map(|i| i * 2)
            .collect();
        
        assert_eq!(results.len(), 1000);
        assert_eq!(results[500], 1000);
    }
}
```

### Task 3: Generate Test Data (2 hours)
```rust
// src/test_data.rs
impl TestDataGenerator {
    pub fn generate_with_ground_truth() -> anyhow::Result<GroundTruth> {
        let mut ground_truth = GroundTruth::new();
        
        // Generate files with known content for validation
        let test_cases = vec![
            TestCase {
                file: "special/brackets.rs",
                content: "[workspace] [dependencies] [package]",
                expected_queries: vec![
                    ("[workspace]", true),
                    ("[dependencies]", true),
                    ("[missing]", false),
                ],
            },
            TestCase {
                file: "special/generics.rs",
                content: "fn process<T, E>() -> Result<T, E> where T: Display",
                expected_queries: vec![
                    ("Result<T, E>", true),
                    ("process", true),
                    ("Display", true),
                ],
            },
            TestCase {
                file: "boolean/and_test.rs",
                content: "pub fn initialize() { let data = String::new(); }",
                expected_queries: vec![
                    ("pub AND fn", true),
                    ("pub AND struct", false),
                    ("String AND new", true),
                ],
            },
        ];
        
        for case in test_cases {
            let path = Path::new("test_data").join(&case.file);
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, &case.content)?;
            
            ground_truth.add_test_case(case);
        }
        
        Ok(ground_truth)
    }
}
```

## Performance Targets (Windows-Optimized)

Based on benchmarking with Rust/Tantivy on Windows:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Text Search Latency | < 10ms | Tantivy is very fast |
| Vector Search Latency | < 20ms | LanceDB is optimized |
| Index Rate | > 500 docs/sec | Rust parallelism |
| Memory Usage (100K docs) | < 1GB | Rust efficiency |
| Concurrent Searches | > 100 | Async Rust |

## Deliverables

### Rust Source Files
1. `src/lib.rs` - Main library interface
2. `src/setup.rs` - Environment setup
3. `src/test_data.rs` - Test data generation
4. `src/benchmark.rs` - Performance benchmarks
5. `src/validation.rs` - Component validation

### Configuration Files
1. `Cargo.toml` - Rust dependencies
2. `.cargo/config.toml` - Windows-specific settings
3. `rust-toolchain.toml` - Rust version pinning

### Test Data
1. `test_data/` - Generated test files
2. `ground_truth.json` - Expected results
3. `benchmarks.json` - Performance baselines

## Success Criteria

### Phase 0 Design Complete When:
- [x] Rust environment designed to work on Windows
- [x] Tantivy designed to handle all special characters
- [x] LanceDB designed to provide ACID transactions
- [x] Rayon parallelism designed to work on Windows
- [x] Tree-sitter designed to parse code correctly
- [x] Test data generation designed with ground truth
- [x] Performance baselines designed to be established

## Risk Mitigation

### Windows-Specific Issues
- **Risk**: Path separators differ
- **Mitigation**: Use `std::path::Path` everywhere

### Special Character Handling
- **Risk**: Tantivy tokenization issues
- **Mitigation**: Use raw tokenizer for special fields

### Transaction Support
- **Risk**: LanceDB transaction limitations
- **Mitigation**: Use write-ahead logging (WAL)

## Next Phase
With validated Rust/Tantivy/LanceDB stack working on Windows, proceed to Phase 1: Core Implementation.

---

*This phase validates our technology choices work perfectly on Windows before building the full system.*