# Micro-Task 046a: Create Similarity Search Validation Framework

## Objective
Create a validation framework to test similarity search accuracy and performance.

## Prerequisites
- Task 045e completed (vector storage persistence tests passing and committed)

## Time Estimate
8 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Create `src/validation.rs` with validation framework:
   ```rust
   //! Similarity search validation and testing framework
   
   use crate::mock_storage::{MockVectorStorage, VectorRecord, SimilarityResult, SearchConfig};
   use std::collections::HashMap;
   use uuid::Uuid;
   
   /// Test case for similarity search validation
   #[derive(Debug, Clone)]
   pub struct SimilarityTestCase {
       pub name: String,
       pub query_content: String,
       pub expected_results: Vec<String>, // Expected content snippets in order
       pub tolerance: f32, // Acceptable ranking difference
   }
   
   /// Validation results for a test case
   #[derive(Debug, Clone)]
   pub struct ValidationResult {
       pub test_case_name: String,
       pub passed: bool,
       pub score: f32, // 0.0 to 1.0
       pub found_results: Vec<String>,
       pub expected_results: Vec<String>,
       pub ranking_accuracy: f32,
       pub recall_at_k: f32,
       pub precision_at_k: f32,
   }
   
   /// Similarity search validator
   pub struct SimilarityValidator {
       storage: MockVectorStorage,
       test_cases: Vec<SimilarityTestCase>,
   }
   
   impl SimilarityValidator {
       pub fn new(dimension: usize) -> Self {
           Self {
               storage: MockVectorStorage::new(dimension),
               test_cases: Vec::new(),
           }
       }
       
       pub fn add_test_case(&mut self, test_case: SimilarityTestCase) {
           self.test_cases.push(test_case);
       }
       
       pub fn add_test_document(&self, content: &str, title: &str) -> Result<Uuid, String> {
           let record = VectorRecord::new(
               content.to_string(),
               title.to_string(),
               None,
               None,
           );
           let id = record.id;
           
           self.storage.insert(record)
               .map_err(|e| e.to_string())?;
           
           Ok(id)
       }
   }
   ```
3. Add validation module to `src/lib.rs`:
   ```rust
   pub mod validation;
   pub use validation::{SimilarityValidator, SimilarityTestCase, ValidationResult};
   ```
4. Test: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] Validation framework structure created
- [ ] SimilarityTestCase and ValidationResult defined
- [ ] SimilarityValidator with basic setup methods
- [ ] Module properly integrated

## Next Task
task_046b_implement_validation_metrics.md

---

# Micro-Task 046b: Implement Validation Metrics

## Objective
Implement ranking accuracy, recall, precision, and other validation metrics.

## Prerequisites
- Task 046a completed (similarity search validation framework created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add validation metrics to `src/validation.rs`:
   ```rust
   impl SimilarityValidator {
       /// Run validation on a single test case
       pub fn validate_test_case(&self, test_case: &SimilarityTestCase, k: usize) -> Result<ValidationResult, String> {
           let config = SearchConfig {
               limit: k,
               threshold: 0.0,
               include_metadata: true,
           };
           
           let results = self.storage.search_by_content(&test_case.query_content, &config)
               .map_err(|e| e.to_string())?;
           
           let found_results: Vec<String> = results.iter()
               .map(|r| r.record.content.clone())
               .collect();
           
           // Calculate metrics
           let ranking_accuracy = Self::calculate_ranking_accuracy(&found_results, &test_case.expected_results);
           let recall_at_k = Self::calculate_recall(&found_results, &test_case.expected_results);
           let precision_at_k = Self::calculate_precision(&found_results, &test_case.expected_results);
           
           // Overall score (weighted combination)
           let score = (ranking_accuracy * 0.4) + (recall_at_k * 0.3) + (precision_at_k * 0.3);
           let passed = score >= (1.0 - test_case.tolerance);
           
           Ok(ValidationResult {
               test_case_name: test_case.name.clone(),
               passed,
               score,
               found_results,
               expected_results: test_case.expected_results.clone(),
               ranking_accuracy,
               recall_at_k,
               precision_at_k,
           })
       }
       
       /// Calculate ranking accuracy (how well the order matches expected)
       fn calculate_ranking_accuracy(found: &[String], expected: &[String]) -> f32 {
           if expected.is_empty() {
               return 1.0;
           }
           
           let mut correct_positions = 0.0;
           let positions_to_check = expected.len().min(found.len());
           
           for (i, expected_item) in expected.iter().take(positions_to_check).enumerate() {
               if let Some(found_pos) = found.iter().position(|item| item.contains(expected_item)) {
                   // Give higher score for closer positions
                   let position_diff = (i as f32 - found_pos as f32).abs();
                   let position_score = 1.0 / (1.0 + position_diff);
                   correct_positions += position_score;
               }
           }
           
           correct_positions / expected.len() as f32
       }
       
       /// Calculate recall@k (how many expected items were found)
       fn calculate_recall(found: &[String], expected: &[String]) -> f32 {
           if expected.is_empty() {
               return 1.0;
           }
           
           let mut found_count = 0;
           for expected_item in expected {
               if found.iter().any(|item| item.contains(expected_item)) {
                   found_count += 1;
               }
           }
           
           found_count as f32 / expected.len() as f32
       }
       
       /// Calculate precision@k (how many found items were expected)
       fn calculate_precision(found: &[String], expected: &[String]) -> f32 {
           if found.is_empty() {
               return if expected.is_empty() { 1.0 } else { 0.0 };
           }
           
           let mut relevant_count = 0;
           for found_item in found {
               if expected.iter().any(|expected| found_item.contains(expected)) {
                   relevant_count += 1;
               }
           }
           
           relevant_count as f32 / found.len() as f32
       }
       
       /// Run all test cases and return aggregated results
       pub fn validate_all(&self, k: usize) -> Result<ValidationSummary, String> {
           let mut results = Vec::new();
           let mut total_score = 0.0;
           let mut passed_count = 0;
           
           for test_case in &self.test_cases {
               let result = self.validate_test_case(test_case, k)?;
               
               total_score += result.score;
               if result.passed {
                   passed_count += 1;
               }
               
               results.push(result);
           }
           
           let average_score = if !results.is_empty() {
               total_score / results.len() as f32
           } else {
               0.0
           };
           
           Ok(ValidationSummary {
               total_tests: results.len(),
               passed_tests: passed_count,
               average_score,
               individual_results: results,
           })
       }
   }
   
   /// Summary of validation results
   #[derive(Debug, Clone)]
   pub struct ValidationSummary {
       pub total_tests: usize,
       pub passed_tests: usize,
       pub average_score: f32,
       pub individual_results: Vec<ValidationResult>,
   }
   
   impl ValidationSummary {
       pub fn pass_rate(&self) -> f32 {
           if self.total_tests == 0 {
               1.0
           } else {
               self.passed_tests as f32 / self.total_tests as f32
           }
       }
       
       pub fn average_ranking_accuracy(&self) -> f32 {
           if self.individual_results.is_empty() {
               0.0
           } else {
               let sum: f32 = self.individual_results.iter().map(|r| r.ranking_accuracy).sum();
               sum / self.individual_results.len() as f32
           }
       }
       
       pub fn average_recall(&self) -> f32 {
           if self.individual_results.is_empty() {
               0.0
           } else {
               let sum: f32 = self.individual_results.iter().map(|r| r.recall_at_k).sum();
               sum / self.individual_results.len() as f32
           }
       }
       
       pub fn average_precision(&self) -> f32 {
           if self.individual_results.is_empty() {
               0.0
           } else {
               let sum: f32 = self.individual_results.iter().map(|r| r.precision_at_k).sum();
               sum / self.individual_results.len() as f32
           }
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Ranking accuracy calculation implemented
- [ ] Recall and precision metrics implemented
- [ ] Test case validation with comprehensive scoring
- [ ] ValidationSummary with aggregated metrics

## Next Task
task_046c_create_standard_test_cases.md

---

# Micro-Task 046c: Create Standard Test Cases

## Objective
Create a comprehensive set of standard test cases for similarity search validation.

## Prerequisites
- Task 046b completed (validation metrics implemented)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add standard test cases to `src/validation.rs`:
   ```rust
   impl SimilarityValidator {
       /// Create standard test cases for code similarity
       pub fn create_code_similarity_tests(&mut self) {
           self.test_cases.extend(vec![
               SimilarityTestCase {
                   name: "Exact Match".to_string(),
                   query_content: "fn main() { println!(\"Hello, world!\"); }".to_string(),
                   expected_results: vec!["fn main() { println!(\"Hello, world!\"); }".to_string()],
                   tolerance: 0.1, // Very strict for exact matches
               },
               
               SimilarityTestCase {
                   name: "Function Signature Similarity".to_string(),
                   query_content: "fn calculate_sum(a: i32, b: i32) -> i32".to_string(),
                   expected_results: vec![
                       "fn calculate_sum(a: i32, b: i32) -> i32".to_string(),
                       "fn add_numbers(x: i32, y: i32) -> i32".to_string(),
                       "fn compute_total(first: i32, second: i32) -> i32".to_string(),
                   ],
                   tolerance: 0.3,
               },
               
               SimilarityTestCase {
                   name: "Loop Structure Similarity".to_string(),
                   query_content: "for i in 0..10 { println!(\"{}\", i); }".to_string(),
                   expected_results: vec![
                       "for i in 0..10 { println!(\"{}\", i); }".to_string(),
                       "for j in 0..5 { print!(\"{} \", j); }".to_string(),
                       "for item in items { process(item); }".to_string(),
                   ],
                   tolerance: 0.4,
               },
               
               SimilarityTestCase {
                   name: "Variable Declaration Pattern".to_string(),
                   query_content: "let mut counter = 0;".to_string(),
                   expected_results: vec![
                       "let mut counter = 0;".to_string(),
                       "let mut index = 1;".to_string(),
                       "let mut value = 42;".to_string(),
                   ],
                   tolerance: 0.3,
               },
               
               SimilarityTestCase {
                   name: "Error Handling Pattern".to_string(),
                   query_content: "match result { Ok(value) => value, Err(e) => panic!(\"Error: {}\", e) }".to_string(),
                   expected_results: vec![
                       "match result { Ok(value) => value, Err(e) => panic!(\"Error: {}\", e) }".to_string(),
                       "match outcome { Ok(data) => data, Err(error) => return Err(error) }".to_string(),
                       "result.unwrap_or_else(|e| panic!(\"Failed: {}\", e))".to_string(),
                   ],
                   tolerance: 0.4,
               },
           ]);
       }
       
       /// Create test cases for natural language similarity
       pub fn create_text_similarity_tests(&mut self) {
           self.test_cases.extend(vec![
               SimilarityTestCase {
                   name: "Synonym Detection".to_string(),
                   query_content: "The quick brown fox jumps over the lazy dog".to_string(),
                   expected_results: vec![
                       "The quick brown fox jumps over the lazy dog".to_string(),
                       "A fast brown fox leaps over the sleepy dog".to_string(),
                       "The rapid fox bounds over the tired hound".to_string(),
                   ],
                   tolerance: 0.5,
               },
               
               SimilarityTestCase {
                   name: "Conceptual Similarity".to_string(),
                   query_content: "Machine learning algorithms for data analysis".to_string(),
                   expected_results: vec![
                       "Machine learning algorithms for data analysis".to_string(),
                       "AI techniques for analyzing datasets".to_string(),
                       "Statistical methods for data mining".to_string(),
                       "Deep learning approaches to pattern recognition".to_string(),
                   ],
                   tolerance: 0.6,
               },
               
               SimilarityTestCase {
                   name: "Technical Documentation".to_string(),
                   query_content: "How to install and configure the software package".to_string(),
                   expected_results: vec![
                       "How to install and configure the software package".to_string(),
                       "Installation and setup guide for the application".to_string(),
                       "Setting up and configuring the system".to_string(),
                   ],
                   tolerance: 0.4,
               },
           ]);
       }
       
       /// Add test documents for code similarity validation
       pub fn add_code_test_documents(&self) -> Result<(), String> {
           let code_samples = vec![
               ("fn main() { println!(\"Hello, world!\"); }", "Hello World Program"),
               ("fn calculate_sum(a: i32, b: i32) -> i32 { a + b }", "Sum Function"),
               ("fn add_numbers(x: i32, y: i32) -> i32 { x + y }", "Add Function"),
               ("fn compute_total(first: i32, second: i32) -> i32 { first + second }", "Total Function"),
               ("for i in 0..10 { println!(\"{}\", i); }", "Basic For Loop"),
               ("for j in 0..5 { print!(\"{} \", j); }", "Print Loop"),
               ("for item in items { process(item); }", "Item Processing Loop"),
               ("let mut counter = 0;", "Counter Declaration"),
               ("let mut index = 1;", "Index Declaration"),
               ("let mut value = 42;", "Value Declaration"),
               ("match result { Ok(value) => value, Err(e) => panic!(\"Error: {}\", e) }", "Match Expression"),
               ("match outcome { Ok(data) => data, Err(error) => return Err(error) }", "Result Handling"),
               ("result.unwrap_or_else(|e| panic!(\"Failed: {}\", e))", "Unwrap with Panic"),
           ];
           
           for (content, title) in code_samples {
               self.add_test_document(content, title)?;
           }
           
           Ok(())
       }
       
       /// Add test documents for text similarity validation
       pub fn add_text_test_documents(&self) -> Result<(), String> {
           let text_samples = vec![
               ("The quick brown fox jumps over the lazy dog", "Classic Pangram"),
               ("A fast brown fox leaps over the sleepy dog", "Similar Pangram"),
               ("The rapid fox bounds over the tired hound", "Synonym Pangram"),
               ("Machine learning algorithms for data analysis", "ML Analysis"),
               ("AI techniques for analyzing datasets", "AI Data Analysis"),
               ("Statistical methods for data mining", "Statistical Mining"),
               ("Deep learning approaches to pattern recognition", "Deep Learning Patterns"),
               ("How to install and configure the software package", "Installation Guide"),
               ("Installation and setup guide for the application", "Setup Instructions"),
               ("Setting up and configuring the system", "System Configuration"),
               ("Database optimization techniques and best practices", "DB Optimization"),
               ("Performance tuning for database systems", "DB Performance"),
               ("Web development with modern JavaScript frameworks", "Modern Web Dev"),
               ("Building responsive websites using CSS and HTML", "Responsive Design"),
           ];
           
           for (content, title) in text_samples {
               self.add_test_document(content, title)?;
           }
           
           Ok(())
       }
       
       /// Create a complete validation setup with all standard tests
       pub fn setup_standard_validation(&mut self) -> Result<(), String> {
           // Add all test documents
           self.add_code_test_documents()?;
           self.add_text_test_documents()?;
           
           // Create all test cases
           self.create_code_similarity_tests();
           self.create_text_similarity_tests();
           
           Ok(())
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Code similarity test cases for function signatures, loops, patterns
- [ ] Text similarity test cases for synonyms and concepts
- [ ] Standard test document collections
- [ ] Complete validation setup method

## Next Task
task_046d_add_performance_benchmarking.md

---

# Micro-Task 046d: Add Performance Benchmarking

## Objective
Add performance benchmarking capabilities to measure search speed and efficiency.

## Prerequisites
- Task 046c completed (standard test cases created)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add performance benchmarking to `src/validation.rs`:
   ```rust
   use std::time::{Duration, Instant};
   
   /// Performance benchmark results
   #[derive(Debug, Clone)]
   pub struct BenchmarkResult {
       pub test_name: String,
       pub total_queries: usize,
       pub total_duration: Duration,
       pub average_query_time: Duration,
       pub queries_per_second: f64,
       pub min_query_time: Duration,
       pub max_query_time: Duration,
   }
   
   /// Performance benchmark configuration
   #[derive(Debug, Clone)]
   pub struct BenchmarkConfig {
       pub warmup_queries: usize,
       pub benchmark_queries: usize,
       pub result_limit: usize,
   }
   
   impl Default for BenchmarkConfig {
       fn default() -> Self {
           Self {
               warmup_queries: 10,
               benchmark_queries: 100,
               result_limit: 10,
           }
       }
   }
   
   impl SimilarityValidator {
       /// Run performance benchmark on similarity search
       pub fn benchmark_similarity_search(&self, config: &BenchmarkConfig) -> Result<BenchmarkResult, String> {
           let search_config = SearchConfig {
               limit: config.result_limit,
               threshold: 0.0,
               include_metadata: true,
           };
           
           // Prepare test queries
           let test_queries = vec![
               "function definition",
               "loop iteration",
               "variable declaration",
               "error handling",
               "data processing",
               "algorithm implementation",
               "pattern matching",
               "memory management",
               "async programming",
               "web development",
           ];
           
           // Warmup phase
           for _ in 0..config.warmup_queries {
               let query = &test_queries[fastrand::usize(..test_queries.len())];
               let _ = self.storage.search_by_content(query, &search_config);
           }
           
           // Benchmark phase
           let mut query_times = Vec::new();
           let start_time = Instant::now();
           
           for _ in 0..config.benchmark_queries {
               let query = &test_queries[fastrand::usize(..test_queries.len())];
               
               let query_start = Instant::now();
               let _ = self.storage.search_by_content(query, &search_config)
                   .map_err(|e| e.to_string())?;
               let query_duration = query_start.elapsed();
               
               query_times.push(query_duration);
           }
           
           let total_duration = start_time.elapsed();
           
           // Calculate statistics
           let average_query_time = total_duration / config.benchmark_queries as u32;
           let queries_per_second = config.benchmark_queries as f64 / total_duration.as_secs_f64();
           let min_query_time = query_times.iter().min().copied().unwrap_or(Duration::ZERO);
           let max_query_time = query_times.iter().max().copied().unwrap_or(Duration::ZERO);
           
           Ok(BenchmarkResult {
               test_name: "Similarity Search".to_string(),
               total_queries: config.benchmark_queries,
               total_duration,
               average_query_time,
               queries_per_second,
               min_query_time,
               max_query_time,
           })
       }
       
       /// Benchmark indexed vs non-indexed search
       pub fn benchmark_indexed_vs_regular(&self, config: &BenchmarkConfig) -> Result<(BenchmarkResult, BenchmarkResult), String> {
           // Benchmark regular search first
           let regular_result = self.benchmark_similarity_search(config)?;
           
           // Build index for comparison
           let index_config = crate::mock_storage::IndexConfig::default();
           self.storage.build_index(&index_config)
               .map_err(|e| e.to_string())?;
           
           // Benchmark indexed search
           let indexed_result = self.benchmark_indexed_search(config)?;
           
           Ok((regular_result, indexed_result))
       }
       
       /// Benchmark indexed search performance
       fn benchmark_indexed_search(&self, config: &BenchmarkConfig) -> Result<BenchmarkResult, String> {
           let search_config = SearchConfig {
               limit: config.result_limit,
               threshold: 0.0,
               include_metadata: true,
           };
           
           let test_queries = vec![
               "function definition",
               "loop iteration", 
               "variable declaration",
               "error handling",
               "data processing",
           ];
           
           // Warmup
           for _ in 0..config.warmup_queries {
               let query = &test_queries[fastrand::usize(..test_queries.len())];
               let query_embedding = crate::mock_storage::VectorRecord::generate_mock_embedding(query);
               let _ = self.storage.indexed_similarity_search(&query_embedding, &search_config);
           }
           
           // Benchmark
           let mut query_times = Vec::new();
           let start_time = Instant::now();
           
           for _ in 0..config.benchmark_queries {
               let query = &test_queries[fastrand::usize(..test_queries.len())];
               let query_embedding = crate::mock_storage::VectorRecord::generate_mock_embedding(query);
               
               let query_start = Instant::now();
               let _ = self.storage.indexed_similarity_search(&query_embedding, &search_config)
                   .map_err(|e| e.to_string())?;
               let query_duration = query_start.elapsed();
               
               query_times.push(query_duration);
           }
           
           let total_duration = start_time.elapsed();
           let average_query_time = total_duration / config.benchmark_queries as u32;
           let queries_per_second = config.benchmark_queries as f64 / total_duration.as_secs_f64();
           let min_query_time = query_times.iter().min().copied().unwrap_or(Duration::ZERO);
           let max_query_time = query_times.iter().max().copied().unwrap_or(Duration::ZERO);
           
           Ok(BenchmarkResult {
               test_name: "Indexed Similarity Search".to_string(),
               total_queries: config.benchmark_queries,
               total_duration,
               average_query_time,
               queries_per_second,
               min_query_time,
               max_query_time,
           })
       }
       
       /// Run comprehensive performance test suite
       pub fn run_performance_suite(&self) -> Result<PerformanceSuite, String> {
           let config = BenchmarkConfig::default();
           
           // Benchmark different scenarios
           let similarity_benchmark = self.benchmark_similarity_search(&config)?;
           let (regular_benchmark, indexed_benchmark) = self.benchmark_indexed_vs_regular(&config)?;
           
           Ok(PerformanceSuite {
               similarity_search: similarity_benchmark,
               regular_search: regular_benchmark,
               indexed_search: indexed_benchmark,
               speedup_factor: regular_benchmark.queries_per_second / indexed_benchmark.queries_per_second,
           })
       }
   }
   
   /// Complete performance test suite results
   #[derive(Debug, Clone)]
   pub struct PerformanceSuite {
       pub similarity_search: BenchmarkResult,
       pub regular_search: BenchmarkResult,
       pub indexed_search: BenchmarkResult,
       pub speedup_factor: f64,
   }
   
   impl BenchmarkResult {
       pub fn format_summary(&self) -> String {
           format!(
               "{}: {:.2} queries/sec, avg {:.2}ms (min: {:.2}ms, max: {:.2}ms)",
               self.test_name,
               self.queries_per_second,
               self.average_query_time.as_secs_f64() * 1000.0,
               self.min_query_time.as_secs_f64() * 1000.0,
               self.max_query_time.as_secs_f64() * 1000.0
           )
       }
   }
   ```
3. Add fastrand to dependencies in `Cargo.toml`:
   ```toml
   [dependencies]
   fastrand = "2.0"
   ```
4. Test: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] Performance benchmarking framework with comprehensive metrics
- [ ] Indexed vs regular search comparison
- [ ] Warmup and proper statistical measurement
- [ ] Performance suite for multiple scenarios

## Next Task
task_046e_add_validation_tests_and_commit.md

---

# Micro-Task 046e: Add Validation Tests and Commit

## Objective
Write tests for the validation framework and commit the similarity search validation.

## Prerequisites
- Task 046d completed (performance benchmarking added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add validation tests to `src/validation.rs`:
   ```rust
   #[cfg(test)]
   mod validation_tests {
       use super::*;
       
       #[test]
       fn test_validation_framework_setup() {
           let mut validator = SimilarityValidator::new(384);
           
           // Test adding documents and test cases
           let doc_id = validator.add_test_document("test content", "test title").unwrap();
           assert!(!doc_id.is_nil());
           
           let test_case = SimilarityTestCase {
               name: "Basic Test".to_string(),
               query_content: "test".to_string(),
               expected_results: vec!["test content".to_string()],
               tolerance: 0.3,
           };
           
           validator.add_test_case(test_case);
           assert_eq!(validator.test_cases.len(), 1);
       }
       
       #[test]
       fn test_ranking_accuracy_calculation() {
           let found = vec!["first".to_string(), "second".to_string(), "third".to_string()];
           let expected = vec!["first".to_string(), "second".to_string()];
           
           let accuracy = SimilarityValidator::calculate_ranking_accuracy(&found, &expected);
           
           // Should be high since first two match in order
           assert!(accuracy > 0.8);
           
           // Test with reordered results
           let found_reordered = vec!["second".to_string(), "first".to_string(), "third".to_string()];
           let accuracy_reordered = SimilarityValidator::calculate_ranking_accuracy(&found_reordered, &expected);
           
           // Should be lower due to incorrect order
           assert!(accuracy_reordered < accuracy);
       }
       
       #[test]
       fn test_recall_and_precision_calculation() {
           let found = vec!["relevant1".to_string(), "irrelevant".to_string(), "relevant2".to_string()];
           let expected = vec!["relevant1".to_string(), "relevant2".to_string(), "relevant3".to_string()];
           
           let recall = SimilarityValidator::calculate_recall(&found, &expected);
           let precision = SimilarityValidator::calculate_precision(&found, &expected);
           
           // Recall: 2 out of 3 expected found = 2/3
           assert!((recall - 0.667).abs() < 0.01);
           
           // Precision: 2 out of 3 found are relevant = 2/3
           assert!((precision - 0.667).abs() < 0.01);
       }
       
       #[test]
       fn test_validation_with_exact_match() {
           let mut validator = SimilarityValidator::new(384);
           
           // Add exact match document
           validator.add_test_document("exact match content", "exact match").unwrap();
           
           let test_case = SimilarityTestCase {
               name: "Exact Match Test".to_string(),
               query_content: "exact match content".to_string(),
               expected_results: vec!["exact match content".to_string()],
               tolerance: 0.1,
           };
           
           let result = validator.validate_test_case(&test_case, 5).unwrap();
           
           assert!(result.passed);
           assert!(result.score > 0.9); // Should be very high for exact match
           assert_eq!(result.found_results.len(), 1);
           assert!(result.found_results[0].contains("exact match content"));
       }
       
       #[test]
       fn test_standard_validation_setup() {
           let mut validator = SimilarityValidator::new(384);
           
           // Setup standard validation
           validator.setup_standard_validation().unwrap();
           
           // Should have both code and text test cases
           assert!(!validator.test_cases.is_empty());
           
           // Should have test documents
           assert!(validator.storage.count().unwrap() > 0);
           
           // Test cases should include both code and text scenarios
           let has_code_test = validator.test_cases.iter().any(|tc| tc.name.contains("Function"));
           let has_text_test = validator.test_cases.iter().any(|tc| tc.name.contains("Synonym"));
           
           assert!(has_code_test);
           assert!(has_text_test);
       }
       
       #[test]
       fn test_validation_summary() {
           let mut validator = SimilarityValidator::new(384);
           
           // Add some test data
           validator.add_test_document("hello world", "greeting").unwrap();
           validator.add_test_document("goodbye world", "farewell").unwrap();
           
           let test_case1 = SimilarityTestCase {
               name: "Hello Test".to_string(),
               query_content: "hello".to_string(),
               expected_results: vec!["hello world".to_string()],
               tolerance: 0.3,
           };
           
           let test_case2 = SimilarityTestCase {
               name: "Goodbye Test".to_string(),
               query_content: "goodbye".to_string(),
               expected_results: vec!["goodbye world".to_string()],
               tolerance: 0.3,
           };
           
           validator.add_test_case(test_case1);
           validator.add_test_case(test_case2);
           
           let summary = validator.validate_all(5).unwrap();
           
           assert_eq!(summary.total_tests, 2);
           assert!(summary.passed_tests <= 2);
           assert!(summary.average_score >= 0.0);
           assert!(summary.pass_rate() >= 0.0);
           assert!(summary.pass_rate() <= 1.0);
       }
       
       #[test]
       fn test_performance_benchmarking() {
           let mut validator = SimilarityValidator::new(384);
           
           // Add some test documents for benchmarking
           for i in 0..50 {
               validator.add_test_document(
                   &format!("benchmark content number {}", i),
                   &format!("benchmark {}", i)
               ).unwrap();
           }
           
           let config = BenchmarkConfig {
               warmup_queries: 2,
               benchmark_queries: 10,
               result_limit: 5,
           };
           
           let benchmark_result = validator.benchmark_similarity_search(&config).unwrap();
           
           assert_eq!(benchmark_result.total_queries, 10);
           assert!(benchmark_result.queries_per_second > 0.0);
           assert!(benchmark_result.average_query_time.as_nanos() > 0);
           assert!(benchmark_result.min_query_time <= benchmark_result.max_query_time);
           
           println!("{}", benchmark_result.format_summary());
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\lancedb-integration && git commit -m "Implement comprehensive similarity search validation with metrics and benchmarking"`

## Success Criteria
- [ ] Validation framework setup tests implemented and passing
- [ ] Ranking accuracy calculation tests implemented and passing  
- [ ] Recall and precision calculation tests implemented and passing
- [ ] Exact match validation tests implemented and passing
- [ ] Performance benchmarking tests implemented and passing
- [ ] Similarity search validation committed to Git

## Next Task
task_047_implement_cross_validation_testing.md