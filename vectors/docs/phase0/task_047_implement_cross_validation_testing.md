# Micro-Task 047a: Create Cross-Validation Framework

## Objective
Implement cross-validation testing to ensure robust similarity search performance across different data distributions.

## Prerequisites
- Task 046e completed (similarity search validation tests passing and committed)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add cross-validation framework to `src/validation.rs`:
   ```rust
   /// Cross-validation configuration
   #[derive(Debug, Clone)]
   pub struct CrossValidationConfig {
       pub num_folds: usize,
       pub test_ratio: f32,
       pub shuffle_data: bool,
       pub random_seed: Option<u64>,
   }
   
   impl Default for CrossValidationConfig {
       fn default() -> Self {
           Self {
               num_folds: 5,
               test_ratio: 0.2,
               shuffle_data: true,
               random_seed: Some(42),
           }
       }
   }
   
   /// Cross-validation fold data
   #[derive(Debug, Clone)]
   pub struct ValidationFold {
       pub fold_id: usize,
       pub training_documents: Vec<VectorRecord>,
       pub test_documents: Vec<VectorRecord>,
       pub test_cases: Vec<SimilarityTestCase>,
   }
   
   /// Cross-validation results
   #[derive(Debug, Clone)]
   pub struct CrossValidationResults {
       pub config: CrossValidationConfig,
       pub fold_results: Vec<FoldResult>,
       pub overall_metrics: CrossValidationMetrics,
   }
   
   #[derive(Debug, Clone)]
   pub struct FoldResult {
       pub fold_id: usize,
       pub validation_summary: ValidationSummary,
       pub performance_metrics: BenchmarkResult,
   }
   
   #[derive(Debug, Clone)]
   pub struct CrossValidationMetrics {
       pub mean_accuracy: f32,
       pub std_accuracy: f32,
       pub mean_recall: f32,
       pub std_recall: f32,
       pub mean_precision: f32,
       pub std_precision: f32,
       pub mean_queries_per_second: f64,
       pub consistency_score: f32,
   }
   
   impl SimilarityValidator {
       /// Perform k-fold cross-validation
       pub fn cross_validate(&self, config: &CrossValidationConfig) -> Result<CrossValidationResults, String> {
           // Get all records from storage
           let all_records = self.storage.get_all().map_err(|e| e.to_string())?;
           
           if all_records.len() < config.num_folds {
               return Err("Not enough documents for cross-validation".to_string());
           }
           
           // Create folds
           let folds = self.create_folds(&all_records, config)?;
           
           let mut fold_results = Vec::new();
           
           for fold in folds {
               let result = self.validate_fold(&fold)?;
               fold_results.push(result);
           }
           
           let overall_metrics = self.calculate_cross_validation_metrics(&fold_results);
           
           Ok(CrossValidationResults {
               config: config.clone(),
               fold_results,
               overall_metrics,
           })
       }
       
       /// Create validation folds from documents
       fn create_folds(&self, all_records: &[VectorRecord], config: &CrossValidationConfig) -> Result<Vec<ValidationFold>, String> {
           let mut records = all_records.to_vec();
           
           // Shuffle if requested
           if config.shuffle_data {
               if let Some(seed) = config.random_seed {
                   fastrand::seed(seed);
               }
               for i in (1..records.len()).rev() {
                   let j = fastrand::usize(..=i);
                   records.swap(i, j);
               }
           }
           
           let fold_size = records.len() / config.num_folds;
           let mut folds = Vec::new();
           
           for fold_id in 0..config.num_folds {
               let start_idx = fold_id * fold_size;
               let end_idx = if fold_id == config.num_folds - 1 {
                   records.len() // Include remaining documents in last fold
               } else {
                   (fold_id + 1) * fold_size
               };
               
               let test_documents = records[start_idx..end_idx].to_vec();
               let mut training_documents = Vec::new();
               
               // Add all other documents to training set
               for (i, record) in records.iter().enumerate() {
                   if i < start_idx || i >= end_idx {
                       training_documents.push(record.clone());
                   }
               }
               
               // Generate test cases from test documents
               let test_cases = self.generate_test_cases_from_documents(&test_documents, 3)?;
               
               folds.push(ValidationFold {
                   fold_id,
                   training_documents,
                   test_documents,
                   test_cases,
               });
           }
           
           Ok(folds)
       }
       
       /// Generate test cases from a set of documents
       fn generate_test_cases_from_documents(&self, documents: &[VectorRecord], cases_per_doc: usize) -> Result<Vec<SimilarityTestCase>, String> {
           let mut test_cases = Vec::new();
           
           for (doc_idx, doc) in documents.iter().enumerate() {
               for case_idx in 0..cases_per_doc {
                   // Create different types of test cases
                   let test_case = match case_idx {
                       0 => {
                           // Exact match test
                           SimilarityTestCase {
                               name: format!("Exact_{}_{}", doc_idx, case_idx),
                               query_content: doc.content.clone(),
                               expected_results: vec![doc.content.clone()],
                               tolerance: 0.1,
                           }
                       }
                       1 => {
                           // Partial match test (first half of content)
                           let partial_content = if doc.content.len() > 20 {
                               doc.content.chars().take(doc.content.len() / 2).collect()
                           } else {
                               doc.content.clone()
                           };
                           
                           SimilarityTestCase {
                               name: format!("Partial_{}_{}", doc_idx, case_idx),
                               query_content: partial_content,
                               expected_results: vec![doc.content.clone()],
                               tolerance: 0.4,
                           }
                       }
                       2 => {
                           // Title-based test
                           SimilarityTestCase {
                               name: format!("Title_{}_{}", doc_idx, case_idx),
                               query_content: doc.title.clone(),
                               expected_results: vec![doc.content.clone()],
                               tolerance: 0.5,
                           }
                       }
                       _ => continue,
                   };
                   
                   test_cases.push(test_case);
               }
           }
           
           Ok(test_cases)
       }
       
       /// Validate a single fold
       fn validate_fold(&self, fold: &ValidationFold) -> Result<FoldResult, String> {
           // Create temporary validator for this fold
           let fold_validator = SimilarityValidator::new(self.storage.dimension);
           
           // Add training documents
           for doc in &fold.training_documents {
               fold_validator.storage.insert(doc.clone()).map_err(|e| e.to_string())?;
           }
           
           // Add test cases
           for test_case in &fold.test_cases {
               fold_validator.test_cases.push(test_case.clone());
           }
           
           // Run validation
           let validation_summary = fold_validator.validate_all(10)?;
           
           // Run performance benchmark
           let benchmark_config = BenchmarkConfig {
               warmup_queries: 5,
               benchmark_queries: 20,
               result_limit: 5,
           };
           
           let performance_metrics = fold_validator.benchmark_similarity_search(&benchmark_config)?;
           
           Ok(FoldResult {
               fold_id: fold.fold_id,
               validation_summary,
               performance_metrics,
           })
       }
       
       /// Calculate overall cross-validation metrics
       fn calculate_cross_validation_metrics(&self, fold_results: &[FoldResult]) -> CrossValidationMetrics {
           if fold_results.is_empty() {
               return CrossValidationMetrics {
                   mean_accuracy: 0.0,
                   std_accuracy: 0.0,
                   mean_recall: 0.0,
                   std_recall: 0.0,
                   mean_precision: 0.0,
                   std_precision: 0.0,
                   mean_queries_per_second: 0.0,
                   consistency_score: 0.0,
               };
           }
           
           let scores: Vec<f32> = fold_results.iter().map(|r| r.validation_summary.average_score).collect();
           let recalls: Vec<f32> = fold_results.iter().map(|r| r.validation_summary.average_recall()).collect();
           let precisions: Vec<f32> = fold_results.iter().map(|r| r.validation_summary.average_precision()).collect();
           let qps: Vec<f64> = fold_results.iter().map(|r| r.performance_metrics.queries_per_second).collect();
           
           let mean_accuracy = scores.iter().sum::<f32>() / scores.len() as f32;
           let mean_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
           let mean_precision = precisions.iter().sum::<f32>() / precisions.len() as f32;
           let mean_queries_per_second = qps.iter().sum::<f64>() / qps.len() as f64;
           
           let std_accuracy = Self::calculate_std_dev(&scores, mean_accuracy);
           let std_recall = Self::calculate_std_dev(&recalls, mean_recall);
           let std_precision = Self::calculate_std_dev(&precisions, mean_precision);
           
           // Consistency score: higher when standard deviations are lower
           let consistency_score = 1.0 - (std_accuracy + std_recall + std_precision) / 3.0;
           
           CrossValidationMetrics {
               mean_accuracy,
               std_accuracy,
               mean_recall,
               std_recall,
               mean_precision,
               std_precision,
               mean_queries_per_second,
               consistency_score: consistency_score.max(0.0),
           }
       }
       
       /// Calculate standard deviation
       fn calculate_std_dev(values: &[f32], mean: f32) -> f32 {
           if values.len() <= 1 {
               return 0.0;
           }
           
           let variance = values.iter()
               .map(|x| (x - mean).powi(2))
               .sum::<f32>() / values.len() as f32;
           
           variance.sqrt()
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Cross-validation framework with k-fold splitting
- [ ] Automatic test case generation from documents
- [ ] Statistical metrics calculation with consistency scoring
- [ ] Proper data shuffling and fold creation

## Next Task
task_047b_add_stratified_sampling.md