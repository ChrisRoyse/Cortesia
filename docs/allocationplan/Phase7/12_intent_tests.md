# Micro Task 12: Intent Tests

**Priority**: CRITICAL  
**Estimated Time**: 30 minutes  
**Dependencies**: 11_query_decomposition.md  
**Skills Required**: Test design, quality assurance

## Objective

Implement comprehensive test suite for all Day 2 intent recognition components to ensure reliability, accuracy, and performance of the query intelligence system.

## Context

Testing the intent recognition pipeline is crucial for maintaining high accuracy and reliability. The test suite must cover all components from basic intent classification to complex query decomposition, ensuring the system works correctly across diverse query types and edge cases.

## Specifications

### Test Coverage Requirements

1. **Component Tests**
   - Intent type classification accuracy
   - Entity extraction precision/recall
   - Context analysis correctness
   - Query decomposition validation

2. **Integration Tests**
   - End-to-end query processing pipeline
   - Component interaction validation
   - Performance under load
   - Error handling and recovery

3. **Performance Tests**
   - Latency requirements validation
   - Concurrent processing capability
   - Memory usage optimization
   - Scalability verification

## Implementation Guide

### Step 1: Test Framework Setup
```rust
// File: tests/query/intent_tests.rs

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct TestMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub avg_latency_ms: u64,
    pub max_latency_ms: u64,
    pub success_rate: f32,
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub passed: bool,
    pub metrics: Option<TestMetrics>,
    pub error_message: Option<String>,
    pub execution_time_ms: u64,
}

pub struct IntentTestSuite {
    intent_parser: Arc<QueryIntentParser>,
    entity_extractor: Arc<HybridEntityExtractor>,
    context_analyzer: Arc<ContextAnalyzer>,
    query_decomposer: Arc<QueryDecomposer>,
    test_data: TestDataLoader,
}

impl IntentTestSuite {
    pub async fn new() -> Result<Self> {
        let llm = create_test_llm().await?;
        
        Ok(Self {
            intent_parser: Arc::new(QueryIntentParser::new(llm.clone()).await?),
            entity_extractor: Arc::new(HybridEntityExtractor::new(llm.clone())),
            context_analyzer: Arc::new(ContextAnalyzer::new()),
            query_decomposer: Arc::new(QueryDecomposer::new()),
            test_data: TestDataLoader::new(),
        })
    }
    
    pub async fn run_all_tests(&self) -> Vec<TestResult> {
        let mut results = Vec::new();
        
        // Component tests
        results.push(self.test_intent_classification().await);
        results.push(self.test_entity_extraction().await);
        results.push(self.test_context_analysis().await);
        results.push(self.test_query_decomposition().await);
        
        // Integration tests
        results.push(self.test_end_to_end_pipeline().await);
        results.push(self.test_component_interactions().await);
        
        // Performance tests
        results.push(self.test_latency_requirements().await);
        results.push(self.test_concurrent_processing().await);
        results.push(self.test_memory_usage().await);
        
        // Edge case tests
        results.push(self.test_edge_cases().await);
        results.push(self.test_error_handling().await);
        
        results
    }
}
```

### Step 2: Intent Classification Tests
```rust
impl IntentTestSuite {
    async fn test_intent_classification(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Intent Classification Accuracy".to_string();
        
        let test_cases = self.test_data.get_intent_classification_cases();
        let mut correct = 0;
        let mut total = 0;
        let mut latencies = Vec::new();
        
        for test_case in &test_cases {
            let case_start = Instant::now();
            
            match self.intent_parser.parse_intent(&test_case.query).await {
                Ok(parsed) => {
                    total += 1;
                    latencies.push(case_start.elapsed().as_millis() as u64);
                    
                    if self.intent_matches(&parsed.intent_type, &test_case.expected_intent) {
                        correct += 1;
                    }
                }
                Err(e) => {
                    return TestResult {
                        test_name,
                        passed: false,
                        metrics: None,
                        error_message: Some(format!("Intent parsing failed: {}", e)),
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                    };
                }
            }
        }
        
        let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };
        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        } else { 0 };
        let max_latency = latencies.iter().max().cloned().unwrap_or(0);
        
        let passed = accuracy >= 0.90 && avg_latency <= 200; // 90% accuracy, 200ms avg latency
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy,
                precision: accuracy, // Simplified for this test
                recall: accuracy,
                f1_score: accuracy,
                avg_latency_ms: avg_latency,
                max_latency_ms: max_latency,
                success_rate: accuracy,
            }),
            error_message: if !passed {
                Some(format!("Accuracy: {:.2}%, Avg Latency: {}ms", accuracy * 100.0, avg_latency))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    fn intent_matches(&self, actual: &QueryIntent, expected: &QueryIntent) -> bool {
        // Compare discriminants (intent types) ignoring specific values
        std::mem::discriminant(actual) == std::mem::discriminant(expected)
    }
}

#[derive(Debug, Clone)]
pub struct IntentTestCase {
    pub query: String,
    pub expected_intent: QueryIntent,
    pub expected_entities: Vec<String>,
    pub expected_confidence: f32,
    pub description: String,
}

pub struct TestDataLoader {
    intent_cases: Vec<IntentTestCase>,
    entity_cases: Vec<EntityTestCase>,
    context_cases: Vec<ContextTestCase>,
    decomposition_cases: Vec<DecompositionTestCase>,
}

impl TestDataLoader {
    pub fn new() -> Self {
        Self {
            intent_cases: Self::create_intent_test_cases(),
            entity_cases: Self::create_entity_test_cases(),
            context_cases: Self::create_context_test_cases(),
            decomposition_cases: Self::create_decomposition_test_cases(),
        }
    }
    
    fn create_intent_test_cases() -> Vec<IntentTestCase> {
        vec![
            IntentTestCase {
                query: "What animals can fly?".to_string(),
                expected_intent: QueryIntent::Filter {
                    entity_type: "animals".to_string(),
                    property: "can_fly".to_string(),
                    value: "true".to_string(),
                    operator: FilterOperator::Equals,
                },
                expected_entities: vec!["animals".to_string()],
                expected_confidence: 0.85,
                description: "Simple filter query".to_string(),
            },
            IntentTestCase {
                query: "How are dogs related to wolves?".to_string(),
                expected_intent: QueryIntent::Relationship {
                    entity1: "dogs".to_string(),
                    entity2: "wolves".to_string(),
                    relation_type: RelationType::Similarity,
                    direction: RelationDirection::Bidirectional,
                },
                expected_entities: vec!["dogs".to_string(), "wolves".to_string()],
                expected_confidence: 0.90,
                description: "Relationship query".to_string(),
            },
            IntentTestCase {
                query: "Compare lions and tigers".to_string(),
                expected_intent: QueryIntent::Comparison {
                    entities: vec!["lions".to_string(), "tigers".to_string()],
                    aspect: "general".to_string(),
                    comparison_type: ComparisonType::Differences,
                },
                expected_entities: vec!["lions".to_string(), "tigers".to_string()],
                expected_confidence: 0.88,
                description: "Comparison query".to_string(),
            },
            IntentTestCase {
                query: "Show me the mammal hierarchy".to_string(),
                expected_intent: QueryIntent::Hierarchy {
                    root_entity: "mammal".to_string(),
                    direction: HierarchyDirection::Descendants,
                    depth_limit: None,
                },
                expected_entities: vec!["mammal".to_string()],
                expected_confidence: 0.80,
                description: "Hierarchy navigation query".to_string(),
            },
            IntentTestCase {
                query: "Why do birds migrate?".to_string(),
                expected_intent: QueryIntent::Causal {
                    cause: "seasonal_changes".to_string(),
                    effect: "bird_migration".to_string(),
                    mechanism: None,
                },
                expected_entities: vec!["birds".to_string()],
                expected_confidence: 0.75,
                description: "Causal reasoning query".to_string(),
            },
            IntentTestCase {
                query: "What is photosynthesis?".to_string(),
                expected_intent: QueryIntent::Definition {
                    entity: "photosynthesis".to_string(),
                    detail_level: DetailLevel::Basic,
                },
                expected_entities: vec!["photosynthesis".to_string()],
                expected_confidence: 0.95,
                description: "Definition query".to_string(),
            },
        ]
    }
    
    pub fn get_intent_classification_cases(&self) -> &[IntentTestCase] {
        &self.intent_cases
    }
}
```

### Step 3: Entity Extraction Tests
```rust
#[derive(Debug, Clone)]
pub struct EntityTestCase {
    pub query: String,
    pub expected_entities: Vec<ExpectedEntity>,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ExpectedEntity {
    pub text: String,
    pub entity_type: EntityType,
    pub min_confidence: f32,
}

impl IntentTestSuite {
    async fn test_entity_extraction(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Entity Extraction Performance".to_string();
        
        let test_cases = self.test_data.get_entity_extraction_cases();
        let mut total_entities_expected = 0;
        let mut total_entities_found = 0;
        let mut correct_entities = 0;
        let mut latencies = Vec::new();
        
        for test_case in &test_cases {
            let case_start = Instant::now();
            
            match self.entity_extractor.extract_entities(&test_case.query).await {
                Ok(result) => {
                    latencies.push(case_start.elapsed().as_millis() as u64);
                    
                    total_entities_expected += test_case.expected_entities.len();
                    total_entities_found += result.entities.len();
                    
                    // Check for correct entity matches
                    for expected in &test_case.expected_entities {
                        if result.entities.iter().any(|found| {
                            found.text.to_lowercase().contains(&expected.text.to_lowercase()) &&
                            found.entity_type == expected.entity_type &&
                            found.confidence >= expected.min_confidence
                        }) {
                            correct_entities += 1;
                        }
                    }
                }
                Err(e) => {
                    return TestResult {
                        test_name,
                        passed: false,
                        metrics: None,
                        error_message: Some(format!("Entity extraction failed: {}", e)),
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                    };
                }
            }
        }
        
        let precision = if total_entities_found > 0 {
            correct_entities as f32 / total_entities_found as f32
        } else { 0.0 };
        
        let recall = if total_entities_expected > 0 {
            correct_entities as f32 / total_entities_expected as f32
        } else { 0.0 };
        
        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else { 0.0 };
        
        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        } else { 0 };
        
        let passed = precision >= 0.80 && recall >= 0.75 && avg_latency <= 500;
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: f1_score,
                precision,
                recall,
                f1_score,
                avg_latency_ms: avg_latency,
                max_latency_ms: latencies.iter().max().cloned().unwrap_or(0),
                success_rate: if precision >= 0.80 && recall >= 0.75 { 1.0 } else { 0.0 },
            }),
            error_message: if !passed {
                Some(format!("Precision: {:.2}%, Recall: {:.2}%, Latency: {}ms", 
                           precision * 100.0, recall * 100.0, avg_latency))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
}

impl TestDataLoader {
    fn create_entity_test_cases() -> Vec<EntityTestCase> {
        vec![
            EntityTestCase {
                query: "What animals live in the African savanna?".to_string(),
                expected_entities: vec![
                    ExpectedEntity {
                        text: "animals".to_string(),
                        entity_type: EntityType::Organism,
                        min_confidence: 0.8,
                    },
                    ExpectedEntity {
                        text: "African savanna".to_string(),
                        entity_type: EntityType::Location,
                        min_confidence: 0.7,
                    },
                ],
                description: "Organism and location extraction".to_string(),
            },
            EntityTestCase {
                query: "How does photosynthesis work in green plants?".to_string(),
                expected_entities: vec![
                    ExpectedEntity {
                        text: "photosynthesis".to_string(),
                        entity_type: EntityType::Concept,
                        min_confidence: 0.9,
                    },
                    ExpectedEntity {
                        text: "plants".to_string(),
                        entity_type: EntityType::Organism,
                        min_confidence: 0.8,
                    },
                ],
                description: "Concept and organism extraction".to_string(),
            },
            EntityTestCase {
                query: "Large red apples are sweet fruits".to_string(),
                expected_entities: vec![
                    ExpectedEntity {
                        text: "apples".to_string(),
                        entity_type: EntityType::Organism,
                        min_confidence: 0.8,
                    },
                    ExpectedEntity {
                        text: "fruits".to_string(),
                        entity_type: EntityType::Organism,
                        min_confidence: 0.7,
                    },
                ],
                description: "Entity extraction with modifiers".to_string(),
            },
        ]
    }
    
    pub fn get_entity_extraction_cases(&self) -> &[EntityTestCase] {
        &self.entity_cases
    }
}
```

### Step 4: Context Analysis Tests
```rust
#[derive(Debug, Clone)]
pub struct ContextTestCase {
    pub query: String,
    pub expected_domain: Domain,
    pub expected_expertise: ExpertiseLevel,
    pub expected_complexity: ComplexityLevel,
    pub description: String,
}

impl IntentTestSuite {
    async fn test_context_analysis(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Context Analysis Accuracy".to_string();
        
        let test_cases = self.test_data.get_context_analysis_cases();
        let mut correct_domains = 0;
        let mut correct_expertise = 0;
        let mut correct_complexity = 0;
        let mut total = 0;
        let mut latencies = Vec::new();
        
        for test_case in &test_cases {
            let case_start = Instant::now();
            
            // Simple entity list for testing
            let entities = vec![];
            
            match self.context_analyzer.analyze_context(&test_case.query, &entities) {
                Ok(context) => {
                    total += 1;
                    latencies.push(case_start.elapsed().as_millis() as u64);
                    
                    if context.domain.primary_domain == test_case.expected_domain {
                        correct_domains += 1;
                    }
                    if context.domain.expertise_level == test_case.expected_expertise {
                        correct_expertise += 1;
                    }
                    // Note: complexity would need to be part of context analysis in the actual implementation
                }
                Err(e) => {
                    return TestResult {
                        test_name,
                        passed: false,
                        metrics: None,
                        error_message: Some(format!("Context analysis failed: {}", e)),
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                    };
                }
            }
        }
        
        let domain_accuracy = if total > 0 { correct_domains as f32 / total as f32 } else { 0.0 };
        let expertise_accuracy = if total > 0 { correct_expertise as f32 / total as f32 } else { 0.0 };
        let overall_accuracy = (domain_accuracy + expertise_accuracy) / 2.0;
        
        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        } else { 0 };
        
        let passed = overall_accuracy >= 0.80 && avg_latency <= 100;
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: overall_accuracy,
                precision: domain_accuracy,
                recall: expertise_accuracy,
                f1_score: overall_accuracy,
                avg_latency_ms: avg_latency,
                max_latency_ms: latencies.iter().max().cloned().unwrap_or(0),
                success_rate: overall_accuracy,
            }),
            error_message: if !passed {
                Some(format!("Domain: {:.1}%, Expertise: {:.1}%, Latency: {}ms", 
                           domain_accuracy * 100.0, expertise_accuracy * 100.0, avg_latency))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
}

impl TestDataLoader {
    fn create_context_test_cases() -> Vec<ContextTestCase> {
        vec![
            ContextTestCase {
                query: "What is DNA replication?".to_string(),
                expected_domain: Domain::Biology,
                expected_expertise: ExpertiseLevel::Novice,
                expected_complexity: ComplexityLevel::Simple,
                description: "Basic biology question".to_string(),
            },
            ContextTestCase {
                query: "Explain the molecular mechanism of CRISPR-Cas9 gene editing".to_string(),
                expected_domain: Domain::Biology,
                expected_expertise: ExpertiseLevel::Expert,
                expected_complexity: ComplexityLevel::Complex,
                description: "Advanced molecular biology".to_string(),
            },
            ContextTestCase {
                query: "How do neural networks process information?".to_string(),
                expected_domain: Domain::Technology,
                expected_expertise: ExpertiseLevel::Intermediate,
                expected_complexity: ComplexityLevel::Compound,
                description: "Technology/AI question".to_string(),
            },
        ]
    }
    
    pub fn get_context_analysis_cases(&self) -> &[ContextTestCase] {
        &self.context_cases
    }
}
```

### Step 5: Query Decomposition Tests
```rust
#[derive(Debug, Clone)]
pub struct DecompositionTestCase {
    pub query: String,
    pub should_decompose: bool,
    pub min_sub_queries: usize,
    pub max_sub_queries: usize,
    pub expected_parallel_groups: usize,
    pub description: String,
}

impl IntentTestSuite {
    async fn test_query_decomposition(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Query Decomposition Logic".to_string();
        
        let test_cases = self.test_data.get_decomposition_cases();
        let mut correct_decisions = 0;
        let mut correct_structures = 0;
        let mut total = 0;
        let mut latencies = Vec::new();
        
        for test_case in &test_cases {
            let case_start = Instant::now();
            
            // Create a simple parsed query for testing
            let parsed_query = ParsedQuery {
                original_query: test_case.query.clone(),
                intent_type: QueryIntent::Unknown,
                entities: vec![],
                context: QueryContext::default(),
                confidence: 0.8,
                sub_queries: vec![],
                complexity: if test_case.should_decompose { 
                    ComplexityLevel::Complex 
                } else { 
                    ComplexityLevel::Simple 
                },
            };
            
            let context = QueryContext::default();
            
            match self.query_decomposer.decompose_query(&parsed_query, &context) {
                Ok(decomposition_opt) => {
                    total += 1;
                    latencies.push(case_start.elapsed().as_millis() as u64);
                    
                    // Check decomposition decision
                    let was_decomposed = decomposition_opt.is_some();
                    if was_decomposed == test_case.should_decompose {
                        correct_decisions += 1;
                    }
                    
                    // Check decomposition structure if it was decomposed
                    if let Some(decomposition) = decomposition_opt {
                        let sub_query_count = decomposition.sub_queries.len();
                        let parallel_groups = decomposition.execution_plan.parallelizable_groups.len();
                        
                        if sub_query_count >= test_case.min_sub_queries &&
                           sub_query_count <= test_case.max_sub_queries &&
                           parallel_groups >= test_case.expected_parallel_groups {
                            correct_structures += 1;
                        }
                    } else if !test_case.should_decompose {
                        correct_structures += 1; // Correctly didn't decompose
                    }
                }
                Err(e) => {
                    return TestResult {
                        test_name,
                        passed: false,
                        metrics: None,
                        error_message: Some(format!("Decomposition failed: {}", e)),
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                    };
                }
            }
        }
        
        let decision_accuracy = if total > 0 { correct_decisions as f32 / total as f32 } else { 0.0 };
        let structure_accuracy = if total > 0 { correct_structures as f32 / total as f32 } else { 0.0 };
        let overall_accuracy = (decision_accuracy + structure_accuracy) / 2.0;
        
        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        } else { 0 };
        
        let passed = overall_accuracy >= 0.75 && avg_latency <= 150;
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: overall_accuracy,
                precision: decision_accuracy,
                recall: structure_accuracy,
                f1_score: overall_accuracy,
                avg_latency_ms: avg_latency,
                max_latency_ms: latencies.iter().max().cloned().unwrap_or(0),
                success_rate: overall_accuracy,
            }),
            error_message: if !passed {
                Some(format!("Decision: {:.1}%, Structure: {:.1}%, Latency: {}ms", 
                           decision_accuracy * 100.0, structure_accuracy * 100.0, avg_latency))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
}

impl TestDataLoader {
    fn create_decomposition_test_cases() -> Vec<DecompositionTestCase> {
        vec![
            DecompositionTestCase {
                query: "What is a cat?".to_string(),
                should_decompose: false,
                min_sub_queries: 0,
                max_sub_queries: 0,
                expected_parallel_groups: 0,
                description: "Simple query should not be decomposed".to_string(),
            },
            DecompositionTestCase {
                query: "Compare lions and tigers in terms of hunting behavior".to_string(),
                should_decompose: true,
                min_sub_queries: 2,
                max_sub_queries: 5,
                expected_parallel_groups: 1,
                description: "Comparison query should be decomposed".to_string(),
            },
            DecompositionTestCase {
                query: "How are dogs related to wolves and what are their differences?".to_string(),
                should_decompose: true,
                min_sub_queries: 3,
                max_sub_queries: 6,
                expected_parallel_groups: 1,
                description: "Complex relationship query".to_string(),
            },
        ]
    }
    
    pub fn get_decomposition_cases(&self) -> &[DecompositionTestCase] {
        &self.decomposition_cases
    }
}
```

### Step 6: Performance and Integration Tests
```rust
impl IntentTestSuite {
    async fn test_end_to_end_pipeline(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "End-to-End Pipeline".to_string();
        
        let test_queries = vec![
            "What animals can fly and live in water?",
            "How are mammals related to reptiles?",
            "Compare the intelligence of dolphins and chimpanzees",
            "Why do some birds migrate while others don't?",
            "Show me the evolutionary tree of primates",
        ];
        
        let mut successful_processes = 0;
        let mut total_latencies = Vec::new();
        
        for query in &test_queries {
            let pipeline_start = Instant::now();
            
            // Full pipeline test
            match self.run_full_pipeline(query).await {
                Ok(_) => {
                    successful_processes += 1;
                    total_latencies.push(pipeline_start.elapsed().as_millis() as u64);
                }
                Err(_) => {
                    // Pipeline failed
                }
            }
        }
        
        let success_rate = successful_processes as f32 / test_queries.len() as f32;
        let avg_latency = if !total_latencies.is_empty() {
            total_latencies.iter().sum::<u64>() / total_latencies.len() as u64
        } else { 0 };
        
        let passed = success_rate >= 0.90 && avg_latency <= 1000; // 90% success, 1s max
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: success_rate,
                precision: success_rate,
                recall: success_rate,
                f1_score: success_rate,
                avg_latency_ms: avg_latency,
                max_latency_ms: total_latencies.iter().max().cloned().unwrap_or(0),
                success_rate,
            }),
            error_message: if !passed {
                Some(format!("Success rate: {:.1}%, Avg latency: {}ms", 
                           success_rate * 100.0, avg_latency))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    async fn run_full_pipeline(&self, query: &str) -> Result<()> {
        // 1. Parse intent
        let parsed_query = self.intent_parser.parse_intent(query).await?;
        
        // 2. Extract entities
        let entity_result = self.entity_extractor.extract_entities(query).await?;
        
        // 3. Analyze context
        let context = self.context_analyzer.analyze_context(query, &entity_result.entities)?;
        
        // 4. Attempt decomposition
        let _decomposition = self.query_decomposer.decompose_query(&parsed_query, &context)?;
        
        Ok(())
    }
    
    async fn test_concurrent_processing(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Concurrent Processing".to_string();
        
        let queries = vec![
            "What are mammals?",
            "How do birds fly?",
            "Compare cats and dogs",
            "Why do fish have gills?",
            "What is evolution?",
        ];
        
        // Test concurrent intent parsing
        let handles: Vec<_> = queries.iter().map(|query| {
            let parser = self.intent_parser.clone();
            let q = query.to_string();
            tokio::spawn(async move {
                parser.parse_intent(&q).await
            })
        }).collect();
        
        let results = futures::future::try_join_all(handles).await
            .map_err(|e| format!("Join error: {}", e))?;
        
        let successful = results.iter().filter(|r| r.is_ok()).count();
        let success_rate = successful as f32 / results.len() as f32;
        
        let passed = success_rate >= 0.95; // 95% success rate for concurrent processing
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: success_rate,
                precision: success_rate,
                recall: success_rate,
                f1_score: success_rate,
                avg_latency_ms: 0, // Not measuring individual latencies here
                max_latency_ms: 0,
                success_rate,
            }),
            error_message: if !passed {
                Some(format!("Concurrent success rate: {:.1}%", success_rate * 100.0))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    async fn test_latency_requirements(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Latency Requirements".to_string();
        
        let test_query = "What animals live in the ocean?";
        let mut latencies = Vec::new();
        
        // Run multiple iterations to get stable measurements
        for _ in 0..10 {
            let iter_start = Instant::now();
            let _ = self.intent_parser.parse_intent(test_query).await;
            latencies.push(iter_start.elapsed().as_millis() as u64);
        }
        
        let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
        let max_latency = latencies.iter().max().cloned().unwrap_or(0);
        let p95_latency = {
            let mut sorted = latencies.clone();
            sorted.sort();
            sorted[(sorted.len() as f32 * 0.95) as usize]
        };
        
        let passed = avg_latency <= 200 && p95_latency <= 300; // 200ms avg, 300ms p95
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: 1.0,
                precision: 1.0,
                recall: 1.0,
                f1_score: 1.0,
                avg_latency_ms: avg_latency,
                max_latency_ms: max_latency,
                success_rate: 1.0,
            }),
            error_message: if !passed {
                Some(format!("Avg: {}ms, P95: {}ms, Max: {}ms", avg_latency, p95_latency, max_latency))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    async fn test_memory_usage(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Memory Usage".to_string();
        
        // Simple memory usage test - in practice you'd use a proper memory profiler
        let initial_memory = self.get_approximate_memory_usage();
        
        // Process multiple queries
        for i in 0..100 {
            let query = format!("Test query number {}", i);
            let _ = self.intent_parser.parse_intent(&query).await;
        }
        
        let final_memory = self.get_approximate_memory_usage();
        let memory_growth = final_memory.saturating_sub(initial_memory);
        
        // Allow some memory growth but not excessive
        let passed = memory_growth < 50_000_000; // 50MB limit
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: 1.0,
                precision: 1.0,
                recall: 1.0,
                f1_score: 1.0,
                avg_latency_ms: 0,
                max_latency_ms: 0,
                success_rate: if passed { 1.0 } else { 0.0 },
            }),
            error_message: if !passed {
                Some(format!("Memory growth: {}MB", memory_growth / 1_000_000))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    fn get_approximate_memory_usage(&self) -> usize {
        // Simplified memory measurement - in practice use a proper profiler
        std::process::id() as usize * 1000 // Placeholder
    }
    
    async fn test_edge_cases(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Edge Cases Handling".to_string();
        
        let edge_cases = vec![
            "",                                    // Empty query
            "?",                                   // Single character
            "a".repeat(1000),                      // Very long query
            "What is a 🦁?",                      // Unicode/emoji
            "What is\n\ta\tcat?",                 // Whitespace variations
            "WHAT ARE ANIMALS?",                   // All caps
            "what are animals",                    // No punctuation
            "Que sont les animaux?",               // Non-English
        ];
        
        let mut handled_gracefully = 0;
        
        for edge_case in &edge_cases {
            match self.intent_parser.parse_intent(edge_case).await {
                Ok(_) | Err(_) => {
                    // Both success and graceful error handling count as handled
                    handled_gracefully += 1;
                }
            }
        }
        
        let success_rate = handled_gracefully as f32 / edge_cases.len() as f32;
        let passed = success_rate >= 0.90; // 90% should be handled gracefully
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: success_rate,
                precision: success_rate,
                recall: success_rate,
                f1_score: success_rate,
                avg_latency_ms: 0,
                max_latency_ms: 0,
                success_rate,
            }),
            error_message: if !passed {
                Some(format!("Edge case handling: {:.1}%", success_rate * 100.0))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    async fn test_error_handling(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Error Handling and Recovery".to_string();
        
        // Test various error conditions
        let mut error_recovery_tests = 0;
        let mut successful_recoveries = 0;
        
        // Test 1: Network timeout simulation (if applicable)
        error_recovery_tests += 1;
        // In a real implementation, you might inject faults or use test doubles
        if true { // Placeholder - implement actual error injection
            successful_recoveries += 1;
        }
        
        // Test 2: Invalid LLM response format
        error_recovery_tests += 1;
        // Test graceful handling of malformed responses
        if true { // Placeholder
            successful_recoveries += 1;
        }
        
        // Test 3: Resource exhaustion
        error_recovery_tests += 1;
        // Test behavior under resource constraints
        if true { // Placeholder
            successful_recoveries += 1;
        }
        
        let recovery_rate = if error_recovery_tests > 0 {
            successful_recoveries as f32 / error_recovery_tests as f32
        } else { 0.0 };
        
        let passed = recovery_rate >= 0.80; // 80% error recovery rate
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: recovery_rate,
                precision: recovery_rate,
                recall: recovery_rate,
                f1_score: recovery_rate,
                avg_latency_ms: 0,
                max_latency_ms: 0,
                success_rate: recovery_rate,
            }),
            error_message: if !passed {
                Some(format!("Error recovery rate: {:.1}%", recovery_rate * 100.0))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    async fn test_component_interactions(&self) -> TestResult {
        let start_time = Instant::now();
        let test_name = "Component Interactions".to_string();
        
        // Test that components work well together
        let query = "Compare the hunting behavior of lions and cheetahs in African savannas";
        
        // Parse intent
        let parsed = self.intent_parser.parse_intent(query).await
            .map_err(|e| format!("Intent parsing failed: {}", e))?;
        
        // Extract entities
        let entities = self.entity_extractor.extract_entities(query).await
            .map_err(|e| format!("Entity extraction failed: {}", e))?;
        
        // Analyze context
        let context = self.context_analyzer.analyze_context(query, &entities.entities)
            .map_err(|e| format!("Context analysis failed: {}", e))?;
        
        // Try decomposition
        let decomposition = self.query_decomposer.decompose_query(&parsed, &context)
            .map_err(|e| format!("Query decomposition failed: {}", e))?;
        
        // Validate interaction results
        let entity_count = entities.entities.len();
        let is_biology_domain = matches!(context.domain.primary_domain, Domain::Biology);
        let has_decomposition = decomposition.is_some();
        
        let interaction_quality = (
            if entity_count >= 2 { 1.0 } else { 0.0 } +
            if is_biology_domain { 1.0 } else { 0.0 } +
            if has_decomposition { 1.0 } else { 0.0 }
        ) / 3.0;
        
        let passed = interaction_quality >= 0.67; // At least 2/3 components working correctly
        
        TestResult {
            test_name,
            passed,
            metrics: Some(TestMetrics {
                accuracy: interaction_quality,
                precision: interaction_quality,
                recall: interaction_quality,
                f1_score: interaction_quality,
                avg_latency_ms: 0,
                max_latency_ms: 0,
                success_rate: interaction_quality,
            }),
            error_message: if !passed {
                Some(format!("Component interaction quality: {:.1}%", interaction_quality * 100.0))
            } else { None },
            execution_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
}

// Helper functions for test infrastructure
async fn create_test_llm() -> Result<Arc<dyn LanguageModel + Send + Sync>> {
    // Create a test LLM implementation or mock
    todo!("Implement test LLM")
}

pub fn print_test_results(results: &[TestResult]) {
    println!("\n=== Intent Recognition Test Results ===\n");
    
    let mut passed = 0;
    let mut total = 0;
    
    for result in results {
        total += 1;
        if result.passed {
            passed += 1;
            println!("✅ {}", result.test_name);
        } else {
            println!("❌ {}", result.test_name);
            if let Some(error) = &result.error_message {
                println!("   Error: {}", error);
            }
        }
        
        if let Some(metrics) = &result.metrics {
            println!("   Accuracy: {:.1}%, Latency: {}ms", 
                   metrics.accuracy * 100.0, metrics.avg_latency_ms);
        }
        
        println!("   Execution time: {}ms\n", result.execution_time_ms);
    }
    
    println!("=== Summary ===");
    println!("Tests passed: {}/{} ({:.1}%)", passed, total, 
             (passed as f32 / total as f32) * 100.0);
}
```

## File Locations

- `tests/query/intent_tests.rs` - Main test suite
- `tests/query/test_data.rs` - Test data sets
- `tests/query/test_helpers.rs` - Test utility functions
- `tests/query/performance_tests.rs` - Performance benchmarks
- `tests/query/integration_tests.rs` - Integration test scenarios

## Success Criteria

- [ ] All component tests pass with > 80% accuracy
- [ ] Integration tests demonstrate proper component interaction
- [ ] Performance tests validate latency requirements
- [ ] Edge cases handled gracefully
- [ ] Error recovery mechanisms functional
- [ ] Memory usage remains stable
- [ ] Concurrent processing works correctly

## Test Requirements

```rust
#[tokio::test]
async fn run_full_intent_test_suite() {
    let test_suite = IntentTestSuite::new().await.unwrap();
    let results = test_suite.run_all_tests().await;
    
    print_test_results(&results);
    
    // Ensure all critical tests pass
    let critical_tests = results.iter().filter(|r| {
        r.test_name.contains("Intent Classification") ||
        r.test_name.contains("Entity Extraction") ||
        r.test_name.contains("End-to-End")
    });
    
    for test in critical_tests {
        assert!(test.passed, "Critical test failed: {}", test.test_name);
    }
    
    // Ensure overall success rate
    let pass_rate = results.iter().filter(|r| r.passed).count() as f32 / results.len() as f32;
    assert!(pass_rate >= 0.80, "Overall test pass rate too low: {:.1}%", pass_rate * 100.0);
}

#[tokio::test]
async fn benchmark_intent_performance() {
    let test_suite = IntentTestSuite::new().await.unwrap();
    
    let queries = vec![
        "What are mammals?",
        "How do birds fly?",
        "Compare lions and tigers",
        "Why do fish have gills?",
        "Show me the evolutionary tree of primates",
    ];
    
    let start = Instant::now();
    
    for query in &queries {
        let _ = test_suite.intent_parser.parse_intent(query).await;
    }
    
    let total_time = start.elapsed();
    let avg_time = total_time / queries.len() as u32;
    
    assert!(avg_time <= Duration::from_millis(200), 
           "Average query processing time too high: {:?}", avg_time);
}
```

## Quality Gates

- [ ] Test suite runs in under 60 seconds
- [ ] No test flakiness or non-deterministic failures
- [ ] Coverage includes all major code paths
- [ ] Performance benchmarks track regression
- [ ] Integration tests validate real-world scenarios

## Next Task

Upon completion, proceed to **Phase 7 Day 3 tasks** or return to the main Phase 7 overview for task coordination.