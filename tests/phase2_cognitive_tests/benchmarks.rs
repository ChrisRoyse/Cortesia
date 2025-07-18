use std::time::{Duration, Instant};
use std::sync::Arc;
use llmkg::cognitive::*;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::neural::neural_server::NeuralProcessingServer;
use super::test_data_generator::TestDataGenerator;

/// Benchmarking utilities for Phase 2 cognitive patterns
pub struct CognitiveBenchmark {
    pub graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub neural_server: Arc<NeuralProcessingServer>,
    pub generator: TestDataGenerator,
}

impl CognitiveBenchmark {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut generator = TestDataGenerator::new().await?;
        generator.generate_comprehensive_data().await?;
        
        Ok(Self {
            graph: generator.graph.clone(),
            neural_server: generator.neural_server.clone(),
            generator,
        })
    }
    
    /// Benchmark convergent thinking performance
    pub async fn benchmark_convergent_thinking(&self, queries: &[&str]) -> BenchmarkResult {
        let convergent = ConvergentThinking::new(
            self.graph.clone(),
            self.neural_server.clone(),
        );
        
        let mut results = Vec::new();
        let total_start = Instant::now();
        
        for query in queries {
            let start = Instant::now();
            let result = convergent.execute_convergent_query(query, None).await;
            let duration = start.elapsed();
            
            results.push(QueryBenchmark {
                query: query.to_string(),
                duration,
                success: result.is_ok(),
                confidence: result.as_ref().map(|r| r.confidence).unwrap_or(0.0),
                reasoning_steps: result.as_ref().map(|r| r.reasoning_trace.len()).unwrap_or(0),
            });
        }
        
        let total_duration = total_start.elapsed();
        
        BenchmarkResult {
            pattern_type: "Convergent".to_string(),
            total_duration,
            average_duration: total_duration / queries.len() as u32,
            success_rate: self.calculate_success_rate(&results),
            average_confidence: self.calculate_average_confidence(&results),
            query_results: results,
        }
    }
    
    /// Benchmark divergent thinking performance
    pub async fn benchmark_divergent_thinking(&self, seed_concepts: &[&str]) -> BenchmarkResult {
        let divergent = DivergentThinking::new_with_params(
            self.graph.clone(),
            self.neural_server.clone(),
            15, // exploration_breadth
            0.3, // creativity_threshold
        );
        
        let mut results = Vec::new();
        let total_start = Instant::now();
        
        for concept in seed_concepts {
            let start = Instant::now();
            let result = divergent.execute_divergent_exploration(concept, ExplorationType::Instances).await;
            let duration = start.elapsed();
            
            results.push(QueryBenchmark {
                query: format!("Explore: {}", concept),
                duration,
                success: result.is_ok(),
                confidence: result.as_ref().map(|r| r.creativity_scores.iter().sum::<f32>() / r.creativity_scores.len() as f32).unwrap_or(0.0),
                reasoning_steps: result.as_ref().map(|r| r.explorations.len()).unwrap_or(0),
            });
        }
        
        let total_duration = total_start.elapsed();
        
        BenchmarkResult {
            pattern_type: "Divergent".to_string(),
            total_duration,
            average_duration: total_duration / seed_concepts.len() as u32,
            success_rate: self.calculate_success_rate(&results),
            average_confidence: self.calculate_average_confidence(&results),
            query_results: results,
        }
    }
    
    /// Benchmark lateral thinking performance
    pub async fn benchmark_lateral_thinking(&self, concept_pairs: &[(&str, &str)]) -> BenchmarkResult {
        let lateral = LateralThinking::new(
            self.graph.clone(),
            self.neural_server.clone(),
        );
        
        let mut results = Vec::new();
        let total_start = Instant::now();
        
        for (concept_a, concept_b) in concept_pairs {
            let start = Instant::now();
            let result = lateral.find_creative_connections(concept_a, concept_b, Some(5)).await;
            let duration = start.elapsed();
            
            results.push(QueryBenchmark {
                query: format!("Connect: {} -> {}", concept_a, concept_b),
                duration,
                success: result.is_ok(),
                confidence: result.as_ref().map(|r| r.bridges.iter().map(|b| b.plausibility_score).sum::<f32>() / r.bridges.len() as f32).unwrap_or(0.0),
                reasoning_steps: result.as_ref().map(|r| r.bridges.len()).unwrap_or(0),
            });
        }
        
        let total_duration = total_start.elapsed();
        
        BenchmarkResult {
            pattern_type: "Lateral".to_string(),
            total_duration,
            average_duration: total_duration / concept_pairs.len() as u32,
            success_rate: self.calculate_success_rate(&results),
            average_confidence: self.calculate_average_confidence(&results),
            query_results: results,
        }
    }
    
    /// Run comprehensive benchmark suite
    pub async fn run_comprehensive_benchmark(&self) -> ComprehensiveBenchmark {
        let start_time = Instant::now();
        
        // Convergent thinking queries
        let convergent_queries = vec![
            "What type is dog?",
            "What properties do mammals have?",
            "What is artificial intelligence?",
            "How do neural networks work?",
            "What animals are warm blooded?",
        ];
        
        // Divergent thinking seed concepts
        let divergent_concepts = vec![
            "animal",
            "technology",
            "intelligence",
            "pattern",
            "creativity",
        ];
        
        // Lateral thinking concept pairs
        let lateral_pairs = vec![
            ("art", "ai"),
            ("dog", "technology"),
            ("creativity", "problem_solving"),
            ("neural_network", "brain"),
            ("pattern", "intelligence"),
        ];
        
        // Run benchmarks
        let convergent_result = self.benchmark_convergent_thinking(&convergent_queries).await;
        let divergent_result = self.benchmark_divergent_thinking(&divergent_concepts).await;
        let lateral_result = self.benchmark_lateral_thinking(&lateral_pairs).await;
        
        let total_duration = start_time.elapsed();
        
        ComprehensiveBenchmark {
            total_duration,
            convergent: convergent_result,
            divergent: divergent_result,
            lateral: lateral_result,
            data_stats: self.generator.get_statistics().await.unwrap(),
        }
    }
    
    fn calculate_success_rate(&self, results: &[QueryBenchmark]) -> f32 {
        let successful = results.iter().filter(|r| r.success).count();
        successful as f32 / results.len() as f32
    }
    
    fn calculate_average_confidence(&self, results: &[QueryBenchmark]) -> f32 {
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        if successful_results.is_empty() {
            return 0.0;
        }
        
        let total_confidence: f32 = successful_results.iter().map(|r| r.confidence).sum();
        total_confidence / successful_results.len() as f32
    }
}

#[derive(Debug, Clone)]
pub struct QueryBenchmark {
    pub query: String,
    pub duration: Duration,
    pub success: bool,
    pub confidence: f32,
    pub reasoning_steps: usize,
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub pattern_type: String,
    pub total_duration: Duration,
    pub query_results: Vec<QueryBenchmark>,
    pub average_duration: Duration,
    pub success_rate: f32,
    pub average_confidence: f32,
}

#[derive(Debug)]
pub struct ComprehensiveBenchmark {
    pub total_duration: Duration,
    pub convergent: BenchmarkResult,
    pub divergent: BenchmarkResult,
    pub lateral: BenchmarkResult,
    pub data_stats: super::test_data_generator::TestDataStatistics,
}

impl ComprehensiveBenchmark {
    pub fn print_detailed_report(&self) {
        println!("\n=== COMPREHENSIVE COGNITIVE PATTERNS BENCHMARK ===");
        println!("Total benchmark duration: {:?}", self.total_duration);
        println!();
        
        // Print data statistics
        println!("=== Test Data Statistics ===");
        self.data_stats.print_summary();
        println!();
        
        // Print individual pattern results
        self.print_pattern_results(&self.convergent);
        self.print_pattern_results(&self.divergent);
        self.print_pattern_results(&self.lateral);
        
        // Print summary
        println!("=== SUMMARY ===");
        println!("Overall success rate: {:.2}%", self.calculate_overall_success_rate() * 100.0);
        println!("Average confidence: {:.3}", self.calculate_average_confidence());
        println!("Total queries executed: {}", self.get_total_queries());
        println!("===========================================");
    }
    
    fn print_pattern_results(&self, result: &BenchmarkResult) {
        println!("=== {} Thinking Results ===", result.pattern_type);
        println!("Total duration: {:?}", result.total_duration);
        println!("Average duration: {:?}", result.average_duration);
        println!("Success rate: {:.2}%", result.success_rate * 100.0);
        println!("Average confidence: {:.3}", result.average_confidence);
        println!("Queries executed: {}", result.query_results.len());
        
        println!("Individual query results:");
        for query_result in &result.query_results {
            println!("  '{}' - {:?} (success: {}, confidence: {:.3}, steps: {})", 
                query_result.query, 
                query_result.duration, 
                query_result.success, 
                query_result.confidence,
                query_result.reasoning_steps
            );
        }
        println!();
    }
    
    pub fn calculate_overall_success_rate(&self) -> f32 {
        let total_queries = self.get_total_queries();
        if total_queries == 0 {
            return 0.0;
        }
        
        let successful_queries = 
            self.convergent.query_results.iter().filter(|r| r.success).count() +
            self.divergent.query_results.iter().filter(|r| r.success).count() +
            self.lateral.query_results.iter().filter(|r| r.success).count();
        
        successful_queries as f32 / total_queries as f32
    }
    
    pub fn calculate_average_confidence(&self) -> f32 {
        let all_results: Vec<_> = self.convergent.query_results.iter()
            .chain(self.divergent.query_results.iter())
            .chain(self.lateral.query_results.iter())
            .filter(|r| r.success && r.confidence.is_finite())
            .collect();
        
        if all_results.is_empty() {
            return 0.0;
        }
        
        let total_confidence: f32 = all_results.iter().map(|r| r.confidence).sum();
        total_confidence / all_results.len() as f32
    }
    
    pub fn get_total_queries(&self) -> usize {
        self.convergent.query_results.len() + 
        self.divergent.query_results.len() + 
        self.lateral.query_results.len()
    }
}