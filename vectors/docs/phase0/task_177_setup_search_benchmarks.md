# Micro-Task 177: Setup Search Benchmarks

## Objective
Setup comprehensive search benchmarking framework for text and vector search operations.

## Prerequisites
- Task 176 completed (Performance profile generated)

## Time Estimate
9 minutes

## Instructions
1. Create search benchmark configuration `search_benchmark_config.toml`:
   ```toml
   [text_search]
   enabled = true
   query_types = ["exact", "fuzzy", "phrase", "boolean"]
   dataset_sizes = ["small", "medium", "large"]
   
   [vector_search]
   enabled = true
   dimensions = [128, 384, 768]
   similarity_metrics = ["cosine", "euclidean", "dot_product"]
   
   [hybrid_search]
   enabled = true
   fusion_methods = ["rrf", "linear", "weighted"]
   
   [performance_targets]
   text_search_ms = 50
   vector_search_ms = 100
   hybrid_search_ms = 150
   accuracy_threshold = 0.85
   
   [benchmark_settings]
   warmup_queries = 100
   benchmark_queries = 1000
   concurrent_threads = [1, 2, 4, 8]
   iterations = 5
   ```
2. Create search benchmark framework `search_benchmark_framework.rs`:
   ```rust
   use std::time::{Duration, Instant};
   use std::collections::HashMap;
   
   pub struct SearchBenchmarkFramework {
       config: SearchBenchmarkConfig,
       results: HashMap<String, BenchmarkResult>,
   }
   
   pub struct SearchBenchmarkConfig {
       pub warmup_queries: usize,
       pub benchmark_queries: usize,
       pub iterations: usize,
   }
   
   pub struct BenchmarkResult {
       pub min_latency: Duration,
       pub max_latency: Duration,
       pub avg_latency: Duration,
       pub p95_latency: Duration,
       pub p99_latency: Duration,
       pub throughput: f64,
       pub accuracy: f64,
   }
   
   impl SearchBenchmarkFramework {
       pub fn new(config: SearchBenchmarkConfig) -> Self {
           Self {
               config,
               results: HashMap::new(),
           }
       }
       
       pub fn benchmark_text_search(&mut self, query_type: &str) -> BenchmarkResult {
           println!("Benchmarking {} text search...", query_type);
           
           let mut latencies = Vec::new();
           let start_time = Instant::now();
           
           // Warmup
           for _ in 0..self.config.warmup_queries {
               let _ = self.simulate_text_search(query_type);
           }
           
           // Actual benchmark
           for _ in 0..self.config.benchmark_queries {
               let query_start = Instant::now();
               let _results = self.simulate_text_search(query_type);
               latencies.push(query_start.elapsed());
           }
           
           let total_time = start_time.elapsed();
           self.calculate_benchmark_result(latencies, total_time)
       }
       
       pub fn benchmark_vector_search(&mut self, dimensions: usize) -> BenchmarkResult {
           println!("Benchmarking {}D vector search...", dimensions);
           
           let mut latencies = Vec::new();
           let start_time = Instant::now();
           
           // Warmup
           for _ in 0..self.config.warmup_queries {
               let _ = self.simulate_vector_search(dimensions);
           }
           
           // Actual benchmark
           for _ in 0..self.config.benchmark_queries {
               let query_start = Instant::now();
               let _results = self.simulate_vector_search(dimensions);
               latencies.push(query_start.elapsed());
           }
           
           let total_time = start_time.elapsed();
           self.calculate_benchmark_result(latencies, total_time)
       }
       
       pub fn benchmark_hybrid_search(&mut self, fusion_method: &str) -> BenchmarkResult {
           println!("Benchmarking hybrid search with {}...", fusion_method);
           
           let mut latencies = Vec::new();
           let start_time = Instant::now();
           
           // Warmup
           for _ in 0..self.config.warmup_queries {
               let _ = self.simulate_hybrid_search(fusion_method);
           }
           
           // Actual benchmark
           for _ in 0..self.config.benchmark_queries {
               let query_start = Instant::now();
               let _results = self.simulate_hybrid_search(fusion_method);
               latencies.push(query_start.elapsed());
           }
           
           let total_time = start_time.elapsed();
           self.calculate_benchmark_result(latencies, total_time)
       }
       
       fn simulate_text_search(&self, query_type: &str) -> Vec<u32> {
           // Simulate text search latency based on query type
           let base_latency = match query_type {
               "exact" => 10,
               "fuzzy" => 25,
               "phrase" => 15,
               "boolean" => 30,
               _ => 20,
           };
           std::thread::sleep(Duration::from_micros(base_latency));
           vec![1, 2, 3, 4, 5]
       }
       
       fn simulate_vector_search(&self, dimensions: usize) -> Vec<u32> {
           // Simulate vector search latency based on dimensions
           let base_latency = 50 + (dimensions / 10);
           std::thread::sleep(Duration::from_micros(base_latency as u64));
           vec![6, 7, 8, 9, 10]
       }
       
       fn simulate_hybrid_search(&self, fusion_method: &str) -> Vec<u32> {
           // Simulate hybrid search (combination of text + vector)
           let text_results = self.simulate_text_search("exact");
           let vector_results = self.simulate_vector_search(384);
           
           let fusion_latency = match fusion_method {
               "rrf" => 5,
               "linear" => 3,
               "weighted" => 8,
               _ => 5,
           };
           std::thread::sleep(Duration::from_micros(fusion_latency));
           
           let mut combined = text_results;
           combined.extend(vector_results);
           combined
       }
       
       fn calculate_benchmark_result(&self, mut latencies: Vec<Duration>, total_time: Duration) -> BenchmarkResult {
           latencies.sort();
           
           let count = latencies.len();
           let min_latency = latencies[0];
           let max_latency = latencies[count - 1];
           let avg_latency = Duration::from_nanos(
               latencies.iter().map(|d| d.as_nanos()).sum::<u128>() as u64 / count as u64
           );
           let p95_latency = latencies[(count as f64 * 0.95) as usize];
           let p99_latency = latencies[(count as f64 * 0.99) as usize];
           let throughput = count as f64 / total_time.as_secs_f64();
           
           BenchmarkResult {
               min_latency,
               max_latency,
               avg_latency,
               p95_latency,
               p99_latency,
               throughput,
               accuracy: 0.92, // Simulated accuracy
           }
       }
   }
   
   fn main() {
       let config = SearchBenchmarkConfig {
           warmup_queries: 100,
           benchmark_queries: 1000,
           iterations: 3,
       };
       
       let mut framework = SearchBenchmarkFramework::new(config);
       
       // Test text search
       let text_result = framework.benchmark_text_search("exact");
       println!("Text search - Avg: {:.3}ms, P95: {:.3}ms, Throughput: {:.1} qps", 
               text_result.avg_latency.as_millis(), 
               text_result.p95_latency.as_millis(),
               text_result.throughput);
       
       // Test vector search
       let vector_result = framework.benchmark_vector_search(384);
       println!("Vector search - Avg: {:.3}ms, P95: {:.3}ms, Throughput: {:.1} qps", 
               vector_result.avg_latency.as_millis(), 
               vector_result.p95_latency.as_millis(),
               vector_result.throughput);
       
       // Test hybrid search
       let hybrid_result = framework.benchmark_hybrid_search("rrf");
       println!("Hybrid search - Avg: {:.3}ms, P95: {:.3}ms, Throughput: {:.1} qps", 
               hybrid_result.avg_latency.as_millis(), 
               hybrid_result.p95_latency.as_millis(),
               hybrid_result.throughput);
   }
   ```
3. Run framework test: `cargo run --release --bin search_benchmark_framework`
4. Commit: `git add search_benchmark_config.toml src/bin/search_benchmark_framework.rs && git commit -m "Setup comprehensive search benchmarking framework"`

## Success Criteria
- [ ] Search benchmark configuration created
- [ ] Benchmark framework implemented
- [ ] Text, vector, and hybrid search supported
- [ ] Performance metrics calculated
- [ ] Framework tested and committed

## Next Task
task_178_benchmark_exact_text_search.md