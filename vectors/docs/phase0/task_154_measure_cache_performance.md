# Micro-Task 154: Measure Cache Performance

## Objective
Benchmark caching effectiveness and measure cache hit/miss ratios for search operations.

## Prerequisites
- Task 153 completed (Concurrent operations benchmarked)

## Time Estimate
8 minutes

## Instructions
1. Create cache benchmark `bench_cache.rs`:
   ```rust
   use std::collections::HashMap;
   use std::time::Instant;
   
   struct SearchCache {
       cache: HashMap<String, Vec<u32>>,
       hits: usize,
       misses: usize,
   }
   
   impl SearchCache {
       fn new() -> Self {
           Self {
               cache: HashMap::new(),
               hits: 0,
               misses: 0,
           }
       }
       
       fn search(&mut self, query: &str) -> Vec<u32> {
           if let Some(results) = self.cache.get(query) {
               self.hits += 1;
               results.clone()
           } else {
               self.misses += 1;
               let results = expensive_search(query);
               self.cache.insert(query.to_string(), results.clone());
               results
           }
       }
       
       fn hit_ratio(&self) -> f64 {
           self.hits as f64 / (self.hits + self.misses) as f64
       }
   }
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Benchmarking cache performance...");
       
       benchmark_cache_scenario("Cold cache", 1000, 0.0)?;
       benchmark_cache_scenario("Warm cache", 1000, 0.3)?;
       benchmark_cache_scenario("Hot cache", 1000, 0.7)?;
       
       Ok(())
   }
   
   fn benchmark_cache_scenario(scenario: &str, queries: usize, repeat_ratio: f64) -> Result<(), Box<dyn std::error::Error>> {
       println!("Testing {}: {} queries, {:.1}% repeats", scenario, queries, repeat_ratio * 100.0);
       
       let mut cache = SearchCache::new();
       let start = Instant::now();
       
       for i in 0..queries {
           let query = if (i as f64 / queries as f64) < repeat_ratio {
               format!("repeat_query_{}", i % 10)
           } else {
               format!("unique_query_{}", i)
           };
           
           let _results = cache.search(&query);
       }
       
       let duration = start.elapsed();
       let avg_latency = duration.as_millis() as f64 / queries as f64;
       
       println!("  Duration: {:.3}ms, Avg: {:.3}ms/query", duration.as_millis(), avg_latency);
       println!("  Cache hit ratio: {:.1}%", cache.hit_ratio() * 100.0);
       println!("  Hits: {}, Misses: {}", cache.hits, cache.misses);
       
       Ok(())
   }
   
   fn expensive_search(query: &str) -> Vec<u32> {
       // Simulate expensive search operation
       std::thread::sleep(std::time::Duration::from_micros(100));
       vec![1, 2, 3, 4, 5]
   }
   ```
2. Run: `cargo run --release --bin bench_cache`
3. Commit: `git add src/bin/bench_cache.rs && git commit -m "Measure cache performance and hit ratios"`

## Success Criteria
- [ ] Cache benchmark created
- [ ] Hit/miss ratios measured
- [ ] Cache effectiveness quantified
- [ ] Results committed

## Next Task
task_155_benchmark_io_operations.md