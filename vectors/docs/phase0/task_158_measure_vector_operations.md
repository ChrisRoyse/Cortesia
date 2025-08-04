# Micro-Task 158: Measure Vector Operations

## Objective
Benchmark core vector operations including dot product, cosine similarity, and vector arithmetic.

## Prerequisites
- Task 157 completed (Text processing benchmarked)

## Time Estimate
8 minutes

## Instructions
1. Create vector operations benchmark `bench_vectors.rs`:
   ```rust
   use std::time::Instant;
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Benchmarking vector operations...");
       
       let vectors_128 = generate_vectors(1000, 128);
       let vectors_384 = generate_vectors(1000, 384);
       let vectors_768 = generate_vectors(1000, 768);
       
       benchmark_dot_product("128D", &vectors_128)?;
       benchmark_dot_product("384D", &vectors_384)?;
       benchmark_dot_product("768D", &vectors_768)?;
       
       benchmark_cosine_similarity("128D", &vectors_128)?;
       benchmark_cosine_similarity("384D", &vectors_384)?;
       benchmark_cosine_similarity("768D", &vectors_768)?;
       
       benchmark_vector_addition("128D", &vectors_128)?;
       benchmark_vector_addition("384D", &vectors_384)?;
       benchmark_vector_addition("768D", &vectors_768)?;
       
       Ok(())
   }
   
   fn generate_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
       (0..count).map(|i| {
           (0..dimensions).map(|j| ((i + j) as f32) / 1000.0).collect()
       }).collect()
   }
   
   fn benchmark_dot_product(dims: &str, vectors: &[Vec<f32>]) -> Result<(), Box<dyn std::error::Error>> {
       println!("Dot product benchmark ({}):", dims);
       let start = Instant::now();
       
       let mut total_ops = 0;
       for i in 0..vectors.len() {
           for j in (i+1)..vectors.len().min(i+10) { // Limit pairs for performance
               let _dot = dot_product(&vectors[i], &vectors[j]);
               total_ops += 1;
           }
       }
       
       let duration = start.elapsed();
       let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
       
       println!("  {} operations in {:.3}ms ({:.0} ops/sec)", total_ops, duration.as_millis(), ops_per_sec);
       
       Ok(())
   }
   
   fn benchmark_cosine_similarity(dims: &str, vectors: &[Vec<f32>]) -> Result<(), Box<dyn std::error::Error>> {
       println!("Cosine similarity benchmark ({}):", dims);
       let start = Instant::now();
       
       let mut total_ops = 0;
       for i in 0..vectors.len() {
           for j in (i+1)..vectors.len().min(i+10) {
               let _sim = cosine_similarity(&vectors[i], &vectors[j]);
               total_ops += 1;
           }
       }
       
       let duration = start.elapsed();
       let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
       
       println!("  {} operations in {:.3}ms ({:.0} ops/sec)", total_ops, duration.as_millis(), ops_per_sec);
       
       Ok(())
   }
   
   fn benchmark_vector_addition(dims: &str, vectors: &[Vec<f32>]) -> Result<(), Box<dyn std::error::Error>> {
       println!("Vector addition benchmark ({}):", dims);
       let start = Instant::now();
       
       let mut total_ops = 0;
       for i in 0..vectors.len() {
           for j in (i+1)..vectors.len().min(i+10) {
               let _sum = vector_add(&vectors[i], &vectors[j]);
               total_ops += 1;
           }
       }
       
       let duration = start.elapsed();
       let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
       
       println!("  {} operations in {:.3}ms ({:.0} ops/sec)", total_ops, duration.as_millis(), ops_per_sec);
       
       Ok(())
   }
   
   fn dot_product(a: &[f32], b: &[f32]) -> f32 {
       a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
   }
   
   fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
       let dot = dot_product(a, b);
       let norm_a = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
       let norm_b = (b.iter().map(|x| x * x).sum::<f32>()).sqrt();
       dot / (norm_a * norm_b)
   }
   
   fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
       a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
   }
   ```
2. Run: `cargo run --release --bin bench_vectors`
3. Commit: `git add src/bin/bench_vectors.rs && git commit -m "Benchmark core vector operations"`

## Success Criteria
- [ ] Vector operations benchmark created
- [ ] Dot product, cosine similarity, addition measured
- [ ] Performance across different dimensions assessed
- [ ] Results committed

## Next Task
task_159_benchmark_index_operations.md