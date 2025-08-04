# Micro-Task 157: Benchmark Text Processing

## Objective
Measure text processing performance for tokenization, stemming, and text analysis operations.

## Prerequisites
- Task 156 completed (Serialization performance measured)

## Time Estimate
9 minutes

## Instructions
1. Create text processing benchmark `bench_text.rs`:
   ```rust
   use std::time::Instant;
   use std::collections::HashMap;
   
   fn main() -> Result<(), Box<dyn std::error::Error>> {
       println!("Benchmarking text processing...");
       
       let texts = generate_text_samples();
       
       benchmark_tokenization(&texts)?;
       benchmark_stemming(&texts)?;
       benchmark_ngram_generation(&texts)?;
       benchmark_term_frequency(&texts)?;
       
       Ok(())
   }
   
   fn generate_text_samples() -> Vec<String> {
       vec![
           "The quick brown fox jumps over the lazy dog".repeat(10),
           "Natural language processing with Rust is efficient and fast".repeat(20),
           "Vector search enables semantic similarity matching across documents".repeat(15),
           "Performance benchmarking helps optimize search algorithms".repeat(25),
       ]
   }
   
   fn benchmark_tokenization(texts: &[String]) -> Result<(), Box<dyn std::error::Error>> {
       println!("Tokenization benchmark:");
       let start = Instant::now();
       
       let mut total_tokens = 0;
       for text in texts {
           let tokens: Vec<&str> = text.split_whitespace().collect();
           total_tokens += tokens.len();
       }
       
       let duration = start.elapsed();
       let tokens_per_sec = total_tokens as f64 / duration.as_secs_f64();
       
       println!("  {} tokens in {:.3}ms ({:.0} tokens/sec)", total_tokens, duration.as_millis(), tokens_per_sec);
       
       Ok(())
   }
   
   fn benchmark_stemming(texts: &[String]) -> Result<(), Box<dyn std::error::Error>> {
       println!("Stemming benchmark:");
       let start = Instant::now();
       
       let mut total_words = 0;
       for text in texts {
           let tokens: Vec<&str> = text.split_whitespace().collect();
           for token in tokens {
               let _stemmed = simple_stem(token);
               total_words += 1;
           }
       }
       
       let duration = start.elapsed();
       let words_per_sec = total_words as f64 / duration.as_secs_f64();
       
       println!("  {} words in {:.3}ms ({:.0} words/sec)", total_words, duration.as_millis(), words_per_sec);
       
       Ok(())
   }
   
   fn benchmark_ngram_generation(texts: &[String]) -> Result<(), Box<dyn std::error::Error>> {
       println!("N-gram generation benchmark:");
       let start = Instant::now();
       
       let mut total_ngrams = 0;
       for text in texts {
           let tokens: Vec<&str> = text.split_whitespace().collect();
           let ngrams = generate_bigrams(&tokens);
           total_ngrams += ngrams.len();
       }
       
       let duration = start.elapsed();
       let ngrams_per_sec = total_ngrams as f64 / duration.as_secs_f64();
       
       println!("  {} n-grams in {:.3}ms ({:.0} n-grams/sec)", total_ngrams, duration.as_millis(), ngrams_per_sec);
       
       Ok(())
   }
   
   fn benchmark_term_frequency(texts: &[String]) -> Result<(), Box<dyn std::error::Error>> {
       println!("Term frequency benchmark:");
       let start = Instant::now();
       
       let mut total_terms = 0;
       for text in texts {
           let tf = calculate_term_frequency(text);
           total_terms += tf.len();
       }
       
       let duration = start.elapsed();
       let terms_per_sec = total_terms as f64 / duration.as_secs_f64();
       
       println!("  {} unique terms in {:.3}ms ({:.0} terms/sec)", total_terms, duration.as_millis(), terms_per_sec);
       
       Ok(())
   }
   
   fn simple_stem(word: &str) -> String {
       // Simple stemming - remove common suffixes
       if word.ends_with("ing") {
           word[..word.len()-3].to_string()
       } else if word.ends_with("ed") {
           word[..word.len()-2].to_string()
       } else {
           word.to_string()
       }
   }
   
   fn generate_bigrams(tokens: &[&str]) -> Vec<String> {
       tokens.windows(2).map(|window| format!("{} {}", window[0], window[1])).collect()
   }
   
   fn calculate_term_frequency(text: &str) -> HashMap<String, usize> {
       let mut tf = HashMap::new();
       for word in text.split_whitespace() {
           *tf.entry(word.to_lowercase()).or_insert(0) += 1;
       }
       tf
   }
   ```
2. Run: `cargo run --release --bin bench_text`
3. Commit: `git add src/bin/bench_text.rs && git commit -m "Benchmark text processing operations"`

## Success Criteria
- [ ] Text processing benchmark created
- [ ] Tokenization, stemming, n-gram performance measured
- [ ] Processing rates calculated
- [ ] Results committed

## Next Task
task_158_measure_vector_operations.md