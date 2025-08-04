# Task 159: Benchmark Index Operations

**Time: 9 minutes**

## Context
Vector operations benchmarked. Now measure index-specific CRUD performance for optimization analysis.

## Objective
Create index operations benchmark measuring insertion, deletion, update, and search performance.

## Implementation
Create `bench_index_ops.rs`:

```rust
struct SimpleIndex {
    documents: HashMap<u64, String>,
    term_index: HashMap<String, Vec<u64>>,
}

impl SimpleIndex {
    fn insert(&mut self, id: u64, content: &str) {
        self.documents.insert(id, content.to_string());
        for term in content.split_whitespace() {
            self.term_index.entry(term.to_lowercase()).or_default().push(id);
        }
    }
    
    fn delete(&mut self, id: u64) { /* Remove from both maps */ }
    fn update(&mut self, id: u64, content: &str) { /* Delete + Insert */ }
    fn search(&self, term: &str) -> Vec<u64> { /* Return doc IDs */ }
}

fn main() -> Result<()> {
    benchmark_insertions(1000)?;
    benchmark_deletions(1000)?;
    benchmark_updates(1000)?;
    benchmark_searches(10000, 100)?;
    Ok(())
}
```

## Success Criteria
- [ ] Benchmark measures insert, delete, update, search operations
- [ ] Reports latency and throughput for each operation type
- [ ] Results show baseline performance metrics
- [ ] Code compiles and runs: `cargo run --release --bin bench_index_ops`

## Next Task
Task 160: Validate Windows-specific performance characteristics