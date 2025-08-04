# Task 07: Implement Intelligent Chunk Overlap Calculation with Boundary Optimization

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 06 completed (AST boundary detection implemented)
**Input Files:**
- `C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs` (with boundary detection methods)
- `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` (all dependencies)

## Complete Context (For AI with ZERO Knowledge)

**What is Chunk Overlap?** Chunk overlap is a technique where adjacent text chunks share some common content at their boundaries. This prevents search queries from missing relevant results that span the boundary between two chunks. Without overlap, a search for "function definition" might miss results where "function" appears at the end of one chunk and "definition" at the start of the next.

**Why Boundary-Aware Overlap?** Traditional text chunking uses fixed-size overlap (e.g., always overlap 200 characters). However, this can split semantic units awkwardly. Boundary-aware overlap uses the AST boundaries identified in Task 06 to place overlaps at semantically safe positions, such as between functions or at the end of import statements.

**The Overlap Algorithm:**
1. **Chunk Size Calculation**: Start with max_chunk_size limit and find the largest boundary within that limit
2. **Overlap Position**: Calculate where overlap should begin, preferring semantic boundaries over arbitrary character positions
3. **Boundary Optimization**: Adjust overlap start/end to align with AST boundaries when possible
4. **Context Preservation**: Ensure enough context is preserved in overlaps for meaningful search results

**Example Scenario:**
```rust
fn function_a() { /* 500 chars */ }
fn function_b() { /* 400 chars */ }
struct Data { /* 300 chars */ }
```
With 800-char chunks and 100-char overlap:
- Chunk 1: `function_a() + function_b()` (900 chars, but boundary at function_b end)
- Chunk 2: `function_b() + struct Data` (overlap starts at function_b, not mid-function)

This approach maintains semantic integrity while providing robust search coverage across chunk boundaries.

## Exact Steps

1. **Navigate to project directory** (30 seconds):
```bash
cd C:/code/LLMKG/vectors/tantivy_search
```

2. **Add chunk creation method to src/chunker.rs** (5 minutes):
Add this EXACT method to the `impl SmartChunker` block in `C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs`:

```rust
    /// Create chunks with overlap from content and boundaries
    pub fn create_chunks(&self, content: &str, boundaries: &[usize]) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut i = 0;
        
        while i < boundaries.len() - 1 {
            let start = boundaries[i];
            let mut end = start + self.max_chunk_size;
            
            // Find the best boundary within max size
            let mut best_boundary = boundaries[i + 1];
            for &boundary in &boundaries[i + 1..] {
                if boundary <= end {
                    best_boundary = boundary;
                } else {
                    break;
                }
            }
            
            end = best_boundary;
            
            // Create overlap for next chunk (if not last chunk)
            let has_overlap = end < content.len();
            let overlap_start = if has_overlap {
                let overlap_pos = end.saturating_sub(self.overlap_size);
                // Find boundary closest to desired overlap position
                boundaries.iter()
                    .filter(|&&b| b >= overlap_pos && b < end)
                    .last()
                    .copied()
                    .unwrap_or(overlap_pos)
            } else {
                end
            };
            
            let chunk_content = content[start..end].to_string();
            chunks.push(Chunk::new(chunk_content, start, end, i > 0));
            
            // Find next start position
            i += 1;
            while i < boundaries.len() && boundaries[i] <= overlap_start {
                i += 1;
            }
            
            if i >= boundaries.len() - 1 {
                break;
            }
        }
        
        chunks
    }
```

3. **Add chunking test** (1 minute):
Add this EXACT test to the `mod tests` section in `C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs`:

```rust
    #[test]
    fn test_chunk_creation_with_overlap() -> Result<()> {
        let chunker = SmartChunker::new()?.with_sizes(100, 20);
        
        let boundaries = vec![0, 50, 80, 120, 150];
        let content = "a".repeat(150);
        
        let chunks = chunker.create_chunks(&content, &boundaries);
        
        // Should create multiple chunks
        assert!(chunks.len() >= 2);
        
        // First chunk should start at 0
        assert_eq!(chunks[0].start, 0);
        
        // Chunks should have reasonable sizes
        for chunk in &chunks {
            assert!(chunk.content.len() <= 120); // Max size + some tolerance
        }
        
        Ok(())
    }
```

4. **Verify compilation** (1 minute):
```bash
cargo check
```

5. **Run chunking test** (1 minute):
```bash
cargo test test_chunk_creation
```

## Verification Steps (2 minutes)

### Verify 1: Overlap calculation compiles successfully
```bash
cargo check
```
**Expected output:**
```
   Compiling tantivy_search v0.1.0 (C:\code\LLMKG\vectors\tantivy_search)
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
```

### Verify 2: Chunk creation test passes
```bash
cargo test test_chunk_creation_with_overlap -- --nocapture
```
**Expected output:**
```
test chunker::tests::test_chunk_creation_with_overlap ... ok
test result: ok. 1 passed; 0 failed; 0 ignored
```

### Verify 3: Integration with boundary detection works
```bash
cargo test chunker -- --nocapture | grep -E "boundary|chunk"
```
**Expected output:** All chunker tests pass, showing boundary detection and chunking work together

## If This Task Fails

**Error 1: "method `with_sizes` not found for struct `SmartChunker`"**
```bash
# Error: error[E0599]: no method named `with_sizes` found
# Solution: Add the with_sizes method to SmartChunker impl
pub fn with_sizes(mut self, max_chunk_size: usize, overlap_size: usize) -> Self {
    self.max_chunk_size = max_chunk_size;
    self.overlap_size = overlap_size;
    self
}
```

**Error 2: "index out of bounds" during chunk creation**
```bash
# Error: thread 'main' panicked at 'index out of bounds'
# Solution: Add bounds checking for content slicing
let chunk_content = if start < content.len() && end <= content.len() {
    content[start..end].to_string()
} else {
    String::new()
};
```

**Error 3: "infinite loop" in chunk creation algorithm**
```bash
# Error: Test timeout during chunk creation
# Solution: Add loop protection and progress validation
let mut iterations = 0;
const MAX_ITERATIONS: usize = 1000;
while i < boundaries.len() - 1 && iterations < MAX_ITERATIONS {
    iterations += 1;
    // ... existing loop body ...
}
```

**Error 4: "overlap calculation produces negative values"**
```bash
# Error: attempt to subtract with overflow
# Solution: Use saturating arithmetic for overlap calculations
let overlap_start = end.saturating_sub(self.overlap_size);
let safe_start = overlap_start.max(start);
```

**Error 5: "chunks have incorrect overlap metadata"**
```bash
# Error: Assertion failed - chunk.has_overlap should be true
# Solution: Fix overlap detection logic
let has_overlap = i > 0 && start < previous_chunk_end;
chunks.push(Chunk::new(chunk_content, start, end, has_overlap));
```

**Error 6: "semantic boundaries not respected in overlap"**
```bash
# Error: Overlap splits function definition
# Solution: Improve boundary selection in overlap calculation
let overlap_boundary = boundaries.iter()
    .filter(|&&b| b >= overlap_pos && b < end)
    .min_by_key(|&&b| (overlap_pos as i32 - b as i32).abs())
    .copied()
    .unwrap_or(overlap_pos);
```

## Troubleshooting Checklist
- [ ] Task 06 boundary detection is working correctly
- [ ] SmartChunker has max_chunk_size and overlap_size fields
- [ ] All array indexing uses proper bounds checking
- [ ] Chunk metadata (start, end, has_overlap) is calculated correctly
- [ ] Overlap positions respect semantic boundaries when possible
- [ ] Algorithm handles edge cases (empty content, single boundary, etc.)
- [ ] Memory usage is reasonable for large files

## Recovery Procedures

### Chunk Size Miscalculation
If chunks are too large or too small:
1. **Debug boundary selection**: Log which boundaries are being chosen
2. **Validate size limits**: Ensure max_chunk_size is respected
3. **Check boundary spacing**: Verify boundaries aren't too sparse or dense
4. **Test with known content**: Use predictable test data to verify algorithm

### Overlap Logic Errors
If overlaps are missing or incorrect:
1. **Trace overlap calculation**: Add logging to overlap position calculation
2. **Validate boundary filtering**: Ensure boundary selection logic is correct
3. **Check edge cases**: Test with content smaller than chunk size
4. **Verify metadata**: Ensure has_overlap flag is set correctly

### Performance Issues
If chunking is slow for large files:
1. **Profile boundary operations**: Check if boundary array operations are O(n²)
2. **Optimize boundary searches**: Use binary search instead of linear scan
3. **Limit chunk count**: Cap maximum chunks per file to prevent memory issues
4. **Stream processing**: Process files in segments rather than loading entirely

## Success Validation Checklist
- [ ] File `src/chunker.rs` contains `create_chunks` method implementation
- [ ] File `src/chunker.rs` contains `with_sizes` method for configuration
- [ ] Test `test_chunk_creation_with_overlap` exists and passes
- [ ] Chunks respect maximum size limits
- [ ] Overlaps are calculated at semantic boundaries when possible
- [ ] Chunk metadata (start, end, has_overlap) is accurate
- [ ] Algorithm handles edge cases without panicking
- [ ] Memory usage is reasonable for typical file sizes

## Files Created For Next Task

Task 08 expects these EXACT files to exist:
1. **C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs** - Enhanced with overlap calculation methods
2. **All previous files from Tasks 01-06** - Unchanged but required for compilation

## Context for Task 08

Task 08 will implement automatic language detection to determine which tree-sitter parser to use for boundary detection. The chunking system developed here will integrate with language detection to provide appropriate semantic chunking for different file types (Rust, Python, JavaScript, etc.). The boundary-aware overlap system ensures high-quality search results regardless of programming language.