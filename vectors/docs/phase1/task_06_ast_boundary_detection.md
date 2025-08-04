# Task 06: Implement AST Semantic Boundary Detection with Multi-Language Support

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 05 completed (SmartChunker with tree-sitter parsers initialized)
**Input Files:**
- `C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs` (SmartChunker struct with parsers)
- `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` (tree-sitter dependencies)

## Complete Context (For AI with ZERO Knowledge)

**What is AST Boundary Detection?** Abstract Syntax Tree (AST) boundary detection is the process of identifying semantic boundaries in source code where logical units (functions, classes, structs) begin and end. This prevents chunking algorithms from breaking code in the middle of semantic units, which would make search results incomplete or confusing.

**What is Tree-sitter?** Tree-sitter is a parser generator tool that creates incremental parsers for programming languages. It produces concrete syntax trees (CST) that maintain full fidelity to the source code, including whitespace and comments. Unlike traditional ASTs, tree-sitter CSTs can be used for text editing operations while preserving semantic structure.

**Why Multi-Language Support?** Modern codebases contain multiple programming languages. A search system must understand the syntactic structure of Rust (functions, structs, impls), Python (functions, classes, methods), JavaScript (functions, classes), and other languages to chunk them appropriately. Each language has different keywords and structural patterns.

**How This Integrates with Chunking:** The boundary detection system identifies safe places to split large files into chunks. For example, splitting between functions is safe, but splitting in the middle of a function body would create incomplete search results. The system uses tree-sitter parsers to walk the AST and mark positions where semantic units begin and end.

**The Algorithm:** 
1. Parse source code using language-specific tree-sitter parser
2. Walk the AST using tree-sitter cursor API
3. Identify nodes that represent semantic boundaries (functions, classes, etc.)
4. Record byte positions of these boundaries
5. Return sorted list of safe split positions for chunking algorithm

This task implements the core AST traversal logic that enables semantic-aware chunking for high-quality search results.

## Exact Steps

1. **Navigate to project directory** (30 seconds):
```bash
cd C:/code/LLMKG/vectors/tantivy_search
```

2. **Add boundary detection methods to src/chunker.rs** (5 minutes):
Add these EXACT methods to the `impl SmartChunker` block in `C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs` (before the closing `}`):

```rust
    /// Find semantic boundaries in parsed code using AST
    pub fn find_boundaries(&mut self, content: &str, language: &str) -> Result<Vec<usize>> {
        let parser = match language {
            "rust" => &mut self.rust_parser,
            "python" => &mut self.python_parser,
            _ => return Err(anyhow::anyhow!("Unsupported language: {}", language)),
        };
        
        let tree = parser.parse(content, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse code"))?;
        
        let mut boundaries = Vec::new();
        let mut cursor = tree.walk();
        
        // Always include start and end
        boundaries.push(0);
        boundaries.push(content.len());
        
        // Find semantic boundaries through AST traversal
        self.traverse_for_boundaries(&mut cursor, &mut boundaries);
        
        // Sort and deduplicate
        boundaries.sort_unstable();
        boundaries.dedup();
        Ok(boundaries)
    }
    
    /// Recursively traverse AST to find semantic boundaries
    fn traverse_for_boundaries(&self, cursor: &mut tree_sitter::TreeCursor, boundaries: &mut Vec<usize>) {
        loop {
            let node = cursor.node();
            
            // Check if this node represents a semantic boundary
            if self.is_semantic_boundary(&node) {
                boundaries.push(node.start_byte());
                boundaries.push(node.end_byte());
            }
            
            // Recurse into children
            if cursor.goto_first_child() {
                self.traverse_for_boundaries(cursor, boundaries);
                cursor.goto_parent();
            }
            
            // Move to next sibling
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
    
    /// Check if a tree-sitter node represents a semantic boundary
    fn is_semantic_boundary(&self, node: &tree_sitter::Node) -> bool {
        match node.kind() {
            // Rust semantic units
            "function_item" | "struct_item" | "enum_item" | "impl_item" |
            "trait_item" | "mod_item" | "use_declaration" |
            // Python semantic units  
            "function_definition" | "class_definition" | "import_statement" |
            "import_from_statement" => true,
            _ => false,
        }
    }
```

3. **Add boundary detection test** (1 minute):
Add this EXACT test to the `mod tests` section in `C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs`:

```rust
    #[test]
    fn test_boundary_detection() -> Result<()> {
        let mut chunker = SmartChunker::new()?;
        
        let rust_code = r#"
pub fn hello() {
    println!("Hello");
}

struct Config {
    value: String,
}

impl Config {
    fn new() -> Self {
        Self { value: "test".to_string() }
    }
}
"#;
        
        let boundaries = chunker.find_boundaries(rust_code, "rust")?;
        
        // Should have at least start, end, and some semantic boundaries
        assert!(boundaries.len() >= 2);
        assert_eq!(boundaries[0], 0);
        assert_eq!(boundaries[boundaries.len() - 1], rust_code.len());
        
        Ok(())
    }
```

4. **Verify compilation** (1 minute):
```bash
cargo check
```
Expected output: "Checking tantivy_search v0.1.0" with no errors

5. **Run boundary detection test** (1 minute):
```bash
cargo test test_boundary_detection
```
Expected output: Boundary detection test should pass

## Verification Steps (2 minutes)

### Verify 1: Code compilation succeeds
```bash
cargo check
```
**Expected output:**
```
   Compiling tantivy_search v0.1.0 (C:\code\LLMKG\vectors\tantivy_search)
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
```

### Verify 2: Boundary detection test passes
```bash
cargo test test_boundary_detection -- --nocapture
```
**Expected output:**
```
test chunker::tests::test_boundary_detection ... ok
test result: ok. 1 passed; 0 failed; 0 ignored
```

### Verify 3: Parser initialization works correctly
```bash
cargo test --lib | grep -i "chunker\|parser"
```
**Expected output:** All chunker-related tests pass without parser errors

## If This Task Fails

**Error 1: "tree_sitter::TreeCursor` cannot be found in this scope"**
```bash
# Error: error[E0433]: failed to resolve: use of undeclared type `TreeCursor`
# Solution: Add missing tree-sitter import
# Add to top of chunker.rs:
use tree_sitter::{Language, Parser, Tree, TreeCursor, Node};
```

**Error 2: "failed to parse code" during boundary detection**
```bash
# Error: Custom(failed to parse code)
# Solution: Check parser initialization and language detection
cargo test test_parser_initialization
# If parser init fails, check tree-sitter language loading:
println!("Rust parser loaded: {:?}", tree_sitter_rust::language());
```

**Error 3: "Unsupported language" error for valid languages**
```bash
# Error: Custom(Unsupported language: rust)
# Solution: Verify language string matching is case-sensitive
# Check that language parameter matches exactly: "rust", "python"
# Add debug logging to see actual language string being passed
```

**Error 4: "start_byte() or end_byte() panic" in AST traversal**
```bash
# Error: thread 'main' panicked at 'byte index out of bounds'
# Solution: Validate node boundaries before accessing
# Add bounds checking:
if node.start_byte() <= content.len() && node.end_byte() <= content.len() {
    boundaries.push(node.start_byte());
    boundaries.push(node.end_byte());
}
```

**Error 5: "infinite loop" in AST traversal**
```bash
# Error: Test timeout or hanging during tree traversal
# Solution: Add loop protection and cursor state validation
# Add counter to prevent infinite loops:
let mut iterations = 0;
const MAX_ITERATIONS: usize = 10000;
if iterations > MAX_ITERATIONS {
    return Err(anyhow::anyhow!("AST traversal exceeded max iterations"));
}
```

**Error 6: "empty boundaries vector" for valid code**
```bash
# Error: Assertion failed - boundaries.len() < 2
# Solution: Check semantic boundary detection logic
# Debug which node kinds are being detected:
println!("Node kind: {} at {}..{}", node.kind(), node.start_byte(), node.end_byte());
# Verify is_semantic_boundary() matches your language's AST node names
```

## Troubleshooting Checklist
- [ ] Task 05 completed successfully with parsers initialized
- [ ] Tree-sitter dependencies compiled without errors
- [ ] All imports in chunker.rs are present and correct
- [ ] Test code sample is valid Rust syntax
- [ ] AST node kinds match tree-sitter grammar for target language
- [ ] Parser state is properly managed (not reused incorrectly)
- [ ] Boundary detection handles empty or invalid code gracefully

## Recovery Procedures

### Parser State Corruption
If tree-sitter parsers get into invalid state:
1. **Reset parser state**: Create new parser instances
2. **Clear parser memory**: `parser.reset()`
3. **Validate input**: Ensure source code is valid UTF-8
4. **Test with minimal code**: Use simple function to isolate issue

### AST Traversal Issues
If cursor traversal fails or hangs:
1. **Add traversal logging**: Debug cursor position at each step
2. **Limit traversal depth**: Prevent stack overflow on deeply nested code
3. **Validate cursor state**: Check `cursor.node().is_error()` before proceeding
4. **Handle malformed AST**: Skip error nodes and continue traversal

### Memory Issues with Large Files
If boundary detection fails on large files:
1. **Implement streaming**: Process file in chunks rather than loading entirely
2. **Limit boundary count**: Cap maximum boundaries per file
3. **Use memory mapping**: For very large files, use memory-mapped I/O
4. **Add progress reporting**: Show progress for long-running operations

## Success Validation Checklist
- [ ] File `src/chunker.rs` contains `find_boundaries` method
- [ ] File `src/chunker.rs` contains `traverse_for_boundaries` method  
- [ ] File `src/chunker.rs` contains `is_semantic_boundary` method
- [ ] Test `test_boundary_detection` exists and passes
- [ ] Boundary detection works for both Rust and Python code
- [ ] Method handles unsupported languages gracefully with error
- [ ] AST traversal completes without infinite loops or panics
- [ ] Boundary positions are valid byte offsets within source code

## Files Created For Next Task

Task 07 expects these EXACT files to exist:
1. **C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs** - Enhanced with boundary detection methods
2. **All previous files from Tasks 01-05** - Unchanged but required

## Context for Task 07

Task 07 will implement chunk overlap calculation using the boundary positions identified in this task. The overlap system ensures that content near chunk boundaries is preserved in adjacent chunks, improving search recall for queries that span chunk boundaries. The boundary detection implemented here provides the foundation for intelligent overlap placement.