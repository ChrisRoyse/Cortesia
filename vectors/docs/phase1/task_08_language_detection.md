# Task 08: Implement Multi-Strategy Language Detection with Content Analysis

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 07 completed (intelligent chunk overlap calculation)
**Input Files:**
- `C:/code/LLMKG/vectors/tantivy_search/src/utils.rs` (basic utility stubs)
- `C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs` (boundary detection needing language input)

## Complete Context (For AI with ZERO Knowledge)

**What is Language Detection?** Language detection is the process of automatically identifying which programming language a source code file is written in. This is essential for selecting the correct tree-sitter parser for AST-based boundary detection. Without accurate language detection, the chunking system would use the wrong parser and miss semantic boundaries.

**Why Multi-Strategy Detection?** Files don't always have clear extensions (.txt files might contain Rust code, .config files might be TOML, etc.). A robust detection system uses multiple strategies:
1. **File Extension Analysis**: Fast first-pass detection based on .rs, .py, .js extensions
2. **Content Pattern Matching**: Fallback analysis looking for language-specific keywords and syntax
3. **Shebang Analysis**: Detection of #!/usr/bin/python type headers
4. **Statistical Analysis**: Counting language-specific patterns to determine likelihood

**Language-Specific Patterns:**
- **Rust**: `fn main()`, `pub struct`, `impl`, `use std::`, `Result<T, E>`, `Option<T>`
- **Python**: `def `, `import `, `from ... import`, `if __name__ == "__main__"`, `self.`
- **JavaScript**: `function`, `const`, `let`, `=>`, `require()`, `module.exports`
- **Markdown**: `#`, `##`, `**bold**`, `[link](url)`, ` ```code``` `

**Why This Matters for Search:** Accurate language detection enables:
- Proper AST parsing for semantic chunking
- Language-specific syntax highlighting in search results
- Targeted search within specific language contexts
- Better understanding of code structure for relevance ranking

The detection system balances speed (extension checking) with accuracy (content analysis) to provide reliable language identification for the chunking pipeline.

## Exact Steps

1. **Add language detection to src/utils.rs** (5 minutes):
Replace entire content of `C:/code/LLMKG/vectors/tantivy_search/src/utils.rs` with:

```rust
//! Utility functions and helpers

use anyhow::Result;
use std::path::Path;

/// Detect programming language from file path and content
pub fn detect_language(file_path: &Path, content: &str) -> String {
    // First try file extension
    if let Some(ext) = file_path.extension() {
        match ext.to_str() {
            Some("rs") => return "rust".to_string(),
            Some("py") => return "python".to_string(),
            _ => {}
        }
    }
    
    // Fallback to content-based detection
    detect_language_from_content(content)
}

/// Detect language from content patterns
fn detect_language_from_content(content: &str) -> String {
    let rust_patterns = ["fn main(", "pub fn", "struct ", "impl ", "use std::", "cargo"];
    let python_patterns = ["def ", "import ", "from ", "__init__", "if __name__"];
    
    let rust_score = rust_patterns.iter()
        .map(|&pattern| content.matches(pattern).count())
        .sum::<usize>();
    
    let python_score = python_patterns.iter()
        .map(|&pattern| content.matches(pattern).count())
        .sum::<usize>();
    
    if rust_score > python_score {
        "rust".to_string()
    } else if python_score > 0 {
        "python".to_string()
    } else {
        "text".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_language_detection_by_extension() {
        let rust_path = PathBuf::from("test.rs");
        let python_path = PathBuf::from("test.py");
        
        assert_eq!(detect_language(&rust_path, ""), "rust");
        assert_eq!(detect_language(&python_path, ""), "python");
    }
    
    #[test]
    fn test_language_detection_by_content() {
        let rust_content = "fn main() { println!(\"Hello\"); }";
        let python_content = "def main(): print(\"Hello\")";
        
        assert_eq!(detect_language_from_content(rust_content), "rust");
        assert_eq!(detect_language_from_content(python_content), "python");
    }
}
```

2. **Verify compilation** (1 minute):
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
```

3. **Test language detection** (1 minute):
```bash
cargo test test_language_detection
```

## Verification Steps (2 minutes)

### Verify 1: Language detection compiles successfully
```bash
cargo check
```
**Expected output:**
```
   Compiling tantivy_search v0.1.0 (C:\code\LLMKG\vectors\tantivy_search)
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
```

### Verify 2: Language detection tests pass
```bash
cargo test test_language_detection -- --nocapture
```
**Expected output:**
```
test utils::tests::test_language_detection_by_extension ... ok
test utils::tests::test_language_detection_by_content ... ok
test result: ok. 2 passed; 0 failed; 0 ignored
```

### Verify 3: Integration with chunker works
```bash
# Test that chunker can use detected language
cargo test boundary_detection -- --nocapture
```
**Expected output:** Boundary detection tests pass using language detection

## If This Task Fails

**Error 1: "PathBuf does not have method extension()"**
```bash
# Error: error[E0599]: no method named `extension` found for struct `PathBuf`
# Solution: Import Path trait
use std::path::{Path, PathBuf};
# Or use as_path():
file_path.as_path().extension()
```

**Error 2: "cannot find function `detect_language` in module"**
```bash
# Error: error[E0425]: cannot find function `detect_language`
# Solution: Export function in lib.rs
# Add to lib.rs:
pub use utils::detect_language;
```

**Error 3: "extension() returns Option<&OsStr> not &str"**
```bash
# Error: type mismatch in match statement
# Solution: Handle OsStr to str conversion properly
if let Some(ext) = file_path.extension() {
    if let Some(ext_str) = ext.to_str() {
        match ext_str {
            "rs" => return "rust".to_string(),
            // ...
        }
    }
}
```

**Error 4: "language detection returns incorrect results"**
```bash
# Error: Assertion failed - expected 'rust', got 'text'
# Solution: Debug pattern matching logic
println!("Rust score: {}, Python score: {}", rust_score, python_score);
# Check if patterns are being found in test content
# Verify pattern matching is case-sensitive
```

**Error 5: "content analysis is too slow for large files"**
```bash
# Error: Test timeout or performance issues
# Solution: Limit content analysis to first N characters
let analysis_content = if content.len() > 1000 {
    &content[..1000]
} else {
    content
};
detect_language_from_content(analysis_content)
```

**Error 6: "false positives in content detection"**
```bash
# Error: Comments containing language keywords cause misdetection
# Solution: Improve pattern matching to avoid comments
let rust_patterns = ["fn ", "pub fn", "struct ", "impl "];
// Use word boundaries or more specific patterns
let rust_score = rust_patterns.iter()
    .map(|&pattern| content.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .map(|line| line.matches(pattern).count())
        .sum::<usize>()
    ).sum::<usize>();
```

## Troubleshooting Checklist
- [ ] All path handling uses proper std::path types
- [ ] Content analysis patterns are language-specific and accurate
- [ ] Extension detection handles all supported file types
- [ ] Function is exported properly from utils module
- [ ] Test cases cover both extension-based and content-based detection
- [ ] Performance is acceptable for typical file sizes
- [ ] False positive rate is minimized through better pattern matching

## Recovery Procedures

### Pattern Matching Accuracy Issues
If language detection has high false positive/negative rates:
1. **Collect test samples**: Gather representative files from each language
2. **Analyze failure cases**: Identify which patterns cause misdetection
3. **Refine patterns**: Use more specific keywords that are unique to each language
4. **Add statistical weighting**: Weight patterns by their discriminative power

### Performance Optimization
If detection is too slow for large files:
1. **Limit analysis scope**: Only analyze first 1KB of file content
2. **Early termination**: Stop analysis once confidence threshold is reached
3. **Cache results**: Store language detection results to avoid recomputation
4. **Parallel processing**: Detect language while reading file in background

### Extension Handling Edge Cases
If extension detection fails for non-standard files:
1. **Add more extensions**: Support .config, .toml, .yaml extensions
2. **Check multiple extensions**: Handle .test.rs, .spec.py compound extensions
3. **Shebang detection**: Parse #!/usr/bin/python headers
4. **MIME type integration**: Use system MIME type detection if available

## Success Validation Checklist
- [ ] File `src/utils.rs` contains `detect_language` function
- [ ] File `src/utils.rs` contains `detect_language_from_content` function
- [ ] Function exported in `src/lib.rs` for use by other modules
- [ ] Test `test_language_detection_by_extension` passes
- [ ] Test `test_language_detection_by_content` passes
- [ ] Rust files (.rs) detected correctly
- [ ] Python files (.py) detected correctly
- [ ] Content-based fallback works for files without extensions
- [ ] Performance is acceptable for files up to 100KB

## Files Created For Next Task

Task 09 expects these EXACT files to exist:
1. **C:/code/LLMKG/vectors/tantivy_search/src/utils.rs** - Complete language detection implementation
2. **C:/code/LLMKG/vectors/tantivy_search/src/lib.rs** - Updated to export detect_language function
3. **All previous files from Tasks 01-07** - Required for document indexer integration

## Context for Task 09

Task 09 will implement the DocumentIndexer that orchestrates the entire pipeline: reading files, detecting their language, using appropriate parsers for boundary detection, creating semantic chunks with intelligent overlap, and indexing them in Tantivy. The language detection implemented here is the critical first step that enables all subsequent semantic processing.