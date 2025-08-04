# Task 11: Generate Edge Case Test Files

## Context
You are continuing the test data generation phase (Phase 0, Task 11). Task 10 created special character test files. Now you need to generate edge case test files that test boundary conditions, extreme inputs, and corner cases that could break the search system.

## Objective
Generate comprehensive edge case test files that challenge the limits of the search system, including empty files, very large files, malformed content, unicode edge cases, and boundary conditions.

## Requirements
1. Create empty and minimal content test files
2. Create very large test files (for performance testing)
3. Create malformed and broken syntax test files
4. Create unicode and encoding edge case files
5. Create chunk boundary test files
6. Create nested and deeply recursive content files

## Implementation for test_data.rs (extend existing)
```rust
impl SpecialCharTestGenerator {
    // ... existing methods ...
}

pub struct EdgeCaseTestGenerator;

impl EdgeCaseTestGenerator {
    /// Generate comprehensive edge case test files
    pub fn generate_edge_case_files() -> Result<()> {
        info!("Generating edge case test files");
        
        // Create test data directory
        std::fs::create_dir_all("test_data/edge_cases")?;
        
        // Generate different categories of edge case files
        Self::generate_empty_minimal_files()?;
        Self::generate_large_files()?;
        Self::generate_malformed_files()?;
        Self::generate_unicode_files()?;
        Self::generate_chunk_boundary_files()?;
        Self::generate_deeply_nested_files()?;
        Self::generate_encoding_files()?;
        Self::generate_extreme_cases()?;
        
        info!("Edge case test files generated successfully");
        Ok(())
    }
    
    fn generate_empty_minimal_files() -> Result<()> {
        debug!("Generating empty and minimal test files");
        
        let minimal_cases = vec![
            // Completely empty
            ("empty.txt", ""),
            ("empty.rs", ""),
            ("empty.toml", ""),
            
            // Single characters
            ("single_char.txt", "a"),
            ("single_bracket.rs", "["),
            ("single_paren.rs", "("),
            ("single_brace.rs", "{"),
            ("single_angle.rs", "<"),
            
            // Minimal valid Rust
            ("minimal.rs", "fn main(){}"),
            ("minimal_struct.rs", "struct A;"),
            ("minimal_enum.rs", "enum E{}"),
            ("minimal_trait.rs", "trait T{}"),
            ("minimal_impl.rs", "impl A{}"),
            
            // Minimal valid Cargo.toml
            ("minimal.toml", r#"[package]
name = "a"
version = "0.1.0""#),
            
            // Just whitespace
            ("whitespace_only.txt", "   \n\t\r\n   "),
            ("newlines_only.txt", "\n\n\n\n\n"),
            ("tabs_only.txt", "\t\t\t\t\t"),
            
            // Single tokens
            ("single_keyword.rs", "pub"),
            ("single_type.rs", "String"),
            ("single_generic.rs", "T"),
            ("single_lifetime.rs", "'static"),
        ];
        
        for (filename, content) in minimal_cases {
            let path = Path::new("test_data/edge_cases").join(filename);
            fs::write(path, content)?;
            debug!("Created minimal test file: {}", filename);
        }
        
        Ok(())
    }
    
    fn generate_large_files() -> Result<()> {
        debug!("Generating large test files");
        
        // Large repetitive file (10MB)
        let large_content = "pub fn function_".to_string() + 
            &(0..100_000)
                .map(|i| format!("{}() -> i32 {{ {} }}\n", i, i))
                .collect::<String>();
        
        let path = Path::new("test_data/edge_cases").join("large_repetitive.rs");
        fs::write(path, large_content)?;
        debug!("Created large repetitive file (~10MB)");
        
        // Large single line (1MB single line)
        let single_line = format!(
            "pub static LARGE_ARRAY: [i32; 100000] = [{}];",
            (0..100_000).map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
        );
        
        let path = Path::new("test_data/edge_cases").join("single_huge_line.rs");
        fs::write(path, single_line)?;
        debug!("Created single huge line file (~1MB)");
        
        // Large nested structure
        let mut nested_content = String::new();
        nested_content.push_str("pub mod root {\n");
        
        for i in 0..1000 {
            nested_content.push_str(&format!("    pub mod module_{} {{\n", i));
            for j in 0..10 {
                nested_content.push_str(&format!(
                    "        pub fn function_{}_{}_{}() -> Result<String, Error> {{\n",
                    i, j, "very_long_function_name_that_repeats"
                ));
                nested_content.push_str(&format!(
                    "            Ok(\"result_{}_{}\".to_string())\n",
                    i, j
                ));
                nested_content.push_str("        }\n");
            }
            nested_content.push_str("    }\n");
        }
        
        nested_content.push_str("}\n");
        
        let path = Path::new("test_data/edge_cases").join("large_nested.rs");
        fs::write(path, nested_content)?;
        debug!("Created large nested structure file");
        
        Ok(())
    }
    
    fn generate_malformed_files() -> Result<()> {
        debug!("Generating malformed syntax test files");
        
        let malformed_cases = vec![
            // Unmatched brackets
            ("unmatched_bracket.rs", "fn test() { if true { println!(\"test\");"),
            ("unmatched_paren.rs", "fn test(a: i32, b: i32 { a + b }"),
            ("unmatched_angle.rs", "struct Test<T { value: T }"),
            
            // Incomplete syntax
            ("incomplete_fn.rs", "pub fn incomplete("),
            ("incomplete_struct.rs", "struct Data { field: String"),
            ("incomplete_impl.rs", "impl<T> MyTrait for"),
            ("incomplete_generic.rs", "Result<T,"),
            
            // Invalid tokens
            ("invalid_chars.rs", "fn test() { let x = @#$%^; }"),
            ("invalid_keywords.rs", "invalid_keyword struct Test {}"),
            ("invalid_operators.rs", "let x = 1 +++ 2 --- 3;"),
            
            // Mixed languages (invalid Rust)
            ("mixed_languages.rs", r#"
fn rust_function() -> i32 { 42 }
function jsFunction() { return 42; }  // JavaScript in Rust file
def python_function(): return 42     # Python in Rust file
            "#),
            
            // Broken Cargo.toml
            ("broken.toml", r#"
[package
name = "broken"
version = 0.1.0"
[dependencies]
serde = { version = "1.0", features = ["derive"
            "#),
            
            // Unterminated strings and comments
            ("unterminated_string.rs", r#"fn test() { let s = "unterminated string"#),
            ("unterminated_comment.rs", "fn test() { /* unterminated comment"),
            
            // Circular references (textual)
            ("circular.rs", r#"
use crate::circular;
mod circular {
    use super::circular;
    pub fn recurse() { circular::recurse(); }
}
            "#),
        ];
        
        for (filename, content) in malformed_cases {
            let path = Path::new("test_data/edge_cases").join(filename);
            fs::write(path, content)?;
            debug!("Created malformed test file: {}", filename);
        }
        
        Ok(())
    }
    
    fn generate_unicode_files() -> Result<()> {
        debug!("Generating unicode edge case test files");
        
        let unicode_cases = vec![
            // Different scripts
            ("chinese.rs", r#"
// 中文注释
pub fn 函数名() -> String {
    "你好世界".to_string()
}
            "#),
            
            ("russian.rs", r#"
// Русский комментарий
pub fn функция() -> String {
    "Привет мир".to_string()
}
            "#),
            
            ("arabic.rs", r#"
// تعليق عربي
pub fn وظيفة() -> String {
    "مرحبا بالعالم".to_string()
}
            "#),
            
            ("japanese.rs", r#"
// 日本語のコメント
pub fn 関数() -> String {
    "こんにちは世界".to_string()
}
            "#),
            
            // Mathematical symbols
            ("math_symbols.rs", r#"
// Mathematical notation: ∀x∈ℝ, ∃y∈ℕ: x < y
pub fn mathematical_function() -> f64 {
    std::f64::consts::PI * 2.0  // π * 2
}

// Greek letters often used in math
pub struct ΔData {
    pub α: f64,  // alpha
    pub β: f64,  // beta  
    pub γ: f64,  // gamma
    pub λ: f64,  // lambda
}
            "#),
            
            // Emojis and symbols
            ("emojis.rs", r#"
// Emojis in code 🦀🔍💻
pub fn emoji_function() -> String {
    "Search 🔍 in Rust 🦀 code 💻".to_string()
}

pub static EMOJI_ARRAY: &[&str] = &[
    "🚀", "⚡", "🔥", "💡", "🎯", "📊", "🔧", "🛠️"
];
            "#),
            
            // Zero-width and control characters
            ("zero_width.rs", format!(
                "pub fn function_with_zero_width{}characters() -> i32 {{ 42 }}",
                "\u{200B}\u{200C}\u{200D}"  // Zero-width space, ZWNJ, ZWJ
            )),
            
            // Combining characters
            ("combining.rs", "pub fn café_function() -> String { \"naïve\".to_string() }"),
            
            // Unusual whitespace
            ("unusual_whitespace.rs", format!(
                "pub{}fn{}unusual{}spacing() -> i32 {{ 42 }}",
                "\u{2000}",  // EN QUAD
                "\u{2001}",  // EM QUAD  
                "\u{2002}"   // EN SPACE
            )),
            
            // BOM (Byte Order Mark)
            ("with_bom.rs", format!("{}pub fn with_bom() -> i32 {{ 42 }}", "\u{FEFF}")),
            
            // Mixed encodings simulation
            ("mixed_encoding.rs", r#"
// This file contains mixed encoding-like content
pub fn ascii_function() -> &'static str { "ASCII" }
pub fn latin1_café() -> &'static str { "café" }  
pub fn utf8_функция() -> &'static str { "функция" }
pub fn emoji_🦀() -> &'static str { "🦀" }
            "#),
        ];
        
        for (filename, content) in unicode_cases {
            let path = Path::new("test_data/edge_cases").join(filename);
            fs::write(path, content)?;
            debug!("Created unicode test file: {}", filename);
        }
        
        Ok(())
    }
    
    fn generate_chunk_boundary_files() -> Result<()> {
        debug!("Generating chunk boundary test files");
        
        // Test files designed to test chunking at specific boundaries
        let chunk_size = 1000; // Assume 1KB chunks for testing
        
        // Function that spans chunk boundary
        let padding = "x".repeat(chunk_size - 50);
        let boundary_function = format!(
            r#"
// Padding to approach chunk boundary
// {}
pub fn function_that_spans_chunk_boundary() -> Result<String, Error> {{
    // This function intentionally spans a chunk boundary
    let result = "test".to_string();
    Ok(result)
}}
            "#,
            padding
        );
        
        let path = Path::new("test_data/edge_cases").join("chunk_boundary_function.rs");
        fs::write(path, boundary_function)?;
        
        // Struct that gets split
        let struct_content = format!(
            r#"
// {}
#[derive(Debug, Clone)]
pub struct ChunkBoundaryStruct {{
    pub field1: String,
    pub field2: i32,
    pub field3: Vec<String>,
}}
            "#,
            "x".repeat(chunk_size - 100)
        );
        
        let path = Path::new("test_data/edge_cases").join("chunk_boundary_struct.rs");
        fs::write(path, struct_content)?;
        
        // Generic that gets split
        let generic_content = format!(
            r#"
// {}
pub fn generic_function<T, E>() -> Result<T, E>
where
    T: Clone + Send + Sync + 'static,
    E: std::error::Error + Send + Sync + 'static,
{{
    todo!()
}}
            "#,
            "x".repeat(chunk_size - 150)
        );
        
        let path = Path::new("test_data/edge_cases").join("chunk_boundary_generic.rs");
        fs::write(path, generic_content)?;
        
        debug!("Created chunk boundary test files");
        Ok(())
    }
    
    fn generate_deeply_nested_files() -> Result<()> {
        debug!("Generating deeply nested test files");
        
        // Deeply nested modules
        let mut nested_modules = String::new();
        let depth = 50;
        
        for i in 0..depth {
            nested_modules.push_str(&format!("{}pub mod level_{} {{\n", "    ".repeat(i), i));
        }
        
        nested_modules.push_str(&format!(
            "{}pub fn deeply_nested_function() -> i32 {{ {} }}\n",
            "    ".repeat(depth),
            depth
        ));
        
        for i in (0..depth).rev() {
            nested_modules.push_str(&format!("{}}}\n", "    ".repeat(i)));
        }
        
        let path = Path::new("test_data/edge_cases").join("deeply_nested_modules.rs");
        fs::write(path, nested_modules)?;
        
        // Deeply nested generic types
        let mut nested_generics = "Result<".to_string();
        for i in 0..20 {
            nested_generics.push_str(&format!("Option<HashMap<String, Vec<Result<{}, Error", i));
        }
        
        // Close all the generics
        for _ in 0..20 {
            nested_generics.push_str(">>>>");
        }
        
        let generic_content = format!(
            r#"
pub type DeepType = {};

pub fn function_with_deep_type() -> DeepType {{
    todo!()
}}
            "#,
            nested_generics
        );
        
        let path = Path::new("test_data/edge_cases").join("deeply_nested_generics.rs");
        fs::write(path, generic_content)?;
        
        // Deeply nested match statements
        let mut nested_match = String::new();
        nested_match.push_str("pub fn deeply_nested_match(input: i32) -> String {\n");
        nested_match.push_str("    match input {\n");
        
        for i in 0..30 {
            nested_match.push_str(&format!(
                "        {} => match input % {} {{\n",
                i, i + 1
            ));
            for j in 0..5 {
                nested_match.push_str(&format!(
                    "            {} => \"level_{}_{}\".to_string(),\n",
                    j, i, j
                ));
            }
            nested_match.push_str("            _ => match input {\n");
        }
        
        nested_match.push_str(&format!(
            "{}_ => \"deeply_nested\".to_string(),\n",
            "                ".repeat(30)
        ));
        
        // Close all the matches
        for _ in 0..31 {
            nested_match.push_str("            },\n        },\n");
        }
        nested_match.push_str("    }\n}\n");
        
        let path = Path::new("test_data/edge_cases").join("deeply_nested_match.rs");
        fs::write(path, nested_match)?;
        
        debug!("Created deeply nested test files");
        Ok(())
    }
    
    fn generate_encoding_files() -> Result<()> {
        debug!("Generating encoding edge case files");
        
        // Files with different line endings
        let content = "pub fn test() -> i32 { 42 }";
        
        // Unix line endings (LF)
        let unix_content = content.replace("\r\n", "\n").replace("\r", "\n");
        let path = Path::new("test_data/edge_cases").join("unix_endings.rs");
        fs::write(path, unix_content)?;
        
        // Windows line endings (CRLF)
        let windows_content = content.replace("\n", "\r\n");
        let path = Path::new("test_data/edge_cases").join("windows_endings.rs");
        fs::write(path, windows_content.as_bytes())?;
        
        // Old Mac line endings (CR)
        let mac_content = content.replace("\n", "\r");
        let path = Path::new("test_data/edge_cases").join("mac_endings.rs");
        fs::write(path, mac_content)?;
        
        // Mixed line endings
        let mixed_content = "pub fn test1() -> i32 { 42 }\r\npub fn test2() -> i32 { 43 }\npub fn test3() -> i32 { 44 }\r";
        let path = Path::new("test_data/edge_cases").join("mixed_endings.rs");
        fs::write(path, mixed_content.as_bytes())?;
        
        debug!("Created encoding test files");
        Ok(())
    }
    
    fn generate_extreme_cases() -> Result<()> {
        debug!("Generating extreme edge cases");
        
        // File with only special characters
        let special_only = "[]{}<>()#@$%^&*!?~`|\\\"';:,./+-=_";
        let path = Path::new("test_data/edge_cases").join("special_chars_only.txt");
        fs::write(path, special_only)?;
        
        // File with repeating patterns
        let repeating = "Result<T, E>".repeat(10000);
        let path = Path::new("test_data/edge_cases").join("repeating_pattern.rs");
        fs::write(path, repeating)?;
        
        // File with very long identifier
        let long_identifier = format!(
            "pub fn {}() -> i32 {{ 42 }}",
            "very_long_function_name_that_keeps_going_and_going_".repeat(100)
        );
        let path = Path::new("test_data/edge_cases").join("long_identifier.rs");
        fs::write(path, long_identifier)?;
        
        // Binary-like content (not actually binary, but looks like it)
        let binary_like = (0..256u8)
            .cycle()
            .take(10000)
            .map(|b| format!("\\x{:02x}", b))
            .collect::<String>();
        let binary_content = format!("pub static BINARY_DATA: &str = \"{}\";", binary_like);
        let path = Path::new("test_data/edge_cases").join("binary_like.rs");
        fs::write(path, binary_content)?;
        
        debug!("Created extreme edge case files");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_edge_case_files() {
        EdgeCaseTestGenerator::generate_edge_case_files().unwrap();
        
        // Verify some key files were created
        assert!(Path::new("test_data/edge_cases/empty.txt").exists());
        assert!(Path::new("test_data/edge_cases/large_repetitive.rs").exists());
        assert!(Path::new("test_data/edge_cases/unmatched_bracket.rs").exists());
        assert!(Path::new("test_data/edge_cases/chinese.rs").exists());
        assert!(Path::new("test_data/edge_cases/deeply_nested_modules.rs").exists());
    }
    
    #[test]
    fn test_edge_case_file_sizes() {
        EdgeCaseTestGenerator::generate_edge_case_files().unwrap();
        
        // Check that large files are actually large
        let large_file = fs::metadata("test_data/edge_cases/large_repetitive.rs").unwrap();
        assert!(large_file.len() > 1_000_000, "Large file should be > 1MB");
        
        // Check that empty file is actually empty
        let empty_file = fs::metadata("test_data/edge_cases/empty.txt").unwrap();
        assert_eq!(empty_file.len(), 0, "Empty file should be 0 bytes");
    }
}
```

## Implementation Steps
1. Add EdgeCaseTestGenerator struct to test_data.rs
2. Implement generate_empty_minimal_files() for boundary cases
3. Implement generate_large_files() for performance testing
4. Implement generate_malformed_files() for error handling
5. Implement generate_unicode_files() for encoding issues
6. Implement generate_chunk_boundary_files() for chunking edge cases
7. Implement generate_deeply_nested_files() for parser limits
8. Implement generate_encoding_files() for line ending issues
9. Implement generate_extreme_cases() for stress testing

## Success Criteria
- [ ] EdgeCaseTestGenerator struct implemented and compiling
- [ ] Empty and minimal files created (0-10 bytes)
- [ ] Large files created (>1MB for performance testing)
- [ ] Malformed syntax files that should not crash the parser
- [ ] Unicode files with various scripts and symbols
- [ ] Chunk boundary files that test splitting behavior
- [ ] Deeply nested files that test parser recursion limits
- [ ] Different encoding and line ending files
- [ ] Extreme cases for stress testing

## Test Command
```bash
cargo test test_generate_edge_case_files
cargo test test_edge_case_file_sizes
ls -la test_data/edge_cases/
```

## Generated Files Categories
After completion, these file types should exist:

**Empty/Minimal:** empty.txt, single_char.txt, minimal.rs
**Large Files:** large_repetitive.rs (~10MB), single_huge_line.rs (~1MB)
**Malformed:** unmatched_bracket.rs, incomplete_fn.rs, broken.toml
**Unicode:** chinese.rs, russian.rs, emojis.rs, math_symbols.rs
**Boundaries:** chunk_boundary_function.rs, chunk_boundary_struct.rs
**Nested:** deeply_nested_modules.rs, deeply_nested_generics.rs
**Encoding:** unix_endings.rs, windows_endings.rs, mixed_endings.rs
**Extreme:** special_chars_only.txt, repeating_pattern.rs

## Time Estimate
10 minutes

## Next Task
Task 12: Generate chunk boundary test files for semantic parsing validation.