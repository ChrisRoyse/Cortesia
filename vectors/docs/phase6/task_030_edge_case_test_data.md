# Task 030: Generate Edge Case Test Data

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 010-029. Edge case testing is critical for ensuring the search system handles unusual inputs, boundary conditions, and potential failure scenarios without crashing or producing incorrect results.

## Project Structure
```
src/
  validation/
    test_data.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `generate_edge_case_tests()` method that creates test files specifically designed to expose edge cases, boundary conditions, and potential failure modes in the search indexing system.

## Requirements
1. Add to existing `src/validation/test_data.rs`
2. Generate files with zero-length content and extremely long lines
3. Create files with boundary conditions (1-byte, exactly 64KB, etc.)
4. Include files with unusual character encodings and byte sequences
5. Test files with no newlines, only newlines, and mixed line endings
6. Generate files that test memory and buffer edge cases
7. Include search patterns that test regex engine limits

## Expected Code Structure to Add
```rust
impl TestDataGenerator {
    fn generate_edge_case_tests(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // Empty file - zero bytes
        let empty_content = String::new();
        let mut empty_file = self.create_test_file("edge_empty.txt", &empty_content, TestFileType::EdgeCase)?;
        empty_file.expected_matches = vec![]; // No matches possible in empty file
        files.push(empty_file);
        
        // Single character file
        let single_char = "a";
        let mut single_file = self.create_test_file("edge_single_char.txt", single_char, TestFileType::EdgeCase)?;
        single_file.expected_matches = vec!["a".to_string()];
        files.push(single_file);
        
        // File with only whitespace
        let whitespace_content = self.generate_whitespace_only_content()?;
        let mut whitespace_file = self.create_test_file("edge_whitespace_only.txt", &whitespace_content, TestFileType::EdgeCase)?;
        whitespace_file.expected_matches = vec![
            " ".to_string(),
            "\t".to_string(),
            "\n".to_string(),
            "\\s+".to_string(), // Regex for whitespace
        ];
        files.push(whitespace_file);
        
        // Extremely long single line (no newlines)
        let long_line_content = self.generate_extremely_long_line()?;
        let mut long_line_file = self.create_test_file("edge_long_line.txt", &long_line_content, TestFileType::EdgeCase)?;
        long_line_file.expected_matches = vec![
            "START_MARKER".to_string(),
            "MIDDLE_MARKER".to_string(),
            "END_MARKER".to_string(),
            "REPEATED_PATTERN".to_string(),
        ];
        files.push(long_line_file);
        
        // File with only newlines
        let newlines_only = self.generate_newlines_only_content()?;
        let mut newlines_file = self.create_test_file("edge_newlines_only.txt", &newlines_only, TestFileType::EdgeCase)?;
        newlines_file.expected_matches = vec![
            "\n".to_string(),
            "^$".to_string(), // Empty lines
        ];
        files.push(newlines_file);
        
        // Mixed line endings (Windows, Unix, Mac classic)
        let mixed_endings_content = self.generate_mixed_line_endings()?;
        let mut mixed_endings_file = self.create_test_file("edge_mixed_endings.txt", &mixed_endings_content, TestFileType::EdgeCase)?;
        mixed_endings_file.expected_matches = vec![
            "line_unix".to_string(),
            "line_windows".to_string(),
            "line_mac".to_string(),
            "\r\n".to_string(),
            "\r".to_string(),
        ];
        files.push(mixed_endings_file);
        
        // Binary-like content with null bytes
        let binary_content = self.generate_binary_like_content()?;
        let mut binary_file = self.create_test_file("edge_binary_like.bin", &binary_content, TestFileType::EdgeCase)?;
        binary_file.expected_matches = vec![
            "TEXT_MARKER".to_string(),
            "BINARY_SECTION".to_string(),
        ];
        files.push(binary_file);
        
        // Boundary size files
        let boundary_files = self.generate_boundary_size_files()?;
        files.extend(boundary_files);
        
        // Unicode edge cases
        let unicode_edge_files = self.generate_unicode_edge_cases()?;
        files.extend(unicode_edge_files);
        
        // Pathological regex patterns
        let regex_edge_files = self.generate_regex_edge_cases()?;
        files.extend(regex_edge_files);
        
        Ok(files)
    }
    
    /// Generate content with various whitespace characters only
    fn generate_whitespace_only_content(&self) -> Result<String> {
        let mut content = String::new();
        
        // Regular spaces
        content.push_str("    ");
        content.push('\n');
        
        // Tabs
        content.push_str("\t\t\t");
        content.push('\n');
        
        // Mixed spaces and tabs
        content.push_str(" \t \t ");
        content.push('\n');
        
        // Unicode whitespace characters
        content.push('\u{00A0}'); // Non-breaking space
        content.push('\u{2000}'); // En quad
        content.push('\u{2001}'); // Em quad
        content.push('\u{2002}'); // En space
        content.push('\u{2003}'); // Em space
        content.push('\u{2004}'); // Three-per-em space
        content.push('\u{2005}'); // Four-per-em space
        content.push('\u{2006}'); // Six-per-em space
        content.push('\u{2007}'); // Figure space
        content.push('\u{2008}'); // Punctuation space
        content.push('\u{2009}'); // Thin space
        content.push('\u{200A}'); // Hair space
        content.push('\n');
        
        // Vertical whitespace
        content.push('\u{000B}'); // Vertical tab
        content.push('\u{000C}'); // Form feed
        content.push('\u{000D}'); // Carriage return
        content.push('\u{0085}'); // Next line
        content.push('\u{2028}'); // Line separator
        content.push('\u{2029}'); // Paragraph separator
        
        Ok(content)
    }
    
    /// Generate extremely long single line (1MB+)
    fn generate_extremely_long_line(&self) -> Result<String> {
        let mut content = String::with_capacity(2_000_000);
        
        content.push_str("START_MARKER");
        
        // Generate repetitive pattern that could stress regex engines
        for i in 0..50_000 {
            content.push_str(&format!("REPEATED_PATTERN_{:05}_", i));
            
            // Add complexity to prevent simple optimizations
            if i % 1000 == 0 {
                content.push_str("MIDDLE_MARKER_");
            }
            
            if i % 10 == 0 {
                content.push_str("abcdefghijklmnopqrstuvwxyz");
            }
            
            if i % 100 == 0 {
                content.push_str("0123456789");
            }
        }
        
        content.push_str("END_MARKER");
        
        // Ensure no newlines in the entire content
        assert!(!content.contains('\n'));
        assert!(content.len() > 1_000_000);
        
        Ok(content)
    }
    
    /// Generate file with only newlines of different types
    fn generate_newlines_only_content(&self) -> Result<String> {
        let mut content = String::new();
        
        // Unix newlines
        for _ in 0..100 {
            content.push('\n');
        }
        
        // Windows newlines
        for _ in 0..50 {
            content.push_str("\r\n");
        }
        
        // Mac classic newlines
        for _ in 0..25 {
            content.push('\r');
        }
        
        // Mixed Unicode line separators
        content.push('\u{2028}'); // Line separator
        content.push('\u{2029}'); // Paragraph separator
        content.push('\u{0085}'); // Next line
        
        Ok(content)
    }
    
    /// Generate content with mixed line endings
    fn generate_mixed_line_endings(&self) -> Result<String> {
        let mut content = String::new();
        
        content.push_str("line_unix");
        content.push('\n');
        
        content.push_str("line_windows");
        content.push_str("\r\n");
        
        content.push_str("line_mac");
        content.push('\r');
        
        content.push_str("line_unicode_ls");
        content.push('\u{2028}');
        
        content.push_str("line_unicode_ps");
        content.push('\u{2029}');
        
        content.push_str("line_nel");
        content.push('\u{0085}');
        
        // Line with no ending (EOF)
        content.push_str("line_no_ending");
        
        Ok(content)
    }
    
    /// Generate binary-like content with embedded text
    fn generate_binary_like_content(&self) -> Result<String> {
        let mut content = String::new();
        
        // Start with some binary-looking data
        for i in 0..256 {
            if i % 32 == 0 {
                content.push_str("TEXT_MARKER");
            }
            
            // Add some control characters and high-bit characters
            if (i as u8) < 32 || (i as u8) > 126 {
                // Skip null bytes and other problematic characters for now
                if i != 0 {
                    content.push(char::from(i as u8));
                }
            } else {
                content.push(char::from(i as u8));
            }
            
            if i % 64 == 63 {
                content.push_str("BINARY_SECTION\n");
            }
        }
        
        // Add some readable text mixed in
        content.push_str("\nThis is readable text mixed with binary data\n");
        content.push_str("TEXT_MARKER: Another searchable pattern\n");
        
        // Add more binary-like patterns
        for i in 0..100 {
            content.push_str(&format!("\x01\x02BINARY_SECTION_{:03}\x03\x04", i));
            if i % 10 == 9 {
                content.push('\n');
            }
        }
        
        Ok(content)
    }
    
    /// Generate files with specific boundary sizes
    fn generate_boundary_size_files(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // Exactly 1 byte
        let one_byte = "X";
        let mut one_byte_file = self.create_test_file("edge_1_byte.txt", one_byte, TestFileType::EdgeCase)?;
        one_byte_file.expected_matches = vec!["X".to_string()];
        files.push(one_byte_file);
        
        // Exactly 4KB (common page size)
        let four_kb_content = "A".repeat(4096);
        let mut four_kb_file = self.create_test_file("edge_4kb.txt", &four_kb_content, TestFileType::EdgeCase)?;
        four_kb_file.expected_matches = vec!["A".to_string(), "AA".to_string()];
        files.push(four_kb_file);
        
        // Exactly 64KB (common buffer size)
        let mut sixty_four_kb_content = String::with_capacity(65536);
        for i in 0..(65536 / 32) {
            sixty_four_kb_content.push_str(&format!("BOUNDARY_TEST_PATTERN_{:08}", i));
        }
        let mut sixty_four_kb_file = self.create_test_file("edge_64kb.txt", &sixty_four_kb_content, TestFileType::EdgeCase)?;
        sixty_four_kb_file.expected_matches = vec![
            "BOUNDARY_TEST_PATTERN_".to_string(),
            "BOUNDARY_TEST_PATTERN_00000000".to_string(),
        ];
        files.push(sixty_four_kb_file);
        
        // Exactly 1MB - 1 byte
        let mb_minus_one_size = 1024 * 1024 - 1;
        let mut mb_minus_one_content = "B".repeat(mb_minus_one_size);
        mb_minus_one_content.push('Z'); // Last character is different
        let mut mb_minus_one_file = self.create_test_file("edge_1mb_minus_1.txt", &mb_minus_one_content, TestFileType::EdgeCase)?;
        mb_minus_one_file.expected_matches = vec!["B".to_string(), "Z".to_string()];
        files.push(mb_minus_one_file);
        
        Ok(files)
    }
    
    /// Generate Unicode edge cases
    fn generate_unicode_edge_cases(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // BOM (Byte Order Mark) variations
        let bom_content = format!("{}\nContent after BOM\nSEARCH_TARGET_BOM\n", '\u{FEFF}');
        let mut bom_file = self.create_test_file("edge_unicode_bom.txt", &bom_content, TestFileType::EdgeCase)?;
        bom_file.expected_matches = vec![
            "Content after BOM".to_string(),
            "SEARCH_TARGET_BOM".to_string(),
        ];
        files.push(bom_file);
        
        // Zero-width characters
        let mut zero_width_content = String::new();
        zero_width_content.push_str("normal");
        zero_width_content.push('\u{200B}'); // Zero-width space
        zero_width_content.push_str("text");
        zero_width_content.push('\u{200C}'); // Zero-width non-joiner
        zero_width_content.push_str("SEARCH_TARGET");
        zero_width_content.push('\u{200D}'); // Zero-width joiner
        zero_width_content.push_str("more");
        zero_width_content.push('\u{FEFF}'); // Zero-width no-break space
        zero_width_content.push_str("text\n");
        
        let mut zero_width_file = self.create_test_file("edge_zero_width.txt", &zero_width_content, TestFileType::EdgeCase)?;
        zero_width_file.expected_matches = vec![
            "normal".to_string(),
            "text".to_string(),
            "SEARCH_TARGET".to_string(),
            "more".to_string(),
        ];
        files.push(zero_width_file);
        
        // Combining characters and normalization edge cases
        let combining_content = format!(
            "e\u{0301}motion\n{}SEARCH_TARGET_COMBINING\ncafe\u{0301}\n",
            "a\u{0300}bcde\u{0301}fgh\u{0302}ijk\u{0303}lmn\u{0304}opq\u{0305}rst\u{0306}uvw\u{0307}xyz\n"
        );
        let mut combining_file = self.create_test_file("edge_combining_chars.txt", &combining_content, TestFileType::EdgeCase)?;
        combining_file.expected_matches = vec![
            "emotion".to_string(), // Should match despite combining characters
            "SEARCH_TARGET_COMBINING".to_string(),
            "cafe".to_string(),
        ];
        files.push(combining_file);
        
        // Surrogate pairs and high Unicode code points
        let high_unicode_content = format!(
            "Regular text\n{} SEARCH_TARGET_EMOJI {}\n{}\nEnd text\n",
            "🔍🌟📝💻🚀", // Emojis using surrogate pairs
            "🎯🔥⭐",
            "𝕌𝕟𝕚𝕔𝕠𝕕𝕖 𝕚𝕤 𝕕𝕚𝔣𝔣𝕚𝔠𝕦𝕝𝕥" // Mathematical script characters
        );
        let mut high_unicode_file = self.create_test_file("edge_high_unicode.txt", &high_unicode_content, TestFileType::EdgeCase)?;
        high_unicode_file.expected_matches = vec![
            "Regular text".to_string(),
            "SEARCH_TARGET_EMOJI".to_string(),
            "End text".to_string(),
            "🔍".to_string(),
            "💻".to_string(),
        ];
        files.push(high_unicode_file);
        
        Ok(files)
    }
    
    /// Generate files that test regex engine edge cases
    fn generate_regex_edge_cases(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // Catastrophic backtracking patterns
        let backtracking_content = r#"
SEARCH_START
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaX
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
ccccccccccccccccccccccccccccccccccccccccccccccccccY
ddddddddddddddddddddddddddddddddddddddddddddddddddZ
BACKTRACKING_TARGET_FOUND
SEARCH_END
"#;
        let mut backtracking_file = self.create_test_file("edge_regex_backtracking.txt", backtracking_content, TestFileType::EdgeCase)?;
        backtracking_file.expected_matches = vec![
            "SEARCH_START".to_string(),
            "SEARCH_END".to_string(),
            "BACKTRACKING_TARGET_FOUND".to_string(),
            "a+X".to_string(), // Test greedy quantifiers
            "b+".to_string(),
            "c+Y".to_string(),
            "d+Z".to_string(),
        ];
        files.push(backtracking_file);
        
        // Nested quantifiers and complex patterns
        let complex_patterns_content = r#"
NESTED_START
((((((((((pattern))))))))))
[[[[[[[[[[[[[[[bracket_pattern]]]]]]]]]]]]]]]
{{{{{{{{{{brace_pattern}}}}}}}}}}
REGEX_STRESS_TEST: a*b*c*d*e*f*g*h*i*j*k*l*m*n*o*p*q*r*s*t*u*v*w*x*y*z*
ALTERNATION_TEST: (option1|option2|option3|option4|option5|option6|option7|option8|option9|option10)
NESTED_END
"#;
        let mut complex_file = self.create_test_file("edge_regex_complex.txt", complex_patterns_content, TestFileType::EdgeCase)?;
        complex_file.expected_matches = vec![
            "NESTED_START".to_string(),
            "NESTED_END".to_string(),
            "pattern".to_string(),
            "bracket_pattern".to_string(),
            "brace_pattern".to_string(),
            "REGEX_STRESS_TEST".to_string(),
            "ALTERNATION_TEST".to_string(),
        ];
        files.push(complex_file);
        
        // Special character combinations that might break parsers
        let special_chars_content = r#"
SPECIAL_CHARS_START
Line with \backslashes\ and /forward/slashes/
Line with "quotes" and 'apostrophes' and `backticks`
Line with (parentheses) and [brackets] and {braces}
Line with ^carets^ and $dollars$ and %percents%
Line with &ampersands& and |pipes| and ~tildes~
Line with *asterisks* and +plus+ and -minus- and =equals=
Line with ?questions? and !exclamations! and @at@ and #hash#
UNICODE_SPECIAL: ±²³¼½¾×÷‰‱°′″‴‼‽⁇⁈⁉⁏⁐⁑⁒⁓
CONTROL_CHARS: \t\r\n\v\f\a\b
SPECIAL_CHARS_END
"#;
        let mut special_chars_file = self.create_test_file("edge_special_chars.txt", special_chars_content, TestFileType::EdgeCase)?;
        special_chars_file.expected_matches = vec![
            "SPECIAL_CHARS_START".to_string(),
            "SPECIAL_CHARS_END".to_string(),
            "backslashes".to_string(),
            "quotes".to_string(),
            "parentheses".to_string(),
            "UNICODE_SPECIAL".to_string(),
            "CONTROL_CHARS".to_string(),
        ];
        files.push(special_chars_file);
        
        Ok(files)
    }
}
```

## Success Criteria
- Method generates 15+ edge case test files covering various boundary conditions
- Files test zero-length, single-character, and extremely large content
- Unicode edge cases including BOM, combining characters, and surrogate pairs
- Line ending variations (Unix, Windows, Mac) are properly handled
- Binary-like content with embedded text patterns
- Regex engine stress tests with complex patterns and quantifiers
- Boundary size files (1 byte, 4KB, 64KB, 1MB-1) are generated
- All files include appropriate expected_matches arrays

## Time Limit
10 minutes maximum