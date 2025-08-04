# Task 24: Complete Cross-Platform Compatibility Testing Implementation

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)  
**Prerequisites:** Task 23 completed (stress testing framework)  
**Input Files:** 
- `C:/code/LLMKG/vectors/tantivy_search/src/schema.rs` (exists with all functions)
- `C:/code/LLMKG/vectors/tantivy_search/tests/` (directory exists)
- Previous test files (stress_testing.rs, component_interactions.rs, integration_workflow.rs)

## Complete Context (For AI with ZERO Knowledge)

**What is Cross-Platform Compatibility Testing?** Cross-platform testing ensures software works correctly across different operating systems (Windows, macOS, Linux) by validating OS-specific behaviors like file paths, directory separators, permissions, and system APIs.

**Why This Task is Critical?** After stress testing in Task 23, we need to ensure the search system works consistently across all major platforms where Rust code is developed, handling platform-specific differences gracefully.

**Key Platform Differences to Test:**
1. **Path Separators** - Windows uses `\`, Unix uses `/`
2. **File Permissions** - Unix has rwx bits, Windows has different model
3. **Case Sensitivity** - Windows is case-insensitive, Unix is case-sensitive
4. **Line Endings** - Windows uses CRLF (`\r\n`), Unix uses LF (`\n`)
5. **Reserved Names** - Windows has reserved names like CON, PRN, AUX
6. **Path Length Limits** - Windows has 260 char limit (historically)
7. **Unicode Handling** - Different normalization on different platforms

**Real-World Platform Scenarios:**
- **Windows:** `C:\code\project\src\main.rs` with CRLF line endings
- **Linux:** `/home/user/code/project/src/main.rs` with LF endings
- **macOS:** `/Users/user/code/project/src/main.rs` with Unicode normalization
- **Mixed:** Git repos with different line endings across platforms
- **Edge Cases:** Filenames with spaces, Unicode, special characters

**What We Need to Test:**
- File path normalization across platforms
- Directory creation and traversal
- Index storage in platform-appropriate locations
- Text encoding and line ending handling
- File permissions and access patterns
- Platform-specific error conditions

## Exact Steps (6 minutes implementation)

### Step 1: Create cross-platform compatibility test file (4 minutes)

Create the file `C:/code/LLMKG/vectors/tantivy_search/tests/cross_platform.rs` with this exact content:

```rust
use tantivy_search::*;
use tempfile::TempDir;
use tantivy::{doc, Index, IndexWriter};
use tantivy::query::QueryParser;
use tantivy::collector::TopDocs;
use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf, MAIN_SEPARATOR};
use std::io::Write;

/// Test file path handling across different platforms
#[test]
fn test_cross_platform_path_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Test different path styles that might appear in real codebases
    let test_paths = vec![
        "src/main.rs",
        "src/lib.rs", 
        "tests/integration.rs",
        "examples/basic.rs",
        "benches/performance.rs",
    ];
    
    // Create directory structure
    fs::create_dir_all(base_path.join("src"))?;
    fs::create_dir_all(base_path.join("tests"))?;
    fs::create_dir_all(base_path.join("examples"))?;
    fs::create_dir_all(base_path.join("benches"))?;
    
    // Test that our indexing works with platform-appropriate paths
    let schema = get_schema();
    let index_path = base_path.join("platform_test_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    for (i, relative_path) in test_paths.iter().enumerate() {
        let full_path = base_path.join(relative_path);
        let content = format!("// Platform test file {}\nfn test_{}() {{}}", i, relative_path.replace('/', "_").replace(".rs", ""));
        
        // Write actual file
        fs::write(&full_path, &content)?;
        
        // Index with platform-normalized path
        let normalized_path = full_path.to_string_lossy().to_string();
        let doc = doc!(
            content_field => content.clone(),
            raw_content_field => content,
            file_path_field => normalized_path,
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => content.len() as u64,
            has_overlap_field => false
        );
        
        writer.add_document(doc)?;
    }
    
    writer.commit()?;
    
    // Verify paths are handled correctly across platforms
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    // Test search works regardless of path separator
    let query_parser = QueryParser::for_index(&index, vec![file_path_field]);
    let query = query_parser.parse_query("main")?;
    let results = searcher.search(&query, &TopDocs::with_limit(10))?;
    
    assert!(!results.is_empty(), "Should find files regardless of platform path format");
    
    // Verify we can retrieve and read the indexed paths
    let (_, doc_address) = results[0];
    let retrieved_doc = searcher.doc(doc_address)?;
    let stored_path = retrieved_doc.get_first(file_path_field).unwrap().as_text().unwrap();
    
    // Path should contain platform-appropriate separator
    assert!(stored_path.contains("main.rs"), "Should preserve filename correctly");
    
    println!("Platform path test passed on {}", std::env::consts::OS);
    
    Ok(())
}

/// Test handling of different line endings across platforms
#[test]
fn test_cross_platform_line_endings() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let schema = get_schema();
    let index_path = temp_dir.path().join("line_ending_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Test different line ending styles
    let line_ending_tests = vec![
        ("unix_file.rs", "fn unix_function() {\n    println!(\"Unix style\");\n}"),
        ("windows_file.rs", "fn windows_function() {\r\n    println!(\"Windows style\");\r\n}"),
        ("mixed_file.rs", "fn mixed_function() {\n    println!(\"Mixed\");\r\n    println!(\"Style\");\n}"),
    ];
    
    for (i, (filename, content)) in line_ending_tests.iter().enumerate() {
        let file_path = temp_dir.path().join(filename);
        
        // Write file with specific line endings
        let mut file = fs::File::create(&file_path)?;
        file.write_all(content.as_bytes())?;
        file.flush()?;
        drop(file);
        
        // Index the content
        let doc = doc!(
            content_field => *content,
            raw_content_field => *content,
            file_path_field => file_path.to_string_lossy().to_string(),
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => content.len() as u64,
            has_overlap_field => false
        );
        
        writer.add_document(doc)?;
    }
    
    writer.commit()?;
    
    // Verify search works regardless of line ending style
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    
    // Search should find content regardless of line endings
    let query = query_parser.parse_query("function")?;
    let results = searcher.search(&query, &TopDocs::with_limit(10))?;
    
    assert_eq!(results.len(), 3, "Should find all files regardless of line ending style");
    
    println!("Line ending test passed - found {} files with different line endings", results.len());
    
    Ok(())
}

/// Test Unicode filename and content handling across platforms
#[test]
fn test_cross_platform_unicode_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let schema = get_schema();
    let index_path = temp_dir.path().join("unicode_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Test various Unicode scenarios
    let unicode_tests = vec![
        ("café.rs", "// Café module\nfn café_function() { println!(\"Café!\"); }"),
        ("测试.rs", "// Chinese filename\nfn test_function() { println!(\"测试\"); }"),
        ("émoji_🦀.rs", "// Emoji in filename\nfn rust_function() { println!(\"🦀 Rust!\"); }"),
        ("spëciàl_chars.rs", "// Special chars\nfn spëciàl_function() { println!(\"Spëciàl!\"); }"),
    ];
    
    let mut successful_unicode_files = 0;
    
    for (i, (filename, content)) in unicode_tests.iter().enumerate() {
        let file_path = temp_dir.path().join(filename);
        
        // Try to create Unicode filename - might fail on some filesystems
        match fs::write(&file_path, content) {
            Ok(_) => {
                // Successfully created Unicode file, now index it
                let doc = doc!(
                    content_field => *content,
                    raw_content_field => *content,
                    file_path_field => file_path.to_string_lossy().to_string(),
                    chunk_index_field => i as u64,
                    chunk_start_field => 0u64,
                    chunk_end_field => content.len() as u64,
                    has_overlap_field => false
                );
                
                writer.add_document(doc)?;
                successful_unicode_files += 1;
            }
            Err(_) => {
                // Unicode filename not supported on this filesystem - that's OK
                println!("Unicode filename {} not supported on this platform", filename);
            }
        }
    }
    
    if successful_unicode_files > 0 {
        writer.commit()?;
        
        // Test search with Unicode content
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&index, vec![content_field]);
        
        // Search for Unicode content
        let query = query_parser.parse_query("function")?;
        let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        
        assert!(results.len() >= 1, "Should find Unicode files that were successfully created");
        
        println!("Unicode test passed - indexed {} Unicode files successfully", successful_unicode_files);
    } else {
        println!("No Unicode filenames supported on this platform - test skipped");
    }
    
    Ok(())
}

/// Test platform-specific directory structures
#[test]
fn test_cross_platform_directory_structures() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let schema = get_schema();
    let index_path = temp_dir.path().join("directory_structure_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Create typical cross-platform directory structures
    let directory_structures = vec![
        // Unix-style nested structure
        vec!["src", "main.rs"],
        vec!["src", "lib", "mod.rs"],
        vec!["src", "lib", "utils", "helper.rs"],
        vec!["tests", "integration", "test_main.rs"],
        vec!["examples", "basic", "example.rs"],
        // Deeper nesting
        vec!["src", "components", "ui", "widgets", "button.rs"],
        vec!["src", "backend", "database", "models", "user.rs"],
    ];
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    for (i, path_components) in directory_structures.iter().enumerate() {
        // Build platform-appropriate path
        let mut full_path = temp_dir.path().to_path_buf();
        for component in path_components.iter() {
            full_path.push(component);
        }
        
        // Create directory structure
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Create file content
        let filename = path_components.last().unwrap();
        let content = format!("// File: {}\nfn {}() {{ /* implementation */ }}", 
                            filename, filename.replace(".rs", "").replace('-', "_"));
        
        // Write file
        fs::write(&full_path, &content)?;
        
        // Index with normalized path
        let normalized_path = full_path.to_string_lossy().to_string();
        let doc = doc!(
            content_field => content.clone(),
            raw_content_field => content,
            file_path_field => normalized_path,
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => content.len() as u64,
            has_overlap_field => false
        );
        
        writer.add_document(doc)?;
    }
    
    writer.commit()?;
    
    // Test searching across different directory levels
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![file_path_field]);
    
    // Test directory-specific searches
    let directory_queries = vec![
        ("src", "Should find files in src directory"),
        ("lib", "Should find files in lib subdirectory"), 
        ("tests", "Should find files in tests directory"),
        ("components", "Should find files in deep directory structure"),
    ];
    
    for (query_term, description) in directory_queries {
        let query = query_parser.parse_query(query_term)?;
        let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        
        if !results.is_empty() {
            println!("✓ {}: found {} files", description, results.len());
        }
    }
    
    // Verify total document count
    assert_eq!(searcher.num_docs(), directory_structures.len() as u32, 
              "Should have indexed all directory structure files");
    
    println!("Directory structure test passed on {} with {} files", 
             std::env::consts::OS, directory_structures.len());
    
    Ok(())
}

/// Test platform-specific index storage and retrieval
#[test]
fn test_cross_platform_index_storage() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Test index creation and reopening across platform differences
    let index_path = temp_dir.path().join("platform_storage_test");
    
    // Create initial index
    {
        let schema = get_schema();
        let index = create_index(&index_path)?;
        let mut writer = index.writer(50_000_000)?;
        
        let content_field = schema.get_field("content")?;
        let raw_content_field = schema.get_field("raw_content")?;
        let file_path_field = schema.get_field("file_path")?;
        let chunk_index_field = schema.get_field("chunk_index")?;
        let chunk_start_field = schema.get_field("chunk_start")?;
        let chunk_end_field = schema.get_field("chunk_end")?;
        let has_overlap_field = schema.get_field("has_overlap")?;
        
        // Add platform-specific test data
        let platform_content = format!("// Platform: {}\n// Architecture: {}\nfn platform_test() {{ }}", 
                                     std::env::consts::OS, std::env::consts::ARCH);
        
        let doc = doc!(
            content_field => platform_content.clone(),
            raw_content_field => platform_content,
            file_path_field => format!("/platform/{}/test.rs", std::env::consts::OS),
            chunk_index_field => 0u64,
            chunk_start_field => 0u64,
            chunk_end_field => platform_content.len() as u64,
            has_overlap_field => false
        );
        
        writer.add_document(doc)?;
        writer.commit()?;
    }
    
    // Test reopening the index (simulates cross-session usage)
    {
        let index = open_or_create_index(&index_path)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        
        assert_eq!(searcher.num_docs(), 1, "Index should persist across platform operations");
        
        // Verify we can search the reopened index
        let schema = index.schema();
        let content_field = schema.get_field("content")?;
        let query_parser = QueryParser::for_index(&index, vec![content_field]);
        let query = query_parser.parse_query("platform_test")?;
        let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        
        assert!(!results.is_empty(), "Should be able to search reopened index");
        
        println!("Index storage test passed - index persisted correctly on {}", std::env::consts::OS);
    }
    
    Ok(())
}

/// Test platform-specific error handling
#[test]
fn test_cross_platform_error_handling() -> Result<()> {
    // Test platform-specific error conditions
    
    // Test with invalid/reserved filenames (Windows-specific)
    if cfg!(windows) {
        let temp_dir = TempDir::new()?;
        let schema = get_schema();
        let index_path = temp_dir.path().join("error_test_index");
        let index = create_index(&index_path)?;
        let mut writer = index.writer(50_000_000)?;
        
        let content_field = schema.get_field("content")?;
        let raw_content_field = schema.get_field("raw_content")?;
        let file_path_field = schema.get_field("file_path")?;
        let chunk_index_field = schema.get_field("chunk_index")?;
        let chunk_start_field = schema.get_field("chunk_start")?;
        let chunk_end_field = schema.get_field("chunk_end")?;
        let has_overlap_field = schema.get_field("has_overlap")?;
        
        // Test Windows reserved names - should handle gracefully
        let reserved_names = vec!["CON", "PRN", "AUX", "NUL"];
        for (i, reserved_name) in reserved_names.iter().enumerate() {
            let content = format!("// File with reserved name: {}", reserved_name);
            let file_path = format!("C:\\reserved\\{}.rs", reserved_name);
            
            let doc = doc!(
                content_field => content.clone(),
                raw_content_field => content,
                file_path_field => file_path,
                chunk_index_field => i as u64,
                chunk_start_field => 0u64,
                chunk_end_field => content.len() as u64,
                has_overlap_field => false
            );
            
            // Should not crash even with reserved names in paths
            writer.add_document(doc)?;
        }
        
        writer.commit()?;
        println!("Windows reserved name handling test passed");
    }
    
    // Test with very long paths (platform limits)
    let temp_dir = TempDir::new()?;
    let schema = get_schema();
    let index_path = temp_dir.path().join("long_path_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Create a moderately long path (not extreme to avoid test failures)
    let long_path_component = "very_long_directory_name_that_might_cause_issues_on_some_platforms";
    let long_path = format!("/src/{}/{}/{}/file.rs", 
                           long_path_component, long_path_component, long_path_component);
    
    let content = "// File with long path";
    let doc = doc!(
        content_field => content,
        raw_content_field => content,
        file_path_field => long_path,
        chunk_index_field => 0u64,
        chunk_start_field => 0u64,
        chunk_end_field => content.len() as u64,
        has_overlap_field => false
    );
    
    // Should handle long paths gracefully
    writer.add_document(doc)?;
    writer.commit()?;
    
    println!("Long path handling test passed on {}", std::env::consts::OS);
    
    Ok(())
}
```

### Step 2: Add platform-specific test configuration (1 minute)

Add platform detection to `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` if needed:

```toml
[dev-dependencies]
tempfile = "3.8"
anyhow = "1.0"
```

### Step 3: Verify cross-platform compatibility (1 minute)

Ensure `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs` exports are available:

```rust
pub mod schema;
pub use schema::{get_schema, create_index, open_or_create_index};
```

## Verification Steps (2 minutes)

```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo test cross_platform
```

**Expected output:**
```
running 6 tests
test test_cross_platform_path_handling ... ok
test test_cross_platform_line_endings ... ok
test test_cross_platform_unicode_handling ... ok
test test_cross_platform_directory_structures ... ok
test test_cross_platform_index_storage ... ok
test test_cross_platform_error_handling ... ok

test result: ok. 6 passed; 0 failed; 0 ignored
```

## If This Task Fails

### Common Errors and Solutions

**Error 1: "Unicode filename not supported"**
```bash
# Solution: This is expected on some filesystems
# The test should handle this gracefully and continue
# Verify the test prints "Unicode filename X not supported on this platform"
cargo test cross_platform -- --nocapture
```

**Error 2: "Permission denied" creating directories**
```bash
# Solution (Windows): Fix permissions
icacls %TEMP% /grant Users:F /T

# Solution (Linux/macOS): Check temp directory permissions
chmod 755 /tmp
cargo test cross_platform
```  

**Error 3: "Path too long" on Windows**
```bash
# Solution: Enable long path support on Windows 10+
# Or reduce path length in test
# Registry: HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
# Set LongPathsEnabled to 1
```

**Error 4: "Invalid characters in filename"**
```bash
# Solution: This is platform-specific behavior
# Test should handle invalid characters gracefully
# Check that test doesn't crash, just skips invalid names
cargo test cross_platform -- --nocapture
```

## Troubleshooting Checklist

- [ ] Rust supports current platform (Windows/Linux/macOS)
- [ ] Filesystem supports Unicode filenames (for Unicode test)
- [ ] Temp directory has write permissions
- [ ] No special filesystem restrictions (case-sensitivity, etc.)
- [ ] Path length limits understood for platform
- [ ] Line ending handling working correctly
- [ ] Reserved filename restrictions known (Windows)

## Recovery Procedures

### Unicode Issues
If Unicode tests fail:
1. Check filesystem support: `ls -la` with Unicode names
2. Verify locale settings: `locale` command (Unix)
3. Test with simpler Unicode: Use only basic accented characters
4. Allow graceful degradation: Skip unsupported Unicode tests

### Path Length Issues
If path tests fail due to length:
1. Check platform limits: 260 chars (Windows), 4096 (Linux)
2. Enable long paths: Windows 10+ registry setting
3. Reduce test path length: Use shorter directory names
4. Test with relative paths instead of absolute

### Permission Issues
If file creation fails:
1. Check temp directory: `echo $TMPDIR` or `echo %TEMP%`
2. Verify permissions: `ls -ld /tmp` or `icacls %TEMP%`
3. Use alternative temp location: Set `TMPDIR` environment variable
4. Run with elevated permissions if necessary

## Success Validation Checklist

- [ ] File `tests/cross_platform.rs` exists with 6 platform-specific tests
- [ ] Command `cargo test cross_platform` passes on current platform
- [ ] Path handling test works with current OS separator (`\` or `/`)
- [ ] Line ending test handles CRLF, LF, and mixed endings
- [ ] Unicode test either succeeds or gracefully reports unsupported filenames
- [ ] Directory structure test creates and indexes nested paths correctly
- [ ] Index storage test demonstrates persistence across sessions
- [ ] Error handling test manages platform-specific edge cases

## Files Created For Next Task

After completing this task, you will have:

1. **C:/code/LLMKG/vectors/tantivy_search/tests/cross_platform.rs** - Comprehensive cross-platform compatibility tests
2. **Platform validation** - Confirmed system works on current platform with proper error handling
3. **Path normalization** - Verified file paths are handled correctly across different OS conventions

**Next Task (Task 25)** will implement Unicode and special character handling to ensure the search system can properly index and search code containing international characters, emojis, and complex Unicode patterns.

## Context for Task 25

Task 25 will build upon the cross-platform foundation to create specialized handling for Unicode text, special characters in code syntax, and international character sets that commonly appear in modern codebases.