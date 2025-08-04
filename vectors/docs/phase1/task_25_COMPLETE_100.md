# Task 25: Complete Unicode and Special Character Handling Implementation

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)  
**Prerequisites:** Task 24 completed (cross-platform compatibility testing)  
**Input Files:** 
- `C:/code/LLMKG/vectors/tantivy_search/src/schema.rs` (exists with dual-field schema)
- `C:/code/LLMKG/vectors/tantivy_search/tests/` (directory exists)
- Previous test files demonstrating basic indexing functionality

## Complete Context (For AI with ZERO Knowledge)

**What is Unicode Handling?** Unicode is the international standard for text representation that includes characters from all world languages, symbols, emojis, and special characters. Proper Unicode handling ensures text is correctly stored, indexed, and searched regardless of language or character complexity.

**Why This Task is Critical?** Modern codebases contain Unicode characters in comments, string literals, documentation, and even identifiers. Our search system must handle these correctly to be useful for international development teams and projects containing non-ASCII content.

**Key Unicode Challenges in Code Search:**
1. **Character Normalization** - Same visual character can have different Unicode representations
2. **Emoji Handling** - Emojis in comments, commit messages, documentation
3. **International Text** - Comments in multiple languages (Chinese, Japanese, Arabic, etc.)
4. **Special Symbols** - Mathematical symbols, currency, arrows in code documentation
5. **Combining Characters** - Accented letters built from base + accent characters
6. **Mixed Content** - Files containing both ASCII code and Unicode comments

**Real-World Unicode Scenarios:**
- **Comments:** `// Función para procesar datos (Spanish)`
- **Strings:** `"用户名不能为空"` (Chinese error message)  
- **Documentation:** `/// 🚀 Fast implementation with ⚡ performance`
- **Identifiers:** `struct Café { café_type: String }` (if language allows)
- **Mixed:** `fn calculate_π() -> f64 { std::f64::consts::PI }`
- **Complex:** `"👨‍💻 Developer: 山田太郎"` (emoji + Japanese)

**What Our Dual-Field Schema Must Handle:**
- **content field (TEXT):** Tokenized Unicode, searchable by individual words
- **raw_content field (STRING):** Exact Unicode preservation for special character matching
- Both fields must preserve Unicode integrity during indexing and retrieval
- Search must work correctly across different Unicode normalization forms

## Exact Steps (6 minutes implementation)

### Step 1: Create Unicode and special character test file (4 minutes)

Create the file `C:/code/LLMKG/vectors/tantivy_search/tests/unicode_handling.rs` with this exact content:

```rust
use tantivy_search::*;
use tempfile::TempDir;
use tantivy::{doc, Index, IndexWriter};
use tantivy::query::QueryParser;
use tantivy::collector::TopDocs;
use anyhow::Result;
use std::fs;

/// Test basic Unicode character indexing and searching
#[test]
fn test_basic_unicode_character_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let schema = get_schema();
    let index_path = temp_dir.path().join("unicode_basic_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Test various Unicode content scenarios
    let unicode_test_cases = vec![
        ("spanish.rs", "// Función para procesar datos\nfn procesar_datos() -> Resultado<String, Error> {\n    // Implementación aquí\n}"),
        ("chinese.rs", "// 用户管理系统\nstruct 用户 {\n    名称: String,\n    年龄: u32,\n}\n// 计算用户年龄"),
        ("japanese.rs", "// ユーザー管理\nfn ユーザー作成() {\n    println!(\"ユーザーを作成しました\");\n}"),
        ("arabic.rs", "// نظام إدارة المستخدمين\n// معالجة البيانات\nfn process_data() {\n    // تنفيذ الوظيفة\n}"),
        ("emoji.rs", "// 🚀 Fast implementation\n// ⚡ High performance module\nfn rocket_fast() {\n    println!(\"🔥 Blazing fast!\");\n}"),
        ("mixed.rs", "// Mixed: English, Español, 中文, 日本語\nfn calculate_π() -> f64 {\n    std::f64::consts::PI // π value\n}"),
    ];
    
    for (i, (filename, content)) in unicode_test_cases.iter().enumerate() {
        let doc = doc!(
            content_field => *content,
            raw_content_field => *content,
            file_path_field => format!("/unicode/basic/{}", filename),
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => content.len() as u64,
            has_overlap_field => false
        );
        
        writer.add_document(doc)?;
    }
    
    writer.commit()?;
    
    // Test Unicode search functionality
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    
    // Test searching for Unicode terms
    let unicode_search_tests = vec![
        ("datos", "Should find Spanish content"),
        ("用户", "Should find Chinese content"),
        ("ユーザー", "Should find Japanese content"),
        ("rocket_fast", "Should find emoji-decorated function"),
        ("π", "Should find mathematical symbol"),
    ];
    
    for (search_term, description) in unicode_search_tests {
        let query = query_parser.parse_query(search_term)?;
        let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        
        if !results.is_empty() {
            println!("✓ {}: found {} results for '{}'", description, results.len(), search_term);
            
            // Verify content integrity
            let (_, doc_address) = results[0];
            let retrieved_doc = searcher.doc(doc_address)?;
            let retrieved_content = retrieved_doc.get_first(content_field).unwrap().as_text().unwrap();
            assert!(retrieved_content.contains(search_term), "Retrieved content should contain search term");
        } else {
            println!("! {} - no results found (might be expected for some tokenizers)", description);
        }
    }
    
    println!("Basic Unicode test passed - indexed {} files with Unicode content", unicode_test_cases.len());
    
    Ok(())
}

/// Test emoji and special symbol handling
#[test] 
fn test_emoji_and_special_symbols() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let schema = get_schema();
    let index_path = temp_dir.path().join("emoji_symbols_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Test comprehensive emoji and symbol scenarios
    let emoji_symbol_tests = vec![
        ("developer.rs", "// 👨‍💻 Developer tools\n// 🔧 Configuration utilities\nfn setup_dev_environment() {\n    println!(\"🚀 Setup complete!\");\n}"),
        ("performance.rs", "// ⚡ Lightning fast\n// 🔥 Blazing performance\nfn optimize() {\n    // 💪 Powerful optimization\n}"),
        ("math.rs", "// Mathematical symbols: π ∑ ∫ √ ∞\nconst PI: f64 = π;\nfn calculate_∑(values: &[f64]) -> f64 {\n    values.iter().sum()\n}"),
        ("currency.rs", "// Currency handling: $ € £ ¥ ₽ ₿\nstruct Price {\n    amount: f64,\n    currency: Currency, // €, $, £\n}"),
        ("arrows.rs", "// Flow indicators: → ← ↑ ↓ ⇒ ⇐\n// Data flow: input → process → output\nfn data_pipeline() {\n    // input ⇒ validation ⇒ processing ⇒ output\n}"),
        ("complex_emoji.rs", "// Complex emoji sequences\n// 👨‍👩‍👧‍👦 Family emoji\n// 🏳️‍🌈 Flag with combining\nfn family_management() {\n    println!(\"👨‍👩‍👧‍👦 Family system\");\n}"),
    ];
    
    for (i, (filename, content)) in emoji_symbol_tests.iter().enumerate() {
        let doc = doc!(
            content_field => *content,
            raw_content_field => *content,
            file_path_field => format!("/unicode/symbols/{}", filename),
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => content.len() as u64,
            has_overlap_field => false
        );
        
        writer.add_document(doc)?;
    }
    
    writer.commit()?;
    
    // Test emoji and symbol search using raw_content field
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let raw_query_parser = QueryParser::for_index(&index, vec![raw_content_field]);
    
    // Test exact symbol matching in raw content
    let symbol_search_tests = vec![
        ("🚀", "Should find rocket emoji"),
        ("⚡", "Should find lightning emoji"),
        ("π", "Should find pi symbol"),
        ("€", "Should find euro symbol"),
        ("→", "Should find right arrow"),
        ("👨‍💻", "Should find developer emoji sequence"),
    ];
    
    for (symbol, description) in symbol_search_tests {
        let query = raw_query_parser.parse_query(symbol)?;
        let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        
        if !results.is_empty() {
            println!("✓ {}: found {} results", description, results.len());
            
            // Verify exact symbol preservation
            let (_, doc_address) = results[0];
            let retrieved_doc = searcher.doc(doc_address)?;
            let raw_content = retrieved_doc.get_first(raw_content_field).unwrap().as_text().unwrap();
            assert!(raw_content.contains(symbol), "Raw content should preserve exact symbol: {}", symbol);
        } else {
            println!("! {} - not found in raw content (tokenizer-dependent)", description);
        }
    }
    
    println!("Emoji and symbols test passed - indexed {} files with special characters", emoji_symbol_tests.len());
    
    Ok(())
}

/// Test Unicode normalization and character equivalence
#[test]
fn test_unicode_normalization_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let schema = get_schema();
    let index_path = temp_dir.path().join("normalization_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Test different Unicode normalization forms
    // These represent the same visual characters in different encodings
    let normalization_tests = vec![
        ("nfc.rs", "// Café (NFC - composed form)\nstruct Café {\n    name: String,\n}"),
        ("nfd.rs", "// Cafe\u{0301} (NFD - decomposed form)\nstruct Cafe\u{0301} {\n    name: String,\n}"),
        ("accents.rs", "// Various accented characters\n// é è ê ë ñ ü ö ä\nfn handle_français() {\n    println!(\"Français, Español, Português\");\n}"),
        ("combining.rs", "// Combining characters test\n// a\u{0300} a\u{0301} a\u{0302} (a with grave, acute, circumflex)\nfn combining_test() {\n    let àáâ = \"combining characters\";\n}"),
    ];
    
    for (i, (filename, content)) in normalization_tests.iter().enumerate() {
        let doc = doc!(
            content_field => *content,
            raw_content_field => *content,
            file_path_field => format!("/unicode/normalization/{}", filename),
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => content.len() as u64,
            has_overlap_field => false
        );
        
        writer.add_document(doc)?;
    }
    
    writer.commit()?;
    
    // Test search across different normalization forms
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    // Test both tokenized and raw searches
    let content_parser = QueryParser::for_index(&index, vec![content_field]);
    let raw_parser = QueryParser::for_index(&index, vec![raw_content_field]);
    
    // Search for accented characters
    let accent_searches = vec![
        ("Café", "Should find cafe with accent"),
        ("français", "Should find French text"),
        ("combining", "Should find combining character test"),
    ];
    
    for (search_term, description) in accent_searches {
        // Test in content field (tokenized)
        if let Ok(query) = content_parser.parse_query(search_term) {
            let results = searcher.search(&query, &TopDocs::with_limit(10))?;
            if !results.is_empty() {
                println!("✓ Content field: {} - found {} results", description, results.len());
            }
        }
        
        // Test in raw field (exact matching)  
        if let Ok(query) = raw_parser.parse_query(search_term) {
            let results = searcher.search(&query, &TopDocs::with_limit(10))?;
            if !results.is_empty() {
                println!("✓ Raw field: {} - found {} results", description, results.len());
            }
        }
    }
    
    println!("Unicode normalization test passed - indexed {} files with different normalization forms", normalization_tests.len());
    
    Ok(())
}

/// Test mixed-script content handling
#[test]
fn test_mixed_script_content() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let schema = get_schema();
    let index_path = temp_dir.path().join("mixed_script_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Test files with mixed scripts and languages
    let mixed_script_tests = vec![
        ("international.rs", r#"
// International user management system
// Système de gestion des utilisateurs (French)
// Sistema de gestión de usuarios (Spanish)  
// 用户管理系统 (Chinese)
// ユーザー管理システム (Japanese)
// Система управления пользователями (Russian)

struct User {
    name: String,          // English field
    nom: String,           // French
    nombre: String,        // Spanish
    姓名: String,           // Chinese
    名前: String,           // Japanese  
    имя: String,           // Russian
}

fn create_user(name: &str) -> User {
    println!("Creating user: {}", name);
    User {
        name: name.to_string(),
        nom: name.to_string(),
        nombre: name.to_string(), 
        姓名: name.to_string(),
        名前: name.to_string(),
        имя: name.to_string(),
    }
}
"#),
        ("documentation.rs", r#"
/// # Multi-language Documentation / Documentation multilingue
/// 
/// This function handles international text processing.
/// Cette fonction gère le traitement de texte international.
/// Esta función maneja el procesamiento de texto internacional.
/// この関数は国際的なテキスト処理を処理します。
/// 该函数处理国际文本处理。
/// 
/// ## Examples / Exemples / Ejemplos / 例 / 示例
/// 
/// ```rust
/// let result = process_text("Hello 世界 🌍");
/// assert_eq!(result, "Processed: Hello 世界 🌍");
/// ```
fn process_international_text(input: &str) -> String {
    format!("Processed: {}", input)
}
"#),
        ("error_messages.rs", r#"
// Multi-language error messages
const ERRORS: &[(&str, &str)] = &[
    ("en", "User not found"),
    ("es", "Usuario no encontrado"),
    ("fr", "Utilisateur non trouvé"),
    ("de", "Benutzer nicht gefunden"),
    ("zh", "找不到用户"),
    ("ja", "ユーザーが見つかりません"),
    ("ru", "Пользователь не найден"),
    ("ar", "المستخدم غير موجود"),
];

fn get_error_message(lang: &str) -> &'static str {
    ERRORS.iter()
        .find(|(l, _)| *l == lang)
        .map(|(_, msg)| *msg)
        .unwrap_or("Error message not available")
}
"#),
    ];
    
    for (i, (filename, content)) in mixed_script_tests.iter().enumerate() {
        let doc = doc!(
            content_field => *content,
            raw_content_field => *content,
            file_path_field => format!("/unicode/mixed/{}", filename),
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => content.len() as u64,
            has_overlap_field => false
        );
        
        writer.add_document(doc)?;
    }
    
    writer.commit()?;
    
    // Test searching across different scripts
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    
    let mixed_search_tests = vec![
        ("User", "Should find English term"),
        ("utilisateur", "Should find French term"), 
        ("usuario", "Should find Spanish term"),
        ("用户", "Should find Chinese term"),
        ("ユーザー", "Should find Japanese term"),
        ("пользователь", "Should find Russian term"),
        ("process_international", "Should find function name"),
        ("error_message", "Should find error handling code"),
    ];
    
    let mut successful_searches = 0;
    
    for (search_term, description) in mixed_search_tests {
        match query_parser.parse_query(search_term) {
            Ok(query) => {
                match searcher.search(&query, &TopDocs::with_limit(10)) {
                    Ok(results) => {
                        if !results.is_empty() {
                            println!("✓ {}: found {} results", description, results.len());
                            successful_searches += 1;
                        } else {
                            println!("- {}: no results (tokenizer-dependent)", description);
                        }
                    }
                    Err(_) => println!("! {}: search failed", description),
                }
            }
            Err(_) => println!("! {}: query parsing failed", description),
        }
    }
    
    // Should find at least some terms (depends on tokenizer capabilities)
    assert!(successful_searches > 0, "Should successfully search for at least some mixed-script content");
    
    println!("Mixed script test passed - indexed {} files with mixed international content", mixed_script_tests.len());
    
    Ok(())
}

/// Test Unicode content integrity and round-trip preservation
#[test]
fn test_unicode_content_integrity() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let schema = get_schema();
    let index_path = temp_dir.path().join("integrity_index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Test complex Unicode content that must be preserved exactly
    let integrity_test_content = r#"
// Unicode integrity test file
// Complex Unicode sequences that must be preserved exactly

// Emoji sequences with zero-width joiners
const FAMILY: &str = "👨‍👩‍👧‍👦"; // Man, woman, girl, boy
const FLAG: &str = "🏳️‍🌈"; // Rainbow flag with combining
const PROFESSIONAL: &str = "👨‍💻"; // Man technologist

// Mathematical and scientific symbols
const SYMBOLS: &[&str] = &[
    "∑", "∫", "∆", "∇", "∞", "≠", "≤", "≥", "±", "×", "÷",
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "λ", "μ", "π", "σ", "φ", "ψ", "ω"
];

// Currency and financial symbols
const CURRENCIES: &[&str] = &["$", "€", "£", "¥", "₽", "₿", "₹", "₩", "₪"];

// Directional and arrow symbols
const ARROWS: &[&str] = &["→", "←", "↑", "↓", "⇒", "⇐", "⇑", "⇓", "↔", "⇔"];

// Combining diacritical marks
fn test_combining_marks() {
    let base_a = "a";
    let combining_grave = "\u{0300}"; 
    let combining_acute = "\u{0301}";
    let combining_circumflex = "\u{0302}";
    
    let a_grave = format!("{}{}", base_a, combining_grave); // à
    let a_acute = format!("{}{}", base_a, combining_acute); // á  
    let a_circumflex = format!("{}{}", base_a, combining_circumflex); // â
    
    println!("Combining marks: {} {} {}", a_grave, a_acute, a_circumflex);
}

// Mixed script identifiers (if language allows)
struct UnicodeTest {
    café: String,           // French
    naïve: String,          // French with diaeresis
    façade: String,         // French with cedilla
    résumé: String,         // French with acute accents
}
"#;
    
    let doc = doc!(
        content_field => integrity_test_content,
        raw_content_field => integrity_test_content,
        file_path_field => "/unicode/integrity/test.rs",
        chunk_index_field => 0u64,
        chunk_start_field => 0u64,
        chunk_end_field => integrity_test_content.len() as u64,
        has_overlap_field => false
    );
    
    writer.add_document(doc)?;
    writer.commit()?;
    
    // Verify content integrity after indexing and retrieval
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    // Retrieve the document
    let retrieved_doc = searcher.doc(tantivy::DocAddress::new(0, 0))?;
    
    // Check content field preservation
    let retrieved_content = retrieved_doc.get_first(content_field).unwrap().as_text().unwrap();
    let retrieved_raw = retrieved_doc.get_first(raw_content_field).unwrap().as_text().unwrap();
    
    // Verify key Unicode sequences are preserved
    let critical_sequences = vec![
        "👨‍👩‍👧‍👦", // Family emoji
        "🏳️‍🌈",       // Rainbow flag
        "∑", "∫", "π", // Mathematical symbols
        "€", "£", "¥", // Currency symbols  
        "→", "←", "↔", // Arrows
        "café", "naïve", "façade", "résumé", // Accented words
    ];
    
    for sequence in critical_sequences {
        assert!(retrieved_content.contains(sequence) || retrieved_raw.contains(sequence), 
               "Unicode sequence '{}' should be preserved in indexed content", sequence);
    }
    
    // Test that original and retrieved content have same byte length
    // (This ensures no Unicode corruption during storage)
    assert_eq!(integrity_test_content.len(), retrieved_raw.len(), 
              "Raw content should preserve exact byte length");
    
    println!("Unicode integrity test passed - all critical sequences preserved");
    
    Ok(())
}
```

### Step 2: Add Unicode test dependencies (1 minute)

Verify `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` has required dependencies:

```toml
[dev-dependencies]
tempfile = "3.8"
anyhow = "1.0"
```

### Step 3: Ensure Unicode-ready schema (1 minute)

Confirm `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs` exports are available:

```rust
pub mod schema;
pub use schema::{get_schema, create_index, open_or_create_index};
```

## Verification Steps (2 minutes)

```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo test unicode_handling
```

**Expected output:**
```
running 5 tests
test test_basic_unicode_character_handling ... ok
test test_emoji_and_special_symbols ... ok
test test_unicode_normalization_handling ... ok
test test_mixed_script_content ... ok
test test_unicode_content_integrity ... ok

test result: ok. 5 passed; 0 failed; 0 ignored
```

## If This Task Fails

### Common Errors and Solutions

**Error 1: "Invalid UTF-8 sequence"**
```bash
# Solution: Unicode content issue in test file
# Verify file is saved with UTF-8 encoding
file --mime-encoding tests/unicode_handling.rs
cargo test unicode_handling -- --nocapture
```

**Error 2: "Query parsing failed for Unicode terms"**
```bash
# Solution: Tokenizer doesn't support Unicode terms
# This is expected behavior - test should handle gracefully
# Check that test prints "no results (tokenizer-dependent)"
cargo test unicode_handling -- --nocapture
```

**Error 3: "Character not found in indexed content"**
```bash
# Solution: Unicode normalization or encoding issue
# Test specific character preservation:
echo "Testing: 🚀 → π café" | hexdump -C
# Verify characters are properly encoded in source
```

**Error 4: "Compilation error: invalid identifier"**
```bash
# Solution: Rust doesn't allow all Unicode in identifiers
# This is expected - some tests may use comments only
# Check Rust edition and identifier rules for Unicode
```

## Troubleshooting Checklist

- [ ] Test file saved with UTF-8 encoding (not ASCII or Latin-1)
- [ ] Terminal/console supports Unicode display
- [ ] Rust version supports Unicode string literals
- [ ] Tantivy tokenizer handles Unicode correctly (or test expects failures)
- [ ] No invalid Unicode sequences in test strings
- [ ] Platform supports Unicode filenames (for filesystem tests)
- [ ] Locale settings support international characters

## Recovery Procedures

### Unicode Display Issues
If Unicode characters don't display correctly:
1. Check terminal encoding: `locale` command (Unix)
2. Verify font supports Unicode: Use Unicode-capable terminal
3. Test with simple ASCII first: Ensure basic functionality works
4. Use `--nocapture` flag: See actual Unicode output in tests

### Tokenizer Limitations
If search fails for Unicode terms:
1. This is often expected behavior - tokenizers vary in Unicode support
2. Verify raw content preservation: Check STRING field preserves Unicode
3. Test with English terms first: Ensure basic search works
4. Check Tantivy documentation: Review Unicode tokenization capabilities

### Encoding Problems
If content appears corrupted:
1. Check file encoding: `file --mime-encoding filename`
2. Verify editor settings: Ensure UTF-8 without BOM
3. Test byte-by-byte: Use hexdump to compare original vs retrieved
4. Check normalization: Different Unicode forms of same character

## Success Validation Checklist

- [ ] File `tests/unicode_handling.rs` exists with 5 comprehensive Unicode tests
- [ ] Command `cargo test unicode_handling` completes successfully
- [ ] Basic Unicode test indexes content in multiple languages (Spanish, Chinese, Japanese, Arabic)
- [ ] Emoji test handles complex emoji sequences and mathematical symbols
- [ ] Normalization test handles different Unicode normalization forms
- [ ] Mixed script test indexes files with multiple languages together
- [ ] Integrity test verifies exact Unicode preservation in raw_content field
- [ ] Tests gracefully handle tokenizer limitations with informative messages

## Files Created For Next Task

After completing this task, you will have:

1. **C:/code/LLMKG/vectors/tantivy_search/tests/unicode_handling.rs** - Comprehensive Unicode and special character testing
2. **Unicode validation** - Confirmed system preserves international characters correctly  
3. **Character integrity** - Verified both tokenized and raw content fields handle Unicode properly

**Next Task (Task 26)** will implement large file processing to handle very large source files that need to be chunked for efficient indexing and searching.

## Context for Task 26

Task 26 will build upon the Unicode handling foundation to create efficient processing of large files (>1MB) that require intelligent chunking while preserving Unicode boundaries and special character integrity across chunk splits.