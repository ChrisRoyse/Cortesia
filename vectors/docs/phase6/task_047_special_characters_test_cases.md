# Task 047: Special Characters Test Cases

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates special characters test cases that validates the system correctly handles all special characters used in programming languages and markup.

## Project Structure
tests/
  special_characters_test_cases.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive test cases for special characters commonly found in code files including brackets, operators, quotes, escapes, and Unicode characters.

## Requirements
1. Create comprehensive integration test
2. Test all special character categories
3. Validate search accuracy with special characters
4. Handle edge cases and combinations
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_bracket_characters() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    // Generate test files with various bracket types
    let bracket_test_files = test_generator.generate_bracket_test_files().await?;
    let validator = CorrectnessValidator::new(&bracket_test_files.index_path, &bracket_test_files.vector_path).await?;
    
    let bracket_test_cases = vec![
        // Round brackets/parentheses
        ("()", vec!["functions.rs", "math.py"], "empty parentheses"),
        ("(condition)", vec!["if_statements.rs", "conditionals.py"], "condition in parentheses"),
        ("function(args)", vec!["function_calls.rs", "method_calls.py"], "function call"),
        ("((nested))", vec!["complex_expr.rs", "nested_calls.py"], "nested parentheses"),
        
        // Square brackets
        ("[]", vec!["arrays.rs", "lists.py"], "empty array/list"),
        ("[index]", vec!["array_access.rs", "list_access.py"], "array indexing"),
        ("[1, 2, 3]", vec!["array_literals.rs", "list_literals.py"], "array literal"),
        ("array[0][1]", vec!["multidim_arrays.rs", "nested_lists.py"], "multi-dimensional access"),
        
        // Curly braces
        ("{}", vec!["structs.rs", "dicts.py", "objects.js"], "empty block/object"),
        ("{field: value}", vec!["object_literals.js", "struct_init.rs"], "object literal"),
        ("if condition {", vec!["control_flow.rs", "blocks.rs"], "block start"),
        ("{{nested}}", vec!["templates.html", "nested_blocks.rs"], "nested braces"),
        
        // Angle brackets
        ("<>", vec!["generics.rs", "templates.cpp"], "empty angle brackets"),
        ("<T>", vec!["generic_types.rs", "templates.cpp"], "generic type parameter"),
        ("<html>", vec!["markup.html", "xml_files.xml"], "HTML/XML tag"),
        ("Vec<String>", vec!["collections.rs", "generic_usage.rs"], "generic collection"),
        
        // Mixed bracket combinations
        ("([{}])", vec!["complex_syntax.rs", "nested_structures.py"], "mixed nested brackets"),
        ("{[key]: (value)}", vec!["complex_objects.js", "mixed_syntax.py"], "complex bracket mixing"),
    ];
    
    for (query, expected_files, description) in bracket_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::TextSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed bracket test {}: {} - {}", description, query, result.summary());
        assert!(result.precision >= 0.85, "Low precision for {}: {:.2}", description, result.precision);
    }
    
    println!("Bracket characters test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_operator_characters() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let operator_test_files = test_generator.generate_operator_test_files().await?;
    let validator = CorrectnessValidator::new(&operator_test_files.index_path, &operator_test_files.vector_path).await?;
    
    let operator_test_cases = vec![
        // Arithmetic operators
        ("+", vec!["arithmetic.rs", "math_ops.py", "calculations.js"], "addition operator"),
        ("-", vec!["arithmetic.rs", "math_ops.py", "calculations.js"], "subtraction operator"),
        ("*", vec!["arithmetic.rs", "math_ops.py", "calculations.js"], "multiplication operator"),
        ("/", vec!["arithmetic.rs", "math_ops.py", "calculations.js"], "division operator"),
        ("%", vec!["modulo.rs", "remainder.py", "mod_ops.js"], "modulo operator"),
        ("**", vec!["power.py", "exponent.js"], "power operator"),
        
        // Comparison operators
        ("==", vec!["comparisons.rs", "equality.py", "equals.js"], "equality operator"),
        ("!=", vec!["comparisons.rs", "inequality.py", "not_equals.js"], "inequality operator"),
        ("<", vec!["comparisons.rs", "less_than.py", "comparison.js"], "less than operator"),
        (">", vec!["comparisons.rs", "greater_than.py", "comparison.js"], "greater than operator"),
        ("<=", vec!["comparisons.rs", "less_equal.py", "comparison.js"], "less than or equal"),
        (">=", vec!["comparisons.rs", "greater_equal.py", "comparison.js"], "greater than or equal"),
        
        // Logical operators
        ("&&", vec!["logical.rs", "boolean_ops.cpp"], "logical AND"),
        ("||", vec!["logical.rs", "boolean_ops.cpp"], "logical OR"),
        ("!", vec!["logical.rs", "boolean_ops.py", "negation.js"], "logical NOT"),
        ("and", vec!["logical.py", "boolean_logic.py"], "Python logical AND"),
        ("or", vec!["logical.py", "boolean_logic.py"], "Python logical OR"),
        ("not", vec!["logical.py", "boolean_logic.py"], "Python logical NOT"),
        
        // Bitwise operators
        ("&", vec!["bitwise.rs", "bit_ops.cpp"], "bitwise AND"),
        ("|", vec!["bitwise.rs", "bit_ops.cpp"], "bitwise OR"),
        ("^", vec!["bitwise.rs", "xor_ops.cpp"], "bitwise XOR"),
        ("<<", vec!["bitshift.rs", "shift_ops.cpp"], "left shift"),
        (">>", vec!["bitshift.rs", "shift_ops.cpp"], "right shift"),
        ("~", vec!["bitwise.rs", "complement.cpp"], "bitwise complement"),
        
        // Assignment operators
        ("=", vec!["assignments.rs", "variables.py", "assign.js"], "assignment operator"),
        ("+=", vec!["compound_assign.rs", "operators.py"], "add assignment"),
        ("-=", vec!["compound_assign.rs", "operators.py"], "subtract assignment"),
        ("*=", vec!["compound_assign.rs", "operators.py"], "multiply assignment"),
        ("/=", vec!["compound_assign.rs", "operators.py"], "divide assignment"),
        
        // Special operators
        ("->", vec!["pointers.cpp", "arrow_functions.js", "closures.rs"], "arrow operator"),
        ("=>", vec!["match_arms.rs", "arrow_functions.js"], "fat arrow"),
        ("::", vec!["namespaces.cpp", "modules.rs"], "scope resolution"),
        ("?", vec!["optional.rs", "ternary.js", "nullable.ts"], "question mark operator"),
        ("??", vec!["null_coalescing.js", "optional_chaining.ts"], "null coalescing"),
        ("...", vec!["spread.js", "variadic.cpp", "rest_params.ts"], "spread/rest operator"),
    ];
    
    for (query, expected_files, description) in operator_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::TextSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.80,
            recall_threshold: 0.75,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed operator test {}: {} - {}", description, query, result.summary());
    }
    
    println!("Operator characters test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_quote_and_string_characters() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let quote_test_files = test_generator.generate_quote_test_files().await?;
    let validator = CorrectnessValidator::new(&quote_test_files.index_path, &quote_test_files.vector_path).await?;
    
    let quote_test_cases = vec![
        // Single quotes
        ("'hello'", vec!["strings.py", "char_literals.cpp", "quotes.js"], "single quoted string"),
        ("'world'", vec!["string_literals.py", "characters.rs"], "single quoted literal"),
        ("'\\'escaped\\''", vec!["escaped_strings.py", "escape_sequences.js"], "escaped single quotes"),
        
        // Double quotes
        ("\"hello\"", vec!["strings.rs", "string_literals.py", "quotes.js"], "double quoted string"),
        ("\"world\"", vec!["string_literals.rs", "text_data.py"], "double quoted literal"),
        ("\"\\\\\\\"\\\\\\\"\", vec![\"escaped_strings.rs\", \"json_data.json\"], \"escaped double quotes\"),
        
        // Backticks/template literals
        ("`template`", vec!["template_strings.js", "backticks.ts"], "template literal"),
        ("`${variable}`", vec!["interpolation.js", "template_vars.ts"], "template interpolation"),
        
        // Raw strings
        ("r\"raw string\"", vec!["raw_strings.py", "regex_patterns.py"], "Python raw string"),
        ("r#\"raw string\"#", vec!["raw_strings.rs", "regex_literals.rs"], "Rust raw string"),
        
        // Multi-line strings
        ("\"\"\"multi-line\"\"\"", vec!["docstrings.py", "multiline.py"], "Python triple quotes"),
        ("'''multi-line'''", vec!["docstrings.py", "text_blocks.py"], "Python single triple quotes"),
        
        // Escape sequences
        ("\\n", vec!["newlines.rs", "escape_chars.py", "line_breaks.js"], "newline escape"),
        ("\\t", vec!["tabs.rs", "whitespace.py", "formatting.js"], "tab escape"),
        ("\\r", vec!["carriage_return.rs", "line_endings.py"], "carriage return"),
        ("\\\\", vec!["backslashes.rs", "paths.py", "escapes.js"], "escaped backslash"),
        ("\\0", vec!["null_chars.rs", "terminators.cpp"], "null character"),
        ("\\x20", vec!["hex_escapes.rs", "ascii_codes.py"], "hex escape sequence"),
        ("\\u{1F600}", vec!["unicode_escapes.rs", "emoji_codes.py"], "Unicode escape"),
        
        // String concatenation
        ("\"hello\" + \"world\"", vec!["string_concat.js", "text_joining.py"], "string concatenation"),
        ("\"hello\" \"world\"", vec!["adjacent_strings.cpp", "literal_concat.py"], "adjacent string literals"),
    ];
    
    for (query, expected_files, description) in quote_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::TextSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed quote test {}: {} - {}", description, query, result.summary());
    }
    
    println!("Quote and string characters test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_punctuation_and_symbols() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let punctuation_test_files = test_generator.generate_punctuation_test_files().await?;
    let validator = CorrectnessValidator::new(&punctuation_test_files.index_path, &punctuation_test_files.vector_path).await?;
    
    let punctuation_test_cases = vec![
        // Semicolons and colons
        (";", vec!["statements.rs", "line_endings.cpp", "semicolons.js"], "semicolon"),
        (":", vec!["types.rs", "dictionaries.py", "labels.cpp"], "colon"),
        ("::", vec!["namespaces.cpp", "modules.rs", "scope.hpp"], "double colon"),
        
        // Commas and periods
        (",", vec!["lists.py", "arrays.rs", "parameters.js"], "comma"),
        (".", vec!["methods.rs", "attributes.py", "properties.js"], "period/dot"),
        ("...", vec!["ellipsis.py", "spread.js", "variadic.cpp"], "ellipsis"),
        
        // Hash/pound symbols
        ("#", vec!["comments.py", "directives.cpp", "hashtags.md"], "hash symbol"),
        ("##", vec!["markdown.md", "headings.md", "double_hash.py"], "double hash"),
        ("###", vec!["markdown.md", "triple_hash.py"], "triple hash"),
        
        // At symbols
        ("@", vec!["decorators.py", "annotations.java", "mentions.md"], "at symbol"),
        ("@@", vec!["sql_variables.sql", "double_at.py"], "double at symbol"),
        
        // Dollar signs
        ("$", vec!["variables.bash", "jquery.js", "shell_vars.sh"], "dollar sign"),
        ("${", vec!["template_vars.js", "bash_expansion.sh"], "dollar brace"),
        ("$$", vec!["double_dollar.sql", "shell_pid.sh"], "double dollar"),
        
        // Percent signs
        ("%", vec!["modulo.rs", "formatting.py", "percent_ops.js"], "percent sign"),
        ("%%", vec!["jupyter.py", "double_percent.py"], "double percent"),
        
        // Ampersands
        ("&", vec!["references.rs", "bitwise.cpp", "html_entities.html"], "ampersand"),
        ("&&", vec!["logical_and.rs", "boolean_ops.js"], "double ampersand"),
        
        // Pipe symbols
        ("|", vec!["bitwise_or.rs", "pipes.bash", "or_ops.js"], "pipe symbol"),
        ("||", vec!["logical_or.rs", "boolean_ops.js"], "double pipe"),
        
        // Tilde
        ("~", vec!["bitwise_not.rs", "home_dir.bash", "complement.cpp"], "tilde"),
        ("~~", vec!["double_tilde.js", "strikethrough.md"], "double tilde"),
        
        // Caret
        ("^", vec!["bitwise_xor.rs", "exponent.py", "regex_start.py"], "caret"),
        ("^^", vec!["double_caret.rs", "xor_ops.cpp"], "double caret"),
        
        // Backslashes
        ("\\", vec!["escapes.rs", "windows_paths.py", "backslash.js"], "backslash"),
        ("\\\\", vec!["double_backslash.rs", "escaped_paths.py"], "double backslash"),
        
        // Forward slashes
        ("/", vec!["division.rs", "paths.py", "comments.js"], "forward slash"),
        ("//", vec!["comments.cpp", "floor_division.py", "line_comments.js"], "double slash"),
        ("///", vec!["doc_comments.rs", "triple_slash.cpp"], "triple slash"),
        
        // Underscores
        ("_", vec!["identifiers.rs", "snake_case.py", "private_vars.js"], "underscore"),
        ("__", vec!["dunder_methods.py", "double_underscore.rs"], "double underscore"),
        ("___", vec!["triple_underscore.py", "separators.md"], "triple underscore"),
    ];
    
    for (query, expected_files, description) in punctuation_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::TextSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.80,
            recall_threshold: 0.75,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed punctuation test {}: {} - {}", description, query, result.summary());
    }
    
    println!("Punctuation and symbols test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_unicode_and_international_characters() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let unicode_test_files = test_generator.generate_unicode_test_files().await?;
    let validator = CorrectnessValidator::new(&unicode_test_files.index_path, &unicode_test_files.vector_path).await?;
    
    let unicode_test_cases = vec![
        // Mathematical symbols
        ("π", vec!["math_constants.py", "greek_letters.rs"], "pi symbol"),
        ("∑", vec!["summation.py", "math_symbols.rs"], "summation symbol"),
        ("∆", vec!["delta.py", "change_symbols.rs"], "delta symbol"),
        ("∞", vec!["infinity.py", "math_limits.rs"], "infinity symbol"),
        ("√", vec!["square_root.py", "math_functions.rs"], "square root symbol"),
        ("∫", vec!["integration.py", "calculus.rs"], "integral symbol"),
        
        // Arrow symbols
        ("→", vec!["arrows.rs", "flow_charts.md"], "right arrow"),
        ("←", vec!["left_arrows.rs", "reverse_flow.md"], "left arrow"),
        ("↑", vec!["up_arrows.rs", "increase.md"], "up arrow"),
        ("↓", vec!["down_arrows.rs", "decrease.md"], "down arrow"),
        ("⇒", vec!["double_arrows.rs", "implies.md"], "double right arrow"),
        ("⇐", vec!["double_left_arrows.rs", "implied_by.md"], "double left arrow"),
        
        // Emojis
        ("🚀", vec!["emojis.md", "rocket_symbols.py"], "rocket emoji"),
        ("💡", vec!["idea_emojis.md", "lightbulb.py"], "lightbulb emoji"),
        ("🔥", vec!["fire_emojis.md", "hot_topics.py"], "fire emoji"),
        ("⚡", vec!["lightning.md", "fast_symbols.py"], "lightning emoji"),
        ("🎯", vec!["target_emojis.md", "goal_symbols.py"], "target emoji"),
        
        // Currency symbols
        ("$", vec!["dollar_amounts.py", "currency.rs"], "dollar sign"),
        ("€", vec!["euro_amounts.py", "european_currency.rs"], "euro symbol"),
        ("¢", vec!["cent_amounts.py", "small_currency.rs"], "cent symbol"),
        ("£", vec!["pound_amounts.py", "british_currency.rs"], "pound symbol"),
        ("¥", vec!["yen_amounts.py", "japanese_currency.rs"], "yen symbol"),
        
        // Accented characters
        ("café", vec!["french_words.py", "accented_text.rs"], "French accented word"),
        ("naïve", vec!["accented_words.py", "diaeresis.rs"], "word with diaeresis"),
        ("résumé", vec!["resume_files.py", "french_accents.rs"], "resume with accents"),
        ("piñata", vec!["spanish_words.py", "tilde_accent.rs"], "Spanish tilde"),
        
        // Diacritical marks
        ("é", vec!["acute_accents.py", "french_chars.rs"], "e with acute accent"),
        ("ñ", vec!["spanish_chars.py", "tilde_letters.rs"], "n with tilde"),
        ("ü", vec!["german_chars.py", "umlaut_letters.rs"], "u with umlaut"),
        ("ç", vec!["cedilla_chars.py", "french_letters.rs"], "c with cedilla"),
        
        // Non-Latin scripts
        ("こんにちは", vec!["japanese_text.py", "hiragana.rs"], "Japanese hiragana"),
        ("你好", vec!["chinese_text.py", "mandarin.rs"], "Chinese characters"),
        ("مرحبا", vec!["arabic_text.py", "rtl_scripts.rs"], "Arabic text"),
        ("Здравствуйте", vec!["russian_text.py", "cyrillic.rs"], "Russian Cyrillic"),
        ("안녕하세요", vec!["korean_text.py", "hangul.rs"], "Korean Hangul"),
        
        // Special Unicode categories
        ("\u{200B}", vec!["zero_width_space.py", "invisible_chars.rs"], "zero-width space"),
        ("\u{FEFF}", vec!["byte_order_mark.py", "bom_chars.rs"], "byte order mark"),
        ("\u{202E}", vec!["rtl_override.py", "text_direction.rs"], "right-to-left override"),
    ];
    
    for (query, expected_files, description) in unicode_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::TextSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.85,
            recall_threshold: 0.80,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed Unicode test {}: {} - {}", description, query, result.summary());
    }
    
    println!("Unicode and international characters test completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_complex_special_character_combinations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_generator = TestDataGenerator::new(temp_dir.path())?;
    
    let complex_test_files = test_generator.generate_complex_character_test_files().await?;
    let validator = CorrectnessValidator::new(&complex_test_files.index_path, &complex_test_files.vector_path).await?;
    
    let complex_test_cases = vec![
        // Programming constructs with multiple special characters
        ("fn main() -> Result<(), Error>", vec!["rust_functions.rs"], "Rust function signature"),
        ("std::collections::HashMap<String, i32>", vec!["rust_types.rs"], "Rust generic types"),
        ("console.log(`Hello ${name}!`);", vec!["javascript_template.js"], "JavaScript template literal"),
        ("@pytest.mark.parametrize(\"input,expected\", [(1, 2), (3, 4)])", vec!["python_decorators.py"], "Python decorator with parameters"),
        ("SELECT * FROM users WHERE name LIKE '%john%' AND age >= 18;", vec!["sql_queries.sql"], "SQL query with wildcards"),
        
        // Regular expressions with special characters
        ("/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/", vec!["email_regex.js", "validation.py"], "email regex pattern"),
        ("/\\d{3}-\\d{3}-\\d{4}/", vec!["phone_regex.js", "phone_validation.py"], "phone number regex"),
        ("/^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]{8,}$/", vec!["password_regex.js"], "complex password regex"),
        
        // JSON and data structures
        ("{\"name\": \"John\", \"age\": 30, \"active\": true}", vec!["json_data.json", "objects.js"], "JSON object"),
        ("[{\"id\": 1, \"tags\": [\"urgent\", \"bug\"]}, {\"id\": 2, \"tags\": [\"feature\"]}]", vec!["json_arrays.json"], "JSON array of objects"),
        
        // Configuration and markup
        ("<div class=\"container\" id=\"main-content\" data-value=\"123\">", vec!["html_elements.html"], "HTML element with attributes"),
        ("<!-- This is a comment with special chars: @#$%^&*()_+ -->", vec!["html_comments.html"], "HTML comment with specials"),
        
        // Mathematical expressions
        ("f(x) = ax² + bx + c", vec!["math_equations.py", "quadratic.rs"], "quadratic equation"),
        ("∂f/∂x = lim(h→0) [f(x+h) - f(x)]/h", vec!["calculus.py", "derivatives.rs"], "derivative definition"),
        
        // File paths and URLs
        ("C:\\Users\\Documents\\file with spaces & symbols!.txt", vec!["windows_paths.py"], "Windows path with specials"),
        ("https://example.com/api/v1/users?filter=active&sort=name#section-1", vec!["urls.py", "api_endpoints.rs"], "URL with query params"),
        
        // Command line and shell
        ("grep -r \"pattern\" --include=\"*.rs\" --exclude-dir=\"target\" .", vec!["shell_commands.sh"], "grep command with options"),
        ("find . -name \"*.py\" -exec python -m py_compile {} \\;", vec!["find_commands.sh"], "find with exec"),
        
        // Markdown and documentation
        ("## Heading with `code` and **bold** and *italic* text", vec!["markdown_files.md"], "markdown with formatting"),
        ("- [x] Task with ~strikethrough~ and [link](https://example.com)", vec!["task_lists.md"], "markdown task list"),
        
        // Complex string literals
        ("r#\"This is a \"raw\" string with 'quotes' and \\backslashes\"#", vec!["rust_raw_strings.rs"], "Rust raw string"),
        ("f\"Hello {user.name}, you have {len(items)} item{'s' if len(items) != 1 else ''}\"", vec!["python_f_strings.py"], "Python f-string"),
    ];
    
    for (query, expected_files, description) in complex_test_cases {
        let case = GroundTruthCase {
            query: query.to_string(),
            expected_files: expected_files.into_iter().map(|f| f.to_string()).collect(),
            query_type: QueryType::TextSearch,
            expected_count: expected_files.len(),
            precision_threshold: 0.90,
            recall_threshold: 0.85,
        };
        
        let result = validator.validate(&case).await?;
        assert!(result.is_correct, "Failed complex test {}: {} - {}", description, query, result.summary());
        assert!(result.precision >= 0.90, "Low precision for complex test {}: {:.2}", description, result.precision);
    }
    
    println!("Complex special character combinations test completed successfully");
    Ok(())
}
```

## Success Criteria
- All bracket types ((), [], {}, <>) are correctly indexed and searchable
- Operator characters (+, -, *, /, ==, !=, etc.) work in search queries
- Quote characters and string literals are handled properly
- Punctuation and symbols are correctly tokenized
- Unicode characters including emojis and international text work
- Complex combinations of special characters are indexed accurately
- Precision >= 85% for bracket and quote tests
- Precision >= 80% for operator and punctuation tests
- Precision >= 90% for complex combination tests
- All test categories complete without errors

## Time Limit
10 minutes maximum