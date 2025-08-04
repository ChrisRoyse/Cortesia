# Micro-Task 092: Generate Regex Patterns

## Objective
Generate files containing regular expression patterns with special characters to test vector search handling of regex syntax.

## Context
Regular expressions use special characters with specific meanings that differ from normal text. Vector search must handle regex patterns correctly while maintaining search accuracy for pattern matching use cases.

## Prerequisites
- Task 091 completed (Markup patterns generated)

## Time Estimate
7 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create regex pattern script `generate_regex_patterns.py`:
   ```python
   #!/usr/bin/env python3
   """
   Generate regex pattern files for vector search testing.
   """
   
   import os
   import sys
   from pathlib import Path
   sys.path.append('templates')
   from template_generator import TestFileGenerator
   
   def generate_regex_files():
       """Generate files with regex pattern examples."""
       generator = TestFileGenerator()
       
       # Comprehensive regex patterns file
       regex_patterns = '''Regular Expression Patterns for Vector Search Testing
   
   # Basic Character Classes
   [a-z]          # Lowercase letters
   [A-Z]          # Uppercase letters  
   [0-9]          # Digits
   [a-zA-Z0-9]    # Alphanumeric characters
   [^a-z]         # Not lowercase letters
   \\d             # Digits (\\d equivalent to [0-9])
   \\w             # Word characters (\\w equivalent to [a-zA-Z0-9_])
   \\s             # Whitespace characters
   \\D             # Non-digits
   \\W             # Non-word characters
   \\S             # Non-whitespace characters
   
   # Quantifiers
   a?             # Zero or one 'a'
   a*             # Zero or more 'a'
   a+             # One or more 'a'
   a{3}           # Exactly 3 'a's
   a{2,5}         # Between 2 and 5 'a's
   a{3,}          # 3 or more 'a's
   
   # Anchors and Boundaries
   ^start         # Beginning of line
   end$           # End of line
   \\bword\\b      # Word boundary
   \\Bnon\\B       # Non-word boundary
   
   # Groups and Alternation
   (abc)          # Capturing group
   (?:abc)        # Non-capturing group
   (?<name>abc)   # Named capturing group
   a|b            # Alternation (a or b)
   (red|blue|green) # Multiple alternatives
   
   # Special Characters and Escaping
   \\.             # Literal dot
   \\*             # Literal asterisk
   \\+             # Literal plus
   \\?             # Literal question mark
   \\[             # Literal opening bracket
   \\]             # Literal closing bracket
   \\{             # Literal opening brace
   \\}             # Literal closing brace
   \\(             # Literal opening parenthesis
   \\)             # Literal closing parenthesis
   \\|             # Literal pipe
   \\\\            # Literal backslash
   
   # Common Programming Patterns
   
   ## Rust Type Patterns
   Result<\\w+,\\s*\\w+>                    # Result<T, E>
   Vec<[^>]+>                              # Vec<Type>
   HashMap<[^,]+,\\s*[^>]+>                 # HashMap<K, V>
   Option<[^>]+>                           # Option<T>
   Box<dyn\\s+\\w+>                         # Box<dyn Trait>
   
   ## Function Signatures
   fn\\s+\\w+\\([^)]*\\)\\s*->\\s*[^\\{]+    # Function with return type
   async\\s+fn\\s+\\w+\\([^)]*\\)           # Async function
   pub\\s+fn\\s+\\w+                        # Public function
   
   ## Macro Patterns
   #\\[derive\\([^)]+\\)\\]                 # Derive macro
   #\\[cfg\\([^)]+\\)\\]                    # Config macro
   #\\[test\\]                              # Test attribute
   #\\[allow\\([^)]+\\)\\]                  # Allow attribute
   
   ## Generic Constraints
   <T:\\s*\\w+>                             # Simple trait bound
   <T:\\s*\\w+\\s*\\+\\s*\\w+>              # Multiple trait bounds
   where\\s+T:\\s*\\w+                      # Where clause
   
   # Data Format Patterns
   
   ## Email Addresses
   [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}
   
   ## URLs
   https?://[^\\s/$.?#].[^\\s]*
   
   ## IP Addresses  
   (?:[0-9]{1,3}\\.){3}[0-9]{1,3}
   
   ## Phone Numbers
   \\+?[1-9]\\d{1,14}                       # International format
   \\(?[0-9]{3}\\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4} # US format
   
   ## Dates
   \\d{4}-\\d{2}-\\d{2}                     # ISO date (YYYY-MM-DD)
   \\d{1,2}/\\d{1,2}/\\d{4}                 # US date (M/D/YYYY)
   \\d{1,2}\\.\\d{1,2}\\.\\d{4}             # European date (D.M.YYYY)
   
   ## Version Numbers
   \\d+\\.\\d+\\.\\d+(-[a-zA-Z0-9]+)?      # Semantic versioning
   v?\\d+\\.\\d+(\\.\\d+)?                  # Simple version
   
   # Complex Patterns
   
   ## JSON Value Extraction
   "([^"\\\\]|\\\\.)*"                      # JSON string
   -?\\d+(\\.\\d+)?([eE][+-]?\\d+)?         # JSON number
   
   ## Configuration Sections
   \\[([^\\]]+)\\]                          # TOML/INI section headers
   (\\w+)\\s*=\\s*(.+)                      # Key-value pairs
   
   ## Code Comments
   //.*$                                   # Single line comment
   /\\*[\\s\\S]*?\\*/                       # Multi-line comment
   #.*$                                    # Hash comment
   
   ## String Literals
   "([^"\\\\]|\\\\.)*"                      # Double-quoted string
   '([^'\\\\]|\\\\.)*'                      # Single-quoted string
   `([^\\\\`]|\\\\.)*`                      # Backtick string
   r"[^"]*"                                # Raw string
   
   # Windows-Specific Patterns
   
   ## File Paths
   [A-Za-z]:\\\\[^<>:"|?*\\n\\r]*           # Windows absolute path
   \\.\\\\[^<>:"|?*\\n\\r]*                 # Relative path with backslash
   \\\\\\\\[^\\\\]+\\\\[^<>:"|?*\\n\\r]*     # UNC path
   
   ## Registry Keys
   HKEY_[A-Z_]+\\\\[^\\n\\r]*               # Registry key path
   
   ## Environment Variables
   %[A-Za-z_][A-Za-z0-9_]*%                # Windows env var
   \\$\\{[A-Za-z_][A-Za-z0-9_]*\\}          # Unix-style env var
   
   # Advanced Patterns
   
   ## Lookahead and Lookbehind
   (?=pattern)                             # Positive lookahead
   (?!pattern)                             # Negative lookahead
   (?<=pattern)                            # Positive lookbehind
   (?<!pattern)                            # Negative lookbehind
   
   ## Unicode Categories
   \\p{L}                                   # Any letter
   \\p{N}                                   # Any number
   \\p{P}                                   # Any punctuation
   \\p{S}                                   # Any symbol
   
   ## Case-Insensitive Patterns
   (?i)pattern                             # Case-insensitive flag
   [Tt]est                                 # Manual case variants
   
   # Testing Patterns for Vector Search
   
   ## These patterns should be found correctly:
   Result<String, Error>
   fn main() -> ()
   #[derive(Debug, Clone)]
   [workspace]
   https://api.example.com/v1
   user@example.com
   192.168.1.1
   2024-08-04
   
   ## Edge cases to handle:
   Result<Vec<HashMap<String, Value>>, Box<dyn Error>>
   async fn process<T: Send + Sync>() -> impl Future<Output = Result<T, Error>>
   #[cfg(all(target_os = "windows", feature = "advanced"))]
   \\\\server\\share\\path\\to\\file.txt
   ${DATABASE_URL:-postgresql://localhost/db}'''
       
       # Generate the regex patterns file
       samples = [
           ("regex_patterns.txt", "Comprehensive regex patterns with special characters", regex_patterns)
       ]
       
       generated_files = []
       for filename, pattern_focus, content in samples:
           output_path = generator.generate_text_file(
               filename,
               "code_samples",
               pattern_focus,
               content,
               "code_samples"
           )
           generated_files.append(output_path)
           print(f"Generated: {output_path}")
       
       return generated_files
   
   def main():
       """Main generation function."""
       print("Generating regex pattern files...")
       
       # Ensure output directory exists
       os.makedirs("code_samples", exist_ok=True)
       
       try:
           files = generate_regex_files()
           print(f"\nSuccessfully generated {len(files)} regex pattern files:")
           for file_path in files:
               print(f"  - {file_path}")
           
           print("\nRegex pattern generation completed successfully!")
           return 0
       
       except Exception as e:
           print(f"Error generating regex files: {e}")
           return 1
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
3. Run generation: `python generate_regex_patterns.py`
4. Return to root: `cd ..\..`  
5. Commit: `git add data\test_files\generate_regex_patterns.py data\test_files\code_samples && git commit -m "task_092: Generate regex patterns with special characters"`

## Expected Output
- Comprehensive regex pattern reference file
- Programming language pattern examples
- Data format validation patterns
- Windows-specific path and variable patterns

## Success Criteria
- [ ] Regex patterns file generated with comprehensive examples
- [ ] Programming language patterns included
- [ ] Data format patterns covered
- [ ] Windows-specific patterns included

## Validation Commands
```cmd
cd data\test_files
python generate_regex_patterns.py
dir code_samples
```

## Next Task
task_093_validate_code_generation.md

## Notes
- Regex patterns test special character interpretation
- Programming patterns validate code-specific syntax handling
- Data format patterns ensure structured content recognition
- Windows patterns test platform-specific syntax support