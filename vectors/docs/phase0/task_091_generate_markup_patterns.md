# Micro-Task 091: Generate Markup Patterns

## Objective
Generate markup files (HTML, XML, Markdown) with special characters to test vector search handling of structured document formats.

## Context
Markup languages use special characters for structure definition that should be distinguished from content. Vector search must properly handle markup syntax while extracting meaningful text content.

## Prerequisites
- Task 090 completed (Shell script patterns generated)

## Time Estimate
8 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create markup generation script `generate_markup_patterns.py`:
   ```python
   #!/usr/bin/env python3
   """
   Generate markup pattern files for vector search testing.
   """
   
   import os
   import sys
   from pathlib import Path
   sys.path.append('templates')
   from template_generator import TestFileGenerator
   
   def generate_markup_files():
       """Generate markup files with special character patterns."""
       generator = TestFileGenerator()
       
       # HTML with special characters
       html_content = '''<!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>Vector Search Test Page &mdash; Special Characters</title>
       <style>
           .code { font-family: monospace; background: #f0f0f0; padding: 2px 4px; }
           .highlight { background: yellow; }
           .error { color: red; font-weight: bold; }
       </style>
   </head>
   <body>
       <header>
           <h1>Vector Search System Test &amp; Validation</h1>
           <nav>
               <a href="#overview">Overview</a> |
               <a href="#patterns">Patterns</a> |
               <a href="#examples">Examples</a>
           </nav>
       </header>
       
       <main>
           <section id="overview">
               <h2>Special Character Handling</h2>
               <p>This document tests various <strong>special characters</strong> in HTML context:</p>
               <ul>
                   <li>HTML entities: &lt;tag&gt;, &amp;amp;, &quot;quotes&quot;, &#39;apostrophe&#39;</li>
                   <li>Unicode characters: &#8364; (Euro), &#8482; (Trademark), &#169; (Copyright)</li>
                   <li>Mathematical symbols: &lt;= &gt;= &ne; &plusmn; &times; &divide;</li>
                   <li>Arrows and symbols: &larr; &rarr; &uarr; &darr; &harr;</li>
               </ul>
           </section>
           
           <section id="patterns">
               <h2>Code Patterns in HTML</h2>
               <p>Testing code representation within markup:</p>
               
               <h3>Rust Code Example</h3>
               <pre><code>
   fn process_data&lt;T&gt;(input: Vec&lt;T&gt;) -&gt; Result&lt;HashMap&lt;String, T&gt;, Error&gt; {
       let mut result = HashMap::new();
       
       for (index, item) in input.iter().enumerate() {
           let key = format!("item_{}", index);
           result.insert(key, item.clone());
       }
       
       Ok(result)
   }
               </code></pre>
               
               <h3>JSON Configuration</h3>
               <pre><code>
   {
       "api": {
           "base_url": "https://api.example.com/v1",
           "timeout": 30,
           "retries": 3
       },
       "features": {
           "user_auth": true,
           "rate_limiting": {
               "enabled": true,
               "requests_per_minute": 100
           }
       }
   }
               </code></pre>
           </section>
           
           <section id="examples">
               <h2>Interactive Examples</h2>
               
               <form action="/search" method="post">
                   <fieldset>
                       <legend>Search Configuration</legend>
                       
                       <label for="query">Search Query:</label>
                       <input type="text" id="query" name="query" 
                              placeholder="Enter search terms..." 
                              value="Result&lt;T, E&gt;" />
                       
                       <label for="filters">Filters:</label>
                       <select id="filters" name="filters[]" multiple>
                           <option value="rust">Rust (.rs)</option>
                           <option value="json">JSON (.json)</option>
                           <option value="toml">TOML (.toml)</option>
                           <option value="md">Markdown (.md)</option>
                       </select>
                       
                       <label>
                           <input type="checkbox" name="case_sensitive" value="1" />
                           Case sensitive search
                       </label>
                       
                       <button type="submit">Search &rarr;</button>
                   </fieldset>
               </form>
               
               <table border="1">
                   <thead>
                       <tr>
                           <th>Pattern</th>
                           <th>Description</th>
                           <th>Example</th>
                       </tr>
                   </thead>
                   <tbody>
                       <tr>
                           <td><code>&lt;T&gt;</code></td>
                           <td>Generic type parameter</td>
                           <td><code>Vec&lt;String&gt;</code></td>
                       </tr>
                       <tr>
                           <td><code>-&gt;</code></td>
                           <td>Function return type</td>
                           <td><code>fn name() -&gt; Result&lt;(), Error&gt;</code></td>
                       </tr>
                       <tr>
                           <td><code>#[derive]</code></td>
                           <td>Rust derive macro</td>
                           <td><code>#[derive(Debug, Clone)]</code></td>
                       </tr>
                   </tbody>
               </table>
           </section>
       </main>
       
       <footer>
           <p>&copy; 2024 Vector Search Testing Suite. All rights reserved.</p>
           <p>Generated on: <time datetime="2024-08-04">August 4, 2024</time></p>
       </footer>
   </body>
   </html>'''
       
       # XML with namespaces and special characters
       xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
   <project xmlns="http://example.com/schema"
            xmlns:config="http://example.com/config"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://example.com/schema project.xsd">
   
       <metadata>
           <name>VectorSearchApp</name>
           <version>1.0.0</version>
           <description><![CDATA[
               Vector search system with special character handling.
               Supports patterns like: <T>, Result<T, E>, fn() -> Type
           ]]></description>
           <authors>
               <author email="dev@example.com">Developer</author>
           </authors>
       </metadata>
       
       <config:settings>
           <config:database>
               <config:connection_string>
                   postgresql://user:pass@localhost:5432/db?sslmode=require&amp;timeout=30
               </config:connection_string>
               <config:pool_size min="2" max="20" />
               <config:timeout unit="seconds">30</config:timeout>
           </config:database>
           
           <config:api>
               <config:base_url>https://api.example.com/v1</config:base_url>
               <config:endpoints>
                   <config:endpoint name="users" path="/users" methods="GET,POST,PUT,DELETE" />
                   <config:endpoint name="search" path="/search" methods="GET,POST">
                       <config:parameters>
                           <config:param name="q" type="string" required="true" />
                           <config:param name="filters" type="array" required="false" />
                           <config:param name="limit" type="integer" default="10" />
                       </config:parameters>
                   </config:endpoint>
               </config:endpoints>
           </config:api>
       </config:settings>
       
       <build>
           <dependencies>
               <dependency name="serde" version="1.0" features="derive" />
               <dependency name="tokio" version="1.0" features="full,rt-multi-thread" />
               <dependency name="reqwest" version="0.11" features="json,rustls-tls" />
           </dependencies>
           
           <targets>
               <target name="debug" type="binary">
                   <source>src/main.rs</source>
                   <flags>--cfg debug_assertions</flags>
               </target>
               <target name="release" type="binary">
                   <source>src/main.rs</source>
                   <flags>--release -C target-cpu=native</flags>
                   <optimizations level="3" lto="true" />
               </target>
           </targets>
       </build>
       
       <tests>
           <test_suite name="unit_tests">
               <test name="test_generics" file="tests/generics.rs">
                   <pattern>Result&lt;T, E&gt;</pattern>
                   <pattern>Vec&lt;String&gt;</pattern>
                   <pattern>HashMap&lt;K, V&gt;</pattern>
               </test>
               <test name="test_functions" file="tests/functions.rs">
                   <pattern>fn name() -&gt; Type</pattern>
                   <pattern>async fn name() -&gt; impl Future</pattern>
               </test>
               <test name="test_macros" file="tests/macros.rs">
                   <pattern>#[derive(Debug)]</pattern>
                   <pattern>#[cfg(test)]</pattern>
               </test>
           </test_suite>
       </tests>
       
       <deployment>
           <environments>
               <environment name="development">
                   <variables>
                       <variable name="LOG_LEVEL" value="debug" />
                       <variable name="DB_URL" value="postgresql://localhost/dev_db" />
                   </variables>
               </environment>
               <environment name="production">
                   <variables>
                       <variable name="LOG_LEVEL" value="info" />
                       <variable name="DB_URL" value="${DATABASE_URL}" />
                   </variables>
               </environment>
           </environments>
       </deployment>
   </project>'''
       
       # Markdown with special characters
       markdown_content = '''# Vector Search System Documentation
   
   ## Overview
   
   This document describes the **Vector Search System** with special character handling capabilities. The system processes various programming patterns including:
   
   - Rust generics: `Result<T, E>`, `Vec<String>`, `HashMap<K, V>`
   - Function signatures: `fn name() -> Type`, `async fn() -> impl Future`
   - Macro patterns: `#[derive(Debug)]`, `#[cfg(target_os = "windows")]`
   
   ## Installation & Setup
   
   ### Prerequisites
   
   1. **Rust toolchain** (version >= 1.70)
   2. **System dependencies**:
      - Windows: Visual Studio Build Tools
      - Linux: `build-essential`, `pkg-config`
      - macOS: Xcode Command Line Tools
   
   ### Build Instructions
   
   ```bash
   # Clone the repository
   git clone https://github.com/example/vector-search.git
   cd vector-search
   
   # Build in release mode
   cargo build --release
   
   # Run tests
   cargo test
   ```
   
   ## Configuration
   
   The system uses TOML configuration with the following structure:
   
   ```toml
   [workspace]
   members = ["core", "api", "cli"]
   
   [workspace.dependencies]
   serde = { version = "1.0", features = ["derive"] }
   tokio = { version = "1.0", features = ["full"] }
   
   [package]
   name = "vector-search"
   version = "1.0.0"
   edition = "2021"
   
   [dependencies]
   # Core dependencies
   tantivy = "0.19"
   lancedb = "0.1"
   
   [features]
   default = ["search", "indexing"]
   search = []
   indexing = []
   advanced = ["search", "indexing", "ml"]
   ```
   
   ## API Reference
   
   ### Core Types
   
   | Type | Description | Example |
   |------|-------------|---------|
   | `SearchResult<T>` | Generic search result | `SearchResult<Document>` |
   | `Index<T, E>` | Index with error handling | `Index<Document, IndexError>` |
   | `Query<F>` | Query with filter type | `Query<DocumentFilter>` |
   
   ### Functions
   
   #### `search_documents`
   
   ```rust
   pub async fn search_documents<T>(
       query: &str,
       filters: Option<Vec<Filter>>,
       limit: usize,
   ) -> Result<Vec<T>, SearchError>
   where
       T: Serialize + DeserializeOwned,
   {
       // Implementation details...
   }
   ```
   
   #### `create_index`
   
   ```rust
   pub fn create_index<T: Indexable>(
       config: IndexConfig,
   ) -> Result<Index<T>, IndexError> {
       // Implementation details...
   }
   ```
   
   ## Special Character Handling
   
   The system handles various special character patterns:
   
   ### Code Patterns
   
   - **Generics**: `Vec<T>`, `Result<T, E>`, `HashMap<K, V>`
   - **References**: `&str`, `&mut T`, `&'a str`
   - **Operators**: `->`, `=>`, `::`, `..=`, `..`
   - **Attributes**: `#[derive]`, `#[cfg]`, `#[test]`
   
   ### Configuration Patterns  
   
   - **TOML sections**: `[package]`, `[dependencies]`, `[workspace]`
   - **Environment variables**: `${VAR}`, `$VAR`, `%VAR%`
   - **URLs**: `https://api.example.com/v1?param=value&other=123`
   
   ### Mathematical Expressions
   
   The system supports mathematical notation in search queries:
   
   - Comparisons: `>=`, `<=`, `==`, `!=`
   - Set operations: `∈`, `∉`, `⊆`, `⊇`
   - Logical operators: `∧`, `∨`, `¬`
   - Greek letters: α, β, γ, δ, ε, θ, λ, π, σ, φ, ψ, ω
   
   ## Testing
   
   ### Unit Tests
   
   ```bash
   # Run all tests
   cargo test
   
   # Run specific test pattern
   cargo test test_generics
   
   # Run with verbose output
   cargo test -- --nocapture
   ```
   
   ### Integration Tests
   
   Integration tests validate special character handling:
   
   - [ ] Rust generic patterns: `Result<T, E>`
   - [ ] Function arrows: `fn() -> Type`  
   - [ ] Macro attributes: `#[derive(Debug)]`
   - [ ] TOML sections: `[dependencies]`
   - [ ] URL patterns: `https://example.com`
   
   ## Performance Metrics
   
   | Metric | Value | Notes |
   |--------|-------|-------|
   | Index time | ~1ms/doc | For documents < 1KB |
   | Search time | ~5ms | For queries < 100 terms |
   | Memory usage | ~100MB | Base + 2MB per 10K docs |
   | Throughput | ~1000 QPS | On modern hardware |
   
   ## Troubleshooting
   
   ### Common Issues
   
   1. **Build errors with special characters**
      - Ensure UTF-8 encoding in source files
      - Check for BOM (Byte Order Mark) issues
      - Validate special character escaping
   
   2. **Search accuracy problems**
      - Verify pattern normalization
      - Check tokenization boundaries
      - Review similarity thresholds
   
   3. **Performance degradation**
      - Monitor index size growth
      - Check memory usage patterns
      - Profile query complexity
   
   ### Error Codes
   
   | Code | Description | Solution |
   |------|-------------|----------|
   | `E001` | Invalid UTF-8 sequence | Re-encode file as UTF-8 |
   | `E002` | Pattern parse error | Check special character escaping |
   | `E003` | Index corruption | Rebuild index from source |
   
   ---
   
   > **Note**: This documentation covers special character handling in vector search systems. For more details, see the [API documentation](https://docs.example.com/api) and [GitHub repository](https://github.com/example/vector-search).'''
       
       # Generate markup files
       samples = [
           ("test_document.html", "HTML with special characters and entities", html_content),
           ("project_config.xml", "XML with namespaces and special characters", xml_content),
           ("documentation.md", "Markdown with code patterns and special chars", markdown_content)
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
       print("Generating markup pattern files...")
       
       # Ensure output directory exists
       os.makedirs("code_samples", exist_ok=True)
       
       try:
           files = generate_markup_files()
           print(f"\nSuccessfully generated {len(files)} markup files:")
           for file_path in files:
               print(f"  - {file_path}")
           
           print("\nMarkup pattern generation completed successfully!")
           return 0
       
       except Exception as e:
           print(f"Error generating markup files: {e}")
           return 1
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
3. Run generation: `python generate_markup_patterns.py`
4. Return to root: `cd ..\..`
5. Commit: `git add data\test_files\generate_markup_patterns.py data\test_files\code_samples && git commit -m "task_091: Generate markup patterns with special characters"`

## Expected Output
- HTML file with entities and embedded code patterns
- XML file with namespaces and configuration structures  
- Markdown file with code blocks and documentation patterns
- Comprehensive markup syntax coverage

## Success Criteria
- [ ] HTML file with proper entity encoding
- [ ] XML file with namespace and structure patterns
- [ ] Markdown file with code blocks and tables
- [ ] All markup-specific special characters included

## Validation Commands
```cmd
cd data\test_files
python generate_markup_patterns.py
dir code_samples
```

## Next Task
task_092_generate_regex_patterns.md

## Notes
- Markup files test structured document parsing
- HTML entities test character encoding handling
- XML namespaces test complex structure processing
- Markdown code blocks test mixed content extraction