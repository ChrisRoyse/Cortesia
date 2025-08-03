# Universal RAG Indexing System - Simulation Environments

This directory contains three comprehensive simulation environments designed to test the Universal RAG Indexing System across various challenging scenarios.

## Overview

Each simulation environment tests different aspects of the indexing system's capabilities:

1. **Multi-Language Project** - Tests language detection and cross-language code understanding
2. **Evolving Codebase** - Tests incremental indexing and change detection
3. **Edge Cases** - Tests robustness against challenging input patterns

## Simulation Environments

### 1. Multi-Language Project (`1_multi_language/`)

A realistic full-stack e-commerce application featuring:

**Backend (Python)**
- `app.py` - FastAPI application with complex models and relationships
- `models.py` - Advanced SQLAlchemy models with business logic

**Frontend (JavaScript/TypeScript)**
- `app.js` - React-like frontend application with modern patterns
- `components.tsx` - TypeScript React components with hooks and context

**Microservice (Rust)**
- `src/main.rs` - High-performance recommendation engine
- `Cargo.toml` - Rust project configuration

**Configuration Files**
- `database.yml` - Multi-environment database configuration
- `application.toml` - Comprehensive application settings
- `deployment.json` - Kubernetes deployment configuration

**Documentation**
- `README.md` - Comprehensive project documentation

**Key Features for Testing:**
- Multiple programming languages in one project
- Complex nested data structures
- Modern frameworks and patterns
- Configuration in various formats (YAML, TOML, JSON)
- Realistic code complexity and structure
- Cross-language imports and dependencies

### 2. Evolving Codebase (`2_evolving_codebase/`)

A Git repository simulating natural code evolution:

**Evolution Timeline:**
1. **Initial Commit** - Basic calculator with add/subtract operations
2. **Feature Addition** - Added multiply, divide, and advanced operations
3. **Enhancement** - Improved interface and error handling
4. **Major Refactor** - Replaced CLI with GUI, added configuration
5. **Final Polish** - Added documentation and comprehensive tests

**Files Through Evolution:**
- `calculator.py` - Core calculator class (modified)
- `utils.py` - Utility functions (enhanced)
- `main.py` - Command-line interface (deleted)
- `advanced_operations.py` - Mathematical functions (added)
- `gui_calculator.py` - GUI interface (added)
- `config.py` - Configuration management (added)
- `test_calculator.py` - Unit tests (added)
- `README.md` - Documentation (added)

**Key Features for Testing:**
- Incremental indexing as code evolves
- File creation, modification, and deletion
- Code refactoring and restructuring
- Change impact analysis
- Git history integration
- Documentation evolution

### 3. Edge Cases (`3_edge_cases/`)

Challenging patterns to test system robustness:

**Files:**

1. **`massive_function.py`** - Function with 1000+ lines
   - Tests chunking strategies for very long functions
   - Complex nested logic and extensive comments
   - Multiple responsibilities in single function (anti-pattern)

2. **`deeply_nested.py`** - Extremely nested class structures
   - Classes within classes within classes (10+ levels)
   - Tests extraction of deeply nested code elements
   - Complex inheritance and composition patterns

3. **`syntax_errors.py`** - Various syntax errors
   - Missing colons, unmatched parentheses
   - Incorrect indentation, unclosed strings
   - Invalid variable names, broken imports
   - Tests error handling and recovery

4. **`mixed_languages.py`** - Multiple languages in one file
   - Python with embedded SQL, HTML, CSS, JavaScript
   - Tests language detection and boundary identification
   - Realistic web application patterns

5. **`unicode_content.py`** - Extensive Unicode and multilingual content
   - Multiple scripts (Latin, Cyrillic, Chinese, Arabic, etc.)
   - Emoji, mathematical symbols, currency symbols
   - Tests encoding handling and character recognition

6. **`duplicate_code.py`** - Intentional code duplication
   - Exact duplicates with different names
   - Similar logic with minor variations
   - Structural duplicates with different implementations
   - Tests deduplication algorithms

7. **`empty_file.py`** - Completely empty file
8. **`empty_file.js`** - Empty JavaScript file
9. **`whitespace_only.py`** - File with only whitespace

**Key Features for Testing:**
- Handling of malformed or incomplete code
- Language detection in mixed-content files
- Unicode and character encoding challenges
- Code deduplication across variations
- Empty file handling
- Performance with very large functions
- Extraction from deeply nested structures

## Testing Scenarios

### Language Detection
- **Test:** Process files from all three simulations
- **Expected:** Accurate identification of Python, JavaScript, TypeScript, Rust, SQL, HTML, CSS, YAML, TOML, JSON

### Code Extraction
- **Test:** Extract functions, classes, and methods from complex files
- **Expected:** Proper identification and extraction despite nesting, syntax errors, or length

### Chunking Strategies
- **Test:** Process `massive_function.py` with 1000+ line function
- **Expected:** Intelligent chunking that preserves context and meaning

### Change Detection
- **Test:** Process evolving codebase commits sequentially
- **Expected:** Accurate identification of additions, modifications, deletions

### Error Recovery
- **Test:** Process files with syntax errors
- **Expected:** Graceful handling, extraction of valid portions

### Deduplication
- **Test:** Process duplicate code patterns
- **Expected:** Identification of similar/identical code blocks

### Performance Testing
- **Test:** Process all simulations together
- **Expected:** Reasonable performance across diverse content types

## Usage Instructions

### Running Individual Simulations

```bash
# Test multi-language project
python indexer_universal.py --path ./vectors/simulation/1_multi_language/

# Test evolving codebase (process each commit)
cd ./vectors/simulation/2_evolving_codebase/
git log --oneline  # See available commits
git checkout <commit-hash>
python ../../indexer_universal.py --path ./

# Test edge cases
python indexer_universal.py --path ./vectors/simulation/3_edge_cases/
```

### Comprehensive Testing

```bash
# Test all simulations
python test_improvements.py --comprehensive
```

### Performance Benchmarking

```bash
# Benchmark against all simulation environments
python benchmark_indexer.py --simulations all
```

## Expected Outcomes

### Language Detection
- Multi-language project: 5+ languages detected
- Mixed languages file: 6+ languages in single file
- Edge cases: Robust handling of malformed content

### Code Structure Extraction
- Functions: 100+ extracted across all simulations
- Classes: 50+ extracted with proper nesting
- Methods: 200+ extracted from various contexts

### Chunking Performance
- Massive function: Intelligently split into meaningful chunks
- Deeply nested: Proper handling of 10+ nesting levels
- Regular code: Standard chunking maintains context

### Change Tracking
- Evolving codebase: 5 commits properly processed
- File changes: Additions, modifications, deletions tracked
- Incremental updates: Only changed code re-indexed

### Error Resilience
- Syntax errors: System continues processing despite errors
- Empty files: Gracefully handled without failures
- Unicode content: Proper encoding and character handling

### Deduplication Accuracy
- Exact duplicates: 100% identification rate
- Similar code: 80%+ identification of variants
- Structural similarity: Detection of common patterns

## Metrics to Track

1. **Processing Time** per simulation environment
2. **Memory Usage** during indexing
3. **Accuracy Rates** for language detection
4. **Extraction Completeness** (functions/classes found vs. expected)
5. **Error Recovery Rate** (valid content extracted from malformed files)
6. **Deduplication Precision** (true positives vs. false positives)
7. **Change Detection Accuracy** (incremental updates)

These simulation environments provide comprehensive testing coverage for the Universal RAG Indexing System, ensuring robust performance across real-world scenarios and edge cases.