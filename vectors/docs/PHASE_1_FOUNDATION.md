# Phase 1: Foundation - Core Text Search Without FTS5

## Objective
Build a robust text search system that handles ALL special characters correctly without using SQLite FTS5.

## Duration
2 Days (16 hours of parallel subagent work)

## Problem Statement
SQLite FTS5 cannot handle special characters like `##`, `[]`, `<>` despite escaping attempts. We need a custom solution that works 100% of the time.

## Technical Approach

### 1. Custom Text Index
```python
# Instead of FTS5, use regular SQLite with Python string matching
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    content TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    chunk_index INTEGER,
    metadata JSON
);
CREATE INDEX idx_file_path ON documents(file_path);
```

### 2. Python-Based Search
- Use Python's string operations for exact matching
- Implement our own tokenization
- Handle special characters natively

### 3. Document/Chunk Management
- Track document-chunk relationships
- Enable cross-chunk searching
- Maintain chunk overlap context

## Subagent Tasks

### SA1-1: Core Database Schema (2 hours)
**Task**: Create database schema without FTS5
**Test First**: 
```python
def test_database_creation():
    db = TextSearchDB("test.db")
    assert db.table_exists("documents")
    assert db.can_store_special_chars("##", "[]", "<>")
```
**Success Criteria**: 
- Database accepts all Unicode characters
- No SQL syntax errors on special characters
- Proper indexing for performance

### SA1-2: Document Storage (3 hours)
**Task**: Implement document storage with chunk tracking
**Test First**:
```python
def test_document_storage():
    db = TextSearchDB("test.db")
    doc_id = db.add_document(
        file_path="/test/file.rs",
        content="pub fn test() { ## comment }",
        chunk_index=0
    )
    assert db.get_document(doc_id).content == "pub fn test() { ## comment }"
```
**Success Criteria**:
- Store documents with all special characters
- Track chunk relationships
- Retrieve by doc_id efficiently

### SA1-3: Exact String Search (4 hours)
**Task**: Implement Python-based exact string search
**Test First**:
```python
def test_exact_search():
    db = TextSearchDB("test.db")
    db.add_document("test.rs", "[workspace]")
    results = db.search("[workspace]")
    assert len(results) == 1
    assert "[workspace]" in results[0].content
    # Should NOT match [dependencies]
    results = db.search("[dependencies]")
    assert len(results) == 0
```
**Success Criteria**:
- Find exact string matches
- Handle all special characters
- No false positives

### SA1-4: Tokenization Engine (3 hours)
**Task**: Build tokenizer for code-aware searching
**Test First**:
```python
def test_tokenization():
    tokenizer = CodeTokenizer()
    tokens = tokenizer.tokenize("pub fn test<T>() -> Result<T, E>")
    assert "pub" in tokens
    assert "fn" in tokens
    assert "test" in tokens
    assert "Result<T, E>" in tokens  # Keep generics together
```
**Success Criteria**:
- Extract meaningful tokens
- Preserve special constructs
- Language-aware tokenization

### SA1-5: Query Parser Foundation (2 hours)
**Task**: Create base query parser
**Test First**:
```python
def test_query_parser():
    parser = QueryParser()
    query = parser.parse("Result<T, E>")
    assert query.type == "exact"
    assert query.escaped == "Result<T, E>"  # No escaping needed
```
**Success Criteria**:
- Parse queries without breaking on special chars
- Identify query types
- No SQL injection vulnerabilities

### SA1-6: Integration Tests (2 hours)
**Task**: Verify all components work together
**Test First**:
```python
def test_end_to_end():
    system = TextSearchSystem()
    system.index_file("Cargo.toml")  # Has [workspace]
    
    # Test 1: Exact bracket match
    results = system.search("[workspace]")
    assert any("[workspace]" in r.content for r in results)
    assert not any("[dependencies]" in r.content for r in results)
    
    # Test 2: Hash symbols
    results = system.search("##")
    assert all("##" in r.content for r in results)
    
    # Test 3: Generics
    results = system.search("Result<T, E>")
    assert len(results) > 0
```
**Success Criteria**:
- All special characters work
- No SQL errors
- Correct results returned

## Review Subagents

### RSA1-1: Code Review (1 hour per task)
- Review each implementation
- Check for SQL injection
- Verify error handling
- Ensure no FTS5 usage

### RSA1-2: Performance Review (2 hours)
- Benchmark search speed
- Check memory usage
- Verify index efficiency
- Test with 10,000 documents

## Deliverables

### Code Files
1. `text_search_db.py` - Core database without FTS5
2. `document_store.py` - Document/chunk management
3. `search_engine.py` - Python-based search
4. `tokenizer.py` - Code-aware tokenization
5. `query_parser.py` - Base query parsing

### Test Files
1. `test_text_search_db.py` - Database tests
2. `test_document_store.py` - Storage tests
3. `test_search_engine.py` - Search tests
4. `test_tokenizer.py` - Tokenization tests
5. `test_integration.py` - End-to-end tests

### Documentation
1. API documentation
2. Performance benchmarks
3. Special character support matrix

## Success Metrics

### Functional
- [ ] 100% of special characters searchable
- [ ] Zero SQL syntax errors
- [ ] Exact matches only (no false positives)

### Performance  
- [ ] Search < 50ms for 1,000 documents
- [ ] Index 100 documents/second
- [ ] Memory < 100MB for 10,000 documents

### Quality
- [ ] All tests passing
- [ ] No FTS5 dependencies
- [ ] Clean error handling

## Risks & Mitigations

### Risk 1: Performance without FTS5
**Mitigation**: Use proper SQL indexes, implement caching

### Risk 2: Memory usage for large documents
**Mitigation**: Implement streaming, chunk size limits

### Risk 3: Unicode edge cases
**Mitigation**: Comprehensive Unicode test suite

## Dependencies
- Python 3.8+
- SQLite3 (without FTS5)
- No external search libraries

## Next Phase
Once Phase 1 is complete and all tests pass, move to Phase 2: Boolean Logic Implementation