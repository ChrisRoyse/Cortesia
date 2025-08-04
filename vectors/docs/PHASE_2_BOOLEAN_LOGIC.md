# Phase 2: Boolean Logic - True AND/OR/NOT Implementation

## Objective
Implement REAL boolean logic that correctly handles AND (all terms required), OR (any term), and NOT (exclusion).

## Duration
2 Days (16 hours of parallel subagent work)

## Problem Statement
Current "boolean" search returns wrong results:
- AND returns files with ANY term (acting like OR)
- NOT doesn't properly exclude
- No understanding of document vs chunk boundaries

## Technical Approach

### 1. Document-Level Boolean Logic
```python
# Track which documents contain which terms
class BooleanEngine:
    def search_and(self, terms: List[str]) -> List[Document]:
        # Return only documents containing ALL terms
        # Terms can be in different chunks of same document
        
    def search_or(self, terms: List[str]) -> List[Document]:
        # Return documents containing ANY term
        
    def search_not(self, include: str, exclude: str) -> List[Document]:
        # Return documents with include but WITHOUT exclude
```

### 2. Cross-Chunk Term Tracking
- Build term occurrence map per document
- Handle terms split across chunks
- Maintain document coherence

## Subagent Tasks

### SA2-1: Boolean AND Implementation (4 hours)
**Task**: Build true AND logic requiring ALL terms
**Test First**:
```python
def test_boolean_and():
    engine = BooleanEngine()
    engine.index_document("file1.rs", "pub struct MyStruct")
    engine.index_document("file2.rs", "fn process()")
    engine.index_document("file3.rs", "pub fn initialize()")
    
    results = engine.search_and(["pub", "fn"])
    assert len(results) == 1  # Only file3 has both
    assert "file3.rs" in results[0].path
    assert "file1.rs" not in [r.path for r in results]  # Has pub but not fn
    assert "file2.rs" not in [r.path for r in results]  # Has fn but not pub
```
**Success Criteria**:
- Returns ONLY documents with ALL terms
- Zero false positives
- Handle terms in different chunks

### SA2-2: Boolean OR Implementation (3 hours)
**Task**: Build OR logic for ANY term matching
**Test First**:
```python
def test_boolean_or():
    engine = BooleanEngine()
    engine.index_document("file1.rs", "struct Data")
    engine.index_document("file2.rs", "enum Status")
    engine.index_document("file3.rs", "fn process()")
    
    results = engine.search_or(["struct", "enum"])
    assert len(results) == 2  # file1 and file2
    assert "file3.rs" not in [r.path for r in results]
```
**Success Criteria**:
- Returns documents with ANY term
- No duplicates
- Proper result merging

### SA2-3: Boolean NOT Implementation (3 hours)
**Task**: Build NOT logic for exclusion
**Test First**:
```python
def test_boolean_not():
    engine = BooleanEngine()
    engine.index_document("file1.rs", "impl Display for Error")
    engine.index_document("file2.rs", "impl Debug")
    engine.index_document("file3.rs", "impl Display for Success")
    
    results = engine.search_not("impl", "Error")
    assert len(results) == 2  # file2 and file3
    assert "file1.rs" not in [r.path for r in results]  # Excluded due to Error
```
**Success Criteria**:
- Excludes ALL documents containing exclude term
- Include term must be present
- Handle exclusion across chunks

### SA2-4: Nested Boolean Expressions (4 hours)
**Task**: Handle complex nested expressions
**Test First**:
```python
def test_nested_boolean():
    engine = BooleanEngine()
    # (pub OR private) AND struct
    results = engine.search_nested("(pub OR private) AND struct")
    
    # Verify: Document must have (pub OR private) AND also have struct
    for result in results:
        has_visibility = "pub" in result.content or "private" in result.content
        has_struct = "struct" in result.content
        assert has_visibility and has_struct
```
**Success Criteria**:
- Parse nested expressions correctly
- Evaluate in correct order
- Handle arbitrary nesting depth

### SA2-5: Cross-Chunk Boolean (2 hours)
**Task**: Handle boolean logic across chunk boundaries
**Test First**:
```python
def test_cross_chunk_boolean():
    engine = BooleanEngine()
    # Document split into chunks
    engine.index_chunk("file.rs", chunk=0, content="pub struct")
    engine.index_chunk("file.rs", chunk=1, content="impl Display")
    
    # Should find document even though terms in different chunks
    results = engine.search_and(["pub", "Display"])
    assert len(results) == 1
    assert "file.rs" in results[0].path
```
**Success Criteria**:
- Track terms across chunks
- Maintain document coherence
- Correct aggregation

## Review Subagents

### RSA2-1: Logic Verification (2 hours)
**Task**: Verify boolean logic correctness
**Tests**:
```python
def test_logic_verification():
    # Test 50 different boolean combinations
    # Verify against ground truth
    # No false positives or negatives
```

### RSA2-2: Performance Testing (2 hours)
**Task**: Benchmark boolean operations
**Requirements**:
- AND query < 100ms for 10,000 documents
- OR query < 100ms for 10,000 documents
- Complex nested < 200ms

## Deliverables

### Code Files
1. `boolean_engine.py` - Core boolean logic
2. `expression_parser.py` - Parse boolean expressions
3. `term_tracker.py` - Track terms across documents
4. `result_aggregator.py` - Combine boolean results

### Test Files
1. `test_boolean_and.py` - AND logic tests
2. `test_boolean_or.py` - OR logic tests
3. `test_boolean_not.py` - NOT logic tests
4. `test_nested_boolean.py` - Complex expression tests
5. `test_cross_chunk.py` - Multi-chunk tests

## Success Metrics

### Correctness
- [ ] AND returns ONLY docs with ALL terms
- [ ] OR returns docs with ANY term
- [ ] NOT properly excludes documents
- [ ] Nested expressions evaluate correctly

### Performance
- [ ] Boolean queries < 100ms
- [ ] Scales to 10,000 documents
- [ ] Memory efficient

### Quality
- [ ] 100% test coverage
- [ ] No false positives
- [ ] No false negatives

## Common Pitfalls to Avoid

### Pitfall 1: Chunk-Level Instead of Document-Level
**Wrong**: Return chunks that match
**Right**: Return documents where all chunks collectively match

### Pitfall 2: String Contains Instead of Token Match
**Wrong**: "function" matches "functionality"
**Right**: Word boundary matching

### Pitfall 3: Case Sensitivity Issues
**Wrong**: "Pub" doesn't match "pub"
**Right**: Case-insensitive option available

## Integration Points

### From Phase 1
- Use TextSearchDB for storage
- Use tokenizer for term extraction
- Use query parser for expression parsing

### To Phase 3
- Boolean engine becomes base for proximity search
- Expression parser extended for advanced queries

## Next Phase
Once Phase 2 is complete with 100% accuracy on boolean logic, move to Phase 3: Advanced Search Features