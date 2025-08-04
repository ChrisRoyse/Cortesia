# Phase 3: Advanced Search - Proximity, Phrases, and Wildcards

## Objective
Implement advanced search features with ACTUAL functionality, not fake implementations.

## Duration
2 Days (16 hours of parallel subagent work)

## Problem Statement
Current implementations are fake:
- Proximity search doesn't calculate actual word distance
- Phrase search doesn't match exact sequences
- Wildcards don't work correctly

## Technical Approach

### 1. Real Proximity Calculation
```python
def calculate_proximity(text: str, term1: str, term2: str, max_distance: int) -> List[Match]:
    # Find actual positions of terms
    # Calculate word distance between them
    # Return only if within max_distance
```

### 2. Exact Phrase Matching
```python
def match_phrase(text: str, phrase: str) -> List[Match]:
    # Match exact sequence including spaces
    # Handle punctuation correctly
    # Case-sensitive option
```

### 3. Wildcard Pattern Matching
```python
def match_wildcard(text: str, pattern: str) -> List[Match]:
    # Convert wildcards to regex
    # * = zero or more characters
    # ? = exactly one character
    # Handle escape sequences
```

## Subagent Tasks

### SA3-1: Proximity Search Implementation (4 hours)
**Task**: Build proximity search with actual distance calculation
**Test First**:
```python
def test_proximity_search():
    engine = ProximityEngine()
    
    # Test 1: Within proximity
    engine.index("file1.rs", "pub static fn new")
    results = engine.search_proximity("pub", "fn", max_distance=3)
    assert len(results) == 1  # Within 3 words
    
    # Test 2: Outside proximity
    engine.index("file2.rs", "pub is a visibility modifier that is often used while fn creates functions")
    results = engine.search_proximity("pub", "fn", max_distance=3)
    assert len(results) == 0  # More than 3 words apart
    
    # Test 3: Exact distance calculation
    engine.index("file3.rs", "pub one two fn")
    match = engine.search_proximity("pub", "fn", max_distance=3)
    assert match[0].distance == 3  # Exactly 3 words apart
```
**Success Criteria**:
- Calculate actual word distance
- Respect max_distance strictly
- Handle multiple occurrences

### SA3-2: Phrase Search Implementation (3 hours)
**Task**: Build exact phrase matching
**Test First**:
```python
def test_phrase_search():
    engine = PhraseEngine()
    
    # Test 1: Exact match
    engine.index("file1.rs", "pub fn initialize() -> Result")
    results = engine.search_phrase("pub fn initialize")
    assert len(results) == 1
    
    # Test 2: No partial matches
    results = engine.search_phrase("fn pub")  # Wrong order
    assert len(results) == 0
    
    # Test 3: Handle spaces correctly
    engine.index("file2.rs", "pub  fn  test()")  # Multiple spaces
    results = engine.search_phrase("pub fn test")
    assert len(results) == 1  # Should normalize spaces
```
**Success Criteria**:
- Match exact sequences
- Handle whitespace normalization
- Case sensitivity option

### SA3-3: Wildcard Implementation (3 hours)
**Task**: Build working wildcard patterns
**Test First**:
```python
def test_wildcard_search():
    engine = WildcardEngine()
    
    # Test 1: Star wildcard
    engine.index("file1.rs", "SpikingNeuralNetwork")
    results = engine.search_wildcard("Spike*Network")
    assert len(results) == 1
    
    # Test 2: Question mark wildcard
    engine.index("file2.rs", "test1 test2 test3")
    results = engine.search_wildcard("test?")
    assert len(results) == 1
    assert all(r.match in ["test1", "test2", "test3"] for r in results)
    
    # Test 3: Complex patterns
    engine.index("file3.rs", "get_user_by_id")
    results = engine.search_wildcard("get_*_by_*")
    assert len(results) == 1
```
**Success Criteria**:
- Convert wildcards to regex correctly
- Handle edge cases
- Efficient pattern matching

### SA3-4: Combined Query Support (3 hours)
**Task**: Combine proximity, phrase, and boolean
**Test First**:
```python
def test_combined_queries():
    engine = CombinedEngine()
    
    # Test: "pub fn" NEAR/5 initialize
    results = engine.search_combined('"pub fn" NEAR/5 initialize')
    
    # Verify: Must have exact phrase "pub fn" within 5 words of "initialize"
    for result in results:
        assert "pub fn" in result.content
        distance = engine.calculate_distance("pub fn", "initialize", result.content)
        assert distance <= 5
```
**Success Criteria**:
- Parse combined syntax
- Apply operations in correct order
- Maintain performance

### SA3-5: Regex Support (3 hours)
**Task**: Add full regex pattern support
**Test First**:
```python
def test_regex_search():
    engine = RegexEngine()
    
    # Test 1: Function pattern
    engine.index("file1.rs", "pub fn process_data()")
    results = engine.search_regex(r"pub fn \w+\(\)")
    assert len(results) == 1
    
    # Test 2: Generic pattern
    engine.index("file2.rs", "Result<String, Error>")
    results = engine.search_regex(r"Result<\w+, \w+>")
    assert len(results) == 1
```
**Success Criteria**:
- Support standard regex syntax
- Handle capture groups
- Prevent regex DoS

## Review Subagents

### RSA3-1: Accuracy Verification (2 hours)
**Task**: Verify search accuracy
- Test 100 proximity queries
- Test 100 phrase queries
- Test 100 wildcard patterns
- Zero false positives

### RSA3-2: Edge Case Testing (2 hours)
**Task**: Test edge cases
- Empty patterns
- Special characters in phrases
- Overlapping matches
- Unicode handling

## Deliverables

### Code Files
1. `proximity_engine.py` - Distance calculation
2. `phrase_engine.py` - Exact phrase matching
3. `wildcard_engine.py` - Pattern matching
4. `regex_engine.py` - Regex support
5. `combined_engine.py` - Combined query handling

### Test Files
1. `test_proximity.py` - Distance tests
2. `test_phrase.py` - Phrase match tests
3. `test_wildcard.py` - Pattern tests
4. `test_regex.py` - Regex tests
5. `test_combined.py` - Integration tests

## Success Metrics

### Functional
- [ ] Proximity calculates actual distance
- [ ] Phrases match exact sequences
- [ ] Wildcards work as documented
- [ ] Regex patterns supported

### Performance
- [ ] Proximity search < 150ms
- [ ] Phrase search < 100ms
- [ ] Wildcard search < 200ms
- [ ] Regex search < 300ms

### Accuracy
- [ ] 100% correct distance calculation
- [ ] Zero false phrase matches
- [ ] Wildcards match expected patterns

## Algorithm Details

### Proximity Algorithm
```python
def calculate_proximity(text: str, term1: str, term2: str) -> int:
    words = text.split()
    positions1 = [i for i, w in enumerate(words) if w.lower() == term1.lower()]
    positions2 = [i for i, w in enumerate(words) if w.lower() == term2.lower()]
    
    min_distance = float('inf')
    for p1 in positions1:
        for p2 in positions2:
            distance = abs(p2 - p1) - 1  # Words between
            min_distance = min(min_distance, distance)
    
    return min_distance
```

### Phrase Algorithm
```python
def match_phrase(text: str, phrase: str) -> List[int]:
    # Normalize spaces
    phrase_normalized = ' '.join(phrase.split())
    text_normalized = ' '.join(text.split())
    
    # Find all occurrences
    positions = []
    start = 0
    while True:
        pos = text_normalized.find(phrase_normalized, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    
    return positions
```

## Next Phase
Once Phase 3 is complete with all advanced search features working correctly, move to Phase 4: Scale & Performance