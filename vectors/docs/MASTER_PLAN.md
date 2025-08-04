# Vector Indexing System - Master Redesign Plan

## Overview
Complete redesign of the vector indexing system to achieve TRUE 100% accuracy on all query types with production-ready quality.

## Current Reality Assessment

### What's Actually Broken
1. **No Actual System**: Most Python files don't exist in filesystem
2. **FTS5 Fundamentally Broken**: Cannot handle special characters despite escaping attempts
3. **Boolean Logic Fake**: AND returns OR results, NOT doesn't exclude properly
4. **Proximity Search Fake**: Doesn't calculate actual word distance
5. **Tests Pass on Wrong Results**: Validation checks presence, not correctness
6. **Parallel Processing Broken**: Cannot pickle local functions
7. **Incremental Indexing Fake**: MD5 hashing doesn't trigger properly

### What We Learned
- FTS5 is incompatible with special characters
- Chunking before indexing breaks cross-chunk searches
- Semantic fallbacks hide failures
- Need unified query language
- Must validate correctness, not just presence

## System Requirements

### Core Requirements
1. **100% Special Character Support**: `[]`, `<>`, `##`, `::`, `->`, `&mut`, etc.
2. **True Boolean Logic**: AND requires ALL terms, OR requires ANY, NOT excludes
3. **Accurate Proximity Search**: NEAR/N checks actual word distance
4. **Exact Phrase Matching**: Quotes match exact sequences
5. **Wildcard Support**: `*` and `?` patterns work correctly
6. **Performance**: <100ms query time for 10,000 documents
7. **Scale**: Parallel processing for 100,000+ files
8. **Incremental**: Only reindex changed files

## Architecture Design

### Core Components
1. **Text Index**: Custom implementation without FTS5 limitations
2. **Vector Store**: ChromaDB for semantic search
3. **Document Store**: Unified document/chunk management
4. **Query Engine**: Single parser for all query types
5. **Result Validator**: Ensures correctness before returning
6. **Parallel Processor**: Proper multiprocessing implementation
7. **Change Tracker**: Real incremental indexing

## Implementation Phases

### Phase 1: Foundation (Days 1-2)
- Build core text search without FTS5
- Implement document storage with chunk tracking
- Create unified query parser
- **Deliverable**: Basic exact search working with all special characters

### Phase 2: Boolean Logic (Days 3-4)
- Implement true AND/OR/NOT logic
- Handle cross-chunk boolean operations
- Add nested expression support
- **Deliverable**: Boolean queries return correct results

### Phase 3: Advanced Search (Days 5-6)
- Add proximity search with distance calculation
- Implement phrase matching
- Add wildcard support
- **Deliverable**: All query types working correctly

### Phase 4: Scale & Performance (Days 7-8)
- Fix parallel processing
- Implement true incremental indexing
- Add caching layer
- **Deliverable**: Process 10,000 files in parallel

### Phase 5: Integration (Days 9-10)
- Integrate with ChromaDB for hybrid search
- Add result ranking and relevance
- Implement query optimization
- **Deliverable**: Complete system with semantic + exact search

### Phase 6: Validation (Days 11-12)
- Create comprehensive test suite
- Validate against ground truth
- Performance benchmarking
- **Deliverable**: 100% accuracy proven with metrics

## Success Criteria

### Functional Requirements
- [ ] All special characters searchable without errors
- [ ] Boolean AND returns only documents with ALL terms
- [ ] Boolean OR returns documents with ANY term
- [ ] Boolean NOT properly excludes documents
- [ ] Proximity search measures actual distance
- [ ] Phrase search matches exact sequences
- [ ] Wildcards work as expected

### Performance Requirements
- [ ] Query response < 100ms for 10,000 documents
- [ ] Index 1,000 files/minute
- [ ] Parallel processing scales to CPU cores
- [ ] Memory usage < 1GB for 100,000 documents

### Quality Requirements
- [ ] 100% test coverage
- [ ] No mock data or stubs
- [ ] All integration points verified
- [ ] Production error handling
- [ ] Clear documentation

## Subagent Task Distribution

Each phase will spawn multiple subagents working in parallel:
- **Core Subagents**: Implement main functionality
- **Test Subagents**: Write failing tests first (TDD)
- **Review Subagents**: Validate each implementation
- **Integration Subagents**: Ensure components work together

## Next Steps

1. Review and approve this plan
2. Begin Phase 1 implementation
3. Daily progress reviews
4. Iterate until 100% accuracy achieved

---

*This plan follows CLAUDE.md principles: brutal honesty, TDD, one feature at a time, break things internally, optimize only after it works.*