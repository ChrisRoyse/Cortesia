# Task 01: Create ProximitySearchEngine Struct

**Estimated Time: 10 minutes**

## Context
You are implementing Phase 3 of an advanced vector search system using Rust and Tantivy. The system currently has a `BooleanSearchEngine` that handles basic boolean queries. You need to create a new `ProximitySearchEngine` that wraps the boolean engine and adds proximity search capabilities.

## Current System Overview
- Project is located in `src/` directory with modular structure
- Existing `BooleanSearchEngine` handles basic search queries 
- Tantivy is used as the underlying search engine
- The system processes code files and creates searchable indices

## Your Task
Create the basic `ProximitySearchEngine` struct in `src/proximity.rs`.

## Required Implementation

```rust
use crate::boolean::BooleanSearchEngine;
use crate::types::SearchResult;
use anyhow::Result;

pub struct ProximitySearchEngine {
    pub boolean_engine: BooleanSearchEngine,
}

impl ProximitySearchEngine {
    pub fn new(index_path: &std::path::Path) -> Result<Self> {
        let boolean_engine = BooleanSearchEngine::new(index_path)?;
        Ok(Self { boolean_engine })
    }
}
```

## Success Criteria
- [ ] File `src/proximity.rs` created
- [ ] `ProximitySearchEngine` struct defined with `boolean_engine` field
- [ ] `new()` constructor implemented that creates `BooleanSearchEngine`
- [ ] Proper imports for dependencies
- [ ] Code compiles without errors

## Dependencies Needed
- `anyhow` for error handling
- Access to existing `BooleanSearchEngine` 
- Access to `SearchResult` type

## Next Task
After completing this struct, the next task will implement the `search_proximity` method.