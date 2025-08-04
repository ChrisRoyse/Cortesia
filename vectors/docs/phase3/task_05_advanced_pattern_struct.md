# Task 05: Create AdvancedPatternEngine Struct

**Estimated Time: 10 minutes**

## Context
You have completed `ProximitySearchEngine` with proximity, phrase, and NEAR search. Now create `AdvancedPatternEngine` to handle wildcards, regex, and fuzzy search patterns.

## Current System State
- `ProximitySearchEngine` is fully implemented in `src/proximity.rs`
- Need higher-level pattern engine for wildcard/regex functionality

## Your Task
Create the `AdvancedPatternEngine` struct in `src/patterns.rs`.

## Required Implementation

```rust
use crate::proximity::ProximitySearchEngine;
use crate::types::SearchResult;
use anyhow::Result;

pub struct AdvancedPatternEngine {
    proximity_engine: ProximitySearchEngine,
}

impl AdvancedPatternEngine {
    pub fn new(index_path: &std::path::Path) -> Result<Self> {
        let proximity_engine = ProximitySearchEngine::new(index_path)?;
        Ok(Self { proximity_engine })
    }
    
    // Provide access to proximity features
    pub fn search_proximity(&self, term1: &str, term2: &str, max_distance: u32) -> Result<Vec<SearchResult>> {
        self.proximity_engine.search_proximity(term1, term2, max_distance)
    }
    
    pub fn search_phrase(&self, phrase: &str) -> Result<Vec<SearchResult>> {
        self.proximity_engine.search_phrase(phrase)
    }
    
    pub fn search_near(&self, query: &str) -> Result<Vec<SearchResult>> {
        self.proximity_engine.search_near(query)
    }
}
```

## Architecture Design
- `AdvancedPatternEngine` wraps `ProximitySearchEngine`
- Provides delegation methods for proximity features
- Will add wildcard, regex, and fuzzy methods in subsequent tasks
- Single entry point for all advanced search functionality

## Success Criteria
- [ ] File `src/patterns.rs` created
- [ ] `AdvancedPatternEngine` struct defined with `proximity_engine` field
- [ ] `new()` constructor creates `ProximitySearchEngine`
- [ ] Delegation methods for proximity, phrase, and NEAR search
- [ ] Proper imports and error handling
- [ ] Code compiles without errors

## Module Structure
```
src/
├── proximity.rs       # ProximitySearchEngine (completed)
├── patterns.rs        # AdvancedPatternEngine (this task)
├── boolean.rs         # BooleanSearchEngine (existing)
└── types.rs          # SearchResult type (existing)
```

## Next Task
After creating this struct, you'll implement wildcard search functionality.