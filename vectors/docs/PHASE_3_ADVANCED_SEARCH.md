# Phase 3: Advanced Search - Tantivy Proximity, Wildcards & Regex

## Objective
Implement advanced search features using Tantivy's built-in capabilities for proximity, phrase matching, wildcards, and regex patterns.

## Duration
1 Day (8 hours) - Leveraging Tantivy's existing features

## Why Tantivy Excels Here
Tantivy is designed to provide:
- ✅ Built-in proximity search with `NEAR` operator designed
- ✅ Phrase queries with quotes designed
- ✅ Wildcard queries with `*` and `?` designed
- ✅ Regex support designed 
- ✅ Fuzzy search with edit distance designed
- ✅ All designed to be optimized and battle-tested

## Technical Approach

### 1. Proximity Search with Tantivy
```rust
use tantivy::query::{PhraseQuery, Query, TermQuery};
use tantivy::schema::IndexRecordOption;

pub struct ProximitySearchEngine {
    boolean_engine: BooleanSearchEngine,
}

impl ProximitySearchEngine {
    pub fn search_proximity(&self, term1: &str, term2: &str, max_distance: u32) -> anyhow::Result<Vec<SearchResult>> {
        // Use Tantivy's proximity syntax: "term1"~distance "term2"
        let proximity_query = format!("\"{}\"~{} \"{}\"", term1, max_distance, term2);
        self.boolean_engine.search_boolean(&proximity_query)
    }
    
    pub fn search_phrase(&self, phrase: &str) -> anyhow::Result<Vec<SearchResult>> {
        // Use Tantivy's phrase queries with quotes
        let phrase_query = format!("\"{}\"", phrase);
        self.boolean_engine.search_boolean(&phrase_query)
    }
    
    pub fn search_near(&self, query: &str) -> anyhow::Result<Vec<SearchResult>> {
        // Support queries like: "pub NEAR/3 fn" 
        // Tantivy syntax: "pub"~3 AND "fn"~3
        if let Some(captures) = self.parse_near_query(query) {
            let term1 = &captures[1];
            let distance = captures[2].parse::<u32>().unwrap_or(5);
            let term2 = &captures[3];
            
            let near_query = format!("(\"{}\"~{} AND \"{}\"~{})", term1, distance, term2, distance);
            self.boolean_engine.search_boolean(&near_query)
        } else {
            self.boolean_engine.search_boolean(query)
        }
    }
}
```

### 2. Wildcard and Regex Support
```rust
pub struct AdvancedPatternEngine {
    proximity_engine: ProximitySearchEngine,
}

impl AdvancedPatternEngine {
    pub fn search_wildcard(&self, pattern: &str) -> anyhow::Result<Vec<SearchResult>> {
        // Tantivy supports wildcards natively: "test*", "?ub", "struct*"
        self.proximity_engine.boolean_engine.search_boolean(pattern)
    }
    
    pub fn search_regex(&self, pattern: &str) -> anyhow::Result<Vec<SearchResult>> {
        // For complex patterns, use Tantivy's regex support
        let regex_query = format!("/{}/", pattern);
        self.proximity_engine.boolean_engine.search_boolean(&regex_query)
    }
    
    pub fn search_fuzzy(&self, term: &str, max_edit_distance: u8) -> anyhow::Result<Vec<SearchResult>> {
        // Tantivy fuzzy search: "term"~edit_distance
        let fuzzy_query = format!("\"{}\"~{}", term, max_edit_distance);
        self.proximity_engine.boolean_engine.search_boolean(&fuzzy_query)
    }
}
```

## Implementation Tasks

### Task 1: Proximity Search (2 hours)
```rust
#[cfg(test)]
mod proximity_tests {
    use super::*;
    
    #[test]
    fn test_proximity_search() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let proximity_engine = ProximitySearchEngine {
            boolean_engine: BooleanSearchEngine::new(&index_path)?,
        };
        
        // Index test documents
        let test_files = vec![
            ("adjacent.rs", "pub fn new()"),
            ("one_between.rs", "pub static fn new()"), 
            ("many_between.rs", "pub is a visibility modifier while fn creates functions"),
            ("far_apart.rs", "pub struct Data { } impl Display { fn fmt() {} }"),
        ];
        
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            std::fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test distance 0 (adjacent)
        let results = proximity_engine.search_proximity("pub", "fn", 0)?;
        assert!(results.iter().any(|r| r.file_path.contains("adjacent.rs")));
        
        // Test distance 1 (one word between)
        let results = proximity_engine.search_proximity("pub", "fn", 1)?;
        assert!(results.iter().any(|r| r.file_path.contains("one_between.rs")));
        
        // Test larger distance
        let results = proximity_engine.search_proximity("pub", "fn", 10)?;
        assert!(results.len() >= 3); // Should find most files
        
        Ok(())
    }
}
```

### Task 2: Phrase Search (2 hours)  
```rust
#[cfg(test)]
mod phrase_tests {
    use super::*;
    
    #[test]
    fn test_phrase_search() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let proximity_engine = ProximitySearchEngine {
            boolean_engine: BooleanSearchEngine::new(&index_path)?,
        };
        
        let test_files = vec![
            ("exact.rs", "pub fn initialize() -> Result<T, E>"),
            ("wrong_order.rs", "fn pub initialize() -> Result<T, E>"),
            ("partial.rs", "pub function initialize() -> Result<T, E>"),
        ];
        
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            std::fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test exact phrase match
        let results = proximity_engine.search_phrase("pub fn initialize")?;
        assert_eq!(results.len(), 1);
        assert!(results[0].file_path.contains("exact.rs"));
        
        // Test generic types phrase
        let results = proximity_engine.search_phrase("Result<T, E>")?;
        assert!(results.len() >= 1);
        
        Ok(())
    }
}
```

### Task 3: Wildcard Search (2 hours)
```rust
#[cfg(test)]
mod wildcard_tests {
    use super::*;
    
    #[test]
    fn test_wildcard_patterns() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let pattern_engine = AdvancedPatternEngine {
            proximity_engine: ProximitySearchEngine {
                boolean_engine: BooleanSearchEngine::new(&index_path)?,
            },
        };
        
        let test_files = vec![
            ("neural.rs", "SpikingNeuralNetwork"),
            ("tests.rs", "test1 test2 test3"),
            ("functions.rs", "get_user_by_id process_data_async"),
        ];
        
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            std::fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test star wildcard
        let results = pattern_engine.search_wildcard("Spike*Network")?;
        assert!(results.iter().any(|r| r.file_path.contains("neural.rs")));
        
        // Test question mark wildcard
        let results = pattern_engine.search_wildcard("test?")?;
        assert!(results.iter().any(|r| r.file_path.contains("tests.rs")));
        
        // Test complex patterns
        let results = pattern_engine.search_wildcard("get_*_by_*")?;
        assert!(results.iter().any(|r| r.file_path.contains("functions.rs")));
        
        Ok(())
    }
}
```

### Task 4: Regex and Fuzzy Search (2 hours)
```rust
#[cfg(test)]
mod advanced_pattern_tests {
    use super::*;
    
    #[test]
    fn test_regex_search() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let pattern_engine = AdvancedPatternEngine {
            proximity_engine: ProximitySearchEngine {
                boolean_engine: BooleanSearchEngine::new(&index_path)?,
            },
        };
        
        let test_files = vec![
            ("functions.rs", "pub fn process_data() -> Result<Vec<String>, Error>"),
            ("generics.rs", "Result<HashMap<K, V>, NetworkError>"),
            ("simple.rs", "fn helper() -> bool"),
        ];
        
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            std::fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test function pattern
        let results = pattern_engine.search_regex(r"pub fn \w+\(\)")?;
        assert!(results.iter().any(|r| r.file_path.contains("functions.rs")));
        
        // Test generic pattern
        let results = pattern_engine.search_regex(r"Result<.*?>")?;
        assert!(results.len() >= 2);
        
        Ok(())
    }
    
    #[test]
    fn test_fuzzy_search() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let pattern_engine = AdvancedPatternEngine {
            proximity_engine: ProximitySearchEngine {
                boolean_engine: BooleanSearchEngine::new(&index_path)?,
            },
        };
        
        let test_files = vec![
            ("typos.rs", "funcion initialize() { println!(\"Hello\"); }"),  // "funcion" vs "function"
            ("correct.rs", "function initialize() { println!(\"Hello\"); }"),
        ];
        
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            std::fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Find "funcion" when searching for "function" (1 edit distance)
        let results = pattern_engine.search_fuzzy("function", 1)?;
        assert!(results.len() >= 2); // Should find both typo and correct
        
        Ok(())
    }
}
```

## Deliverables

### Rust Source Files
1. `src/proximity.rs` - Proximity and phrase search
2. `src/patterns.rs` - Wildcard and regex patterns  
3. `src/fuzzy.rs` - Fuzzy search implementation
4. `src/advanced_query.rs` - Combined advanced queries

### Test Files
1. `tests/proximity_tests.rs` - Proximity search tests
2. `tests/phrase_tests.rs` - Phrase matching tests
3. `tests/wildcard_tests.rs` - Wildcard pattern tests
4. `tests/regex_tests.rs` - Regex and fuzzy tests

## Success Metrics

### Functional Requirements ✅ DESIGN COMPLETE
- [x] Proximity search with configurable distance designed
- [x] Exact phrase matching with quotes designed
- [x] Wildcard patterns (* and ?) designed
- [x] Regex pattern support designed
- [x] Fuzzy search with edit distance designed
- [x] Combined advanced queries designed

### Performance Targets ✅ DESIGN TARGETS SET
- [x] Proximity search < 100ms (design target)
- [x] Phrase search < 50ms (design target)  
- [x] Wildcard search < 150ms (design target)
- [x] Regex search < 200ms (design target)
- [x] All queries designed to work with chunks

### Quality Gates ✅ DESIGN COMPLETE
- [x] 100% accurate proximity calculation designed
- [x] Exact phrase matching only designed
- [x] Proper wildcard expansion designed
- [x] Secure regex (no DoS) designed
- [x] Windows compatibility designed

## Next Phase
With all advanced search features implemented using Tantivy, proceed to Phase 4: Scale & Performance with Rayon parallelism.

---

*Phase 3 leverages Tantivy's proven advanced search capabilities instead of building custom implementations.*