# Task 093: Define Search Configuration and Mode Enums

## Prerequisites Check
- [ ] Task 092 completed: UnifiedSearchSystem struct defined
- [ ] Serde traits are imported
- [ ] Run: `cargo check` (should pass)

## Context
Define configuration struct and search mode enum for flexible search behavior.

## Task Objective
Create UnifiedSearchConfig struct and SearchMode enum with proper defaults.

## Steps
1. Add UnifiedSearchConfig struct:
   ```rust
   /// Configuration for unified search
   #[derive(Debug, Clone)]
   pub struct UnifiedSearchConfig {
       /// Default search limit
       pub default_limit: usize,
       /// Enable caching
       pub enable_caching: bool,
       /// Cache TTL in seconds
       pub cache_ttl: u64,
       /// Default search mode
       pub default_search_mode: SearchMode,
   }
   ```
2. Add SearchMode enum:
   ```rust
   /// Search mode options
   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub enum SearchMode {
       /// Text search only
       TextOnly,
       /// Vector search only
       VectorOnly,
       /// Hybrid search combining both
       Hybrid,
       /// Adaptive mode choosing based on query
       Adaptive,
   }
   ```
3. Verify compilation

## Success Criteria
- [ ] UnifiedSearchConfig struct with all required fields
- [ ] SearchMode enum with all modes defined
- [ ] Proper derive traits for serialization
- [ ] Compiles without errors

## Time: 3 minutes

## Next Task
Task 094 will define result structures and enums.

## Notes
Configuration enables runtime behavior adjustment while SearchMode provides flexibility in search strategies.