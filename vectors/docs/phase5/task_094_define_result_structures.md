# Task 094: Define Result Structures and Source Enum

## Prerequisites Check
- [ ] Task 093 completed: configuration and search mode enums defined
- [ ] HashMap is imported from std::collections
- [ ] Run: `cargo check` (should pass)

## Context
Define structures for unified search results and tracking result sources.

## Task Objective
Create UnifiedResult struct and ResultSource enum for search result handling.

## Steps
1. Add UnifiedResult struct:
   ```rust
   /// Unified search result
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct UnifiedResult {
       /// Document ID
       pub id: String,
       /// Document content
       pub content: String,
       /// Combined relevance score
       pub score: f32,
       /// Source of the result
       pub source: ResultSource,
       /// Additional metadata
       pub metadata: HashMap<String, String>,
   }
   ```
2. Add ResultSource enum:
   ```rust
   /// Source of search result
   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub enum ResultSource {
       /// From text search engine
       TextSearch,
       /// From vector search
       VectorSearch,
       /// From cache
       Cache,
       /// Fused from multiple sources
       Hybrid,
   }
   ```
3. Verify compilation

## Success Criteria
- [ ] UnifiedResult struct with all required fields
- [ ] ResultSource enum tracking all result origins
- [ ] Proper serialization traits for caching
- [ ] Compiles without errors

## Time: 3 minutes

## Next Task
Task 095 will implement basic constructors and default implementations.

## Notes
Result structures support metadata for enhanced functionality and proper source tracking.