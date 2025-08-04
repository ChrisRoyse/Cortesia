# Micro-Task 056: Create Query Builder Foundation

## Objective
Initialize query builder module for structured query construction.

## Context
Query builder provides a fluent API for constructing complex search queries programmatically.

## Prerequisites
- Task 055 completed (fusion algorithms done)
- Search API crate configured

## Time Estimate
10 minutes

## Instructions
1. Navigate to search-api: `cd search-api`
2. Create query builder module: `mkdir -p src/query_builder`
3. Create `src/query_builder/mod.rs`:
   ```rust
   pub struct QueryBuilder {
       pub query_text: Option<String>,
       pub filters: Vec<Filter>,
       pub limit: usize,
       pub offset: usize,
   }
   
   impl QueryBuilder {
       pub fn new() -> Self {
           Self {
               query_text: None,
               filters: Vec::new(),
               limit: 10,
               offset: 0,
           }
       }
       
       pub fn with_text(mut self, text: &str) -> Self {
           self.query_text = Some(text.to_string());
           self
       }
       
       pub fn with_limit(mut self, limit: usize) -> Self {
           self.limit = limit;
           self
       }
   }
   
   #[derive(Debug, Clone)]
   pub struct Filter {
       pub field: String,
       pub operator: FilterOp,
       pub value: String,
   }
   
   #[derive(Debug, Clone)]
   pub enum FilterOp {
       Equals,
       NotEquals,
       Contains,
       GreaterThan,
       LessThan,
   }
   ```
4. Add to `src/lib.rs`:
   ```rust
   pub mod query_builder;
   ```
5. Test compilation: `cargo check`

## Expected Output
- Query builder module created
- Fluent API methods implemented
- Basic filter types defined
- Module integrated successfully

## Success Criteria
- [ ] QueryBuilder struct created with fluent API
- [ ] Filter and FilterOp types defined
- [ ] Module added to lib.rs
- [ ] `cargo check` passes without errors

## Next Task
task_057_implement_query_filters.md