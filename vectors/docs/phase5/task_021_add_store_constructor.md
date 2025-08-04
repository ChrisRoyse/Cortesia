# Task 021: Add TransactionalVectorStore new() Method

## Prerequisites Check
- [ ] Task 020 completed: TransactionalVectorStore struct defined
- [ ] Struct has connection and db_path fields
- [ ] LanceDB imports are available in scope
- [ ] Run: `cargo check` (should pass with struct definition)

## Context
TransactionalVectorStore struct defined. Adding async constructor that connects to LanceDB.

## Task Objective
Add async new() method that creates LanceDB connection

## Steps
1. Open src/vector_store.rs in editor
2. Add impl block after TransactionalVectorStore struct:
   ```rust
   impl TransactionalVectorStore {
       /// Create a new TransactionalVectorStore
       /// 
       /// # Arguments
       /// * `db_path` - Path to the LanceDB database directory
       /// 
       /// # Returns
       /// * `Result<Self>` - New TransactionalVectorStore or error
       pub async fn new(db_path: &str) -> Result<Self> {
           let connection = connect(db_path).execute().await?;
           
           Ok(Self {
               connection,
               db_path: db_path.to_string(),
           })
       }
   }
   ```
3. Save file

## Success Criteria
- [ ] new() method is async and returns Result<Self>
- [ ] Uses connect(db_path).execute().await? pattern
- [ ] Stores connection and db_path in struct
- [ ] Has proper documentation comments
- [ ] Method is public

## Time: 6 minutes

## Next Task
Task 022: Add connection accessor methods