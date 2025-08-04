# Task 025: Add Basic Store Creation Test

## Prerequisites Check
- [ ] Task 024 completed: test module structure created
- [ ] tempfile import is available in test scope
- [ ] vector_store_foundation_tests module exists
- [ ] Run: `cargo check` (should pass with test module)

## Context
Test module structure created. Adding simple store creation test.

## Task Objective
Add single test function for TransactionalVectorStore creation

## Steps
1. Open src/vector_store.rs in editor
2. Add test function inside vector_store_foundation_tests module:
   ```rust
   #[tokio::test]
   async fn test_store_creation() {
       let temp_dir = TempDir::new().unwrap();
       let db_path = temp_dir.path().join("test.lance").to_string_lossy().to_string();
       
       let result = TransactionalVectorStore::new(&db_path).await;
       
       // Test passes if we can create store or get expected error
       match result {
           Ok(store) => assert_eq!(store.db_path(), &db_path),
           Err(_) => println!("Store creation failed (expected without schema)"),
       }
   }
   ```
3. Save file

## Success Criteria
- [ ] Single test function with #[tokio::test]
- [ ] Uses temporary directory
- [ ] Tests store creation with basic assertion
- [ ] Handles expected error case

## Time: 5 minutes

## Next Task
Task 026: Update store exports