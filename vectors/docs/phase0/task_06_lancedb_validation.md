# Task 06: Test LanceDB Basic Functionality on Windows

## Context
You are continuing architecture validation (Phase 0, Task 6). Task 05 validated Tantivy text search. Now you need to validate that LanceDB (vector database) works correctly on Windows with ACID transactions, vector operations, and embedding storage.

## Objective
Implement and test basic LanceDB functionality on Windows, focusing on vector storage, retrieval, ACID transactions, and Windows file system compatibility.

## Requirements
1. Test LanceDB connection and table creation on Windows
2. Test vector insertion and retrieval operations
3. Test ACID transaction support
4. Test vector similarity search
5. Verify Windows path handling for database storage
6. Test with sample 384-dimensional embeddings (common size)

## Implementation for validation.rs (extend existing)
```rust
use lancedb::{connect, Connection, Result as LanceResult};
use arrow_array::{RecordBatch, Int32Array, FixedSizeListArray, Float32Array, StringArray};
use arrow_schema::{Schema, Field, DataType};
use std::sync::Arc;
use anyhow::Result;
use tracing::{info, debug, error};

impl TantivyValidator {
    // ... existing Tantivy methods ...
}

pub struct LanceDBValidator;

impl LanceDBValidator {
    /// Test basic LanceDB functionality on Windows
    pub async fn validate_lancedb_windows() -> Result<()> {
        info!("Starting LanceDB validation on Windows");
        
        // Test basic connection
        Self::test_connection().await?;
        
        // Test table creation and schema
        Self::test_table_operations().await?;
        
        // Test vector operations
        Self::test_vector_operations().await?;
        
        // Test ACID transactions
        Self::test_transactions().await?;
        
        info!("LanceDB validation completed successfully");
        Ok(())
    }
    
    async fn test_connection() -> Result<()> {
        debug!("Testing LanceDB connection on Windows");
        
        // Ensure directory exists
        std::fs::create_dir_all("indexes/lancedb")?;
        
        // Test connection with Windows path
        let uri = "indexes/lancedb/test.lance";
        let db = connect(uri).execute().await?;
        
        debug!("LanceDB connection successful");
        Ok(())
    }
    
    async fn test_table_operations() -> Result<()> {
        debug!("Testing LanceDB table operations");
        
        let uri = "indexes/lancedb/test.lance";
        let db = connect(uri).execute().await?;
        
        // Create schema for code embeddings
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
            Field::new("file_path", DataType::Utf8, true),
            Field::new("vector", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                384  // Common embedding dimension
            ), true),
        ]));
        
        // Create sample data
        let ids = Int32Array::from(vec![1, 2, 3]);
        let texts = StringArray::from(vec![
            Some("pub fn test() -> Result<T, E>"),
            Some("[workspace] dependencies"),
            Some("struct Data<T> { value: T }")
        ]);
        let file_paths = StringArray::from(vec![
            Some("src/main.rs"),
            Some("Cargo.toml"),
            Some("src/lib.rs")
        ]);
        
        // Create sample 384-dimensional vectors
        let vector_data: Vec<f32> = (0..384*3).map(|i| (i as f32) * 0.001).collect();
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vector_data.chunks(384).map(|chunk| Some(chunk.to_vec())),
            384
        );
        
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(texts),
                Arc::new(file_paths),
                Arc::new(vectors),
            ]
        )?;
        
        // Create table
        let table = db.create_table("code_embeddings", Box::new(batch.into_reader()))
            .execute()
            .await?;
        
        debug!("Table creation successful: {}", table.name());
        Ok(())
    }
    
    async fn test_vector_operations() -> Result<()> {
        debug!("Testing vector search operations");
        
        let uri = "indexes/lancedb/test.lance";
        let db = connect(uri).execute().await?;
        
        // Open existing table
        let table = db.open_table("code_embeddings").execute().await?;
        
        // Test vector similarity search
        let query_vector: Vec<f32> = (0..384).map(|i| (i as f32) * 0.001).collect();
        
        let results = table
            .vector_search(query_vector)?
            .limit(5)
            .execute()
            .await?;
        
        debug!("Vector search returned {} results", results.len());
        assert!(results.len() > 0, "Vector search should return results");
        
        Ok(())
    }
    
    async fn test_transactions() -> Result<()> {
        debug!("Testing ACID transactions");
        
        let uri = "indexes/lancedb/transactions.lance";
        let db = connect(uri).execute().await?;
        
        // Create simple schema for transaction test
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        
        let ids = Int32Array::from(vec![1]);
        let values = StringArray::from(vec![Some("test_value")]);
        
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids), Arc::new(values)]
        )?;
        
        // Test transaction-like behavior
        let table = db.create_table("transactions", Box::new(batch.into_reader()))
            .execute()
            .await?;
        
        // Verify data was written atomically
        let count = table.count_rows(None).await?;
        assert_eq!(count, 1, "Transaction should commit all or nothing");
        
        debug!("Transaction test passed");
        Ok(())
    }
    
    /// Test Windows-specific paths and file handling
    async fn test_windows_paths() -> Result<()> {
        debug!("Testing Windows path handling");
        
        // Test with Windows-style path
        let windows_path = "indexes\\lancedb\\windows_test.lance";
        let db = connect(windows_path).execute().await?;
        
        debug!("Windows path handling successful");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_lancedb_validation() {
        LanceDBValidator::validate_lancedb_windows().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_windows_paths() {
        LanceDBValidator::test_windows_paths().await.unwrap();
    }
}
```

## Implementation Steps
1. Add LanceDBValidator struct to validation.rs
2. Implement connection testing with Windows paths
3. Implement table creation with proper schema for code embeddings
4. Create sample 384-dimensional vectors for testing
5. Implement vector similarity search testing
6. Implement basic transaction testing
7. Add Windows-specific path testing
8. Run async tests to verify functionality

## Success Criteria
- [ ] LanceDBValidator struct implemented and compiling
- [ ] Database connection works with Windows paths
- [ ] Table creation with vector schema works
- [ ] Vector insertion and retrieval work correctly
- [ ] Vector similarity search returns results
- [ ] Transaction-like behavior is verified
- [ ] All async tests pass (`cargo test`)
- [ ] Database files are created in indexes/lancedb/

## Test Command
```bash
cargo test test_lancedb_validation
cargo test test_windows_paths
```

## Verification
After completion, check that:
1. `indexes/lancedb/` directory contains database files
2. Vector operations complete without errors
3. Both forward slash and backslash paths work on Windows
4. 384-dimensional vectors are stored and searchable

## Time Estimate
10 minutes

## Next Task
Task 07: Test Rayon parallelism functionality on Windows with thread safety validation.