// Simple test to verify federation functionality works
use crate::federation::coordinator::{FederationCoordinator, TransactionMetadata, TransactionPriority, IsolationLevel, ConsistencyMode};
use crate::federation::registry::{DatabaseRegistry, DatabaseDescriptor, DatabaseType, DatabaseStatus, DatabaseMetadata};
use crate::federation::types::{DatabaseId, DatabaseCapabilities};
use std::sync::Arc;
use std::time::SystemTime;

pub async fn test_basic_federation() -> crate::error::Result<()> {
    // Create registry
    let registry = Arc::new(DatabaseRegistry::new()?);
    
    // Register a test database
    let db_id = DatabaseId::new("test_db".to_string());
    let descriptor = DatabaseDescriptor {
        id: db_id.clone(),
        name: "Test Database".to_string(),
        description: Some("Test database for federation".to_string()),
        connection_string: "test_federation.db".to_string(),
        database_type: DatabaseType::SQLite,
        capabilities: DatabaseCapabilities::default(),
        metadata: DatabaseMetadata {
            version: "1.0.0".to_string(),
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
            owner: Some("test".to_string()),
            tags: vec!["test".to_string()],
            entity_count: Some(0),
            relationship_count: Some(0),
            storage_size_bytes: Some(0),
        },
        status: DatabaseStatus::Online,
    };
    
    registry.register(descriptor).await?;
    
    // Create coordinator
    let coordinator = FederationCoordinator::new(registry).await?;
    
    // Begin transaction
    let metadata = TransactionMetadata {
        initiator: Some("test".to_string()),
        description: Some("Basic federation test".to_string()),
        priority: TransactionPriority::Normal,
        isolation_level: IsolationLevel::ReadCommitted,
        consistency_mode: ConsistencyMode::Strong,
    };
    
    let tx_id = coordinator.begin_transaction(vec![db_id], metadata).await?;
    
    println!("✓ Federation test transaction created: {}", tx_id.as_str());
    
    // Test prepare
    let prepare_result = coordinator.prepare_transaction(&tx_id).await?;
    
    if prepare_result {
        println!("✓ Transaction prepared successfully");
        
        // Test commit
        let commit_result = coordinator.commit_transaction(&tx_id).await?;
        println!("✓ Transaction committed: success={}", commit_result.success);
        
        return Ok(());
    } else {
        println!("❌ Transaction prepare failed");
        let _abort = coordinator.abort_transaction(&tx_id).await?;
        return Err(crate::error::GraphError::TransactionError("Prepare failed".to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_federation_basic() {
        let result = test_basic_federation().await;
        assert!(result.is_ok(), "Federation test failed: {:?}", result);
    }
}