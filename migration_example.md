# Migration Tools Usage Example

This example demonstrates how to use the LLMKG migration tools to upgrade existing data from v1 to v2 format.

## Basic Usage

```rust
use llmkg::{MigrationTool, MigrationConfig, ValidationLevel, KnowledgeEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a knowledge engine with existing data
    let mut engine = KnowledgeEngine::new(128, 10000)?;
    
    // Add some sample legacy data
    engine.add_triple("Alice", "knows", "Bob", 0.9)?;
    engine.add_triple("Bob", "works_at", "Company", 0.8)?;
    engine.add_knowledge_chunk("Introduction", "Alice is a software engineer who knows Bob.", None, None)?;
    
    // Create migration tool with default configuration
    let migration_tool = MigrationTool::new();
    
    // Run migration from v1 to v2
    println!("Starting migration...");
    let report = migration_tool.migrate_v1_to_v2(&mut engine).await?;
    
    // Display migration results
    println!("Migration completed!");
    println!("Duration: {:.2} seconds", report.duration_seconds);
    println!("Nodes processed: {}", report.total_nodes_processed);
    println!("Successful migrations: {}", report.successful_migrations);
    println!("Failed migrations: {}", report.failed_migrations);
    println!("Entities reprocessed: {}", report.entities_reprocessed);
    println!("Relationships updated: {}", report.relationships_updated);
    
    if let Some(validation_report) = &report.validation_results {
        println!("\nValidation Results:");
        println!("Data consistency score: {:.1}%", validation_report.data_consistency_score);
        println!("Structural integrity score: {:.1}%", validation_report.structural_integrity_score);
        println!("Checks passed: {}/{}", validation_report.passed_checks, validation_report.total_checks);
    }
    
    Ok(())
}
```

## Advanced Configuration

```rust
use llmkg::{MigrationTool, MigrationConfig, ValidationLevel};

async fn advanced_migration_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create custom migration configuration
    let config = MigrationConfig {
        batch_size: 500,                           // Process 500 nodes at a time
        max_memory_usage: 1024 * 1024 * 256,     // 256MB memory limit
        create_backup: true,                       // Create backup before migration
        validation_level: ValidationLevel::Comprehensive, // Full validation
        preserve_metadata: true,                   // Keep original metadata
        max_retries: 5,                           // Allow 5 retry attempts
    };
    
    let migration_tool = MigrationTool::with_config(config);
    let mut engine = KnowledgeEngine::new(128, 10000)?;
    
    // Monitor progress during migration
    let progress_handle = tokio::spawn({
        let migration_tool = migration_tool.clone();
        async move {
            loop {
                let progress = migration_tool.get_progress();
                println!("Progress: {:.1}% ({}/{})", 
                    progress.progress_percentage(),
                    progress.processed_nodes,
                    progress.total_nodes
                );
                
                if progress.processed_nodes >= progress.total_nodes {
                    break;
                }
                
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        }
    });
    
    // Run migration
    let report = migration_tool.migrate_v1_to_v2(&mut engine).await?;
    progress_handle.abort();
    
    println!("Migration completed with {} errors", report.errors.len());
    
    Ok(())
}
```

## Error Handling and Rollback

```rust
async fn migration_with_rollback() -> Result<(), Box<dyn std::error::Error>> {
    let migration_tool = MigrationTool::new();
    let mut engine = KnowledgeEngine::new(128, 10000)?;
    
    // Attempt migration
    match migration_tool.migrate_v1_to_v2(&mut engine).await {
        Ok(report) => {
            if report.failed_migrations > 0 {
                println!("Migration had {} failures, considering rollback", report.failed_migrations);
                
                // Check if rollback is needed based on failure rate
                let failure_rate = report.failed_migrations as f64 / report.total_nodes_processed as f64;
                if failure_rate > 0.1 { // More than 10% failure rate
                    println!("Rolling back migration due to high failure rate");
                    migration_tool.rollback_migration(&mut engine).await?;
                    return Err("Migration failed and was rolled back".into());
                }
            }
            
            println!("Migration successful!");
        }
        Err(e) => {
            println!("Migration failed: {}", e);
            
            // Attempt rollback
            if let Err(rollback_err) = migration_tool.rollback_migration(&mut engine).await {
                println!("Rollback also failed: {}", rollback_err);
                return Err("Both migration and rollback failed".into());
            }
            
            println!("Successfully rolled back to previous state");
            return Err(e.into());
        }
    }
    
    Ok(())
}
```

## Validation Only

```rust
async fn validate_existing_data() -> Result<(), Box<dyn std::error::Error>> {
    let migration_tool = MigrationTool::new();
    let engine = KnowledgeEngine::new(128, 10000)?;
    
    // Run validation without migration
    let validation_report = migration_tool.validate_migration(&engine).await?;
    
    println!("Validation Results:");
    println!("Total checks: {}", validation_report.total_checks);
    println!("Passed: {}", validation_report.passed_checks);
    println!("Failed: {}", validation_report.failed_checks);
    
    for item in &validation_report.detailed_results {
        match item.status {
            llmkg::ValidationStatus::Failed => {
                println!("❌ {}: {}", item.check_name, item.message);
            }
            llmkg::ValidationStatus::Warning => {
                println!("⚠️  {}: {}", item.check_name, item.message);
            }
            llmkg::ValidationStatus::Passed => {
                println!("✅ {}: {}", item.check_name, item.message);
            }
            _ => {}
        }
    }
    
    Ok(())
}
```

## Key Features

### 1. **Backup and Rollback**
- Automatic backup creation before migration
- Full rollback capability if migration fails
- Checksum verification for backup integrity

### 2. **Progress Monitoring**
- Real-time progress tracking
- Memory usage monitoring
- Estimated completion time
- Error reporting during migration

### 3. **Batch Processing**
- Memory-efficient processing of large datasets
- Configurable batch sizes
- Automatic memory management

### 4. **Comprehensive Validation**
- Multiple validation levels (Basic, Standard, Comprehensive)
- Data consistency checks
- Structural integrity verification
- Performance impact assessment

### 5. **Enhanced Entity Processing**
- Re-extraction of entities using advanced NLP
- Relationship type normalization
- Index rebuilding for optimal performance

### 6. **Error Recovery**
- Retry mechanisms for transient failures
- Detailed error reporting with context
- Graceful degradation on partial failures

## Migration Process

1. **Backup Creation**: Original data is backed up for rollback capability
2. **Batch Processing**: Data is processed in configurable batches to manage memory
3. **Entity Re-extraction**: Text chunks are re-processed with enhanced entity extraction
4. **Relationship Updates**: Predicates are normalized and relationship types updated
5. **Index Rebuilding**: All indices are rebuilt for optimal performance
6. **Validation**: Comprehensive validation ensures data integrity
7. **Cleanup**: Temporary resources are cleaned up

This migration system ensures safe, reliable upgrades of LLMKG installations while preserving data integrity and providing full rollback capabilities.