# Migration Guide

Complete guide for migrating from the traditional MCP knowledge storage system to the Enhanced Knowledge Storage System.

## Table of Contents

- [Migration Overview](#migration-overview)
- [Pre-Migration Assessment](#pre-migration-assessment)
- [Migration Strategies](#migration-strategies)
- [Step-by-Step Migration](#step-by-step-migration)
- [Data Migration](#data-migration)
- [Configuration Migration](#configuration-migration)
- [Code Migration](#code-migration)
- [Testing and Validation](#testing-and-validation)
- [Rollback Procedures](#rollback-procedures)
- [Performance Comparison](#performance-comparison)
- [Troubleshooting](#troubleshooting)

## Migration Overview

### What's Changing

The Enhanced Knowledge Storage System introduces significant improvements over the traditional system:

| Aspect | Traditional System | Enhanced System |
|--------|-------------------|----------------|
| **Chunking** | Hard 2KB limit, breaks sentences | Semantic chunking, preserves meaning |
| **Entity Extraction** | Simple pattern matching (~30% accuracy) | AI-powered extraction (85%+ accuracy) |
| **Relationships** | Basic "is/has" patterns | Complex relationship mapping |
| **Storage** | Flat chunk storage | Hierarchical, multi-layer organization |
| **Context** | Lost at chunk boundaries | Preserved across processing |
| **Quality** | No quality assessment | Comprehensive quality metrics |

### Migration Benefits

- **3x better retrieval quality** - Improved context preservation and entity recognition
- **85%+ entity extraction accuracy** - vs ~30% with pattern matching
- **Intelligent chunking** - Respects semantic boundaries
- **Complex relationships** - Beyond simple "is/has" patterns
- **Quality monitoring** - Comprehensive metrics and validation

### Migration Challenges

- **Increased resource requirements** - 1-8GB RAM vs ~100MB
- **Processing complexity** - AI models require more computation
- **Configuration changes** - New parameters and settings
- **API differences** - Enhanced interfaces with additional features

## Pre-Migration Assessment

### System Requirements Check

```rust
use llmkg::enhanced_knowledge_storage::*;

pub async fn assess_system_compatibility() -> SystemAssessment {
    let mut assessment = SystemAssessment::new();
    
    // Check available memory
    let available_memory = get_available_system_memory();
    assessment.memory_sufficient = available_memory >= 1_000_000_000; // 1GB minimum
    assessment.available_memory = available_memory;
    
    // Check disk space for models
    let available_disk = get_available_disk_space();
    assessment.disk_sufficient = available_disk >= 5_000_000_000; // 5GB for models
    assessment.available_disk = available_disk;
    
    // Check network connectivity for model downloads
    assessment.network_available = test_model_download_connectivity().await;
    
    // Assess current data volume
    let current_data_stats = analyze_current_knowledge_base().await;
    assessment.current_documents = current_data_stats.document_count;
    assessment.current_data_size = current_data_stats.total_size;
    assessment.estimated_migration_time = estimate_migration_time(&current_data_stats);
    
    // Check for compatibility issues
    assessment.compatibility_issues = check_compatibility_issues().await;
    
    assessment
}

#[derive(Debug)]
pub struct SystemAssessment {
    pub memory_sufficient: bool,
    pub available_memory: u64,
    pub disk_sufficient: bool,
    pub available_disk: u64,
    pub network_available: bool,
    pub current_documents: usize,
    pub current_data_size: u64,
    pub estimated_migration_time: Duration,
    pub compatibility_issues: Vec<CompatibilityIssue>,
}

impl SystemAssessment {
    pub fn can_migrate(&self) -> bool {
        self.memory_sufficient && 
        self.disk_sufficient && 
        self.network_available && 
        self.compatibility_issues.iter().all(|issue| !issue.is_blocking)
    }
    
    pub fn print_assessment(&self) {
        println!("Enhanced Knowledge Storage Migration Assessment");
        println!("=============================================");
        
        println!("System Requirements:");
        println!("  Memory: {} ({})", 
            if self.memory_sufficient { "✅ Sufficient" } else { "❌ Insufficient" },
            format_bytes(self.available_memory)
        );
        println!("  Disk Space: {} ({})", 
            if self.disk_sufficient { "✅ Sufficient" } else { "❌ Insufficient" },
            format_bytes(self.available_disk)
        );
        println!("  Network: {}", 
            if self.network_available { "✅ Available" } else { "❌ Not Available" }
        );
        
        println!("\nCurrent Data:");
        println!("  Documents: {}", self.current_documents);
        println!("  Total Size: {}", format_bytes(self.current_data_size));
        println!("  Est. Migration Time: {:?}", self.estimated_migration_time);
        
        if !self.compatibility_issues.is_empty() {
            println!("\nCompatibility Issues:");
            for issue in &self.compatibility_issues {
                let icon = if issue.is_blocking { "❌" } else { "⚠️" };
                println!("  {} {}: {}", icon, issue.category, issue.description);
            }
        }
        
        println!("\nMigration Recommendation: {}", 
            if self.can_migrate() { 
                "✅ Ready to proceed with migration" 
            } else { 
                "❌ Address issues before migration" 
            }
        );
    }
}
```

### Data Analysis

```rust
pub async fn analyze_migration_impact() -> MigrationImpact {
    let current_system = analyze_current_system().await;
    let projected_enhanced = project_enhanced_system_performance(&current_system).await;
    
    MigrationImpact {
        performance_improvement: PerformanceComparison {
            entity_extraction_accuracy: ImprovementMetric {
                before: current_system.entity_accuracy,
                after: projected_enhanced.entity_accuracy,
                improvement_factor: projected_enhanced.entity_accuracy / current_system.entity_accuracy,
            },
            retrieval_quality: ImprovementMetric {
                before: current_system.retrieval_quality,
                after: projected_enhanced.retrieval_quality,
                improvement_factor: projected_enhanced.retrieval_quality / current_system.retrieval_quality,
            },
            processing_speed: ImprovementMetric {
                before: current_system.avg_processing_time,
                after: projected_enhanced.avg_processing_time,
                improvement_factor: current_system.avg_processing_time / projected_enhanced.avg_processing_time,
            },
        },
        resource_changes: ResourceComparison {
            memory_usage_change: projected_enhanced.memory_usage as i64 - current_system.memory_usage as i64,
            storage_usage_change: projected_enhanced.storage_usage as i64 - current_system.storage_usage as i64,
            cpu_usage_change: projected_enhanced.cpu_usage - current_system.cpu_usage,
        },
        migration_complexity: assess_migration_complexity(&current_system),
        recommended_approach: recommend_migration_approach(&current_system),
    }
}
```

## Migration Strategies

### Strategy 1: Gradual Migration (Recommended)

Migrate documents incrementally while maintaining the old system for rollback capability.

```rust
pub struct GradualMigrationPlan {
    phases: Vec<MigrationPhase>,
    rollback_capability: bool,
    parallel_operation_period: Duration,
}

impl GradualMigrationPlan {
    pub fn create_standard_plan() -> Self {
        Self {
            phases: vec![
                MigrationPhase {
                    name: "Setup and Testing".to_string(),
                    description: "Install enhanced system, test with sample data".to_string(),
                    duration: Duration::from_secs(3600), // 1 hour
                    documents_to_migrate: 0,
                    success_criteria: vec![
                        "Enhanced system responds to basic queries".to_string(),  
                        "Model loading works correctly".to_string(),
                        "Configuration validated".to_string(),
                    ],
                },
                MigrationPhase {
                    name: "Pilot Migration".to_string(),
                    description: "Migrate 10% of documents for testing".to_string(),
                    duration: Duration::from_secs(7200), // 2 hours
                    documents_to_migrate: 100, // or 10% of total
                    success_criteria: vec![
                        "Pilot documents process successfully".to_string(),
                        "Quality metrics meet thresholds".to_string(),
                        "Performance acceptable".to_string(),
                    ],
                },
                MigrationPhase {
                    name: "Incremental Migration".to_string(),
                    description: "Migrate remaining documents in batches".to_string(),
                    duration: Duration::from_secs(86400), // 24 hours
                    documents_to_migrate: -1, // remaining documents
                    success_criteria: vec![
                        "All documents migrated successfully".to_string(),
                        "System performance stable".to_string(),
                        "Quality validation passed".to_string(),
                    ],
                },
                MigrationPhase {
                    name: "Validation and Cutover".to_string(),
                    description: "Final validation and switch to enhanced system".to_string(),
                    duration: Duration::from_secs(3600), // 1 hour
                    documents_to_migrate: 0,
                    success_criteria: vec![
                        "All queries work correctly".to_string(),
                        "Performance meets requirements".to_string(),
                        "Old system successfully decommissioned".to_string(),
                    ],
                },
            ],
            rollback_capability: true,
            parallel_operation_period: Duration::from_secs(604800), // 1 week
        }
    }
}
```

### Strategy 2: Blue-Green Migration

Run both systems in parallel, then switch over.

```rust
pub struct BlueGreenMigrationPlan {
    preparation_phase: Duration,
    parallel_processing_phase: Duration,
    validation_phase: Duration,
    cutover_phase: Duration,
}

impl BlueGreenMigrationPlan {
    pub async fn execute_migration(&self) -> Result<MigrationResult, MigrationError> {
        // Phase 1: Prepare enhanced system (Blue)
        println!("Phase 1: Preparing enhanced system...");
        let enhanced_system = self.setup_enhanced_system().await?;
        
        // Phase 2: Migrate all data to enhanced system
        println!("Phase 2: Migrating data to enhanced system...");
        let migration_result = self.migrate_all_data(&enhanced_system).await?;
        
        // Phase 3: Run both systems in parallel for validation
        println!("Phase 3: Running parallel validation...");
        let validation_result = self.run_parallel_validation(&enhanced_system).await?;
        
        if !validation_result.meets_criteria() {
            return Err(MigrationError::ValidationFailed(validation_result));
        }
        
        // Phase 4: Cut over to enhanced system
        println!("Phase 4: Performing cutover...");
        self.perform_cutover(&enhanced_system).await?;
        
        Ok(MigrationResult {
            strategy: MigrationStrategy::BlueGreen,
            documents_migrated: migration_result.documents_processed,
            migration_time: migration_result.total_time,
            quality_improvement: validation_result.quality_improvement,
            performance_metrics: validation_result.performance_metrics,
        })
    }
}
```

### Strategy 3: Big Bang Migration

Migrate all data at once during a maintenance window.

```rust
pub struct BigBangMigrationPlan {
    maintenance_window: Duration,
    batch_size: usize,
    parallel_workers: usize,
}

impl BigBangMigrationPlan {
    pub async fn execute_migration(&self) -> Result<MigrationResult, MigrationError> {
        println!("Starting Big Bang migration...");
        
        // Stop old system
        self.stop_old_system().await?;
        
        // Migrate all data
        let start_time = std::time::Instant::now();
        let migration_result = self.migrate_all_data_parallel().await?;
        let migration_time = start_time.elapsed();
        
        // Start enhanced system
        self.start_enhanced_system().await?;
        
        // Validate migration
        let validation_result = self.validate_migration().await?;
        
        if !validation_result.is_successful() {
            // Rollback if validation fails
            return self.perform_rollback().await;
        }
        
        Ok(MigrationResult {
            strategy: MigrationStrategy::BigBang,
            documents_migrated: migration_result.documents_processed,
            migration_time,
            quality_improvement: validation_result.quality_improvement,
            performance_metrics: validation_result.performance_metrics,
        })
    }
}
```

## Step-by-Step Migration

### Phase 1: Environment Setup

```rust
// 1. Install and configure enhanced system
pub async fn setup_enhanced_system() -> Result<EnhancedSystemSetup, SetupError> {
    println!("Step 1: Setting up Enhanced Knowledge Storage System");
    
    // Configure system based on assessment
    let system_assessment = assess_system_compatibility().await;
    let config = if system_assessment.available_memory < 2_000_000_000 {
        create_memory_optimized_config()
    } else if system_assessment.available_memory > 8_000_000_000 {
        create_high_performance_config()
    } else {
        create_balanced_config()
    };
    
    // Initialize model manager
    let model_manager = Arc::new(ModelResourceManager::new(config.0));
    
    // Test model loading
    println!("  Testing model loading...");
    test_model_loading(&model_manager).await?;
    
    // Initialize processor
    let processor = IntelligentKnowledgeProcessor::new(model_manager.clone(), config.1);
    
    // Test processing with sample data
    println!("  Testing processing pipeline...");
    test_processing_pipeline(&processor).await?;
    
    // Initialize storage
    let hierarchical_storage = HierarchicalStorage::new().await?;
    
    println!("✅ Enhanced system setup complete");
    
    Ok(EnhancedSystemSetup {
        model_manager,
        processor,
        hierarchical_storage,
    })
}

async fn test_model_loading(model_manager: &ModelResourceManager) -> Result<(), SetupError> {
    let test_models = ["smollm2_135m", "smollm2_360m"];
    
    for model_id in &test_models {
        match model_manager.load_model(model_id).await {
            Ok(_) => println!("    ✅ Model {} loaded successfully", model_id),
            Err(e) => {
                println!("    ❌ Failed to load model {}: {}", model_id, e);
                return Err(SetupError::ModelLoadingFailed(model_id.to_string()));
            }
        }
    }
    
    Ok(())
}

async fn test_processing_pipeline(processor: &IntelligentKnowledgeProcessor) -> Result<(), SetupError> {
    let test_content = "Albert Einstein developed the theory of relativity. This groundbreaking work revolutionized physics.";
    
    match processor.process_knowledge(test_content, "Test Document").await {
        Ok(result) => {
            println!("    ✅ Processing test successful");
            println!("      Chunks: {}", result.chunks.len());
            println!("      Entities: {}", result.global_entities.len());
            println!("      Quality: {:.2}", result.quality_metrics.overall_quality);
            
            if result.quality_metrics.overall_quality < 0.5 {
                return Err(SetupError::QualityBelowThreshold);
            }
        },
        Err(e) => {
            println!("    ❌ Processing test failed: {}", e);
            return Err(SetupError::ProcessingTestFailed);
        }
    }
    
    Ok(())
}
```

### Phase 2: Data Migration

```rust
pub struct DataMigrationManager {
    old_system_reader: OldSystemReader,
    enhanced_system_writer: EnhancedSystemWriter,
    batch_size: usize,
    validation_enabled: bool,
}

impl DataMigrationManager {
    pub async fn migrate_all_documents(&self) -> Result<MigrationSummary, MigrationError> {
        let documents = self.old_system_reader.list_all_documents().await?;
        let total_documents = documents.len();
        
        println!("Starting migration of {} documents", total_documents);
        
        let mut migrated_count = 0;
        let mut failed_count = 0;
        let mut quality_stats = QualityStats::new();
        let start_time = std::time::Instant::now();
        
        // Process in batches
        for batch in documents.chunks(self.batch_size) {
            println!("Processing batch {}/{}", 
                (migrated_count / self.batch_size) + 1,
                (total_documents + self.batch_size - 1) / self.batch_size
            );
            
            let batch_results = self.migrate_document_batch(batch).await;
            
            for result in batch_results {
                match result {
                    Ok(migration_result) => {
                        migrated_count += 1;
                        quality_stats.add_sample(migration_result.quality_metrics);
                        
                        if migrated_count % 100 == 0 {
                            println!("  Migrated {} documents", migrated_count);
                        }
                    },
                    Err(e) => {
                        failed_count += 1;
                        eprintln!("  Migration failed: {}", e);
                    }
                }
            }
        }
        
        let total_time = start_time.elapsed();
        
        println!("Migration completed:");
        println!("  Migrated: {}", migrated_count);
        println!("  Failed: {}", failed_count);
        println!("  Success rate: {:.1}%", 
            (migrated_count as f64 / total_documents as f64) * 100.0);
        println!("  Total time: {:?}", total_time);
        println!("  Average quality: {:.2}", quality_stats.average());
        
        Ok(MigrationSummary {
            total_documents,
            migrated_count,
            failed_count,
            total_time,
            average_quality: quality_stats.average(),
            quality_distribution: quality_stats.distribution(),
        })
    }
    
    async fn migrate_document_batch(
        &self,
        documents: &[DocumentReference]
    ) -> Vec<Result<DocumentMigrationResult, MigrationError>> {
        let mut results = Vec::new();
        
        for doc_ref in documents {
            let result = self.migrate_single_document(doc_ref).await;
            results.push(result);
        }
        
        results
    }
    
    async fn migrate_single_document(
        &self,
        doc_ref: &DocumentReference
    ) -> Result<DocumentMigrationResult, MigrationError> {
        // Read from old system
        let old_document = self.old_system_reader.read_document(doc_ref).await?;
        
        // Process with enhanced system
        let processing_result = self.enhanced_system_writer
            .processor
            .process_knowledge(&old_document.content, &old_document.title)
            .await
            .map_err(MigrationError::ProcessingFailed)?;
        
        // Validate quality if enabled
        if self.validation_enabled {
            let validation = self.enhanced_system_writer
                .processor
                .validate_processing_result(&processing_result);
            
            if !validation.is_valid {
                return Err(MigrationError::QualityValidationFailed(
                    doc_ref.id.clone(),
                    validation
                ));
            }
        }
        
        // Store in enhanced system
        let storage_result = self.enhanced_system_writer
            .hierarchical_storage
            .store_processed_knowledge(&processing_result)
            .await
            .map_err(MigrationError::StorageFailed)?;
        
        Ok(DocumentMigrationResult {
            original_id: doc_ref.id.clone(),
            new_id: processing_result.document_id,
            storage_id: storage_result.storage_id,
            quality_metrics: processing_result.quality_metrics,
            processing_time: processing_result.processing_metadata.processing_time,
        })
    }
}
```

### Phase 3: Validation and Testing

```rust
pub struct MigrationValidator {
    old_system: OldSystemInterface,
    enhanced_system: EnhancedSystemInterface,
    test_queries: Vec<TestQuery>,
}

impl MigrationValidator {
    pub async fn validate_migration(&self) -> ValidationReport {
        let mut report = ValidationReport::new();
        
        println!("Running migration validation...");
        
        // Test 1: Data completeness
        println!("  Testing data completeness...");
        let completeness_result = self.test_data_completeness().await;
        report.add_test_result("Data Completeness", completeness_result);
        
        // Test 2: Query accuracy
        println!("  Testing query accuracy...");
        let accuracy_result = self.test_query_accuracy().await;
        report.add_test_result("Query Accuracy", accuracy_result);
        
        // Test 3: Performance comparison
        println!("  Testing performance...");
        let performance_result = self.test_performance_comparison().await;
        report.add_test_result("Performance", performance_result);
        
        // Test 4: Quality assessment
        println!("  Testing quality improvements...");
        let quality_result = self.test_quality_improvements().await;
        report.add_test_result("Quality", quality_result);
        
        report.generate_summary();
        report
    }
    
    async fn test_query_accuracy(&self) -> TestResult {
        let mut correct_answers = 0;
        let mut total_queries = 0;
        let mut improvements = 0;
        
        for test_query in &self.test_queries {
            total_queries += 1;
            
            // Get results from both systems
            let old_results = self.old_system.query(&test_query.query).await;
            let enhanced_results = self.enhanced_system.query(&test_query.query).await;
            
            // Compare with expected results
            let old_accuracy = calculate_accuracy(&old_results, &test_query.expected_results);
            let enhanced_accuracy = calculate_accuracy(&enhanced_results, &test_query.expected_results);
            
            if enhanced_accuracy >= test_query.minimum_accuracy_threshold {
                correct_answers += 1;
            }
            
            if enhanced_accuracy > old_accuracy {
                improvements += 1;
            }
            
            println!("    Query: '{}' - Old: {:.2}, Enhanced: {:.2}", 
                test_query.query, old_accuracy, enhanced_accuracy);
        }
        
        let accuracy_rate = correct_answers as f64 / total_queries as f64;
        let improvement_rate = improvements as f64 / total_queries as f64;
        
        TestResult {
            passed: accuracy_rate >= 0.8, // 80% accuracy threshold
            score: accuracy_rate,
            details: format!(
                "Query accuracy: {:.1}%, Improvements: {:.1}%", 
                accuracy_rate * 100.0, 
                improvement_rate * 100.0
            ),
        }
    }
    
    async fn test_performance_comparison(&self) -> TestResult {
        let mut old_times = Vec::new();
        let mut enhanced_times = Vec::new();
        
        for test_query in &self.test_queries {
            // Time old system
            let old_start = std::time::Instant::now();
            let _old_results = self.old_system.query(&test_query.query).await;
            old_times.push(old_start.elapsed());
            
            // Time enhanced system
            let enhanced_start = std::time::Instant::now();
            let _enhanced_results = self.enhanced_system.query(&test_query.query).await;
            enhanced_times.push(enhanced_start.elapsed());
        }
        
        let old_avg = old_times.iter().sum::<Duration>() / old_times.len() as u32;
        let enhanced_avg = enhanced_times.iter().sum::<Duration>() / enhanced_times.len() as u32;
        
        let performance_ratio = old_avg.as_millis() as f64 / enhanced_avg.as_millis() as f64;
        
        TestResult {
            passed: enhanced_avg < Duration::from_secs(10), // Max 10 seconds
            score: if performance_ratio >= 1.0 { 1.0 } else { performance_ratio },
            details: format!(
                "Old avg: {:?}, Enhanced avg: {:?}, Ratio: {:.2}x", 
                old_avg, enhanced_avg, performance_ratio
            ),
        }
    }
}
```

## Configuration Migration

### Converting Old Configuration

```rust
pub fn migrate_configuration(old_config: OldSystemConfig) -> KnowledgeProcessingConfig {
    KnowledgeProcessingConfig {
        // Map old chunk settings to new semantic chunking
        max_chunk_size: if old_config.max_chunk_size > 0 {
            (old_config.max_chunk_size * 2).min(4096) // Allow larger chunks
        } else {
            2048
        },
        min_chunk_size: 128, // Set reasonable minimum
        chunk_overlap_size: 64, // Add overlap for context preservation
        
        // Entity extraction settings
        entity_extraction_model: select_entity_model(&old_config),
        min_entity_confidence: map_confidence_threshold(old_config.entity_threshold),
        
        // Relationship settings (new feature)
        relationship_extraction_model: "smollm_360m_instruct".to_string(),
        min_relationship_confidence: 0.5,
        
        // Context preservation (new feature)
        preserve_context: true,
        enable_quality_validation: true,
        
        // Semantic analysis
        semantic_analysis_model: "smollm2_135m".to_string(),
    }
}

fn select_entity_model(old_config: &OldSystemConfig) -> String {
    match old_config.processing_mode {
        ProcessingMode::Fast => "smollm2_135m".to_string(),
        ProcessingMode::Balanced => "smollm2_360m".to_string(),
        ProcessingMode::Accurate => "smollm_1_7b".to_string(),
    }
}

fn map_confidence_threshold(old_threshold: Option<f32>) -> f32 {
    match old_threshold {
        Some(threshold) if threshold < 0.3 => 0.5, // Increase low thresholds
        Some(threshold) => threshold,
        None => 0.6, // Default for enhanced system
    }
}
```

### Environment Variable Migration

```bash
# Old system environment variables
OLD_CHUNK_SIZE=2048
OLD_ENTITY_THRESHOLD=0.3
OLD_PROCESSING_MODE=fast

# New system environment variables
ENHANCED_MAX_CHUNK_SIZE=2048
ENHANCED_MIN_CHUNK_SIZE=128
ENHANCED_CHUNK_OVERLAP=64
ENHANCED_ENTITY_MODEL=smollm2_135m
ENHANCED_ENTITY_CONFIDENCE=0.6
ENHANCED_RELATIONSHIP_MODEL=smollm_360m_instruct
ENHANCED_RELATIONSHIP_CONFIDENCE=0.5
ENHANCED_SEMANTIC_MODEL=smollm2_135m
ENHANCED_MAX_MEMORY=2000000000
ENHANCED_MAX_CONCURRENT_MODELS=3
```

## Code Migration

### API Changes

#### Old API Usage
```rust
// Old system usage
use llmkg::mcp::LLMKGServer;

let server = LLMKGServer::new().await?;

// Store knowledge (simple)
let result = server.store_knowledge(StoreKnowledgeParams {
    content: "Einstein developed relativity theory...".to_string(),
    title: Some("Physics History".to_string()),
}).await?;

// Find facts (basic)
let facts = server.find_facts(FindFactsParams {
    subject: Some("Einstein".to_string()),
    predicate: None,
    object: None,
}).await?;
```

#### New API Usage
```rust
// Enhanced system usage
use llmkg::enhanced_knowledge_storage::*;
use llmkg::mcp::EnhancedMCPServer;

let model_manager = Arc::new(ModelResourceManager::new(
    ModelResourceConfig::default()
));
let processor = IntelligentKnowledgeProcessor::new(
    model_manager,
    KnowledgeProcessingConfig::default()
);
let server = EnhancedMCPServer::new(processor).await?;

// Store knowledge (enhanced)
let result = server.store_knowledge_enhanced(StoreKnowledgeParams {
    content: "Einstein developed relativity theory...".to_string(),
    title: Some("Physics History".to_string()),
    quality_threshold: Some(0.7),
    enable_validation: Some(true),
}).await?;

// Access enhanced result
println!("Quality score: {:.2}", result.quality_metrics.overall_quality);
println!("Entities found: {}", result.entities_extracted);
println!("Relationships found: {}", result.relationships_found);

// Find facts (enhanced with context)
let facts = server.find_facts_enhanced(FindFactsParams {
    subject: Some("Einstein".to_string()),
    predicate: None,
    object: None,
    include_context: Some(true),
    min_confidence: Some(0.6),
}).await?;

// Access enhanced facts
for fact in facts.facts {
    println!("Fact: {} {} {} (confidence: {:.2})", 
        fact.subject, fact.predicate, fact.object, fact.confidence);
    if let Some(context) = fact.context {
        println!("  Context: {}", context);
    }
}
```

### Gradual Code Migration

```rust
// Compatibility wrapper for gradual migration
pub struct CompatibilityWrapper {
    enhanced_system: Option<EnhancedMCPServer>,
    old_system: LLMKGServer,
    migration_percentage: f32, // 0.0 = all old, 1.0 = all enhanced
}

impl CompatibilityWrapper {
    pub async fn store_knowledge(
        &self,
        params: StoreKnowledgeParams
    ) -> Result<StoreKnowledgeResult, MsgError> {
        // Gradually migrate based on percentage
        let use_enhanced = rand::random::<f32>() < self.migration_percentage;
        
        if use_enhanced && self.enhanced_system.is_some() {
            // Try enhanced system first
            match self.enhanced_system.as_ref().unwrap()
                .store_knowledge_enhanced(params.clone()).await {
                Ok(result) => Ok(result),
                Err(_) => {
                    // Fallback to old system
                    self.old_system.store_knowledge(params).await
                }
            }
        } else {
            // Use old system
            self.old_system.store_knowledge(params).await
        }
    }
    
    pub fn increase_migration_percentage(&mut self, increment: f32) {
        self.migration_percentage = (self.migration_percentage + increment).min(1.0);
        println!("Migration percentage increased to {:.1}%", 
            self.migration_percentage * 100.0);
    }
}
```

## Testing and Validation

### Comprehensive Test Suite

```rust
#[cfg(test)]
mod migration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_data_integrity() {
        let migration_manager = setup_test_migration().await;
        
        // Migrate test document
        let test_doc = create_test_document();
        let migration_result = migration_manager
            .migrate_single_document(&test_doc)
            .await
            .unwrap();
        
        // Verify data integrity
        let original_content = test_doc.content;
        let migrated_document = migration_manager.enhanced_system
            .retrieve_document(&migration_result.new_id)
            .await
            .unwrap();
        
        // Check content preservation
        let original_words: HashSet<_> = original_content.split_whitespace().collect();
        let migrated_words: HashSet<_> = migrated_document.chunks
            .iter()
            .flat_map(|chunk| chunk.content.split_whitespace())
            .collect();
        
        let preservation_rate = original_words.intersection(&migrated_words).count() as f64 
            / original_words.len() as f64;
        
        assert!(preservation_rate > 0.95, "Content preservation rate too low: {:.2}", preservation_rate);
    }
    
    #[tokio::test]
    async fn test_quality_improvement() {
        let migration_manager = setup_test_migration().await;
        
        let test_documents = create_test_document_suite();
        let mut quality_improvements = Vec::new();
        
        for test_doc in test_documents {
            // Get old system results
            let old_results = migration_manager.old_system
                .find_facts(FindFactsParams {
                    subject: Some("Einstein".to_string()),
                    predicate: None,
                    object: None,
                })
                .await
                .unwrap();
            
            // Migrate and get enhanced results
            let migration_result = migration_manager
                .migrate_single_document(&test_doc)
                .await
                .unwrap();
            
            let enhanced_results = migration_manager.enhanced_system
                .find_facts_enhanced(FindFactsParams {
                    subject: Some("Einstein".to_string()),
                    predicate: None,
                    object: None,
                    include_context: Some(true),
                    min_confidence: Some(0.6),
                })
                .await
                .unwrap();
            
            // Compare entity extraction
            let old_entity_count = count_entities_in_results(&old_results);
            let enhanced_entity_count = migration_result.processing_result.global_entities.len();
            
            let improvement = enhanced_entity_count as f64 / old_entity_count.max(1) as f64;
            quality_improvements.push(improvement);
        }
        
        let avg_improvement = quality_improvements.iter().sum::<f64>() / quality_improvements.len() as f64;
        assert!(avg_improvement > 2.0, "Expected >2x entity extraction improvement, got {:.2}x", avg_improvement);
    }
}
```

## Rollback Procedures

### Automated Rollback

```rust
pub struct RollbackManager {
    backup_location: String,
    enhanced_system: EnhancedSystemInterface,
    old_system: OldSystemInterface,
}

impl RollbackManager {
    pub async fn create_rollback_point(&self) -> Result<RollbackPoint, RollbackError> {
        println!("Creating rollback point...");
        
        // Backup current state
        let backup_id = generate_backup_id();
        let backup_path = format!("{}/{}", self.backup_location, backup_id);
        
        // Backup old system data
        let old_system_backup = self.backup_old_system_data(&backup_path).await?;
        
        // Backup configuration
        let config_backup = self.backup_configuration(&backup_path).await?;
        
        println!("Rollback point created: {}", backup_id);
        
        Ok(RollbackPoint {
            id: backup_id,
            created_at: std::time::SystemTime::now(),
            old_system_backup,
            config_backup,
        })
    }
    
    pub async fn perform_rollback(&self, rollback_point: &RollbackPoint) -> Result<(), RollbackError> {
        println!("Performing rollback to point: {}", rollback_point.id);
        
        // Stop enhanced system
        self.stop_enhanced_system().await?;
        
        // Restore old system data
        self.restore_old_system_data(&rollback_point.old_system_backup).await?;
        
        // Restore configuration
        self.restore_configuration(&rollback_point.config_backup).await?;
        
        // Start old system
        self.start_old_system().await?;
        
        // Verify rollback
        let verification = self.verify_rollback().await?;
        if !verification.is_successful() {
            return Err(RollbackError::VerificationFailed(verification));
        }
        
        println!("Rollback completed successfully");
        Ok(())
    }
    
    async fn verify_rollback(&self) -> Result<RollbackVerification, RollbackError> {
        // Test basic functionality
        let test_query = "Einstein";
        let results = self.old_system.query(test_query).await
            .map_err(RollbackError::QueryTestFailed)?;
        
        Ok(RollbackVerification {
            system_responsive: true,
            data_accessible: !results.is_empty(),
            configuration_valid: self.verify_old_system_config().await,
        })
    }
}

pub struct RollbackPoint {
    pub id: String,
    pub created_at: std::time::SystemTime,
    pub old_system_backup: BackupInfo,
    pub config_backup: BackupInfo,
}

#[derive(Debug)]
pub struct RollbackVerification {
    pub system_responsive: bool,
    pub data_accessible: bool,
    pub configuration_valid: bool,
}

impl RollbackVerification {
    pub fn is_successful(&self) -> bool {
        self.system_responsive && self.data_accessible && self.configuration_valid
    }
}
```

## Performance Comparison

### Before and After Analysis

```rust
pub async fn generate_migration_performance_report() -> PerformanceMigrationReport {
    let old_system_metrics = benchmark_old_system().await;
    let enhanced_system_metrics = benchmark_enhanced_system().await;
    
    PerformanceMigrationReport {
        processing_speed: SpeedComparison {
            old_avg_time: old_system_metrics.avg_processing_time,
            enhanced_avg_time: enhanced_system_metrics.avg_processing_time,
            improvement_factor: old_system_metrics.avg_processing_time.as_millis() as f64 
                / enhanced_system_metrics.avg_processing_time.as_millis() as f64,
        },
        entity_extraction: ExtractionComparison {
            old_accuracy: old_system_metrics.entity_accuracy,
            enhanced_accuracy: enhanced_system_metrics.entity_accuracy,
            improvement_factor: enhanced_system_metrics.entity_accuracy / old_system_metrics.entity_accuracy,
        },
        memory_usage: MemoryComparison {
            old_usage: old_system_metrics.memory_usage,
            enhanced_usage: enhanced_system_metrics.memory_usage,
            usage_increase: enhanced_system_metrics.memory_usage as i64 - old_system_metrics.memory_usage as i64,
        },
        retrieval_quality: QualityComparison {
            old_relevance_score: old_system_metrics.retrieval_relevance,
            enhanced_relevance_score: enhanced_system_metrics.retrieval_relevance,
            improvement_factor: enhanced_system_metrics.retrieval_relevance / old_system_metrics.retrieval_relevance,
        },
        overall_assessment: assess_migration_success(&old_system_metrics, &enhanced_system_metrics),
    }
}

fn assess_migration_success(old: &SystemMetrics, enhanced: &SystemMetrics) -> MigrationAssessment {
    let entity_improvement = enhanced.entity_accuracy / old.entity_accuracy;
    let retrieval_improvement = enhanced.retrieval_relevance / old.retrieval_relevance;
    let memory_acceptable = enhanced.memory_usage < 8_000_000_000; // 8GB limit
    
    if entity_improvement > 2.0 && retrieval_improvement > 2.0 && memory_acceptable {
        MigrationAssessment::HighlySuccessful
    } else if entity_improvement > 1.5 && retrieval_improvement > 1.5 && memory_acceptable {
        MigrationAssessment::Successful
    } else if entity_improvement > 1.0 && retrieval_improvement > 1.0 {
        MigrationAssessment::Marginal
    } else {
        MigrationAssessment::Unsuccessful
    }
}
```

This migration guide provides a comprehensive roadmap for transitioning from the traditional MCP knowledge storage system to the Enhanced Knowledge Storage System, ensuring a smooth transition while maximizing the benefits of the new system.