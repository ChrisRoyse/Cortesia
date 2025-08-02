# Usage Guide

Practical guide for using the Enhanced Knowledge Storage System with real-world examples and best practices.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage Patterns](#basic-usage-patterns)
- [Advanced Configuration](#advanced-configuration)
- [MCP Integration](#mcp-integration)
- [Performance Optimization](#performance-optimization)
- [Quality Management](#quality-management)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements

- **Memory**: Minimum 1GB available RAM, recommended 4GB
- **Storage**: 2-5GB for model files (downloaded automatically)
- **CPU**: Modern multi-core processor recommended
- **Network**: Internet connection for initial model downloads

### Basic Setup

```rust
use llmkg::enhanced_knowledge_storage::*;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize with default settings
    let model_manager = Arc::new(ModelResourceManager::new(
        ModelResourceConfig::default()
    ));
    
    let processor = IntelligentKnowledgeProcessor::new(
        model_manager,
        KnowledgeProcessingConfig::default()
    );
    
    println!("Enhanced Knowledge Storage System initialized successfully!");
    Ok(())
}
```

## Basic Usage Patterns

### Processing Simple Documents

```rust
async fn process_simple_document() -> Result<(), Box<dyn std::error::Error>> {
    let processor = create_processor().await?;
    
    let content = r#"
        Marie Curie was a pioneering physicist and chemist who conducted 
        groundbreaking research on radioactivity. Born in Poland in 1867, 
        she later moved to France where she completed her studies at the 
        University of Paris. She became the first woman to win a Nobel Prize 
        and remains the only person to win Nobel Prizes in two different 
        scientific fields: Physics and Chemistry.
    "#;
    
    let result = processor.process_knowledge(content, "Marie Curie Biography").await?;
    
    println!("Processing Results:");
    println!("  Document ID: {}", result.document_id);
    println!("  Chunks created: {}", result.chunks.len());
    println!("  Entities found: {}", result.global_entities.len());
    println!("  Relationships found: {}", result.global_relationships.len());
    println!("  Quality score: {:.2}", result.quality_metrics.overall_quality);
    
    // Examine extracted entities
    for entity in &result.global_entities {
        println!("  Entity: {} ({}), confidence: {:.2}", 
                 entity.name, 
                 format!("{:?}", entity.entity_type),
                 entity.confidence);
    }
    
    Ok(())
}
```

### Processing Technical Documentation

```rust
async fn process_technical_document() -> Result<(), Box<dyn std::error::Error>> {
    let processor = create_processor().await?;
    
    let content = r#"
        The HTTP/2 protocol introduces several key improvements over HTTP/1.1.
        Binary framing is the foundation of all performance improvements in HTTP/2.
        Unlike HTTP/1.1, which uses textual format, HTTP/2 splits the communication
        into smaller binary frames. This enables multiplexing, where multiple 
        requests can be sent simultaneously over a single TCP connection.
        
        Server push is another significant feature that allows the server to 
        send resources to the client before they are requested. This reduces
        latency by eliminating additional round trips. Stream prioritization
        ensures that critical resources are delivered first, improving the
        overall user experience.
    "#;
    
    let result = processor.process_knowledge(content, "HTTP/2 Protocol Overview").await?;
    
    // Analyze technical concepts
    println!("Technical Analysis:");
    for chunk in &result.chunks {
        println!("Chunk {}: {} key concepts", chunk.id, chunk.key_concepts.len());
        for concept in &chunk.key_concepts {
            println!("  - {}", concept);
        }
    }
    
    // Examine technical relationships
    for relationship in &result.global_relationships {
        println!("Relationship: {} -> {} ({})", 
                 relationship.source,
                 relationship.target,
                 format!("{:?}", relationship.predicate));
    }
    
    Ok(())
}
```

### Processing Scientific Papers

```rust
async fn process_scientific_paper() -> Result<(), Box<dyn std::error::Error>> {
    let processor = create_processor_with_high_quality_config().await?;
    
    let content = std::fs::read_to_string("scientific_paper.txt")?;
    
    let result = processor.process_knowledge(&content, "Quantum Computing Research").await?;
    
    // Validate quality for scientific content
    let validation = processor.validate_processing_result(&result);
    
    if validation.quality_score > 0.8 {
        println!("High-quality processing achieved!");
        
        // Analyze document structure
        println!("Document Structure:");
        println!("  Complexity: {:?}", result.document_structure.complexity_level);
        println!("  Key themes: {:?}", result.document_structure.key_themes);
        println!("  Reading time: {:?}", result.document_structure.estimated_reading_time);
        
        // Extract scientific entities
        let scientific_entities: Vec<_> = result.global_entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Concept | EntityType::Technology))
            .collect();
            
        println!("Scientific concepts found: {}", scientific_entities.len());
        for entity in scientific_entities {
            println!("  {}: {:.2} confidence", entity.name, entity.confidence);
        }
    } else {
        println!("Quality below threshold. Warnings:");
        for warning in &validation.warnings {
            println!("  {}", warning);
        }
    }
    
    Ok(())
}
```

## Advanced Configuration

### Memory-Constrained Environments

```rust
async fn create_memory_efficient_processor() -> Result<IntelligentKnowledgeProcessor, Box<dyn std::error::Error>> {
    // Configuration for limited memory environments
    let model_config = ModelResourceConfig {
        max_memory_usage: 1_000_000_000, // 1GB limit
        max_concurrent_models: 2,        // Only 2 models at once
        idle_timeout: Duration::from_secs(120), // Aggressive eviction
        min_memory_threshold: 50_000_000, // 50MB minimum
    };
    
    let processing_config = KnowledgeProcessingConfig {
        // Use smaller, more efficient models
        entity_extraction_model: "smollm2_135m".to_string(),
        relationship_extraction_model: "smollm_135m".to_string(),
        semantic_analysis_model: "smollm2_135m".to_string(),
        
        // Smaller chunks to reduce memory usage
        max_chunk_size: 1024,
        min_chunk_size: 64,
        chunk_overlap_size: 32,
        
        // Relaxed thresholds for efficiency
        min_entity_confidence: 0.5,
        min_relationship_confidence: 0.4,
        
        preserve_context: true,
        enable_quality_validation: false, // Disable for speed
    };
    
    let model_manager = Arc::new(ModelResourceManager::new(model_config));
    Ok(IntelligentKnowledgeProcessor::new(model_manager, processing_config))
}
```

### High-Quality Processing

```rust
async fn create_high_quality_processor() -> Result<IntelligentKnowledgeProcessor, Box<dyn std::error::Error>> {
    // Configuration for maximum quality
    let model_config = ModelResourceConfig {
        max_memory_usage: 8_000_000_000, // 8GB for large models
        max_concurrent_models: 5,        // More models for specialized tasks
        idle_timeout: Duration::from_secs(1800), // Keep models loaded longer
        min_memory_threshold: 500_000_000, // 500MB minimum
    };
    
    let processing_config = KnowledgeProcessingConfig {
        // Use largest, most capable models
        entity_extraction_model: "smollm_1_7b".to_string(),
        relationship_extraction_model: "smollm_360m_instruct".to_string(),
        semantic_analysis_model: "smollm2_360m".to_string(),
        
        // Larger chunks for better context
        max_chunk_size: 4096,
        min_chunk_size: 256,
        chunk_overlap_size: 128,
        
        // High confidence thresholds
        min_entity_confidence: 0.8,
        min_relationship_confidence: 0.7,
        
        preserve_context: true,
        enable_quality_validation: true,
    };
    
    let model_manager = Arc::new(ModelResourceManager::new(model_config));
    Ok(IntelligentKnowledgeProcessor::new(model_manager, processing_config))
}
```

### Domain-Specific Configuration

```rust
// Configuration for medical/scientific documents
async fn create_scientific_processor() -> Result<IntelligentKnowledgeProcessor, Box<dyn std::error::Error>> {
    let processing_config = KnowledgeProcessingConfig {
        // Models good at technical content
        entity_extraction_model: "smollm2_360m".to_string(),
        relationship_extraction_model: "smollm_360m_instruct".to_string(),
        semantic_analysis_model: "smollm2_360m".to_string(),
        
        // Medium chunks for technical complexity
        max_chunk_size: 2048,
        min_chunk_size: 256,
        chunk_overlap_size: 100,
        
        // High confidence for precision
        min_entity_confidence: 0.75,
        min_relationship_confidence: 0.65,
        
        preserve_context: true,
        enable_quality_validation: true,
    };
    
    let model_manager = Arc::new(ModelResourceManager::new(ModelResourceConfig::default()));
    Ok(IntelligentKnowledgeProcessor::new(model_manager, processing_config))
}

// Configuration for general web content
async fn create_web_content_processor() -> Result<IntelligentKnowledgeProcessor, Box<dyn std::error::Error>> {
    let processing_config = KnowledgeProcessingConfig {
        // Balanced models for varied content
        entity_extraction_model: "smollm2_135m".to_string(),
        relationship_extraction_model: "smollm_135m_instruct".to_string(),
        semantic_analysis_model: "smollm2_135m".to_string(),
        
        // Flexible chunking for varied formats
        max_chunk_size: 1800,
        min_chunk_size: 100,
        chunk_overlap_size: 50,
        
        // Moderate confidence thresholds
        min_entity_confidence: 0.6,
        min_relationship_confidence: 0.5,
        
        preserve_context: true,
        enable_quality_validation: false, // Speed over perfect quality
    };
    
    let model_manager = Arc::new(ModelResourceManager::new(ModelResourceConfig::default()));
    Ok(IntelligentKnowledgeProcessor::new(model_manager, processing_config))
}
```

## MCP Integration

### Basic MCP Handler Integration

```rust
use llmkg::mcp::*;
use serde_json::Value;

pub struct EnhancedMCPServer {
    processor: Arc<IntelligentKnowledgeProcessor>,
    hierarchical_storage: Arc<HierarchicalStorage>,
}

impl EnhancedMCPServer {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let model_manager = Arc::new(ModelResourceManager::new(
            ModelResourceConfig::default()
        ));
        
        let processor = Arc::new(IntelligentKnowledgeProcessor::new(
            model_manager.clone(),
            KnowledgeProcessingConfig::default()
        ));
        
        let hierarchical_storage = Arc::new(HierarchicalStorage::new().await?);
        
        Ok(Self {
            processor,
            hierarchical_storage,
        })
    }
    
    /// Enhanced store_knowledge handler
    pub async fn handle_store_knowledge(
        &self, 
        params: Value
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let content: String = params["content"].as_str()
            .ok_or("Missing content parameter")?
            .to_string();
            
        let title: String = params["title"].as_str()
            .unwrap_or("Untitled")
            .to_string();
        
        // Process with enhanced system
        let result = self.processor.process_knowledge(&content, &title).await?;
        
        // Store in hierarchical system
        let storage_result = self.hierarchical_storage
            .store_processed_knowledge(&result)
            .await?;
        
        // Validate quality
        let validation = self.processor.validate_processing_result(&result);
        
        let response = serde_json::json!({
            "success": true,
            "document_id": result.document_id,
            "chunks_created": result.chunks.len(),
            "entities_extracted": result.global_entities.len(),
            "relationships_found": result.global_relationships.len(),
            "quality_score": result.quality_metrics.overall_quality,
            "storage_id": storage_result.storage_id,
            "warnings": validation.warnings,
        });
        
        Ok(response)
    }
    
    /// Enhanced find_facts handler with hierarchical search
    pub async fn handle_find_facts(
        &self,
        params: Value
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let query = params["query"].as_str()
            .ok_or("Missing query parameter")?;
        
        // Use hierarchical retrieval system
        let retrieval_engine = RetrievalEngine::new(
            self.hierarchical_storage.clone()
        );
        
        let search_results = retrieval_engine
            .search_knowledge(query)
            .await?;
        
        let facts: Vec<Value> = search_results.into_iter()
            .map(|result| serde_json::json!({
                "subject": result.subject,
                "predicate": result.predicate,
                "object": result.object,
                "confidence": result.confidence,
                "context": result.context,
                "source_document": result.source_document_id,
            }))
            .collect();
        
        Ok(serde_json::json!({
            "facts": facts,
            "total_found": facts.len(),
        }))
    }
}
```

### Advanced MCP Integration with Quality Monitoring

```rust
pub struct QualityAwareMCPServer {
    processor: Arc<IntelligentKnowledgeProcessor>,
    quality_monitor: QualityMonitor,
    performance_tracker: PerformanceTracker,
}

impl QualityAwareMCPServer {
    pub async fn handle_store_knowledge_with_monitoring(
        &self,
        params: Value
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        let content: String = params["content"].as_str()
            .ok_or("Missing content parameter")?
            .to_string();
            
        let title: String = params["title"].as_str()
            .unwrap_or("Untitled")
            .to_string();
        
        // Process with quality monitoring
        let result = self.processor.process_knowledge(&content, &title).await?;
        let processing_time = start_time.elapsed();
        
        // Validate and assess quality
        let validation = self.processor.validate_processing_result(&result);
        let stats = self.processor.get_processing_stats(&result);
        
        // Record metrics
        self.quality_monitor.record_processing_quality(&result, &validation);
        self.performance_tracker.record_processing_time(processing_time, content.len());
        
        // Determine response based on quality
        if validation.quality_score < 0.6 {
            // Low quality processing - provide detailed feedback
            return Ok(serde_json::json!({
                "success": false,
                "error": "Processing quality below acceptable threshold",
                "quality_score": validation.quality_score,
                "issues": validation.warnings,
                "suggestions": [
                    "Try with shorter content segments",
                    "Check content formatting and structure",
                    "Consider using higher quality configuration"
                ]
            }));
        }
        
        // High quality processing - store and return results
        let storage_result = self.hierarchical_storage
            .store_processed_knowledge(&result)
            .await?;
        
        Ok(serde_json::json!({
            "success": true,
            "document_id": result.document_id,
            "storage_id": storage_result.storage_id,
            "quality_metrics": {
                "overall_quality": result.quality_metrics.overall_quality,
                "entity_extraction_quality": result.quality_metrics.entity_extraction_quality,
                "relationship_extraction_quality": result.quality_metrics.relationship_extraction_quality,
                "semantic_coherence": result.quality_metrics.semantic_coherence,
                "context_preservation": result.quality_metrics.context_preservation,
            },
            "performance_metrics": {
                "processing_time_ms": processing_time.as_millis(),
                "chunks_created": stats.total_chunks,
                "entities_extracted": stats.total_entities,
                "relationships_found": stats.total_relationships,
                "average_chunk_size": stats.average_chunk_size,
            },
            "validation": {
                "is_valid": validation.is_valid,
                "warnings": validation.warnings,
            }
        }))
    }
}
```

## Performance Optimization

### Batch Processing

```rust
async fn batch_process_documents(
    processor: &IntelligentKnowledgeProcessor,
    documents: Vec<(String, String)> // (content, title) pairs
) -> Result<Vec<KnowledgeProcessingResult>, Box<dyn std::error::Error>> {
    use tokio::task::JoinSet;
    
    let mut join_set = JoinSet::new();
    let processor = Arc::new(processor);
    
    // Process documents concurrently
    for (content, title) in documents {
        let processor_clone = processor.clone();
        join_set.spawn(async move {
            processor_clone.process_knowledge(&content, &title).await
        });
    }
    
    let mut results = Vec::new();
    while let Some(result) = join_set.join_next().await {
        match result? {
            Ok(processing_result) => results.push(processing_result),
            Err(e) => eprintln!("Document processing failed: {}", e),
        }
    }
    
    Ok(results)
}
```

### Streaming Processing for Large Documents

```rust
async fn stream_process_large_document(
    processor: &IntelligentKnowledgeProcessor,
    file_path: &str
) -> Result<KnowledgeProcessingResult, Box<dyn std::error::Error>> {
    use tokio::fs::File;
    use tokio::io::{AsyncBufReadExt, BufReader};
    
    let file = File::open(file_path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    let mut content_chunks = Vec::new();
    let mut current_chunk = String::new();
    const MAX_CHUNK_SIZE: usize = 8192; // 8KB chunks
    
    // Read file in chunks
    while let Some(line) = lines.next_line().await? {
        if current_chunk.len() + line.len() > MAX_CHUNK_SIZE {
            if !current_chunk.is_empty() {
                content_chunks.push(current_chunk.clone());
                current_chunk.clear();
            }
        }
        current_chunk.push_str(&line);
        current_chunk.push('\n');
    }
    
    if !current_chunk.is_empty() {
        content_chunks.push(current_chunk);
    }
    
    // Process chunks and merge results
    let mut all_chunks = Vec::new();
    let mut all_entities = Vec::new();
    let mut all_relationships = Vec::new();
    
    for (i, chunk_content) in content_chunks.iter().enumerate() {
        let chunk_title = format!("Large Document - Part {}", i + 1);
        let result = processor.process_knowledge(chunk_content, &chunk_title).await?;
        
        all_chunks.extend(result.chunks);
        all_entities.extend(result.global_entities);
        all_relationships.extend(result.global_relationships);
    }
    
    // Create merged result
    let merged_result = KnowledgeProcessingResult {
        document_id: format!("large_doc_{}", 
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs()),
        chunks: all_chunks,
        global_entities: all_entities,
        global_relationships: all_relationships,
        // ... other fields would need proper merging logic
    };
    
    Ok(merged_result)
}
```

### Memory Usage Monitoring

```rust
pub struct MemoryAwareProcessor {
    processor: IntelligentKnowledgeProcessor,
    memory_monitor: MemoryMonitor,
}

impl MemoryAwareProcessor {
    pub async fn process_with_memory_management(
        &self,
        content: &str,
        title: &str
    ) -> Result<KnowledgeProcessingResult, Box<dyn std::error::Error>> {
        // Check memory before processing
        let initial_memory = self.memory_monitor.get_current_usage();
        
        if initial_memory > 0.8 * self.memory_monitor.get_total_memory() {
            // High memory usage - trigger cleanup
            self.processor.model_manager.evict_idle_models().await;
            tokio::task::yield_now().await; // Allow cleanup to complete
        }
        
        let result = self.processor.process_knowledge(content, title).await?;
        
        // Monitor peak memory usage
        let peak_memory = self.memory_monitor.get_peak_usage();
        if peak_memory > 0.9 * self.memory_monitor.get_total_memory() {
            eprintln!("Warning: Near memory limit during processing");
        }
        
        Ok(result)
    }
}
```

## Quality Management

### Quality Validation Pipeline

```rust
pub struct QualityValidator {
    min_quality_threshold: f32,
    entity_confidence_threshold: f32,
    relationship_confidence_threshold: f32,
}

impl QualityValidator {
    pub fn validate_comprehensive(
        &self, 
        result: &KnowledgeProcessingResult
    ) -> QualityValidationReport {
        let mut report = QualityValidationReport::new();
        
        // Overall quality check
        if result.quality_metrics.overall_quality < self.min_quality_threshold {
            report.add_error(format!(
                "Overall quality {:.2} below threshold {:.2}",
                result.quality_metrics.overall_quality,
                self.min_quality_threshold
            ));
        }
        
        // Entity quality analysis
        let low_confidence_entities = result.global_entities
            .iter()
            .filter(|e| e.confidence < self.entity_confidence_threshold)
            .count();
            
        if low_confidence_entities > result.global_entities.len() / 2 {
            report.add_warning(format!(
                "High number of low-confidence entities: {}/{}",
                low_confidence_entities,
                result.global_entities.len()
            ));
        }
        
        // Relationship quality analysis
        let low_confidence_relationships = result.global_relationships
            .iter()
            .filter(|r| r.confidence < self.relationship_confidence_threshold)
            .count();
        
        if low_confidence_relationships > result.global_relationships.len() / 2 {
            report.add_warning(format!(
                "High number of low-confidence relationships: {}/{}",
                low_confidence_relationships,
                result.global_relationships.len()
            ));
        }
        
        // Chunk coherence analysis
        let low_coherence_chunks = result.chunks
            .iter()
            .filter(|c| c.semantic_coherence < 0.5)
            .count();
            
        if low_coherence_chunks > 0 {
            report.add_warning(format!(
                "Found {} chunks with low semantic coherence",
                low_coherence_chunks
            ));
        }
        
        // Coverage analysis
        let entities_per_chunk = result.global_entities.len() as f32 / result.chunks.len() as f32;
        if entities_per_chunk < 1.0 {
            report.add_warning("Low entity density - may indicate poor extraction".to_string());
        }
        
        report
    }
}

pub struct QualityValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

impl QualityValidationReport {
    pub fn is_acceptable(&self) -> bool {
        self.errors.is_empty()
    }
    
    pub fn print_report(&self) {
        if !self.errors.is_empty() {
            println!("Quality Validation Errors:");
            for error in &self.errors {
                println!("  ❌ {}", error);
            }
        }
        
        if !self.warnings.is_empty() {
            println!("Quality Warnings:");
            for warning in &self.warnings {
                println!("  ⚠️  {}", warning);
            }
        }
        
        if !self.recommendations.is_empty() {
            println!("Recommendations:");
            for rec in &self.recommendations {
                println!("  💡 {}", rec);
            }
        }
    }
}
```

## Error Handling

### Comprehensive Error Handling

```rust
pub async fn robust_knowledge_processing(
    processor: &IntelligentKnowledgeProcessor,
    content: &str,
    title: &str
) -> Result<KnowledgeProcessingResult, ProcessingError> {
    let max_retries = 3;
    let mut attempt = 0;
    
    loop {
        attempt += 1;
        
        match processor.process_knowledge(content, title).await {
            Ok(result) => {
                // Validate result quality
                let validation = processor.validate_processing_result(&result);
                
                if validation.is_valid {
                    return Ok(result);
                } else if attempt < max_retries {
                    // Try with relaxed configuration
                    eprintln!("Attempt {}: Quality issues, retrying with relaxed settings", attempt);
                    continue;
                } else {
                    return Err(ProcessingError::QualityValidationFailed(validation));
                }
            },
            Err(EnhancedStorageError::ModelNotFound(model)) => {
                eprintln!("Model {} not found locally - ensure all required models are in model_weights directory", model);
                // All models must be available locally - no fallbacks available
                return Err(ProcessingError::ModelUnavailable(model));
            },
            Err(EnhancedStorageError::InsufficientResources(msg)) => {
                eprintln!("Resource issue: {}, cleaning up", msg);
                
                // Trigger cleanup
                processor.model_manager.evict_idle_models().await;
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                
                if attempt < max_retries {
                    continue;
                } else {
                    return Err(ProcessingError::ResourceExhaustion(msg));
                }
            },
            Err(e) => {
                eprintln!("Processing error on attempt {}: {}", attempt, e);
                if attempt < max_retries {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100 * attempt as u64)).await;
                    continue;
                } else {
                    return Err(ProcessingError::ProcessingFailed(e.to_string()));
                }
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    #[error("Quality validation failed: {0:?}")]
    QualityValidationFailed(ProcessingValidation),
    
    #[error("Required model unavailable: {0}")]
    ModelUnavailable(String),
    
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
    
    #[error("Processing failed after retries: {0}")]
    ProcessingFailed(String),
}
```

## Best Practices

### 1. Configuration Management

```rust
// Use environment-specific configurations
pub fn get_processing_config() -> KnowledgeProcessingConfig {
    match std::env::var("ENVIRONMENT").as_deref() {
        Ok("production") => KnowledgeProcessingConfig {
            entity_extraction_model: "smollm2_360m".to_string(),
            max_chunk_size: 2048,
            min_entity_confidence: 0.7,
            enable_quality_validation: true,
            ..Default::default()
        },
        Ok("development") => KnowledgeProcessingConfig {
            entity_extraction_model: "smollm2_135m".to_string(),
            max_chunk_size: 1024,
            min_entity_confidence: 0.5,
            enable_quality_validation: false,
            ..Default::default()
        },
        _ => KnowledgeProcessingConfig::default(),
    }
}
```

### 2. Resource Management

```rust
// Always use Arc for shared resources
let model_manager = Arc::new(ModelResourceManager::new(config));

// Monitor resource usage
let usage = model_manager.get_resource_usage();
if usage.memory_usage > 0.8 * usage.max_memory {
    model_manager.evict_idle_models().await;
}
```

### 3. Quality Assurance

```rust
// Always validate processing results
let result = processor.process_knowledge(content, title).await?;
let validation = processor.validate_processing_result(&result);

if !validation.is_valid {
    // Handle quality issues
    log::warn!("Processing quality issues: {:?}", validation.warnings);
}
```

### 4. Performance Monitoring

```rust
// Track processing performance
let start_time = std::time::Instant::now();
let result = processor.process_knowledge(content, title).await?;
let processing_time = start_time.elapsed();

// Log performance metrics
log::info!("Processed {} chars in {:?}, quality: {:.2}",
    content.len(), 
    processing_time, 
    result.quality_metrics.overall_quality
);
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Out of Memory Errors
```rust
// Solution: Reduce memory limits and concurrent models
let config = ModelResourceConfig {
    max_memory_usage: 1_000_000_000, // Reduce to 1GB
    max_concurrent_models: 2,        // Limit concurrent models
    idle_timeout: Duration::from_secs(60), // Aggressive eviction
    ..Default::default()
};
```

#### Issue: Poor Entity Extraction Quality
```rust
// Solution: Use larger models or adjust confidence thresholds
let config = KnowledgeProcessingConfig {
    entity_extraction_model: "smollm2_360m".to_string(), // Larger model
    min_entity_confidence: 0.5,  // Lower threshold
    ..Default::default()
};
```

#### Issue: Slow Processing Times
```rust
// Solution: Use smaller models or reduce chunk sizes
let config = KnowledgeProcessingConfig {
    entity_extraction_model: "smollm2_135m".to_string(), // Smaller model
    max_chunk_size: 1024,        // Smaller chunks
    enable_quality_validation: false, // Skip validation
    ..Default::default()
};
```

#### Issue: Model Loading Failures
```rust
// Solution: Check model availability and network connectivity
async fn verify_models(manager: &ModelResourceManager) -> Result<(), Box<dyn std::error::Error>> {
    let models = ["smollm2_135m", "smollm2_360m", "smollm_360m_instruct"];
    
    for model_id in &models {
        match manager.load_model(model_id).await {
            Ok(_) => println!("✅ Model {} loaded successfully", model_id),
            Err(e) => println!("❌ Failed to load model {}: {}", model_id, e),
        }
    }
    
    Ok(())
}
```

### Debugging Tools

```rust
// Enable detailed logging
pub fn enable_debug_logging() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();
}

// Processing diagnostics
pub fn diagnose_processing_result(result: &KnowledgeProcessingResult) {
    println!("Processing Diagnostics:");
    println!("  Document ID: {}", result.document_id);
    println!("  Chunks: {} (avg size: {:.0})", 
        result.chunks.len(),
        result.chunks.iter().map(|c| c.content.len()).sum::<usize>() as f32 / result.chunks.len() as f32
    );
    println!("  Entities: {} (avg confidence: {:.2})",
        result.global_entities.len(),
        result.global_entities.iter().map(|e| e.confidence).sum::<f32>() / result.global_entities.len() as f32
    );
    println!("  Relationships: {} (avg confidence: {:.2})",
        result.global_relationships.len(),
        result.global_relationships.iter().map(|r| r.confidence).sum::<f32>() / result.global_relationships.len() as f32
    );
    println!("  Processing time: {:?}", result.processing_metadata.processing_time);
    println!("  Quality metrics:");
    println!("    Overall: {:.2}", result.quality_metrics.overall_quality);
    println!("    Entity extraction: {:.2}", result.quality_metrics.entity_extraction_quality);
    println!("    Relationship extraction: {:.2}", result.quality_metrics.relationship_extraction_quality);
    println!("    Semantic coherence: {:.2}", result.quality_metrics.semantic_coherence);
    println!("    Context preservation: {:.2}", result.quality_metrics.context_preservation);
}
```

This usage guide provides comprehensive examples and patterns for effectively using the Enhanced Knowledge Storage System in various scenarios and environments.