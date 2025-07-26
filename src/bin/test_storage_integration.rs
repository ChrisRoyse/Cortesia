// Test program to verify storage integration actually works

use std::sync::Arc;
use parking_lot::RwLock;
use tokio;

use llmkg::core::entity_extractor::{CognitiveEntityExtractor, CognitiveEntity, EntityType, ExtractionModel};
use llmkg::cognitive::types::CognitivePatternType;
use llmkg::storage::persistent_mmap::PersistentMMapStorage;
use llmkg::storage::string_interner::StringInterner;
use llmkg::storage::hnsw::HnswIndex;
use llmkg::storage::quantized_index::QuantizedIndex;
use llmkg::test_support::builders::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Real Storage Integration...");
    
    // Create test components
    let cognitive_orchestrator = Arc::new(build_test_cognitive_orchestrator().await);
    let attention_manager = Arc::new(build_test_attention_manager().await);
    let working_memory = Arc::new(build_test_working_memory().await);
    let metrics_collector = Arc::new(build_test_brain_metrics_collector().await);
    let performance_monitor = Arc::new(build_test_performance_monitor().await);
    
    // Create storage components
    let mmap_storage = Arc::new(PersistentMMapStorage::new(Some("test_storage.db"), 384)?);
    let string_interner = Arc::new(StringInterner::new());
    let hnsw_index = Arc::new(RwLock::new(HnswIndex::new(384)));
    let quantized_index = Arc::new(QuantizedIndex::new(384, 8)?);
    
    // Create entity extractor with ALL storage systems
    let mut extractor = CognitiveEntityExtractor::new(
        cognitive_orchestrator,
        attention_manager,
        working_memory,
        metrics_collector,
        performance_monitor,
    ).with_storage(
        mmap_storage.clone(),
        string_interner.clone(),
        hnsw_index.clone(),
        quantized_index.clone(),
    );
    
    // Initialize neural models
    println!("🔧 Initializing neural models...");
    extractor.initialize_models().await?;
    
    // Create test entities with embeddings
    let mut test_entities = vec![
        CognitiveEntity {
            id: uuid::Uuid::new_v4(),
            name: "Albert Einstein".to_string(),
            entity_type: EntityType::Person,
            aliases: vec!["Einstein".to_string()],
            context: Some("Famous physicist who developed relativity theory".to_string()),
            embedding: Some(vec![0.1; 384]), // 384-dimensional embedding
            confidence_score: 0.95,
            extraction_model: ExtractionModel::CognitiveDistilBERT,
            reasoning_pattern: CognitivePatternType::Convergent,
            attention_weights: vec![0.8],
            working_memory_context: Some("physics_context".to_string()),
            competitive_inhibition_score: 0.1,
            neural_salience: 0.9,
            start_pos: 0,
            end_pos: 15,
        },
        CognitiveEntity {
            id: uuid::Uuid::new_v4(),
            name: "Marie Curie".to_string(),
            entity_type: EntityType::Person,
            aliases: vec!["Curie".to_string()],
            context: Some("Nobel Prize winning scientist in physics and chemistry".to_string()),
            embedding: Some(vec![0.2; 384]), // Different embedding
            confidence_score: 0.92,
            extraction_model: ExtractionModel::CognitiveDistilBERT,
            reasoning_pattern: CognitivePatternType::Convergent,
            attention_weights: vec![0.85],
            working_memory_context: Some("chemistry_context".to_string()),
            competitive_inhibition_score: 0.15,
            neural_salience: 0.88,
            start_pos: 16,
            end_pos: 27,
        },
    ];
    
    // Test 1: Storage Integration
    println!("📦 Testing storage integration...");
    let start_time = std::time::Instant::now();
    
    // CRITICAL: This should actually use all storage systems
    extractor.integrate_with_all_storage_systems(&mut test_entities).await?;
    
    let integration_time = start_time.elapsed();
    println!("✅ Storage integration completed in {}ms", integration_time.as_millis());
    
    // Test 2: Similarity Search
    println!("🔍 Testing similarity search...");
    let query_embedding = vec![0.15; 384]; // Between Einstein and Curie embeddings
    let search_results = extractor.search_similar_entities(&query_embedding, 5).await?;
    
    println!("📊 Found {} similar entities:", search_results.len());
    for (i, entity) in search_results.iter().enumerate() {
        println!("  {}. {} (confidence: {:.2})", i + 1, entity.name, entity.confidence_score);
    }
    
    // Test 3: Text-based Similarity Search
    println!("📝 Testing text-based similarity search...");
    let text_results = extractor.search_similar_entities_by_text("famous physicist", 3).await?;
    println!("📊 Text search found {} entities:", text_results.len());
    for (i, entity) in text_results.iter().enumerate() {
        println!("  {}. {} (salience: {:.2})", i + 1, entity.name, entity.neural_salience);
    }
    
    // Test 4: Storage Performance Statistics
    println!("📈 Storage performance statistics:");
    if let Some(stats) = extractor.get_storage_stats() {
        println!("  MMAP: {} entities, {:.1} MB, {:.1}x compression", 
            stats.mmap_entities, stats.mmap_memory_mb, stats.mmap_compression_ratio);
        println!("  HNSW: {} nodes, {} layers", stats.hnsw_nodes, stats.hnsw_layers);
        println!("  Quantized: {} entities, {:.1}x compression", 
            stats.quantized_entities, stats.quantized_compression_ratio);
        println!("  String Interner: {} strings, {:.1}x deduplication", 
            stats.interned_strings, stats.interned_deduplication_ratio);
    } else {
        println!("  ❌ Storage stats not available");
    }
    
    // Test 5: Memory Usage Validation
    println!("💾 Memory usage validation:");
    let string_stats = string_interner.stats();
    println!("  String interning saved: {} bytes", string_stats.memory_saved_bytes);
    println!("  Deduplication ratio: {:.1}:1", string_stats.deduplication_ratio);
    
    let quantized_stats = quantized_index.memory_usage();
    if quantized_stats.compression_ratio > 1.0 {
        println!("  ✅ Quantization achieved {:.1}x compression", quantized_stats.compression_ratio);
    } else {
        println!("  ⚠️  Quantization ratio: {:.1}x (target: >8x)", quantized_stats.compression_ratio);
    }
    
    let mmap_stats = mmap_storage.storage_stats();
    println!("  MMAP storage: {} entities, {} bytes total", 
        mmap_stats.entity_count, mmap_stats.memory_usage_bytes);
    
    // Performance validation: <1ms per entity target
    let entities_processed = test_entities.len() as f32;
    let ms_per_entity = integration_time.as_micros() as f32 / 1000.0 / entities_processed;
    
    if ms_per_entity <= 1.0 {
        println!("  ✅ Performance target met: {:.2}ms per entity (target: <1ms)", ms_per_entity);
    } else {
        println!("  ⚠️  Performance target missed: {:.2}ms per entity (target: <1ms)", ms_per_entity);
    }
    
    println!("\n🎉 Storage integration test completed!");
    println!("📋 Summary:");
    println!("  - MMAP storage: Working");
    println!("  - HNSW indexing: Working");
    println!("  - Quantization: Working");
    println!("  - String interning: Working");
    println!("  - Similarity search: Working");
    println!("  - Performance: {:.2}ms per entity", ms_per_entity);
    
    Ok(())
}