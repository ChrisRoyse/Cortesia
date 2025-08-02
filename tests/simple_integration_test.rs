//! Simple Integration Test for Enhanced Knowledge Storage System
//! 
//! Demonstrates basic functionality of the system components

use llmkg::enhanced_knowledge_storage::{
    ai_components::{
        AIModelBackend, RealEntityExtractor, RealSemanticChunker, 
        RealReasoningEngine, PerformanceMonitor, ModelConfig
    },
    production::{
        ProductionConfig, Environment, PerformanceMonitor as ProdPerfMonitor
    }
};
use std::sync::Arc;

#[tokio::test]
async fn test_basic_ai_components() {
    println!("🚀 Testing Enhanced Knowledge Storage AI Components");
    
    // Initialize configuration
    let model_config = ModelConfig {
        model_135m_path: Some("models/135M".to_string()),
        model_360m_path: Some("models/360M".to_string()),
        model_1_5b_path: Some("models/1.5B".to_string()),
        model_7b_path: Some("models/7B".to_string()),
        cache_size: 100,
        max_concurrent_models: 2,
        model_timeout_secs: 30,
        enable_gpu: false,
    };
    
    // Initialize AI Model Backend
    println!("\n🤖 Initializing AI Model Backend...");
    let model_backend = Arc::new(
        AIModelBackend::new(model_config)
            .await
            .expect("Failed to initialize model backend")
    );
    println!("✅ Model backend initialized");
    
    // Initialize Performance Monitor
    println!("\n📊 Initializing Performance Monitor...");
    let perf_monitor = Arc::new(PerformanceMonitor::new());
    println!("✅ Performance monitor ready");
    
    // Test Entity Extractor
    println!("\n🔍 Testing Entity Extractor...");
    let entity_extractor = RealEntityExtractor::new(
        model_backend.clone(),
        perf_monitor.clone()
    ).await.expect("Failed to create entity extractor");
    
    let test_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California.";
    let entities = entity_extractor.extract_entities(test_text, None)
        .await
        .expect("Failed to extract entities");
    
    println!("✅ Extracted {} entities from test text", entities.len());
    for entity in &entities {
        println!("  - {}: {} (confidence: {:.2})", 
            entity.entity_type, entity.text, entity.confidence);
    }
    
    // Test Semantic Chunker
    println!("\n📄 Testing Semantic Chunker...");
    let semantic_chunker = RealSemanticChunker::new(
        model_backend.clone(),
        perf_monitor.clone()
    ).await.expect("Failed to create semantic chunker");
    
    let long_text = r#"
        Artificial Intelligence has revolutionized many industries. 
        Machine learning models can now understand and generate human-like text.
        
        In healthcare, AI assists doctors in diagnosing diseases.
        The technology analyzes medical images with high accuracy.
        
        Financial services use AI for fraud detection and risk assessment.
        Automated trading systems make decisions in milliseconds.
    "#;
    
    let chunks = semantic_chunker.chunk_text(long_text, None)
        .await
        .expect("Failed to chunk text");
    
    println!("✅ Created {} semantic chunks", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  - Chunk {}: {} chars, coherence: {:.2}", 
            i + 1, chunk.content.len(), chunk.coherence_score);
    }
    
    // Test Reasoning Engine
    println!("\n🧠 Testing Reasoning Engine...");
    let reasoning_engine = RealReasoningEngine::new(
        model_backend.clone(),
        perf_monitor.clone()
    ).await.expect("Failed to create reasoning engine");
    
    let query = "What is the relationship between AI and healthcare?";
    let context = vec![
        "AI assists doctors in diagnosis".to_string(),
        "Machine learning analyzes medical images".to_string(),
    ];
    
    let reasoning_result = reasoning_engine.reason(query, &context, 3)
        .await
        .expect("Failed to perform reasoning");
    
    println!("✅ Reasoning completed with {} steps", reasoning_result.steps.len());
    println!("  Final answer: {}", reasoning_result.final_answer);
    println!("  Confidence: {:.2}", reasoning_result.confidence);
    
    // Check performance metrics
    println!("\n📈 Performance Metrics Summary:");
    let metrics = perf_monitor.get_recent_metrics(
        std::time::Duration::from_secs(60)
    ).await;
    
    println!("  Total operations: {}", metrics.len());
    for metric in metrics.iter().take(5) {
        println!("  - {:?}: {:.2}ms, {} tokens", 
            metric.operation_type,
            metric.duration.as_millis(),
            metric.tokens_processed
        );
    }
    
    println!("\n✨ All AI components tested successfully!");
}

#[tokio::test]
async fn test_production_config() {
    println!("🔧 Testing Production Configuration System");
    
    // Test environment-specific configs
    let environments = vec![
        Environment::Development,
        Environment::Testing,
        Environment::Staging,
        Environment::Production,
    ];
    
    for env in environments {
        println!("\n📋 Loading config for {:?} environment", env);
        let config = ProductionConfig::from_environment(env)
            .expect(&format!("Failed to load {:?} config", env));
        
        println!("✅ Configuration loaded:");
        println!("  - AI models enabled: {}", config.model_config.model_135m_path.is_some());
        println!("  - Cache size: {}", config.model_config.cache_size);
        println!("  - Monitoring enabled: {}", config.monitoring_config.enable_monitoring);
        println!("  - Error recovery: {:?}", config.error_handling_config.recovery_strategies);
        
        // Validate config
        config.validate().expect(&format!("Invalid {:?} config", env));
        println!("✅ Configuration validated");
    }
    
    println!("\n✨ Production configuration system working correctly!");
}

#[tokio::test]
async fn test_caching_system() {
    use llmkg::enhanced_knowledge_storage::production::{
        MultiLevelCache, CacheConfig, WriteStrategy
    };
    
    println!("💾 Testing Multi-Level Caching System");
    
    let cache_config = CacheConfig {
        l1_capacity: 100,
        l1_max_bytes: 10 * 1024 * 1024, // 10MB
        l2_cache_dir: Some("./test_cache".to_string()),
        l2_max_bytes: 100 * 1024 * 1024, // 100MB
        l3_redis_url: None, // Skip Redis for test
        write_strategy: WriteStrategy::WriteThrough,
        ttl_seconds: 300,
        compression_enabled: true,
        compression_level: 6,
    };
    
    let cache = MultiLevelCache::new(cache_config)
        .await
        .expect("Failed to create cache");
    
    println!("\n📥 Testing cache operations...");
    
    // Test put and get
    let key = "test_key_1";
    let value = "This is test data for caching";
    
    cache.put(key.to_string(), value.to_string(), None)
        .await
        .expect("Failed to put to cache");
    println!("✅ Stored value in cache");
    
    let retrieved: Option<String> = cache.get(key).await;
    assert_eq!(retrieved, Some(value.to_string()));
    println!("✅ Retrieved value from cache");
    
    // Test cache stats
    let stats = cache.get_statistics().await;
    println!("\n📊 Cache Statistics:");
    println!("  - L1 Hits: {}", stats.l1_hits);
    println!("  - L1 Misses: {}", stats.l1_misses);
    println!("  - L1 Hit Rate: {:.2}%", stats.l1_hit_rate * 100.0);
    
    // Test pattern invalidation
    cache.put("pattern:1".to_string(), "data1".to_string(), None).await.unwrap();
    cache.put("pattern:2".to_string(), "data2".to_string(), None).await.unwrap();
    cache.put("other:1".to_string(), "data3".to_string(), None).await.unwrap();
    
    let invalidated = cache.invalidate_pattern("pattern:*")
        .await
        .expect("Failed to invalidate pattern");
    
    println!("✅ Invalidated {} entries matching pattern", invalidated);
    assert_eq!(invalidated, 2);
    
    // Verify pattern entries are gone
    let result1: Option<String> = cache.get("pattern:1").await;
    let result2: Option<String> = cache.get("pattern:2").await;
    let result3: Option<String> = cache.get("other:1").await;
    
    assert!(result1.is_none());
    assert!(result2.is_none());
    assert!(result3.is_some());
    
    println!("\n✨ Caching system working perfectly!");
    
    // Cleanup
    std::fs::remove_dir_all("./test_cache").ok();
}

#[tokio::test]
async fn test_monitoring_system() {
    use llmkg::enhanced_knowledge_storage::production::{
        PerformanceMonitor as ProdMonitor, MonitoringConfig
    };
    
    println!("📊 Testing Production Monitoring System");
    
    let config = MonitoringConfig {
        enable_monitoring: true,
        metrics_retention_days: 7,
        sampling_rate: 1.0,
        export_interval_secs: 60,
        alert_thresholds: Default::default(),
        dashboard_config: Default::default(),
    };
    
    let monitor = ProdMonitor::new(config);
    
    println!("\n📈 Recording test metrics...");
    
    // Simulate document processing
    monitor.record_document_processing(
        "doc_001",
        std::time::Duration::from_millis(250),
        150 * 1024, // 150KB
    ).await;
    
    // Simulate query processing
    monitor.record_query_processing(
        "query_001",
        std::time::Duration::from_millis(45),
        0.92, // confidence
    ).await;
    
    // Get performance report
    let report = monitor.generate_performance_report(
        llmkg::enhanced_knowledge_storage::production::TimeRange::LastHour
    ).await.expect("Failed to generate report");
    
    println!("\n📄 Performance Report:");
    println!("  Period: {} to {}", report.start_time, report.end_time);
    println!("  Total requests: {}", report.total_requests);
    println!("  Average latency: {:.2}ms", report.average_latency_ms);
    println!("  Success rate: {:.2}%", report.success_rate * 100.0);
    
    println!("\n✨ Monitoring system operational!");
}

fn main() {
    println!("Run tests with: cargo test --test simple_integration_test");
}