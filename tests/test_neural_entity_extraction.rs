use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use llmkg::cognitive::attention::{AttentionManager, AttentionConfig};
use llmkg::cognitive::working_memory::WorkingMemory;
use llmkg::cognitive::brain_metrics::BrainMetricsCollector;
use llmkg::monitoring::performance::PerformanceMonitor;
use llmkg::neural::neural_server::NeuralProcessingServer;
use llmkg::core::entity_extractor::CognitiveEntityExtractor;
use llmkg::models::model_loader::{ModelLoader, ModelLoaderConfig};
use std::sync::Arc;
use tokio;

#[tokio::test]
async fn test_neural_entity_extraction_accuracy() {
    // Initialize all required components
    let cognitive_orchestrator = Arc::new(CognitiveOrchestrator::new().await.unwrap());
    let attention_manager = Arc::new(AttentionManager::new(AttentionConfig::default()));
    let working_memory = Arc::new(WorkingMemory::new(Default::default()));
    let metrics_collector = Arc::new(BrainMetricsCollector::new());
    let performance_monitor = Arc::new(PerformanceMonitor::new(Default::default()));
    
    // Initialize model loader and neural server
    let model_loader_config = ModelLoaderConfig {
        cache_dir: std::path::PathBuf::from("./test_models"),
        enable_onnx: true,
        enable_rust_bert: false,
        enable_candle: true,
        max_memory_gb: 4.0,
    };
    
    let model_loader = Arc::new(ModelLoader::new(model_loader_config).await.unwrap());
    let neural_server = Arc::new(NeuralProcessingServer::new(model_loader).await.unwrap());
    
    // Create entity extractor with neural server
    let extractor = CognitiveEntityExtractor::new(
        cognitive_orchestrator,
        attention_manager,
        working_memory,
        metrics_collector,
        performance_monitor,
    ).with_neural_server(neural_server);
    
    // Test sentences
    let test_cases = vec![
        (
            "Albert Einstein developed the Theory of Relativity in 1905",
            vec!["Albert Einstein", "Theory of Relativity", "1905"],
        ),
        (
            "Marie Curie won the Nobel Prize in Physics and Chemistry",
            vec!["Marie Curie", "Nobel Prize", "Physics", "Chemistry"],
        ),
    ];
    
    let mut total_accuracy = 0.0;
    let mut total_time_ms = 0.0;
    
    for (text, expected_entities) in test_cases {
        let start = tokio::time::Instant::now();
        let entities = extractor.extract_entities(text).await.unwrap();
        let duration = start.elapsed();
        
        // Check accuracy
        let extracted_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
        let mut correct = 0;
        for expected in &expected_entities {
            if extracted_names.iter().any(|name| name.contains(expected)) {
                correct += 1;
            }
        }
        
        let accuracy = correct as f32 / expected_entities.len() as f32;
        total_accuracy += accuracy;
        
        // Check performance
        let ms_per_sentence = duration.as_millis() as f32;
        total_time_ms += ms_per_sentence;
        
        println!("Text: {}", text);
        println!("Extracted entities: {:?}", extracted_names);
        println!("Accuracy: {:.2}%", accuracy * 100.0);
        println!("Time: {:.2}ms", ms_per_sentence);
        println!("---");
        
        // Verify neural extraction was used
        assert!(entities.iter().any(|e| matches!(
            e.extraction_model,
            llmkg::core::types::ExtractionModel::NeuralServer |
            llmkg::core::types::ExtractionModel::CognitiveDistilBERT |
            llmkg::core::types::ExtractionModel::CognitiveTinyBERT
        )));
    }
    
    let avg_accuracy = total_accuracy / test_cases.len() as f32;
    let avg_time_ms = total_time_ms / test_cases.len() as f32;
    
    println!("\nFinal Results:");
    println!("Average accuracy: {:.2}%", avg_accuracy * 100.0);
    println!("Average time per sentence: {:.2}ms", avg_time_ms);
    
    // Verify success criteria
    assert!(avg_accuracy >= 0.95, "Accuracy must be >95%, got {:.2}%", avg_accuracy * 100.0);
    assert!(avg_time_ms <= 8.0, "Time must be <8ms per sentence, got {:.2}ms", avg_time_ms);
}