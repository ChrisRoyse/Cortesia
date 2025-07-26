use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use llmkg::cognitive::attention::{AttentionManager, AttentionConfig};
use llmkg::cognitive::working_memory::WorkingMemory;
use llmkg::cognitive::brain_metrics::BrainMetricsCollector;
use llmkg::monitoring::performance::PerformanceMonitor;
use llmkg::neural::neural_server::NeuralProcessingServer;
use llmkg::core::entity_extractor::CognitiveEntityExtractor;
use llmkg::models::model_loader::{ModelLoader, ModelLoaderConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Entity Extraction Demo - Using CognitiveOrchestrator + NeuralProcessingServer");
    println!("=" .repeat(80));

    // Initialize all required components
    println!("\n1. Initializing Cognitive Components...");
    let cognitive_orchestrator = Arc::new(CognitiveOrchestrator::new().await?);
    let attention_manager = Arc::new(AttentionManager::new(AttentionConfig::default()));
    let working_memory = Arc::new(WorkingMemory::new(Default::default()));
    let metrics_collector = Arc::new(BrainMetricsCollector::new());
    let performance_monitor = Arc::new(PerformanceMonitor::new(Default::default()));
    
    // Initialize model loader and neural server
    println!("\n2. Loading Neural Models...");
    let model_loader_config = ModelLoaderConfig {
        cache_dir: std::path::PathBuf::from("./models"),
        enable_onnx: true,
        enable_rust_bert: false,
        enable_candle: true,
        max_memory_gb: 4.0,
    };
    
    let model_loader = Arc::new(ModelLoader::new(model_loader_config).await?);
    let neural_server = Arc::new(NeuralProcessingServer::new(model_loader).await?);
    
    // Create entity extractor with neural server
    println!("\n3. Creating Cognitive Entity Extractor with Neural Integration...");
    let extractor = CognitiveEntityExtractor::new(
        cognitive_orchestrator,
        attention_manager,
        working_memory,
        metrics_collector,
        performance_monitor,
    ).with_neural_server(neural_server);
    
    // Test sentences
    let test_sentences = vec![
        "Albert Einstein developed the Theory of Relativity in 1905",
        "Marie Curie won the Nobel Prize in Physics and Chemistry",
        "The European Union announced new climate policies today",
        "Amazon Web Services launched a new quantum computing service",
    ];
    
    println!("\n4. Extracting Entities with Neural Models...\n");
    
    for (i, text) in test_sentences.iter().enumerate() {
        println!("Test #{}: \"{}\"", i + 1, text);
        println!("-" .repeat(60));
        
        let start = tokio::time::Instant::now();
        
        // Extract entities using cognitive orchestration + neural processing
        match extractor.extract_entities(text).await {
            Ok(entities) => {
                let duration = start.elapsed();
                
                println!("✅ Extraction completed in {:.2}ms", duration.as_millis());
                println!("📊 Found {} entities:", entities.len());
                
                for entity in &entities {
                    println!("   • {} ({:?})", entity.name, entity.entity_type);
                    println!("     - Confidence: {:.2}%", entity.confidence_score * 100.0);
                    println!("     - Model: {:?}", entity.extraction_model);
                    println!("     - Pattern: {:?}", entity.reasoning_pattern);
                    println!("     - Neural Salience: {:.2}", entity.neural_salience);
                }
                
                // Verify performance target
                let ms_per_sentence = duration.as_millis() as f32;
                if ms_per_sentence <= 8.0 {
                    println!("⚡ Performance: PASS ({:.2}ms <= 8ms target)", ms_per_sentence);
                } else {
                    println!("⚠️  Performance: SLOW ({:.2}ms > 8ms target)", ms_per_sentence);
                }
            }
            Err(e) => {
                println!("❌ Error extracting entities: {}", e);
            }
        }
        
        println!();
    }
    
    println!("\n✨ Demo completed successfully!");
    println!("\nKey Integration Points:");
    println!("1. CognitiveOrchestrator.reason() - Plans extraction strategy");
    println!("2. NeuralProcessingServer.neural_predict() - Performs actual neural inference");
    println!("3. AttentionManager.compute_attention() - Computes attention weights");
    println!("4. WorkingMemory.store_entities() - Stores entities for context");
    
    Ok(())
}