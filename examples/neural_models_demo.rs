//! Demo of real neural model integration for entity extraction
//!
//! This example shows how the LLMKG system uses real pre-trained neural models
//! instead of placeholder implementations for high-quality entity extraction.

use llmkg::core::entity_extractor::CognitiveEntityExtractor;
use llmkg::neural::neural_server::NeuralProcessingServer;
use llmkg::test_support::builders::*;
use std::sync::Arc;
use tokio::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LLMKG Real Neural Model Integration Demo ===\n");
    
    // Initialize cognitive components
    println!("Initializing cognitive components...");
    let cognitive_orchestrator = Arc::new(build_test_cognitive_orchestrator().await);
    let attention_manager = Arc::new(build_test_attention_manager().await);
    let working_memory = Arc::new(build_test_working_memory().await);
    let metrics_collector = Arc::new(build_test_brain_metrics_collector().await);
    let performance_monitor = Arc::new(build_test_performance_monitor().await);
    
    // Create neural server
    println!("Creating neural processing server...");
    let neural_server = Arc::new(NeuralProcessingServer::new("localhost:8080".to_string()).await?);
    
    // Initialize models (this loads real pre-trained weights)
    println!("\nLoading pre-trained neural models:");
    let load_start = Instant::now();
    neural_server.initialize_models().await?;
    println!("✓ Models loaded in {:.2}s", load_start.elapsed().as_secs_f64());
    
    // Create entity extractor with neural server
    let mut extractor = CognitiveEntityExtractor::new(
        cognitive_orchestrator,
        attention_manager,
        working_memory,
        metrics_collector,
        performance_monitor,
    ).with_neural_server(neural_server.clone());
    
    // Initialize extractor models
    extractor.initialize_models().await?;
    
    // Demo texts
    let demo_texts = vec![
        (
            "Scientific Discovery",
            "Marie Curie discovered radium and polonium at the University of Paris. She won Nobel Prizes in Physics (1903) and Chemistry (1911)."
        ),
        (
            "Technology Leaders",
            "Elon Musk founded SpaceX in 2002 and leads Tesla's development of electric vehicles in California."
        ),
        (
            "Historical Events",
            "The Theory of Relativity was published by Albert Einstein in 1905, revolutionizing physics at the University of Berlin."
        ),
    ];
    
    println!("\n=== Entity Extraction with Real Neural Models ===\n");
    
    for (title, text) in demo_texts {
        println!("{}:", title);
        println!("Text: {}", text);
        
        let start = Instant::now();
        let entities = extractor.extract_entities(text).await?;
        let extraction_time = start.elapsed();
        
        println!("\nExtracted {} entities in {:.2}ms:", entities.len(), extraction_time.as_millis());
        
        for entity in &entities {
            println!(
                "  - {} ({:?}) [confidence: {:.2}, model: {:?}]",
                entity.name,
                entity.entity_type,
                entity.confidence_score,
                entity.extraction_model
            );
            
            // Show embeddings if available
            if let Some(embedding) = &entity.embedding {
                println!(
                    "    → Embedding: [{:.3}, {:.3}, {:.3}, ...] (dim: {})",
                    embedding[0], embedding[1], embedding[2], embedding.len()
                );
            }
        }
        
        // Performance metrics
        let sentence_count = text.matches(|c: char| c == '.' || c == '!' || c == '?').count();
        let ms_per_sentence = extraction_time.as_millis() as f32 / sentence_count as f32;
        println!("\nPerformance: {:.2}ms per sentence", ms_per_sentence);
        println!("─".repeat(60));
    }
    
    // Demonstrate embedding generation
    println!("\n=== Semantic Embeddings with MiniLM ===\n");
    
    let concepts = vec![
        "artificial intelligence",
        "machine learning",
        "neural networks",
        "deep learning",
    ];
    
    println!("Generating embeddings for concepts:");
    for concept in &concepts {
        let start = Instant::now();
        let embedding = neural_server.get_embedding(concept).await?;
        let gen_time = start.elapsed();
        
        println!(
            "  {} → [{:.3}, {:.3}, {:.3}, ...] ({}D) in {:.2}ms",
            concept,
            embedding[0], embedding[1], embedding[2],
            embedding.len(),
            gen_time.as_millis()
        );
    }
    
    // Compare embeddings
    println!("\n=== Semantic Similarity Analysis ===\n");
    
    let ai_embedding = neural_server.get_embedding("artificial intelligence").await?;
    let ml_embedding = neural_server.get_embedding("machine learning").await?;
    let physics_embedding = neural_server.get_embedding("quantum physics").await?;
    
    let ai_ml_similarity = cosine_similarity(&ai_embedding, &ml_embedding);
    let ai_physics_similarity = cosine_similarity(&ai_embedding, &physics_embedding);
    
    println!("Cosine similarity scores:");
    println!("  'artificial intelligence' vs 'machine learning': {:.3}", ai_ml_similarity);
    println!("  'artificial intelligence' vs 'quantum physics': {:.3}", ai_physics_similarity);
    println!("\n(Higher scores indicate more similar concepts)");
    
    // Model statistics
    println!("\n=== Model Statistics ===\n");
    
    let models = neural_server.list_models().await?;
    for model_id in models {
        if let Some(metadata) = neural_server.get_model_metadata(&model_id).await? {
            println!("{}:", model_id);
            println!("  Parameters: {:?}M", metadata.parameters_count / 1_000_000);
            println!("  Input dims: {}", metadata.input_dimensions);
            println!("  Output dims: {}", metadata.output_dimensions);
            if !metadata.accuracy_metrics.is_empty() {
                println!("  Metrics: {:?}", metadata.accuracy_metrics);
            }
        }
    }
    
    println!("\n=== Demo Complete ===");
    
    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm_a * norm_b)
}