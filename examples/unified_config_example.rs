//! Example demonstrating the unified models system with various state-of-the-art models

use llmkg::models::{
    ModelRegistry, ModelLoader, LoadingConfig, DeviceConfig, QuantizationConfig,
    smollm, tinyllama, openelm, minilm
};
use llmkg::model_utils;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 LLMKG Models Example - State-of-the-Art Small Language Models\n");

    // Create a model registry
    let registry = ModelRegistry::new();
    println!("📚 Model Registry initialized with {} models", registry.model_count());

    // Display registry statistics
    let stats = registry.get_statistics();
    println!("\n📊 Registry Statistics:");
    println!("  Total Models: {}", stats.total_models);
    println!("  Total Families: {}", stats.total_families);
    println!("  Total Parameters: {}", model_utils::format_parameters(stats.total_parameters));
    println!("  Average Parameters: {}", model_utils::format_parameters(stats.average_parameters));

    // Show models by family
    println!("\n🏠 Models by Family:");
    for family in ["SmolLM", "TinyLlama", "OpenELM", "MiniLM"] {
        let family_models = registry.list_models_by_family(family);
        println!("  {}: {} models", family, family_models.len());
        for model in family_models.iter().take(3) {
            println!("    - {} ({})", model.name, model_utils::format_parameters(model.parameters));
        }
    }

    // Show models by size category
    println!("\n📏 Models by Size Category:");
    for size in [llmkg::ModelSize::Tiny, llmkg::ModelSize::Small, llmkg::ModelSize::Medium] {
        let size_models = registry.list_models_by_size(size);
        println!("  {:?}: {} models", size, size_models.len());
    }

    // Get recommendations
    println!("\n⭐ Recommended Models:");
    let recommendations = registry.get_recommended_models();
    if let Some(best_overall) = &recommendations.best_overall {
        println!("  Best Overall: {} ({})", best_overall.name, model_utils::format_parameters(best_overall.parameters));
    }
    if let Some(most_efficient) = &recommendations.most_efficient {
        println!("  Most Efficient: {} ({})", most_efficient.name, model_utils::format_parameters(most_efficient.parameters));
    }
    if let Some(best_chat) = &recommendations.best_chat {
        println!("  Best for Chat: {} ({})", best_chat.name, model_utils::format_parameters(best_chat.parameters));
    }

    // Create models using builders
    println!("\n🏗️  Creating Models:");

    // SmolLM models
    let smol_135m = smollm::smollm_135m().build()?;
    println!("  ✅ SmolLM-135M: {} parameters", model_utils::format_parameters(smol_135m.parameter_count()));

    let smol_360m_instruct = smollm::smollm_360m_instruct().build()?;
    println!("  ✅ SmolLM-360M-Instruct: {} parameters (chat: {})", 
             model_utils::format_parameters(smol_360m_instruct.parameter_count()),
             smol_360m_instruct.metadata.capabilities.chat);

    // TinyLlama models
    let tinyllama_chat = tinyllama::tinyllama_1_1b_chat().build()?;
    println!("  ✅ TinyLlama-1.1B-Chat: {} parameters", model_utils::format_parameters(tinyllama_chat.parameter_count()));

    // OpenELM models
    let openelm_270m = openelm::openelm_270m().build()?;
    println!("  ✅ OpenELM-270M: {} parameters", model_utils::format_parameters(openelm_270m.parameter_count()));

    let openelm_450m_instruct = openelm::openelm_450m_instruct().build()?;
    println!("  ✅ OpenELM-450M-Instruct: {} parameters", model_utils::format_parameters(openelm_450m_instruct.parameter_count()));

    // MiniLM models
    let minilm_l6 = minilm::all_minilm_l6_v2().build()?;
    println!("  ✅ all-MiniLM-L6-v2: {} parameters (embeddings)", model_utils::format_parameters(minilm_l6.parameter_count()));

    // Model comparison
    println!("\n🔍 Model Comparison:");
    let comparison_models = vec![&smol_135m.metadata, &smol_360m_instruct.metadata, &openelm_270m.metadata];
    let comparisons = model_utils::compare_models(&comparison_models);
    
    for comparison in &comparisons {
        println!("\n  📋 {}", comparison.name);
        println!("    Parameters: {}", model_utils::format_parameters(comparison.parameters));
        println!("    Performance Tier: {:?}", comparison.performance_tier);
        println!("    Model Size: {}", model_utils::format_file_size(comparison.memory_requirements.model_size));
        println!("    Min RAM: {}", model_utils::format_file_size(comparison.memory_requirements.minimum_ram));
        println!("    CPU Speed: {:.1} tokens/sec", comparison.estimated_inference_speed.cpu_tokens_per_second);
        println!("    Mobile Compatible: {}", comparison.device_compatibility.mobile_phone);
    }

    // Model search
    println!("\n🔎 Model Search:");
    let search_results = registry.search_models("instruct");
    println!("  Found {} models matching 'instruct':", search_results.len());
    for model in search_results.iter().take(5) {
        println!("    - {} ({})", model.name, model.huggingface_id);
    }

    // Capability filtering
    println!("\n🎯 Models with Chat Capability:");
    let chat_models = registry.list_models_with_capability(|caps| caps.chat);
    for model in chat_models.iter().take(5) {
        println!("    - {} ({})", model.name, model_utils::format_parameters(model.parameters));
    }

    // Parameter range filtering
    println!("\n📊 Models in 100M-400M Parameter Range:");
    let mid_range_models = registry.list_models_in_parameter_range(100_000_000, 400_000_000);
    for model in &mid_range_models {
        println!("    - {} ({})", model.name, model_utils::format_parameters(model.parameters));
    }

    // Create a model loader
    println!("\n📥 Model Loader Example:");
    let loading_config = LoadingConfig {
        device: DeviceConfig::Cpu,
        quantization: QuantizationConfig::F16,
        batch_size: 1,
        cache_dir: None,
        trust_remote_code: false,
        revision: None,
    };

    let loader = ModelLoader::new(None, loading_config);
    println!("  📁 Cache directory: {}", loader.cache_dir().display());

    // Generate a detailed report for one model
    println!("\n📄 Detailed Model Report:");
    if let Some(model_metadata) = registry.get_model("HuggingFaceTB/SmolLM-360M") {
        let report = model_utils::generate_model_report(model_metadata);
        // Print first few lines of the report
        for line in report.lines().take(15) {
            println!("  {line}");
        }
        println!("  ... (truncated for brevity)");
    }

    // Model recommendation
    println!("\n💡 Model Recommendation:");
    let recommendation_req = model_utils::RecommendationRequest {
        max_parameters: Some(400_000_000),
        target_device: Some("laptop".to_string()),
        required_capabilities: vec!["chat".to_string(), "text_generation".to_string()],
        preferred_families: vec!["SmolLM".to_string()],
        max_memory_mb: Some(2048), // 2GB RAM limit
        min_inference_speed: None,
    };

    let all_models: Vec<_> = registry.list_models();
    if let Some(recommended) = model_utils::recommend_model(&all_models, &recommendation_req) {
        println!("  🎯 Recommended: {} ({})", recommended.name, model_utils::format_parameters(recommended.parameters));
        println!("     HuggingFace ID: {}", recommended.huggingface_id);
        println!("     Why: Matches size constraints, has chat capability, from preferred family");
    }

    println!("\n✨ Example completed successfully! All {} models are ready to use.", registry.model_count());
    println!("💡 These models range from {}M to {}B parameters, perfect for edge deployment!",
             model_utils::format_parameters(registry.get_smallest_model().unwrap().parameters),
             model_utils::format_parameters(registry.get_largest_model().unwrap_or(registry.get_smallest_model().unwrap()).parameters));

    Ok(())
}