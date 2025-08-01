//! Utility functions for model management and operations

use super::ModelMetadata;
use serde::{Serialize, Deserialize};

/// Format parameter count in human-readable form
pub fn format_parameters(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{}M", params / 1_000_000)
    } else if params >= 1_000 {
        format!("{}K", params / 1_000)
    } else {
        params.to_string()
    }
}

/// Format token count in human-readable form
pub fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000_000_000 {
        format!("{:.1}T", tokens as f64 / 1_000_000_000_000.0)
    } else if tokens >= 1_000_000_000 {
        format!("{:.1}B", tokens as f64 / 1_000_000_000.0)
    } else if tokens >= 1_000_000 {
        format!("{}M", tokens / 1_000_000)
    } else if tokens >= 1_000 {
        format!("{}K", tokens / 1_000)
    } else {
        tokens.to_string()
    }
}

/// Format file size in human-readable form
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Calculate estimated model size on disk (rough estimation)
pub fn estimate_model_size(parameters: u64, precision: ModelPrecision) -> u64 {
    let bytes_per_param = match precision {
        ModelPrecision::Float32 => 4,
        ModelPrecision::Float16 => 2,
        ModelPrecision::BFloat16 => 2,
        ModelPrecision::Int8 => 1,
        ModelPrecision::Int4 => 1, // Approximation, actual is 0.5 bytes per param
    };
    
    // Add some overhead for tokenizer, config, etc. (roughly 10%)
    let model_size = parameters * bytes_per_param;
    (model_size as f64 * 1.1) as u64
}

/// Model precision types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelPrecision {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int4,
}

/// Memory requirements estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    pub model_size: u64,
    pub minimum_vram: u64,
    pub recommended_vram: u64,
    pub minimum_ram: u64,
    pub recommended_ram: u64,
}

/// Estimate memory requirements for a model
pub fn estimate_memory_requirements(parameters: u64, precision: ModelPrecision) -> MemoryRequirements {
    let model_size = estimate_model_size(parameters, precision);
    
    // Rule of thumb: need 2x model size for inference (weights + activations)
    let minimum_vram = model_size * 2;
    let recommended_vram = model_size * 3; // For better performance
    
    // RAM requirements (if not using GPU)
    let minimum_ram = model_size * 2;
    let recommended_ram = model_size * 4;
    
    MemoryRequirements {
        model_size,
        minimum_vram,
        recommended_vram,
        minimum_ram,
        recommended_ram,
    }
}

/// Performance tier based on model size
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PerformanceTier {
    UltraFast,  // < 50M params
    Fast,       // 50M - 200M params
    Balanced,   // 200M - 500M params
    Powerful,   // 500M+ params
}

/// Get performance tier for a model
pub fn get_performance_tier(parameters: u64) -> PerformanceTier {
    if parameters < 50_000_000 {
        PerformanceTier::UltraFast
    } else if parameters < 200_000_000 {
        PerformanceTier::Fast
    } else if parameters < 500_000_000 {
        PerformanceTier::Balanced
    } else {
        PerformanceTier::Powerful
    }
}

/// Device compatibility assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCompatibility {
    pub mobile_phone: bool,
    pub tablet: bool,
    pub laptop: bool,
    pub desktop: bool,
    pub edge_device: bool,
    pub recommended_devices: Vec<String>,
}

/// Assess device compatibility for a model
pub fn assess_device_compatibility(parameters: u64) -> DeviceCompatibility {
    let tier = get_performance_tier(parameters);
    
    match tier {
        PerformanceTier::UltraFast => DeviceCompatibility {
            mobile_phone: true,
            tablet: true,
            laptop: true,
            desktop: true,
            edge_device: true,
            recommended_devices: vec![
                "Modern smartphones".to_string(),
                "Tablets".to_string(),
                "Raspberry Pi 4+".to_string(),
                "Edge computing devices".to_string(),
            ],
        },
        PerformanceTier::Fast => DeviceCompatibility {
            mobile_phone: true,
            tablet: true,
            laptop: true,
            desktop: true,
            edge_device: true,
            recommended_devices: vec![
                "High-end smartphones".to_string(),
                "Tablets".to_string(),
                "Laptops".to_string(),
                "Desktop computers".to_string(),
            ],
        },
        PerformanceTier::Balanced => DeviceCompatibility {
            mobile_phone: false,
            tablet: true,
            laptop: true,
            desktop: true,
            edge_device: false,
            recommended_devices: vec![
                "Laptops with 8GB+ RAM".to_string(),
                "Desktop computers".to_string(),
                "High-end tablets".to_string(),
            ],
        },
        PerformanceTier::Powerful => DeviceCompatibility {
            mobile_phone: false,
            tablet: false,
            laptop: true,
            desktop: true,
            edge_device: false,
            recommended_devices: vec![
                "Desktop computers with GPU".to_string(),
                "Gaming laptops".to_string(),
                "Workstations".to_string(),
            ],
        },
    }
}

/// Model comparison metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub name: String,
    pub parameters: u64,
    pub memory_requirements: MemoryRequirements,
    pub performance_tier: PerformanceTier,
    pub device_compatibility: DeviceCompatibility,
    pub estimated_inference_speed: InferenceSpeed,
}

/// Inference speed estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSpeed {
    pub cpu_tokens_per_second: f64,
    pub gpu_tokens_per_second: f64,
    pub mobile_tokens_per_second: f64,
}

/// Estimate inference speed for a model
pub fn estimate_inference_speed(parameters: u64) -> InferenceSpeed {
    // These are rough estimates based on typical hardware
    let base_cpu_speed = 1000.0 / (parameters as f64 / 1_000_000.0).sqrt();
    let base_gpu_speed = base_cpu_speed * 5.0; // GPUs are typically 5x faster
    let base_mobile_speed = base_cpu_speed * 0.3; // Mobile devices are slower
    
    InferenceSpeed {
        cpu_tokens_per_second: base_cpu_speed.max(0.1),
        gpu_tokens_per_second: base_gpu_speed.max(0.5),
        mobile_tokens_per_second: base_mobile_speed.max(0.05),
    }
}

/// Create a detailed model comparison
pub fn compare_models(models: &[&ModelMetadata]) -> Vec<ModelComparison> {
    models.iter().map(|metadata| {
        let memory_req = estimate_memory_requirements(metadata.parameters, ModelPrecision::Float16);
        let performance_tier = get_performance_tier(metadata.parameters);
        let device_compat = assess_device_compatibility(metadata.parameters);
        let inference_speed = estimate_inference_speed(metadata.parameters);
        
        ModelComparison {
            name: metadata.name.clone(),
            parameters: metadata.parameters,
            memory_requirements: memory_req,
            performance_tier,
            device_compatibility: device_compat,
            estimated_inference_speed: inference_speed,
        }
    }).collect()
}

/// Model recommendation based on requirements
#[derive(Debug, Clone)]
pub struct RecommendationRequest {
    pub max_parameters: Option<u64>,
    pub target_device: Option<String>,
    pub required_capabilities: Vec<String>,
    pub preferred_families: Vec<String>,
    pub max_memory_mb: Option<u64>,
    pub min_inference_speed: Option<f64>,
}

/// Find the best model matching requirements
pub fn recommend_model<'a>(models: &'a [&'a ModelMetadata], requirements: &RecommendationRequest) -> Option<&'a ModelMetadata> {
    let mut candidates: Vec<_> = models.iter().cloned().collect();
    
    // Filter by max parameters
    if let Some(max_params) = requirements.max_parameters {
        candidates.retain(|model| model.parameters <= max_params);
    }
    
    // Filter by memory requirements
    if let Some(max_memory) = requirements.max_memory_mb {
        let max_memory_bytes = max_memory * 1024 * 1024;
        candidates.retain(|model| {
            let memory_req = estimate_memory_requirements(model.parameters, ModelPrecision::Float16);
            memory_req.minimum_ram <= max_memory_bytes
        });
    }
    
    // Filter by preferred families
    if !requirements.preferred_families.is_empty() {
        candidates.retain(|model| requirements.preferred_families.contains(&model.family));
    }
    
    // Score remaining candidates (higher is better)
    let mut scored_candidates: Vec<_> = candidates.iter().map(|model| {
        let mut score = 0.0;
        
        // Prefer models with required capabilities
        for capability in &requirements.required_capabilities {
            match capability.as_str() {
                "text_generation" if model.capabilities.text_generation => score += 10.0,
                "chat" if model.capabilities.chat => score += 10.0,
                "instruction_following" if model.capabilities.instruction_following => score += 10.0,
                "multilingual" if model.capabilities.multilingual => score += 5.0,
                "code_generation" if model.capabilities.code_generation => score += 5.0,
                "reasoning" if model.capabilities.reasoning => score += 5.0,
                _ => {}
            }
        }
        
        // Prefer more parameters (within limits) for better quality
        score += (model.parameters as f64 / 1_000_000.0).log10();
        
        (model, score)
    }).collect();
    
    // Sort by score (descending)
    scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    scored_candidates.first().map(|(model, _)| **model)
}

/// Generate a model summary report
pub fn generate_model_report(metadata: &ModelMetadata) -> String {
    let memory_req = estimate_memory_requirements(metadata.parameters, ModelPrecision::Float16);
    let performance_tier = get_performance_tier(metadata.parameters);
    let device_compat = assess_device_compatibility(metadata.parameters);
    let inference_speed = estimate_inference_speed(metadata.parameters);
    
    format!(
        r#"# {} Model Report

## Basic Information
- **Family**: {}
- **Parameters**: {} ({})
- **Size Category**: {:?}
- **Architecture**: {}
- **HuggingFace ID**: {}
- **License**: {}
- **Release Date**: {}

## Capabilities
- Text Generation: {}
- Instruction Following: {}
- Chat: {}
- Code Generation: {}
- Reasoning: {}
- Multilingual: {}

## Performance & Requirements
- **Performance Tier**: {:?}
- **Model Size**: {}
- **Minimum RAM**: {}
- **Recommended RAM**: {}
- **Minimum VRAM**: {}
- **Recommended VRAM**: {}

## Device Compatibility
- Mobile Phone: {}
- Tablet: {}
- Laptop: {}
- Desktop: {}
- Edge Device: {}

## Estimated Inference Speed
- CPU: {:.1} tokens/second
- GPU: {:.1} tokens/second
- Mobile: {:.1} tokens/second

## Recommended Devices
{}

## Description
{}
"#,
        metadata.name,
        metadata.family,
        format_parameters(metadata.parameters),
        metadata.parameters,
        metadata.size_category,
        metadata.architecture,
        metadata.huggingface_id,
        metadata.license,
        metadata.release_date,
        if metadata.capabilities.text_generation { "✓" } else { "✗" },
        if metadata.capabilities.instruction_following { "✓" } else { "✗" },
        if metadata.capabilities.chat { "✓" } else { "✗" },
        if metadata.capabilities.code_generation { "✓" } else { "✗" },
        if metadata.capabilities.reasoning { "✓" } else { "✗" },
        if metadata.capabilities.multilingual { "✓" } else { "✗" },
        performance_tier,
        format_file_size(memory_req.model_size),
        format_file_size(memory_req.minimum_ram),
        format_file_size(memory_req.recommended_ram),
        format_file_size(memory_req.minimum_vram),
        format_file_size(memory_req.recommended_vram),
        if device_compat.mobile_phone { "✓" } else { "✗" },
        if device_compat.tablet { "✓" } else { "✗" },
        if device_compat.laptop { "✓" } else { "✗" },
        if device_compat.desktop { "✓" } else { "✗" },
        if device_compat.edge_device { "✓" } else { "✗" },
        inference_speed.cpu_tokens_per_second,
        inference_speed.gpu_tokens_per_second,
        inference_speed.mobile_tokens_per_second,
        device_compat.recommended_devices.join(", "),
        metadata.description
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_parameters() {
        assert_eq!(format_parameters(1_000), "1K");
        assert_eq!(format_parameters(1_000_000), "1M");
        assert_eq!(format_parameters(135_000_000), "135M");
        assert_eq!(format_parameters(1_100_000_000), "1.1B");
        assert_eq!(format_parameters(500), "500");
    }

    #[test]
    fn test_format_tokens() {
        assert_eq!(format_tokens(1_000_000_000_000), "1.0T");
        assert_eq!(format_tokens(600_000_000_000), "600.0B");
        assert_eq!(format_tokens(1_000_000), "1M");
        assert_eq!(format_tokens(1_000), "1K");
        assert_eq!(format_tokens(500), "500");
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(1024), "1.0 KB");
        assert_eq!(format_file_size(1048576), "1.0 MB");
        assert_eq!(format_file_size(1073741824), "1.0 GB");
        assert_eq!(format_file_size(500), "500 B");
    }

    #[test]
    fn test_estimate_model_size() {
        let size_fp32 = estimate_model_size(100_000_000, ModelPrecision::Float32);
        let size_fp16 = estimate_model_size(100_000_000, ModelPrecision::Float16);
        let size_int8 = estimate_model_size(100_000_000, ModelPrecision::Int8);
        
        assert!(size_fp32 > size_fp16);
        assert!(size_fp16 > size_int8);
        assert_eq!(size_fp32, 440_000_000); // 100M * 4 bytes * 1.1 overhead
    }

    #[test]
    fn test_performance_tier() {
        assert_eq!(get_performance_tier(30_000_000), PerformanceTier::UltraFast);
        assert_eq!(get_performance_tier(135_000_000), PerformanceTier::Fast);
        assert_eq!(get_performance_tier(360_000_000), PerformanceTier::Balanced);
        assert_eq!(get_performance_tier(1_100_000_000), PerformanceTier::Powerful);
    }

    #[test]
    fn test_device_compatibility() {
        let compat_small = assess_device_compatibility(30_000_000);
        assert!(compat_small.mobile_phone);
        assert!(compat_small.edge_device);
        
        let compat_large = assess_device_compatibility(1_100_000_000);
        assert!(!compat_large.mobile_phone);
        assert!(!compat_large.edge_device);
        assert!(compat_large.desktop);
    }

    #[test]
    fn test_memory_estimation() {
        let memory_req = estimate_memory_requirements(135_000_000, ModelPrecision::Float16);
        assert!(memory_req.model_size > 0);
        assert!(memory_req.minimum_vram >= memory_req.model_size * 2);
        assert!(memory_req.recommended_vram >= memory_req.minimum_vram);
    }
}