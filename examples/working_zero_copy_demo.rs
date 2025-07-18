// Phase 4.4: Working Zero-Copy Serialization Demo
// Demonstrates the performance benefits of zero-copy serialization

use llmkg::core::types::EntityData;
use llmkg::storage::zero_copy::{ZeroCopySerializer, ZeroCopyDeserializer};
use std::time::Instant;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 LLMKG Zero-Copy Serialization Demo");
    println!("=====================================");
    
    // Initialize randomness
    let mut rng = rand::thread_rng();
    
    // Test different scales
    let test_scales = vec![
        (100, "Small scale"),
        (1_000, "Medium scale"),
        (10_000, "Large scale"),
    ];
    
    for (entity_count, scale_name) in test_scales {
        println!("\n📊 {} - {} entities", scale_name, entity_count);
        println!("{}", "─".repeat(50));
        
        run_zero_copy_demo(entity_count, &mut rng)?;
    }
    
    // Memory efficiency demonstration
    println!("\n💾 Memory Efficiency Demonstration");
    println!("══════════════════════════════════");
    demonstrate_memory_efficiency(&mut rng)?;
    
    // Performance comparison
    println!("\n⚡ Performance Characteristics");
    println!("{}", "═".repeat(40));
    demonstrate_performance_characteristics(&mut rng)?;
    
    Ok(())
}

fn run_zero_copy_demo(entity_count: usize, rng: &mut rand::rngs::ThreadRng) -> Result<(), Box<dyn std::error::Error>> {
    let embedding_dim = 96;
    
    // Generate test data
    println!("  🔄 Generating {} entities...", entity_count);
    let start = Instant::now();
    
    let entities: Vec<EntityData> = (0..entity_count)
        .map(|i| generate_realistic_entity(i as u32, embedding_dim, rng))
        .collect();
    
    let generation_time = start.elapsed();
    println!("  ✅ Generation: {:.2}ms", generation_time.as_millis());
    
    // Serialize with zero-copy
    println!("  🔄 Serializing...");
    let start = Instant::now();
    let mut serializer = ZeroCopySerializer::new();
    
    for entity in &entities {
        serializer.add_entity(entity, embedding_dim)?;
    }
    
    let serialized_data = serializer.finalize()?;
    let serialization_time = start.elapsed();
    
    println!("  ✅ Serialization: {:.2}ms ({:.0} entities/ms)", 
        serialization_time.as_millis(),
        entity_count as f64 / serialization_time.as_millis() as f64
    );
    
    // Deserialize with zero-copy
    println!("  🔄 Deserializing...");
    let start = Instant::now();
    let deserializer = unsafe { ZeroCopyDeserializer::new(&serialized_data)? };
    let deserialization_time = start.elapsed();
    
    println!("  ✅ Deserialization: {:.2}μs ({:.0} entities/μs)", 
        deserialization_time.as_micros(),
        entity_count as f64 / deserialization_time.as_micros() as f64
    );
    
    // Access performance test
    println!("  🔄 Testing access performance...");
    let start = Instant::now();
    let mut total_properties_len = 0usize;
    
    for i in 0..entity_count.min(1000) { // Limit to 1000 for reasonable test time
        if let Some(entity) = deserializer.get_entity(i as u32) {
            let properties = deserializer.get_entity_properties(entity);
            total_properties_len += properties.len();
        }
    }
    
    let access_time = start.elapsed();
    let tested_entities = entity_count.min(1000);
    
    println!("  ✅ Access performance: {:.2}μs ({:.0} entities/ms)", 
        access_time.as_micros(),
        tested_entities as f64 / access_time.as_millis() as f64
    );
    
    // Memory statistics
    let raw_entity_size = entities.iter()
        .map(|e| e.properties.len() + e.embedding.len() * 4 + 8) // rough estimate
        .sum::<usize>();
    
    let compression_ratio = raw_entity_size as f32 / serialized_data.len() as f32;
    let bytes_per_entity = serialized_data.len() as f32 / entity_count as f32;
    
    println!("  📈 Results:");
    println!("    • Serialized size: {:.2} MB", serialized_data.len() as f64 / (1024.0 * 1024.0));
    println!("    • Bytes per entity: {:.1}", bytes_per_entity);
    println!("    • Compression ratio: {:.2}:1", compression_ratio);
    println!("    • Accessed properties: {} chars", total_properties_len);
    
    // Performance rating
    let serialization_rate = entity_count as f64 / serialization_time.as_millis() as f64;
    if serialization_rate > 1000.0 && compression_ratio > 2.0 {
        println!("    🏆 Performance: EXCELLENT");
    } else if serialization_rate > 500.0 && compression_ratio > 1.5 {
        println!("    🥈 Performance: VERY GOOD");
    } else {
        println!("    🥉 Performance: GOOD");
    }
    
    Ok(())
}

fn demonstrate_memory_efficiency(rng: &mut rand::rngs::ThreadRng) -> Result<(), Box<dyn std::error::Error>> {
    let embedding_dim = 128;
    
    // Create entities with different property patterns
    println!("  Creating entities with varying data patterns...");
    
    let patterns = vec![
        ("Short properties", 20..50),
        ("Medium properties", 100..200),
        ("Long properties", 300..500),
    ];
    
    for (pattern_name, property_range) in patterns {
        let entity_count = 1000;
        let mut serializer = ZeroCopySerializer::new();
        
        let mut total_property_size = 0;
        for i in 0..entity_count {
            let property_size = rng.gen_range(property_range.clone());
            let properties = generate_random_text(property_size, rng);
            total_property_size += properties.len();
            
            let entity = EntityData {
                type_id: (i % 10) as u16,
                properties,
                embedding: (0..embedding_dim).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            };
            
            serializer.add_entity(&entity, embedding_dim)?;
        }
        
        let serialized_data = serializer.finalize()?;
        let raw_size = total_property_size + (entity_count * embedding_dim * 4);
        let compression_ratio = raw_size as f32 / serialized_data.len() as f32;
        
        println!("    • {}: {:.2}:1 compression, {:.1} bytes/entity", 
            pattern_name, compression_ratio, 
            serialized_data.len() as f32 / entity_count as f32);
    }
    
    Ok(())
}

fn demonstrate_performance_characteristics(rng: &mut rand::rngs::ThreadRng) -> Result<(), Box<dyn std::error::Error>> {
    let embedding_dim = 64;
    let entity_count = 5000;
    
    println!("  🔄 Creating {} entities for performance testing...", entity_count);
    
    let entities: Vec<EntityData> = (0..entity_count)
        .map(|i| EntityData {
            type_id: (i % 20) as u16,
            properties: format!("Performance test entity {} with additional data", i),
            embedding: (0..embedding_dim).map(|j| (i + j) as f32 / 1000.0).collect(),
        })
        .collect();
    
    // Serialization benchmark
    let start = Instant::now();
    let mut serializer = ZeroCopySerializer::new();
    for entity in &entities {
        serializer.add_entity(entity, embedding_dim)?;
    }
    let serialized_data = serializer.finalize()?;
    let serialization_time = start.elapsed();
    
    // Deserialization benchmark  
    let start = Instant::now();
    let deserializer = unsafe { ZeroCopyDeserializer::new(&serialized_data)? };
    let deserialization_time = start.elapsed();
    
    // Iteration benchmark
    let start = Instant::now();
    let mut entity_count_check = 0;
    for _entity in deserializer.iter_entities() {
        entity_count_check += 1;
    }
    let iteration_time = start.elapsed();
    
    // Random access benchmark
    let start = Instant::now();
    let mut access_count = 0;
    for _ in 0..1000 {
        let id = rng.gen_range(0..entity_count as u32);
        if deserializer.get_entity(id).is_some() {
            access_count += 1;
        }
    }
    let random_access_time = start.elapsed();
    
    println!("  📊 Performance Characteristics:");
    println!("    • Serialization: {:.2} ms ({:.0} entities/ms)", 
        serialization_time.as_millis(),
        entity_count as f64 / serialization_time.as_millis() as f64);
    println!("    • Deserialization: {:.2} μs ({:.0} entities/μs)", 
        deserialization_time.as_micros(),
        entity_count as f64 / deserialization_time.as_micros() as f64);
    println!("    • Full iteration: {:.2} ms ({:.0} entities/ms)", 
        iteration_time.as_millis(),
        entity_count as f64 / iteration_time.as_millis() as f64);
    println!("    • Random access: {:.2} ms ({:.0} lookups/ms)", 
        random_access_time.as_millis(),
        1000.0 / random_access_time.as_millis() as f64);
    println!("    • Access success rate: {:.1}%", 
        access_count as f64 / 1000.0 * 100.0);
    
    // Memory efficiency
    let data_size_mb = serialized_data.len() as f64 / (1024.0 * 1024.0);
    let bytes_per_entity = serialized_data.len() as f64 / entity_count as f64;
    
    println!("    • Total size: {:.2} MB", data_size_mb);
    println!("    • Memory efficiency: {:.1} bytes/entity", bytes_per_entity);
    
    // Verify iteration correctness
    assert_eq!(entity_count_check, entity_count);
    println!("    ✅ All {} entities verified through iteration", entity_count);
    
    Ok(())
}

fn generate_realistic_entity(id: u32, embedding_dim: usize, rng: &mut rand::rngs::ThreadRng) -> EntityData {
    let property_types = vec!["name", "description", "category", "metadata"];
    let base_properties = format!(r#"{{"id": {}, "timestamp": {}"#, id, rng.gen::<u64>());
    
    let mut properties = base_properties;
    for prop_type in &property_types {
        if rng.gen_bool(0.6) { // 60% chance to include each property
            let value_length = rng.gen_range(10..80);
            let value = generate_random_text(value_length, rng);
            properties.push_str(&format!(r#", "{}": "{}""#, prop_type, value));
        }
    }
    properties.push('}');
    
    // Generate embedding with some structure
    let base_value = (id % 10) as f32 / 10.0;
    let embedding: Vec<f32> = (0..embedding_dim)
        .map(|i| base_value + rng.gen_range(-0.2..0.2) + (i as f32 / embedding_dim as f32) * 0.1)
        .collect();
    
    EntityData {
        type_id: (id % 50) as u16 + 1, // 1-50 type range
        properties,
        embedding,
    }
}

fn generate_random_text(length: usize, rng: &mut rand::rngs::ThreadRng) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .-_";
    (0..length)
        .map(|_| CHARS[rng.gen_range(0..CHARS.len())] as char)
        .collect()
}