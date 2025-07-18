// String Interning Performance Demo
// Phase 4.3: String interning for properties to reduce memory usage

use llmkg::{
    StringInterner, 
    InternedEntityCollection,
    EntityKey,
    Result,
    intern_string,
    interner_stats,
    clear_interner,
};
use std::time::Instant;
use std::collections::HashMap;

fn generate_random_embedding(dimension: usize) -> Vec<f32> {
    (0..dimension).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect()
}

fn generate_realistic_properties(entity_id: usize) -> HashMap<String, String> {
    let mut props = HashMap::new();
    
    // Common property names that will be repeated across entities
    props.insert("name".to_string(), format!("Entity_{}", entity_id));
    props.insert("type".to_string(), match entity_id % 5 {
        0 => "Person".to_string(),
        1 => "Organization".to_string(),
        2 => "Location".to_string(),
        3 => "Product".to_string(),
        _ => "Concept".to_string(),
    });
    props.insert("status".to_string(), match entity_id % 3 {
        0 => "Active".to_string(),
        1 => "Inactive".to_string(),
        _ => "Pending".to_string(),
    });
    props.insert("category".to_string(), match entity_id % 4 {
        0 => "Primary".to_string(),
        1 => "Secondary".to_string(),
        2 => "Tertiary".to_string(),
        _ => "Other".to_string(),
    });
    props.insert("description".to_string(), format!("Description for entity {}", entity_id));
    props.insert("created_date".to_string(), "2024-01-01".to_string()); // Many will have same date
    props.insert("version".to_string(), "1.0.0".to_string()); // Repeated value
    
    props
}

fn main() -> Result<()> {
    println!("\n🏷️  === PHASE 4.3: STRING INTERNING DEMO ===");
    println!("Testing string interning for properties to reduce memory usage\n");

    let dimension = 384;
    let entity_count = 10000;
    
    // === Test 1: Basic String Interning ===
    println!("📊 Test 1: Basic String Interning");
    let start = Instant::now();
    
    let interner = StringInterner::new();
    
    // Test interning of repeated strings
    let repeated_strings = vec!["Active", "Inactive", "Pending", "Person", "Organization"];
    let mut intern_ids = Vec::new();
    
    for _ in 0..1000 {
        for string in &repeated_strings {
            intern_ids.push(interner.intern(string));
        }
    }
    
    let interning_time = start.elapsed();
    let stats = interner.stats();
    
    println!("✅ Interned {} references in {:.2}ms", intern_ids.len(), interning_time.as_millis());
    println!("📊 Interning Statistics:");
    println!("  • Unique strings: {}", stats.unique_strings);
    println!("  • Total references: {}", stats.total_references);
    println!("  • Deduplication ratio: {:.1}:1", stats.deduplication_ratio);
    println!("  • Memory saved: {} bytes", stats.memory_saved_bytes);

    // === Test 2: Memory Usage Comparison ===
    println!("\n💾 Test 2: Memory Usage Comparison");
    
    // Regular string storage
    let start = Instant::now();
    let mut regular_storage: Vec<HashMap<String, String>> = Vec::new();
    for i in 0..entity_count {
        regular_storage.push(generate_realistic_properties(i));
    }
    let regular_time = start.elapsed();
    
    // Calculate regular memory usage
    let regular_memory: usize = regular_storage.iter()
        .map(|props| {
            props.iter()
                .map(|(k, v)| k.len() + v.len() + 32) // 32 bytes overhead per entry
                .sum::<usize>()
        })
        .sum();
    
    // Interned string storage
    let start = Instant::now();
    let mut interned_collection = InternedEntityCollection::new();
    
    for i in 0..entity_count {
        let key = EntityKey::from_hash(&format!("entity_{}", i));
        let properties = generate_realistic_properties(i);
        let embedding = generate_random_embedding(dimension);
        
        interned_collection.add_entity(key, (i % 100) as u16, &properties, embedding);
    }
    let interned_time = start.elapsed();
    
    let interned_stats = interned_collection.stats();
    
    println!("✅ Regular storage: {:.2}ms", regular_time.as_millis());
    println!("✅ Interned storage: {:.2}ms", interned_time.as_millis());
    println!("\n📊 Memory Comparison:");
    println!("  • Regular storage: {} KB", regular_memory / 1024);
    println!("  • Interned storage: {} KB", interned_stats.properties_memory_bytes / 1024);
    println!("  • Memory saved: {} KB ({:.1}%)", 
             (regular_memory - interned_stats.properties_memory_bytes) / 1024,
             (1.0 - interned_stats.properties_memory_bytes as f32 / regular_memory as f32) * 100.0);
    println!("  • Compression ratio: {:.1}:1", 
             regular_memory as f32 / interned_stats.properties_memory_bytes as f32);

    // === Test 3: Property Access Performance ===
    println!("\n⚡ Test 3: Property Access Performance");
    
    let test_iterations = 10000;
    
    // Regular property access
    let start = Instant::now();
    let mut found_count = 0;
    for _ in 0..test_iterations {
        let idx = rand::random::<usize>() % regular_storage.len();
        if regular_storage[idx].get("type").is_some() {
            found_count += 1;
        }
    }
    let regular_access_time = start.elapsed();
    
    // Interned property access
    let start = Instant::now();
    let mut interned_found_count = 0;
    for _ in 0..test_iterations {
        let idx = rand::random::<usize>() % entity_count;
        let key = EntityKey::from_hash(&format!("entity_{}", idx));
        if let Some(entity) = interned_collection.entities.get(&key) {
            if entity.get_property(&interned_collection.interner, "type").is_some() {
                interned_found_count += 1;
            }
        }
    }
    let interned_access_time = start.elapsed();
    
    println!("✅ Regular access: {:.2}ms ({} found)", regular_access_time.as_millis(), found_count);
    println!("✅ Interned access: {:.2}ms ({} found)", interned_access_time.as_millis(), interned_found_count);
    println!("📊 Access performance: {:.1}x {} than regular",
             if interned_access_time < regular_access_time {
                 regular_access_time.as_micros() as f32 / interned_access_time.as_micros() as f32
             } else {
                 interned_access_time.as_micros() as f32 / regular_access_time.as_micros() as f32
             },
             if interned_access_time < regular_access_time { "faster" } else { "slower" });

    // === Test 4: Search by Property Performance ===
    println!("\n🔍 Test 4: Search by Property Performance");
    
    let start = Instant::now();
    let regular_results = regular_storage.iter()
        .enumerate()
        .filter(|(_, props)| props.get("type") == Some(&"Person".to_string()))
        .count();
    let regular_search_time = start.elapsed();
    
    let start = Instant::now();
    let interned_results = interned_collection.find_by_property("type", "Person");
    let interned_search_time = start.elapsed();
    
    println!("✅ Regular search: {:.2}ms ({} results)", regular_search_time.as_millis(), regular_results);
    println!("✅ Interned search: {:.2}ms ({} results)", interned_search_time.as_millis(), interned_results.len());

    // === Test 5: Global String Interner ===
    println!("\n🌐 Test 5: Global String Interner");
    
    clear_interner(); // Reset global interner
    
    let start = Instant::now();
    let mut global_ids = Vec::new();
    
    for i in 0..1000 {
        global_ids.push(intern_string(&format!("global_string_{}", i % 10))); // 10 unique strings, 1000 references
    }
    
    let global_time = start.elapsed();
    let global_stats = interner_stats();
    
    println!("✅ Global interning: {:.2}ms", global_time.as_millis());
    println!("📊 Global Stats: {} unique, {} references, {:.1}:1 deduplication",
             global_stats.unique_strings,
             global_stats.total_references,
             global_stats.deduplication_ratio);

    // === Test 6: Complex Property Operations ===
    println!("\n🔧 Test 6: Complex Property Operations");
    
    // Add tags and categories to entities
    let start = Instant::now();
    let mut tagged_entities = 0;
    
    for (key, entity) in interned_collection.entities.iter_mut() {
        if let Some(entity_type) = entity.get_property(&interned_collection.interner, "type") {
            match entity_type.as_str() {
                "Person" => {
                    entity.add_tag(&interned_collection.interner, "human");
                    entity.add_tag(&interned_collection.interner, "individual");
                    entity.set_category(&interned_collection.interner, "biology");
                },
                "Organization" => {
                    entity.add_tag(&interned_collection.interner, "business");
                    entity.add_tag(&interned_collection.interner, "institution");
                    entity.set_category(&interned_collection.interner, "organization");
                },
                "Location" => {
                    entity.add_tag(&interned_collection.interner, "geography");
                    entity.add_tag(&interned_collection.interner, "place");
                    entity.set_category(&interned_collection.interner, "spatial");
                },
                _ => {
                    entity.add_tag(&interned_collection.interner, "general");
                }
            }
            tagged_entities += 1;
        }
    }
    
    let tagging_time = start.elapsed();
    
    println!("✅ Tagged {} entities in {:.2}ms", tagged_entities, tagging_time.as_millis());
    
    // Search by tags
    let human_entities = interned_collection.find_by_tag("human");
    let business_entities = interned_collection.find_by_tag("business");
    
    println!("🏷️  Found {} entities with 'human' tag", human_entities.len());
    println!("🏷️  Found {} entities with 'business' tag", business_entities.len());

    // === Test 7: JSON Export Performance ===
    println!("\n📄 Test 7: JSON Export Performance");
    
    let start = Instant::now();
    let json_export = interned_collection.export_sample_json(10).map_err(|_| llmkg::GraphError::IndexCorruption)?;
    let export_time = start.elapsed();
    
    println!("✅ JSON export (10 entities): {:.2}ms", export_time.as_millis());
    println!("📄 JSON size: {} characters", json_export.len());

    // === Final Statistics ===
    println!("\n📋 === PHASE 4.3 SUMMARY ===");
    let final_stats = interned_collection.stats();
    
    println!("✅ String Interning Features Implemented:");
    println!("  • Thread-safe string interner with deduplication");
    println!("  • Interned property containers for entities");
    println!("  • Global string interner for convenience");
    println!("  • Complex property operations (tags, categories)");
    println!("  • Efficient search by property values");
    println!("  • JSON serialization with string reconstruction");
    
    println!("\n📊 Performance Results:");
    println!("  • Property memory savings: {:.1}%", 
             (1.0 - final_stats.properties_memory_bytes as f32 / regular_memory as f32) * 100.0);
    println!("  • String deduplication: {:.1}:1", final_stats.interner_stats.deduplication_ratio);
    println!("  • Entities processed: {}", final_stats.entity_count);
    println!("  • Unique strings: {}", final_stats.interner_stats.unique_strings);
    println!("  • Total string references: {}", final_stats.interner_stats.total_references);
    println!("  • Avg properties per entity: {:.1}", final_stats.avg_properties_per_entity);
    
    println!("\n💾 Memory Efficiency:");
    println!("  • Total memory: {} KB", final_stats.total_memory_bytes / 1024);
    println!("  • Properties memory: {} KB", final_stats.properties_memory_bytes / 1024);
    println!("  • Embeddings memory: {} KB", final_stats.embedding_memory_bytes / 1024);
    println!("  • String interner memory: {} bytes", final_stats.interner_stats.total_memory_bytes);
    
    println!("\n🎯 Phase 4.3 Completed Successfully!");
    println!("Ready for Phase 4.4: Zero-copy serialization\n");
    
    Ok(())
}