use llmkg::core::entity_extractor::EntityExtractor;

fn main() {
    let extractor = EntityExtractor::new();
    
    // Test 1: Organization entity extraction
    println!("Test 1: Organization entity extraction");
    let text1 = "Microsoft Corporation partnered with OpenAI Inc to develop advanced AI systems.";
    let entities1 = extractor.extract_entities(text1);
    println!("Text: {}", text1);
    println!("Extracted entities:");
    for e in &entities1 {
        println!("  - '{}' ({:?}) at pos {}-{}", e.name, e.entity_type, e.start_pos, e.end_pos);
    }
    println!();
    
    // Test 2: No overlapping entities
    println!("Test 2: No overlapping entities");
    let text2 = "New York City is different from New York State.";
    let entities2 = extractor.extract_entities(text2);
    println!("Text: {}", text2);
    println!("Extracted entities:");
    for e in &entities2 {
        println!("  - '{}' ({:?}) at pos {}-{}", e.name, e.entity_type, e.start_pos, e.end_pos);
    }
    let ny_entities: Vec<_> = entities2.iter()
        .filter(|e| e.name.contains("New York"))
        .collect();
    println!("New York entities count: {}", ny_entities.len());
    println!();
    
    // Test 3: Complex entities with connectors
    println!("Test 3: Complex entities with connectors");
    let text3 = "The University of California at Berkeley is located in the San Francisco Bay Area.";
    let entities3 = extractor.extract_entities(text3);
    println!("Text: {}", text3);
    println!("Extracted entities:");
    for e in &entities3 {
        println!("  - '{}' ({:?}) at pos {}-{}", e.name, e.entity_type, e.start_pos, e.end_pos);
    }
}