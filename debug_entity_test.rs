use llmkg::core::entity_extractor::{EntityExtractor, EntityType};

fn main() {
    let extractor = EntityExtractor::new();
    
    // Test 1: Theory of Relativity
    println!("=== Test 1: Theory of Relativity ===");
    let text1 = "Albert Einstein developed the Theory of Relativity in 1905.";
    let entities1 = extractor.extract_entities(text1);
    for entity in &entities1 {
        println!("Entity: '{}' -> {:?} (pos: {}-{})", entity.name, entity.entity_type, entity.start_pos, entity.end_pos);
    }
    
    // Test 2: University of California
    println!("\n=== Test 2: University of California ===");
    let text2 = "The University of California at Berkeley is located in the San Francisco Bay Area.";
    let entities2 = extractor.extract_entities(text2);
    for entity in &entities2 {
        println!("Entity: '{}' -> {:?} (pos: {}-{})", entity.name, entity.entity_type, entity.start_pos, entity.end_pos);
    }
    
    // Test 3: New York
    println!("\n=== Test 3: New York ===");
    let text3 = "New York City is different from New York State.";
    let entities3 = extractor.extract_entities(text3);
    for entity in &entities3 {
        println!("Entity: '{}' -> {:?} (pos: {}-{})", entity.name, entity.entity_type, entity.start_pos, entity.end_pos);
    }
    
    // Test 4: Quoted entities
    println!("\n=== Test 4: Quoted Entities ===");
    let text4 = "The concept of 'quantum entanglement' was described as 'spooky action at a distance'.";
    let entities4 = extractor.extract_entities(text4);
    for entity in &entities4 {
        println!("Entity: '{}' -> {:?} (pos: {}-{})", entity.name, entity.entity_type, entity.start_pos, entity.end_pos);
    }
}