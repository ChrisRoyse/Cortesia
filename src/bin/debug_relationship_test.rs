use llmkg::core::entity_extractor::EntityExtractor;

fn main() {
    let extractor = EntityExtractor::new();
    
    // Test the failing relationship test
    println!("=== Relationship Test ===");
    let text = "The Eiffel Tower is located in Paris, France.";
    let entities = extractor.extract_entities(text);
    for entity in &entities {
        println!("Entity: '{}' -> {:?} (pos: {}-{})", entity.name, entity.entity_type, entity.start_pos, entity.end_pos);
    }
    
    // Test the other failing test
    println!("\n=== Complex Knowledge Test ===");
    let text2 = "Warsaw is the capital of Poland. The city has over 1.7 million inhabitants.";
    let entities2 = extractor.extract_entities(text2);
    for entity in &entities2 {
        println!("Entity: '{}' -> {:?} (pos: {}-{})", entity.name, entity.entity_type, entity.start_pos, entity.end_pos);
    }
}