use llmkg::core::entity_extractor::{EntityExtractor, EntityType};

fn main() {
    let text = "Marie Curie, born Maria Sklodowska in Warsaw, Poland, was a pioneering physicist and chemist. \
                She discovered the elements polonium and radium in 1898. Marie Curie was the first woman to \
                win a Nobel Prize and the only person to win Nobel Prizes in two different sciences.";
    
    let entity_extractor = EntityExtractor::new();
    let entities = entity_extractor.extract_entities(text);
    
    println!("Extracted entities:");
    for entity in &entities {
        println!("  '{}' -> {:?} (pos: {}-{})", entity.name, entity.entity_type, entity.start_pos, entity.end_pos);
    }
    
    // Check the specific assertions from the test
    println!("\nTesting specific assertions:");
    println!("Marie Curie as Person: {}", entities.iter().any(|e| e.name == "Marie Curie" && e.entity_type == EntityType::Person));
    println!("Contains Warsaw: {}", entities.iter().any(|e| e.name == "Warsaw"));
    println!("Contains Poland: {}", entities.iter().any(|e| e.name == "Poland"));
    println!("Contains Nobel Prize: {}", entities.iter().any(|e| e.name == "Nobel Prize"));
    println!("Contains 1898 as Time: {}", entities.iter().any(|e| e.name == "1898" && e.entity_type == EntityType::Time));
}