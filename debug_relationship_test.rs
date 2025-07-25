use llmkg::core::entity_extractor::EntityExtractor;
use llmkg::core::relationship_extractor::RelationshipExtractor;

fn main() {
    let entity_extractor = EntityExtractor::new();
    let rel_extractor = RelationshipExtractor::new();
    
    let text = "The Eiffel Tower is located in Paris, France.";
    println!("Text: {}", text);
    
    let entities = entity_extractor.extract_entities(text);
    println!("\nEntities extracted:");
    for entity in &entities {
        println!("  '{}' ({:?}) at {}..{}", entity.name, entity.entity_type, entity.start_pos, entity.end_pos);
    }
    
    let relationships = rel_extractor.extract_relationships(text, &entities);
    println!("\nRelationships extracted:");
    for rel in &relationships {
        println!("  '{}' -[{}]-> '{}' (confidence: {})", rel.subject, rel.predicate, rel.object, rel.confidence);
    }
    
    // Check for the specific relationship the test is looking for
    let found = relationships.iter().any(|r| 
        r.subject == "Eiffel Tower" && 
        r.predicate == "located in" && 
        r.object == "Paris"
    );
    
    println!("\nLooking for: 'Eiffel Tower' -[located in]-> 'Paris'");
    println!("Found: {}", found);
    
    if !found {
        println!("\nDebug: Let's check what we have:");
        for rel in &relationships {
            if rel.subject == "Eiffel Tower" {
                println!("  Found Eiffel Tower relationship: '{}' -[{}]-> '{}'", rel.subject, rel.predicate, rel.object);
            }
        }
    }
}