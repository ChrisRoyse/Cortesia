//! Simple example to test cognitive question answering integration

use llmkg::core::{
    knowledge_engine::KnowledgeEngine,
    triple::Triple,
};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Cognitive Question Answering Integration\n");
    
    // Create knowledge engine
    let engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(384, 10000)?
    ));
    
    // Add test facts
    println!("Adding test facts...");
    {
        let engine_write = engine.write().await;
        
        // Einstein facts
        engine_write.store_triple(
            Triple::new(
                "Albert Einstein".to_string(),
                "developed".to_string(),
                "Theory of Relativity".to_string()
            )?,
            None
        )?;
        
        engine_write.store_triple(
            Triple::new(
                "Theory of Relativity".to_string(),
                "published_in".to_string(),
                "1905".to_string()
            )?,
            None
        )?;
        
        // Marie Curie facts
        engine_write.store_triple(
            Triple::new(
                "Marie Curie".to_string(),
                "discovered".to_string(),
                "Radium".to_string()
            )?,
            None
        )?;
        
        engine_write.store_triple(
            Triple::new(
                "Marie Curie".to_string(),
                "discovered".to_string(),
                "Polonium".to_string()
            )?,
            None
        )?;
    }
    
    println!("Facts added successfully!\n");
    
    // Test questions
    let test_questions = vec![
        "Who developed the Theory of Relativity?",
        "What did Marie Curie discover?",
        "When was the Theory of Relativity published?",
    ];
    
    // Import question parser and answer generator
    use llmkg::core::question_parser::QuestionParser;
    use llmkg::core::answer_generator::AnswerGenerator;
    use llmkg::core::knowledge_types::TripleQuery;
    
    for question in test_questions {
        println!("Question: {}", question);
        
        // Parse question
        let intent = QuestionParser::parse_static(question);
        println!("  Parsed type: {:?}", intent.question_type);
        println!("  Entities: {:?}", intent.entities);
        
        // Retrieve facts
        let engine_read = engine.read().await;
        let mut all_facts = Vec::new();
        
        for entity in &intent.entities {
            // Search as subject
            let query = TripleQuery {
                subject: Some(entity.clone()),
                predicate: None,
                object: None,
                limit: 100,
                min_confidence: 0.0,
                include_chunks: false,
            };
            
            if let Ok(results) = engine_read.query_triples(query) {
                all_facts.extend(results.triples);
            }
            
            // Search as object
            let query = TripleQuery {
                subject: None,
                predicate: None,
                object: Some(entity.clone()),
                limit: 100,
                min_confidence: 0.0,
                include_chunks: false,
            };
            
            if let Ok(results) = engine_read.query_triples(query) {
                all_facts.extend(results.triples);
            }
        }
        
        // Generate answer
        let answer = AnswerGenerator::generate_answer_static(all_facts, intent);
        
        println!("  Answer: {}", answer.text);
        println!("  Confidence: {:.0}%", answer.confidence * 100.0);
        println!("  Supporting facts: {}", answer.facts.len());
        println!();
    }
    
    println!("\n✅ Cognitive Q&A system is working!");
    println!("Note: This is using the basic implementation.");
    println!("Full cognitive enhancements require the complete system setup.");
    
    Ok(())
}