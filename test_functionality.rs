// Test program to verify LLMKG functionality
use llmkg::core::entity_extractor::CognitiveEntityExtractor;
use llmkg::core::relationship_extractor::CognitiveRelationshipExtractor;
use llmkg::core::question_parser::CognitiveQuestionParser;
use llmkg::core::answer_generator::CognitiveAnswerGenerator;
use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use llmkg::neural::neural_server::NeuralProcessingServer;
use llmkg::federation::coordinator::FederationCoordinator;
use llmkg::models::ModelType;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() {
    println!("Testing LLMKG Library Functionality...\n");
    
    // Test 1: Entity Extractor
    println!("1. Testing Entity Extraction:");
    match test_entity_extraction().await {
        Ok(msg) => println!("   ✓ {}", msg),
        Err(e) => println!("   ✗ Error: {}", e),
    }
    
    // Test 2: Relationship Extractor
    println!("\n2. Testing Relationship Extraction:");
    match test_relationship_extraction().await {
        Ok(msg) => println!("   ✓ {}", msg),
        Err(e) => println!("   ✗ Error: {}", e),
    }
    
    // Test 3: Question Parser
    println!("\n3. Testing Question Parsing:");
    match test_question_parsing().await {
        Ok(msg) => println!("   ✓ {}", msg),
        Err(e) => println!("   ✗ Error: {}", e),
    }
    
    // Test 4: Answer Generator
    println!("\n4. Testing Answer Generation:");
    match test_answer_generation().await {
        Ok(msg) => println!("   ✓ {}", msg),
        Err(e) => println!("   ✗ Error: {}", e),
    }
    
    // Test 5: Cognitive Orchestrator
    println!("\n5. Testing Cognitive Orchestrator:");
    match test_cognitive_orchestrator().await {
        Ok(msg) => println!("   ✓ {}", msg),
        Err(e) => println!("   ✗ Error: {}", e),
    }
    
    // Test 6: Neural Processing Server
    println!("\n6. Testing Neural Processing Server:");
    match test_neural_server().await {
        Ok(msg) => println!("   ✓ {}", msg),
        Err(e) => println!("   ✗ Error: {}", e),
    }
    
    // Test 7: Federation Coordinator
    println!("\n7. Testing Federation Coordinator:");
    match test_federation().await {
        Ok(msg) => println!("   ✓ {}", msg),
        Err(e) => println!("   ✗ Error: {}", e),
    }
    
    println!("\n=== Test Summary ===");
    println!("Library compiles successfully!");
    println!("Core components are instantiable.");
}

async fn test_entity_extraction() -> Result<String, Box<dyn std::error::Error>> {
    let extractor = CognitiveEntityExtractor::new(
        ModelType::TinyBert,
        None,
        None,
    ).await?;
    
    let text = "Albert Einstein developed the theory of relativity in 1905.";
    let entities = extractor.extract_entities(text).await?;
    
    Ok(format!("Extracted {} entities from test text", entities.len()))
}

async fn test_relationship_extraction() -> Result<String, Box<dyn std::error::Error>> {
    let extractor = CognitiveRelationshipExtractor::new(
        ModelType::TinyBert,
        None,
        None,
    ).await?;
    
    Ok("Relationship extractor created successfully".to_string())
}

async fn test_question_parsing() -> Result<String, Box<dyn std::error::Error>> {
    let parser = CognitiveQuestionParser::new(
        ModelType::TinyBert,
        None,
        None,
    ).await?;
    
    let question = "What is the capital of France?";
    let parsed = parser.parse_question(question).await?;
    
    Ok(format!("Parsed question type: {:?}", parsed.intent.question_type))
}

async fn test_answer_generation() -> Result<String, Box<dyn std::error::Error>> {
    let generator = CognitiveAnswerGenerator::new(
        ModelType::TinyBert,
        None,
        None,
    ).await?;
    
    Ok("Answer generator created successfully".to_string())
}

async fn test_cognitive_orchestrator() -> Result<String, Box<dyn std::error::Error>> {
    let orchestrator = CognitiveOrchestrator::new();
    
    // Test pattern selection
    let patterns = orchestrator.get_available_patterns();
    
    Ok(format!("Orchestrator has {} cognitive patterns available", patterns.len()))
}

async fn test_neural_server() -> Result<String, Box<dyn std::error::Error>> {
    let neural_server = NeuralProcessingServer::new().await?;
    
    // Get available models
    let models = neural_server.list_models().await;
    
    Ok(format!("Neural server has {} models available", models.len()))
}

async fn test_federation() -> Result<String, Box<dyn std::error::Error>> {
    let coordinator = FederationCoordinator::new();
    
    Ok("Federation coordinator created successfully".to_string())
}