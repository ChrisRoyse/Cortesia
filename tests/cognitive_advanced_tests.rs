use llmkg::cognitive::test_orchestrator::CognitiveOrchestrator;
use llmkg::cognitive::types::QueryContext;
use llmkg::graph::{Graph, Node, Edge, Property};
use llmkg::graph::operations::GraphOps;
use llmkg::graph::types::{NodeType, EdgeType, PropertyKey, PropertyValue};
use std::collections::HashMap;
use std::sync::Arc;

/// Create a complex knowledge graph for testing advanced cognitive reasoning
fn create_complex_knowledge_graph() -> Graph {
    let mut graph = Graph::new();
    
    // Create a multi-domain knowledge graph spanning:
    // 1. Scientific concepts (physics, chemistry, biology)
    // 2. Historical events and causality
    // 3. Technological innovations and dependencies
    // 4. Social structures and relationships
    // 5. Abstract mathematical concepts
    
    // Physics domain
    let einstein = graph.add_node(Node::new(
        "Einstein".to_string(),
        NodeType::Entity("Person".to_string()),
        PropertyKey::Custom("physicist".to_string()),
        PropertyValue::String("theoretical physicist".to_string()),
    ));
    
    let relativity = graph.add_node(Node::new(
        "General Relativity".to_string(),
        NodeType::Entity("Theory".to_string()),
        PropertyKey::Custom("field".to_string()),
        PropertyValue::String("physics".to_string()),
    ));
    
    let spacetime = graph.add_node(Node::new(
        "Spacetime".to_string(),
        NodeType::Entity("Concept".to_string()),
        PropertyKey::Custom("dimension".to_string()),
        PropertyValue::String("4D".to_string()),
    ));
    
    let gravity = graph.add_node(Node::new(
        "Gravity".to_string(),
        NodeType::Entity("Force".to_string()),
        PropertyKey::Custom("type".to_string()),
        PropertyValue::String("fundamental force".to_string()),
    ));
    
    // Chemistry domain
    let water = graph.add_node(Node::new(
        "Water".to_string(),
        NodeType::Entity("Molecule".to_string()),
        PropertyKey::Custom("formula".to_string()),
        PropertyValue::String("H2O".to_string()),
    ));
    
    let hydrogen = graph.add_node(Node::new(
        "Hydrogen".to_string(),
        NodeType::Entity("Element".to_string()),
        PropertyKey::Custom("atomic_number".to_string()),
        PropertyValue::Int(1),
    ));
    
    let oxygen = graph.add_node(Node::new(
        "Oxygen".to_string(),
        NodeType::Entity("Element".to_string()),
        PropertyKey::Custom("atomic_number".to_string()),
        PropertyValue::Int(8),
    ));
    
    // Biology domain
    let dna = graph.add_node(Node::new(
        "DNA".to_string(),
        NodeType::Entity("Molecule".to_string()),
        PropertyKey::Custom("type".to_string()),
        PropertyValue::String("nucleic acid".to_string()),
    ));
    
    let evolution = graph.add_node(Node::new(
        "Evolution".to_string(),
        NodeType::Entity("Process".to_string()),
        PropertyKey::Custom("mechanism".to_string()),
        PropertyValue::String("natural selection".to_string()),
    ));
    
    // Technology domain
    let computer = graph.add_node(Node::new(
        "Computer".to_string(),
        NodeType::Entity("Technology".to_string()),
        PropertyKey::Custom("type".to_string()),
        PropertyValue::String("electronic device".to_string()),
    ));
    
    let internet = graph.add_node(Node::new(
        "Internet".to_string(),
        NodeType::Entity("Network".to_string()),
        PropertyKey::Custom("scope".to_string()),
        PropertyValue::String("global".to_string()),
    ));
    
    let ai = graph.add_node(Node::new(
        "Artificial Intelligence".to_string(),
        NodeType::Entity("Field".to_string()),
        PropertyKey::Custom("goal".to_string()),
        PropertyValue::String("machine intelligence".to_string()),
    ));
    
    // Historical domain
    let wwii = graph.add_node(Node::new(
        "World War II".to_string(),
        NodeType::Entity("Event".to_string()),
        PropertyKey::Custom("period".to_string()),
        PropertyValue::String("1939-1945".to_string()),
    ));
    
    let manhattan = graph.add_node(Node::new(
        "Manhattan Project".to_string(),
        NodeType::Entity("Project".to_string()),
        PropertyKey::Custom("goal".to_string()),
        PropertyValue::String("atomic bomb".to_string()),
    ));
    
    // Mathematical concepts
    let calculus = graph.add_node(Node::new(
        "Calculus".to_string(),
        NodeType::Entity("Mathematics".to_string()),
        PropertyKey::Custom("branch".to_string()),
        PropertyValue::String("analysis".to_string()),
    ));
    
    let infinity = graph.add_node(Node::new(
        "Infinity".to_string(),
        NodeType::Entity("Concept".to_string()),
        PropertyKey::Custom("type".to_string()),
        PropertyValue::String("mathematical concept".to_string()),
    ));
    
    // Add complex relationships with different edge types and confidence scores
    
    // Physics relationships
    graph.add_edge(Edge::new(
        einstein,
        relativity,
        EdgeType::Custom("developed".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(1.0),
    ));
    
    graph.add_edge(Edge::new(
        relativity,
        spacetime,
        EdgeType::Custom("describes".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(0.95),
    ));
    
    graph.add_edge(Edge::new(
        relativity,
        gravity,
        EdgeType::Custom("explains".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(0.9),
    ));
    
    // Chemistry relationships
    graph.add_edge(Edge::new(
        water,
        hydrogen,
        EdgeType::Custom("contains".to_string()),
        PropertyKey::Custom("count".to_string()),
        PropertyValue::Int(2),
    ));
    
    graph.add_edge(Edge::new(
        water,
        oxygen,
        EdgeType::Custom("contains".to_string()),
        PropertyKey::Custom("count".to_string()),
        PropertyValue::Int(1),
    ));
    
    // Cross-domain relationships
    graph.add_edge(Edge::new(
        einstein,
        manhattan,
        EdgeType::Custom("contributed_to".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(0.7),
    ));
    
    graph.add_edge(Edge::new(
        manhattan,
        wwii,
        EdgeType::Custom("during".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(1.0),
    ));
    
    graph.add_edge(Edge::new(
        computer,
        ai,
        EdgeType::Custom("enables".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(0.95),
    ));
    
    graph.add_edge(Edge::new(
        ai,
        calculus,
        EdgeType::Custom("uses".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(0.8),
    ));
    
    // Inhibitory relationships (contradictions)
    let classical_physics = graph.add_node(Node::new(
        "Classical Physics".to_string(),
        NodeType::Entity("Theory".to_string()),
        PropertyKey::Custom("field".to_string()),
        PropertyValue::String("physics".to_string()),
    ));
    
    graph.add_edge(Edge::new(
        relativity,
        classical_physics,
        EdgeType::Custom("contradicts".to_string()),
        PropertyKey::Custom("aspect".to_string()),
        PropertyValue::String("absolute time".to_string()),
    ));
    
    // Temporal relationships
    let newton = graph.add_node(Node::new(
        "Newton".to_string(),
        NodeType::Entity("Person".to_string()),
        PropertyKey::Custom("physicist".to_string()),
        PropertyValue::String("classical physicist".to_string()),
    ));
    
    graph.add_edge(Edge::new(
        newton,
        calculus,
        EdgeType::Custom("invented".to_string()),
        PropertyKey::Custom("year".to_string()),
        PropertyValue::Int(1666),
    ));
    
    graph.add_edge(Edge::new(
        newton,
        einstein,
        EdgeType::Custom("preceded".to_string()),
        PropertyKey::Custom("years".to_string()),
        PropertyValue::Int(200),
    ));
    
    // Hierarchical relationships
    let physics = graph.add_node(Node::new(
        "Physics".to_string(),
        NodeType::Entity("Field".to_string()),
        PropertyKey::Custom("type".to_string()),
        PropertyValue::String("natural science".to_string()),
    ));
    
    graph.add_edge(Edge::new(
        relativity,
        physics,
        EdgeType::IsA,
        PropertyKey::Confidence,
        PropertyValue::Float(1.0),
    ));
    
    graph.add_edge(Edge::new(
        classical_physics,
        physics,
        EdgeType::IsA,
        PropertyKey::Confidence,
        PropertyValue::Float(1.0),
    ));
    
    // Abstract relationships
    graph.add_edge(Edge::new(
        infinity,
        calculus,
        EdgeType::Custom("fundamental_to".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(0.9),
    ));
    
    graph.add_edge(Edge::new(
        evolution,
        dna,
        EdgeType::Custom("operates_on".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(0.85),
    ));
    
    // Causal chains
    let industrial_revolution = graph.add_node(Node::new(
        "Industrial Revolution".to_string(),
        NodeType::Entity("Event".to_string()),
        PropertyKey::Custom("period".to_string()),
        PropertyValue::String("1760-1840".to_string()),
    ));
    
    graph.add_edge(Edge::new(
        industrial_revolution,
        computer,
        EdgeType::Custom("led_to".to_string()),
        PropertyKey::Custom("indirectly".to_string()),
        PropertyValue::Bool(true),
    ));
    
    graph.add_edge(Edge::new(
        computer,
        internet,
        EdgeType::Custom("enabled".to_string()),
        PropertyKey::Confidence,
        PropertyValue::Float(1.0),
    ));
    
    graph
}

#[cfg(test)]
mod advanced_cognitive_tests {
    use super::*;
    
    #[test]
    fn test_multi_hop_causal_reasoning() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "What historical events indirectly led to the development of artificial intelligence?";
        let context = QueryContext {
            domain: Some("history and technology".to_string()),
            confidence_threshold: 0.6,
            max_depth: Some(5),
            required_evidence: Some(2),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        // Should trace: Industrial Revolution -> Computer -> AI
        // And possibly: WWII -> Manhattan Project -> Computer -> AI
        assert!(result.confidence > 0.7);
        assert!(result.answer.contains("Industrial Revolution") || 
                result.answer.contains("World War II"));
        assert!(!result.reasoning_path.is_empty());
    }
    
    #[test]
    fn test_contradiction_detection_and_resolution() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "How does Einstein's relativity relate to Newton's classical physics?";
        let context = QueryContext {
            domain: Some("physics".to_string()),
            confidence_threshold: 0.5,
            max_depth: Some(3),
            required_evidence: Some(2),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        // Should detect the contradiction and explain the relationship
        assert!(result.answer.contains("contradict") || 
                result.answer.contains("supersede") ||
                result.answer.contains("replace"));
        assert!(result.confidence > 0.8);
    }
    
    #[test]
    fn test_cross_domain_analogy_reasoning() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "What is the DNA of the internet?";
        let context = QueryContext {
            domain: None, // Should infer cross-domain
            confidence_threshold: 0.4,
            max_depth: Some(4),
            required_evidence: Some(1),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        // Should make an analogy between biological DNA and internet protocols/structure
        assert!(result.pattern_used.contains("Lateral") || 
                result.pattern_used.contains("Abstract"));
        assert!(result.confidence > 0.5);
    }
    
    #[test]
    fn test_temporal_reasoning_with_precedence() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "Who came first, Newton or Einstein, and by how much?";
        let context = QueryContext {
            domain: Some("history of science".to_string()),
            confidence_threshold: 0.8,
            max_depth: Some(2),
            required_evidence: Some(1),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        assert!(result.answer.contains("Newton"));
        assert!(result.answer.contains("200") || result.answer.contains("preceded"));
        assert!(result.confidence > 0.9);
    }
    
    #[test]
    fn test_hierarchical_inheritance_reasoning() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "What scientific theories belong to physics?";
        let context = QueryContext {
            domain: Some("science".to_string()),
            confidence_threshold: 0.7,
            max_depth: Some(3),
            required_evidence: Some(2),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        assert!(result.answer.contains("General Relativity"));
        assert!(result.answer.contains("Classical Physics"));
        assert!(result.pattern_used.contains("Systems"));
    }
    
    #[test]
    fn test_compositional_reasoning() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "What elements make up water and in what proportions?";
        let context = QueryContext {
            domain: Some("chemistry".to_string()),
            confidence_threshold: 0.9,
            max_depth: Some(2),
            required_evidence: Some(2),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        assert!(result.answer.contains("Hydrogen"));
        assert!(result.answer.contains("Oxygen"));
        assert!(result.answer.contains("2") || result.answer.contains("two"));
        assert!(result.confidence > 0.95);
    }
    
    #[test]
    fn test_abstract_pattern_extraction() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "What patterns exist in scientific revolutions?";
        let context = QueryContext {
            domain: Some("history of science".to_string()),
            confidence_threshold: 0.5,
            max_depth: Some(4),
            required_evidence: Some(2),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        // Should identify patterns like: new theory contradicts old, paradigm shifts
        assert!(result.pattern_used.contains("Abstract"));
        assert!(result.reasoning_path.len() >= 2);
    }
    
    #[test]
    fn test_ensemble_reasoning_complex_query() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "How did Einstein's work influence modern technology, considering both direct and indirect impacts?";
        let context = QueryContext {
            domain: None, // Should use multiple domains
            confidence_threshold: 0.6,
            max_depth: Some(5),
            required_evidence: Some(3),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        // Should use ensemble of patterns
        assert!(result.pattern_used.contains("Ensemble"));
        // Should find multiple paths of influence
        assert!(!result.reasoning_path.is_empty());
        assert!(result.confidence > 0.7);
    }
    
    #[test]
    fn test_creative_hypothesis_generation() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "What would happen if we applied evolutionary principles to internet protocols?";
        let context = QueryContext {
            domain: None,
            confidence_threshold: 0.3, // Lower threshold for creative queries
            max_depth: Some(4),
            required_evidence: Some(1),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        // Should use lateral or divergent thinking
        assert!(result.pattern_used.contains("Lateral") || 
                result.pattern_used.contains("Divergent"));
        // Should generate creative connections
        assert!(!result.answer.is_empty());
    }
    
    #[test]
    fn test_meta_reasoning_about_knowledge() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "What areas of knowledge in this graph are most interconnected?";
        let context = QueryContext {
            domain: None,
            confidence_threshold: 0.5,
            max_depth: Some(6),
            required_evidence: Some(3),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        // Should identify physics or technology as highly connected
        assert!(result.pattern_used.contains("Systems") || 
                result.pattern_used.contains("Abstract"));
        assert!(result.answer.contains("Physics") || 
                result.answer.contains("Technology"));
    }
    
    #[test]
    fn test_counterfactual_reasoning() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "What if Einstein had not developed relativity theory?";
        let context = QueryContext {
            domain: Some("physics".to_string()),
            confidence_threshold: 0.4,
            max_depth: Some(4),
            required_evidence: Some(1),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        // Should engage in hypothetical reasoning
        assert!(result.pattern_used.contains("Adaptive") || 
                result.pattern_used.contains("Lateral"));
        assert!(!result.answer.is_empty());
    }
    
    #[test]
    fn test_implicit_knowledge_inference() {
        let graph = Arc::new(create_complex_knowledge_graph());
        let orchestrator = CognitiveOrchestrator::new(graph.clone());
        
        let query = "Can water exist without hydrogen?";
        let context = QueryContext {
            domain: Some("chemistry".to_string()),
            confidence_threshold: 0.9,
            max_depth: Some(2),
            required_evidence: Some(1),
            reasoning_trace: true,
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        
        // Should infer from H2O composition
        assert!(result.answer.contains("No") || result.answer.contains("cannot"));
        assert!(result.confidence > 0.9);
    }
}

#[test]
fn test_performance_under_load() {
    use std::time::Instant;
    
    let graph = Arc::new(create_complex_knowledge_graph());
    let orchestrator = CognitiveOrchestrator::new(graph.clone());
    
    let queries = vec![
        "What is the relationship between Einstein and modern AI?",
        "How does water relate to evolution?",
        "What mathematical concepts are fundamental to physics?",
        "Which came first, the computer or the internet?",
        "What connects Newton to artificial intelligence?",
    ];
    
    let start = Instant::now();
    
    for query in queries {
        let context = QueryContext {
            domain: None,
            confidence_threshold: 0.5,
            max_depth: Some(4),
            required_evidence: Some(2),
            reasoning_trace: false, // Disable for performance
        };
        
        let result = orchestrator.process_query(query, context).unwrap();
        assert!(!result.answer.is_empty());
    }
    
    let elapsed = start.elapsed();
    
    // Should complete all queries in reasonable time
    assert!(elapsed.as_secs() < 5, "Queries took too long: {:?}", elapsed);
}

#[test]
fn test_error_handling_and_recovery() {
    let graph = Arc::new(create_complex_knowledge_graph());
    let orchestrator = CognitiveOrchestrator::new(graph.clone());
    
    // Test with impossible query
    let query = "What is the favorite color of General Relativity?";
    let context = QueryContext {
        domain: Some("physics".to_string()),
        confidence_threshold: 0.9,
        max_depth: Some(2),
        required_evidence: Some(3),
        reasoning_trace: true,
    };
    
    let result = orchestrator.process_query(query, context);
    
    // Should handle gracefully
    match result {
        Ok(r) => assert!(r.confidence < 0.3), // Very low confidence
        Err(_) => assert!(true), // Or error is fine
    }
}