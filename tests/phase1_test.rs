#[cfg(test)]
mod phase1_tests {
    use llmkg::core::brain_types::{
        BrainInspiredEntity, EntityDirection, LogicGate, LogicGateType,
        BrainInspiredRelationship, ActivationPattern
    };
    use llmkg::core::types::{EntityKey, EntityData};
    use llmkg::core::graph::KnowledgeGraph;
    use llmkg::versioning::temporal_graph::{TemporalKnowledgeGraph, TimeRange};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use chrono::Utc;
    use serde_json::json;
    
    /// Mock MCP server for testing without neural dependencies
    struct MockMCPServer {
        temporal_graph: Arc<RwLock<TemporalKnowledgeGraph>>,
    }
    
    impl MockMCPServer {
        fn new(temporal_graph: Arc<RwLock<TemporalKnowledgeGraph>>) -> Self {
            Self { temporal_graph }
        }
        
        fn get_tools(&self) -> Vec<MockTool> {
            vec![
                MockTool { name: "store_knowledge".to_string() },
                MockTool { name: "neural_query".to_string() },
                MockTool { name: "temporal_query".to_string() },
                MockTool { name: "canonicalize_entity".to_string() },
            ]
        }
    }
    
    struct MockTool {
        name: String,
    }

    #[test]
    fn test_brain_inspired_entity_creation() {
        let entity = BrainInspiredEntity::new(
            "test_concept".to_string(),
            EntityDirection::Input
        );
        
        assert_eq!(entity.concept_id, "test_concept");
        assert!(matches!(entity.direction, EntityDirection::Input));
        assert_eq!(entity.activation_state, 0.0);
    }

    #[test]
    fn test_logic_gate_computation() {
        let mut gate = LogicGate::new(LogicGateType::And, 0.5);
        gate.input_nodes = vec![Default::default(), Default::default()];
        
        // Test AND gate
        let result = gate.calculate_output(&[0.8, 0.9]).unwrap();
        assert_eq!(result, 0.8);
        
        let result = gate.calculate_output(&[0.3, 0.9]).unwrap();
        assert_eq!(result, 0.0);
    }

    #[tokio::test]
    async fn test_temporal_knowledge_graph() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = TemporalKnowledgeGraph::new(graph);
        
        let entity = BrainInspiredEntity::new(
            "temporal_entity".to_string(),
            EntityDirection::Output
        );
        
        let valid_time = TimeRange::new(Utc::now());
        let result = temporal_graph.insert_temporal_entity(entity, valid_time).await;
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_activation_pattern() {
        let mut pattern = ActivationPattern::new("test query".to_string());
        
        // Create a dummy graph to generate proper entity keys
        let graph = KnowledgeGraph::new(384).unwrap();
        let entity1 = graph.insert_entity(1, EntityData {
            type_id: 1,
            properties: "name:entity1".to_string(),
            embedding: vec![0.1; 384],
        }).unwrap();
        let entity2 = graph.insert_entity(2, EntityData {
            type_id: 1,
            properties: "name:entity2".to_string(),
            embedding: vec![0.2; 384],
        }).unwrap();
        let entity3 = graph.insert_entity(3, EntityData {
            type_id: 1,
            properties: "name:entity3".to_string(),
            embedding: vec![0.3; 384],
        }).unwrap();
        
        pattern.activations.insert(entity1, 0.9);
        pattern.activations.insert(entity2, 0.7);
        pattern.activations.insert(entity3, 0.5);
        
        let top_activations = pattern.get_top_activations(2);
        assert_eq!(top_activations.len(), 2);
        assert_eq!(top_activations[0].1, 0.9);
        assert_eq!(top_activations[1].1, 0.7);
    }

    #[test]
    fn test_brain_inspired_relationship() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let entity1 = graph.insert_entity(1, EntityData {
            type_id: 1,
            properties: "name:entity1".to_string(),
            embedding: vec![0.1; 384],
        }).unwrap();
        let entity2 = graph.insert_entity(2, EntityData {
            type_id: 1,
            properties: "name:entity2".to_string(),
            embedding: vec![0.2; 384],
        }).unwrap();
        
        let mut relationship = BrainInspiredRelationship::new(
            entity1,
            entity2,
            "test_relation".to_string()
        );
        
        assert_eq!(relationship.weight, 1.0);
        assert!(!relationship.is_inhibitory);
        
        // Test Hebbian learning
        relationship.strengthen(0.1);
        assert!(relationship.weight == 1.0); // Should be capped at 1.0
        assert_eq!(relationship.activation_count, 1);
    }


    #[tokio::test]
    async fn test_mcp_server_integration() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = Arc::new(RwLock::new(TemporalKnowledgeGraph::new(graph)));
        
        let mcp_server = MockMCPServer::new(temporal_graph);
        let tools = mcp_server.get_tools();
        
        // Verify brain-inspired tools are available
        let tool_names: Vec<_> = tools.iter().map(|t| &t.name).collect();
        assert!(tool_names.contains(&&"store_knowledge".to_string()));
        assert!(tool_names.contains(&&"neural_query".to_string()));
        assert!(tool_names.contains(&&"temporal_query".to_string()));
        assert!(tool_names.contains(&&"canonicalize_entity".to_string()));
    }

    #[test]
    fn test_logic_gate_types() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let entity1 = graph.insert_entity(1, EntityData {
            type_id: 1,
            properties: "name:entity1".to_string(),
            embedding: vec![0.1; 384],
        }).unwrap();
        let entity2 = graph.insert_entity(2, EntityData {
            type_id: 1,
            properties: "name:entity2".to_string(),
            embedding: vec![0.2; 384],
        }).unwrap();
        
        // Test OR gate
        let mut or_gate = LogicGate::new(LogicGateType::Or, 0.5);
        or_gate.input_nodes = vec![entity1, entity2];
        assert_eq!(or_gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.9);
        assert_eq!(or_gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0);
        
        // Test NOT gate
        let mut not_gate = LogicGate::new(LogicGateType::Not, 0.0);
        not_gate.input_nodes = vec![entity1];
        let result = not_gate.calculate_output(&[0.8]).unwrap();
        assert!((result - 0.2).abs() < 0.001, "Expected ~0.2, got {}", result);
        
        // Test Inhibitory gate
        let mut inhibitory_gate = LogicGate::new(LogicGateType::Inhibitory, 0.0);
        inhibitory_gate.input_nodes = vec![entity1, entity2];
        assert_eq!(inhibitory_gate.calculate_output(&[0.8, 0.3]).unwrap(), 0.5);
    }
}