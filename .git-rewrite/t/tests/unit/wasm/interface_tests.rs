//! WASM Interface Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::wasm::fast_interface::*;

#[cfg(test)]
mod interface_tests {
    use super::*;

    #[test]
    fn test_wasm_graph_operations() {
        let mut wasm_interface = WasmInterface::new();
        
        // Test entity creation through WASM
        let entity_id = wasm_interface.create_entity("test_entity", "Test Entity").unwrap();
        assert!(entity_id > 0);
        
        // Test entity retrieval
        let entity = wasm_interface.get_entity(entity_id).unwrap();
        assert_eq!(entity.name, "Test Entity");
        
        // Test relationship creation
        let entity2_id = wasm_interface.create_entity("test_entity_2", "Test Entity 2").unwrap();
        let rel_id = wasm_interface.create_relationship(entity_id, entity2_id, "connects", 1.0).unwrap();
        assert!(rel_id > 0);
    }

    #[test]
    fn test_wasm_memory_management() {
        let mut interface = WasmInterface::new();
        let initial_memory = interface.get_memory_usage();
        
        // Create many entities
        for i in 0..1000 {
            interface.create_entity(&format!("entity_{}", i), &format!("Entity {}", i)).unwrap();
        }
        
        let peak_memory = interface.get_memory_usage();
        assert!(peak_memory > initial_memory);
        
        // Clean up
        interface.cleanup();
        let final_memory = interface.get_memory_usage();
        
        // Memory should be released
        assert!(final_memory < peak_memory);
    }
}