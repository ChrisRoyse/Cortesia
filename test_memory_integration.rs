// Simple test to verify memory integration fixes
use std::collections::HashMap;
use std::time::Duration;

// Test basic compilation of types we fixed
#[allow(dead_code)]
fn test_memory_types() {
    // Test MemoryType enum
    let memory_type = llmkg::cognitive::MemoryType::WorkingMemory;
    
    // Test that we can create collections with memory types
    let mut utilization: HashMap<llmkg::cognitive::MemoryType, f32> = HashMap::new();
    utilization.insert(memory_type, 0.5);
    
    // Test Duration usage
    let duration = Duration::from_secs(30);
    
    println!("Memory integration types compile successfully");
    println!("Duration: {:?}", duration);
    println!("Utilization: {:?}", utilization);
}

#[allow(dead_code)]
fn main() {
    test_memory_types();
}