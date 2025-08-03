use temporal_memory::{BranchManager, BranchConfig, ConsolidationState};
use std::time::Duration;

fn main() {
    println!("Branch Manager Demo");
    println!("==================");
    
    // Create branch manager with config
    let config = BranchConfig {
        max_age: Duration::from_secs(3600),
        max_divergence: 0.5,
        auto_consolidate_threshold: 0.8,
        detect_conflicts: true,
        auto_consolidation_check_interval: Duration::from_secs(300),
    };
    
    let manager = BranchManager::new(config);
    
    // Create some branches
    let branch1 = manager.create_branch("feature-1", None).unwrap();
    println!("Created branch: {}", branch1);
    
    let branch2 = manager.create_branch("feature-2", Some(branch1.clone())).unwrap();
    println!("Created child branch: {}", branch2);
    
    // Add some concepts
    let concept1 = uuid::Uuid::new_v4();
    let concept2 = uuid::Uuid::new_v4();
    
    manager.allocate_concept(&branch1, concept1).unwrap();
    manager.allocate_concept(&branch2, concept2).unwrap();
    println!("Allocated concepts to branches");
    
    // Get ancestry
    let ancestry = manager.get_ancestry(&branch2).unwrap();
    println!("\nAncestry of branch2:");
    for (i, id) in ancestry.iter().enumerate() {
        println!("  {}: {}", i, id);
    }
    
    // Get descendants
    let descendants = manager.get_descendants(&branch1).unwrap();
    println!("\nDescendants of branch1:");
    for id in &descendants {
        println!("  - {}", id);
    }
    
    // List all branches
    let all_branches = manager.list_branches();
    println!("\nAll branches ({} total):", all_branches.len());
    for id in &all_branches {
        println!("  - {}", id);
    }
    
    // Find branches by state
    let active_branches = manager.find_by_state(ConsolidationState::WorkingMemory);
    println!("\nActive branches ({}):", active_branches.len());
    for id in &active_branches {
        println!("  - {}", id);
    }
    
    // Get stats
    let stats = manager.stats();
    println!("\nStatistics:");
    println!("  Total branches: {}", stats.total_branches);
    println!("  Active branches: {}", stats.active_branches);
    println!("  Total concepts: {}", stats.total_concepts);
    
    println!("\nDemo completed successfully!");
}