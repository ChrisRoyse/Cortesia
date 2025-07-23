use std::fs;
use std::path::Path;

/// Simple validation script to check our embedding dimension fixes
fn main() {
    println!("🔍 Validating LLMKG Test Fixes");
    println!("================================");
    
    let mut issues_found = 0;
    let mut files_checked = 0;
    
    // Check key files for embedding dimension issues
    let test_files = vec![
        "tests/core/test_brain_advanced_ops.rs",
        "tests/cognitive/test_adaptive.rs", 
        "tests/core/test_graph_core.rs",
        "src/core/brain_enhanced_graph/brain_graph_core.rs",
        "src/core/brain_enhanced_graph/brain_relationship_manager.rs",
    ];
    
    for file_path in test_files {
        files_checked += 1;
        if let Ok(content) = fs::read_to_string(file_path) {
            println!("\n📁 Checking: {}", file_path);
            
            // Check for problematic patterns
            let mut file_issues = 0;
            
            if content.contains("vec![0.0; 64]") {
                println!("   ❌ Found 64D embedding vector");
                file_issues += 1;
            }
            
            if content.contains("vec![0.0; 128]") && !content.contains("// legacy") {
                println!("   ❌ Found 128D embedding vector (non-legacy)");
                file_issues += 1;
            }
            
            if content.contains("vec![0.0; 384]") {
                println!("   ❌ Found 384D embedding vector");
                file_issues += 1;
            }
            
            if content.contains("new(128)") && !content.contains("// legacy") {
                println!("   ❌ Found 128D graph initialization (non-legacy)");
                file_issues += 1;
            }
            
            // Check for good patterns
            if content.contains("vec![0.0; 96]") {
                println!("   ✅ Found 96D embedding vector");
            }
            
            if content.contains("new_for_test()") {
                println!("   ✅ Using standardized test graph creation");
            }
            
            if file_path.contains("brain_relationship_manager") {
                if content.contains("reset_all_activations") {
                    println!("   ✅ Found reset_all_activations method");
                }
                if content.contains("analyze_weight_distribution") {
                    println!("   ✅ Found analyze_weight_distribution method");
                }
                if content.contains("WeightDistribution") {
                    println!("   ✅ Found WeightDistribution struct");
                }
            }
            
            if file_path.contains("brain_graph_core") {
                if content.contains("serde::Serialize, serde::Deserialize") {
                    println!("   ✅ Found serialization traits");
                }
            }
            
            if file_issues == 0 {
                println!("   ✅ No issues found in this file");
            } else {
                issues_found += file_issues;
            }
        } else {
            println!("   ⚠️  Could not read file: {}", file_path);
        }
    }
    
    println!("\n🎯 Validation Summary");
    println!("====================");
    println!("Files checked: {}", files_checked);
    println!("Total issues found: {}", issues_found);
    
    if issues_found == 0 {
        println!("🎉 All validation checks passed!");
        println!("💡 The embedding dimension fixes appear to be properly applied.");
    } else {
        println!("⚠️  {} issues need attention", issues_found);
    }
}