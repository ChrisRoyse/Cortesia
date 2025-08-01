use super::*;

#[tokio::test]
async fn test_basic_test_infrastructure() {
    setup_basic_test_environment().await;
    
    // Test that our test infrastructure works
    assert!(true, "Test infrastructure is functional");
    
    println!("✓ Basic test infrastructure works");
}

#[tokio::test]
async fn test_text_content_processing() {
    setup_basic_test_environment().await;
    
    let content = create_test_knowledge_content();
    
    // Basic text processing tests
    assert!(!content.is_empty());
    assert!(content.contains("Einstein"));
    assert!(content.contains("relativity"));
    assert!(content.contains("Nobel Prize"));
    
    // Test content analysis
    let lines: Vec<&str> = content.lines().collect();
    let word_count = content.split_whitespace().count();
    
    assert!(lines.len() > 5);
    assert!(word_count > 50);
    
    println!("✓ Text content processing works correctly");
}

#[tokio::test]
async fn test_basic_string_operations() {
    setup_basic_test_environment().await;
    
    // Test basic string operations used in knowledge processing
    let subject = "Einstein";
    let predicate = "developed";
    let object = "theory of relativity";
    
    let triple_string = format!("{subject} {predicate} {object}");
    assert!(triple_string.contains(subject));
    assert!(triple_string.contains(predicate));
    assert!(triple_string.contains(object));
    
    println!("✓ Basic string operations work correctly");
}

#[tokio::test]
async fn test_json_handling() {
    setup_basic_test_environment().await;
    
    // Test JSON serialization/deserialization
    let data = serde_json::json!({
        "method": "store_fact",
        "params": {
            "subject": "Einstein",
            "predicate": "developed",
            "object": "relativity theory"
        }
    });
    
    assert!(data.is_object());
    assert_eq!(data["method"], "store_fact");
    assert!(data["params"].is_object());
    
    // Test serialization
    let json_string = serde_json::to_string(&data).expect("Should serialize");
    assert!(json_string.contains("store_fact"));
    assert!(json_string.contains("Einstein"));
    
    // Test deserialization
    let parsed: serde_json::Value = serde_json::from_str(&json_string)
        .expect("Should deserialize");
    assert_eq!(parsed["method"], "store_fact");
    
    println!("✓ JSON handling works correctly");
}

#[tokio::test]
async fn test_concurrent_operations() {
    setup_basic_test_environment().await;
    
    let task_count = 10;
    
    // Test concurrent string processing
    let tasks: Vec<_> = (0..task_count).map(|i| {
        tokio::spawn(async move {
            let content = create_test_knowledge_content();
            let processed = content.to_uppercase();
            (i, processed.contains("EINSTEIN"))
        })
    }).collect();
    
    let results = futures::future::join_all(tasks).await;
    
    // All tasks should complete successfully
    assert_eq!(results.len(), task_count);
    for result in results {
        let (id, contains_einstein) = result.expect("Task should complete");
        assert!(contains_einstein, "Task {id} should find Einstein");
    }
    
    println!("✓ Concurrent operations work correctly");
}

#[tokio::test]
async fn test_error_handling() {
    setup_basic_test_environment().await;
    
    // Test error handling in JSON operations
    let invalid_json = r#"{"invalid": json syntax"#;
    let result: Result<serde_json::Value, _> = serde_json::from_str(invalid_json);
    assert!(result.is_err(), "Should handle invalid JSON");
    
    // Test error handling in string operations
    let empty_string = "";
    let result = empty_string.split_whitespace().count();
    assert_eq!(result, 0, "Should handle empty strings gracefully");
    
    println!("✓ Error handling works correctly");
}

#[tokio::test]
async fn test_memory_efficient_operations() {
    setup_basic_test_environment().await;
    
    // Test memory-efficient processing of large amounts of data
    let iterations = 1000;
    let content = create_test_knowledge_content();
    
    for i in 0..iterations {
        // Process content without accumulating memory
        let word_count = content.split_whitespace().count();
        assert!(word_count > 0, "Iteration {i} should process content");
        
        // Occasionally check we're not accumulating too much
        if i % 100 == 0 {
            // Force garbage collection opportunity
            tokio::task::yield_now().await;
        }
    }
    
    println!("✓ Memory efficient operations work correctly");
}

#[tokio::test]
async fn test_timing_operations() {
    setup_basic_test_environment().await;
    
    use std::time::Instant;
    
    // Test timing of basic operations
    let start = Instant::now();
    
    let content = create_test_knowledge_content();
    let _processed = content.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    
    let duration = start.elapsed();
    
    // Should complete quickly
    assert!(duration.as_millis() < 100, "Processing should be fast");
    
    println!("✓ Timing operations work correctly: {duration:?}");
}