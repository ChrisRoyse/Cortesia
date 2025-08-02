use llmkg::enhanced_knowledge_storage::production::caching::*;
use std::time::Duration;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize the multi-level cache system
    let cache = CacheConfigBuilder::new()
        .l1_capacity(1000)                     // 1000 entries in L1
        .l1_max_bytes(10 * 1024 * 1024)        // 10MB for L1
        .l2_cache_dir("./cache_example")       // L2 cache directory
        .l2_max_bytes(100 * 1024 * 1024)       // 100MB for L2
        .write_strategy(WriteStrategy::WriteThrough)
        .build()
        .await?;

    println!("Multi-level caching system initialized successfully!");

    // Example 1: Basic caching operations
    println!("\n=== Basic Caching Operations ===");
    
    // Store some data
    cache.put("user:123".to_string(), "Alice's Profile Data".to_string(), None).await;
    cache.put("document:456".to_string(), "Important Document Content".to_string(), 
              Some(Duration::from_secs(3600))).await; // 1 hour TTL

    // Retrieve data
    if let Some(user_data) = cache.get::<String>("user:123").await {
        println!("Retrieved user data: {}", user_data);
    }

    if let Some(doc_data) = cache.get::<String>("document:456").await {
        println!("Retrieved document: {}", doc_data);
    }

    // Example 2: Cache statistics
    println!("\n=== Cache Statistics ===");
    let stats = cache.get_statistics().await;
    println!("L1 Cache: {} entries, {} bytes", stats.l1_entry_count, stats.l1_size_bytes);
    println!("L2 Cache: {} entries, {} bytes", stats.l2_entry_count, stats.l2_size_bytes);
    println!("Cache hit rate: {:.2}%", stats.hit_rate() * 100.0);

    // Example 3: Cache warming
    println!("\n=== Cache Warming ===");
    let warmup_data = vec![
        ("session:789".to_string(), b"Session Data 1".to_vec()),
        ("session:790".to_string(), b"Session Data 2".to_vec()),
        ("session:791".to_string(), b"Session Data 3".to_vec()),
    ];
    
    cache.warm_cache(warmup_data).await;
    println!("Cache warmed with 3 session entries");

    // Example 4: Pattern-based invalidation
    println!("\n=== Pattern-based Cache Invalidation ===");
    
    // Add more data
    cache.put("temp:123".to_string(), "Temporary data 1".to_string(), None).await;
    cache.put("temp:456".to_string(), "Temporary data 2".to_string(), None).await;
    cache.put("permanent:789".to_string(), "Permanent data".to_string(), None).await;

    // Invalidate all temporary data
    let invalidated = cache.invalidate_pattern(r"^temp:").await?;
    println!("Invalidated {} cache entries matching 'temp:' pattern", invalidated);

    // Verify invalidation
    if cache.get::<String>("temp:123").await.is_none() {
        println!("✓ temp:123 successfully invalidated");
    }
    if cache.get::<String>("permanent:789").await.is_some() {
        println!("✓ permanent:789 remains in cache");
    }

    // Example 5: Different write strategies
    println!("\n=== Write Strategy Demonstration ===");
    
    // Create cache with WriteBack strategy
    let writeback_cache = CacheConfigBuilder::new()
        .l1_capacity(100)
        .l2_cache_dir("./cache_writeback")
        .write_strategy(WriteStrategy::WriteBack)
        .build()
        .await?;

    writeback_cache.put("writeback_test".to_string(), "Data with WriteBack".to_string(), None).await;
    println!("Data stored with WriteBack strategy (asynchronous L2/L3 writes)");

    // Create cache with WriteBehind strategy
    let writebehind_cache = CacheConfigBuilder::new()
        .l1_capacity(100)
        .l2_cache_dir("./cache_writebehind")
        .write_strategy(WriteStrategy::WriteBehind { delay: Duration::from_millis(100) })
        .build()
        .await?;

    writebehind_cache.put("writebehind_test".to_string(), "Data with WriteBehind".to_string(), None).await;
    println!("Data stored with WriteBehind strategy (delayed L2/L3 writes)");

    // Final statistics
    println!("\n=== Final Statistics ===");
    let final_stats = cache.get_statistics().await;
    println!("Total requests: {}", final_stats.total_requests);
    println!("L1 hits: {}, L1 misses: {}", final_stats.l1_hits, final_stats.l1_misses);
    println!("L2 hits: {}, L2 misses: {}", final_stats.l2_hits, final_stats.l2_misses);
    println!("Overall hit rate: {:.2}%", final_stats.hit_rate() * 100.0);

    // Cleanup
    cache.clear().await?;
    writeback_cache.clear().await?;
    writebehind_cache.clear().await?;
    
    println!("\n✓ Cache demonstration completed successfully!");

    Ok(())
}