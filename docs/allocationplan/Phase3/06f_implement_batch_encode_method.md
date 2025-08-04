# Task 06f: Implement Batch Encode Method

**Estimated Time**: 10 minutes  
**Dependencies**: 06e_implement_single_encode_method.md  
**Stage**: Neural Integration - Batch Operations

## Objective
Implement batch encoding with cache optimization.

## Implementation

Add to `src/integration/ttfs_integration.rs`:
```rust
impl TTFSIntegrationService {
    pub async fn batch_encode_content(
        &self,
        content_items: Vec<&str>,
    ) -> Result<Vec<f32>, TTFSIntegrationError> {
        let mut encodings = Vec::with_capacity(content_items.len());
        let mut uncached_items = Vec::new();
        let mut uncached_indices = Vec::new();
        
        // Check cache for all items first
        for (i, content) in content_items.iter().enumerate() {
            let cache_key = self.generate_content_hash(content);
            if let Some(cached_encoding) = self.encoding_cache.read().await.get(&cache_key) {
                encodings.push(*cached_encoding);
            } else {
                uncached_items.push(*content);
                uncached_indices.push(i);
                encodings.push(0.0); // Placeholder
            }
        }
        
        // Batch encode uncached items
        if !uncached_items.is_empty() {
            let batch_encodings = self.ttfs_encoder.batch_encode_text(uncached_items).await
                .map_err(|e| TTFSIntegrationError::EncodingFailed(e.to_string()))?;
            
            // Update cache and results
            for (i, encoding) in batch_encodings.iter().enumerate() {
                let original_index = uncached_indices[i];
                let cache_key = self.generate_content_hash(content_items[original_index]);
                
                self.encoding_cache.write().await.put(cache_key, *encoding);
                encodings[original_index] = *encoding;
            }
        }
        
        Ok(encodings)
    }
}
```

## Acceptance Criteria
- [ ] Batch method compiles
- [ ] Cache optimization implemented
- [ ] Handles mixed cached/uncached items
- [ ] Updates cache with new encodings

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **06g_implement_similarity_calculation.md**