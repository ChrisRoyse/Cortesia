# Task 06e: Implement Single Encode Method

**Estimated Time**: 8 minutes  
**Dependencies**: 06d_implement_content_hash_helper.md  
**Stage**: Neural Integration - Core Encoding

## Objective
Implement single content encoding with cache support.

## Implementation

Add to `src/integration/ttfs_integration.rs`:
```rust
use std::time::Instant;

impl TTFSIntegrationService {
    pub async fn encode_content(
        &self,
        content: &str,
    ) -> Result<f32, TTFSIntegrationError> {
        // Check cache first
        let cache_key = self.generate_content_hash(content);
        if let Some(cached_encoding) = self.encoding_cache.read().await.get(&cache_key) {
            return Ok(*cached_encoding);
        }
        
        // Generate TTFS encoding using Phase 2 encoder
        let spike_pattern = self.ttfs_encoder.encode_text(content).await
            .map_err(|e| TTFSIntegrationError::EncodingFailed(e.to_string()))?;
        let ttfs_encoding = spike_pattern.first_spike_time;
        
        // Cache the result
        self.encoding_cache.write().await.put(cache_key, ttfs_encoding);
        
        Ok(ttfs_encoding)
    }
}
```

## Acceptance Criteria
- [ ] Method compiles without errors
- [ ] Cache lookup implemented
- [ ] Error handling present
- [ ] Cache storage implemented

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **06f_implement_batch_encode_method.md**