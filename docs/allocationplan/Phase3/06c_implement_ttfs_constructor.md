# Task 06c: Implement TTFS Constructor

**Estimated Time**: 4 minutes  
**Dependencies**: 06b_create_ttfs_integration_struct.md  
**Stage**: Neural Integration - Constructor

## Objective
Implement the constructor for TTFSIntegrationService.

## Implementation

Add to `src/integration/ttfs_integration.rs`:
```rust
impl TTFSIntegrationService {
    pub async fn new(ttfs_encoder: Arc<TTFSEncoder>) -> Result<Self, TTFSIntegrationError> {
        Ok(Self {
            ttfs_encoder,
            encoding_cache: Arc::new(RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(10000).unwrap()
            ))),
            similarity_cache: Arc::new(RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(50000).unwrap()
            ))),
        })
    }
}
```

## Acceptance Criteria
- [ ] Constructor compiles without errors
- [ ] Cache sizes properly configured
- [ ] Returns Result type

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **06d_implement_content_hash_helper.md**