# Task 06d: Implement Content Hash Helper

**Estimated Time**: 5 minutes  
**Dependencies**: 06c_implement_ttfs_constructor.md  
**Stage**: Neural Integration - Helper Functions

## Objective
Implement content hashing helper for cache keys.

## Implementation

Add to `src/integration/ttfs_integration.rs`:
```rust
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

impl TTFSIntegrationService {
    fn generate_content_hash(&self, content: &str) -> String {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}
```

## Acceptance Criteria
- [ ] Hash function compiles
- [ ] Generates consistent hashes for same content
- [ ] Returns String for cache key

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **06e_implement_single_encode_method.md**