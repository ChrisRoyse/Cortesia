# Task 09e: Implement Cache Key Generator

**Estimated Time**: 5 minutes  
**Dependencies**: 09d_implement_allocation_constructor.md  
**Stage**: Neural Integration - Cache Helpers

## Objective
Implement cache key generation for placement decisions.

## Implementation

Add to `src/integration/allocation_placement.rs`:
```rust
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

impl AllocationGuidedPlacement {
    fn generate_placement_cache_key(
        &self,
        content: &str,
        spike_pattern: &TTFSSpikePattern,
    ) -> String {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        spike_pattern.first_spike_time.to_bits().hash(&mut hasher);
        spike_pattern.pattern_hash.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
    
    fn generate_content_hash(&self, content: &str) -> String {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}
```

## Acceptance Criteria
- [ ] Cache key generation compiles
- [ ] Uses spike pattern and content
- [ ] Generates consistent hashes
- [ ] Helper functions available

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **09f_implement_optimal_placement_method.md**