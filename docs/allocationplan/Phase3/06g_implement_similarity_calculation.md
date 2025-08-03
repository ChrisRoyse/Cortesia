# Task 06g: Implement Similarity Calculation

**Estimated Time**: 5 minutes  
**Dependencies**: 06f_implement_batch_encode_method.md  
**Stage**: Neural Integration - Similarity

## Objective
Implement TTFS similarity calculation method.

## Implementation

Add to `src/integration/ttfs_integration.rs`:
```rust
impl TTFSIntegrationService {
    pub async fn calculate_ttfs_similarity(
        &self,
        encoding1: f32,
        encoding2: f32,
    ) -> Result<f32, TTFSIntegrationError> {
        // Use inverse of time difference for similarity
        // Closer spike times = higher similarity
        let time_diff = (encoding1 - encoding2).abs();
        let similarity = 1.0 / (1.0 + time_diff);
        
        Ok(similarity)
    }
}
```

## Acceptance Criteria
- [ ] Similarity calculation compiles
- [ ] Uses inverse time difference
- [ ] Returns value between 0 and 1
- [ ] Handles identical encodings (similarity = 1.0)

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **06h_create_similarity_types.md**