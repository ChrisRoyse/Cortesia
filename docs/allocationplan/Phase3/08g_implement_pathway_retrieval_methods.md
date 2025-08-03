# Task 08g: Implement Pathway Retrieval Methods

**Estimated Time**: 7 minutes  
**Dependencies**: 08f_implement_pathway_creation_method.md  
**Stage**: Neural Integration - Retrieval Logic

## Objective
Implement methods to retrieve pathways by various criteria.

## Implementation

Add to `src/neural_pathways/pathway_storage.rs`:
```rust
impl PathwayStorageService {
    pub async fn get_pathway(&self, pathway_id: &str) -> Option<NeuralPathway> {
        self.pathways.read().await.get(pathway_id).cloned()
    }
    
    pub async fn get_pathways_for_concept(&self, concept_id: &str) -> Vec<NeuralPathway> {
        let concept_pathways = self.concept_pathways.read().await;
        let pathway_ids = match concept_pathways.get(concept_id) {
            Some(ids) => ids.clone(),
            None => return Vec::new(),
        };
        
        let pathways = self.pathways.read().await;
        pathway_ids
            .into_iter()
            .filter_map(|id| pathways.get(&id).cloned())
            .collect()
    }
    
    pub async fn get_pathway_between_concepts(
        &self,
        source_id: &str,
        target_id: &str,
    ) -> Option<NeuralPathway> {
        let index = self.pathway_index.read().await;
        let pathway_id = index.get(&(source_id.to_string(), target_id.to_string()))?;
        
        self.pathways.read().await.get(pathway_id).cloned()
    }
    
    pub async fn get_all_pathways(&self) -> Vec<NeuralPathway> {
        self.pathways.read().await.values().cloned().collect()
    }
}
```

## Acceptance Criteria
- [ ] All retrieval methods compile
- [ ] Individual pathway lookup working
- [ ] Concept-based pathway lookup
- [ ] Bidirectional pathway lookup
- [ ] All pathways retrieval

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **08h_implement_pathway_activation_method.md**