# Phase 1: Immediate Fixes Implementation Guide

## Overview
This guide provides specific code-level solutions to fix the 4 limited tools in the LLMKG MCP system.

---

## 1. Fix generate_graph_query Tool

### Current Issue
The tool only extracts the first word as an entity instead of understanding the full query context.

### Root Cause Analysis
```rust
// Current implementation in handlers/mod.rs
let entities = query.split_whitespace()
    .take(1)  // Only takes first word!
    .map(|s| s.to_string())
    .collect();
```

### Solution Implementation

#### Step 1: Add NLP Entity Extraction
```rust
// In handlers/advanced.rs

use regex::Regex;

fn extract_entities_from_query(query: &str) -> Vec<String> {
    let mut entities = Vec::new();
    
    // Common patterns for entity detection
    let patterns = vec![
        (r"about\s+(\w+)", 1),
        (r"(?:facts?|information|data)\s+(?:about|on|for)\s+([^,\s]+)", 1),
        (r"between\s+(\w+)\s+and\s+(\w+)", 2),
        (r"related to\s+([^,\s]+)", 1),
        (r"(\w+)'s\s+(\w+)", 2),  // possessive relationships
    ];
    
    for (pattern_str, capture_count) in patterns {
        if let Ok(re) = Regex::new(pattern_str) {
            for cap in re.captures_iter(query) {
                for i in 1..=capture_count {
                    if let Some(entity) = cap.get(i) {
                        entities.push(entity.as_str().to_string());
                    }
                }
            }
        }
    }
    
    // Fallback: extract capitalized words as potential entities
    let words: Vec<&str> = query.split_whitespace().collect();
    for word in words {
        if word.chars().next().unwrap_or('a').is_uppercase() {
            entities.push(word.to_string());
        }
    }
    
    entities.dedup();
    entities
}
```

#### Step 2: Create Query Templates
```rust
// Add to handlers/advanced.rs

struct QueryTemplate {
    pattern: Regex,
    cypher_template: &'static str,
    sparql_template: &'static str,
    gremlin_template: &'static str,
}

lazy_static! {
    static ref QUERY_TEMPLATES: Vec<QueryTemplate> = vec![
        QueryTemplate {
            pattern: Regex::new(r"find all (?:facts?|information) about (.+)").unwrap(),
            cypher_template: "MATCH (n:Entity {name: $entity})-[r]->(m) RETURN n, r, m",
            sparql_template: "SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(STR(?s) = $entity) }",
            gremlin_template: "g.V().has('name', '$entity').outE().inV().path()",
        },
        QueryTemplate {
            pattern: Regex::new(r"(?:who|what) (.+) (.+)").unwrap(),
            cypher_template: "MATCH (n)-[:$predicate]->(m) RETURN n, m",
            sparql_template: "SELECT ?s ?o WHERE { ?s :$predicate ?o }",
            gremlin_template: "g.V().out('$predicate')",
        },
        // Add more templates...
    ];
}
```

#### Step 3: Implement Enhanced Query Generation
```rust
pub async fn generate_graph_query_handler(
    Extension(state): Extension<Arc<ServerState>>,
    Json(params): Json<GenerateGraphQueryParams>,
) -> Result<Json<GenerateGraphQueryResponse>, StatusCode> {
    let natural_query = params.natural_query.to_lowercase();
    let query_language = params.query_language;
    
    // Extract entities with improved NLP
    let entities = extract_entities_from_query(&natural_query);
    
    // Match against templates
    let mut generated_query = String::new();
    let mut matched_template = false;
    
    for template in QUERY_TEMPLATES.iter() {
        if template.pattern.is_match(&natural_query) {
            matched_template = true;
            generated_query = match query_language {
                QueryLanguage::Cypher => template.cypher_template.to_string(),
                QueryLanguage::Sparql => template.sparql_template.to_string(),
                QueryLanguage::Gremlin => template.gremlin_template.to_string(),
            };
            
            // Replace placeholders with extracted entities
            for (i, entity) in entities.iter().enumerate() {
                generated_query = generated_query.replace(
                    &format!("$entity{}", if i > 0 { i.to_string() } else { "".to_string() }),
                    entity
                );
            }
            break;
        }
    }
    
    // Fallback if no template matches
    if !matched_template {
        generated_query = generate_fallback_query(&entities, query_language);
    }
    
    Ok(Json(GenerateGraphQueryResponse {
        natural_query: params.natural_query,
        query_language,
        generated_query,
        extracted_entities: entities,
        executable: true,
        explanation: Some(format!("Query generated for: {}", params.natural_query)),
        complexity_score: calculate_complexity(&natural_query),
    }))
}
```

---

## 2. Fix divergent_thinking_engine Tool

### Current Issue
Returns empty exploration paths - no actual creative thinking logic implemented.

### Solution Implementation

#### Step 1: Add Graph Traversal for Creative Connections
```rust
// In handlers/cognitive.rs

use petgraph::algo::dijkstra;
use rand::seq::SliceRandom;

#[derive(Debug, Clone, Serialize)]
struct ExplorationPath {
    id: usize,
    path: Vec<Connection>,
    creativity_score: f32,
    novelty_score: f32,
}

#[derive(Debug, Clone, Serialize)]
struct Connection {
    from: String,
    to: String,
    relationship: String,
    strength: f32,
}

async fn generate_creative_paths(
    graph: &Graph,
    seed_concept: &str,
    creativity_level: f32,
    exploration_depth: usize,
    max_branches: usize,
) -> Vec<ExplorationPath> {
    let mut paths = Vec::new();
    let mut rng = rand::thread_rng();
    
    // Find seed entity in graph
    let seed_node = match graph.find_entity(seed_concept) {
        Some(node) => node,
        None => return paths, // No seed found
    };
    
    // Generate multiple exploration branches
    for branch_id in 0..max_branches {
        let mut current_path = Vec::new();
        let mut current_node = seed_node;
        let mut visited = HashSet::new();
        visited.insert(current_node);
        
        // Explore to specified depth
        for depth in 0..exploration_depth {
            // Get all neighbors
            let neighbors = graph.get_neighbors(current_node);
            
            // Filter by creativity level (higher = more distant connections)
            let mut weighted_neighbors: Vec<(NodeIndex, f32)> = neighbors
                .into_iter()
                .filter(|n| !visited.contains(n))
                .map(|n| {
                    let distance = graph.semantic_distance(current_node, n);
                    let weight = if creativity_level > 0.7 {
                        // Prefer distant connections for high creativity
                        distance
                    } else {
                        // Prefer close connections for low creativity
                        1.0 - distance
                    };
                    (n, weight)
                })
                .collect();
            
            // Sort by weight and add randomness based on creativity
            weighted_neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Select next node with controlled randomness
            let selection_range = (creativity_level * neighbors.len() as f32) as usize;
            let selection_range = selection_range.max(1).min(weighted_neighbors.len());
            
            if let Some(&(next_node, _)) = weighted_neighbors[..selection_range].choose(&mut rng) {
                // Get relationship between nodes
                let relationship = graph.get_relationship(current_node, next_node)
                    .unwrap_or_else(|| "connects_to".to_string());
                
                // Add connection to path
                current_path.push(Connection {
                    from: graph.get_entity_name(current_node),
                    to: graph.get_entity_name(next_node),
                    relationship,
                    strength: graph.get_edge_weight(current_node, next_node),
                });
                
                visited.insert(next_node);
                current_node = next_node;
            } else {
                break; // No more valid neighbors
            }
        }
        
        if !current_path.is_empty() {
            paths.push(ExplorationPath {
                id: branch_id,
                path: current_path,
                creativity_score: calculate_path_creativity(&current_path),
                novelty_score: calculate_path_novelty(&current_path, &paths),
            });
        }
    }
    
    paths
}
```

#### Step 2: Add Creativity Scoring
```rust
fn calculate_path_creativity(path: &[Connection]) -> f32 {
    if path.is_empty() {
        return 0.0;
    }
    
    let mut creativity = 0.0;
    
    // Factor 1: Semantic distance between consecutive nodes
    for window in path.windows(2) {
        let distance = estimate_semantic_distance(&window[0].to, &window[1].from);
        creativity += distance;
    }
    
    // Factor 2: Relationship diversity
    let unique_relationships: HashSet<_> = path.iter()
        .map(|c| &c.relationship)
        .collect();
    let diversity_score = unique_relationships.len() as f32 / path.len() as f32;
    
    // Factor 3: Path length bonus (longer paths are more creative)
    let length_bonus = (path.len() as f32).ln() / 10.0;
    
    // Combine factors
    (creativity / path.len() as f32 + diversity_score + length_bonus) / 3.0
}

fn calculate_path_novelty(path: &[Connection], existing_paths: &[ExplorationPath]) -> f32 {
    if existing_paths.is_empty() {
        return 1.0;
    }
    
    // Check how different this path is from existing ones
    let mut min_similarity = 1.0;
    
    for existing in existing_paths {
        let similarity = calculate_path_similarity(path, &existing.path);
        min_similarity = min_similarity.min(similarity);
    }
    
    1.0 - min_similarity
}
```

#### Step 3: Add Cross-Domain Discovery
```rust
async fn find_cross_domain_connections(
    graph: &Graph,
    seed_concept: &str,
    domains: &[String],
) -> Vec<CrossDomainLink> {
    let mut links = Vec::new();
    
    // Get domain of seed concept
    let seed_domain = graph.get_entity_domain(seed_concept);
    
    // Find bridging concepts
    for target_domain in domains {
        if target_domain == &seed_domain {
            continue;
        }
        
        // Find shortest paths to entities in target domain
        let target_entities = graph.get_entities_in_domain(target_domain);
        
        for target in target_entities.iter().take(10) {
            if let Some(path) = graph.find_shortest_path(seed_concept, target) {
                if path.len() <= 4 { // Reasonable connection distance
                    links.push(CrossDomainLink {
                        source_domain: seed_domain.clone(),
                        target_domain: target_domain.clone(),
                        bridge_concepts: path,
                        strength: 1.0 / path.len() as f32,
                    });
                }
            }
        }
    }
    
    links
}
```

---

## 3. Fix time_travel_query Tool

### Current Issue
No temporal data is being tracked, so temporal queries return empty results.

### Solution Implementation

#### Step 1: Add Timestamp Tracking
```rust
// Modify the Triple struct in lib.rs to include timestamps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
    pub timestamp: DateTime<Utc>,  // Add this
    pub version: u32,              // Add this
    pub previous_version: Option<Box<Triple>>, // Add this for history
}

// Update store_fact to track time
pub async fn store_fact_handler(
    Extension(state): Extension<Arc<ServerState>>,
    Json(params): Json<StoreFactParams>,
) -> Result<Json<StoreFactResponse>, StatusCode> {
    let triple = Triple {
        subject: params.subject.clone(),
        predicate: params.predicate.clone(),
        object: params.object.clone(),
        confidence: params.confidence.unwrap_or(1.0),
        timestamp: Utc::now(),
        version: 1,
        previous_version: None,
    };
    
    // Check if fact already exists
    if let Some(existing) = state.graph.find_triple(&triple) {
        // Create new version
        let new_version = Triple {
            version: existing.version + 1,
            previous_version: Some(Box::new(existing)),
            timestamp: Utc::now(),
            ..triple
        };
        state.graph.update_triple(new_version);
    } else {
        state.graph.add_triple(triple);
    }
    
    // ... rest of implementation
}
```

#### Step 2: Implement Temporal Indexing
```rust
// Add temporal index structure
pub struct TemporalIndex {
    // Entity -> Timeline of changes
    timelines: HashMap<String, Vec<TemporalEvent>>,
}

#[derive(Debug, Clone)]
struct TemporalEvent {
    timestamp: DateTime<Utc>,
    event_type: EventType,
    triple: Triple,
}

#[derive(Debug, Clone)]
enum EventType {
    Created,
    Updated,
    Deleted,
}

impl TemporalIndex {
    pub fn add_event(&mut self, entity: &str, event: TemporalEvent) {
        self.timelines
            .entry(entity.to_string())
            .or_insert_with(Vec::new)
            .push(event);
        
        // Keep sorted by timestamp
        if let Some(timeline) = self.timelines.get_mut(entity) {
            timeline.sort_by_key(|e| e.timestamp);
        }
    }
    
    pub fn get_state_at(&self, entity: &str, timestamp: DateTime<Utc>) -> Option<EntityState> {
        let timeline = self.timelines.get(entity)?;
        
        // Find all events before timestamp
        let mut facts = HashMap::new();
        let mut relationships = HashMap::new();
        
        for event in timeline.iter().filter(|e| e.timestamp <= timestamp) {
            match event.event_type {
                EventType::Created | EventType::Updated => {
                    facts.insert(
                        format!("{}-{}", event.triple.predicate, event.triple.object),
                        event.triple.clone()
                    );
                }
                EventType::Deleted => {
                    facts.remove(&format!("{}-{}", event.triple.predicate, event.triple.object));
                }
            }
        }
        
        Some(EntityState {
            entity: entity.to_string(),
            timestamp,
            facts: facts.into_values().collect(),
            relationships: relationships.into_values().collect(),
        })
    }
}
```

#### Step 3: Implement Temporal Queries
```rust
pub async fn time_travel_query_handler(
    Extension(state): Extension<Arc<ServerState>>,
    Json(params): Json<TimeTravelQueryParams>,
) -> Result<Json<TimeTravelQueryResponse>, StatusCode> {
    match params.query_type {
        QueryType::PointInTime => {
            let timestamp = params.timestamp.unwrap_or_else(Utc::now);
            let entity = params.entity.as_ref().ok_or(StatusCode::BAD_REQUEST)?;
            
            if let Some(state) = state.temporal_index.get_state_at(entity, timestamp) {
                Ok(Json(TimeTravelQueryResponse {
                    query_type: params.query_type,
                    results: json!({
                        "entity_state": state,
                        "facts_count": state.facts.len(),
                        "relationships_count": state.relationships.len(),
                    }),
                    temporal_metadata: TemporalMetadata {
                        query_timestamp: Utc::now(),
                        time_span_covered: Some((timestamp, timestamp)),
                        data_points: state.facts.len() + state.relationships.len(),
                        changes_detected: 0,
                    },
                    insights: vec![
                        format!("{} had {} facts at {}", entity, state.facts.len(), timestamp.format("%Y-%m-%d")),
                    ],
                    trends: vec![],
                }))
            } else {
                Ok(Json(TimeTravelQueryResponse::empty()))
            }
        }
        
        QueryType::EvolutionTracking => {
            let entity = params.entity.as_ref().ok_or(StatusCode::BAD_REQUEST)?;
            let time_range = params.time_range.ok_or(StatusCode::BAD_REQUEST)?;
            
            let timeline = state.temporal_index.get_timeline(entity, time_range.start, time_range.end);
            
            let evolution_points: Vec<_> = timeline.iter()
                .map(|event| json!({
                    "timestamp": event.timestamp,
                    "change_type": format!("{:?}", event.event_type),
                    "fact": format!("{} {} {}", event.triple.subject, event.triple.predicate, event.triple.object),
                }))
                .collect();
            
            Ok(Json(TimeTravelQueryResponse {
                query_type: params.query_type,
                results: json!({
                    "evolution_timeline": evolution_points,
                    "total_changes": timeline.len(),
                }),
                temporal_metadata: TemporalMetadata {
                    query_timestamp: Utc::now(),
                    time_span_covered: Some((time_range.start, time_range.end)),
                    data_points: timeline.len(),
                    changes_detected: timeline.len(),
                },
                insights: analyze_evolution_insights(&timeline),
                trends: detect_temporal_trends(&timeline),
            }))
        }
        
        // ... implement other query types
    }
}
```

---

## 4. Fix cognitive_reasoning_chains Tool

### Current Issue
Returns empty reasoning chains - no actual reasoning logic implemented.

### Solution Implementation

#### Step 1: Implement Reasoning Algorithms
```rust
// In handlers/cognitive.rs

#[derive(Debug, Clone, Serialize)]
struct ReasoningStep {
    step_number: usize,
    statement: String,
    justification: String,
    confidence: f32,
    step_type: StepType,
}

#[derive(Debug, Clone, Serialize)]
enum StepType {
    Premise,
    Inference,
    Conclusion,
    Assumption,
}

async fn build_deductive_chain(
    graph: &Graph,
    premise: &str,
    max_length: usize,
    confidence_threshold: f32,
) -> Vec<ReasoningStep> {
    let mut chain = Vec::new();
    
    // Parse premise into facts
    let premise_facts = parse_premise(premise);
    
    // Add premises to chain
    for (i, fact) in premise_facts.iter().enumerate() {
        chain.push(ReasoningStep {
            step_number: i + 1,
            statement: format!("{} {} {}", fact.subject, fact.predicate, fact.object),
            justification: "Given premise".to_string(),
            confidence: 1.0,
            step_type: StepType::Premise,
        });
    }
    
    // Apply deductive rules
    let rules = vec![
        // Modus Ponens: If A->B and A, then B
        DeductiveRule {
            name: "Modus Ponens",
            pattern: vec!["$A implies $B", "$A"],
            conclusion: "$B",
        },
        // Syllogism: If A->B and B->C, then A->C
        DeductiveRule {
            name: "Syllogism",
            pattern: vec!["$A implies $B", "$B implies $C"],
            conclusion: "$A implies $C",
        },
        // Universal Instantiation: If all X are Y and Z is X, then Z is Y
        DeductiveRule {
            name: "Universal Instantiation",
            pattern: vec!["all $X are $Y", "$Z is $X"],
            conclusion: "$Z is $Y",
        },
    ];
    
    // Try to apply rules iteratively
    let mut current_facts = premise_facts.clone();
    let mut step_number = premise_facts.len() + 1;
    
    while chain.len() < max_length {
        let mut new_inferences = Vec::new();
        
        for rule in &rules {
            if let Some(inference) = apply_rule(&current_facts, rule, graph) {
                if inference.confidence >= confidence_threshold {
                    new_inferences.push(ReasoningStep {
                        step_number,
                        statement: inference.statement,
                        justification: format!("By {} from steps {}", rule.name, inference.from_steps.join(", ")),
                        confidence: inference.confidence,
                        step_type: StepType::Inference,
                    });
                    step_number += 1;
                }
            }
        }
        
        if new_inferences.is_empty() {
            break; // No more inferences possible
        }
        
        // Add new inferences to chain and current facts
        for inference in new_inferences {
            current_facts.push(parse_statement(&inference.statement));
            chain.push(inference);
        }
    }
    
    // Add final conclusion if we have one
    if let Some(conclusion) = derive_conclusion(&chain) {
        chain.push(ReasoningStep {
            step_number,
            statement: conclusion,
            justification: "From the above reasoning".to_string(),
            confidence: calculate_chain_confidence(&chain),
            step_type: StepType::Conclusion,
        });
    }
    
    chain
}
```

#### Step 2: Implement Other Reasoning Types
```rust
async fn build_inductive_chain(
    graph: &Graph,
    premise: &str,
    max_length: usize,
) -> Vec<ReasoningStep> {
    let mut chain = Vec::new();
    
    // Extract observations from premise
    let observations = extract_observations(premise);
    
    // Add observations to chain
    for (i, obs) in observations.iter().enumerate() {
        chain.push(ReasoningStep {
            step_number: i + 1,
            statement: obs.clone(),
            justification: "Observed instance".to_string(),
            confidence: 0.9, // Observations have high but not perfect confidence
            step_type: StepType::Premise,
        });
    }
    
    // Find patterns in observations
    if let Some(pattern) = find_pattern(&observations, graph) {
        chain.push(ReasoningStep {
            step_number: chain.len() + 1,
            statement: format!("Pattern detected: {}", pattern.description),
            justification: "From observations 1-{}".to_string(),
            confidence: pattern.strength,
            step_type: StepType::Inference,
        });
        
        // Generalize pattern
        if let Some(generalization) = generalize_pattern(&pattern, graph) {
            chain.push(ReasoningStep {
                step_number: chain.len() + 1,
                statement: generalization,
                justification: "Inductive generalization".to_string(),
                confidence: pattern.strength * 0.8, // Generalizations are less certain
                step_type: StepType::Conclusion,
            });
        }
    }
    
    chain
}

async fn build_abductive_chain(
    graph: &Graph,
    premise: &str,
    max_length: usize,
) -> Vec<ReasoningStep> {
    let mut chain = Vec::new();
    
    // Add observation
    chain.push(ReasoningStep {
        step_number: 1,
        statement: premise.to_string(),
        justification: "Observed fact".to_string(),
        confidence: 1.0,
        step_type: StepType::Premise,
    });
    
    // Generate possible explanations
    let explanations = generate_explanations(premise, graph);
    
    // Add possible explanations
    for (i, explanation) in explanations.iter().enumerate().take(3) {
        chain.push(ReasoningStep {
            step_number: chain.len() + 1,
            statement: format!("Possible explanation {}: {}", i + 1, explanation.hypothesis),
            justification: "Abductive hypothesis".to_string(),
            confidence: explanation.likelihood,
            step_type: StepType::Assumption,
        });
    }
    
    // Select best explanation
    if let Some(best) = explanations.iter().max_by(|a, b| a.likelihood.partial_cmp(&b.likelihood).unwrap()) {
        chain.push(ReasoningStep {
            step_number: chain.len() + 1,
            statement: format!("Best explanation: {}", best.hypothesis),
            justification: "Highest likelihood based on available evidence".to_string(),
            confidence: best.likelihood,
            step_type: StepType::Conclusion,
        });
    }
    
    chain
}
```

#### Step 3: Add Supporting Logic
```rust
fn parse_premise(premise: &str) -> Vec<Fact> {
    let mut facts = Vec::new();
    
    // Split by periods or semicolons
    let statements = premise.split(&['.', ';'][..])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty());
    
    for statement in statements {
        if let Some(fact) = parse_statement(statement) {
            facts.push(fact);
        }
    }
    
    facts
}

fn apply_rule(facts: &[Fact], rule: &DeductiveRule, graph: &Graph) -> Option<Inference> {
    // Pattern matching logic
    let mut bindings = HashMap::new();
    
    // Check if all patterns in rule match current facts
    for pattern in &rule.pattern {
        let mut matched = false;
        
        for fact in facts {
            if let Some(new_bindings) = match_pattern(pattern, fact) {
                // Merge bindings
                for (var, val) in new_bindings {
                    bindings.insert(var, val);
                }
                matched = true;
                break;
            }
        }
        
        if !matched {
            return None; // Rule doesn't apply
        }
    }
    
    // Apply bindings to conclusion
    let conclusion = apply_bindings(&rule.conclusion, &bindings);
    
    Some(Inference {
        statement: conclusion,
        confidence: 0.9, // High confidence for deductive rules
        from_steps: vec![], // Track which steps were used
    })
}

fn calculate_chain_confidence(chain: &[ReasoningStep]) -> f32 {
    if chain.is_empty() {
        return 0.0;
    }
    
    // Confidence decreases with chain length
    let length_factor = 1.0 / (1.0 + 0.1 * chain.len() as f32);
    
    // Average confidence of all steps
    let avg_confidence = chain.iter()
        .map(|s| s.confidence)
        .sum::<f32>() / chain.len() as f32;
    
    avg_confidence * length_factor
}
```

---

## Testing the Fixes

### Test Suite for Fixed Tools
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_generate_graph_query_improved() {
        let params = GenerateGraphQueryParams {
            natural_query: "Find all facts about Einstein and Tesla".to_string(),
            query_language: QueryLanguage::Cypher,
            include_explanation: true,
        };
        
        let response = generate_graph_query_handler(state, params).await.unwrap();
        
        assert!(response.extracted_entities.contains(&"Einstein".to_string()));
        assert!(response.extracted_entities.contains(&"Tesla".to_string()));
        assert!(response.generated_query.contains("Einstein"));
        assert!(response.generated_query.contains("Tesla"));
    }
    
    #[tokio::test]
    async fn test_divergent_thinking_with_paths() {
        let params = DivergentThinkingParams {
            seed_concept: "artificial intelligence".to_string(),
            creativity_level: 0.7,
            exploration_depth: 3,
            max_branches: 5,
        };
        
        let response = divergent_thinking_handler(state, params).await.unwrap();
        
        assert!(!response.exploration_paths.is_empty());
        assert!(response.exploration_paths[0].path.len() > 0);
        assert!(response.metrics.average_creativity > 0.0);
    }
    
    #[tokio::test]
    async fn test_temporal_queries() {
        // First store some facts with timestamps
        let fact = StoreFactParams {
            subject: "Einstein".to_string(),
            predicate: "won".to_string(),
            object: "Nobel Prize".to_string(),
            confidence: Some(1.0),
        };
        
        store_fact_handler(state.clone(), fact).await.unwrap();
        
        // Query at current time
        let params = TimeTravelQueryParams {
            query_type: QueryType::PointInTime,
            entity: Some("Einstein".to_string()),
            timestamp: Some(Utc::now()),
            time_range: None,
        };
        
        let response = time_travel_query_handler(state, params).await.unwrap();
        
        assert!(response.results.get("entity_state").is_some());
        assert!(response.temporal_metadata.data_points > 0);
    }
    
    #[tokio::test]
    async fn test_reasoning_chains() {
        let params = CognitiveReasoningParams {
            premise: "All humans are mortal. Socrates is human.".to_string(),
            reasoning_type: ReasoningType::Deductive,
            max_chain_length: 5,
            confidence_threshold: 0.7,
            include_alternatives: false,
        };
        
        let response = cognitive_reasoning_handler(state, params).await.unwrap();
        
        assert!(!response.reasoning_chains.is_empty());
        assert_eq!(response.primary_conclusion, "Socrates is mortal");
        assert!(response.logical_validity > 0.8);
    }
}
```

## Deployment Strategy

### Phase 1 Week 1 Tasks:
1. **Day 1-2**: Implement generate_graph_query fixes
2. **Day 3-4**: Implement divergent_thinking_engine fixes  
3. **Day 5-6**: Implement time_travel_query fixes
4. **Day 7-8**: Implement cognitive_reasoning_chains fixes
5. **Day 9-10**: Integration testing and bug fixes

### Success Criteria:
- All 4 tools return meaningful, non-empty results
- Test coverage > 80% for new code
- Performance benchmarks pass (<100ms response time)
- No regression in existing tools

This implementation guide provides the concrete code changes needed to fix all 4 limited tools in Phase 1.