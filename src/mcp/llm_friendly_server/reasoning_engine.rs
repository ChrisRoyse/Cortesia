//! Pure algorithmic reasoning engine - no AI models
//! Implements deductive, inductive, abductive, and analogical reasoning

use crate::core::knowledge_types::KnowledgeResult;
use crate::core::triple::Triple;
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    pub steps: Vec<ReasoningStep>,
    pub conclusion: String,
    pub confidence: f32,
    pub reasoning_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_number: usize,
    pub premise: String,
    pub inference: String,
    pub justification: String,
    pub confidence: f32,
}

pub struct ReasoningResult {
    pub chains: Vec<Value>,
    pub primary_conclusion: String,
    pub logical_validity: f32,
    pub confidence_scores: Vec<f32>,
    pub supporting_evidence: Vec<String>,
    pub counterarguments: Vec<String>,
}

/// Execute deductive reasoning - from general to specific
pub async fn execute_deductive_reasoning(
    premise: &str,
    knowledge: &KnowledgeResult,
    max_chain_length: usize,
    confidence_threshold: f32,
) -> ReasoningResult {
    // Handle empty knowledge base
    if knowledge.triples.is_empty() {
        return ReasoningResult {
            chains: vec![],
            primary_conclusion: format!("No knowledge available for reasoning about '{}'", premise),
            logical_validity: 0.0,
            confidence_scores: vec![],
            supporting_evidence: vec![],
            counterarguments: vec![],
        };
    }
    
    let mut chains = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut cycle_paths: HashSet<String> = HashSet::new();
    let mut contradictions: Vec<String> = Vec::new();
    
    // Start with the premise
    queue.push_back((premise.to_string(), vec![], 1.0));
    visited.insert(premise.to_string());
    
    while let Some((current_entity, chain, confidence)) = queue.pop_front() {
        if chain.len() >= max_chain_length || confidence < confidence_threshold {
            continue;
        }
        
        // Check for cycles in the current chain
        let entities_in_chain: HashSet<&str> = chain.iter()
            .map(|step: &ReasoningStep| step.premise.as_str())
            .chain(std::iter::once(current_entity.as_str()))
            .collect();
        
        // Find all facts about current entity
        let related_facts: Vec<&Triple> = knowledge.triples.iter()
            .filter(|t| t.subject == current_entity || t.object == current_entity)
            .collect();
        
        // Check for contradictions
        let mut facts_by_subject_predicate: HashMap<(String, String), Vec<String>> = HashMap::new();
        for fact in &related_facts {
            let key = (fact.subject.clone(), fact.predicate.clone());
            facts_by_subject_predicate.entry(key).or_insert_with(Vec::new).push(fact.object.clone());
        }
        
        for ((subj, pred), objects) in &facts_by_subject_predicate {
            if objects.len() > 1 {
                // Check for contradictory predicates (e.g., "is" vs "is_not")
                if pred.contains("not") || pred.contains("isn't") {
                    let positive_pred = pred.replace("_not", "").replace("isn't", "is");
                    if facts_by_subject_predicate.contains_key(&(subj.clone(), positive_pred.clone())) {
                        contradictions.push(format!("Contradiction detected: {} {} and {} {}", 
                            subj, positive_pred, subj, pred));
                    }
                }
            }
        }
        
        for fact in related_facts {
            let (next_entity, relationship) = if fact.subject == current_entity {
                (&fact.object, format!("{} {} {}", fact.subject, fact.predicate, fact.object))
            } else {
                (&fact.subject, format!("{} {} {}", fact.subject, fact.predicate, fact.object))
            };
            
            // Detect cycles - check if next_entity is already in the current chain
            if entities_in_chain.contains(next_entity.as_str()) {
                let cycle_path = format!("{} -> {} (cycle detected)", current_entity, next_entity);
                cycle_paths.insert(cycle_path);
                continue; // Skip this path to prevent infinite loop
            }
            
            if !visited.contains(next_entity) {
                visited.insert(next_entity.clone());
                
                let mut new_chain = chain.clone();
                new_chain.push(ReasoningStep {
                    step_number: new_chain.len() + 1,
                    premise: current_entity.clone(),
                    inference: next_entity.clone(),
                    justification: format!("Because: {}", relationship),
                    confidence: confidence * fact.confidence,
                });
                
                if new_chain.len() >= 2 {
                    // Found a reasoning chain
                    let conclusion = format!(
                        "If {} is true, then {} follows",
                        premise,
                        next_entity
                    );
                    
                    chains.push(json!({
                        "type": "deductive",
                        "steps": new_chain.iter().map(|s| json!({
                            "step": s.step_number,
                            "from": s.premise,
                            "to": s.inference,
                            "reason": s.justification,
                            "confidence": s.confidence
                        })).collect::<Vec<_>>(),
                        "conclusion": conclusion,
                        "confidence": confidence * fact.confidence,
                        "validity": calculate_deductive_validity(&new_chain),
                        "cycles_detected": !cycle_paths.is_empty(),
                        "contradictions": contradictions.clone()
                    }));
                }
                
                queue.push_back((next_entity.clone(), new_chain, confidence * fact.confidence));
            }
        }
    }
    
    // Generate primary conclusion
    let primary_conclusion = if chains.is_empty() {
        format!("No deductive conclusions can be drawn from '{}'", premise)
    } else {
        let best_chain = chains.iter()
            .max_by(|a, b| {
                let conf_a = a["confidence"].as_f64().unwrap_or(0.0);
                let conf_b = b["confidence"].as_f64().unwrap_or(0.0);
                conf_a.partial_cmp(&conf_b).unwrap()
            })
            .unwrap();
        best_chain["conclusion"].as_str().unwrap_or("").to_string()
    };
    
    let confidence_scores: Vec<f32> = chains.iter()
        .map(|c| c["confidence"].as_f64().unwrap_or(0.0) as f32)
        .collect();
    
    let avg_validity = if chains.is_empty() {
        0.0
    } else {
        chains.iter()
            .map(|c| c["validity"].as_f64().unwrap_or(0.0) as f32)
            .sum::<f32>() / chains.len() as f32
    };
    
    let mut final_counterarguments = contradictions.clone();
    if !cycle_paths.is_empty() {
        final_counterarguments.push(format!("Warning: {} cycle(s) detected in reasoning paths", cycle_paths.len()));
    }
    
    ReasoningResult {
        chains,
        primary_conclusion,
        logical_validity: avg_validity,
        confidence_scores,
        supporting_evidence: extract_supporting_evidence(knowledge, premise),
        counterarguments: final_counterarguments,
    }
}

/// Execute inductive reasoning - from specific to general
pub async fn execute_inductive_reasoning(
    premise: &str,
    knowledge: &KnowledgeResult,
    max_chain_length: usize,
    confidence_threshold: f32,
) -> ReasoningResult {
    // Handle empty knowledge base
    if knowledge.triples.is_empty() {
        return ReasoningResult {
            chains: vec![],
            primary_conclusion: format!("No knowledge available for inductive reasoning about '{}'", premise),
            logical_validity: 0.0,
            confidence_scores: vec![],
            supporting_evidence: vec![],
            counterarguments: vec![],
        };
    }
    
    let mut chains = Vec::new();
    let mut patterns: HashMap<String, Vec<String>> = HashMap::new();
    
    // Find all instances similar to the premise
    let similar_entities: Vec<&str> = knowledge.triples.iter()
        .filter_map(|t| {
            if t.predicate.contains(&premise) || t.subject.contains(&premise) || t.object.contains(&premise) {
                Some(t.subject.as_str())
            } else {
                None
            }
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    
    // Handle case where no similar entities found
    if similar_entities.is_empty() {
        return ReasoningResult {
            chains: vec![],
            primary_conclusion: format!("No similar instances found for inductive reasoning about '{}'", premise),
            logical_validity: 0.0,
            confidence_scores: vec![],
            supporting_evidence: vec![],
            counterarguments: vec![],
        };
    }
    
    // Look for patterns among similar entities
    for entity in &similar_entities {
        let entity_facts: Vec<&Triple> = knowledge.triples.iter()
            .filter(|t| t.subject == *entity)
            .collect();
        
        for fact in entity_facts {
            let pattern_key = format!("{} -> {}", fact.predicate, fact.object);
            patterns.entry(pattern_key).or_insert_with(Vec::new).push(entity.to_string());
        }
    }
    
    // Generate inductive conclusions from patterns
    for (pattern, instances) in patterns.iter() {
        if instances.len() >= 2 && instances.len() as f32 / similar_entities.len() as f32 > 0.5 {
            let confidence = instances.len() as f32 / similar_entities.len().max(1) as f32;
            
            if confidence >= confidence_threshold {
                let conclusion = format!(
                    "Based on {} instances, entities like '{}' generally exhibit pattern: {}",
                    instances.len(),
                    premise,
                    pattern
                );
                
                let steps: Vec<Value> = instances.iter().take(max_chain_length).enumerate().map(|(i, inst)| {
                    json!({
                        "step": i + 1,
                        "observation": format!("{} exhibits {}", inst, pattern),
                        "confidence": confidence
                    })
                }).collect();
                
                chains.push(json!({
                    "type": "inductive",
                    "pattern": pattern,
                    "instances": instances.len(),
                    "total_samples": similar_entities.len(),
                    "steps": steps,
                    "conclusion": conclusion,
                    "confidence": confidence,
                    "validity": confidence * 0.8 // Inductive reasoning is inherently less certain
                }));
            }
        }
    }
    
    let primary_conclusion = if chains.is_empty() {
        format!("No patterns found for inductive reasoning from '{}'", premise)
    } else {
        chains[0]["conclusion"].as_str().unwrap_or("").to_string()
    };
    
    let confidence_scores: Vec<f32> = chains.iter()
        .map(|c| c["confidence"].as_f64().unwrap_or(0.0) as f32)
        .collect();
    
    ReasoningResult {
        chains,
        primary_conclusion,
        logical_validity: 0.75, // Inductive reasoning has moderate validity
        confidence_scores,
        supporting_evidence: extract_supporting_evidence(knowledge, premise),
        counterarguments: generate_inductive_counterarguments(&patterns),
    }
}

/// Execute abductive reasoning - best explanation
pub async fn execute_abductive_reasoning(
    premise: &str,
    knowledge: &KnowledgeResult,
    max_chain_length: usize,
    confidence_threshold: f32,
) -> ReasoningResult {
    // Handle empty knowledge base
    if knowledge.triples.is_empty() {
        return ReasoningResult {
            chains: vec![],
            primary_conclusion: format!("No knowledge available for abductive reasoning about '{}'", premise),
            logical_validity: 0.0,
            confidence_scores: vec![],
            supporting_evidence: vec![],
            counterarguments: vec![],
        };
    }
    
    let mut chains = Vec::new();
    let mut explanations: Vec<(String, f32, Vec<String>)> = Vec::new();
    
    // Find facts that could explain the premise
    for triple in &knowledge.triples {
        if triple.object == premise {
            // This could be a cause
            let explanation = format!("{} because {}", premise, triple.subject);
            let mut supporting_facts = vec![format!("{} {} {}", triple.subject, triple.predicate, triple.object)];
            
            // Look for additional supporting evidence
            let additional_facts: Vec<String> = knowledge.triples.iter()
                .filter(|t| t.subject == triple.subject || t.object == triple.subject)
                .take(max_chain_length - 1)
                .map(|t| format!("{} {} {}", t.subject, t.predicate, t.object))
                .collect();
            
            supporting_facts.extend(additional_facts);
            
            let confidence = triple.confidence * (supporting_facts.len() as f32 / max_chain_length as f32);
            
            if confidence >= confidence_threshold {
                explanations.push((explanation, confidence, supporting_facts));
            }
        }
    }
    
    // Sort by confidence and convert to chains
    explanations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    for (explanation, confidence, facts) in explanations.iter().take(5) {
        let steps: Vec<Value> = facts.iter().enumerate().map(|(i, fact)| {
            json!({
                "step": i + 1,
                "evidence": fact,
                "supports": explanation
            })
        }).collect();
        
        chains.push(json!({
            "type": "abductive",
            "explanation": explanation,
            "confidence": confidence,
            "supporting_facts": facts.len(),
            "steps": steps,
            "plausibility": calculate_abductive_plausibility(*confidence, facts.len()),
            "validity": confidence * 0.65 // Abductive reasoning is less certain
        }));
    }
    
    let primary_conclusion = if chains.is_empty() {
        format!("No plausible explanations found for '{}'", premise)
    } else {
        chains[0]["explanation"].as_str().unwrap_or("").to_string()
    };
    
    let confidence_scores: Vec<f32> = chains.iter()
        .map(|c| c["confidence"].as_f64().unwrap_or(0.0) as f32)
        .collect();
    
    ReasoningResult {
        chains,
        primary_conclusion,
        logical_validity: 0.65, // Abductive reasoning seeks best explanation, not proof
        confidence_scores,
        supporting_evidence: extract_supporting_evidence(knowledge, premise),
        counterarguments: generate_alternative_explanations(&explanations),
    }
}

/// Execute analogical reasoning - similarity-based inference
pub async fn execute_analogical_reasoning(
    premise: &str,
    knowledge: &KnowledgeResult,
    max_chain_length: usize,
    confidence_threshold: f32,
) -> ReasoningResult {
    // Handle empty knowledge base
    if knowledge.triples.is_empty() {
        return ReasoningResult {
            chains: vec![],
            primary_conclusion: format!("No knowledge available for analogical reasoning about '{}'", premise),
            logical_validity: 0.0,
            confidence_scores: vec![],
            supporting_evidence: vec![],
            counterarguments: vec![],
        };
    }
    
    let mut chains = Vec::new();
    let mut analogies: Vec<(String, String, f32, Vec<(String, String)>)> = Vec::new();
    
    // Find entities with similar properties
    let premise_properties: HashMap<String, String> = knowledge.triples.iter()
        .filter(|t| t.subject == premise)
        .map(|t| (t.predicate.clone(), t.object.clone()))
        .collect();
    
    if premise_properties.is_empty() {
        return ReasoningResult {
            chains: vec![],
            primary_conclusion: format!("No properties found for '{}' to form analogies", premise),
            logical_validity: 0.0,
            confidence_scores: vec![],
            supporting_evidence: vec![],
            counterarguments: vec![],
        };
    }
    
    // Find other entities and calculate similarity
    let mut entity_similarities: HashMap<String, (f32, Vec<(String, String)>)> = HashMap::new();
    
    for triple in &knowledge.triples {
        if triple.subject != premise {
            let entity = &triple.subject;
            let property = (&triple.predicate, &triple.object);
            
            // Check if this property matches any of premise's properties
            for (premise_pred, premise_obj) in &premise_properties {
                if premise_pred == &triple.predicate || premise_obj == &triple.object {
                    let entry = entity_similarities.entry(entity.clone()).or_insert((0.0, vec![]));
                    entry.0 += triple.confidence;
                    entry.1.push((property.0.clone(), property.1.clone()));
                }
            }
        }
    }
    
    // Create analogies from similar entities
    for (entity, (similarity_score, shared_props)) in entity_similarities {
        let normalized_similarity = similarity_score / premise_properties.len() as f32;
        
        if normalized_similarity >= confidence_threshold && shared_props.len() >= 2 {
            // Find properties the analog has that premise might also have
            let analog_unique_props: Vec<(String, String)> = knowledge.triples.iter()
                .filter(|t| t.subject == entity && !premise_properties.contains_key(&t.predicate))
                .map(|t| (t.predicate.clone(), t.object.clone()))
                .collect();
            
            if !analog_unique_props.is_empty() {
                let conclusion = format!(
                    "{} is analogous to {}, suggesting {} might also have properties: {}",
                    premise,
                    entity,
                    premise,
                    analog_unique_props.iter()
                        .map(|(p, o)| format!("{} {}", p, o))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                
                analogies.push((entity, conclusion, normalized_similarity, shared_props));
            }
        }
    }
    
    // Convert to reasoning chains
    for (analog, conclusion, confidence, shared_props) in analogies.iter().take(max_chain_length) {
        let steps: Vec<Value> = shared_props.iter().enumerate().map(|(i, (pred, obj))| {
            json!({
                "step": i + 1,
                "similarity": format!("Both {} and {} have: {} {}", premise, analog, pred, obj),
                "confidence": confidence
            })
        }).collect();
        
        chains.push(json!({
            "type": "analogical",
            "source": premise,
            "analog": analog,
            "shared_properties": shared_props.len(),
            "confidence": confidence,
            "steps": steps,
            "conclusion": conclusion,
            "validity": confidence * 0.7 // Analogical reasoning has moderate validity
        }));
    }
    
    let primary_conclusion = if chains.is_empty() {
        format!("No suitable analogies found for '{}'", premise)
    } else {
        chains[0]["conclusion"].as_str().unwrap_or("").to_string()
    };
    
    let confidence_scores: Vec<f32> = chains.iter()
        .map(|c| c["confidence"].as_f64().unwrap_or(0.0) as f32)
        .collect();
    
    ReasoningResult {
        chains,
        primary_conclusion,
        logical_validity: 0.7, // Analogical reasoning has moderate validity
        confidence_scores,
        supporting_evidence: extract_supporting_evidence(knowledge, premise),
        counterarguments: vec![format!("Analogies may not hold if {} has unique properties", premise)],
    }
}

/// Generate alternative reasoning chains
pub async fn generate_alternative_reasoning_chains(
    premise: &str,
    knowledge: &KnowledgeResult,
    primary_result: &ReasoningResult,
    max_chain_length: usize,
) -> Vec<Value> {
    // Handle empty knowledge base
    if knowledge.triples.is_empty() {
        return vec![json!({
            "type": "none",
            "conclusion": "No alternative reasoning possible with empty knowledge base",
            "confidence": 0.0
        })];
    }
    
    let mut alternatives = Vec::new();
    
    // Try different reasoning types than the primary
    let reasoning_types = ["deductive", "inductive", "abductive", "analogical"];
    
    for reasoning_type in &reasoning_types {
        if !primary_result.chains.is_empty() && 
           primary_result.chains[0]["type"].as_str() == Some(reasoning_type) {
            continue; // Skip the primary type
        }
        
        // Generate a simple alternative based on type
        let alternative = match *reasoning_type {
            "deductive" => json!({
                "type": "deductive",
                "conclusion": format!("Alternative: If {} is true, logical consequences follow", premise),
                "confidence": 0.6,
                "note": "Alternative deductive path"
            }),
            "inductive" => json!({
                "type": "inductive", 
                "conclusion": format!("Alternative: Pattern suggests {} belongs to a larger category", premise),
                "confidence": 0.5,
                "note": "Alternative pattern recognition"
            }),
            "abductive" => json!({
                "type": "abductive",
                "conclusion": format!("Alternative: {} could be explained by hidden factors", premise),
                "confidence": 0.55,
                "note": "Alternative explanation"
            }),
            "analogical" => json!({
                "type": "analogical",
                "conclusion": format!("Alternative: {} shares properties with other concepts", premise),
                "confidence": 0.6,
                "note": "Alternative analogy"
            }),
            _ => continue,
        };
        
        alternatives.push(alternative);
        
        if alternatives.len() >= max_chain_length {
            break;
        }
    }
    
    alternatives
}

// Helper functions

/// Detect contradictions in the knowledge base
fn detect_contradictions(knowledge: &KnowledgeResult) -> Vec<String> {
    let mut contradictions = Vec::new();
    let mut facts_by_subject: HashMap<String, Vec<&Triple>> = HashMap::new();
    
    // Group facts by subject
    for triple in &knowledge.triples {
        facts_by_subject.entry(triple.subject.clone()).or_insert_with(Vec::new).push(triple);
    }
    
    // Check for contradictions
    for (subject, triples) in &facts_by_subject {
        let mut predicates_map: HashMap<String, Vec<&str>> = HashMap::new();
        
        for triple in triples {
            let base_predicate = triple.predicate
                .replace("_not", "")
                .replace("isn't", "is")
                .replace("cannot", "can")
                .replace("doesn't", "does");
            
            predicates_map.entry(base_predicate.clone())
                .or_insert_with(Vec::new)
                .push(&triple.predicate);
        }
        
        // Check for contradictory predicates
        for (base_pred, variations) in &predicates_map {
            if variations.len() > 1 {
                let has_positive = variations.iter().any(|p| !p.contains("not") && !p.contains("n't"));
                let has_negative = variations.iter().any(|p| p.contains("not") || p.contains("n't"));
                
                if has_positive && has_negative {
                    contradictions.push(format!(
                        "Contradiction for '{}': both affirming and negating '{}'",
                        subject, base_pred
                    ));
                }
            }
        }
        
        // Check for mutually exclusive properties
        let properties: Vec<(&str, &str)> = triples.iter()
            .filter(|t| t.predicate == "is" || t.predicate == "type")
            .map(|t| (t.predicate.as_str(), t.object.as_str()))
            .collect();
            
        if properties.len() > 1 {
            // Check for known mutually exclusive categories
            let exclusive_pairs = [
                ("alive", "dead"),
                ("true", "false"),
                ("valid", "invalid"),
                ("possible", "impossible"),
            ];
            
            for (_, obj1) in &properties {
                for (_, obj2) in &properties {
                    for (ex1, ex2) in &exclusive_pairs {
                        if (obj1.contains(ex1) && obj2.contains(ex2)) || 
                           (obj1.contains(ex2) && obj2.contains(ex1)) {
                            contradictions.push(format!(
                                "Contradiction for '{}': cannot be both '{}' and '{}'",
                                subject, obj1, obj2
                            ));
                        }
                    }
                }
            }
        }
    }
    
    contradictions
}

fn calculate_deductive_validity(chain: &[ReasoningStep]) -> f32 {
    if chain.is_empty() {
        return 0.0;
    }
    
    let avg_confidence: f32 = chain.iter().map(|s| s.confidence).sum::<f32>() / chain.len() as f32;
    let consistency_factor = 1.0 - (0.1 * chain.len() as f32).min(0.5); // Longer chains are less certain
    
    avg_confidence * consistency_factor
}

fn calculate_abductive_plausibility(confidence: f32, supporting_facts: usize) -> f32 {
    let fact_weight = (supporting_facts as f32 / 10.0).min(1.0);
    confidence * 0.7 + fact_weight * 0.3
}

fn extract_supporting_evidence(knowledge: &KnowledgeResult, premise: &str) -> Vec<String> {
    let mut evidence: Vec<String> = knowledge.triples.iter()
        .filter(|t| t.subject == premise || t.object == premise)
        .take(5)
        .map(|t| format!("{} {} {}", t.subject, t.predicate, t.object))
        .collect();
    
    // Add any global contradictions as warnings
    let global_contradictions = detect_contradictions(knowledge);
    if !global_contradictions.is_empty() {
        evidence.push(format!("⚠️ {} contradiction(s) detected in knowledge base", global_contradictions.len()));
    }
    
    evidence
}

fn generate_inductive_counterarguments(patterns: &HashMap<String, Vec<String>>) -> Vec<String> {
    let mut counterarguments = Vec::new();
    
    for (pattern, instances) in patterns {
        if instances.len() < 5 {
            counterarguments.push(format!(
                "Pattern '{}' based on only {} instances may not be representative",
                pattern, instances.len()
            ));
        }
    }
    
    if counterarguments.is_empty() {
        counterarguments.push("Inductive conclusions may not hold for all cases".to_string());
    }
    
    counterarguments
}

fn generate_alternative_explanations(explanations: &[(String, f32, Vec<String>)]) -> Vec<String> {
    if explanations.len() <= 1 {
        return vec!["Other unexplored explanations may exist".to_string()];
    }
    
    explanations.iter()
        .skip(1)
        .take(2)
        .map(|(exp, conf, _)| format!("Alternative: {} (confidence: {:.2})", exp, conf))
        .collect()
}

pub fn calculate_chain_confidence(chains: &[Value]) -> f32 {
    if chains.is_empty() {
        return 0.0;
    }
    
    let total: f32 = chains.iter()
        .filter_map(|c| c["confidence"].as_f64())
        .map(|c| c as f32)
        .sum();
    
    total / chains.len() as f32
}