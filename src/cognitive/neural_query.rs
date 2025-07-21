use crate::graph::Graph;
use crate::cognitive::types::QueryContext;
use crate::error::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Advanced neural query processor for concept extraction and understanding
pub struct NeuralQueryProcessor {
    pub graph: Arc<Graph>,
}

#[derive(Debug, Clone)]
pub struct ExtractedConcept {
    pub text: String,
    pub concept_type: ConceptType,
    pub confidence: f32,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConceptType {
    Entity,
    Relationship,
    Property,
    Question,
    Temporal,
    Spatial,
    Quantitative,
    Comparative,
    Causal,
    Hypothetical,
}

#[derive(Debug, Clone)]
pub struct QueryUnderstanding {
    pub intent: QueryIntent,
    pub concepts: Vec<ExtractedConcept>,
    pub relationships: Vec<(String, String, String)>, // (subject, predicate, object)
    pub constraints: QueryConstraints,
    pub domain_hints: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    Factual,           // "What is X?"
    Relational,        // "How does X relate to Y?"
    Causal,           // "What caused X?"
    Temporal,         // "When did X happen?"
    Comparative,      // "Which is better, X or Y?"
    Compositional,    // "What makes up X?"
    Hierarchical,     // "What types of X exist?"
    Counterfactual,   // "What if X?"
    Creative,         // "Generate ideas about X"
    Meta,            // "What do we know about X?"
    MultiHop,        // Complex queries requiring multiple steps
}

#[derive(Debug, Clone)]
pub struct QueryConstraints {
    pub temporal_bounds: Option<(String, String)>,
    pub temporal_context: Option<String>,
    pub spatial_bounds: Option<String>,
    pub quantitative_bounds: Option<(f32, f32)>,
    pub required_properties: Vec<String>,
    pub excluded_concepts: Vec<String>,
}

impl NeuralQueryProcessor {
    pub fn new(graph: Arc<Graph>) -> Self {
        Self { graph }
    }

    /// Enhanced neural query function with advanced concept extraction
    pub fn neural_query(&self, query: &str, context: &QueryContext) -> Result<QueryUnderstanding> {
        // Step 1: Tokenize and analyze query structure
        let tokens = self.tokenize_query(query);
        
        // Step 2: Identify query intent
        let intent = self.identify_intent(&tokens, query);
        
        // Step 3: Extract concepts with context
        let concepts = self.extract_concepts(&tokens, query, &intent);
        
        // Step 4: Extract relationships
        let relationships = self.extract_relationships(&tokens, &concepts);
        
        // Step 5: Extract constraints
        let constraints = self.extract_constraints(&tokens, query);
        
        // Step 6: Identify domain hints
        let domain_hints = self.identify_domains(&concepts, context);
        
        Ok(QueryUnderstanding {
            intent,
            concepts,
            relationships,
            constraints,
            domain_hints,
        })
    }

    /// Understand query intent and extract concepts (alias for neural_query)
    pub fn understand_query(&self, query: &str, context: Option<&QueryContext>) -> Result<QueryUnderstanding> {
        let default_context = QueryContext::new();
        let context = context.unwrap_or(&default_context);
        self.neural_query(query, context)
    }

    fn tokenize_query(&self, query: &str) -> Vec<String> {
        // Advanced tokenization with linguistic analysis
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_quotes = false;
        let mut quote_char = '"';
        
        for ch in query.chars() {
            match ch {
                '"' | '\'' => {
                    if !in_quotes {
                        in_quotes = true;
                        quote_char = ch;
                        if !current_token.is_empty() {
                            tokens.push(current_token.to_lowercase());
                            current_token.clear();
                        }
                    } else if ch == quote_char {
                        in_quotes = false;
                        if !current_token.is_empty() {
                            tokens.push(current_token.to_lowercase());
                            current_token.clear();
                        }
                    } else {
                        current_token.push(ch);
                    }
                }
                '?' | '!' | '.' if !in_quotes => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.to_lowercase());
                        current_token.clear();
                    }
                    tokens.push(ch.to_string());
                }
                ',' | ';' | ':' if !in_quotes => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.to_lowercase());
                        current_token.clear();
                    }
                    tokens.push(ch.to_string());
                }
                c if c.is_whitespace() && !in_quotes => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.to_lowercase());
                        current_token.clear();
                    }
                }
                _ => {
                    current_token.push(ch.to_lowercase().next().unwrap_or(ch));
                }
            }
        }
        
        if !current_token.is_empty() {
            tokens.push(current_token.to_lowercase());
        }
        
        // Remove common stop words but keep important ones
        let keep_words = &["what", "how", "when", "where", "why", "who", "which", "not", "no"];
        tokens.into_iter()
            .filter(|token| {
                !token.is_empty() && 
                (token.len() > 2 || keep_words.contains(&token.as_str()) || token.chars().any(|c| !c.is_alphabetic()))
            })
            .collect()
    }

    fn identify_intent(&self, tokens: &[String], query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();
        
        // Check for specific patterns
        if query_lower.contains("what if") || query_lower.contains("would happen") {
            QueryIntent::Counterfactual
        } else if query_lower.contains("how") && query_lower.contains("relate") {
            QueryIntent::Relational
        } else if query_lower.contains("what caused") || query_lower.contains("why") {
            QueryIntent::Causal
        } else if query_lower.contains("when") || query_lower.contains("time") {
            QueryIntent::Temporal
        } else if query_lower.contains("which") || query_lower.contains("better") || 
                  query_lower.contains("compare") {
            QueryIntent::Comparative
        } else if query_lower.contains("made of") || query_lower.contains("consists") ||
                  query_lower.contains("make up") || query_lower.contains("elements") {
            QueryIntent::Compositional
        } else if query_lower.contains("types") || query_lower.contains("kinds") ||
                  query_lower.contains("belong") {
            QueryIntent::Hierarchical
        } else if query_lower.contains("create") || query_lower.contains("generate") ||
                  query_lower.contains("ideas") {
            QueryIntent::Creative
        } else if query_lower.contains("know about") || query_lower.contains("information") {
            QueryIntent::Meta
        } else if self.is_multi_hop_query(tokens, &query_lower) {
            QueryIntent::MultiHop
        } else {
            QueryIntent::Factual
        }
    }

    fn is_multi_hop_query(&self, tokens: &[String], query: &str) -> bool {
        // Check for indicators of multi-hop reasoning
        let multi_hop_indicators = vec![
            "led to", "influence", "connect", "through", "via",
            "indirect", "result in", "consequence", "chain"
        ];
        
        multi_hop_indicators.iter().any(|ind| query.contains(ind)) ||
        tokens.iter().filter(|t| vec!["and", "then", "which"].contains(&t.as_str())).count() > 1
    }

    fn extract_concepts(&self, tokens: &[String], query: &str, intent: &QueryIntent) -> Vec<ExtractedConcept> {
        let mut concepts = Vec::new();
        let query_lower = query.to_lowercase();
        
        // Extract named entities (simplified - real implementation would use NER)
        let potential_entities = self.find_potential_entities(tokens);
        
        for entity in potential_entities {
            let concept_type = self.classify_concept(&entity, intent);
            let confidence = self.calculate_concept_confidence(&entity, &query_lower);
            
            concepts.push(ExtractedConcept {
                text: entity.clone(),
                concept_type,
                confidence,
                properties: self.extract_concept_properties(&entity, tokens),
            });
        }
        
        // Extract temporal concepts
        if query_lower.contains("when") || query_lower.contains("year") || 
           query_lower.contains("time") {
            concepts.push(ExtractedConcept {
                text: "temporal_query".to_string(),
                concept_type: ConceptType::Temporal,
                confidence: 0.9,
                properties: HashMap::new(),
            });
        }
        
        // Extract quantitative concepts
        for token in tokens {
            if token.parse::<f32>().is_ok() || token.contains("how many") || 
               token.contains("how much") {
                concepts.push(ExtractedConcept {
                    text: token.clone(),
                    concept_type: ConceptType::Quantitative,
                    confidence: 0.95,
                    properties: HashMap::new(),
                });
            }
        }
        
        concepts
    }

    fn find_potential_entities(&self, tokens: &[String]) -> Vec<String> {
        let mut entities = Vec::new();
        let mut i = 0;
        
        // Look for capitalized words or known patterns
        while i < tokens.len() {
            // Check if token is likely an entity
            if self.is_entity_candidate(&tokens[i]) {
                let mut entity = tokens[i].clone();
                
                // Check for multi-word entities
                let mut j = i + 1;
                while j < tokens.len() && self.could_be_continuation(&tokens[j]) {
                    entity.push(' ');
                    entity.push_str(&tokens[j]);
                    j += 1;
                }
                
                entities.push(entity);
                i = j;
            } else {
                i += 1;
            }
        }
        
        // Also check against graph for known entities
        let graph_entities = self.match_graph_entities(tokens);
        entities.extend(graph_entities);
        
        // Deduplicate
        entities.sort();
        entities.dedup();
        
        entities
    }

    fn is_entity_candidate(&self, token: &str) -> bool {
        // Check if token could be an entity
        let first_char = token.chars().next().unwrap_or(' ');
        first_char.is_uppercase() || 
        self.is_known_entity_type(token) ||
        token.len() > 3 && !self.is_stop_word(token)
    }

    fn could_be_continuation(&self, token: &str) -> bool {
        // Check if token could be part of a multi-word entity
        let first_char = token.chars().next().unwrap_or(' ');
        first_char.is_uppercase() || 
        vec!["of", "the", "and", "&"].contains(&token)
    }

    fn is_known_entity_type(&self, token: &str) -> bool {
        let entity_types = vec![
            "einstein", "newton", "relativity", "physics", "chemistry",
            "computer", "internet", "water", "dna", "evolution",
            "theory", "model", "concept", "process", "system"
        ];
        entity_types.contains(&token.to_lowercase().as_str())
    }

    fn is_stop_word(&self, token: &str) -> bool {
        let stop_words = vec![
            "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "a", "an"
        ];
        stop_words.contains(&token.to_lowercase().as_str())
    }

    fn match_graph_entities(&self, tokens: &[String]) -> Vec<String> {
        let mut matched = Vec::new();
        
        // Try to match tokens against known graph nodes
        // This is a simplified version - real implementation would use fuzzy matching
        for window_size in (1..=3).rev() {
            for window in tokens.windows(window_size) {
                let candidate = window.join(" ");
                if self.graph_contains_entity(&candidate) {
                    matched.push(candidate);
                }
            }
        }
        
        matched
    }

    fn graph_contains_entity(&self, entity: &str) -> bool {
        // Check if entity exists in graph (simplified)
        // In real implementation, would check actual graph nodes
        let known_entities = vec![
            "einstein", "newton", "general relativity", "classical physics",
            "water", "hydrogen", "oxygen", "dna", "evolution",
            "computer", "internet", "artificial intelligence",
            "world war ii", "manhattan project", "industrial revolution"
        ];
        
        known_entities.iter().any(|e| e.eq_ignore_ascii_case(entity))
    }

    fn classify_concept(&self, entity: &str, intent: &QueryIntent) -> ConceptType {
        let entity_lower = entity.to_lowercase();
        
        // Classify based on entity content and query intent
        if matches!(intent, QueryIntent::Temporal) || 
           entity_lower.contains("when") || entity_lower.contains("time") {
            ConceptType::Temporal
        } else if matches!(intent, QueryIntent::Causal) ||
                  entity_lower.contains("cause") || entity_lower.contains("effect") {
            ConceptType::Causal
        } else if matches!(intent, QueryIntent::Comparative) {
            ConceptType::Comparative
        } else if matches!(intent, QueryIntent::Counterfactual) {
            ConceptType::Hypothetical
        } else if entity_lower.contains("how") || entity_lower.contains("why") ||
                  entity_lower.contains("what") {
            ConceptType::Question
        } else if self.is_relationship_word(&entity_lower) {
            ConceptType::Relationship
        } else if entity_lower.contains("property") || entity_lower.contains("attribute") {
            ConceptType::Property
        } else {
            ConceptType::Entity
        }
    }

    fn is_relationship_word(&self, word: &str) -> bool {
        let relationship_words = vec![
            "relates", "connected", "influences", "causes", "enables",
            "contains", "consists", "includes", "belongs", "leads",
            "results", "depends", "requires", "uses", "creates"
        ];
        relationship_words.iter().any(|r| word.contains(r))
    }

    fn calculate_concept_confidence(&self, entity: &str, query: &str) -> f32 {
        // Calculate confidence based on various factors
        let mut confidence: f32 = 0.5;
        
        // Exact match in query
        if query.contains(&entity.to_lowercase()) {
            confidence += 0.3;
        }
        
        // Known entity in graph
        if self.graph_contains_entity(entity) {
            confidence += 0.2;
        }
        
        confidence.min(1.0)
    }

    fn extract_concept_properties(&self, entity: &str, tokens: &[String]) -> HashMap<String, String> {
        let mut properties = HashMap::new();
        
        // Look for properties near the entity in the query
        if let Some(pos) = tokens.iter().position(|t| t.contains(&entity.to_lowercase())) {
            // Check tokens before and after
            if pos > 0 && self.is_property_indicator(&tokens[pos - 1]) {
                properties.insert("modifier".to_string(), tokens[pos - 1].clone());
            }
            if pos + 1 < tokens.len() && self.is_property_value(&tokens[pos + 1]) {
                properties.insert("attribute".to_string(), tokens[pos + 1].clone());
            }
        }
        
        properties
    }

    fn is_property_indicator(&self, token: &str) -> bool {
        vec!["modern", "classical", "theoretical", "practical", "new", "old", 
             "direct", "indirect", "primary", "secondary"].contains(&token)
    }

    fn is_property_value(&self, token: &str) -> bool {
        !self.is_stop_word(token) && token.len() > 2
    }

    fn extract_relationships(&self, tokens: &[String], concepts: &[ExtractedConcept]) -> Vec<(String, String, String)> {
        let mut relationships = Vec::new();
        
        // Look for relationship patterns
        let entities: Vec<_> = concepts.iter()
            .filter(|c| c.concept_type == ConceptType::Entity)
            .collect();
        
        if entities.len() >= 2 {
            // Look for relationship words between entities
            for i in 0..entities.len() - 1 {
                for j in i + 1..entities.len() {
                    if let Some(relation) = self.find_relationship_between(
                        &entities[i].text, 
                        &entities[j].text, 
                        tokens
                    ) {
                        relationships.push((
                            entities[i].text.clone(),
                            relation,
                            entities[j].text.clone()
                        ));
                    }
                }
            }
        }
        
        relationships
    }

    fn find_relationship_between(&self, entity1: &str, entity2: &str, tokens: &[String]) -> Option<String> {
        // Find relationship words between two entities
        let pos1 = tokens.iter().position(|t| t.contains(&entity1.to_lowercase()));
        let pos2 = tokens.iter().position(|t| t.contains(&entity2.to_lowercase()));
        
        if let (Some(p1), Some(p2)) = (pos1, pos2) {
            let start = p1.min(p2);
            let end = p1.max(p2);
            
            for i in start + 1..end {
                if self.is_relationship_word(&tokens[i]) {
                    return Some(tokens[i].clone());
                }
            }
        }
        
        None
    }

    fn extract_constraints(&self, tokens: &[String], query: &str) -> QueryConstraints {
        let mut constraints = QueryConstraints {
            temporal_bounds: None,
            temporal_context: None,
            spatial_bounds: None,
            quantitative_bounds: None,
            required_properties: Vec::new(),
            excluded_concepts: Vec::new(),
        };
        
        // Extract temporal constraints
        if let Some(temporal) = self.extract_temporal_bounds(tokens, query) {
            constraints.temporal_bounds = Some(temporal);
        }
        
        // Extract quantitative constraints
        if let Some(quant) = self.extract_quantitative_bounds(tokens) {
            constraints.quantitative_bounds = Some(quant);
        }
        
        // Extract required properties
        constraints.required_properties = self.extract_required_properties(tokens);
        
        // Extract exclusions
        constraints.excluded_concepts = self.extract_exclusions(tokens);
        
        constraints
    }

    fn extract_temporal_bounds(&self, tokens: &[String], query: &str) -> Option<(String, String)> {
        // Look for temporal indicators
        let temporal_patterns = vec![
            (r"between (\d{4}) and (\d{4})", "year_range"),
            (r"after (\d{4})", "after_year"),
            (r"before (\d{4})", "before_year"),
            (r"during (.+)", "during_period"),
        ];
        
        // Simplified extraction - real implementation would use regex
        if query.contains("between") && query.contains("and") {
            // Extract years or periods
            let numbers: Vec<_> = tokens.iter()
                .filter_map(|t| t.parse::<i32>().ok())
                .collect();
            
            if numbers.len() >= 2 {
                return Some((numbers[0].to_string(), numbers[1].to_string()));
            }
        }
        
        None
    }

    fn extract_quantitative_bounds(&self, tokens: &[String]) -> Option<(f32, f32)> {
        let numbers: Vec<f32> = tokens.iter()
            .filter_map(|t| t.parse::<f32>().ok())
            .collect();
        
        if numbers.len() >= 2 {
            let min = numbers.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = numbers.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            Some((min, max))
        } else if numbers.len() == 1 {
            Some((0.0, numbers[0]))
        } else {
            None
        }
    }

    fn extract_required_properties(&self, tokens: &[String]) -> Vec<String> {
        let mut properties = Vec::new();
        
        // Look for property indicators
        for i in 0..tokens.len() {
            if tokens[i] == "with" || tokens[i] == "having" {
                if i + 1 < tokens.len() {
                    properties.push(tokens[i + 1].clone());
                }
            }
        }
        
        properties
    }

    fn extract_exclusions(&self, tokens: &[String]) -> Vec<String> {
        let mut exclusions = Vec::new();
        
        // Look for exclusion indicators
        for i in 0..tokens.len() {
            if tokens[i] == "not" || tokens[i] == "except" || tokens[i] == "without" {
                if i + 1 < tokens.len() {
                    exclusions.push(tokens[i + 1].clone());
                }
            }
        }
        
        exclusions
    }

    fn identify_domains(&self, concepts: &[ExtractedConcept], context: &QueryContext) -> Vec<String> {
        let mut domains = HashSet::new();
        
        // Use provided domain if available
        if let Some(ref domain) = context.domain {
            domains.insert(domain.clone());
        }
        
        // Infer domains from concepts
        for concept in concepts {
            if let Some(domain) = self.infer_domain(&concept.text) {
                domains.insert(domain);
            }
        }
        
        domains.into_iter().map(|s| s.to_string()).collect()
    }

    fn infer_domain(&self, concept: &str) -> Option<String> {
        let concept_lower = concept.to_lowercase();
        
        let domain_mappings = vec![
            (vec!["einstein", "newton", "relativity", "gravity", "physics"], "physics"),
            (vec!["water", "hydrogen", "oxygen", "element", "molecule"], "chemistry"),
            (vec!["dna", "evolution", "biology", "life"], "biology"),
            (vec!["computer", "internet", "ai", "artificial intelligence"], "technology"),
            (vec!["war", "revolution", "project", "event"], "history"),
            (vec!["calculus", "infinity", "mathematics", "equation"], "mathematics"),
        ];
        
        for (keywords, domain) in domain_mappings {
            if keywords.iter().any(|k| concept_lower.contains(k)) {
                return Some(domain.to_string());
            }
        }
        
        None
    }

    /// Find entities in the query by matching against the graph
    pub fn find_entities_in_query(&self, query: &str) -> Vec<(String, f32)> {
        let understanding = self.neural_query(query, &QueryContext::default()).unwrap();
        
        understanding.concepts.into_iter()
            .filter(|c| c.concept_type == ConceptType::Entity)
            .map(|c| (c.text, c.confidence))
            .collect()
    }

    /// Extract the main question type from the query
    pub fn extract_question_type(&self, query: &str) -> QueryIntent {
        let understanding = self.neural_query(query, &QueryContext::default()).unwrap();
        understanding.intent
    }

    /// Get related concepts for a given entity
    pub fn get_related_concepts(&self, entity: &str, _max_depth: usize) -> Vec<String> {
        // This would traverse the graph to find related concepts
        // For now, return a mock implementation
        vec![format!("{}_related", entity)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Helper function to create a test processor
    fn create_test_processor() -> NeuralQueryProcessor {
        let graph = Arc::new(Graph::new());
        NeuralQueryProcessor::new(graph)
    }
    
    #[test]
    fn test_identify_query_intent() {
        let processor = create_test_processor();
        let test_cases = vec![
            ("What is quantum computing?", QueryIntent::Factual),
            ("What makes up an atom?", QueryIntent::Compositional),
            ("How does photosynthesis relate to solar panels?", QueryIntent::Relational),
            ("Why is the sky blue?", QueryIntent::Causal),
            ("What if we had no gravity?", QueryIntent::Counterfactual),
            ("Which is better, solar or wind energy?", QueryIntent::Comparative),
            ("When did the moon landing happen?", QueryIntent::Temporal),
            ("What types of renewable energy exist?", QueryIntent::Hierarchical),
            ("Generate ideas about space exploration", QueryIntent::Creative),
            ("What do we know about black holes?", QueryIntent::Meta),
        ];
        
        for (query, expected_intent) in test_cases {
            let tokens = processor.tokenize_query(query);
            let intent = processor.identify_intent(&tokens, query);
            assert_eq!(
                intent, expected_intent,
                "Failed for query: '{}', expected {:?}, got {:?}",
                query, expected_intent, intent
            );
        }
    }

    #[test]
    fn test_concept_extraction() {
        let processor = create_test_processor();
        let test_cases = vec![
            ("What is quantum computing?", vec!["quantum", "computing"]),
            ("How does DNA relate to heredity?", vec!["DNA", "heredity"]),
            ("Einstein's theory of relativity", vec!["einstein", "theory", "relativity"]),
        ];
        
        for (query, expected_concepts) in test_cases {
            let tokens = processor.tokenize_query(query);
            let intent = processor.identify_intent(&tokens, query);
            let concepts = processor.extract_concepts(&tokens, query, &intent);
            
            // Check that we extracted some concepts
            assert!(!concepts.is_empty(), "No concepts extracted for query: '{}'", query);
            
            // Check that expected concepts are found (case-insensitive)
            for expected in expected_concepts {
                let found = concepts.iter().any(|c| 
                    c.text.to_lowercase().contains(&expected.to_lowercase()) ||
                    expected.to_lowercase().contains(&c.text.to_lowercase())
                );
                assert!(found, "Expected concept '{}' not found in query: '{}'", expected, query);
            }
        }
    }

    #[test]
    fn test_relationship_extraction() {
        let processor = create_test_processor();
        
        // Create test concepts
        let concepts = vec![
            ExtractedConcept {
                text: "photosynthesis".to_string(),
                concept_type: ConceptType::Entity,
                confidence: 0.9,
                properties: HashMap::new(),
            },
            ExtractedConcept {
                text: "solar energy".to_string(),
                concept_type: ConceptType::Entity,
                confidence: 0.8,
                properties: HashMap::new(),
            },
        ];
        
        let query = "How does photosynthesis relate to solar energy?";
        let tokens = processor.tokenize_query(query);
        let relationships = processor.extract_relationships(&tokens, &concepts);
        
        // Should find a relationship between the concepts
        assert!(!relationships.is_empty(), "No relationships extracted");
        
        // Check that the relationship involves our test concepts
        let found_relationship = relationships.iter().any(|(subj, _pred, obj)| {
            (subj.contains("photosynthesis") && obj.contains("solar")) ||
            (subj.contains("solar") && obj.contains("photosynthesis"))
        });
        assert!(found_relationship, "Expected relationship not found between concepts");
    }

    #[test]
    fn test_constraint_extraction() {
        let processor = create_test_processor();
        let test_cases = vec![
            ("What happened between 1940 and 1945?", true, false),
            ("How many electrons does carbon have?", false, true),
            ("Show me results with high accuracy", false, false),
            ("Find concepts not related to chemistry", false, false),
        ];
        
        for (query, expect_temporal, expect_quantitative) in test_cases {
            let tokens = processor.tokenize_query(query);
            let constraints = processor.extract_constraints(&tokens, query);
            
            if expect_temporal {
                assert!(constraints.temporal_bounds.is_some(), 
                    "Expected temporal constraints for query: '{}'", query);
            }
            
            if expect_quantitative {
                assert!(constraints.quantitative_bounds.is_some(), 
                    "Expected quantitative constraints for query: '{}'", query);
            }
        }
    }

    #[test]
    fn test_domain_inference() {
        let processor = create_test_processor();
        let test_cases = vec![
            ("einstein", Some("physics")),
            ("DNA", Some("biology")),
            ("computer", Some("technology")),
            ("water", Some("chemistry")),
            ("revolution", Some("history")),
            ("calculus", Some("mathematics")),
            ("random_concept", None),
        ];
        
        for (concept, expected_domain) in test_cases {
            let result = processor.infer_domain(concept);
            match expected_domain {
                Some(expected) => {
                    assert!(result.is_some(), "Expected domain for concept: '{}'", concept);
                    let actual = result.unwrap();
                    assert_eq!(actual, expected, 
                        "Wrong domain for concept: '{}', expected '{}', got '{}'", 
                        concept, expected, actual);
                }
                None => {
                    assert!(result.is_none(), "Unexpected domain found for concept: '{}'", concept);
                }
            }
        }
    }

    #[test]
    fn test_tokenize_query() {
        let processor = create_test_processor();
        let test_cases = vec![
            ("What is quantum computing?", vec!["what", "quantum", "computing", "?"]),
            ("How does DNA relate to heredity?", vec!["how", "does", "dna", "relate", "heredity", "?"]),
            ("Einstein's theory", vec!["einstein", "theory"]),
        ];
        
        for (query, expected_tokens) in test_cases {
            let tokens = processor.tokenize_query(query);
            
            // Check that expected tokens are present (allowing for additional tokens)
            for expected in expected_tokens {
                assert!(tokens.contains(&expected.to_string()), 
                    "Expected token '{}' not found in tokenized query: '{}'", expected, query);
            }
        }
    }

    #[test]
    fn test_is_multi_hop_query() {
        let processor = create_test_processor();
        let test_cases = vec![
            ("How did Einstein's theories influence modern physics?", true),
            ("What led to the development of quantum computing?", true),
            ("What is water?", false),
            ("How does photosynthesis work?", false),
        ];
        
        for (query, expected_multi_hop) in test_cases {
            let tokens = processor.tokenize_query(query);
            let is_multi_hop = processor.is_multi_hop_query(&tokens, query);
            assert_eq!(is_multi_hop, expected_multi_hop, 
                "Multi-hop detection failed for query: '{}'", query);
        }
    }

    #[test]
    fn test_find_potential_entities() {
        let processor = create_test_processor();
        let test_cases = vec![
            (vec!["what", "is", "quantum", "computing"], vec!["quantum", "computing"]),
            (vec!["einstein", "theory", "of", "relativity"], vec!["einstein", "theory", "relativity"]),
            (vec!["how", "does", "dna", "work"], vec!["dna"]),
        ];
        
        for (tokens, expected_entities) in test_cases {
            let entities = processor.find_potential_entities(&tokens.iter().map(|s| s.to_string()).collect::<Vec<_>>());
            
            // Check that some entities were found
            assert!(!entities.is_empty(), "No entities found for tokens: {:?}", tokens);
            
            // Check that expected entities are found (case-insensitive)
            for expected in expected_entities {
                let found = entities.iter().any(|e| 
                    e.to_lowercase().contains(&expected.to_lowercase())
                );
                assert!(found, "Expected entity '{}' not found in tokens: {:?}", expected, tokens);
            }
        }
    }
}

