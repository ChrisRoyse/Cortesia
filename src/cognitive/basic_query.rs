/// Basic query processing using graph-based methods
use crate::cognitive::types::QueryContext;
use crate::error::Result;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BasicQueryProcessor {
    // Basic processor doesn't need graph reference for simple operations
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

impl BasicQueryProcessor {
    pub fn new() -> Self {
        Self {}
    }

    /// Basic query understanding using graph-based processing
    pub fn understand_query(&self, query: &str, _context: Option<&QueryContext>) -> Result<QueryUnderstanding> {
        let tokens = self.tokenize_query(query);
        let intent = self.identify_intent(&tokens, query);
        let concepts = self.extract_concepts(&tokens, query, &intent);
        let relationships = self.extract_relationships(&tokens, &concepts);
        let constraints = self.extract_constraints(&tokens, query);
        let domain_hints = self.identify_domains(&concepts);

        Ok(QueryUnderstanding {
            intent,
            concepts,
            relationships,
            constraints,
            domain_hints,
        })
    }

    fn tokenize_query(&self, query: &str) -> Vec<String> {
        // Simple tokenization
        query.split_whitespace()
            .map(|s| s.to_lowercase())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .collect()
    }

    fn identify_intent(&self, _tokens: &[String], query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("what if") || query_lower.contains("would happen") {
            QueryIntent::Counterfactual
        } else if query_lower.contains("how") && query_lower.contains("relate") {
            QueryIntent::Relational
        } else if query_lower.contains("what caused") || query_lower.contains("why") {
            QueryIntent::Causal
        } else if query_lower.contains("when") || query_lower.contains("time") {
            QueryIntent::Temporal
        } else if query_lower.contains("which") || query_lower.contains("better") {
            QueryIntent::Comparative
        } else if query_lower.contains("made of") || query_lower.contains("consists") {
            QueryIntent::Compositional
        } else if query_lower.contains("types") || query_lower.contains("kinds") {
            QueryIntent::Hierarchical
        } else if query_lower.contains("create") || query_lower.contains("generate") {
            QueryIntent::Creative
        } else if query_lower.contains("know about") || query_lower.contains("information") {
            QueryIntent::Meta
        } else {
            QueryIntent::Factual
        }
    }

    fn extract_concepts(&self, tokens: &[String], query: &str, intent: &QueryIntent) -> Vec<ExtractedConcept> {
        let mut concepts = Vec::new();
        
        // Basic concept extraction - look for capitalized words and important terms
        for token in tokens {
            if token.len() > 3 && !self.is_stop_word(token) {
                let concept_type = self.classify_concept(token, intent);
                let confidence = self.calculate_basic_confidence(token, query);
                
                concepts.push(ExtractedConcept {
                    text: token.clone(),
                    concept_type,
                    confidence,
                    properties: HashMap::new(),
                });
            }
        }
        
        concepts
    }

    fn extract_relationships(&self, _tokens: &[String], concepts: &[ExtractedConcept]) -> Vec<(String, String, String)> {
        let mut relationships = Vec::new();
        
        // Simple relationship extraction - if we have multiple entities, assume they're related
        let entities: Vec<_> = concepts.iter()
            .filter(|c| c.concept_type == ConceptType::Entity)
            .collect();
        
        if entities.len() >= 2 {
            relationships.push((
                entities[0].text.clone(),
                "relates_to".to_string(),
                entities[1].text.clone()
            ));
        }
        
        relationships
    }

    fn extract_constraints(&self, _tokens: &[String], _query: &str) -> QueryConstraints {
        QueryConstraints {
            temporal_bounds: None,
            temporal_context: None,
            spatial_bounds: None,
            quantitative_bounds: None,
            required_properties: Vec::new(),
            excluded_concepts: Vec::new(),
        }
    }

    fn identify_domains(&self, concepts: &[ExtractedConcept]) -> Vec<String> {
        let mut domains = Vec::new();
        
        for concept in concepts {
            if let Some(domain) = self.infer_domain(&concept.text) {
                domains.push(domain);
            }
        }
        
        domains.sort();
        domains.dedup();
        domains
    }

    fn classify_concept(&self, token: &str, intent: &QueryIntent) -> ConceptType {
        match intent {
            QueryIntent::Temporal => ConceptType::Temporal,
            QueryIntent::Causal => ConceptType::Causal,
            QueryIntent::Comparative => ConceptType::Comparative,
            QueryIntent::Counterfactual => ConceptType::Hypothetical,
            _ => {
                if token.contains("how") || token.contains("what") || token.contains("why") {
                    ConceptType::Question
                } else if self.is_relationship_word(token) {
                    ConceptType::Relationship
                } else {
                    ConceptType::Entity
                }
            }
        }
    }

    fn calculate_basic_confidence(&self, token: &str, query: &str) -> f32 {
        let mut confidence: f32 = 0.5;
        
        if query.to_lowercase().contains(&token.to_lowercase()) {
            confidence += 0.3;
        }
        
        if token.len() > 5 {
            confidence += 0.1;
        }
        
        confidence.min(1.0)
    }

    fn is_stop_word(&self, token: &str) -> bool {
        let stop_words = ["the", "is", "are", "was", "were", "be", "been", "being",
                         "have", "has", "had", "do", "does", "did", "will", "would",
                         "could", "should", "may", "might", "must", "can", "and", "or"];
        stop_words.contains(&token)
    }

    fn is_relationship_word(&self, word: &str) -> bool {
        let relationship_words = ["relates", "connected", "influences", "causes", "enables",
                                "contains", "consists", "includes", "belongs", "leads"];
        relationship_words.iter().any(|r| word.contains(r))
    }

    fn infer_domain(&self, concept: &str) -> Option<String> {
        let concept_lower = concept.to_lowercase();
        
        let domain_mappings = [
            (vec!["physics", "quantum", "relativity", "energy"], "physics"),
            (vec!["chemistry", "molecule", "element", "compound"], "chemistry"),
            (vec!["biology", "dna", "evolution", "organism"], "biology"),
            (vec!["computer", "technology", "internet", "digital"], "technology"),
            (vec!["history", "war", "revolution", "ancient"], "history"),
            (vec!["mathematics", "equation", "number", "calculation"], "mathematics"),
        ];
        
        for (keywords, domain) in &domain_mappings {
            if keywords.iter().any(|k| concept_lower.contains(k)) {
                return Some(domain.to_string());
            }
        }
        
        None
    }
}

impl Default for BasicQueryProcessor {
    fn default() -> Self {
        Self::new()
    }
}