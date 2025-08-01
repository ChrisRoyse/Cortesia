//! Query Processor
//! 
//! Processes natural language queries, performs query understanding,
//! expansion, and transformation for optimal retrieval.

use std::sync::Arc;
// use std::collections::{HashMap, HashSet}; // Removed unused imports
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    retrieval_system::types::*,
};

/// Query processor for understanding and expanding queries
pub struct QueryProcessor {
    model_manager: Arc<ModelResourceManager>,
    config: RetrievalConfig,
}

impl QueryProcessor {
    /// Create new query processor
    pub fn new(
        model_manager: Arc<ModelResourceManager>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Process a retrieval query
    pub async fn process_query(
        &self,
        query: &RetrievalQuery,
    ) -> RetrievalResult2<ProcessedQuery> {
        // Step 1: Understand query intent and extract components
        let understanding = self.understand_query(&query.natural_language_query).await?;
        
        // Step 2: Expand query if enabled
        let expansion = if query.enable_query_expansion {
            Some(self.expand_query(&query.natural_language_query, &understanding).await?)
        } else {
            None
        };
        
        // Step 3: Generate structured search components
        let search_components = self.generate_search_components(query, &understanding, &expansion)?;
        
        // Step 4: Create query embedding for semantic search
        let query_embedding = self.generate_query_embedding(&query.natural_language_query).await?;
        
        // Step 5: Identify temporal context
        let temporal_context = self.extract_temporal_context(&query.natural_language_query);
        
        Ok(ProcessedQuery {
            original_query: query.clone(),
            understanding,
            expansion,
            search_components,
            query_embedding,
            temporal_context,
            processing_metadata: ProcessingMetadata {
                memory_used: 1000,
                cache_hit: false,
                model_load_time: None,
                inference_time: std::time::Duration::from_millis(50),
            },
        })
    }
    
    /// Understand query intent and extract key components
    async fn understand_query(
        &self,
        query: &str,
    ) -> RetrievalResult2<QueryUnderstanding> {
        let prompt = format!(
            r#"Analyze this search query and extract key information:

Query: "{query}"

Provide:
1. Primary intent (factual_lookup, concept_exploration, relationship_query, causal_analysis, temporal_sequence, comparison, definition, example, aggregation)
2. Key entities mentioned (proper nouns, specific objects)
3. Key concepts mentioned (abstract ideas, categories)
4. Temporal context if any (time references, sequences)
5. Complexity level (low, medium, high)
6. Suggested query expansions (related terms, synonyms)
7. Ambiguities or unclear aspects

Return as JSON:
{{
  "intent": "primary_intent",
  "entities": ["entity1", "entity2"],
  "concepts": ["concept1", "concept2"],
  "temporal_context": "optional temporal reference",
  "complexity": "medium",
  "suggested_expansions": ["expansion1", "expansion2"],
  "ambiguities": ["ambiguity1", "ambiguity2"]
}}

JSON Response:"#,
            query = query
        );
        
        let task = ProcessingTask::new(ComplexityLevel::Medium, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| RetrievalError::ModelError(e.to_string()))?;
        
        self.parse_query_understanding(&result.output)
    }
    
    /// Parse query understanding response
    fn parse_query_understanding(&self, response: &str) -> RetrievalResult2<QueryUnderstanding> {
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];
        
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| RetrievalError::QueryProcessingError(format!("JSON parse error: {}", e)))?;
        
        let intent = match parsed["intent"].as_str().unwrap_or("factual_lookup") {
            "factual_lookup" => QueryIntent::FactualLookup,
            "concept_exploration" => QueryIntent::ConceptExploration,
            "relationship_query" => QueryIntent::RelationshipQuery,
            "causal_analysis" => QueryIntent::CausalAnalysis,
            "temporal_sequence" => QueryIntent::TemporalSequence,
            "comparison" => QueryIntent::Comparison,
            "definition" => QueryIntent::Definition,
            "example" => QueryIntent::Example,
            "aggregation" => QueryIntent::Aggregation,
            _ => QueryIntent::FactualLookup,
        };
        
        let entities = parsed["entities"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_default();
        
        let concepts = parsed["concepts"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_default();
        
        let temporal_context = parsed["temporal_context"].as_str().map(|s| s.to_string());
        
        let complexity_level = match parsed["complexity"].as_str().unwrap_or("medium") {
            "low" => ComplexityLevel::Low,
            "high" => ComplexityLevel::High,
            _ => ComplexityLevel::Medium,
        };
        
        let suggested_expansions = parsed["suggested_expansions"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_default();
        
        let ambiguities = parsed["ambiguities"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_default();
        
        Ok(QueryUnderstanding {
            intent,
            extracted_entities: entities,
            extracted_concepts: concepts,
            temporal_context,
            complexity_level,
            suggested_expansions,
            ambiguities,
        })
    }
    
    /// Expand query with related terms and concepts
    async fn expand_query(
        &self,
        query: &str,
        understanding: &QueryUnderstanding,
    ) -> RetrievalResult2<QueryExpansion> {
        let prompt = format!(
            r#"Expand this search query with related terms:

Original query: "{query}"
Intent: {intent:?}
Entities: {entities:?}
Concepts: {concepts:?}

Generate:
1. Synonyms for key terms
2. Related entities
3. Related concepts
4. Suggested filters

Return as JSON:
{{
  "expanded_terms": [
    {{"term": "expanded_term", "type": "synonym|hypernym|hyponym|related|acronym", "confidence": 0.8, "source": "reasoning"}}
  ],
  "related_entities": ["entity1", "entity2"],
  "related_concepts": ["concept1", "concept2"],
  "suggested_filters": ["filter1", "filter2"]
}}

JSON Response:"#,
            query = query,
            intent = understanding.intent,
            entities = understanding.extracted_entities,
            concepts = understanding.extracted_concepts
        );
        
        let task = ProcessingTask::new(ComplexityLevel::Low, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| RetrievalError::ModelError(e.to_string()))?;
        
        self.parse_query_expansion(&result.output, query)
    }
    
    /// Parse query expansion response
    fn parse_query_expansion(&self, response: &str, original_query: &str) -> RetrievalResult2<QueryExpansion> {
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];
        
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| RetrievalError::QueryProcessingError(format!("JSON parse error: {}", e)))?;
        
        let expanded_terms = if let Some(terms_array) = parsed["expanded_terms"].as_array() {
            terms_array
                .iter()
                .filter_map(|term| {
                    let term_str = term["term"].as_str()?;
                    let type_str = term["type"].as_str().unwrap_or("related");
                    let confidence = term["confidence"].as_f64().unwrap_or(0.5) as f32;
                    let source = term["source"].as_str().unwrap_or("model").to_string();
                    
                    let expansion_type = match type_str {
                        "synonym" => ExpansionType::Synonym,
                        "hypernym" => ExpansionType::Hypernym,
                        "hyponym" => ExpansionType::Hyponym,
                        "acronym" => ExpansionType::Acronym,
                        "spelling" => ExpansionType::Spelling,
                        _ => ExpansionType::Related,
                    };
                    
                    Some(ExpandedTerm {
                        term: term_str.to_string(),
                        expansion_type,
                        confidence,
                        source,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        
        let related_entities = parsed["related_entities"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_default();
        
        let related_concepts = parsed["related_concepts"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_default();
        
        let suggested_filters = parsed["suggested_filters"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_default();
        
        Ok(QueryExpansion {
            original_query: original_query.to_string(),
            expanded_terms,
            related_entities,
            related_concepts,
            suggested_filters,
        })
    }
    
    /// Generate search components from processed query
    fn generate_search_components(
        &self,
        query: &RetrievalQuery,
        understanding: &QueryUnderstanding,
        expansion: &Option<QueryExpansion>,
    ) -> RetrievalResult2<SearchComponents> {
        let mut keywords = self.extract_keywords(&query.natural_language_query);
        let mut entities = understanding.extracted_entities.clone();
        let mut concepts = understanding.extracted_concepts.clone();
        
        // Add expanded terms if available
        if let Some(exp) = expansion {
            for term in &exp.expanded_terms {
                if term.confidence >= 0.7 {
                    keywords.push(term.term.clone());
                }
            }
            entities.extend(exp.related_entities.clone());
            concepts.extend(exp.related_concepts.clone());
        }
        
        // Add from structured constraints if provided
        if let Some(constraints) = &query.structured_constraints {
            entities.extend(constraints.required_entities.clone());
            concepts.extend(constraints.required_concepts.clone());
        }
        
        // Remove duplicates
        keywords.sort();
        keywords.dedup();
        entities.sort();
        entities.dedup();
        concepts.sort();
        concepts.dedup();
        
        // Generate boolean queries
        let boolean_queries = self.generate_boolean_queries(&keywords, &entities, &concepts);
        let fuzzy_terms = if self.config.enable_fuzzy_matching {
            self.identify_fuzzy_terms(&keywords)
        } else {
            Vec::new()
        };
        
        Ok(SearchComponents {
            keywords,
            entities,
            concepts,
            relationships: query.structured_constraints
                .as_ref()
                .map(|c| c.required_relationships.clone())
                .unwrap_or_default(),
            boolean_queries,
            fuzzy_terms,
        })
    }
    
    /// Generate query embedding for semantic search
    async fn generate_query_embedding(&self, query: &str) -> RetrievalResult2<Vec<f32>> {
        // Use embedding model to generate query embedding
        let prompt = format!("Generate embedding for query: {}", query);
        let task = ProcessingTask::new(ComplexityLevel::Low, &prompt);
        
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| RetrievalError::ModelError(e.to_string()))?;
        
        // Placeholder: Generate random embedding (would use actual embedding model)
        let embedding_dim = 384;
        let mut embedding = Vec::with_capacity(embedding_dim);
        
        for i in 0..embedding_dim {
            let value = ((result.output.len() + i) % 256) as f32 / 128.0 - 1.0;
            embedding.push(value);
        }
        
        Ok(embedding)
    }
    
    /// Extract temporal context from query
    fn extract_temporal_context(&self, query: &str) -> Option<TemporalContext> {
        let lower_query = query.to_lowercase();
        
        // Simple temporal pattern matching
        let temporal_markers = [
            "before", "after", "during", "since", "until", "between",
            "yesterday", "today", "tomorrow", "last", "next", "recent",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "2020", "2021", "2022", "2023", "2024", "2025"
        ];
        
        let mut found_markers = Vec::new();
        for marker in temporal_markers {
            if lower_query.contains(marker) {
                found_markers.push(marker.to_string());
            }
        }
        
        if found_markers.is_empty() {
            return None;
        }
        
        Some(TemporalContext {
            temporal_markers: found_markers,
            inferred_time_range: None, // Would parse actual time range
            temporal_type: if lower_query.contains("before") || lower_query.contains("after") {
                TemporalType::Relative
            } else if lower_query.contains("during") || lower_query.contains("between") {
                TemporalType::Range
            } else {
                TemporalType::Absolute
            },
        })
    }
    
    // Helper methods
    
    /// Extract keywords from query
    fn extract_keywords(&self, query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|word| !word.is_empty() && word.len() > 2 && !self.is_stop_word(word))
            .collect()
    }
    
    /// Check if word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "been", "be",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these", "those",
            "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        ];
        
        STOP_WORDS.contains(&word)
    }
    
    /// Generate boolean queries from components
    fn generate_boolean_queries(
        &self,
        keywords: &[String],
        entities: &[String],
        concepts: &[String],
    ) -> Vec<BooleanQuery> {
        let mut queries = Vec::new();
        
        // AND query for all keywords
        if !keywords.is_empty() {
            queries.push(BooleanQuery {
                must_terms: keywords.to_vec(),
                should_terms: Vec::new(),
                must_not_terms: Vec::new(),
                boost: 1.0,
            });
        }
        
        // OR query for entities
        if !entities.is_empty() {
            queries.push(BooleanQuery {
                must_terms: Vec::new(),
                should_terms: entities.to_vec(),
                must_not_terms: Vec::new(),
                boost: 1.5,
            });
        }
        
        // Mixed query for concepts
        if !concepts.is_empty() {
            queries.push(BooleanQuery {
                must_terms: concepts.iter().take(1).cloned().collect(),
                should_terms: concepts.iter().skip(1).cloned().collect(),
                must_not_terms: Vec::new(),
                boost: 1.2,
            });
        }
        
        queries
    }
    
    /// Identify terms that should use fuzzy matching
    fn identify_fuzzy_terms(&self, keywords: &[String]) -> Vec<String> {
        keywords
            .iter()
            .filter(|word| word.len() > 5) // Only fuzzy match longer words
            .cloned()
            .collect()
    }
}

/// Processed query with all components
#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    pub original_query: RetrievalQuery,
    pub understanding: QueryUnderstanding,
    pub expansion: Option<QueryExpansion>,
    pub search_components: SearchComponents,
    pub query_embedding: Vec<f32>,
    pub temporal_context: Option<TemporalContext>,
    pub processing_metadata: ProcessingMetadata,
}

/// Search components extracted from query
#[derive(Debug, Clone)]
pub struct SearchComponents {
    pub keywords: Vec<String>,
    pub entities: Vec<String>,
    pub concepts: Vec<String>,
    pub relationships: Vec<String>,
    pub boolean_queries: Vec<BooleanQuery>,
    pub fuzzy_terms: Vec<String>,
}

/// Boolean query structure
#[derive(Debug, Clone)]
pub struct BooleanQuery {
    pub must_terms: Vec<String>,
    pub should_terms: Vec<String>,
    pub must_not_terms: Vec<String>,
    pub boost: f32,
}

/// Temporal context information
#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub temporal_markers: Vec<String>,
    pub inferred_time_range: Option<TimeRange>,
    pub temporal_type: TemporalType,
}

/// Types of temporal references
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalType {
    Absolute,  // Specific dates/times
    Relative,  // Before/after references
    Range,     // Time periods
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    
    #[tokio::test]
    async fn test_query_processor_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let retrieval_config = RetrievalConfig::default();
        
        let processor = QueryProcessor::new(model_manager, retrieval_config);
        
        assert!(processor.config.enable_fuzzy_matching);
    }
    
    #[test]
    fn test_keyword_extraction() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let retrieval_config = RetrievalConfig::default();
        
        let processor = QueryProcessor::new(model_manager, retrieval_config);
        
        let query = "What is the relationship between Einstein and relativity theory?";
        let keywords = processor.extract_keywords(query);
        
        assert!(keywords.contains(&"relationship".to_string()));
        assert!(keywords.contains(&"einstein".to_string()));
        assert!(keywords.contains(&"relativity".to_string()));
        assert!(keywords.contains(&"theory".to_string()));
        assert!(!keywords.contains(&"what".to_string())); // Stop word
    }
    
    #[test]
    fn test_temporal_context_extraction() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let retrieval_config = RetrievalConfig::default();
        
        let processor = QueryProcessor::new(model_manager, retrieval_config);
        
        let query1 = "What happened before 2023?";
        let context1 = processor.extract_temporal_context(query1);
        assert!(context1.is_some());
        let ctx1 = context1.unwrap();
        assert!(ctx1.temporal_markers.contains(&"before".to_string()));
        assert!(ctx1.temporal_markers.contains(&"2023".to_string()));
        assert_eq!(ctx1.temporal_type, TemporalType::Relative);
        
        let query2 = "Events during January";
        let context2 = processor.extract_temporal_context(query2);
        assert!(context2.is_some());
        let ctx2 = context2.unwrap();
        assert!(ctx2.temporal_markers.contains(&"during".to_string()));
        assert!(ctx2.temporal_markers.contains(&"january".to_string()));
        assert_eq!(ctx2.temporal_type, TemporalType::Range);
    }
    
    #[test]
    fn test_boolean_query_generation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let retrieval_config = RetrievalConfig::default();
        
        let processor = QueryProcessor::new(model_manager, retrieval_config);
        
        let keywords = vec!["quantum".to_string(), "physics".to_string()];
        let entities = vec!["Einstein".to_string(), "Bohr".to_string()];
        let concepts = vec!["relativity".to_string(), "uncertainty".to_string()];
        
        let queries = processor.generate_boolean_queries(&keywords, &entities, &concepts);
        
        assert_eq!(queries.len(), 3);
        assert_eq!(queries[0].must_terms, keywords);
        assert_eq!(queries[1].should_terms, entities);
        assert_eq!(queries[1].boost, 1.5);
        assert_eq!(queries[2].must_terms.len(), 1);
        assert_eq!(queries[2].should_terms.len(), 1);
    }
}