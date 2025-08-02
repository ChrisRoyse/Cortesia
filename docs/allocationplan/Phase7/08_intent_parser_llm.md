# Micro Task 08: Intent Parser LLM

**Priority**: CRITICAL  
**Estimated Time**: 60 minutes  
**Dependencies**: 07_query_intent_types.md  
**Skills Required**: LLM integration, prompt engineering

## Objective

Implement an LLM-based query intent parser that can understand natural language queries and classify them into structured intent types.

## Context

The intent parser is the intelligence layer that bridges natural language queries and structured activation patterns. It must handle ambiguity, context, and complex linguistic structures while maintaining high accuracy and speed.

## Specifications

### LLM Integration Requirements

1. **Model Interface**
   - Support for local LLM models (SmallLLM)
   - Fallback to cloud models if available
   - Configurable model selection
   - Response caching for performance

2. **Parsing Capabilities**
   - Intent classification (> 90% accuracy)
   - Entity extraction from queries
   - Context inference and enrichment
   - Confidence scoring

3. **Performance Targets**
   - < 200ms per query parsing
   - Support for 100+ concurrent parses
   - Robust error handling
   - Graceful degradation

## Implementation Guide

### Step 1: LLM Interface Setup
```rust
// File: src/query/llm_parser.rs

use std::sync::Arc;
use tokio::sync::RwLock;

pub struct QueryIntentParser {
    llm: Arc<dyn LanguageModel + Send + Sync>,
    prompt_templates: PromptTemplateManager,
    cache: Arc<RwLock<ResponseCache>>,
    pattern_matcher: IntentPatternMatcher,
    entity_extractor: EntityExtractor,
    config: ParserConfig,
}

#[derive(Debug, Clone)]
pub struct ParserConfig {
    pub use_pattern_matching_first: bool,
    pub llm_timeout: Duration,
    pub cache_ttl: Duration,
    pub max_retries: usize,
    pub fallback_to_patterns: bool,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            use_pattern_matching_first: true,
            llm_timeout: Duration::from_millis(5000),
            cache_ttl: Duration::from_secs(300),
            max_retries: 2,
            fallback_to_patterns: true,
        }
    }
}
```

### Step 2: Core Parsing Logic
```rust
impl QueryIntentParser {
    pub async fn parse_intent(&self, query: &str) -> Result<ParsedQuery> {
        let start_time = Instant::now();
        
        // Try cache first
        if let Some(cached) = self.get_cached_result(query).await? {
            return Ok(cached);
        }
        
        // Try pattern matching first (fast path)
        if self.config.use_pattern_matching_first {
            if let Some(pattern_result) = self.try_pattern_matching(query)? {
                if pattern_result.confidence > 0.85 {
                    return Ok(pattern_result);
                }
            }
        }
        
        // Use LLM for complex parsing
        let llm_result = self.parse_with_llm(query).await?;
        
        // Cache the result
        self.cache_result(query, &llm_result).await?;
        
        Ok(llm_result)
    }
    
    async fn parse_with_llm(&self, query: &str) -> Result<ParsedQuery> {
        // Create parsing prompt
        let prompt = self.prompt_templates.create_intent_parsing_prompt(query)?;
        
        // Call LLM with retries
        let mut retries = 0;
        loop {
            match self.call_llm_with_timeout(&prompt).await {
                Ok(response) => {
                    return self.parse_llm_response(query, &response);
                }
                Err(e) if retries < self.config.max_retries => {
                    retries += 1;
                    tokio::time::sleep(Duration::from_millis(100 * retries)).await;
                }
                Err(e) => {
                    if self.config.fallback_to_patterns {
                        return self.fallback_to_pattern_matching(query);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }
    
    async fn call_llm_with_timeout(&self, prompt: &str) -> Result<String> {
        let llm_future = self.llm.generate(prompt);
        
        match tokio::time::timeout(self.config.llm_timeout, llm_future).await {
            Ok(result) => result,
            Err(_) => Err(Error::LLMTimeout),
        }
    }
}
```

### Step 3: Prompt Template Management
```rust
pub struct PromptTemplateManager {
    templates: HashMap<PromptType, PromptTemplate>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum PromptType {
    IntentClassification,
    EntityExtraction,
    ContextAnalysis,
    QueryDecomposition,
}

pub struct PromptTemplate {
    template: String,
    variables: Vec<String>,
    examples: Vec<PromptExample>,
}

impl PromptTemplateManager {
    pub fn new() -> Self {
        Self {
            templates: Self::create_default_templates(),
        }
    }
    
    pub fn create_intent_parsing_prompt(&self, query: &str) -> Result<String> {
        let template = self.templates.get(&PromptType::IntentClassification)
            .ok_or(Error::TemplateNotFound)?;
        
        let prompt = format!(
            r#"You are a query intent classifier. Analyze the following query and classify its intent.

Query: "{}"

Available intent types:
1. FILTER - Find entities matching specific criteria
   Example: "What animals can fly?"
   
2. RELATIONSHIP - Explore connections between entities  
   Example: "How are dogs related to wolves?"
   
3. HIERARCHY - Navigate taxonomic or organizational structures
   Example: "Show me the mammal hierarchy"
   
4. COMPARISON - Compare two or more entities
   Example: "What's the difference between cats and dogs?"
   
5. CAUSAL - Understand cause-effect relationships
   Example: "Why do birds migrate?"
   
6. DEFINITION - Define or describe an entity
   Example: "What is photosynthesis?"

Respond in this exact JSON format:
{{
  "intent_type": "FILTER|RELATIONSHIP|HIERARCHY|COMPARISON|CAUSAL|DEFINITION",
  "entities": ["entity1", "entity2"],
  "properties": ["property1", "property2"],
  "context": {{
    "domain": "domain_name_or_null",
    "temporal": "time_constraint_or_null"
  }},
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Response:"#,
            query
        );
        
        Ok(prompt)
    }
}
```

### Step 4: Response Parsing
```rust
impl QueryIntentParser {
    fn parse_llm_response(&self, original_query: &str, response: &str) -> Result<ParsedQuery> {
        // Parse JSON response
        let parsed: LLMIntentResponse = serde_json::from_str(response)
            .map_err(|e| Error::LLMResponseParsing(e.to_string()))?;
        
        // Convert to internal format
        let intent_type = self.convert_llm_intent_to_internal(&parsed)?;
        let entities = self.process_extracted_entities(&parsed.entities)?;
        let context = self.build_query_context(&parsed.context)?;
        
        // Validate and adjust confidence
        let adjusted_confidence = self.validate_and_adjust_confidence(
            &parsed,
            original_query,
        )?;
        
        Ok(ParsedQuery {
            original_query: original_query.to_string(),
            intent_type,
            entities,
            context,
            confidence: adjusted_confidence,
            sub_queries: Vec::new(), // Will be populated by decomposition if needed
            complexity: self.assess_complexity(&intent_type, &entities),
        })
    }
    
    fn convert_llm_intent_to_internal(&self, response: &LLMIntentResponse) -> Result<QueryIntent> {
        match response.intent_type.as_str() {
            "FILTER" => {
                let entity_type = response.entities.get(0)
                    .unwrap_or(&"unknown".to_string())
                    .clone();
                let property = response.properties.get(0)
                    .unwrap_or(&"unknown".to_string())
                    .clone();
                
                Ok(QueryIntent::Filter {
                    entity_type,
                    property: property.clone(),
                    value: "true".to_string(), // Default, will be refined
                    operator: FilterOperator::HasProperty,
                })
            }
            "RELATIONSHIP" => {
                if response.entities.len() >= 2 {
                    Ok(QueryIntent::Relationship {
                        entity1: response.entities[0].clone(),
                        entity2: response.entities[1].clone(),
                        relation_type: RelationType::Association,
                        direction: RelationDirection::Bidirectional,
                    })
                } else {
                    Err(Error::InsufficientEntities)
                }
            }
            "HIERARCHY" => {
                let root_entity = response.entities.get(0)
                    .ok_or(Error::NoEntitiesFound)?
                    .clone();
                
                Ok(QueryIntent::Hierarchy {
                    root_entity,
                    direction: HierarchyDirection::Descendants,
                    depth_limit: None,
                })
            }
            "COMPARISON" => {
                if response.entities.len() >= 2 {
                    Ok(QueryIntent::Comparison {
                        entities: response.entities.clone(),
                        aspect: response.properties.get(0)
                            .unwrap_or(&"general".to_string())
                            .clone(),
                        comparison_type: ComparisonType::Differences,
                    })
                } else {
                    Err(Error::InsufficientEntities)
                }
            }
            _ => Ok(QueryIntent::Unknown),
        }
    }
}
```

### Step 5: Entity Extraction Enhancement
```rust
pub struct EntityExtractor {
    llm: Arc<dyn LanguageModel + Send + Sync>,
    ner_patterns: Vec<NERPattern>,
}

impl EntityExtractor {
    pub async fn extract_entities(&self, query: &str) -> Result<Vec<ExtractedEntity>> {
        // Try pattern-based extraction first
        let pattern_entities = self.extract_with_patterns(query)?;
        
        // Enhance with LLM if needed
        if pattern_entities.is_empty() || self.needs_llm_enhancement(&pattern_entities) {
            let llm_entities = self.extract_with_llm(query).await?;
            Ok(self.merge_entity_extractions(pattern_entities, llm_entities)?)
        } else {
            Ok(pattern_entities)
        }
    }
    
    async fn extract_with_llm(&self, query: &str) -> Result<Vec<ExtractedEntity>> {
        let prompt = format!(
            r#"Extract all entities from this query and classify them:

Query: "{}"

Find all:
- ORGANISMS (animals, plants, microorganisms)
- CONCEPTS (abstract ideas, processes, properties)  
- LOCATIONS (places, regions, environments)
- OBJECTS (physical things, substances)
- ATTRIBUTES (properties, characteristics)

Respond in JSON format:
{{
  "entities": [
    {{
      "text": "entity_name",
      "type": "ORGANISM|CONCEPT|LOCATION|OBJECT|ATTRIBUTE",
      "start": start_position,
      "end": end_position,
      "confidence": 0.0-1.0
    }}
  ]
}}

Response:"#,
            query
        );
        
        let response = self.llm.generate(&prompt).await?;
        self.parse_entity_response(&response)
    }
}
```

## File Locations

- `src/query/llm_parser.rs` - Main implementation
- `src/query/prompts.rs` - Prompt templates
- `src/query/entity_extraction.rs` - Entity extraction
- `src/query/response_parsing.rs` - LLM response parsing
- `tests/query/llm_parser_tests.rs` - Test implementation

## Success Criteria

- [ ] Intent classification > 90% accuracy
- [ ] Entity extraction functional
- [ ] Response time < 200ms average
- [ ] Robust error handling
- [ ] Caching working correctly
- [ ] Fallback mechanisms functional
- [ ] All tests pass

## Test Requirements

```rust
#[tokio::test]
async fn test_intent_classification_accuracy() {
    let parser = QueryIntentParser::new().await;
    
    let test_cases = vec![
        ("What animals can fly?", QueryIntent::Filter { .. }),
        ("How are dogs related to wolves?", QueryIntent::Relationship { .. }),
        ("Show me the mammal hierarchy", QueryIntent::Hierarchy { .. }),
    ];
    
    let mut correct = 0;
    for (query, expected) in test_cases {
        let result = parser.parse_intent(query).await.unwrap();
        if std::mem::discriminant(&result.intent_type) == std::mem::discriminant(&expected) {
            correct += 1;
        }
    }
    
    let accuracy = correct as f32 / test_cases.len() as f32;
    assert!(accuracy >= 0.9, "Accuracy {} below threshold", accuracy);
}

#[tokio::test]
async fn test_performance_requirements() {
    let parser = QueryIntentParser::new().await;
    
    let start = Instant::now();
    let result = parser.parse_intent("What are the differences between cats and dogs?").await;
    let elapsed = start.elapsed();
    
    assert!(result.is_ok());
    assert!(elapsed < Duration::from_millis(200));
}

#[tokio::test]
async fn test_concurrent_parsing() {
    let parser = Arc::new(QueryIntentParser::new().await);
    
    let queries = vec![
        "What animals live in water?",
        "How are birds related to reptiles?",
        "Find large predators",
        "What is photosynthesis?",
        "Compare lions and tigers",
    ];
    
    let handles: Vec<_> = queries.into_iter().map(|query| {
        let p = parser.clone();
        tokio::spawn(async move {
            p.parse_intent(query).await
        })
    }).collect();
    
    let results = futures::try_join_all(handles).await.unwrap();
    assert!(results.iter().all(|r| r.is_ok()));
}
```

## Quality Gates

- [ ] No false positive classifications > 5%
- [ ] Handles malformed queries gracefully
- [ ] Memory usage stable under load
- [ ] Cache hit ratio > 60% in production
- [ ] Error recovery functional

## Next Task

Upon completion, proceed to **09_entity_extraction.md**