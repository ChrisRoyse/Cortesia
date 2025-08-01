# ask_question - Natural Language Knowledge Query Tool

## Overview

The `ask_question` tool provides natural language querying capabilities for the LLMKG knowledge graph. Unlike `find_facts` which uses structured patterns, this tool processes human-readable questions, extracts key terms, searches for relevant knowledge, and generates contextual answers from the stored facts and knowledge chunks.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/query.rs`
- **Function**: `handle_ask_question`
- **Lines**: 98-198

### Core Functionality

The tool implements intelligent natural language processing:

1. **Question Processing**: Parses natural language questions for key terms
2. **Key Term Extraction**: Identifies important entities and concepts
3. **Multi-Pattern Search**: Searches as both subjects and objects
4. **Result Fusion**: Combines and deduplicates search results
5. **Answer Generation**: Creates contextual responses based on question type
6. **Relevance Scoring**: Calculates relevance of facts to the original question

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "Your question in natural language",
      "maxLength": 500
    },
    "context": {
      "type": "string",
      "description": "Additional context to help answer the question (optional)",
      "maxLength": 500
    },
    "max_results": {
      "type": "integer",
      "description": "Maximum number of relevant pieces to return",
      "minimum": 1,
      "maximum": 20,
      "default": 5
    }
  },
  "required": ["question"]
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_ask_question(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Input Processing Variables
```rust
let question = params.get("question").and_then(|v| v.as_str())
    .ok_or("Missing required field: question")?;
let context = params.get("context").and_then(|v| v.as_str());
let max_results = params.get("max_results")
    .and_then(|v| v.as_u64())
    .unwrap_or(5)
    .min(20) as usize;
```

### Key Term Extraction System

#### Main Extraction Function
```rust
fn extract_key_terms(question: &str) -> Vec<String>
```

The key term extraction uses multiple strategies:

#### 1. Capitalized Word Detection
```rust
for word in &words {
    let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
    if clean_word.chars().next().map_or(false, |c| c.is_uppercase()) {
        terms.push(clean_word.to_string());
    }
}
```

#### 2. Quoted Phrase Extraction
```rust
let mut in_quotes = false;
let mut current_phrase = String::new();

for char in question.chars() {
    if char == '"' || char == '\'' {
        if in_quotes && !current_phrase.is_empty() {
            terms.push(current_phrase.clone());
            current_phrase.clear();
        }
        in_quotes = !in_quotes;
    } else if in_quotes {
        current_phrase.push(char);
    }
}
```

#### 3. Question Word Context Analysis
```rust
for (i, word) in words.iter().enumerate() {
    if matches!(word.to_lowercase().as_str(), "who" | "what" | "where" | "when" | "which") {
        if i + 1 < words.len() {
            let next_word = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric());
            if !next_word.is_empty() && !is_stop_word(next_word) {
                terms.push(next_word.to_string());
            }
        }
    }
}
```

#### Stop Word Filtering
```rust
fn is_stop_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "is" | "are" | "was" | "were" | "the" | "a" | "an" | "and" | "or" | "but" |
        "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by" | "from" | "about"
    )
}
```

### Search Strategy

The tool implements a comprehensive search approach:

#### 1. Multi-Perspective Search
For each extracted key term, the system searches both as subject and object:

```rust
for term in &key_terms {
    // Search as subject
    let subject_query = TripleQuery {
        subject: Some(term.clone()),
        predicate: None,
        object: None,
        limit: 100,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    if let Ok(results) = engine.query_triples(subject_query) {
        all_results.extend(results);
    }
    
    // Search as object
    let object_query = TripleQuery {
        subject: None,
        predicate: None,
        object: Some(term.clone()),
        limit: 100,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    if let Ok(results) = engine.query_triples(object_query) {
        all_results.extend(results);
    }
}
```

#### 2. Result Deduplication and Limiting
```rust
// Deduplicate and limit results
all_results.sort_by(|a, b| a.subject.cmp(&b.subject));
all_results.dedup();
all_results.truncate(max_results);
```

### Relevance Scoring System

#### Relevance Calculation Function
```rust
fn calculate_relevance(triple: &Triple, question: &str) -> f32 {
    let question_lower = question.to_lowercase();
    let mut score = 0.0;
    
    // Check if triple components appear in question
    if question_lower.contains(&triple.subject.to_lowercase()) {
        score += 0.4;
    }
    if question_lower.contains(&triple.predicate.to_lowercase()) {
        score += 0.2;
    }
    if question_lower.contains(&triple.object.to_lowercase()) {
        score += 0.4;
    }
    
    f32::min(score, 1.0)
}
```

**Scoring Weights:**
- **Subject Match**: 0.4 (40% weight)
- **Predicate Match**: 0.2 (20% weight)
- **Object Match**: 0.4 (40% weight)
- **Maximum Score**: 1.0 (100% relevance)

### Answer Generation System

#### Main Answer Generation Function
```rust
fn generate_answer(facts: &[Triple], question: &str) -> String
```

The system uses question type analysis for tailored responses:

#### 1. "What" Questions
```rust
if question_lower.starts_with("what") {
    // Look for "is" relationships
    if let Some(fact) = facts.iter().find(|f| f.predicate == "is") {
        return format!("{} is {}", fact.subject, fact.object);
    }
}
```

#### 2. "Who" Questions
```rust
else if question_lower.starts_with("who") {
    // Look for person-related facts
    if let Some(fact) = facts.iter().find(|f| 
        f.predicate == "created" || f.predicate == "invented" || f.predicate == "wrote"
    ) {
        return format!("{} {} {}", fact.subject, fact.predicate, fact.object);
    }
}
```

#### 3. "Where" Questions
```rust
else if question_lower.starts_with("where") {
    // Look for location relationships
    if let Some(fact) = facts.iter().find(|f| 
        f.predicate == "located_in" || f.predicate == "from" || f.predicate == "in"
    ) {
        return format!("{} is {} {}", fact.subject, fact.predicate, fact.object);
    }
}
```

#### 4. Default Answer Strategy
```rust
// Default: return the most relevant facts
let relevant_facts: Vec<String> = facts.iter()
    .take(3)
    .map(|f| format!("{} {} {}", f.subject, f.predicate, f.object))
    .collect();

relevant_facts.join("; ")
```

### Output Format

#### Success Response Data
```json
{
  "question": "What did Einstein invent or discover?",
  "context": null,
  "key_terms": ["Einstein", "invent", "discover"],
  "relevant_facts": [
    {
      "subject": "Einstein",
      "predicate": "invented",
      "object": "relativity", 
      "relevance": 0.8
    },
    {
      "subject": "Einstein",
      "predicate": "discovered",
      "object": "photoelectric_effect",
      "relevance": 0.8
    }
  ],
  "answer": "Based on the knowledge graph:\n\n1. Einstein invented the theory of relativity\n2. Einstein discovered the photoelectric effect\n3. Einstein developed E=mc²\n\nRelevant knowledge chunks: [Biography of Einstein, Einstein's Major Works]"
}
```

#### Human-Readable Message Format
```rust
let message = if all_results.is_empty() {
    "No relevant information found for your question".to_string()
} else {
    format!("Based on the knowledge graph:\n\n{}\n\nFound {} relevant fact{}",
        generate_answer(&all_results, question),
        all_results.len(),
        if all_results.len() == 1 { "" } else { "s" }
    )
};
```

### Error Handling

#### Input Validation
```rust
if question.is_empty() {
    return Err("Question cannot be empty".to_string());
}
```

#### Key Term Extraction Validation
```rust
if key_terms.is_empty() {
    return Err("Could not extract meaningful terms from the question".to_string());
}
```

#### Empty Results Handling
```rust
let message = if all_results.is_empty() {
    "No relevant information found for your question".to_string()
} else {
    // Generate contextual response
};
```

### Performance Characteristics

#### Complexity Analysis
- **Key Term Extraction**: O(n) where n is question length
- **Search Operations**: O(k × log m) where k is key terms and m is graph size
- **Result Processing**: O(r) where r is total results found
- **Answer Generation**: O(r) for pattern matching and formatting

#### Memory Usage
- **Key Terms Storage**: Temporary vector for extracted terms
- **Result Accumulation**: Storage for all matching triples
- **Deduplication**: Additional memory for sorting and uniqueness

#### Usage Statistics Impact
- **Weight**: 15 points per query (higher than find_facts due to complexity)
- **Operation Type**: `StatsOperation::ExecuteQuery`

### Integration Points

#### With Key Term Extraction Engine
Uses sophisticated NLP techniques:
- Capitalized word recognition for entity identification
- Quoted phrase extraction for multi-word concepts
- Question word context analysis
- Stop word filtering for noise reduction

#### With Multi-Pattern Search System
Executes comprehensive searches:
- Subject-based queries for facts about extracted entities
- Object-based queries for facts pointing to extracted concepts
- Result fusion and deduplication

#### With Answer Generation Engine
Provides intelligent responses:
- Question type classification (what, who, where, etc.)
- Pattern-based answer construction
- Default fallback for unclassified questions

### Advanced Features

#### Context-Aware Processing
The optional context parameter can enhance query understanding:
```rust
let context = params.get("context").and_then(|v| v.as_str());
```

#### Relevance-Based Ranking
Facts are scored and can be ranked by relevance to the original question:
```rust
let relevant_facts: Vec<_> = all_results.iter().map(|t| {
    json!({
        "subject": &t.subject,
        "predicate": &t.predicate,
        "object": &t.object,
        "relevance": calculate_relevance(t, question)
    })
}).collect();
```

#### Extensible Question Types
The answer generation system can be extended with additional question patterns:
- "When" questions for temporal information
- "How" questions for process information
- "Why" questions for causal relationships

### Best Practices for Developers

1. **Question Clarity**: Encourage specific, well-formed questions
2. **Context Usage**: Leverage the context parameter for disambiguation
3. **Result Limits**: Balance comprehensiveness with performance
4. **Key Term Quality**: Monitor key term extraction effectiveness
5. **Answer Quality**: Continuously improve answer generation patterns

### Usage Examples

#### Scientific Inquiry
```json
{
  "question": "What did Einstein discover about physics?",
  "context": "20th century scientific discoveries",
  "max_results": 5
}
```

#### Historical Questions
```json
{
  "question": "Who invented the telephone?",
  "max_results": 3
}
```

#### Relationship Exploration
```json
{
  "question": "Where was Einstein born?",
  "context": "biographical information"
}
```

### Suggestions System

#### Standard Suggestions
```rust
let suggestions = vec![
    "Try rephrasing your question for different results".to_string(),
    "Use 'find_facts' for more specific searches".to_string(),
    "Add context to disambiguate entities".to_string(),
];
```

### Tool Integration Workflow

1. **Question Processing**: Parse and validate natural language input
2. **Key Term Extraction**: Identify important concepts using NLP techniques
3. **Multi-Pattern Search**: Execute comprehensive searches across the knowledge graph
4. **Result Fusion**: Combine, deduplicate, and limit search results
5. **Relevance Scoring**: Calculate relevance scores for retrieved facts
6. **Answer Generation**: Create contextual responses based on question type
7. **Response Formatting**: Structure results for both API and human consumption
8. **Usage Tracking**: Update system analytics and performance metrics

This tool bridges the gap between human natural language and structured knowledge graph queries, providing an intuitive interface for knowledge exploration and discovery within the LLMKG system.