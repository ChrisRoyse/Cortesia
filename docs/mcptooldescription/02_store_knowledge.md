# store_knowledge - Complex Knowledge Chunk Storage Tool

## Overview

The `store_knowledge` tool is designed for storing complex, unstructured knowledge as text chunks in the LLMKG system. Unlike `store_fact`, which handles simple triples, this tool processes larger bodies of text and automatically extracts entities and relationships from the content, making it ideal for storing descriptions, explanations, documents, and other rich textual information.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/storage.rs`
- **Function**: `handle_store_knowledge`
- **Lines**: 109-262

### Core Functionality

The tool provides sophisticated text processing capabilities:

1. **Text Validation**: Validates content length and required fields
2. **Entity Extraction**: Automatically identifies entities within the text
3. **Relationship Extraction**: Discovers relationships between extracted entities
4. **Chunk Storage**: Stores the text as a knowledge chunk with metadata
5. **Triple Generation**: Creates triples for extracted entities and relationships
6. **Temporal Tracking**: Records all operations for time-travel functionality

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "content": {
      "type": "string",
      "description": "The knowledge content to store",
      "maxLength": 50000
    },
    "title": {
      "type": "string",
      "description": "A short title or summary",
      "maxLength": 200
    },
    "category": {
      "type": "string",
      "description": "Category or type of knowledge",
      "maxLength": 50
    },
    "source": {
      "type": "string",
      "description": "Where this knowledge came from (optional)",
      "maxLength": 200
    }
  },
  "required": ["content", "title"]
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_store_knowledge(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Input Processing Variables
- `content`: The main text content (up to 50,000 characters)
- `title`: A descriptive title (up to 200 characters)
- `category`: Optional category classification (defaults to "general")
- `source`: Optional source attribution (up to 200 characters)

#### Entity Extraction Function
```rust
fn extract_entities_from_text(text: &str) -> Vec<String>
```

**Algorithm Details:**
- Identifies capitalized words as potential entities
- Filters out common words using `is_common_word()` function
- Removes non-alphanumeric characters
- Deduplicates results
- Minimum length requirement of 3 characters

```rust
for word in text.split_whitespace() {
    if word.len() > 2 && word.chars().next().map_or(false, |c| c.is_uppercase()) {
        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
        if !clean_word.is_empty() && !is_common_word(clean_word) {
            entities.push(clean_word.to_string());
        }
    }
}
```

#### Relationship Extraction Function
```rust
fn extract_relationships_from_text(text: &str, entities: &[String]) -> Vec<(String, String, String)>
```

**Pattern Matching Logic:**
- Searches for common relationship patterns in text
- Currently implements patterns for `is`, `created`, and other relationships
- Cross-references entities to build relationship triples

```rust
for i in 0..entities.len() {
    for j in 0..entities.len() {
        if i != j {
            let entity1 = &entities[i];
            let entity2 = &entities[j];
            
            if text_lower.contains(&format!("{} is", entity1.to_lowercase())) {
                relationships.push((entity1.clone(), "is".to_string(), entity2.clone()));
            }
            
            if text_lower.contains(&format!("{} created", entity1.to_lowercase())) {
                relationships.push((entity1.clone(), "created".to_string(), entity2.clone()));
            }
        }
    }
}
```

#### Common Word Filter
```rust
fn is_common_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "the" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" |
        "of" | "with" | "by" | "from" | "as" | "is" | "was" | "are" | "were" |
        "been" | "being" | "have" | "has" | "had" | "do" | "does" | "did" |
        "will" | "would" | "could" | "should" | "may" | "might" | "must" |
        "can" | "this" | "that" | "these" | "those" | "a" | "an"
    )
}
```

### Storage Process

#### 1. Chunk ID Generation
```rust
let chunk_id = format!("chunk_{}", uuid::Uuid::new_v4());
```

#### 2. Metadata Creation
```rust
let mut chunk_metadata = json!({
    "title": title,
    "category": category,
    "content_length": content.len(),
    "extracted_entities": extracted_entities.len(),
    "extracted_relationships": extracted_relationships.len(),
});

if let Some(src) = source {
    chunk_metadata["source"] = json!(src);
}
```

#### 3. Chunk Triple Storage
```rust
let chunk_triple = Triple::new(
    chunk_id.clone(),
    "is".to_string(),
    "knowledge_chunk".to_string(),
).map_err(|e| format!("Failed to create chunk triple: {}", e))?;
```

#### 4. Entity-Chunk Relationship Storage
```rust
for entity in &extracted_entities {
    if let Ok(entity_triple) = Triple::new(
        entity.clone(),
        "mentioned_in".to_string(),
        chunk_id.clone(),
    ) {
        // Check for existing relationships to determine operation type
        let existing_query = TripleQuery {
            subject: Some(entity.clone()),
            predicate: Some("mentioned_in".to_string()),
            object: Some(chunk_id.clone()),
            limit: 1,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        if engine.store_triple(entity_triple.clone(), None).is_ok() {
            stored_count += 1;
            let operation = if exists { TemporalOperation::Update } else { TemporalOperation::Create };
            TEMPORAL_INDEX.record_operation(entity_triple, operation, None);
        }
    }
}
```

#### 5. Extracted Relationship Storage
```rust
for (subj, pred, obj) in &extracted_relationships {
    if let Ok(rel_triple) = Triple::new(subj.clone(), pred.clone(), obj.clone()) {
        // Duplicate detection logic
        let existing_query = TripleQuery {
            subject: Some(subj.clone()),
            predicate: Some(pred.clone()),
            object: Some(obj.clone()),
            limit: 1,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        let (operation, previous_value) = if let Some(result) = existing_result {
            if let Some(existing) = result.triples.first() {
                (TemporalOperation::Update, Some(existing.confidence.to_string()))
            } else {
                (TemporalOperation::Create, None)
            }
        } else {
            (TemporalOperation::Create, None)
        };
        
        if engine.store_triple(rel_triple.clone(), None).is_ok() {
            stored_count += 1;
            TEMPORAL_INDEX.record_operation(rel_triple, operation, previous_value);
        }
    }
}
```

### Output Format

#### Success Response Data
```json
{
  "stored": true,
  "chunk_id": "chunk_12345678-1234-1234-1234-123456789abc",
  "title": "Albert Einstein Biography", 
  "category": "biography",
  "extracted": {
    "entities": ["Albert", "Einstein", "German", "Physics", "Nobel", "Prize"],
    "relationships": 5,
    "total_stored": 8
  }
}
```

#### Response Message Format
```rust
let message = format!(
    "✓ Stored knowledge chunk '{}' with {} extracted entities and {} relationships",
    title, extracted_entities.len(), extracted_relationships.len()
);
```

#### Generated Suggestions
```rust
let suggestions = vec![
    format!("Explore extracted entities with: explore_connections(start_entity=\"{}\")", 
        extracted_entities.first().unwrap_or(&"entity".to_string())),
    "Use ask_question to query this knowledge".to_string(),
];
```

### Error Handling

The tool implements comprehensive validation:

1. **Content Validation**: Ensures content and title are not empty
2. **Length Limits**: Enforces 50,000 character limit for content
3. **Triple Creation**: Handles errors in entity and relationship triple creation
4. **Storage Errors**: Manages database storage failures gracefully

```rust
if content.is_empty() || title.is_empty() {
    return Err("Content and title cannot be empty".to_string());
}

if content.len() > 50000 {
    return Err("Content exceeds maximum length of 50,000 characters".to_string());
}
```

### Performance Characteristics

#### Complexity Analysis
- **Entity Extraction**: O(n) where n is the number of words in content
- **Relationship Extraction**: O(e²) where e is the number of extracted entities  
- **Storage**: O(e + r) where e is entities and r is relationships

#### Memory Usage
- Temporary storage for extracted entities and relationships
- UUID generation for unique chunk identification
- Metadata JSON object construction

#### Usage Statistics Impact
- **Weight**: 20 points per operation (higher than simple facts due to complexity)
- **Operation Type**: `StatsOperation::StoreChunk`

### Integration Points

#### With Entity Recognition System
The tool interfaces with the entity extraction subsystem:
- Identifies capitalized words as potential entities
- Filters common words to reduce noise
- Maintains entity uniqueness through deduplication

#### With Relationship Discovery Engine
Implements pattern-based relationship extraction:
- Supports extensible pattern matching
- Creates typed relationships between entities
- Handles bidirectional relationship discovery

#### With Temporal Tracking System
All extracted triples are recorded in the temporal index:
```rust
TEMPORAL_INDEX.record_operation(chunk_triple, TemporalOperation::Create, None);
```

#### With Knowledge Graph
Creates multiple types of connections:
- Chunk-to-type relationships (`chunk_id` -> `is` -> `knowledge_chunk`)
- Entity-to-chunk relationships (`entity` -> `mentioned_in` -> `chunk_id`)
- Entity-to-entity relationships (extracted from content)

### Advanced Features

#### Extensible Pattern Matching
The relationship extraction system can be extended with additional patterns:

```rust
// Example of adding new patterns
if text_lower.contains(&format!("{} invented", entity1.to_lowercase())) {
    relationships.push((entity1.clone(), "invented".to_string(), entity2.clone()));
}

if text_lower.contains(&format!("{} located_in", entity1.to_lowercase())) {
    relationships.push((entity1.clone(), "located_in".to_string(), entity2.clone()));
}
```

#### Metadata Enrichment
The system supports rich metadata for knowledge chunks:
- Content length tracking
- Entity and relationship counts
- Source attribution
- Categorization system

### Best Practices for Developers

1. **Content Chunking**: Break very long texts into logical chunks (one per topic)
2. **Meaningful Titles**: Use descriptive titles for better retrieval
3. **Consistent Categories**: Establish category conventions for organization
4. **Source Attribution**: Always include source when available
5. **Quality Content**: Include dates, names, and specific facts for better extraction

### Usage Examples

#### Scientific Biography
```json
{
  "title": "Albert Einstein Biography",
  "content": "Albert Einstein (1879-1955) was a German-born theoretical physicist who developed the theory of relativity. He received the Nobel Prize in Physics in 1921.",
  "category": "biography",
  "source": "Wikipedia"
}
```

#### Technical Documentation
```json
{
  "title": "Python Programming Language Overview",
  "content": "Python is a high-level programming language created by Guido van Rossum. It emphasizes code readability and simplicity.",
  "category": "technical"
}
```

### Tool Integration Workflow

1. **Input Validation**: Check content length and required fields
2. **Entity Extraction**: Identify potential entities using NLP heuristics
3. **Relationship Discovery**: Find relationships between extracted entities
4. **Chunk Storage**: Create and store the main knowledge chunk
5. **Triple Generation**: Store entity-chunk and entity-entity relationships
6. **Temporal Recording**: Record all operations for time-travel queries
7. **Statistics Update**: Update usage metrics and performance tracking
8. **Response Generation**: Provide success confirmation with extracted insights

This tool enables the LLMKG system to ingest and structure complex textual knowledge, automatically building rich semantic networks from unstructured content while maintaining full traceability and temporal awareness.