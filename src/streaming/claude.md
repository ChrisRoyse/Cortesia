# Directory Overview: Streaming Module

## 1. High-Level Summary

The `streaming` module provides real-time data streaming and incremental update capabilities for the LLMKG (Large Language Model Knowledge Graph) system. It handles continuous processing of graph updates, temporal data management, and incremental indexing to maintain graph consistency and performance during real-time operations.

## 2. Tech Stack

*   **Languages:** Rust
*   **Frameworks:** Tokio (async runtime), Futures (stream processing)
*   **Libraries:** 
    *   `chrono` (date/time handling)
    *   `serde` (serialization)
    *   `async-trait` (async traits)
    *   `futures` (stream abstractions)
*   **Database:** Internal graph storage with CSR (Compressed Sparse Row), Bloom filters, and embedding stores
*   **Architecture:** Event-driven streaming with batched processing and conflict resolution

## 3. Directory Structure

The streaming module contains four core components:
*   **mod.rs**: Module exports and edge streaming trait definitions
*   **update_handler.rs**: Main streaming update processing engine
*   **incremental_indexing.rs**: Real-time index maintenance system
*   **temporal_updates.rs**: Time-versioned update processing with temporal graph integration

## 4. File Breakdown

### `mod.rs`

*   **Purpose:** Module entry point defining public API and edge streaming capabilities
*   **Traits:**
    *   `EdgeStreamer`
        *   **Description:** Trait for streaming graph edges in batches
        *   **Methods:**
            *   `stream_edges(batch_size)`: Stream edges in configurable batch sizes
            *   `stream_filtered_edges(filter, batch_size)`: Stream edges matching a filter predicate
            *   `total_edge_count()`: Get total number of edges for streaming
            *   `reset_stream()`: Reset streaming position to beginning
*   **Classes:**
    *   `SimpleEdgeStreamer`
        *   **Description:** Basic implementation of EdgeStreamer for in-memory edge collections
        *   **Properties:**
            *   `edges`: Vector of Relationship objects
            *   `current_position`: Current streaming position
*   **Exports:** Re-exports all major types from other modules including StreamingUpdateHandler, IncrementalIndexer, and temporal update types

### `update_handler.rs`

*   **Purpose:** Core streaming update processing engine with batching and conflict resolution
*   **Classes:**
    *   `StreamingUpdateHandler`
        *   **Description:** Main handler for processing streams of graph updates
        *   **Properties:**
            *   `graph`: Arc reference to KnowledgeGraph
            *   `update_queue`: Thread-safe queue for buffering updates
            *   `batch_processor`: Groups updates for efficient processing
            *   `conflict_resolver`: Handles conflicting updates
            *   `stats`: Performance and processing statistics
        *   **Methods:**
            *   `handle_update_stream(updates)`: Process a stream of updates
            *   `enqueue_update(update)`: Add single update to processing queue
            *   `process_batch(updates)`: Process batch of updates with grouping
            *   `get_stats()`: Retrieve processing statistics
            *   `flush_queue()`: Process all queued updates immediately
    *   `BatchProcessor`
        *   **Description:** Groups updates by type for efficient batch processing
        *   **Methods:**
            *   `group_updates(updates)`: Group updates into TripleInserts, TripleUpdates, and TripleDeletes
    *   `ConflictResolver`
        *   **Description:** Handles concurrent update conflicts with configurable resolution strategies
        *   **Methods:**
            *   `check_conflicts(triple)`: Detect conflicts for incoming updates
            *   `resolve_conflicts(triple, conflicts, resolution)`: Apply resolution strategy to conflicts
*   **Functions:**
    *   `process_triple_insert(triple)`: Insert new relationship or entity property
    *   `process_triple_update(old_triple, new_triple)`: Update existing triple
    *   `process_triple_delete(triple)`: Remove triple from graph
    *   `entity_name_to_id(name)`: Convert entity names to numeric IDs using hash

### `incremental_indexing.rs`

*   **Purpose:** Real-time maintenance of graph indices during streaming updates
*   **Classes:**
    *   `IncrementalIndexer`
        *   **Description:** Coordinates updates across multiple index types (Bloom filter, CSR, embeddings)
        *   **Properties:**
            *   `bloom_filter_updater`: Updates Bloom filter indices
            *   `csr_updater`: Updates CSR graph structure
            *   `embedding_updater`: Updates vector embeddings
            *   `index_stats`: Performance statistics
        *   **Methods:**
            *   `update_indices(changes)`: Apply array of graph changes to all indices
            *   `get_index_stats()`: Retrieve indexing performance metrics
            *   `optimize_indices()`: Periodic optimization of all indices
    *   `BloomFilterUpdater`
        *   **Description:** Maintains Bloom filter for fast membership testing
        *   **Methods:**
            *   `add_entity(entity_id, data)`: Add entity to Bloom filter
            *   `update_entity(entity_id, old_data, new_data)`: Update entity in filter
            *   `remove_entity(entity_id)`: Mark entity as deleted (maintains deletion set)
            *   `add_relation(from, to, relation_type)`: Add relationship to filter
            *   `add_triple(triple)`: Add triple components to filter
    *   `CSRUpdater`
        *   **Description:** Updates Compressed Sparse Row graph structure
        *   **Methods:**
            *   `add_relation(from, to, relation_type)`: Add edge to CSR structure
            *   `remove_relation(from, to, relation_type)`: Remove edge from structure
            *   `remove_entity(entity_id)`: Remove all edges connected to entity
    *   `EmbeddingUpdater`
        *   **Description:** Maintains vector embeddings for entities and triples
        *   **Properties:**
            *   `embedding_store`: Persistent embedding storage
            *   `embedding_cache`: In-memory embedding cache
        *   **Methods:**
            *   `add_entity(entity_id, data)`: Generate and store entity embedding
            *   `update_entity(entity_id, old_data, new_data)`: Update entity embedding
            *   `add_triple(triple)`: Generate and store triple embedding
            *   `generate_entity_embedding(data)`: Create embedding from entity properties
            *   `simple_text_embedding(text)`: Hash-based embedding generation

### `temporal_updates.rs`

*   **Purpose:** Time-versioned update processing with temporal graph integration
*   **Classes:**
    *   `IncrementalTemporalProcessor`
        *   **Description:** Processes time-stamped updates with batching and latency control
        *   **Properties:**
            *   `update_queue`: Queue of temporal updates
            *   `batch_size`: Maximum updates per batch
            *   `max_latency`: Maximum time before forcing batch processing
            *   `temporal_graph`: Reference to temporal knowledge graph
            *   `statistics`: Processing performance metrics
        *   **Methods:**
            *   `start()`: Begin background update processing
            *   `stop()`: Halt update processing
            *   `enqueue_update(update)`: Add update to processing queue
            *   `process_update_stream(updates)`: Process stream of temporal updates
            *   `get_statistics()`: Retrieve processing statistics
    *   `TemporalUpdateBuilder`
        *   **Description:** Builder pattern for constructing temporal updates
        *   **Methods:**
            *   `with_entity(entity)`: Set the entity being updated
            *   `with_timestamp(timestamp)`: Set update timestamp
            *   `with_source(source)`: Set update source (User, System, Federation, Import)
            *   `with_metadata(metadata)`: Add custom metadata
            *   `build()`: Create TemporalUpdate instance
*   **Functions:**
    *   `processing_loop()`: Main background processing loop for batched updates
    *   `process_batch()`: Process batch of temporal updates with statistics
    *   `process_single_update()`: Apply single temporal update to graph

## 5. Key Data Structures

### Graph Changes and Updates
*   **`GraphChange`**: Enum representing different types of graph modifications
    *   `EntityAdded(u32, EntityData)`: New entity creation
    *   `EntityUpdated(u32, EntityData, EntityData)`: Entity property changes
    *   `EntityRemoved(u32)`: Entity deletion
    *   `RelationAdded(u32, u32, u8)`: New relationship creation
    *   `RelationRemoved(u32, u32, u8)`: Relationship deletion
    *   `TripleAdded(Triple)`: RDF triple insertion
    *   `TripleRemoved(Triple)`: RDF triple deletion

*   **`Update`**: Enum for streaming update operations
    *   `Insert(Triple)`: Insert new triple
    *   `Update(Triple, Triple)`: Update existing triple (old, new)
    *   `Delete(Triple)`: Delete existing triple

*   **`TemporalUpdate`**: Time-stamped update with metadata
    *   `operation`: UpdateOperation (Create, Update, Delete, Merge)
    *   `entity`: BrainInspiredEntity being modified
    *   `timestamp`: DateTime<Utc> of the update
    *   `source`: UpdateSource (User, System, Federation, Import)
    *   `metadata`: Optional JSON metadata

### Statistics and Results
*   **`UpdateStats`**: Performance metrics for streaming operations
    *   `total_updates`: Total number of updates processed
    *   `successful_updates`: Successfully processed updates
    *   `failed_updates`: Failed update attempts
    *   `avg_batch_time`: Average batch processing time
    *   `queued_updates`: Current queue size

*   **`IndexStats`**: Index maintenance performance metrics
    *   `total_updates`: Total index updates
    *   `successful_updates`: Successful index updates
    *   `avg_update_time`: Average update processing time
    *   `bloom_filter_size`: Current Bloom filter size
    *   `csr_edges`: Number of edges in CSR structure

## 6. Conflict Resolution

The streaming system includes sophisticated conflict resolution for concurrent updates:

*   **Conflict Types:**
    *   `SubjectPredicateConflict`: Multiple values for same subject-predicate pair
    *   `ExactDuplicate`: Identical triple already exists
    *   `SimilarEntity`: Similar entities that might be duplicates

*   **Resolution Strategies:**
    *   `KeepExisting`: Preserve current value, reject new update
    *   `OverwriteWithNew`: Replace with new value
    *   `MergeWithHigherConfidence`: Choose update with higher confidence score
    *   `RequestHumanIntervention`: Queue for manual review

## 7. Dependencies

*   **Internal:**
    *   `crate::core::triple::Triple`: RDF triple representation
    *   `crate::core::graph::KnowledgeGraph`: Main graph storage
    *   `crate::core::types::{Relationship, EntityKey}`: Graph data types
    *   `crate::core::brain_types::BrainInspiredEntity`: Entity representation
    *   `crate::storage::{bloom::BloomFilter, csr::CSRGraph}`: Storage backends
    *   `crate::embedding::store::EmbeddingStore`: Vector embedding storage
    *   `crate::versioning::temporal_graph::TemporalKnowledgeGraph`: Temporal storage
    *   `crate::error::{Result, GraphError}`: Error handling

*   **External:**
    *   `tokio`: Async runtime and synchronization primitives
    *   `futures`: Stream processing abstractions
    *   `chrono`: Date and time handling
    *   `serde`: Serialization/deserialization
    *   `async-trait`: Async trait definitions

## 8. Key Algorithms and Logic

### Batch Processing
The system uses adaptive batching to balance latency and throughput:
1. Collect updates up to `batch_size` or `max_latency` timeout
2. Group updates by type (Insert, Update, Delete) for efficient processing
3. Apply conflict resolution before processing
4. Update multiple indices (Bloom filter, CSR, embeddings) atomically
5. Track performance statistics for optimization

### Incremental Indexing
Updates are applied incrementally to maintain index consistency:
1. **Bloom Filter**: Add new items, track deletions separately (no removal capability)
2. **CSR Graph**: Add/remove edges, compact periodically for performance
3. **Embeddings**: Generate embeddings for new entities/triples, cache frequently accessed items

### Temporal Processing
Time-versioned updates support historical querying:
1. Each update includes timestamp and source information
2. Updates are applied to temporal graph with validity periods
3. Conflict resolution considers temporal ordering
4. Statistics track processing latency and queue depth

## 9. Configuration

### StreamingConfig
*   `batch_size`: Maximum updates per batch (default: 100)
*   `batch_timeout`: Maximum time before processing batch (default: 100ms)
*   `max_queue_size`: Maximum queued updates (default: 10,000)
*   `conflict_resolution`: Default conflict resolution strategy

## 10. Performance Considerations

*   **Batching**: Groups updates to reduce processing overhead
*   **Async Processing**: Non-blocking update handling with tokio
*   **Index Optimization**: Periodic compaction and cleanup
*   **Memory Management**: Bounded queues prevent memory exhaustion
*   **Conflict Caching**: Reduces repeated conflict detection overhead
*   **Statistics**: Comprehensive metrics for performance monitoring and tuning

This streaming module enables real-time graph updates while maintaining consistency, performance, and data integrity through sophisticated batching, indexing, and conflict resolution mechanisms.