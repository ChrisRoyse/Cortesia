# Directory Overview: Federation Module

## 1. High-Level Summary

The `federation` module provides a comprehensive federated knowledge graph system that enables operations across multiple heterogeneous databases. It implements distributed query execution, cross-database transactions, result merging, and database discovery/registration. The module follows a modular architecture with clear separation of concerns for registry management, query routing, transaction coordination, and result processing.

This system allows users to treat multiple knowledge graph databases as a single logical database, enabling complex queries and operations that span across different data sources while maintaining consistency and providing unified results.

## 2. Tech Stack

- **Language:** Rust
- **Async Runtime:** Tokio (for async/await, RwLock, JoinSet)
- **Serialization:** Serde (JSON serialization/deserialization)
- **Traits:** async-trait (for async trait implementations)
- **Concurrency:** Arc, RwLock for thread-safe shared state
- **Error Handling:** Custom GraphError types
- **Time Management:** SystemTime, Instant, Duration

## 3. Directory Structure

- **mod.rs**: Main module definition and FederationManager orchestrator
- **types.rs**: Core data structures, enums, and type definitions
- **coordinator.rs**: Cross-database transaction coordination with 2-phase commit
- **registry.rs**: Database discovery, registration, and health monitoring
- **router.rs**: Query planning, optimization, and execution routing
- **merger.rs**: Result merging strategies for combining multi-database results

## 4. File Breakdown

### `mod.rs` - Federation Manager

**Purpose**: Main orchestrator that coordinates all federation operations and provides the primary API.

**Key Structures:**
- `FederationManager`: Main facade that integrates all components
- `SimilarityResult`: Result structure for similarity searches across databases

**Methods:**
- `new()`: Creates a new federation manager with default components
- `register_database(descriptor)`: Registers a new database in the federation
- `execute_federated_query(query)`: Executes queries across multiple databases
- `list_databases()`: Returns all registered databases
- `health_check()`: Checks health of all databases
- `execute_similarity_search(query, threshold, max_results)`: Specialized similarity search

### `types.rs` - Core Type Definitions

**Purpose**: Defines all data structures, enums, and types used throughout the federation system.

**Key Types:**
- `DatabaseId`: Unique identifier for databases in the federation
- `FederatedEntityKey`: Entity key that includes database context
- `CrossDatabaseRelationship`: Represents relationships between entities in different databases
- `DatabaseCapabilities`: Defines what operations each database supports
- `FederatedQuery`: Structure for queries that span multiple databases
- `FederatedQueryResult`: Unified result structure for federated operations

**Enums:**
- `QueryType`: Different types of federated queries (Similarity, Comparison, Relationship, Mathematical, Aggregate)
- `MergeStrategy`: How to combine results (SimilarityMerge, ComparisonMerge, UnionMerge, etc.)
- `MathOperation`: Mathematical operations supported (CosineSimilarity, PageRank, etc.)
- `AggregateFunction`: Aggregation functions (Count, Sum, Average, etc.)

**Key Functions:**
- `generate_query_id()`: Creates unique query identifiers
- Various data structure implementations with serialization support

### `coordinator.rs` - Transaction Coordinator

**Purpose**: Manages cross-database transactions using 2-phase commit protocol for consistency.

**Key Structures:**
- `FederationCoordinator`: Main transaction coordinator
- `CrossDatabaseTransaction`: Transaction that spans multiple databases
- `TransactionOperation`: Individual operations within transactions
- `TransactionResult`: Result of transaction execution

**Enums:**
- `TransactionStatus`: Status of transactions (Pending, Preparing, Committed, etc.)
- `OperationType`: Types of operations (CreateEntity, UpdateEntity, CreateRelationship, etc.)
- `IsolationLevel`: Transaction isolation levels
- `ConsistencyMode`: Consistency guarantees (Eventual, Strong, Causal, Monotonic)

**Methods:**
- `begin_transaction(databases, metadata)`: Starts a new cross-database transaction
- `add_operation(transaction_id, operation)`: Adds operation to transaction
- `prepare_transaction(transaction_id)`: Phase 1 of 2-phase commit
- `commit_transaction(transaction_id)`: Phase 2 of 2-phase commit
- `abort_transaction(transaction_id)`: Rollback transaction
- `cleanup_expired_transactions()`: Removes timed-out transactions
- `ensure_consistency(databases)`: Checks consistency across databases
- `synchronize_databases(source, target, options)`: Data synchronization

### `registry.rs` - Database Registry

**Purpose**: Manages database discovery, registration, and health monitoring within the federation.

**Key Structures:**
- `DatabaseRegistry`: Main registry for managing federated databases
- `DatabaseDescriptor`: Complete description of a database
- `DatabaseMetadata`: Metadata about database (version, counts, etc.)
- `DiscoveryManager`: Automatic database discovery
- `FederationStats`: Statistics about the federation

**Enums:**
- `DatabaseType`: Types of supported databases (KnowledgeGraph, SQLite, PostgreSQL, Neo4j, etc.)
- `DatabaseStatus`: Status of databases (Online, Offline, Maintenance, etc.)
- `DiscoverySource`: Sources for database discovery (Config, Environment, Network)

**Methods:**
- `register(descriptor)`: Register a new database
- `unregister(database_id)`: Remove database from federation
- `get_databases_with_capabilities(capabilities)`: Find databases with specific features
- `health_check(database_id)`: Check health of specific database
- `health_check_all()`: Check health of all databases
- `discover_databases()`: Automatically discover databases
- `get_federation_stats()`: Get comprehensive federation statistics

### `router.rs` - Query Router

**Purpose**: Plans, optimizes, and executes federated queries across multiple databases.

**Key Structures:**
- `QueryRouter`: Main query routing and execution engine
- `QueryPlan`: Execution plan for federated queries
- `ExecutionStep`: Individual step in query execution
- `RawQueryResult`: Raw results from individual databases
- `QueryOptimizer`: Query optimization engine

**Enums:**
- `DatabaseOperation`: Operations executed on individual databases
- `OptimizationRule`: Rules for query optimization

**Methods:**
- `plan_query(query)`: Creates optimized execution plan
- `execute_plan(plan)`: Executes query plan across databases
- `filter_capable_databases(databases, query)`: Filters databases by capability
- `generate_execution_steps(query, databases)`: Creates execution steps
- `execute_parallel(plan)`: Parallel execution of query steps
- `execute_sequential(plan)`: Sequential execution of query steps

### `merger.rs` - Result Merger

**Purpose**: Combines and merges results from multiple databases using various strategies.

**Key Structures:**
- `ResultMerger`: Main result merging engine
- Various merge handlers for different strategies (SimilarityMergeHandler, ComparisonMergeHandler, etc.)

**Traits:**
- `MergeHandler`: Async trait for implementing merge strategies

**Methods:**
- `merge_results(raw_results, strategy)`: Main method to merge results
- Strategy-specific merge implementations:
  - Similarity merging: Combines and ranks similarity results
  - Comparison merging: Creates side-by-side entity comparisons
  - Relationship merging: Combines relationship graphs
  - Mathematical merging: Combines mathematical operation results
  - Aggregation merging: Combines aggregate statistics
  - Union/Intersection merging: Set operations on results

## 5. Key Variables and Logic

### Federation State Management
- **Thread-safe collections**: All shared state uses `Arc<RwLock<HashMap<K, V>>>` for concurrent access
- **Caching**: Health checks and query results are cached with TTL (300 seconds default)
- **Transaction lifecycle**: Transactions have timeouts and automatic cleanup

### Query Execution Flow
1. **Planning**: Query is analyzed and execution plan is created
2. **Optimization**: Plan is optimized using various rules
3. **Execution**: Steps are executed in parallel or sequential order
4. **Merging**: Results are combined using appropriate merge strategy

### Error Handling
- Uses custom `GraphError` enum for structured error handling
- Graceful degradation: failed databases don't prevent partial results
- Timeout handling for all async operations

## 6. Dependencies

### Internal Dependencies
- `crate::core::types::EntityKey`: Core entity key type
- `crate::error::{GraphError, Result}`: Error handling types

### External Dependencies
- `serde`: JSON serialization/deserialization
- `tokio`: Async runtime and synchronization primitives
- `async_trait`: Enables async methods in traits
- `uuid`: Unique identifier generation (inferred from usage)

### Standard Library
- `std::collections::HashMap`: Key-value storage
- `std::sync::Arc`: Reference counting for shared ownership
- `std::time::{SystemTime, Duration, Instant}`: Time handling

## 7. API Endpoints / Public Interface

The federation module provides a programmatic API through the `FederationManager`:

### Primary Methods
- **Database Management**: `register_database()`, `list_databases()`, `health_check()`
- **Query Execution**: `execute_federated_query()`, `execute_similarity_search()`
- **Transaction Management**: Through coordinator (begin, commit, abort transactions)

### Query Types Supported
- **Similarity Search**: Vector-based similarity across databases
- **Entity Comparison**: Compare same entity across different databases
- **Cross-Database Relationships**: Find relationships spanning databases
- **Mathematical Operations**: PageRank, clustering, centrality measures
- **Aggregate Queries**: Count, sum, average across databases

## 8. Architecture Patterns

### Design Patterns Used
- **Facade Pattern**: `FederationManager` provides simplified interface
- **Strategy Pattern**: Different merge strategies for result combination
- **Command Pattern**: `TransactionOperation` encapsulates operations
- **Observer Pattern**: Health monitoring and status updates

### Concurrency Model
- **Actor-like**: Each component manages its own state with message passing
- **Async/Await**: Full async support for I/O operations
- **Lock-free where possible**: Minimal locking, preferring immutable data

### Extensibility
- **Plugin Architecture**: New database types can be added via `DatabaseType` enum
- **Strategy Registration**: New merge strategies can be registered
- **Operation Extension**: New mathematical operations can be added

## 9. Implementation Status

### Fully Implemented
- Type definitions and data structures
- Transaction coordination logic
- Query planning and routing
- Result merging strategies
- Database registry management

### Partially Implemented/Mocked
- Actual database connections (returns `NotImplemented` errors)
- Network discovery of databases
- Real mathematical operations
- Health check implementations

### Extension Points
- Database driver implementations needed for actual connectivity
- Embedding model integration for similarity search
- Network protocol implementations for remote databases
- Advanced optimization rules for query planning

This federation system provides a solid foundation for distributed knowledge graph operations while maintaining flexibility for future enhancements and database integrations.