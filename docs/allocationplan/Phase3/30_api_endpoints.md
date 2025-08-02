# Task 30: REST/GraphQL API Endpoints Implementation

**Estimated Time**: 15-20 minutes  
**Dependencies**: 29_error_handling.md  
**Stage**: Service Layer  

## Objective
Implement production-ready REST and GraphQL API endpoints that expose all knowledge graph functionality with proper authentication, rate limiting, request validation, and comprehensive API documentation.

## Specific Requirements

### 1. RESTful API Design
- Resource-based URL structure following REST conventions
- HTTP status codes and headers for proper API semantics
- Request/response validation with detailed error messages
- OpenAPI 3.0 specification for comprehensive documentation

### 2. GraphQL API Implementation
- Type-safe schema with comprehensive type definitions
- Efficient query resolution with N+1 problem prevention
- Subscription support for real-time updates
- Query complexity analysis and depth limiting

### 3. Production API Features
- JWT-based authentication and authorization
- Rate limiting with different tiers for different users
- Request/response compression and caching
- API versioning and backward compatibility

## Implementation Steps

### 1. Create REST API Server
```rust
// src/api/rest_server.rs
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, put, delete},
    Router,
    middleware,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone)]
pub struct ApiState {
    pub knowledge_graph_service: Arc<KnowledgeGraphService>,
    pub allocation_service: Arc<MemoryAllocationService>,
    pub retrieval_service: Arc<MemoryRetrievalService>,
    pub auth_service: Arc<AuthenticationService>,
    pub rate_limiter: Arc<RateLimiter>,
}

pub fn create_rest_router(state: ApiState) -> Router {
    Router::new()
        // Memory operations
        .route("/api/v1/memory/allocate", post(allocate_memory))
        .route("/api/v1/memory/retrieve", post(retrieve_memory))
        .route("/api/v1/memory/update/:concept_id", put(update_memory))
        .route("/api/v1/memory/deallocate/:concept_id", delete(deallocate_memory))
        
        // Search operations
        .route("/api/v1/search/semantic", post(semantic_search))
        .route("/api/v1/search/ttfs", post(ttfs_search))
        .route("/api/v1/search/spreading-activation", post(spreading_activation_search))
        .route("/api/v1/search/hierarchical", post(hierarchical_search))
        .route("/api/v1/search/hybrid", post(hybrid_search))
        
        // Knowledge graph operations
        .route("/api/v1/concepts/:concept_id", get(get_concept))
        .route("/api/v1/concepts/:concept_id/properties", get(get_concept_properties))
        .route("/api/v1/concepts/:concept_id/relationships", get(get_concept_relationships))
        .route("/api/v1/concepts/:concept_id/inheritance", get(get_inheritance_chain))
        
        // Analytics and monitoring
        .route("/api/v1/metrics/allocation", get(get_allocation_metrics))
        .route("/api/v1/metrics/retrieval", get(get_retrieval_metrics))
        .route("/api/v1/health", get(health_check))
        .route("/api/v1/status", get(system_status))
        
        // Version and branch management
        .route("/api/v1/versions/:concept_id", get(get_concept_versions))
        .route("/api/v1/branches", get(list_branches))
        .route("/api/v1/branches", post(create_branch))
        .route("/api/v1/branches/:branch_id/merge", post(merge_branch))
        
        .with_state(state)
        .layer(middleware::from_fn(auth_middleware))
        .layer(middleware::from_fn(rate_limit_middleware))
        .layer(middleware::from_fn(request_logging_middleware))
}

// Memory allocation endpoint
async fn allocate_memory(
    State(state): State<ApiState>,
    Json(request): Json<AllocationApiRequest>,
) -> Result<Json<AllocationApiResponse>, ApiError> {
    // Validate request
    request.validate().map_err(ApiError::ValidationError)?;
    
    // Convert API request to service request
    let service_request = MemoryAllocationRequest {
        concept_id: request.concept_id,
        concept_type: request.concept_type,
        content: request.content,
        semantic_embedding: request.semantic_embedding,
        priority: request.priority.unwrap_or(AllocationPriority::Normal),
        resource_requirements: request.resource_requirements.unwrap_or_default(),
        locality_hints: request.locality_hints.unwrap_or_default(),
        user_id: extract_user_id_from_request()?,
        request_id: generate_request_id(),
        version_info: request.version_info,
    };
    
    // Execute allocation
    let result = state.allocation_service
        .allocate_memory(service_request)
        .await
        .map_err(ApiError::AllocationError)?;
    
    // Convert service response to API response
    let api_response = AllocationApiResponse {
        concept_id: result.memory_slot.concept_id.unwrap(),
        allocation_id: result.memory_slot.slot_id,
        neural_pathway_id: result.neural_pathway_id,
        cortical_column_id: result.cortical_column_id,
        allocation_time_ms: result.allocation_time_ms,
        confidence_score: result.confidence_score,
        memory_pool_id: result.memory_pool_id,
        ttfs_encoding: result.ttfs_encoding,
        links: AllocationLinks {
            self_link: format!("/api/v1/memory/{}", result.memory_slot.concept_id.unwrap()),
            concept_link: format!("/api/v1/concepts/{}", result.memory_slot.concept_id.unwrap()),
            metrics_link: "/api/v1/metrics/allocation".to_string(),
        },
    };
    
    Ok(Json(api_response))
}

// Search endpoint
async fn semantic_search(
    State(state): State<ApiState>,
    Json(request): Json<SearchApiRequest>,
) -> Result<Json<SearchApiResponse>, ApiError> {
    // Validate request
    request.validate().map_err(ApiError::ValidationError)?;
    
    // Convert to service request
    let service_request = SearchRequest {
        query_text: request.query,
        search_type: SearchType::Semantic,
        similarity_threshold: request.similarity_threshold,
        limit: request.limit,
        user_context: extract_user_context()?,
        ..Default::default()
    };
    
    // Execute search
    let result = state.retrieval_service
        .search_memory(service_request)
        .await
        .map_err(ApiError::RetrievalError)?;
    
    // Convert to API response
    let api_response = SearchApiResponse {
        results: result.results.into_iter().map(|r| SearchResultItem {
            concept_id: r.concept_id,
            title: r.concept.name,
            content: r.concept.content,
            similarity_score: r.similarity_score,
            retrieval_method: r.retrieval_method.to_string(),
            properties: r.resolved_properties.direct_properties,
            links: SearchResultLinks {
                concept_link: format!("/api/v1/concepts/{}", r.concept_id),
                properties_link: format!("/api/v1/concepts/{}/properties", r.concept_id),
            },
        }).collect(),
        total_matches: result.total_matches,
        search_time_ms: result.search_time_ms,
        cache_hit: result.cache_hit,
        pagination: PaginationInfo {
            page: request.page.unwrap_or(1),
            page_size: request.limit.unwrap_or(10),
            total_pages: (result.total_matches as f64 / request.limit.unwrap_or(10) as f64).ceil() as usize,
        },
    };
    
    Ok(Json(api_response))
}

// Health check endpoint
async fn health_check(
    State(state): State<ApiState>,
) -> Result<Json<HealthCheckResponse>, ApiError> {
    let health_status = state.knowledge_graph_service.get_service_health().await;
    
    let response = HealthCheckResponse {
        status: match health_status.overall_status {
            HealthStatus::Healthy => "healthy".to_string(),
            HealthStatus::Degraded => "degraded".to_string(),
            HealthStatus::Unhealthy => "unhealthy".to_string(),
        },
        timestamp: health_status.timestamp,
        checks: vec![
            HealthCheck {
                name: "database".to_string(),
                status: health_status.database_health.status.to_string(),
                response_time_ms: health_status.database_health.response_time.as_millis() as u64,
                details: health_status.database_health.details,
            },
            HealthCheck {
                name: "cache".to_string(),
                status: health_status.cache_health.status.to_string(),
                response_time_ms: 0, // Cache is in-memory
                details: None,
            },
        ],
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: health_status.uptime.as_secs(),
    };
    
    let status_code = match health_status.overall_status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded => StatusCode::OK, // Still operational
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };
    
    Ok((status_code, Json(response)).into())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AllocationApiRequest {
    pub concept_id: String,
    pub concept_type: ConceptType,
    pub content: String,
    pub semantic_embedding: Option<Vec<f32>>,
    pub priority: Option<AllocationPriority>,
    pub resource_requirements: Option<ResourceRequirements>,
    pub locality_hints: Option<Vec<String>>,
    pub version_info: Option<VersionInfo>,
}

impl AllocationApiRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.concept_id.is_empty() {
            return Err(ValidationError::new("concept_id cannot be empty"));
        }
        if self.content.is_empty() {
            return Err(ValidationError::new("content cannot be empty"));
        }
        if self.content.len() > 1_000_000 { // 1MB limit
            return Err(ValidationError::new("content exceeds maximum size"));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchApiRequest {
    pub query: String,
    pub similarity_threshold: Option<f64>,
    pub limit: Option<usize>,
    pub page: Option<usize>,
    pub include_properties: Option<bool>,
    pub include_relationships: Option<bool>,
}

impl SearchApiRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.query.is_empty() {
            return Err(ValidationError::new("query cannot be empty"));
        }
        if let Some(threshold) = self.similarity_threshold {
            if threshold < 0.0 || threshold > 1.0 {
                return Err(ValidationError::new("similarity_threshold must be between 0.0 and 1.0"));
            }
        }
        if let Some(limit) = self.limit {
            if limit == 0 || limit > 1000 {
                return Err(ValidationError::new("limit must be between 1 and 1000"));
            }
        }
        Ok(())
    }
}
```

### 2. Create GraphQL API Server
```rust
// src/api/graphql_server.rs
use async_graphql::{
    Context, Object, Schema, Subscription, EmptyMutation, ID, Result as GqlResult,
    connection::{Connection, Edge, query},
};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use futures_util::Stream;
use std::sync::Arc;

pub type KnowledgeGraphSchema = Schema<QueryRoot, MutationRoot, SubscriptionRoot>;

pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Get concept by ID
    async fn concept(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Concept ID")] id: ID,
    ) -> GqlResult<Option<Concept>> {
        let service = ctx.data::<Arc<KnowledgeGraphService>>()?;
        
        let concept_id = id.to_string();
        match service.get_concept(&concept_id).await {
            Ok(concept) => Ok(Some(Concept::from(concept))),
            Err(RetrievalError::ConceptNotFound) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    
    /// Search concepts using semantic similarity
    async fn search_concepts(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Search query")] query: String,
        #[graphql(desc = "Search method")] method: SearchMethod,
        #[graphql(desc = "Similarity threshold")] threshold: Option<f64>,
        #[graphql(desc = "Maximum results")] limit: Option<i32>,
    ) -> GqlResult<SearchResults> {
        let retrieval_service = ctx.data::<Arc<MemoryRetrievalService>>()?;
        
        let search_request = SearchRequest {
            query_text: query,
            search_type: method.into(),
            similarity_threshold: threshold,
            limit: limit.map(|l| l as usize),
            ..Default::default()
        };
        
        let result = retrieval_service.search_memory(search_request).await?;
        
        Ok(SearchResults {
            results: result.results.into_iter().map(Concept::from).collect(),
            total_count: result.total_matches,
            search_time_ms: result.search_time_ms,
            cache_hit: result.cache_hit,
        })
    }
    
    /// Get concept's inheritance chain
    async fn inheritance_chain(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Concept ID")] concept_id: ID,
        #[graphql(desc = "Include resolved properties")] include_properties: Option<bool>,
    ) -> GqlResult<InheritanceChain> {
        let inheritance_engine = ctx.data::<Arc<PropertyInheritanceEngine>>()?;
        
        let resolved_properties = inheritance_engine
            .resolve_properties(&concept_id.to_string(), include_properties.unwrap_or(false))
            .await?;
        
        Ok(InheritanceChain::from(resolved_properties))
    }
    
    /// Get system metrics
    async fn system_metrics(&self, ctx: &Context<'_>) -> GqlResult<SystemMetrics> {
        let allocation_service = ctx.data::<Arc<MemoryAllocationService>>()?;
        let retrieval_service = ctx.data::<Arc<MemoryRetrievalService>>()?;
        
        let allocation_metrics = allocation_service.get_allocation_metrics().await;
        let retrieval_metrics = retrieval_service.get_retrieval_metrics().await;
        
        Ok(SystemMetrics {
            allocation: AllocationMetrics::from(allocation_metrics),
            retrieval: RetrievalMetrics::from(retrieval_metrics),
            timestamp: chrono::Utc::now(),
        })
    }
}

pub struct MutationRoot;

#[Object]
impl MutationRoot {
    /// Allocate memory for a new concept
    async fn allocate_memory(
        &self,
        ctx: &Context<'_>,
        input: AllocationInput,
    ) -> GqlResult<AllocationResult> {
        let allocation_service = ctx.data::<Arc<MemoryAllocationService>>()?;
        
        let request = MemoryAllocationRequest {
            concept_id: input.concept_id,
            concept_type: input.concept_type.into(),
            content: input.content,
            semantic_embedding: input.semantic_embedding,
            priority: input.priority.map(|p| p.into()).unwrap_or(AllocationPriority::Normal),
            resource_requirements: input.resource_requirements.map(|r| r.into()).unwrap_or_default(),
            locality_hints: input.locality_hints.unwrap_or_default(),
            user_id: extract_user_id_from_context(ctx)?,
            request_id: generate_request_id(),
            version_info: input.version_info.map(|v| v.into()),
        };
        
        let result = allocation_service.allocate_memory(request).await?;
        Ok(AllocationResult::from(result))
    }
    
    /// Update concept content
    async fn update_concept(
        &self,
        ctx: &Context<'_>,
        input: UpdateConceptInput,
    ) -> GqlResult<UpdateResult> {
        let knowledge_graph_service = ctx.data::<Arc<KnowledgeGraphService>>()?;
        
        let request = MemoryUpdateRequest {
            concept_id: input.concept_id,
            update_type: input.update_type.into(),
            new_content: input.new_content,
            property_updates: input.property_updates.map(|p| p.into()),
            relationship_updates: input.relationship_updates.map(|r| r.into()),
            metadata: input.metadata.unwrap_or_default(),
        };
        
        let result = knowledge_graph_service.update_memory(request).await?;
        Ok(UpdateResult::from(result))
    }
}

pub struct SubscriptionRoot;

#[Subscription]
impl SubscriptionRoot {
    /// Subscribe to concept updates
    async fn concept_updates(
        &self,
        #[graphql(desc = "Concept ID to watch")] concept_id: ID,
    ) -> impl Stream<Item = ConceptUpdate> {
        // Implementation for real-time concept updates
        // This would typically use a message broker or event stream
        async_stream::stream! {
            // Placeholder implementation
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                yield ConceptUpdate {
                    concept_id: concept_id.clone(),
                    update_type: UpdateType::ContentModified,
                    timestamp: chrono::Utc::now(),
                };
            }
        }
    }
    
    /// Subscribe to system metrics updates
    async fn metrics_updates(&self) -> impl Stream<Item = SystemMetrics> {
        async_stream::stream! {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                // Emit current system metrics
                yield SystemMetrics::current().await;
            }
        }
    }
}

#[derive(async_graphql::SimpleObject)]
pub struct Concept {
    pub id: ID,
    pub name: String,
    pub content: String,
    pub concept_type: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub properties: Vec<ConceptProperty>,
    pub relationships: Vec<ConceptRelationship>,
}

#[derive(async_graphql::InputObject)]
pub struct AllocationInput {
    pub concept_id: String,
    pub concept_type: ConceptTypeInput,
    pub content: String,
    pub semantic_embedding: Option<Vec<f64>>,
    pub priority: Option<AllocationPriorityInput>,
    pub resource_requirements: Option<ResourceRequirementsInput>,
    pub locality_hints: Option<Vec<String>>,
    pub version_info: Option<VersionInfoInput>,
}
```

### 3. Implement API Authentication and Rate Limiting
```rust
// src/api/middleware.rs
async fn auth_middleware(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    // Extract JWT token from Authorization header
    let auth_header = request.headers()
        .get("Authorization")
        .and_then(|header| header.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    if !auth_header.starts_with("Bearer ") {
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    let token = &auth_header[7..]; // Remove "Bearer " prefix
    
    // Validate JWT token
    let auth_service = request.extensions()
        .get::<Arc<AuthenticationService>>()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
    
    match auth_service.validate_token(token).await {
        Ok(user_claims) => {
            // Add user information to request extensions
            let mut request = request;
            request.extensions_mut().insert(user_claims);
            Ok(next.run(request).await)
        },
        Err(_) => Err(StatusCode::UNAUTHORIZED),
    }
}

async fn rate_limit_middleware(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    let user_id = request.extensions()
        .get::<UserClaims>()
        .map(|claims| claims.user_id.clone())
        .unwrap_or_else(|| "anonymous".to_string());
    
    let rate_limiter = request.extensions()
        .get::<Arc<RateLimiter>>()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
    
    match rate_limiter.check_rate_limit(&user_id).await {
        Ok(_) => Ok(next.run(request).await),
        Err(_) => Err(StatusCode::TOO_MANY_REQUESTS),
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] REST API provides all CRUD operations for memory management
- [ ] GraphQL API supports queries, mutations, and subscriptions
- [ ] API authentication works with JWT tokens
- [ ] Rate limiting protects against abuse
- [ ] Request validation prevents invalid inputs

### Performance Requirements
- [ ] API response times < 100ms for simple operations
- [ ] API handles 1000+ concurrent connections
- [ ] GraphQL N+1 problem is prevented
- [ ] API compression reduces response sizes

### Testing Requirements
- [ ] Unit tests for all API endpoints
- [ ] Integration tests for authentication flow
- [ ] Load tests for concurrent usage
- [ ] API documentation is comprehensive

## Validation Steps

1. **Test REST API endpoints**:
   ```bash
   curl -X POST http://localhost:8080/api/v1/memory/allocate \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"concept_id": "test", "content": "Test content"}'
   ```

2. **Test GraphQL API**:
   ```bash
   curl -X POST http://localhost:8080/graphql \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "{ concept(id: \"test\") { name content } }"}'
   ```

3. **Run API tests**:
   ```bash
   cargo test api_tests
   ```

## Files to Create/Modify
- `src/api/rest_server.rs` - REST API implementation
- `src/api/graphql_server.rs` - GraphQL API implementation
- `src/api/middleware.rs` - Authentication and rate limiting
- `src/api/types.rs` - API type definitions
- `tests/api/api_integration_tests.rs` - API test suite

## Success Metrics
- API response times: < 100ms (95th percentile)
- API uptime: > 99.9%
- Authentication success rate: > 99%
- Rate limiting effectiveness: 0 abuse incidents

## Next Task
Upon completion, proceed to **31_phase2_integration_tests.md** to test integration with Phase 2 components.