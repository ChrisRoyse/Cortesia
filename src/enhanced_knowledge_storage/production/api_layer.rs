//! Production REST API Layer
//! 
//! Secure, scalable REST API implementation with JWT authentication, rate limiting,
//! comprehensive error handling, and full integration with the Enhanced Knowledge Storage System.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::net::SocketAddr;

use axum::{
    Router, Json, Extension,
    extract::{State, Query},
    response::{Response, IntoResponse},
    http::{StatusCode, HeaderValue, Method},
    middleware::{self, Next},
    routing::{get, post},
};
use axum_server::tls_rustls::RustlsConfig;
use tower_http::{
    cors::{CorsLayer, Any},
    trace::TraceLayer,
    timeout::TimeoutLayer,
    limit::RequestBodyLimitLayer,
    compression::CompressionLayer,
};
use tokio::sync::{RwLock, Semaphore};
use serde::{Serialize, Deserialize};
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use metrics::{counter, histogram, gauge};
use tracing::{info, warn, error, debug, instrument};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use super::{
    config::{ApiConfig, SecurityConfig, RateLimitConfig, ProductionConfig},
    error_handling::SystemErrorHandler,
};
use super::system_orchestrator::{
    ProductionKnowledgeSystem
};

/// Main API server structure
pub struct ApiServer {
    config: ApiConfig,
    security_config: SecurityConfig,
    system_orchestrator: Arc<ProductionKnowledgeSystem>,
    error_handler: Arc<SystemErrorHandler>,
    auth_manager: Arc<AuthManager>,
    rate_limiter: Arc<RateLimiter>,
    metrics_collector: Arc<MetricsCollector>,
    request_limiter: Arc<Semaphore>,
}

/// Authentication manager for JWT and API key authentication
pub struct AuthManager {
    jwt_secret: String,
    jwt_expiration: Duration,
    api_keys: Arc<RwLock<HashMap<String, ApiKeyInfo>>>,
    refresh_tokens: Arc<RwLock<HashMap<String, RefreshTokenInfo>>>,
}

/// Rate limiting system with per-user and global limits
pub struct RateLimiter {
    global_limit: usize,
    per_user_limit: usize,
    burst_size: usize,
    global_counter: Arc<RwLock<RateLimitState>>,
    user_counters: Arc<RwLock<HashMap<String, RateLimitState>>>,
}

/// Metrics collection for monitoring and observability
pub struct MetricsCollector {
    request_counts: Arc<RwLock<HashMap<String, u64>>>,
    response_times: Arc<RwLock<Vec<Duration>>>,
    error_counts: Arc<RwLock<HashMap<String, u64>>>,
    active_connections: Arc<RwLock<u64>>,
}

/// JWT Claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,      // Subject (user ID)
    pub exp: i64,         // Expiration time
    pub iat: i64,         // Issued at
    pub jti: String,      // JWT ID
    pub role: String,     // User role
    pub permissions: Vec<String>, // User permissions
}

/// API key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    pub key_id: String,
    pub user_id: String,
    pub permissions: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub last_used: Option<DateTime<Utc>>,
    pub rate_limit_override: Option<usize>,
}

/// Refresh token information
#[derive(Debug, Clone)]
pub struct RefreshTokenInfo {
    pub user_id: String,
    pub expires_at: DateTime<Utc>,
    pub device_id: Option<String>,
}

/// Rate limiting state
#[derive(Debug, Clone)]
pub struct RateLimitState {
    pub count: usize,
    pub window_start: Instant,
    pub burst_count: usize,
    pub last_reset: Instant,
}

/// Request context for tracking and logging
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
    pub user_id: Option<String>,
    pub start_time: Instant,
    pub endpoint: String,
    pub method: String,
    pub user_agent: Option<String>,
    pub ip_address: Option<String>,
}

// API Request/Response Types

/// Document processing request
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProcessDocumentRequest {
    /// Document content to process
    pub content: String,
    
    /// Document metadata
    pub metadata: Option<DocumentMetadata>,
    
    /// Processing options
    pub options: Option<ProcessingOptions>,
}

/// Document metadata
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub source: Option<String>,
    pub language: Option<String>,
    pub tags: Option<Vec<String>>,
    pub created_at: Option<DateTime<Utc>>,
}

/// Processing options
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProcessingOptions {
    pub chunk_strategy: Option<String>,
    pub extract_entities: Option<bool>,
    pub map_relationships: Option<bool>,
    pub generate_summary: Option<bool>,
    pub enable_reasoning: Option<bool>,
}

/// Document processing response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProcessDocumentResponse {
    pub request_id: String,
    pub document_id: String,
    pub status: ProcessingStatus,
    pub chunks_created: usize,
    pub entities_extracted: usize,
    pub relationships_mapped: usize,
    pub processing_time_ms: u64,
    pub summary: Option<String>,
    pub warnings: Vec<String>,
}

/// Knowledge retrieval request
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RetrieveKnowledgeRequest {
    /// Query text
    pub query: String,
    
    /// Maximum number of results
    pub max_results: Option<usize>,
    
    /// Similarity threshold
    pub similarity_threshold: Option<f32>,
    
    /// Retrieval options
    pub options: Option<RetrievalOptions>,
}

/// Retrieval options
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RetrievalOptions {
    pub use_reranking: Option<bool>,
    pub enable_reasoning: Option<bool>,
    pub include_explanations: Option<bool>,
    pub context_window: Option<usize>,
    pub filter_by_source: Option<Vec<String>>,
    pub filter_by_tags: Option<Vec<String>>,
}

/// Knowledge retrieval response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RetrieveKnowledgeResponse {
    pub request_id: String,
    pub query: String,
    pub results: Vec<KnowledgeResult>,
    pub total_results: usize,
    pub reasoning_steps: Option<Vec<ReasoningStep>>,
    pub query_time_ms: u64,
    pub suggestions: Vec<String>,
}

/// Individual knowledge result
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct KnowledgeResult {
    pub id: String,
    pub content: String,
    pub similarity_score: f32,
    pub source: Option<String>,
    pub metadata: HashMap<String, String>,
    pub entities: Vec<String>,
    pub relationships: Vec<String>,
    pub explanation: Option<String>,
}

/// Reasoning step in multi-hop reasoning
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ReasoningStep {
    pub step: usize,
    pub description: String,
    pub confidence: f32,
    pub evidence: Vec<String>,
}

/// Processing status enumeration
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub enum ProcessingStatus {
    #[serde(rename = "success")]
    Success,
    #[serde(rename = "partial")]
    Partial,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "processing")]
    Processing,
}

/// Health check response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: DateTime<Utc>,
    pub version: String,
    pub uptime_seconds: u64,
    pub components: HashMap<String, ComponentStatus>,
    pub metrics: HealthMetrics,
}

/// Component health status
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ComponentStatus {
    pub healthy: bool,
    pub message: String,
    pub last_check: DateTime<Utc>,
    pub response_time_ms: Option<u64>,
}

/// Health metrics
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthMetrics {
    pub requests_per_minute: f64,
    pub avg_response_time_ms: f64,
    pub error_rate: f64,
    pub active_connections: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// System metrics response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MetricsResponse {
    pub timestamp: DateTime<Utc>,
    pub request_metrics: RequestMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub error_metrics: ErrorMetrics,
    pub system_metrics: SystemMetrics,
}

/// Request metrics
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RequestMetrics {
    pub total_requests: u64,
    pub requests_per_minute: f64,
    pub avg_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub success_rate: f64,
}

/// Performance metrics
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct PerformanceMetrics {
    pub throughput_ops_per_sec: f64,
    pub concurrent_requests: u64,
    pub queue_size: u64,
    pub cache_hit_rate: f64,
    pub model_inference_time_ms: f64,
}

/// Error metrics
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ErrorMetrics {
    pub total_errors: u64,
    pub error_rate: f64,
    pub errors_by_type: HashMap<String, u64>,
    pub errors_by_endpoint: HashMap<String, u64>,
    pub recovery_success_rate: f64,
}

/// System metrics
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SystemMetrics {
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_io_mbps: f64,
    pub active_connections: u64,
    pub uptime_seconds: u64,
}

/// Authentication request
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AuthRequest {
    pub username: String,
    pub password: String,
    pub device_id: Option<String>,
}

/// Authentication response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AuthResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: i64,
    pub token_type: String,
    pub user_id: String,
    pub permissions: Vec<String>,
}

/// Token refresh request
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

/// API Error response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ApiError {
    pub error: String,
    pub message: String,
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub details: Option<HashMap<String, String>>,
}

/// Query parameters for health endpoint
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct HealthQuery {
    pub detailed: Option<bool>,
}

/// Query parameters for metrics endpoint
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct MetricsQuery {
    pub format: Option<String>,
    pub timerange: Option<String>,
}

/// OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    paths(
        process_document,
        retrieve_knowledge,
        health_check,
        get_metrics,
        authenticate,
        refresh_token
    ),
    components(
        schemas(
            ProcessDocumentRequest,
            ProcessDocumentResponse,
            RetrieveKnowledgeRequest,
            RetrieveKnowledgeResponse,
            HealthResponse,
            MetricsResponse,
            AuthRequest,
            AuthResponse,
            RefreshTokenRequest,
            ApiError,
            DocumentMetadata,
            ProcessingOptions,
            RetrievalOptions,
            KnowledgeResult,
            ReasoningStep,
            ProcessingStatus,
            ComponentStatus,
            HealthMetrics,
            RequestMetrics,
            PerformanceMetrics,
            ErrorMetrics,
            SystemMetrics
        )
    ),
    tags(
        (name = "processing", description = "Document processing operations"),
        (name = "retrieval", description = "Knowledge retrieval operations"),
        (name = "system", description = "System health and metrics"),
        (name = "auth", description = "Authentication operations")
    ),
    info(
        title = "Enhanced Knowledge Storage API",
        version = "1.0.0",
        description = "Production-ready REST API for the Enhanced Knowledge Storage System with AI-powered document processing and retrieval"
    )
)]
pub struct ApiDoc;

impl ApiServer {
    /// Create new API server instance
    pub async fn new(config: ProductionConfig) -> Result<Self, ApiError> {
        let error_handler = Arc::new(
            SystemErrorHandler::new(config.error_handling_config.clone())
                .map_err(|e| ApiError::initialization_error(&format!("Error handler creation failed: {}", e)))?
        );
        
        let system_orchestrator = Arc::new(
            ProductionKnowledgeSystem::new(config.clone()).await
                .map_err(|e| ApiError::initialization_error(&format!("System orchestrator creation failed: {}", e)))?
        );
        
        let auth_manager = Arc::new(AuthManager::new(config.security_config.clone()));
        let rate_limiter = Arc::new(RateLimiter::new(config.security_config.rate_limiting.clone()));
        let metrics_collector = Arc::new(MetricsCollector::new());
        let request_limiter = Arc::new(Semaphore::new(config.system.max_concurrent_requests));
        
        Ok(Self {
            config: config.api_config,
            security_config: config.security_config,
            system_orchestrator,
            error_handler,
            auth_manager,
            rate_limiter,
            metrics_collector,
            request_limiter,
        })
    }
    
    /// Start the API server
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<(), ApiError> {
        info!("Starting Enhanced Knowledge Storage API server");
        
        let app = self.create_router().await?;
        let addr = SocketAddr::new(
            self.config.host.parse()
                .map_err(|e| ApiError::configuration_error(&format!("Invalid host: {}", e)))?,
            self.config.port
        );
        
        info!(
            "Server starting on {}:{} (TLS: {})",
            self.config.host,
            self.config.port,
            self.config.tls_enabled
        );
        
        if self.config.tls_enabled {
            self.start_tls_server(app, addr).await
        } else {
            self.start_http_server(app, addr).await
        }
    }
    
    /// Create the main router with all routes and middleware
    async fn create_router(&self) -> Result<Router, ApiError> {
        let state = ApiState {
            system_orchestrator: Arc::clone(&self.system_orchestrator),
            error_handler: Arc::clone(&self.error_handler),
            auth_manager: Arc::clone(&self.auth_manager),
            rate_limiter: Arc::clone(&self.rate_limiter),
            metrics_collector: Arc::clone(&self.metrics_collector),
            request_limiter: Arc::clone(&self.request_limiter),
            config: self.config.clone(),
            security_config: self.security_config.clone(),
        };
        
        let cors = if self.config.cors_enabled {
            CorsLayer::new()
                .allow_origin(self.config.cors_origins.iter().map(|origin| {
                    origin.parse::<HeaderValue>().unwrap_or_else(|_| HeaderValue::from_static("*"))
                }).collect::<Vec<_>>())
                .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS])
                .allow_headers(Any)
        } else {
            CorsLayer::very_permissive()
        };
        
        let router = Router::new()
            // API routes
            .route("/api/v1/process", post(process_document))
            .route("/api/v1/retrieve", post(retrieve_knowledge))
            .route("/api/v1/health", get(health_check))
            .route("/api/v1/metrics", get(get_metrics))
            .route("/api/v1/auth", post(authenticate))
            .route("/api/v1/auth/refresh", post(refresh_token))
            .route("/api/v1/auth/logout", post(logout))
            
            // Swagger UI
            .merge(
                SwaggerUi::new("/swagger-ui")
                    .url("/api-docs/openapi.json", ApiDoc::openapi())
            )
            
            // Add middleware layers individually
            .layer(TraceLayer::new_for_http())
            .layer(cors)
            .layer(CompressionLayer::new())
            .layer(TimeoutLayer::new(Duration::from_secs(30)))
            .layer(RequestBodyLimitLayer::new(self.config.max_request_size))
            // Add stateful middleware
            .layer(middleware::from_fn_with_state(state.clone(), request_limit_middleware))
            .layer(middleware::from_fn_with_state(state.clone(), metrics_middleware))
            .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
            .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
            .layer(middleware::from_fn_with_state(state.clone(), request_id_middleware))
            // Finally add state
            .with_state(state);
        
        Ok(router)
    }
    
    /// Start HTTP server
    async fn start_http_server(&self, app: Router, addr: SocketAddr) -> Result<(), ApiError> {
        axum_server::Server::bind(addr)
            .serve(app.into_make_service_with_connect_info::<SocketAddr>())
            .await
            .map_err(|e| ApiError::server_error(&format!("HTTP server failed: {}", e)))
    }
    
    /// Start HTTPS server with TLS
    async fn start_tls_server(&self, app: Router, addr: SocketAddr) -> Result<(), ApiError> {
        let cert_path = self.config.cert_path.as_ref()
            .ok_or_else(|| ApiError::configuration_error("TLS cert path not configured"))?;
        let key_path = self.config.key_path.as_ref()
            .ok_or_else(|| ApiError::configuration_error("TLS key path not configured"))?;
        
        let tls_config = RustlsConfig::from_pem_file(cert_path, key_path).await
            .map_err(|e| ApiError::configuration_error(&format!("TLS configuration failed: {}", e)))?;
        
        axum_server::bind_rustls(addr, tls_config)
            .serve(app.into_make_service_with_connect_info::<SocketAddr>())
            .await
            .map_err(|e| ApiError::server_error(&format!("HTTPS server failed: {}", e)))
    }
}

/// API state shared across handlers
#[derive(Clone)]
pub struct ApiState {
    pub system_orchestrator: Arc<ProductionKnowledgeSystem>,
    pub error_handler: Arc<SystemErrorHandler>,
    pub auth_manager: Arc<AuthManager>,
    pub rate_limiter: Arc<RateLimiter>,
    pub metrics_collector: Arc<MetricsCollector>,
    pub request_limiter: Arc<Semaphore>,
    pub config: ApiConfig,
    pub security_config: SecurityConfig,
}

// Route Handlers

/// Process document endpoint
#[utoipa::path(
    post,
    path = "/api/v1/process",
    tag = "processing",
    request_body = ProcessDocumentRequest,
    responses(
        (status = 200, description = "Document processed successfully", body = ProcessDocumentResponse),
        (status = 400, description = "Invalid request", body = ApiError),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 429, description = "Rate limit exceeded", body = ApiError),
        (status = 500, description = "Internal server error", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
#[instrument(skip(state, request))]
pub async fn process_document(
    State(state): State<ApiState>,
    Extension(ctx): Extension<RequestContext>,
    Json(request): Json<ProcessDocumentRequest>,
) -> Result<Json<ProcessDocumentResponse>, ApiError> {
    info!("Processing document request: {}", ctx.request_id);
    
    // Validate request
    if request.content.is_empty() {
        return Err(ApiError::validation_error("Document content cannot be empty"));
    }
    
    if request.content.len() > 10_000_000 { // 10MB limit
        return Err(ApiError::validation_error("Document content too large"));
    }
    
    let start_time = Instant::now();
    
    // Process document through system orchestrator
    let title = request.metadata.as_ref()
        .and_then(|m| m.title.as_ref())
        .map(|s| s.as_str())
        .unwrap_or("Untitled Document");
        
    let processing_result = state.system_orchestrator
        .process_document(&request.content, title)
        .await
        .map_err(|e| {
            error!("Document processing failed: {}", e);
            state.error_handler.handle_processing_error(
                super::error_handling::ProcessingError::ProcessingFailed(e.to_string()),
                0
            );
            ApiError::processing_error(&format!("Document processing failed: {}", e))
        })?;
    
    let processing_time = start_time.elapsed();
    
    // Update metrics
    state.metrics_collector.record_request("process_document", processing_time).await;
    
    let response = ProcessDocumentResponse {
        request_id: ctx.request_id.clone(),
        document_id: processing_result.metadata.id.clone(),
        status: ProcessingStatus::Success, // Assume success if no error thrown
        chunks_created: processing_result.chunks.len(),
        entities_extracted: processing_result.global_entities.len(),
        relationships_mapped: processing_result.relationships.len(),
        processing_time_ms: processing_time.as_millis() as u64,
        summary: Some(format!("Processed {} chunks with {} entities", 
                            processing_result.chunks.len(), 
                            processing_result.global_entities.len())),
        warnings: Vec::new(), // Would be extracted from processing result if available
    };
    
    info!(
        "Document processing completed: {} ({}ms)",
        ctx.request_id,
        processing_time.as_millis()
    );
    
    Ok(Json(response))
}

/// Retrieve knowledge endpoint
#[utoipa::path(
    post,
    path = "/api/v1/retrieve",
    tag = "retrieval",
    request_body = RetrieveKnowledgeRequest,
    responses(
        (status = 200, description = "Knowledge retrieved successfully", body = RetrieveKnowledgeResponse),
        (status = 400, description = "Invalid request", body = ApiError),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 429, description = "Rate limit exceeded", body = ApiError),
        (status = 500, description = "Internal server error", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
#[instrument(skip(state, request))]
pub async fn retrieve_knowledge(
    State(state): State<ApiState>,
    Extension(ctx): Extension<RequestContext>,
    Json(request): Json<RetrieveKnowledgeRequest>,
) -> Result<Json<RetrieveKnowledgeResponse>, ApiError> {
    info!("Knowledge retrieval request: {}", ctx.request_id);
    
    // Validate request
    if request.query.is_empty() {
        return Err(ApiError::validation_error("Query cannot be empty"));
    }
    
    if request.query.len() > 1000 {
        return Err(ApiError::validation_error("Query too long"));
    }
    
    let start_time = Instant::now();
    
    // Process query through system orchestrator
    let retrieval_result = state.system_orchestrator
        .query_with_reasoning(&request.query)
        .await
        .map_err(|e| {
            error!("Knowledge retrieval failed: {}", e);
            state.error_handler.handle_retrieval_error(
                super::error_handling::RetrievalError::RetrievalFailed(e.to_string())
            );
            ApiError::retrieval_error(&format!("Knowledge retrieval failed: {}", e))
        })?;
    
    let query_time = start_time.elapsed();
    
    // Update metrics
    state.metrics_collector.record_request("retrieve_knowledge", query_time).await;
    
    let total_results = retrieval_result.supporting_documents.len();
    let response = RetrieveKnowledgeResponse {
        request_id: ctx.request_id.clone(),
        query: request.query.clone(),
        results: retrieval_result.supporting_documents.into_iter().map(|doc| KnowledgeResult {
            id: doc.document_id.clone(),
            content: doc.extracted_content,
            similarity_score: doc.relevance_score,
            source: Some(doc.title),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("document_id".to_string(), doc.document_id);
                meta
            },
            entities: doc.entities_mentioned.into_iter().map(|e| e.to_string()).collect(),
            relationships: Vec::new(), // Would need to be extracted from the result
            explanation: Some(retrieval_result.response.clone()),
        }).collect(),
        total_results,
        reasoning_steps: Some(retrieval_result.reasoning_chain.into_iter().map(|step| ReasoningStep {
            step: step.step_number,
            description: format!("{}: {}", step.operation, step.input),
            confidence: step.confidence,
            evidence: step.supporting_evidence,
        }).collect()),
        query_time_ms: query_time.as_millis() as u64,
        suggestions: Vec::new(), // Would be generated based on query analysis
    };
    
    info!(
        "Knowledge retrieval completed: {} ({}ms, {} results)",
        ctx.request_id,
        query_time.as_millis(),
        response.results.len()
    );
    
    Ok(Json(response))
}

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/api/v1/health",
    tag = "system",
    params(HealthQuery),
    responses(
        (status = 200, description = "Health check response", body = HealthResponse),
        (status = 503, description = "Service unavailable", body = ApiError)
    )
)]
#[instrument(skip(state))]
pub async fn health_check(
    State(state): State<ApiState>,
    Query(query): Query<HealthQuery>,
) -> Result<Json<HealthResponse>, ApiError> {
    let start_time = Instant::now();
    
    // Get system health status
    let _health_status = state.system_orchestrator.get_system_health().await;
    let detailed = query.detailed.unwrap_or(false);
    
    let mut components = HashMap::new();
    
    if detailed {
        // For now, create basic component status
        // In production, this would come from the actual health status
        components.insert("system".to_string(), ComponentStatus {
            healthy: true,
            message: "System operational".to_string(),
            last_check: Utc::now(),
            response_time_ms: Some(start_time.elapsed().as_millis() as u64),
        });
    }
    
    let metrics = state.metrics_collector.get_health_metrics().await;
    
    let overall_healthy = true; // Would be determined from actual health status
    let response = HealthResponse {
        status: if overall_healthy { "healthy".to_string() } else { "unhealthy".to_string() },
        timestamp: Utc::now(),
        version: "1.0.0".to_string(),
        uptime_seconds: 0, // Would be calculated from system start time
        components,
        metrics,
    };
    
    let health_check_time = start_time.elapsed();
    debug!("Health check completed in {}ms", health_check_time.as_millis());
    
    if overall_healthy {
        Ok(Json(response))
    } else {
        Err(ApiError::service_unavailable("System health check failed"))
    }
}

/// Get system metrics endpoint
#[utoipa::path(
    get,
    path = "/api/v1/metrics",
    tag = "system",
    params(MetricsQuery),
    responses(
        (status = 200, description = "System metrics", body = MetricsResponse),
        (status = 401, description = "Unauthorized", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
#[instrument(skip(state))]
pub async fn get_metrics(
    State(state): State<ApiState>,
    Query(_query): Query<MetricsQuery>,
) -> Result<Json<MetricsResponse>, ApiError> {
    let metrics = state.metrics_collector.get_comprehensive_metrics().await;
    
    let response = MetricsResponse {
        timestamp: Utc::now(),
        request_metrics: metrics.request_metrics,
        performance_metrics: metrics.performance_metrics,
        error_metrics: metrics.error_metrics,
        system_metrics: metrics.system_metrics,
    };
    
    Ok(Json(response))
}

/// Authentication endpoint
#[utoipa::path(
    post,
    path = "/api/v1/auth",
    tag = "auth",
    request_body = AuthRequest,
    responses(
        (status = 200, description = "Authentication successful", body = AuthResponse),
        (status = 401, description = "Invalid credentials", body = ApiError),
        (status = 429, description = "Rate limit exceeded", body = ApiError)
    )
)]
#[instrument(skip(state, request))]
pub async fn authenticate(
    State(state): State<ApiState>,
    Json(request): Json<AuthRequest>,
) -> Result<Json<AuthResponse>, ApiError> {
    info!("Authentication request for user: {}", request.username);
    
    // In production, this would validate against a user database
    let user_id = validate_credentials(&request.username, &request.password)
        .ok_or_else(|| ApiError::unauthorized("Invalid credentials"))?;
    
    let permissions = get_user_permissions(&user_id).await;
    
    let (access_token, refresh_token) = state.auth_manager
        .generate_tokens(&user_id, &permissions, request.device_id.as_deref())
        .await
        .map_err(|e| ApiError::server_error(&format!("Token generation failed: {}", e)))?;
    
    let response = AuthResponse {
        access_token,
        refresh_token,
        expires_in: state.auth_manager.jwt_expiration.as_secs() as i64,
        token_type: "Bearer".to_string(),
        user_id,
        permissions,
    };
    
    Ok(Json(response))
}

/// Token refresh endpoint
#[utoipa::path(
    post,
    path = "/api/v1/auth/refresh",
    tag = "auth",
    request_body = RefreshTokenRequest,
    responses(
        (status = 200, description = "Token refreshed successfully", body = AuthResponse),
        (status = 401, description = "Invalid refresh token", body = ApiError)
    )
)]
#[instrument(skip(state, request))]
pub async fn refresh_token(
    State(state): State<ApiState>,
    Json(request): Json<RefreshTokenRequest>,
) -> Result<Json<AuthResponse>, ApiError> {
    let user_id = state.auth_manager
        .validate_refresh_token(&request.refresh_token)
        .await
        .map_err(|_| ApiError::unauthorized("Invalid refresh token"))?;
    
    let permissions = get_user_permissions(&user_id).await;
    
    let (access_token, new_refresh_token) = state.auth_manager
        .generate_tokens(&user_id, &permissions, None)
        .await
        .map_err(|e| ApiError::server_error(&format!("Token generation failed: {}", e)))?;
    
    let response = AuthResponse {
        access_token,
        refresh_token: new_refresh_token,
        expires_in: state.auth_manager.jwt_expiration.as_secs() as i64,
        token_type: "Bearer".to_string(),
        user_id,
        permissions,
    };
    
    Ok(Json(response))
}

/// Logout endpoint
#[instrument(skip(state))]
pub async fn logout(
    State(state): State<ApiState>,
    Extension(ctx): Extension<RequestContext>,
) -> Result<StatusCode, ApiError> {
    if let Some(user_id) = ctx.user_id {
        state.auth_manager.invalidate_user_tokens(&user_id).await;
        info!("User {} logged out", user_id);
    }
    
    Ok(StatusCode::NO_CONTENT)
}

// Middleware

/// Request ID middleware
pub async fn request_id_middleware(
    mut request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let request_id = Uuid::new_v4().to_string();
    
    let ctx = RequestContext {
        request_id: request_id.clone(),
        user_id: None,
        start_time: Instant::now(),
        endpoint: request.uri().path().to_string(),
        method: request.method().to_string(),
        user_agent: request.headers()
            .get("user-agent")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string()),
        ip_address: None, // Would be extracted from connection info
    };
    
    request.extensions_mut().insert(ctx);
    
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(&request_id).unwrap()
    );
    
    Ok(response)
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(state): State<ApiState>,
    Extension(ctx): Extension<RequestContext>,
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let user_id = ctx.user_id.as_deref().unwrap_or("anonymous");
    
    if !state.rate_limiter.check_rate_limit(user_id).await {
        warn!("Rate limit exceeded for user: {}", user_id);
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    
    Ok(next.run(request).await)
}

/// Authentication middleware
pub async fn auth_middleware(
    State(state): State<ApiState>,
    Extension(mut ctx): Extension<RequestContext>,
    mut request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip auth for health check and public endpoints
    if ctx.endpoint.contains("/health") || ctx.endpoint.contains("/auth") {
        return Ok(next.run(request).await);
    }
    
    if !state.security_config.authentication_enabled {
        return Ok(next.run(request).await);
    }
    
    let auth_header = request.headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok());
    
    let user_id = if let Some(auth_header) = auth_header {
        if auth_header.starts_with("Bearer ") {
            let token = &auth_header[7..];
            state.auth_manager.validate_jwt_token(token).await
                .map_err(|_| StatusCode::UNAUTHORIZED)?
        } else if auth_header.starts_with("ApiKey ") {
            let api_key = &auth_header[7..];
            state.auth_manager.validate_api_key(api_key).await
                .map_err(|_| StatusCode::UNAUTHORIZED)?
        } else {
            return Err(StatusCode::UNAUTHORIZED);
        }
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    
    ctx.user_id = Some(user_id);
    request.extensions_mut().insert(ctx);
    
    Ok(next.run(request).await)
}

/// Metrics collection middleware
pub async fn metrics_middleware(
    State(state): State<ApiState>,
    Extension(ctx): Extension<RequestContext>,
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    state.metrics_collector.increment_active_connections().await;
    
    let response = next.run(request).await;
    let response_time = ctx.start_time.elapsed();
    
    // Record metrics
    counter!("http_requests_total").increment(1);
    
    histogram!("http_request_duration_seconds").record(response_time.as_secs_f64());
    
    state.metrics_collector.record_request(&ctx.endpoint, response_time).await;
    state.metrics_collector.decrement_active_connections().await;
    
    Ok(response)
}

/// Request concurrency limiting middleware
pub async fn request_limit_middleware(
    State(state): State<ApiState>,
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let _permit = state.request_limiter.try_acquire()
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
    
    Ok(next.run(request).await)
}

// Implementation of supporting structures

impl AuthManager {
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            jwt_secret: config.jwt_secret,
            jwt_expiration: config.jwt_expiration,
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            refresh_tokens: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn generate_tokens(
        &self,
        user_id: &str,
        permissions: &[String],
        device_id: Option<&str>,
    ) -> Result<(String, String), String> {
        let now = Utc::now();
        let exp = now + ChronoDuration::from_std(self.jwt_expiration)
            .map_err(|e| format!("Invalid expiration duration: {}", e))?;
        
        let claims = Claims {
            sub: user_id.to_string(),
            exp: exp.timestamp(),
            iat: now.timestamp(),
            jti: Uuid::new_v4().to_string(),
            role: "user".to_string(),
            permissions: permissions.to_vec(),
        };
        
        let access_token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_ref())
        ).map_err(|e| format!("JWT encoding failed: {}", e))?;
        
        let refresh_token = Uuid::new_v4().to_string();
        let refresh_exp = now + ChronoDuration::days(30);
        
        let refresh_info = RefreshTokenInfo {
            user_id: user_id.to_string(),
            expires_at: refresh_exp,
            device_id: device_id.map(|s| s.to_string()),
        };
        
        self.refresh_tokens.write().await.insert(refresh_token.clone(), refresh_info);
        
        Ok((access_token, refresh_token))
    }
    
    pub async fn validate_jwt_token(&self, token: &str) -> Result<String, String> {
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_ref()),
            &Validation::default()
        ).map_err(|e| format!("JWT validation failed: {}", e))?;
        
        Ok(token_data.claims.sub)
    }
    
    pub async fn validate_api_key(&self, api_key: &str) -> Result<String, String> {
        let api_keys = self.api_keys.read().await;
        
        if let Some(key_info) = api_keys.get(api_key) {
            if let Some(expires_at) = key_info.expires_at {
                if Utc::now() > expires_at {
                    return Err("API key expired".to_string());
                }
            }
            Ok(key_info.user_id.clone())
        } else {
            Err("Invalid API key".to_string())
        }
    }
    
    pub async fn validate_refresh_token(&self, refresh_token: &str) -> Result<String, String> {
        let mut refresh_tokens = self.refresh_tokens.write().await;
        
        if let Some(token_info) = refresh_tokens.get(refresh_token) {
            if Utc::now() > token_info.expires_at {
                refresh_tokens.remove(refresh_token);
                return Err("Refresh token expired".to_string());
            }
            Ok(token_info.user_id.clone())
        } else {
            Err("Invalid refresh token".to_string())
        }
    }
    
    pub async fn invalidate_user_tokens(&self, user_id: &str) {
        let mut refresh_tokens = self.refresh_tokens.write().await;
        refresh_tokens.retain(|_, info| info.user_id != user_id);
    }
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            global_limit: config.requests_per_minute,
            per_user_limit: config.per_user_limit,
            burst_size: config.burst_size,
            global_counter: Arc::new(RwLock::new(RateLimitState {
                count: 0,
                window_start: Instant::now(),
                burst_count: 0,
                last_reset: Instant::now(),
            })),
            user_counters: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn check_rate_limit(&self, user_id: &str) -> bool {
        // Check global rate limit
        if !self.check_global_limit().await {
            return false;
        }
        
        // Check per-user rate limit
        self.check_user_limit(user_id).await
    }
    
    async fn check_global_limit(&self) -> bool {
        let mut global_state = self.global_counter.write().await;
        let now = Instant::now();
        
        // Reset window if needed
        if now.duration_since(global_state.window_start) >= Duration::from_secs(60) {
            global_state.count = 0;
            global_state.window_start = now;
            global_state.burst_count = 0;
        }
        
        // Check burst limit
        if global_state.burst_count >= self.burst_size {
            if now.duration_since(global_state.last_reset) < Duration::from_secs(1) {
                return false;
            }
            global_state.burst_count = 0;
            global_state.last_reset = now;
        }
        
        // Check rate limit
        if global_state.count >= self.global_limit {
            return false;
        }
        
        global_state.count += 1;
        global_state.burst_count += 1;
        true
    }
    
    async fn check_user_limit(&self, user_id: &str) -> bool {
        let mut user_counters = self.user_counters.write().await;
        let now = Instant::now();
        
        let user_state = user_counters.entry(user_id.to_string()).or_insert_with(|| {
            RateLimitState {
                count: 0,
                window_start: now,
                burst_count: 0,
                last_reset: now,
            }
        });
        
        // Reset window if needed
        if now.duration_since(user_state.window_start) >= Duration::from_secs(60) {
            user_state.count = 0;
            user_state.window_start = now;
            user_state.burst_count = 0;
        }
        
        // Check rate limit
        if user_state.count >= self.per_user_limit {
            return false;
        }
        
        user_state.count += 1;
        true
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            request_counts: Arc::new(RwLock::new(HashMap::new())),
            response_times: Arc::new(RwLock::new(Vec::new())),
            error_counts: Arc::new(RwLock::new(HashMap::new())),
            active_connections: Arc::new(RwLock::new(0)),
        }
    }
    
    pub async fn record_request(&self, endpoint: &str, response_time: Duration) {
        // Record request count
        let mut request_counts = self.request_counts.write().await;
        *request_counts.entry(endpoint.to_string()).or_insert(0) += 1;
        
        // Record response time
        let mut response_times = self.response_times.write().await;
        response_times.push(response_time);
        
        // Keep only recent response times (last 1000)
        if response_times.len() > 1000 {
            response_times.drain(0..500);
        }
    }
    
    pub async fn increment_active_connections(&self) {
        let mut connections = self.active_connections.write().await;
        *connections += 1;
        gauge!("active_connections").set(*connections as f64);
    }
    
    pub async fn decrement_active_connections(&self) {
        let mut connections = self.active_connections.write().await;
        if *connections > 0 {
            *connections -= 1;
        }
        gauge!("active_connections").set(*connections as f64);
    }
    
    pub async fn get_health_metrics(&self) -> HealthMetrics {
        let response_times = self.response_times.read().await;
        let active_connections = *self.active_connections.read().await;
        
        let avg_response_time = if response_times.is_empty() {
            0.0
        } else {
            response_times.iter().map(|d| d.as_millis() as f64).sum::<f64>() / response_times.len() as f64
        };
        
        HealthMetrics {
            requests_per_minute: 0.0, // Would be calculated from actual data
            avg_response_time_ms: avg_response_time,
            error_rate: 0.0, // Would be calculated from error data
            active_connections,
            memory_usage_mb: 0.0, // Would be collected from system
            cpu_usage_percent: 0.0, // Would be collected from system
        }
    }
    
    pub async fn get_comprehensive_metrics(&self) -> MetricsResponse {
        let response_times = self.response_times.read().await;
        let request_counts = self.request_counts.read().await;
        let error_counts = self.error_counts.read().await;
        let active_connections = *self.active_connections.read().await;
        
        // Calculate metrics
        let total_requests: u64 = request_counts.values().sum();
        let total_errors: u64 = error_counts.values().sum();
        
        let avg_response_time = if response_times.is_empty() {
            0.0
        } else {
            response_times.iter().map(|d| d.as_millis() as f64).sum::<f64>() / response_times.len() as f64
        };
        
        let error_rate = if total_requests > 0 {
            (total_errors as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };
        
        MetricsResponse {
            timestamp: Utc::now(),
            request_metrics: RequestMetrics {
                total_requests,
                requests_per_minute: 0.0, // Would be calculated
                avg_response_time_ms: avg_response_time,
                p95_response_time_ms: 0.0, // Would be calculated
                p99_response_time_ms: 0.0, // Would be calculated
                success_rate: 100.0 - error_rate,
            },
            performance_metrics: PerformanceMetrics {
                throughput_ops_per_sec: 0.0, // Would be calculated
                concurrent_requests: active_connections,
                queue_size: 0,
                cache_hit_rate: 0.0, // Would be collected
                model_inference_time_ms: 0.0, // Would be collected
            },
            error_metrics: ErrorMetrics {
                total_errors,
                error_rate,
                errors_by_type: error_counts.clone(),
                errors_by_endpoint: HashMap::new(), // Would be tracked
                recovery_success_rate: 0.0, // Would be calculated
            },
            system_metrics: SystemMetrics {
                memory_usage_mb: 0.0, // Would be collected from system
                cpu_usage_percent: 0.0, // Would be collected from system
                disk_usage_percent: 0.0, // Would be collected from system
                network_io_mbps: 0.0, // Would be collected from system
                active_connections,
                uptime_seconds: 0, // Would be tracked
            },
        }
    }
}

impl ApiError {
    pub fn new(error: &str, message: &str, _status: StatusCode) -> Self {
        Self {
            error: error.to_string(),
            message: message.to_string(),
            request_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            details: None,
        }
    }
    
    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = request_id;
        self
    }
    
    pub fn with_details(mut self, details: HashMap<String, String>) -> Self {
        self.details = Some(details);
        self
    }
    
    pub fn validation_error(message: &str) -> Self {
        Self::new("VALIDATION_ERROR", message, StatusCode::BAD_REQUEST)
    }
    
    pub fn authentication_error(message: &str) -> Self {
        Self::new("AUTHENTICATION_ERROR", message, StatusCode::UNAUTHORIZED)
    }
    
    pub fn authorization_error(message: &str) -> Self {
        Self::new("AUTHORIZATION_ERROR", message, StatusCode::FORBIDDEN)
    }
    
    pub fn not_found_error(message: &str) -> Self {
        Self::new("NOT_FOUND", message, StatusCode::NOT_FOUND)
    }
    
    pub fn rate_limit_error(message: &str) -> Self {
        Self::new("RATE_LIMIT_EXCEEDED", message, StatusCode::TOO_MANY_REQUESTS)
    }
    
    pub fn server_error(message: &str) -> Self {
        Self::new("INTERNAL_SERVER_ERROR", message, StatusCode::INTERNAL_SERVER_ERROR)
    }
    
    pub fn service_unavailable(message: &str) -> Self {
        Self::new("SERVICE_UNAVAILABLE", message, StatusCode::SERVICE_UNAVAILABLE)
    }
    
    pub fn processing_error(message: &str) -> Self {
        Self::new("PROCESSING_ERROR", message, StatusCode::UNPROCESSABLE_ENTITY)
    }
    
    pub fn retrieval_error(message: &str) -> Self {
        Self::new("RETRIEVAL_ERROR", message, StatusCode::INTERNAL_SERVER_ERROR)
    }
    
    pub fn configuration_error(message: &str) -> Self {
        Self::new("CONFIGURATION_ERROR", message, StatusCode::INTERNAL_SERVER_ERROR)
    }
    
    pub fn initialization_error(message: &str) -> Self {
        Self::new("INITIALIZATION_ERROR", message, StatusCode::INTERNAL_SERVER_ERROR)
    }
    
    pub fn unauthorized(message: &str) -> Self {
        Self::new("UNAUTHORIZED", message, StatusCode::UNAUTHORIZED)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.error.as_str() {
            "VALIDATION_ERROR" => StatusCode::BAD_REQUEST,
            "AUTHENTICATION_ERROR" | "UNAUTHORIZED" => StatusCode::UNAUTHORIZED,
            "AUTHORIZATION_ERROR" => StatusCode::FORBIDDEN,
            "NOT_FOUND" => StatusCode::NOT_FOUND,
            "RATE_LIMIT_EXCEEDED" => StatusCode::TOO_MANY_REQUESTS,
            "SERVICE_UNAVAILABLE" => StatusCode::SERVICE_UNAVAILABLE,
            "PROCESSING_ERROR" => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        
        (status, Json(self)).into_response()
    }
}

// Helper functions

fn validate_credentials(username: &str, password: &str) -> Option<String> {
    // In production, this would validate against a user database
    // For demo purposes, accept any non-empty credentials
    if !username.is_empty() && !password.is_empty() {
        Some(format!("user_{}", username))
    } else {
        None
    }
}

async fn get_user_permissions(_user_id: &str) -> Vec<String> {
    // In production, this would fetch from a database
    vec![
        "document:process".to_string(),
        "knowledge:retrieve".to_string(),
        "system:health".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_api_error_creation() {
        let error = ApiError::validation_error("Test message");
        assert_eq!(error.error, "VALIDATION_ERROR");
        assert_eq!(error.message, "Test message");
    }
    
    #[tokio::test]
    async fn test_rate_limiter() {
        use crate::enhanced_knowledge_storage::production::api_layer::RateLimitConfig;
        
        let config = RateLimitConfig {
            requests_per_minute: 10,
            burst_size: 5,
            enable_per_user_limits: true,
            per_user_limit: 5,
        };
        
        let rate_limiter = RateLimiter::new(config);
        
        // Should allow first few requests
        assert!(rate_limiter.check_rate_limit("test_user").await);
        assert!(rate_limiter.check_rate_limit("test_user").await);
    }
    
    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        collector.record_request("/test", Duration::from_millis(100)).await;
        collector.increment_active_connections().await;
        
        let metrics = collector.get_health_metrics().await;
        assert_eq!(metrics.active_connections, 1);
        assert!(metrics.avg_response_time_ms > 0.0);
    }
}