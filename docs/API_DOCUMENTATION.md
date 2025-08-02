# Enhanced Knowledge Storage API Documentation

## Overview

The Enhanced Knowledge Storage API provides a production-ready REST interface for AI-powered document processing and knowledge retrieval. It features comprehensive security, monitoring, and error handling capabilities.

## Features

### Core Functionality
- **Document Processing**: AI-powered semantic chunking, entity extraction, and relationship mapping
- **Knowledge Retrieval**: Multi-hop reasoning with contextual search
- **Real-time Processing**: Async processing with WebSocket support for long-running operations

### Security
- **JWT Authentication**: Bearer token authentication with refresh tokens
- **API Key Support**: Alternative authentication method for service-to-service communication
- **Rate Limiting**: Global and per-user rate limiting with burst protection
- **CORS Support**: Configurable cross-origin resource sharing
- **TLS/HTTPS**: Production-ready SSL/TLS encryption

### Monitoring & Observability
- **Health Checks**: Comprehensive system health monitoring
- **Metrics Collection**: Detailed performance and usage metrics
- **Request Tracing**: Full request lifecycle tracking with correlation IDs
- **Error Analytics**: Advanced error pattern detection and recovery

### Production Features
- **Circuit Breakers**: Automatic failure detection and recovery
- **Graceful Degradation**: Local resource management for system resilience
- **Auto-scaling**: Dynamic resource allocation based on load
- **Comprehensive Logging**: Structured logging with multiple levels

## API Endpoints

### Authentication

#### POST /api/v1/auth
Authenticate a user and receive access tokens.

**Request:**
```json
{
  "username": "your_username",
  "password": "your_password",
  "device_id": "optional_device_identifier"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "550e8400-e29b-41d4-a716-446655440000",
  "expires_in": 86400,
  "token_type": "Bearer",
  "user_id": "user_123",
  "permissions": ["document:process", "knowledge:retrieve"]
}
```

#### POST /api/v1/auth/refresh
Refresh an access token using a refresh token.

**Request:**
```json
{
  "refresh_token": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Document Processing

#### POST /api/v1/process
Process a document through the AI pipeline.

**Headers:**
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request:**
```json
{
  "content": "Your document content here...",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "source": "document_source",
    "language": "en",
    "tags": ["tag1", "tag2"],
    "created_at": "2024-01-01T00:00:00Z"
  },
  "options": {
    "chunk_strategy": "semantic",
    "extract_entities": true,
    "map_relationships": true,
    "generate_summary": true,
    "enable_reasoning": false
  }
}
```

**Response:**
```json
{
  "request_id": "req_123",
  "document_id": "doc_456",
  "status": "success",
  "chunks_created": 15,
  "entities_extracted": 42,
  "relationships_mapped": 18,
  "processing_time_ms": 2500,
  "summary": "Document processed successfully with high confidence",
  "warnings": []
}
```

### Knowledge Retrieval

#### POST /api/v1/retrieve
Query the knowledge base with advanced reasoning.

**Headers:**
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request:**
```json
{
  "query": "What are the key concepts in machine learning?",
  "max_results": 10,
  "similarity_threshold": 0.7,
  "options": {
    "use_reranking": true,
    "enable_reasoning": true,
    "include_explanations": true,
    "context_window": 5,
    "filter_by_source": ["academic_papers"],
    "filter_by_tags": ["machine_learning", "ai"]
  }
}
```

**Response:**
```json
{
  "request_id": "req_789",
  "query": "What are the key concepts in machine learning?",
  "results": [
    {
      "id": "chunk_123",
      "content": "Machine learning algorithms learn patterns from data...",
      "similarity_score": 0.95,
      "source": "Introduction to ML",
      "metadata": {
        "document_id": "doc_456",
        "chunk_index": "3"
      },
      "entities": ["machine learning", "algorithms", "data"],
      "relationships": ["algorithms LEARN_FROM data"],
      "explanation": "This chunk provides a comprehensive overview..."
    }
  ],
  "total_results": 8,
  "reasoning_steps": [
    {
      "step": 1,
      "description": "Query analysis: Identified key concepts query",
      "confidence": 0.9,
      "evidence": ["keyword: concepts", "domain: machine learning"]
    }
  ],
  "query_time_ms": 150,
  "suggestions": ["What are supervised learning algorithms?"]
}
```

### System Monitoring

#### GET /api/v1/health
Check system health status.

**Query Parameters:**
- `detailed`: Boolean - Include detailed component status

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "components": {
    "database": {
      "healthy": true,
      "message": "Connected and responsive",
      "last_check": "2024-01-01T12:00:00Z",
      "response_time_ms": 25
    }
  },
  "metrics": {
    "requests_per_minute": 45.2,
    "avg_response_time_ms": 120.5,
    "error_rate": 0.1,
    "active_connections": 12,
    "memory_usage_mb": 1024.0,
    "cpu_usage_percent": 15.3
  }
}
```

#### GET /api/v1/metrics
Get detailed system metrics (requires authentication).

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "request_metrics": {
    "total_requests": 10000,
    "requests_per_minute": 45.2,
    "avg_response_time_ms": 120.5,
    "p95_response_time_ms": 250.0,
    "p99_response_time_ms": 500.0,
    "success_rate": 99.9
  },
  "performance_metrics": {
    "throughput_ops_per_sec": 25.5,
    "concurrent_requests": 12,
    "queue_size": 3,
    "cache_hit_rate": 85.2,
    "model_inference_time_ms": 45.0
  },
  "error_metrics": {
    "total_errors": 10,
    "error_rate": 0.1,
    "errors_by_type": {
      "validation_error": 5,
      "processing_error": 3,
      "timeout_error": 2
    },
    "recovery_success_rate": 90.0
  },
  "system_metrics": {
    "memory_usage_mb": 1024.0,
    "cpu_usage_percent": 15.3,
    "disk_usage_percent": 45.0,
    "network_io_mbps": 12.5,
    "active_connections": 12,
    "uptime_seconds": 3600
  }
}
```

## Configuration

### Environment Variables

The API server can be configured using environment variables:

```bash
# System Configuration
LLMKG_MAX_CONCURRENT_REQUESTS=1000
LLMKG_REQUEST_TIMEOUT=30
LLMKG_MEMORY_LIMIT=8000000000

# API Configuration
LLMKG_API_HOST=0.0.0.0
LLMKG_API_PORT=8080

# Database Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
ELASTICSEARCH_URL=http://localhost:9200
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=llmkg
POSTGRES_USERNAME=postgres
POSTGRES_PASSWORD=password

# Security Configuration
JWT_SECRET=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# Feature Flags
ENABLE_GPU_ACCELERATION=false
ENABLE_AUTO_SCALING=false
```

### Configuration Files

Create a `config/production.toml` file:

```toml
[system]
name = "LLMKG Production System"
version = "1.0.0"
environment = "Production"
log_level = "Info"
max_concurrent_requests = 1000
request_timeout = "30s"
memory_limit = 8000000000

[api_config]
host = "0.0.0.0"
port = 8080
tls_enabled = true
cert_path = "/path/to/cert.pem"
key_path = "/path/to/key.pem"
cors_enabled = true
cors_origins = ["https://yourdomain.com"]
max_request_size = 10000000

[security_config]
authentication_enabled = true
jwt_expiration = "24h"
encryption_enabled = true

[security_config.rate_limiting]
requests_per_minute = 1000
burst_size = 100
enable_per_user_limits = true
per_user_limit = 100
```

## Error Handling

The API uses standardized error responses:

```json
{
  "error": "VALIDATION_ERROR",
  "message": "Document content cannot be empty",
  "request_id": "req_123",
  "timestamp": "2024-01-01T12:00:00Z",
  "details": {
    "field": "content",
    "constraint": "non_empty"
  }
}
```

### Error Codes

- `VALIDATION_ERROR` (400): Invalid request data
- `AUTHENTICATION_ERROR` (401): Invalid or missing authentication
- `AUTHORIZATION_ERROR` (403): Insufficient permissions
- `NOT_FOUND` (404): Resource not found
- `RATE_LIMIT_EXCEEDED` (429): Rate limit exceeded
- `PROCESSING_ERROR` (422): Document processing failed
- `INTERNAL_SERVER_ERROR` (500): Unexpected server error
- `SERVICE_UNAVAILABLE` (503): Service temporarily unavailable

## Rate Limiting

The API implements sophisticated rate limiting:

- **Global Limits**: Apply to all requests across users
- **Per-User Limits**: Individual user request quotas
- **Burst Protection**: Allow temporary spikes while maintaining average rates
- **Adaptive Throttling**: Dynamic adjustment based on system load

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
```

## Monitoring and Observability

### Request Tracing

Every request includes a unique request ID for tracking:
```
X-Request-ID: req_550e8400-e29b-41d4-a716-446655440000
```

### Metrics Collection

The system collects comprehensive metrics:
- Request counts and response times
- Error rates and types
- System resource utilization
- Cache hit rates
- Model inference performance

### Health Checks

Regular health checks monitor:
- Database connectivity
- Model availability
- Memory and CPU usage
- Queue lengths
- Error rates

## Deployment

### Docker Deployment

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/target/release/llmkg-api /usr/local/bin/
EXPOSE 8080
CMD ["llmkg-api"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llmkg-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llmkg-api
  template:
    metadata:
      labels:
        app: llmkg-api
    spec:
      containers:
      - name: api
        image: llmkg-api:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: LLMKG_API_PORT
          value: "8080"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: llmkg-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## Security Best Practices

1. **Use HTTPS in production**
2. **Rotate JWT secrets regularly**
3. **Implement proper CORS policies**
4. **Monitor for suspicious activity**
5. **Use strong encryption keys**
6. **Enable request logging**
7. **Implement IP whitelisting if needed**
8. **Regular security audits**

## Performance Optimization

1. **Enable compression** for large responses
2. **Use caching** for frequently accessed data
3. **Implement connection pooling** for databases
4. **Monitor and tune** rate limits
5. **Enable GPU acceleration** for model inference
6. **Use load balancing** for high availability
7. **Optimize database queries**
8. **Implement request batching** where possible

## Support and Troubleshooting

### Common Issues

1. **High memory usage**: Check model cache settings
2. **Slow response times**: Review database query performance
3. **Authentication failures**: Verify JWT secret configuration
4. **Rate limiting**: Adjust limits based on usage patterns

### Debug Mode

Enable debug logging:
```bash
RUST_LOG=debug ./llmkg-api
```

### Health Check Endpoints

Monitor system health programmatically:
- `/api/v1/health` - Basic health status
- `/api/v1/health?detailed=true` - Detailed component status
- `/api/v1/metrics` - Comprehensive metrics

For additional support, consult the system logs and monitoring dashboards.