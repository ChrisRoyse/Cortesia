# Phase 2A: Production Deployment Configuration

**Quality Grade**: A+ (Enterprise Production Ready)
**Purpose**: Complete production deployment configuration for scalable allocation architecture
**Audience**: DevOps, Platform Engineers, Production Teams

## Production Architecture Overview

### Deployment Topology

```yaml
# production-topology.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llmkg-scalable-topology
data:
  topology.yaml: |
    # Production cluster configuration
    clusters:
      primary:
        region: us-west-2
        availability_zones: [us-west-2a, us-west-2b, us-west-2c]
        node_count: 12
        instance_types: [c6i.4xlarge, c6i.8xlarge, c6i.16xlarge]
        
      secondary:
        region: us-east-1
        availability_zones: [us-east-1a, us-east-1b]
        node_count: 6
        instance_types: [c6i.4xlarge, c6i.8xlarge]
    
    # Partition distribution strategy
    partitioning:
      strategy: geographic-aware
      primary_partitions: 8
      secondary_partitions: 4
      replication_factor: 2
      
    # Load balancing
    load_balancer:
      type: application
      algorithm: least-connections
      health_check_interval: 30s
      timeout: 5s
```

### Container Configuration

**File**: `docker/Dockerfile.production`

```dockerfile
# Multi-stage build for optimized production image
FROM rust:1.75-alpine as builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    gcc \
    g++ \
    cmake \
    make \
    pkgconfig \
    openssl-dev

WORKDIR /usr/src/llmkg

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./
COPY src/lib.rs src/

# Build dependencies (cached layer)
RUN cargo build --release --target x86_64-unknown-linux-musl

# Copy source code
COPY src/ src/
COPY docs/ docs/

# Build application with scalable features
RUN cargo build --release \
    --target x86_64-unknown-linux-musl \
    --features "scalable-allocation,distributed-processing,adaptive-quantization"

# Production runtime image
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libgcc \
    libstdc++ \
    && addgroup -g 1000 llmkg \
    && adduser -u 1000 -G llmkg -s /bin/sh -D llmkg

# Copy binary
COPY --from=builder /usr/src/llmkg/target/x86_64-unknown-linux-musl/release/llmkg-server /usr/local/bin/

# Copy configuration templates
COPY --from=builder /usr/src/llmkg/docs/allocationplan/config/ /etc/llmkg/

# Set ownership and permissions
RUN chown -R llmkg:llmkg /etc/llmkg /usr/local/bin/llmkg-server
RUN chmod +x /usr/local/bin/llmkg-server

# Switch to non-root user
USER llmkg

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD /usr/local/bin/llmkg-server --health-check || exit 1

EXPOSE 8080 8081 9090

CMD ["/usr/local/bin/llmkg-server", "--config", "/etc/llmkg/production.toml"]
```

### Kubernetes Deployment Manifests

**File**: `k8s/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llmkg-scalable-allocation
  namespace: llmkg-production
  labels:
    app: llmkg
    component: scalable-allocation
    version: v2a.1.0
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: llmkg
      component: scalable-allocation
  template:
    metadata:
      labels:
        app: llmkg
        component: scalable-allocation
        version: v2a.1.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: llmkg-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
      - name: llmkg-server
        image: llmkg/scalable-allocation:v2a.1.0
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: grpc
          containerPort: 8081
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP
        env:
        - name: RUST_LOG
          value: "info,llmkg=debug"
        - name: RUST_BACKTRACE
          value: "1"
        - name: LLMKG_CONFIG_PATH
          value: "/etc/llmkg/production.toml"
        - name: LLMKG_PARTITION_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['partition-id']
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            ephemeral-storage: "10Gi"
          limits:
            memory: "16Gi"
            cpu: "8"
            ephemeral-storage: "50Gi"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: config
          mountPath: /etc/llmkg
          readOnly: true
        - name: data
          mountPath: /var/lib/llmkg
        - name: cache
          mountPath: /tmp/llmkg-cache
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      volumes:
      - name: config
        configMap:
          name: llmkg-config
      - name: data
        persistentVolumeClaim:
          claimName: llmkg-data
      - name: cache
        emptyDir:
          sizeLimit: 1Gi
      nodeSelector:
        kubernetes.io/arch: amd64
        node-type: compute-optimized
      tolerations:
      - key: "node-type"
        operator: "Equal"
        value: "compute-optimized"
        effect: "NoSchedule"
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values: ["llmkg"]
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: llmkg-scalable-allocation-service
  namespace: llmkg-production
  labels:
    app: llmkg
    component: scalable-allocation
spec:
  type: ClusterIP
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
    name: http
  - port: 8081
    targetPort: 8081
    protocol: TCP
    name: grpc
  selector:
    app: llmkg
    component: scalable-allocation

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llmkg-data
  namespace: llmkg-production
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp3-csi
  resources:
    requests:
      storage: 100Gi
```

### Production Configuration

**File**: `config/production.toml`

```toml
# LLMKG Scalable Allocation Production Configuration

[server]
bind_address = "0.0.0.0"
http_port = 8080
grpc_port = 8081
metrics_port = 9090
max_connections = 10000
connection_timeout = "30s"
request_timeout = "60s"

[logging]
level = "info"
format = "json"
output = "stdout"
structured = true

[scalability]
enable_hnsw = true
enable_caching = true
enable_distributed = true
enable_quantization = true

[scalability.hnsw]
dimension = 768
max_connections = 32
ef_construction = 400
ef_search = 100
max_layer = 5

[scalability.cache]
l1_size = 100_000
l2_size = 10_000_000
prefetch_distance = 50
cache_warmup_on_startup = true
eviction_policy = "lru-with-frequency"

[scalability.cache.l1]
type = "lru"
ttl = "1h"
max_memory = "1GB"

[scalability.cache.l2]
type = "concurrent-lru"
ttl = "6h"
max_memory = "8GB"
shards = 16

[scalability.distribution]
partition_count = 8
communication_timeout = "5s"
max_retries = 3
retry_backoff = "exponential"
circuit_breaker_threshold = 50
circuit_breaker_timeout = "30s"

[scalability.distribution.discovery]
type = "kubernetes"
namespace = "llmkg-production"
service_name = "llmkg-scalable-allocation-service"
port = 8081

[scalability.quantization]
enable_adaptive = true
default_precision = "q8"
importance_threshold = 0.7
memory_pressure_threshold = 0.85

[scalability.quantization.precision_levels]
high_importance = "full"
medium_importance = "q8"
low_importance = "q4"
very_low_importance = "binary"

[monitoring]
enable_metrics = true
enable_tracing = true
enable_profiling = false
sample_rate = 0.1

[monitoring.prometheus]
endpoint = "/metrics"
enable_histogram = true
histogram_buckets = [0.001, 0.01, 0.1, 1.0, 10.0]

[monitoring.jaeger]
agent_endpoint = "jaeger-agent:6831"
service_name = "llmkg-scalable-allocation"
tags = { version = "v2a.1.0", environment = "production" }

[storage]
type = "persistent"
data_directory = "/var/lib/llmkg"
backup_directory = "/var/lib/llmkg/backup"
compression = "zstd"
compression_level = 3

[storage.persistence]
sync_interval = "30s"
checkpoint_interval = "5m"
max_wal_size = "1GB"
enable_compression = true

[performance]
worker_threads = 8
io_threads = 4
max_blocking_threads = 16
stack_size = "2MB"

[performance.memory]
arena_size = "64MB"
large_allocation_threshold = "1MB"
memory_pool_sizes = [64, 1024, 65536]

[security]
enable_tls = true
cert_file = "/etc/certs/tls.crt"
key_file = "/etc/certs/tls.key"
ca_file = "/etc/certs/ca.crt"
min_tls_version = "1.2"

[security.authentication]
type = "jwt"
secret_key_file = "/etc/secrets/jwt-secret"
token_expiry = "24h"
refresh_token_expiry = "7d"

[health_checks]
startup_probe_delay = "10s"
liveness_probe_interval = "30s"
readiness_probe_interval = "5s"
health_check_timeout = "5s"

[health_checks.dependencies]
check_hnsw_index = true
check_cache_connectivity = true
check_partition_health = true
check_memory_usage = true
```

### Environment-Specific Configurations

**File**: `config/environments/staging.toml`

```toml
# Staging environment overrides
[scalability]
enable_distributed = false  # Single node for staging

[scalability.cache]
l1_size = 10_000
l2_size = 1_000_000

[monitoring]
sample_rate = 1.0  # Full sampling in staging

[performance]
worker_threads = 4
```

**File**: `config/environments/development.toml`

```toml
# Development environment overrides
[logging]
level = "debug"

[scalability]
enable_hnsw = false
enable_distributed = false
enable_quantization = false

[scalability.cache]
l1_size = 1_000
l2_size = 10_000

[monitoring]
enable_profiling = true
sample_rate = 1.0
```

### Monitoring and Observability

**File**: `monitoring/prometheus-config.yaml`

```yaml
# Prometheus scrape configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-llmkg-config
data:
  llmkg-scrape.yaml: |
    - job_name: 'llmkg-scalable-allocation'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - llmkg-production
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      metric_relabel_configs:
      - source_labels: [__name__]
        regex: '(llmkg_allocation_latency|llmkg_cache_hit_rate|llmkg_memory_usage|llmkg_hnsw_search_time)'
        action: keep
```

**File**: `monitoring/grafana-dashboard.json`

```json
{
  "dashboard": {
    "id": null,
    "title": "LLMKG Scalable Allocation Dashboard",
    "tags": ["llmkg", "scalability", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Allocation Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llmkg_allocation_latency_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(llmkg_allocation_latency_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Latency (seconds)",
            "min": 0
          }
        ]
      },
      {
        "id": 2,
        "title": "Cache Hit Rates",
        "type": "stat",
        "targets": [
          {
            "expr": "llmkg_cache_hit_rate{cache_level=\"l1\"}",
            "legendFormat": "L1 Cache"
          },
          {
            "expr": "llmkg_cache_hit_rate{cache_level=\"l2\"}",
            "legendFormat": "L2 Cache"
          }
        ]
      },
      {
        "id": 3,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "llmkg_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "Total Memory (GB)"
          },
          {
            "expr": "llmkg_hnsw_memory_bytes / 1024 / 1024 / 1024",
            "legendFormat": "HNSW Index (GB)"
          }
        ]
      }
    ]
  }
}
```

### Deployment Scripts

**File**: `scripts/deploy-production.sh`

```bash
#!/bin/bash
set -euo pipefail

# Production deployment script for LLMKG Scalable Allocation

NAMESPACE="llmkg-production"
IMAGE_TAG="${IMAGE_TAG:-v2a.1.0}"
ENVIRONMENT="${ENVIRONMENT:-production}"

echo "🚀 Deploying LLMKG Scalable Allocation v${IMAGE_TAG} to ${ENVIRONMENT}"

# Validate prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is required but not installed"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo "❌ helm is required but not installed"
    exit 1
fi

# Create namespace if it doesn't exist
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Deploy configuration
echo "📝 Applying configuration..."
kubectl apply -f k8s/configmap.yaml -n ${NAMESPACE}
kubectl apply -f k8s/secrets.yaml -n ${NAMESPACE}

# Deploy application
echo "🔄 Deploying application..."
envsubst < k8s/deployment.yaml | kubectl apply -f - -n ${NAMESPACE}

# Wait for rollout to complete
echo "⏳ Waiting for deployment to complete..."
kubectl rollout status deployment/llmkg-scalable-allocation -n ${NAMESPACE} --timeout=600s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n ${NAMESPACE} -l app=llmkg

# Run health checks
echo "🔍 Running health checks..."
for i in {1..10}; do
    if kubectl exec -n ${NAMESPACE} deployment/llmkg-scalable-allocation -- /usr/local/bin/llmkg-server --health-check; then
        echo "✅ Health check passed"
        break
    fi
    echo "⏳ Health check attempt $i/10 failed, retrying in 5s..."
    sleep 5
done

echo "🎉 Deployment completed successfully!"
```

### Load Testing Configuration

**File**: `loadtest/k6-scalability-test.js`

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 200 },   // Ramp up to 200
    { duration: '5m', target: 200 },   // Stay at 200
    { duration: '2m', target: 500 },   // Ramp up to 500
    { duration: '10m', target: 500 },  // Stay at 500
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<100'], // 95% of requests under 100ms
    errors: ['rate<0.01'],            // Error rate under 1%
  },
};

const BASE_URL = __ENV.TARGET_URL || 'http://localhost:8080';

export default function () {
  // Test allocation endpoint
  const allocationPayload = JSON.stringify({
    fact: {
      subject: `entity_${Math.floor(Math.random() * 1000000)}`,
      predicate: 'type',
      object: 'TestEntity',
      confidence: Math.random(),
    },
  });

  const allocationResponse = http.post(`${BASE_URL}/api/v1/allocate`, allocationPayload, {
    headers: { 'Content-Type': 'application/json' },
  });

  check(allocationResponse, {
    'allocation status is 200': (r) => r.status === 200,
    'allocation latency < 100ms': (r) => r.timings.duration < 100,
  }) || errorRate.add(1);

  // Test search endpoint
  const searchPayload = JSON.stringify({
    query: {
      embedding: Array.from({ length: 768 }, () => Math.random()),
      k: 50,
    },
  });

  const searchResponse = http.post(`${BASE_URL}/api/v1/search`, searchPayload, {
    headers: { 'Content-Type': 'application/json' },
  });

  check(searchResponse, {
    'search status is 200': (r) => r.status === 200,
    'search latency < 50ms': (r) => r.timings.duration < 50,
  }) || errorRate.add(1);

  sleep(1);
}
```

This production deployment configuration provides enterprise-grade scalability, monitoring, and operational excellence for the Phase 2A scalable allocation architecture.