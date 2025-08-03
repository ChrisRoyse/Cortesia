# E-commerce Platform

A comprehensive, modern e-commerce platform built with a microservices architecture, featuring a Python backend, React/TypeScript frontend, and Rust-based recommendation engine.

## 🏗️ Architecture Overview

### System Components

```mermaid
graph TB
    subgraph "Frontend Layer"
        FE[React/TypeScript Frontend]
        CDN[CDN/Static Assets]
    end
    
    subgraph "API Gateway"
        NGINX[Nginx Load Balancer]
        RATE[Rate Limiter]
    end
    
    subgraph "Backend Services"
        API[Python FastAPI Backend]
        REC[Rust Recommendation Engine]
        SEARCH[Elasticsearch Service]
    end
    
    subgraph "Data Layer"
        DB[(PostgreSQL Database)]
        REDIS[(Redis Cache)]
        S3[(S3 Object Storage)]
    end
    
    subgraph "External Services"
        STRIPE[Stripe Payment]
        EMAIL[Email Service]
        ANALYTICS[Analytics Service]
    end
    
    FE --> NGINX
    CDN --> FE
    NGINX --> API
    NGINX --> REC
    API --> DB
    API --> REDIS
    API --> SEARCH
    API --> S3
    API --> STRIPE
    API --> EMAIL
    REC --> DB
    API --> ANALYTICS
```

### Technology Stack

#### Backend Services
- **API Server**: Python 3.11+ with FastAPI
- **Recommendation Engine**: Rust with Actix-web
- **Database**: PostgreSQL 15+ with advanced features
- **Cache**: Redis 7+ with clustering support
- **Search**: Elasticsearch 8+ for product search

#### Frontend
- **Framework**: React 18+ with TypeScript
- **Build Tool**: Vite with modern optimizations
- **State Management**: React Context + useReducer
- **Styling**: CSS-in-JS with styled-components
- **Testing**: Jest + React Testing Library

#### Infrastructure
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Kubernetes with Helm charts
- **Load Balancing**: Nginx with SSL termination
- **Monitoring**: Prometheus, Grafana, Jaeger
- **CI/CD**: GitHub Actions with automated testing

## 🚀 Quick Start

### Prerequisites

- **Docker** 20.10+ and Docker Compose 2.0+
- **Node.js** 18+ and npm 8+
- **Python** 3.11+ with pip
- **Rust** 1.70+ with Cargo
- **PostgreSQL** 15+ (for local development)
- **Redis** 7+ (for local development)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ecommerce/platform.git
   cd platform
   ```

2. **Environment Configuration**
   ```bash
   # Copy environment templates
   cp .env.example .env
   cp backend/.env.example backend/.env
   cp frontend/.env.example frontend/.env
   
   # Update configuration values
   nano .env
   ```

3. **Database Setup**
   ```bash
   # Start PostgreSQL and Redis
   docker-compose up -d postgres redis
   
   # Run database migrations
   cd backend
   python -m alembic upgrade head
   
   # Seed initial data
   python scripts/seed_data.py
   ```

4. **Backend Development**
   ```bash
   cd backend
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   
   # Start development server
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Frontend Development**
   ```bash
   cd frontend
   
   # Install dependencies
   npm install
   
   # Start development server
   npm run dev
   ```

6. **Recommendation Engine**
   ```bash
   cd microservice
   
   # Build and run
   cargo run
   ```

### Docker Development Environment

For a complete development environment with all services:

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up

# View logs
docker-compose logs -f backend

# Access services
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Documentation: http://localhost:8000/docs
# - Recommendation Engine: http://localhost:8001
# - Database: localhost:5432
# - Redis: localhost:6379
```

## 📚 API Documentation

### Authentication

The platform uses JWT-based authentication with refresh tokens.

#### Register User
```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePass123!",
  "firstName": "John",
  "lastName": "Doe"
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "johndoe",
  "password": "SecurePass123!"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### Product Management

#### List Products
```http
GET /api/products?skip=0&limit=50&category_id=1&search=laptop
Authorization: Bearer {access_token}
```

#### Get Product Details
```http
GET /api/products/{product_id}
Authorization: Bearer {access_token}
```

#### Create Product (Admin Only)
```http
POST /api/products
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "Gaming Laptop",
  "description": "High-performance gaming laptop with RTX 4080",
  "price": 1999.99,
  "stockQuantity": 50,
  "categoryId": 1,
  "sku": "LAPTOP-GAMING-001"
}
```

### Order Management

#### Create Order
```http
POST /api/orders
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "items": [
    {
      "product_id": 1,
      "quantity": 2
    },
    {
      "product_id": 5,
      "quantity": 1
    }
  ],
  "shipping_address": "123 Main St, City, State 12345",
  "billing_address": "123 Main St, City, State 12345",
  "payment_method": "credit_card"
}
```

#### Get User Orders
```http
GET /api/orders
Authorization: Bearer {access_token}
```

### Recommendations

The recommendation engine provides personalized product suggestions using multiple algorithms.

#### Get Recommendations
```http
POST /recommendations
Content-Type: application/json

{
  "user_id": 123,
  "limit": 10,
  "algorithm": "hybrid",
  "include_categories": [1, 2, 3],
  "exclude_products": [45, 67]
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "product": {
        "id": 42,
        "name": "Wireless Headphones",
        "price": 199.99,
        "description": "Premium noise-cancelling headphones"
      },
      "score": 0.95,
      "reason": "Based on your purchase history + users with similar preferences",
      "confidence": 0.87
    }
  ],
  "algorithm_used": "Hybrid",
  "confidence_score": 0.84,
  "generated_at": "2024-01-15T10:30:00Z"
}
```

### Analytics

#### Get Sales Analytics
```http
POST /analytics
Content-Type: application/json

{
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T23:59:59Z",
  "metrics": ["total_sales", "conversion_rate", "average_order_value"],
  "group_by": "product"
}
```

## 🧪 Testing

### Backend Testing

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_products.py

# Run tests with markers
pytest -m "not slow"
```

### Frontend Testing

```bash
cd frontend

# Run unit tests
npm test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e

# Visual regression tests
npm run test:visual
```

### Integration Testing

```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/

# Load testing
k6 run tests/load/basic-load-test.js
```

## 🔧 Configuration

### Environment Variables

#### Backend (`backend/.env`)
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ecommerce
REDIS_URL=redis://localhost:6379/0

# Authentication
JWT_SECRET=your-super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# External Services
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SMTP_USERNAME=noreply@ecommerce.com
EMAIL_SMTP_PASSWORD=your-email-password

# Storage
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET=ecommerce-uploads

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
```

#### Frontend (`frontend/.env`)
```bash
VITE_API_URL=http://localhost:8000
VITE_STRIPE_PUBLIC_KEY=pk_test_your_stripe_public_key
VITE_GOOGLE_ANALYTICS_ID=GA_TRACKING_ID
VITE_RECAPTCHA_SITE_KEY=your_recaptcha_site_key
```

### Database Configuration

The platform uses PostgreSQL with advanced features:

- **Connection Pooling**: SQLAlchemy with configurable pool sizes
- **Read Replicas**: Automatic read/write splitting
- **Migrations**: Alembic for schema versioning
- **Backup**: Automated daily backups with point-in-time recovery
- **Monitoring**: Query performance tracking and slow query logging

### Caching Strategy

Multi-level caching approach:

1. **Application Level**: In-memory caching for frequently accessed data
2. **Redis**: Distributed caching for session data and temporary objects
3. **Database**: Query result caching with automatic invalidation
4. **CDN**: Static asset caching with long TTL

### Security Configuration

- **Authentication**: JWT with refresh token rotation
- **Authorization**: Role-based access control (RBAC)
- **Rate Limiting**: Per-endpoint and per-user limits
- **Data Protection**: Encryption at rest and in transit
- **CORS**: Configurable cross-origin resource sharing
- **CSP**: Content Security Policy headers
- **HTTPS**: Automatic SSL certificate management

## 📊 Monitoring and Observability

### Application Metrics

- **Performance**: Response times, throughput, error rates
- **Business**: Sales metrics, user engagement, conversion rates
- **System**: CPU, memory, disk usage, database performance
- **Custom**: Product views, cart abandonment, search queries

### Logging

Structured logging with:
- **Format**: JSON with consistent schema
- **Levels**: DEBUG, INFO, WARN, ERROR with contextual information
- **Correlation**: Request IDs for distributed tracing
- **Retention**: Configurable retention policies
- **Alerting**: Real-time alerts for critical errors

### Distributed Tracing

Jaeger integration for:
- **Request Flow**: Track requests across services
- **Performance**: Identify bottlenecks and slow operations
- **Dependencies**: Visualize service interactions
- **Error Tracking**: Detailed error context and stack traces

### Health Checks

- **Liveness**: Basic application health
- **Readiness**: Service dependencies and data consistency
- **Custom**: Business logic health checks
- **External**: Third-party service availability

## 🚀 Deployment

### Local Development

```bash
# Start all services
docker-compose up

# Scale specific services
docker-compose up --scale backend=3

# View service logs
docker-compose logs -f backend
```

### Staging Deployment

```bash
# Deploy to staging
kubectl apply -f k8s/staging/

# Update image version
kubectl set image deployment/backend backend=ecommerce/backend:v1.2.0

# Check deployment status
kubectl rollout status deployment/backend
```

### Production Deployment

```bash
# Deploy with Helm
helm upgrade --install ecommerce ./helm/ecommerce \
  --namespace production \
  --values helm/ecommerce/values.prod.yaml

# Blue-green deployment
kubectl apply -f k8s/production/blue-green/

# Canary deployment
flagger create canary backend \
  --target-ref=deployment/backend \
  --progress-deadline=5m \
  --canary-weight=10
```

### CI/CD Pipeline

GitHub Actions workflow:

1. **Test**: Unit tests, integration tests, security scans
2. **Build**: Docker images with multi-stage builds
3. **Deploy**: Automated deployment to staging
4. **Promote**: Manual promotion to production after validation

## 🔍 Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database connectivity
docker-compose exec backend python -c "from app.database import engine; print(engine.execute('SELECT 1').scalar())"

# View database logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up postgres
```

#### Authentication Problems
```bash
# Verify JWT configuration
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}'

# Check token validity
python scripts/verify_token.py <token>
```

#### Performance Issues
```bash
# Check application metrics
curl http://localhost:8000/metrics

# Database query analysis
docker-compose exec postgres psql -U user -d ecommerce -c "
  SELECT query, mean_time, calls 
  FROM pg_stat_statements 
  ORDER BY mean_time DESC 
  LIMIT 10;"

# Redis cache statistics
docker-compose exec redis redis-cli info stats
```

### Debugging Tools

- **Database**: pgAdmin, DataGrip for query analysis
- **API**: Postman collections for endpoint testing
- **Frontend**: React DevTools, Redux DevTools
- **Performance**: Chrome DevTools, Lighthouse
- **Monitoring**: Grafana dashboards, Prometheus alerts

## 🤝 Contributing

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** changes following coding standards
4. **Test** thoroughly with automated tests
5. **Commit** with conventional commit messages
6. **Push** to your fork: `git push origin feature/amazing-feature`
7. **Submit** a Pull Request with detailed description

### Code Standards

- **Python**: Black formatting, flake8 linting, type hints
- **TypeScript**: Prettier formatting, ESLint rules, strict mode
- **Rust**: Rustfmt formatting, Clippy linting, documentation
- **SQL**: Consistent naming, proper indexing, migration scripts
- **Documentation**: Up-to-date README, API docs, code comments

### Review Process

- **Automated**: Tests, linting, security scans must pass
- **Manual**: Code review by at least one team member
- **Performance**: No degradation in response times
- **Security**: No new vulnerabilities introduced
- **Documentation**: Updated for any API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.ecommerce.com](https://docs.ecommerce.com)
- **Issues**: [GitHub Issues](https://github.com/ecommerce/platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ecommerce/platform/discussions)
- **Email**: [support@ecommerce.com](mailto:support@ecommerce.com)
- **Slack**: [#ecommerce-dev](https://company.slack.com/channels/ecommerce-dev)

## 🗺️ Roadmap

### Version 1.1 (Q2 2024)
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Multi-currency support
- [ ] Inventory management system
- [ ] Loyalty program

### Version 1.2 (Q3 2024)
- [ ] Machine learning recommendations
- [ ] Real-time chat support
- [ ] Advanced search with filters
- [ ] Subscription products
- [ ] Multi-vendor marketplace

### Version 2.0 (Q4 2024)
- [ ] Microservices architecture
- [ ] Event-driven architecture
- [ ] GraphQL API
- [ ] Progressive Web App (PWA)
- [ ] International expansion

---

**Built with ❤️ by the E-commerce Team**