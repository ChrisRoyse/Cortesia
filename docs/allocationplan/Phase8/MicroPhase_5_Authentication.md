# MicroPhase 5: Authentication & Security (15 Micro-Tasks)

**Duration**: 4-5 hours  
**Priority**: Critical - Production security requirement  
**Prerequisites**: MicroPhase 1 (Foundation), MicroPhase 3 (Tool Schemas)

## Overview

Implement OAuth 2.1 authentication, JWT session management, rate limiting, and comprehensive security measures for the MCP server through atomic micro-tasks, each delivering ONE concrete security component in 15-20 minutes.

## Micro-Task Breakdown

---

## Micro-Task 5.1.1: Create OAuth2Config Structure
**Duration**: 18 minutes  
**Dependencies**: MicroPhase 1 (Foundation)  
**Input**: OAuth2 requirements and error types  
**Output**: Basic OAuth2 configuration struct  

### Task Prompt for AI
```
Create the OAuth2Config structure with validation:

```rust
use oauth2::{ClientId, ClientSecret, AuthUrl, TokenUrl, RedirectUrl, Scope};
use crate::mcp::errors::{MCPResult, MCPServerError};

#[derive(Debug, Clone)]
pub struct OAuth2Config {
    pub client_id: ClientId,
    pub client_secret: ClientSecret,
    pub auth_url: AuthUrl,
    pub token_url: TokenUrl,
    pub redirect_url: RedirectUrl,
    pub scopes: Vec<Scope>,
}

impl OAuth2Config {
    pub fn new(
        client_id: String,
        client_secret: String,
        auth_url: String,
        token_url: String,
        redirect_url: String,
    ) -> MCPResult<Self> {
        Ok(Self {
            client_id: ClientId::new(client_id),
            client_secret: ClientSecret::new(client_secret),
            auth_url: AuthUrl::new(auth_url)
                .map_err(|e| MCPServerError::AuthenticationError(format!("Invalid auth URL: {}", e)))?,
            token_url: TokenUrl::new(token_url)
                .map_err(|e| MCPServerError::AuthenticationError(format!("Invalid token URL: {}", e)))?,
            redirect_url: RedirectUrl::new(redirect_url)
                .map_err(|e| MCPServerError::AuthenticationError(format!("Invalid redirect URL: {}", e)))?,
            scopes: vec![
                Scope::new("cortex_kg:read".to_string()),
                Scope::new("cortex_kg:write".to_string()),
                Scope::new("cortex_kg:admin".to_string()),
            ],
        })
    }
}
```

Write ONE test verifying OAuth2Config creation with valid URLs.
```

**Expected Deliverable**: `src/mcp/auth/oauth_client.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.1.2: Create PendingAuth State Tracker
**Duration**: 15 minutes  
**Dependencies**: Task 5.1.1  
**Input**: OAuth2Config structure  
**Output**: CSRF state tracking  

### Task Prompt for AI
```
Add PendingAuth structure for CSRF protection:

```rust
use oauth2::CsrfToken;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PendingAuth {
    pub csrf_token: CsrfToken,
    pub created_at: DateTime<Utc>,
    pub scopes: Vec<Scope>,
}

impl PendingAuth {
    pub fn new(csrf_token: CsrfToken, scopes: Vec<Scope>) -> Self {
        Self {
            csrf_token,
            created_at: Utc::now(),
            scopes,
        }
    }
    
    pub fn is_expired(&self, timeout_minutes: i64) -> bool {
        let cutoff = Utc::now() - chrono::Duration::minutes(timeout_minutes);
        self.created_at < cutoff
    }
}
```

Write ONE test verifying PendingAuth expiration logic.
```

**Expected Deliverable**: Updated `src/mcp/auth/oauth_client.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.1.3: Create OAuth2Handler Structure
**Duration**: 17 minutes  
**Dependencies**: Task 5.1.2  
**Input**: OAuth2Config and PendingAuth  
**Output**: Basic OAuth2 client wrapper  

### Task Prompt for AI
```
Create OAuth2Handler with BasicClient:

```rust
use oauth2::basic::BasicClient;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct OAuth2Handler {
    client: BasicClient,
    config: OAuth2Config,
    pending_states: Arc<RwLock<HashMap<String, PendingAuth>>>,
}

impl OAuth2Handler {
    pub fn new(config: OAuth2Config) -> Self {
        let client = BasicClient::new(
            config.client_id.clone(),
            Some(config.client_secret.clone()),
            config.auth_url.clone(),
            Some(config.token_url.clone()),
        )
        .set_redirect_uri(config.redirect_url.clone());

        Self {
            client,
            config,
            pending_states: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}
```

Write ONE test verifying OAuth2Handler creation.
```

**Expected Deliverable**: Updated `src/mcp/auth/oauth_client.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.1.4: Implement Authorization URL Generation
**Duration**: 20 minutes  
**Dependencies**: Task 5.1.3  
**Input**: OAuth2Handler structure  
**Output**: Authorization URL generation method  

### Task Prompt for AI
```
Add authorization URL generation with CSRF protection:

```rust
use url::Url;

impl OAuth2Handler {
    pub async fn generate_authorization_url(&self, user_id: Option<String>) -> MCPResult<(Url, String)> {
        let (auth_url, csrf_token) = self
            .client
            .authorize_url(oauth2::CsrfToken::new_random)
            .add_scopes(self.config.scopes.clone())
            .url();

        let state = csrf_token.secret().clone();
        let pending_auth = PendingAuth::new(csrf_token, self.config.scopes.clone());

        self.pending_states.write().await.insert(state.clone(), pending_auth);
        self.cleanup_expired_states().await;

        Ok((auth_url, state))
    }

    async fn cleanup_expired_states(&self) {
        let mut pending_states = self.pending_states.write().await;
        pending_states.retain(|_, pending| !pending.is_expired(10));
    }
}
```

Write ONE test verifying URL generation and state storage.
```

**Expected Deliverable**: Updated `src/mcp/auth/oauth_client.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.1.5: Create TokenResponse Structure
**Duration**: 16 minutes  
**Dependencies**: Task 5.1.4  
**Input**: OAuth2 token requirements  
**Output**: Token response wrapper  

### Task Prompt for AI
```
Create TokenResponse for OAuth2 tokens:

```rust
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: Option<Duration>,
    pub scopes: Vec<Scope>,
    pub token_type: String,
}

impl TokenResponse {
    pub fn new(
        access_token: String,
        refresh_token: Option<String>,
        expires_in: Option<Duration>,
        scopes: Vec<Scope>,
    ) -> Self {
        Self {
            access_token,
            refresh_token,
            expires_in,
            scopes,
            token_type: "Bearer".to_string(),
        }
    }
    
    pub fn is_expired(&self) -> bool {
        // Placeholder for token expiration logic
        false
    }
}
```

Write ONE test verifying TokenResponse creation and serialization.
```

**Expected Deliverable**: Updated `src/mcp/auth/oauth_client.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.1.6: Implement Token Exchange Method
**Duration**: 19 minutes  
**Dependencies**: Task 5.1.5  
**Input**: OAuth2Handler and TokenResponse  
**Output**: Code-to-token exchange method  

### Task Prompt for AI
```
Add token exchange functionality:

```rust
use oauth2::AuthorizationCode;

impl OAuth2Handler {
    pub async fn exchange_code(
        &self,
        code: AuthorizationCode,
        state: String,
    ) -> MCPResult<TokenResponse> {
        // Verify CSRF state
        let pending_auth = {
            let mut pending_states = self.pending_states.write().await;
            pending_states.remove(&state)
                .ok_or_else(|| MCPServerError::AuthenticationError("Invalid or expired state".to_string()))?
        };

        // Exchange authorization code for access token
        let token_result = self
            .client
            .exchange_code(code)
            .request_async(oauth2::reqwest::async_http_client)
            .await
            .map_err(|e| MCPServerError::AuthenticationError(format!("Token exchange failed: {}", e)))?;

        Ok(TokenResponse::new(
            token_result.access_token().secret().clone(),
            token_result.refresh_token().map(|t| t.secret().clone()),
            token_result.expires_in(),
            token_result.scopes().cloned().unwrap_or_default(),
        ))
    }
}
```

Write ONE test verifying state validation (mock the HTTP client).
```

**Expected Deliverable**: Updated `src/mcp/auth/oauth_client.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.2.1: Create JWT Claims Structure
**Duration**: 18 minutes  
**Dependencies**: MicroPhase 1 (Foundation)  
**Input**: JWT requirements  
**Output**: JWT claims with neuromorphic fields  

### Task Prompt for AI
```
Create JWT Claims structure with custom fields:

```rust
use jsonwebtoken::{Algorithm, Header, Validation};
use serde::{Deserialize, Serialize};
use chrono::{Duration, Utc};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    // Standard JWT claims
    pub sub: String,          // Subject (user ID)
    pub iss: String,          // Issuer
    pub aud: String,          // Audience
    pub exp: i64,             // Expiration time
    pub iat: i64,             // Issued at
    pub nbf: i64,             // Not before
    pub jti: String,          // JWT ID
    
    // Custom claims for neuromorphic memory
    pub session_id: String,
    pub permissions: Vec<String>,
    pub cortex_access_level: String,
    pub memory_quota: Option<usize>,
    pub rate_limit_tier: String,
}

impl Claims {
    pub fn new(
        user_id: String,
        session_id: String,
        permissions: Vec<String>,
        access_level: String,
        expires_in: Duration,
    ) -> Self {
        let now = Utc::now();
        let exp = (now + expires_in).timestamp();
        
        Self {
            sub: user_id,
            iss: "cortex-kg-mcp-server".to_string(),
            aud: "cortex-kg-users".to_string(),
            exp,
            iat: now.timestamp(),
            nbf: now.timestamp(),
            jti: Uuid::new_v4().to_string(),
            session_id,
            permissions,
            cortex_access_level: access_level,
            memory_quota: None,
            rate_limit_tier: "standard".to_string(),
        }
    }
    
    pub fn is_expired(&self) -> bool {
        Utc::now().timestamp() > self.exp
    }
    
    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions.contains(&permission.to_string()) ||
        self.permissions.contains(&"cortex_kg:admin".to_string())
    }
}
```

Write ONE test verifying Claims creation and permission checking.
```

**Expected Deliverable**: `src/mcp/auth/jwt_manager.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.2.2: Create JWTManager Structure
**Duration**: 17 minutes  
**Dependencies**: Task 5.2.1  
**Input**: JWT Claims structure  
**Output**: JWT encoding/decoding manager  

### Task Prompt for AI
```
Create JWTManager with encoding/decoding keys:

```rust
use jsonwebtoken::{DecodingKey, EncodingKey, Validation};
use std::collections::HashSet;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct JWTManager {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    validation: Validation,
    blacklisted_tokens: Arc<RwLock<HashSet<String>>>,
    default_expiry: Duration,
}

impl JWTManager {
    pub fn new(secret: &[u8]) -> MCPResult<Self> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.set_issuer(&["cortex-kg-mcp-server"]);
        validation.set_audience(&["cortex-kg-users"]);
        
        Ok(Self {
            encoding_key: EncodingKey::from_secret(secret),
            decoding_key: DecodingKey::from_secret(secret),
            validation,
            blacklisted_tokens: Arc::new(RwLock::new(HashSet::new())),
            default_expiry: Duration::hours(1),
        })
    }
}
```

Write ONE test verifying JWTManager creation with valid secret.
```

**Expected Deliverable**: Updated `src/mcp/auth/jwt_manager.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.2.3: Implement JWT Token Creation
**Duration**: 16 minutes  
**Dependencies**: Task 5.2.2  
**Input**: JWTManager structure  
**Output**: Token creation method  

### Task Prompt for AI
```
Add JWT token creation method:

```rust
use jsonwebtoken::encode;

impl JWTManager {
    pub async fn create_token(&self, claims: Claims) -> MCPResult<String> {
        let header = Header::new(Algorithm::HS256);
        
        encode(&header, &claims, &self.encoding_key)
            .map_err(|e| MCPServerError::AuthenticationError(format!("JWT creation failed: {}", e)))
    }
}
```

Write ONE test verifying token creation produces valid JWT string.
```

**Expected Deliverable**: Updated `src/mcp/auth/jwt_manager.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.2.4: Implement JWT Token Verification
**Duration**: 19 minutes  
**Dependencies**: Task 5.2.3  
**Input**: JWT token creation method  
**Output**: Token verification with blacklist checking  

### Task Prompt for AI
```
Add JWT token verification with blacklist checking:

```rust
use jsonwebtoken::decode;

impl JWTManager {
    pub async fn verify_token(&self, token: &str) -> MCPResult<Claims> {
        // Check if token is blacklisted
        if self.blacklisted_tokens.read().await.contains(token) {
            return Err(MCPServerError::AuthenticationError("Token has been revoked".to_string()));
        }
        
        let token_data = decode::<Claims>(token, &self.decoding_key, &self.validation)
            .map_err(|e| MCPServerError::AuthenticationError(format!("JWT verification failed: {}", e)))?;
        
        let claims = token_data.claims;
        
        // Additional validation
        if claims.is_expired() {
            return Err(MCPServerError::AuthenticationError("Token has expired".to_string()));
        }
        
        Ok(claims)
    }
    
    pub async fn blacklist_token(&self, token: &str) {
        self.blacklisted_tokens.write().await.insert(token.to_string());
    }
}
```

Write ONE test verifying token verification and blacklisting.
```

**Expected Deliverable**: Updated `src/mcp/auth/jwt_manager.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.3.1: Create RateLimitConfig Structure
**Duration**: 15 minutes  
**Dependencies**: MicroPhase 1 (Foundation)  
**Input**: Rate limiting requirements  
**Output**: Rate limit configuration structure  

### Task Prompt for AI
```
Create RateLimitConfig with default implementations:

```rust
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub requests_per_minute: usize,
    pub requests_per_hour: usize,
    pub burst_limit: usize,
    pub memory_operations_per_minute: usize,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            requests_per_hour: 1000,
            burst_limit: 10,
            memory_operations_per_minute: 30,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RateLimitTier {
    pub name: String,
    pub config: RateLimitConfig,
}

impl RateLimitTier {
    pub fn standard() -> Self {
        Self {
            name: "standard".to_string(),
            config: RateLimitConfig::default(),
        }
    }
    
    pub fn premium() -> Self {
        Self {
            name: "premium".to_string(),
            config: RateLimitConfig {
                requests_per_minute: 120,
                requests_per_hour: 5000,
                burst_limit: 20,
                memory_operations_per_minute: 100,
            },
        }
    }
}
```

Write ONE test verifying tier configurations are sensible.
```

**Expected Deliverable**: `src/mcp/auth/rate_limiter.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.3.2: Create UserLimitTracker Structure
**Duration**: 18 minutes  
**Dependencies**: Task 5.3.1  
**Input**: Rate limit config structure  
**Output**: Per-user request tracking  

### Task Prompt for AI
```
Create UserLimitTracker for per-user rate limiting:

```rust
use chrono::{DateTime, Utc, Duration};

#[derive(Debug, Clone)]
pub struct UserLimitTracker {
    pub minute_requests: Vec<DateTime<Utc>>,
    pub hour_requests: Vec<DateTime<Utc>>,
    pub memory_operations: Vec<DateTime<Utc>>,
    pub last_cleanup: DateTime<Utc>,
}

impl UserLimitTracker {
    pub fn new() -> Self {
        Self {
            minute_requests: Vec::new(),
            hour_requests: Vec::new(),
            memory_operations: Vec::new(),
            last_cleanup: Utc::now(),
        }
    }
    
    pub fn cleanup_expired(&mut self) {
        let now = Utc::now();
        let minute_ago = now - Duration::minutes(1);
        let hour_ago = now - Duration::hours(1);
        
        self.minute_requests.retain(|&time| time > minute_ago);
        self.hour_requests.retain(|&time| time > hour_ago);
        self.memory_operations.retain(|&time| time > minute_ago);
        
        self.last_cleanup = now;
    }
    
    pub fn should_cleanup(&self) -> bool {
        Utc::now() - self.last_cleanup > Duration::seconds(30)
    }
}
```

Write ONE test verifying cleanup removes old entries.
```

**Expected Deliverable**: Updated `src/mcp/auth/rate_limiter.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.3.3: Create RateLimitResult Enum
**Duration**: 16 minutes  
**Dependencies**: Task 5.3.2  
**Input**: User tracking structure  
**Output**: Rate limit result with HTTP headers  

### Task Prompt for AI
```
Create RateLimitResult with header generation:

```rust
#[derive(Debug, Clone)]
pub enum OperationType {
    GeneralRequest,
    MemoryOperation,
}

#[derive(Debug, Clone)]
pub enum RateLimitResult {
    Allowed {
        remaining_requests: usize,
        reset_time: DateTime<Utc>,
    },
    Exceeded {
        limit_type: String,
        current_count: usize,
        limit: usize,
        reset_time: DateTime<Utc>,
    },
}

impl RateLimitResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, RateLimitResult::Allowed { .. })
    }
    
    pub fn to_headers(&self) -> Vec<(String, String)> {
        match self {
            RateLimitResult::Allowed { remaining_requests, reset_time } => {
                vec![
                    ("X-RateLimit-Remaining".to_string(), remaining_requests.to_string()),
                    ("X-RateLimit-Reset".to_string(), reset_time.timestamp().to_string()),
                ]
            },
            RateLimitResult::Exceeded { limit, reset_time, .. } => {
                vec![
                    ("X-RateLimit-Limit".to_string(), limit.to_string()),
                    ("X-RateLimit-Reset".to_string(), reset_time.timestamp().to_string()),
                    ("Retry-After".to_string(), "60".to_string()),
                ]
            },
        }
    }
}
```

Write ONE test verifying header generation for both result types.
```

**Expected Deliverable**: Updated `src/mcp/auth/rate_limiter.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.3.4: Implement RateLimiter Core Logic
**Duration**: 20 minutes  
**Dependencies**: Task 5.3.3  
**Input**: Rate limit structures and results  
**Output**: Core rate limiting implementation  

### Task Prompt for AI
```
Create RateLimiter with check_rate_limit method:

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct RateLimiter {
    user_trackers: Arc<RwLock<HashMap<String, UserLimitTracker>>>,
    tiers: HashMap<String, RateLimitTier>,
}

impl RateLimiter {
    pub fn new() -> Self {
        let mut tiers = HashMap::new();
        tiers.insert("standard".to_string(), RateLimitTier::standard());
        tiers.insert("premium".to_string(), RateLimitTier::premium());
        
        Self {
            user_trackers: Arc::new(RwLock::new(HashMap::new())),
            tiers,
        }
    }
    
    pub async fn check_rate_limit(
        &self,
        user_id: &str,
        tier: &str,
        operation_type: OperationType,
    ) -> MCPResult<RateLimitResult> {
        let config = self.tiers.get(tier)
            .map(|t| &t.config)
            .unwrap_or(&RateLimitConfig::default());
        
        let mut trackers = self.user_trackers.write().await;
        let tracker = trackers.entry(user_id.to_string())
            .or_insert_with(UserLimitTracker::new);
        
        if tracker.should_cleanup() {
            tracker.cleanup_expired();
        }
        
        let now = Utc::now();
        
        // Check minute limit
        if tracker.minute_requests.len() >= config.requests_per_minute {
            return Ok(RateLimitResult::Exceeded {
                limit_type: "requests_per_minute".to_string(),
                current_count: tracker.minute_requests.len(),
                limit: config.requests_per_minute,
                reset_time: now + Duration::minutes(1),
            });
        }
        
        // Record the request
        tracker.minute_requests.push(now);
        
        Ok(RateLimitResult::Allowed {
            remaining_requests: config.requests_per_minute - tracker.minute_requests.len(),
            reset_time: now + Duration::minutes(1),
        })
    }
}
```

Write ONE test verifying basic rate limiting works.
```

**Expected Deliverable**: Updated `src/mcp/auth/rate_limiter.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.4.1: Create AuthContext Structure
**Duration**: 17 minutes  
**Dependencies**: Tasks 5.2.x (JWT)  
**Input**: JWT Claims and access levels  
**Output**: Authentication context wrapper  

### Task Prompt for AI
```
Create AuthContext for request authentication:

```rust
#[derive(Debug, Clone)]
pub enum AccessLevel {
    Read,
    Write,
    Admin,
}

impl std::fmt::Display for AccessLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessLevel::Read => write!(f, "read"),
            AccessLevel::Write => write!(f, "write"),
            AccessLevel::Admin => write!(f, "admin"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: String,
    pub session_id: String,
    pub permissions: Vec<String>,
    pub access_level: AccessLevel,
    pub rate_limit_tier: String,
    pub memory_quota: Option<usize>,
}

impl AuthContext {
    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions.contains(&permission.to_string()) ||
        self.permissions.contains(&"cortex_kg:admin".to_string())
    }
    
    pub fn can_access_memory_operation(&self) -> bool {
        self.has_permission("cortex_kg:write") || self.has_permission("cortex_kg:read")
    }
    
    pub fn can_modify_memory(&self) -> bool {
        self.has_permission("cortex_kg:write") || self.has_permission("cortex_kg:admin")
    }
}
```

Write ONE test verifying permission checking logic.
```

**Expected Deliverable**: `src/mcp/auth/middleware.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.4.2: Create AuthenticationMiddleware Structure
**Duration**: 19 minutes  
**Dependencies**: Task 5.4.1  
**Input**: AuthContext and all auth components  
**Output**: Middleware with authentication logic  

### Task Prompt for AI
```
Create AuthenticationMiddleware for request processing:

```rust
use super::{
    jwt_manager::JWTManager,
    rate_limiter::RateLimiter,
    oauth_client::OAuth2Handler,
};
use std::sync::Arc;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MCPRequest {
    pub method: String,
    pub params: serde_json::Value,
    pub headers: HashMap<String, String>,
    pub id: String,
}

pub struct AuthenticationMiddleware {
    jwt_manager: Arc<JWTManager>,
    rate_limiter: Arc<RateLimiter>,
    oauth_handler: Arc<OAuth2Handler>,
    require_auth: bool,
}

impl AuthenticationMiddleware {
    pub fn new(
        jwt_manager: Arc<JWTManager>,
        rate_limiter: Arc<RateLimiter>,
        oauth_handler: Arc<OAuth2Handler>,
        require_auth: bool,
    ) -> Self {
        Self {
            jwt_manager,
            rate_limiter,
            oauth_handler,
            require_auth,
        }
    }
    
    fn extract_bearer_token(&self, auth_header: &str) -> MCPResult<String> {
        if !auth_header.starts_with("Bearer ") {
            return Err(MCPServerError::AuthenticationError(
                "Invalid authorization header format. Expected 'Bearer <token>'".to_string()
            ));
        }
        
        let token = auth_header.strip_prefix("Bearer ")
            .ok_or_else(|| MCPServerError::AuthenticationError("Invalid bearer token format".to_string()))?;
        
        if token.is_empty() {
            return Err(MCPServerError::AuthenticationError("Empty bearer token".to_string()));
        }
        
        Ok(token.to_string())
    }
}
```

Write ONE test verifying bearer token extraction.
```

**Expected Deliverable**: Updated `src/mcp/auth/middleware.rs`  
**Verification**: Compiles + 1 test passes  

---

## Micro-Task 5.5.1: Create Authentication Integration Test
**Duration**: 18 minutes  
**Dependencies**: All previous auth tasks  
**Input**: Complete authentication system  
**Output**: End-to-end authentication test  

### Task Prompt for AI
```
Create integration test for complete authentication flow:

```rust
use cortex_kg::mcp::auth::{
    jwt_manager::{JWTManager, Claims},
    oauth_client::{OAuth2Config, OAuth2Handler},
    rate_limiter::RateLimiter,
    middleware::{AuthenticationMiddleware, MCPRequest, AccessLevel},
};
use std::sync::Arc;
use std::collections::HashMap;
use chrono::Duration;

async fn setup_auth_components() -> (Arc<JWTManager>, Arc<RateLimiter>, Arc<OAuth2Handler>) {
    let jwt_manager = Arc::new(JWTManager::new(b"test_secret_key_32_bytes_minimum").unwrap());
    let rate_limiter = Arc::new(RateLimiter::new());
    
    let oauth_config = OAuth2Config::new(
        "test_client".to_string(),
        "test_secret".to_string(),
        "https://auth.example.com/oauth/authorize".to_string(),
        "https://auth.example.com/oauth/token".to_string(),
        "https://localhost:8080/callback".to_string(),
    ).unwrap();
    let oauth_handler = Arc::new(OAuth2Handler::new(oauth_config));
    
    (jwt_manager, rate_limiter, oauth_handler)
}

#[tokio::test]
async fn test_full_authentication_flow() {
    let (jwt_manager, rate_limiter, oauth_handler) = setup_auth_components().await;
    let middleware = AuthenticationMiddleware::new(jwt_manager.clone(), rate_limiter, oauth_handler, true);
    
    // Create a valid JWT token
    let claims = Claims::new(
        "user123".to_string(),
        "session456".to_string(),
        vec!["cortex_kg:read".to_string(), "cortex_kg:write".to_string()],
        "write".to_string(),
        Duration::hours(1),
    );
    
    let token = jwt_manager.create_token(claims).await.unwrap();
    
    // Test bearer token extraction
    let result = middleware.extract_bearer_token(&format!("Bearer {}", token));
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), token);
}

#[tokio::test]
async fn test_invalid_authentication() {
    let (jwt_manager, rate_limiter, oauth_handler) = setup_auth_components().await;
    let middleware = AuthenticationMiddleware::new(jwt_manager, rate_limiter, oauth_handler, true);
    
    // Test invalid bearer token format
    let result = middleware.extract_bearer_token("Invalid token format");
    assert!(result.is_err());
    
    // Test empty token
    let result = middleware.extract_bearer_token("Bearer ");
    assert!(result.is_err());
}
```

Write additional tests for token verification and permission checking.
```

**Expected Deliverable**: `tests/authentication_integration_test.rs`  
**Verification**: Compiles + 2 tests pass  

## Summary

**Total Micro-Tasks**: 15 tasks  
**Total Time**: ~4-5 hours (15 × 15-20 minutes)  
**Deliverables**:
- `src/mcp/auth/oauth_client.rs` (OAuth2 client implementation)
- `src/mcp/auth/jwt_manager.rs` (JWT token management)  
- `src/mcp/auth/rate_limiter.rs` (Rate limiting system)
- `src/mcp/auth/middleware.rs` (Authentication middleware)
- `tests/authentication_integration_test.rs` (Integration tests)

**Key Benefits**:
- Each micro-task delivers ONE security component in 15-20 minutes
- Clear progression from config → tokens → middleware → integration
- Incremental testing at each step ensures working security system
- AI can complete each task independently with concrete verification
- Follows established micro-task patterns from Phase 1/2

## Validation Checklist

**Security Components**:
- [ ] All 15 micro-tasks completed
- [ ] OAuth2 client with CSRF protection implemented
- [ ] JWT creation/verification with blacklisting working
- [ ] Multi-tier rate limiting functional
- [ ] Authentication middleware validates requests
- [ ] Tool-specific permission checking operational
- [ ] Memory quota validation implemented
- [ ] Integration tests verify complete auth flow
- [ ] All tests passing (15+ individual unit tests)
- [ ] Security headers properly generated for rate limits

**Quality Gates**:
- [ ] OAuth2 authorization flow generates valid URLs with state tracking
- [ ] JWT tokens can be created, verified, and blacklisted
- [ ] Rate limiting enforces limits per tier and operation type
- [ ] Authentication middleware extracts and validates bearer tokens
- [ ] Permission-based tool access control working
- [ ] Invalid authentication scenarios properly handled
- [ ] Ready for MicroPhase 6 performance optimization integration

## Next Phase Dependencies

This micro-phase provides the security foundation for:
- **MicroPhase 6**: Performance optimization with authenticated requests
- **MicroPhase 7**: Testing authentication scenarios in isolation
- **MicroPhase 8**: Integration testing with full security enabled
- **Production deployment**: Complete security infrastructure ready

**Critical Path**: Authentication is required for production deployment. All secured endpoints depend on this phase completing successfully.

## Validation Checklist

- [ ] OAuth 2.1 client configuration and authorization flow working
- [ ] JWT token creation, verification, and refresh implemented
- [ ] Rate limiting with multiple tiers and operation types functional
- [ ] Authentication middleware properly validates requests
- [ ] Permission-based tool access control implemented
- [ ] Memory quota validation working
- [ ] Token blacklisting for security implemented
- [ ] Comprehensive error handling for all auth scenarios
- [ ] Integration tests verify complete auth flow
- [ ] Security headers and rate limit information provided

## Next Phase Dependencies

This phase provides authentication foundation for:
- MicroPhase 6: Performance optimization with authenticated requests
- MicroPhase 7: Testing authentication scenarios
- MicroPhase 8: Integration testing with security enabled