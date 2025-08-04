# MP055: Authentication & Authorization

## Task Description
Implement comprehensive authentication and authorization system for securing graph algorithm access, user management, and role-based permissions across neuromorphic components.

## Prerequisites
- MP001-MP050 completed
- Understanding of authentication protocols (JWT, OAuth2)
- Knowledge of authorization patterns (RBAC, ABAC)
- Familiarity with cryptographic principles

## Detailed Steps

1. Create `src/neuromorphic/security/auth.rs`

2. Implement JWT-based authentication system:
   ```rust
   use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
   use serde::{Deserialize, Serialize};
   use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
   use argon2::password_hash::{rand_core::OsRng, SaltString};
   use uuid::Uuid;
   use chrono::{DateTime, Duration, Utc};
   use std::collections::HashMap;
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct Claims {
       pub sub: String,           // Subject (user ID)
       pub username: String,      // Username
       pub roles: Vec<String>,    // User roles
       pub permissions: Vec<String>, // Specific permissions
       pub exp: i64,             // Expiration time
       pub iat: i64,             // Issued at
       pub jti: String,          // JWT ID for revocation
       pub scope: Vec<String>,   // API scopes
   }
   
   #[derive(Debug, Clone)]
   pub struct AuthenticationService {
       encoding_key: EncodingKey,
       decoding_key: DecodingKey,
       validation: Validation,
       user_store: Arc<dyn UserStore>,
       session_store: Arc<dyn SessionStore>,
       password_hasher: Argon2<'static>,
       token_config: TokenConfig,
   }
   
   #[derive(Debug, Clone)]
   pub struct TokenConfig {
       pub access_token_expiry: Duration,
       pub refresh_token_expiry: Duration,
       pub issuer: String,
       pub audience: String,
       pub algorithm: Algorithm,
   }
   
   impl AuthenticationService {
       pub fn new(
           secret: &[u8],
           user_store: Arc<dyn UserStore>,
           session_store: Arc<dyn SessionStore>,
           config: TokenConfig,
       ) -> Self {
           let encoding_key = EncodingKey::from_secret(secret);
           let decoding_key = DecodingKey::from_secret(secret);
           
           let mut validation = Validation::new(config.algorithm);
           validation.set_issuer(&[config.issuer.clone()]);
           validation.set_audience(&[config.audience.clone()]);
           
           Self {
               encoding_key,
               decoding_key,
               validation,
               user_store,
               session_store,
               password_hasher: Argon2::default(),
               token_config: config,
           }
       }
       
       pub async fn authenticate(&self, credentials: &LoginCredentials) -> Result<AuthenticationResult, AuthError> {
           // Validate user credentials
           let user = self.user_store.find_by_username(&credentials.username).await?
               .ok_or(AuthError::InvalidCredentials)?;
           
           // Verify password
           let parsed_hash = PasswordHash::new(&user.password_hash)
               .map_err(|_| AuthError::InvalidCredentials)?;
           
           self.password_hasher
               .verify_password(credentials.password.as_bytes(), &parsed_hash)
               .map_err(|_| AuthError::InvalidCredentials)?;
           
           // Check if account is active
           if !user.is_active {
               return Err(AuthError::AccountDisabled);
           }
           
           // Generate access and refresh tokens
           let access_token = self.generate_access_token(&user).await?;
           let refresh_token = self.generate_refresh_token(&user).await?;
           
           // Create session
           let session = UserSession {
               id: Uuid::new_v4(),
               user_id: user.id,
               access_token_jti: access_token.jti.clone(),
               refresh_token_jti: refresh_token.jti.clone(),
               created_at: Utc::now(),
               last_accessed: Utc::now(),
               ip_address: credentials.ip_address.clone(),
               user_agent: credentials.user_agent.clone(),
           };
           
           self.session_store.create_session(session).await?;
           
           Ok(AuthenticationResult {
               access_token: access_token.token,
               refresh_token: refresh_token.token,
               expires_in: self.token_config.access_token_expiry.num_seconds(),
               user_info: UserInfo {
                   id: user.id,
                   username: user.username,
                   roles: user.roles,
                   permissions: user.permissions,
               },
           })
       }
       
       async fn generate_access_token(&self, user: &User) -> Result<TokenData, AuthError> {
           let now = Utc::now();
           let jti = Uuid::new_v4().to_string();
           
           let claims = Claims {
               sub: user.id.to_string(),
               username: user.username.clone(),
               roles: user.roles.clone(),
               permissions: user.permissions.clone(),
               exp: (now + self.token_config.access_token_expiry).timestamp(),
               iat: now.timestamp(),
               jti: jti.clone(),
               scope: vec!["graph_algorithms".to_string(), "neuromorphic_access".to_string()],
           };
           
           let header = Header::new(self.token_config.algorithm);
           let token = encode(&header, &claims, &self.encoding_key)
               .map_err(AuthError::TokenGenerationError)?;
           
           Ok(TokenData { token, jti })
       }
   }
   ```

3. Implement role-based access control (RBAC):
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct Role {
       pub id: Uuid,
       pub name: String,
       pub description: String,
       pub permissions: Vec<Permission>,
       pub inherit_from: Vec<Uuid>, // Role inheritance
       pub created_at: DateTime<Utc>,
       pub updated_at: DateTime<Utc>,
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct Permission {
       pub resource: String,       // e.g., "graph_algorithms", "cortical_columns"
       pub action: String,         // e.g., "read", "write", "execute", "admin"
       pub conditions: Vec<PermissionCondition>, // Attribute-based conditions
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct PermissionCondition {
       pub attribute: String,      // e.g., "algorithm_type", "data_sensitivity"
       pub operator: ConditionOperator, // eq, ne, in, contains, etc.
       pub value: serde_json::Value,
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub enum ConditionOperator {
       Equal,
       NotEqual,
       In,
       NotIn,
       Contains,
       GreaterThan,
       LessThan,
       Regex,
   }
   
   pub struct AuthorizationService {
       role_store: Arc<dyn RoleStore>,
       permission_cache: Arc<Mutex<lru::LruCache<String, Vec<Permission>>>>,
       policy_engine: PolicyEngine,
   }
   
   impl AuthorizationService {
       pub async fn check_permission(
           &self,
           user: &User,
           resource: &str,
           action: &str,
           context: &AccessContext,
       ) -> Result<bool, AuthError> {
           // Get user's effective permissions (including role inheritance)
           let permissions = self.get_effective_permissions(user).await?;
           
           // Check if user has the required permission
           for permission in &permissions {
               if permission.resource == resource && permission.action == action {
                   // Evaluate conditions if any
                   if self.evaluate_conditions(&permission.conditions, context).await? {
                       return Ok(true);
                   }
               }
           }
           
           // Check for wildcard permissions
           for permission in &permissions {
               if (permission.resource == "*" || permission.resource == resource) &&
                  (permission.action == "*" || permission.action == action) {
                   if self.evaluate_conditions(&permission.conditions, context).await? {
                       return Ok(true);
                   }
               }
           }
           
           Ok(false)
       }
       
       async fn get_effective_permissions(&self, user: &User) -> Result<Vec<Permission>, AuthError> {
           let cache_key = format!("user_permissions_{}", user.id);
           
           // Check cache first
           {
               let mut cache = self.permission_cache.lock().await;
               if let Some(cached_permissions) = cache.get(&cache_key) {
                   return Ok(cached_permissions.clone());
               }
           }
           
           let mut effective_permissions = Vec::new();
           
           // Direct user permissions
           effective_permissions.extend(user.permissions.iter().cloned());
           
           // Role-based permissions (with inheritance)
           for role_id in &user.roles {
               let role_permissions = self.get_role_permissions_recursive(*role_id).await?;
               effective_permissions.extend(role_permissions);
           }
           
           // Remove duplicates and cache result
           effective_permissions.sort_by(|a, b| {
               a.resource.cmp(&b.resource).then(a.action.cmp(&b.action))
           });
           effective_permissions.dedup_by(|a, b| {
               a.resource == b.resource && a.action == b.action
           });
           
           // Cache the result
           {
               let mut cache = self.permission_cache.lock().await;
               cache.put(cache_key, effective_permissions.clone());
           }
           
           Ok(effective_permissions)
       }
       
       async fn evaluate_conditions(
           &self,
           conditions: &[PermissionCondition],
           context: &AccessContext,
       ) -> Result<bool, AuthError> {
           if conditions.is_empty() {
               return Ok(true);
           }
           
           for condition in conditions {
               if !self.evaluate_single_condition(condition, context).await? {
                   return Ok(false);
               }
           }
           
           Ok(true)
       }
       
       async fn evaluate_single_condition(
           &self,
           condition: &PermissionCondition,
           context: &AccessContext,
       ) -> Result<bool, AuthError> {
           let context_value = context.get_attribute(&condition.attribute);
           
           match condition.operator {
               ConditionOperator::Equal => Ok(context_value == Some(&condition.value)),
               ConditionOperator::NotEqual => Ok(context_value != Some(&condition.value)),
               ConditionOperator::In => {
                   if let Some(value) = context_value {
                       if let serde_json::Value::Array(allowed_values) = &condition.value {
                           Ok(allowed_values.contains(value))
                       } else {
                           Ok(false)
                       }
                   } else {
                       Ok(false)
                   }
               }
               ConditionOperator::Contains => {
                   if let (Some(serde_json::Value::String(context_str)), serde_json::Value::String(search_str)) =
                       (context_value, &condition.value) {
                       Ok(context_str.contains(search_str))
                   } else {
                       Ok(false)
                   }
               }
               // Add other operators as needed
               _ => Ok(false),
           }
       }
   }
   ```

4. Implement API key and service-to-service authentication:
   ```rust
   #[derive(Debug, Clone)]
   pub struct ApiKey {
       pub id: Uuid,
       pub key_hash: String,
       pub name: String,
       pub owner_id: Uuid,
       pub scopes: Vec<String>,
       pub rate_limit: Option<RateLimit>,
       pub expires_at: Option<DateTime<Utc>>,
       pub last_used: Option<DateTime<Utc>>,
       pub is_active: bool,
       pub created_at: DateTime<Utc>,
   }
   
   #[derive(Debug, Clone)]
   pub struct RateLimit {
       pub requests_per_minute: u32,
       pub requests_per_hour: u32,
       pub requests_per_day: u32,
   }
   
   pub struct ApiKeyService {
       key_store: Arc<dyn ApiKeyStore>,
       rate_limiter: Arc<dyn RateLimiter>,
       hasher: Argon2<'static>,
   }
   
   impl ApiKeyService {
       pub async fn create_api_key(
           &self,
           owner_id: Uuid,
           name: String,
           scopes: Vec<String>,
           expires_in: Option<Duration>,
       ) -> Result<(String, ApiKey), AuthError> {
           // Generate secure random key
           let raw_key = self.generate_secure_key();
           let key_hash = self.hash_api_key(&raw_key)?;
           
           let api_key = ApiKey {
               id: Uuid::new_v4(),
               key_hash,
               name,
               owner_id,
               scopes,
               rate_limit: Some(RateLimit {
                   requests_per_minute: 100,
                   requests_per_hour: 1000,
                   requests_per_day: 10000,
               }),
               expires_at: expires_in.map(|d| Utc::now() + d),
               last_used: None,
               is_active: true,
               created_at: Utc::now(),
           };
           
           self.key_store.store_api_key(&api_key).await?;
           
           Ok((raw_key, api_key))
       }
       
       pub async fn validate_api_key(&self, raw_key: &str) -> Result<ApiKey, AuthError> {
           let key_hash = self.hash_api_key(raw_key)?;
           
           let api_key = self.key_store.find_by_hash(&key_hash).await?
               .ok_or(AuthError::InvalidApiKey)?;
           
           // Check if key is active
           if !api_key.is_active {
               return Err(AuthError::ApiKeyDisabled);
           }
           
           // Check if key is expired
           if let Some(expires_at) = api_key.expires_at {
               if Utc::now() > expires_at {
                   return Err(AuthError::ApiKeyExpired);
               }
           }
           
           // Check rate limits
           if let Some(rate_limit) = &api_key.rate_limit {
               self.rate_limiter.check_rate_limit(&api_key.id, rate_limit).await?;
           }
           
           // Update last used timestamp
           self.key_store.update_last_used(&api_key.id, Utc::now()).await?;
           
           Ok(api_key)
       }
       
       fn generate_secure_key(&self) -> String {
           use rand::Rng;
           const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
           const KEY_LENGTH: usize = 32;
           
           let mut rng = rand::thread_rng();
           let key: String = (0..KEY_LENGTH)
               .map(|_| {
                   let idx = rng.gen_range(0..CHARSET.len());
                   CHARSET[idx] as char
               })
               .collect();
           
           format!("nk_{}", key) // "nk" prefix for "neuromorphic key"
       }
   }
   ```

5. Implement middleware for request authentication:
   ```rust
   use axum::{
       extract::{Request, State},
       http::{HeaderMap, StatusCode},
       middleware::Next,
       response::Response,
   };
   
   pub struct AuthMiddleware {
       auth_service: Arc<AuthenticationService>,
       api_key_service: Arc<ApiKeyService>,
       authorization_service: Arc<AuthorizationService>,
   }
   
   impl AuthMiddleware {
       pub async fn authenticate_request(
           State(auth): State<Arc<AuthMiddleware>>,
           mut request: Request,
           next: Next,
       ) -> Result<Response, StatusCode> {
           let headers = request.headers();
           
           // Try different authentication methods
           let auth_context = if let Some(auth_result) = auth.try_bearer_token(headers).await {
               auth_result?
           } else if let Some(auth_result) = auth.try_api_key(headers).await {
               auth_result?
           } else {
               return Err(StatusCode::UNAUTHORIZED);
           };
           
           // Add authentication context to request extensions
           request.extensions_mut().insert(auth_context);
           
           Ok(next.run(request).await)
       }
       
       async fn try_bearer_token(&self, headers: &HeaderMap) -> Option<Result<AuthContext, StatusCode>> {
           let auth_header = headers.get("authorization")?;
           let auth_str = auth_header.to_str().ok()?;
           
           if !auth_str.starts_with("Bearer ") {
               return None;
           }
           
           let token = &auth_str[7..];
           
           Some(match self.auth_service.validate_token(token).await {
               Ok(claims) => Ok(AuthContext::User(claims)),
               Err(AuthError::TokenExpired) => Err(StatusCode::UNAUTHORIZED),
               Err(AuthError::InvalidToken) => Err(StatusCode::UNAUTHORIZED),
               Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
           })
       }
       
       async fn try_api_key(&self, headers: &HeaderMap) -> Option<Result<AuthContext, StatusCode>> {
           let api_key = headers.get("x-api-key")?.to_str().ok()?;
           
           Some(match self.api_key_service.validate_api_key(api_key).await {
               Ok(api_key) => Ok(AuthContext::ApiKey(api_key)),
               Err(AuthError::InvalidApiKey) => Err(StatusCode::UNAUTHORIZED),
               Err(AuthError::ApiKeyExpired) => Err(StatusCode::UNAUTHORIZED),
               Err(AuthError::RateLimitExceeded) => Err(StatusCode::TOO_MANY_REQUESTS),
               Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
           })
       }
   }
   
   #[derive(Debug, Clone)]
   pub enum AuthContext {
       User(Claims),
       ApiKey(ApiKey),
   }
   
   impl AuthContext {
       pub fn user_id(&self) -> Option<Uuid> {
           match self {
               AuthContext::User(claims) => claims.sub.parse().ok(),
               AuthContext::ApiKey(api_key) => Some(api_key.owner_id),
           }
       }
       
       pub fn has_scope(&self, scope: &str) -> bool {
           match self {
               AuthContext::User(claims) => claims.scope.contains(&scope.to_string()),
               AuthContext::ApiKey(api_key) => api_key.scopes.contains(&scope.to_string()),
           }
       }
   }
   ```

## Expected Output
```rust
pub trait Authentication {
    async fn authenticate(&self, credentials: &LoginCredentials) -> Result<AuthenticationResult, AuthError>;
    async fn validate_token(&self, token: &str) -> Result<Claims, AuthError>;
    async fn refresh_token(&self, refresh_token: &str) -> Result<TokenPair, AuthError>;
    async fn revoke_token(&self, token_jti: &str) -> Result<(), AuthError>;
}

pub trait Authorization {
    async fn check_permission(&self, user: &User, resource: &str, action: &str, context: &AccessContext) -> Result<bool, AuthError>;
    async fn get_user_permissions(&self, user: &User) -> Result<Vec<Permission>, AuthError>;
    async fn assign_role(&self, user_id: Uuid, role_id: Uuid) -> Result<(), AuthError>;
}

#[derive(Debug)]
pub enum AuthError {
    InvalidCredentials,
    TokenExpired,
    InvalidToken,
    AccountDisabled,
    InsufficientPermissions,
    ApiKeyExpired,
    RateLimitExceeded,
    DatabaseError(String),
}
```

## Verification Steps
1. Test JWT token generation and validation
2. Verify role inheritance and permission aggregation
3. Test API key authentication and rate limiting
4. Validate authorization policies under various contexts
5. Test session management and token revocation
6. Benchmark authentication performance under load

## Time Estimate
25 minutes

## Dependencies
- MP001-MP050: Previous implementations
- jsonwebtoken: JWT handling
- argon2: Password hashing
- axum: Web framework middleware
- uuid: Unique identifiers
- serde: Serialization