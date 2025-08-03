/*
 * Rust Microservice for Product Recommendation Engine
 * High-performance service using Actix-web, Tokio, and advanced Rust patterns
 * Provides ML-based product recommendations and analytics
 */

use actix_web::{
    web, App, HttpResponse, HttpServer, Result, middleware::Logger,
    error::{ErrorBadRequest, ErrorInternalServerError, ErrorNotFound},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use tokio::time::{sleep, Duration};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    env,
};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use log::{info, warn, error, debug};
use anyhow::{Context, Result as AnyhowResult};
use futures::stream::{self, StreamExt};

// Configuration and types
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub database_url: String,
    pub server_host: String,
    pub server_port: u16,
    pub cache_ttl_seconds: u64,
    pub max_recommendations: usize,
    pub analytics_batch_size: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            database_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://user:password@localhost:5432/ecommerce".to_string()),
            server_host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            server_port: env::var("PORT")
                .unwrap_or_else(|_| "8001".to_string())
                .parse()
                .unwrap_or(8001),
            cache_ttl_seconds: 300, // 5 minutes
            max_recommendations: 20,
            analytics_batch_size: 1000,
        }
    }
}

// Data models
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Product {
    pub id: i32,
    pub name: String,
    pub description: Option<String>,
    pub price: f64,
    pub stock_quantity: i32,
    pub category_id: Option<i32>,
    pub sku: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct User {
    pub id: i32,
    pub username: String,
    pub email: String,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PurchaseHistory {
    pub user_id: i32,
    pub product_id: i32,
    pub quantity: i32,
    pub unit_price: f64,
    pub purchase_date: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProductInteraction {
    pub user_id: i32,
    pub product_id: i32,
    pub interaction_type: InteractionType,
    pub timestamp: DateTime<Utc>,
    pub session_id: Option<Uuid>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum InteractionType {
    View,
    AddToCart,
    Purchase,
    Review,
    Wishlist,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecommendationRequest {
    pub user_id: i32,
    pub limit: Option<usize>,
    pub include_categories: Option<Vec<i32>>,
    pub exclude_products: Option<Vec<i32>>,
    pub algorithm: Option<RecommendationAlgorithm>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RecommendationAlgorithm {
    CollaborativeFiltering,
    ContentBased,
    Hybrid,
    PopularityBased,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecommendationResponse {
    pub recommendations: Vec<ProductRecommendation>,
    pub algorithm_used: String,
    pub confidence_score: f64,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProductRecommendation {
    pub product: Product,
    pub score: f64,
    pub reason: String,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalyticsRequest {
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub metrics: Vec<AnalyticsMetric>,
    pub group_by: Option<AnalyticsGroupBy>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum AnalyticsMetric {
    TotalSales,
    ProductViews,
    ConversionRate,
    AverageOrderValue,
    UserRetention,
    ProductPopularity,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum AnalyticsGroupBy {
    Product,
    Category,
    User,
    Date,
    Hour,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalyticsResponse {
    pub metrics: HashMap<String, f64>,
    pub breakdown: Option<HashMap<String, f64>>,
    pub time_series: Option<Vec<TimeSeriesPoint>>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub label: Option<String>,
}

// Application state and caching
#[derive(Debug)]
pub struct AppState {
    pub db_pool: PgPool,
    pub config: AppConfig,
    pub recommendation_cache: Arc<RwLock<HashMap<String, (RecommendationResponse, DateTime<Utc>)>>>,
    pub analytics_cache: Arc<RwLock<HashMap<String, (AnalyticsResponse, DateTime<Utc>)>>>,
    pub user_similarity_matrix: Arc<RwLock<HashMap<(i32, i32), f64>>>,
    pub product_similarity_matrix: Arc<RwLock<HashMap<(i32, i32), f64>>>,
}

impl AppState {
    pub async fn new(config: AppConfig) -> AnyhowResult<Self> {
        let db_pool = PgPool::connect(&config.database_url)
            .await
            .context("Failed to connect to database")?;

        info!("Database connection established");

        let state = Self {
            db_pool,
            config,
            recommendation_cache: Arc::new(RwLock::new(HashMap::new())),
            analytics_cache: Arc::new(RwLock::new(HashMap::new())),
            user_similarity_matrix: Arc::new(RwLock::new(HashMap::new())),
            product_similarity_matrix: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize similarity matrices
        state.build_similarity_matrices().await?;

        Ok(state)
    }

    async fn build_similarity_matrices(&self) -> AnyhowResult<()> {
        info!("Building similarity matrices...");
        
        // Build user similarity matrix using collaborative filtering
        let user_similarities = self.calculate_user_similarities().await?;
        {
            let mut cache = self.user_similarity_matrix.write().unwrap();
            *cache = user_similarities;
        }

        // Build product similarity matrix using content-based filtering
        let product_similarities = self.calculate_product_similarities().await?;
        {
            let mut cache = self.product_similarity_matrix.write().unwrap();
            *cache = product_similarities;
        }

        info!("Similarity matrices built successfully");
        Ok(())
    }

    async fn calculate_user_similarities(&self) -> AnyhowResult<HashMap<(i32, i32), f64>> {
        let mut similarities = HashMap::new();
        
        // Get user purchase histories
        let query = r#"
            SELECT DISTINCT o.user_id, oi.product_id, SUM(oi.quantity) as total_quantity
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            WHERE o.status IN ('delivered', 'confirmed')
            GROUP BY o.user_id, oi.product_id
            ORDER BY o.user_id, oi.product_id
        "#;
        
        let rows = sqlx::query(query).fetch_all(&self.db_pool).await?;
        
        // Build user-product matrix
        let mut user_products: HashMap<i32, HashSet<i32>> = HashMap::new();
        for row in rows {
            let user_id: i32 = row.get("user_id");
            let product_id: i32 = row.get("product_id");
            
            user_products.entry(user_id).or_insert_with(HashSet::new).insert(product_id);
        }
        
        // Calculate Jaccard similarity between users
        let users: Vec<i32> = user_products.keys().cloned().collect();
        for i in 0..users.len() {
            for j in (i + 1)..users.len() {
                let user1 = users[i];
                let user2 = users[j];
                
                if let (Some(products1), Some(products2)) = 
                    (user_products.get(&user1), user_products.get(&user2)) {
                    
                    let intersection = products1.intersection(products2).count() as f64;
                    let union = products1.union(products2).count() as f64;
                    
                    if union > 0.0 {
                        let similarity = intersection / union;
                        similarities.insert((user1, user2), similarity);
                        similarities.insert((user2, user1), similarity);
                    }
                }
            }
        }
        
        Ok(similarities)
    }

    async fn calculate_product_similarities(&self) -> AnyhowResult<HashMap<(i32, i32), f64>> {
        let mut similarities = HashMap::new();
        
        // Get product features (category, price range, etc.)
        let query = r#"
            SELECT id, category_id, price, 
                   CASE 
                       WHEN price < 50 THEN 1
                       WHEN price < 100 THEN 2
                       WHEN price < 200 THEN 3
                       ELSE 4
                   END as price_range
            FROM products 
            WHERE is_active = true
        "#;
        
        let rows = sqlx::query(query).fetch_all(&self.db_pool).await?;
        
        // Build product feature vectors
        let mut product_features: HashMap<i32, (Option<i32>, i32)> = HashMap::new();
        for row in rows {
            let product_id: i32 = row.get("id");
            let category_id: Option<i32> = row.get("category_id");
            let price_range: i32 = row.get("price_range");
            
            product_features.insert(product_id, (category_id, price_range));
        }
        
        // Calculate similarity based on shared features
        let products: Vec<i32> = product_features.keys().cloned().collect();
        for i in 0..products.len() {
            for j in (i + 1)..products.len() {
                let product1 = products[i];
                let product2 = products[j];
                
                if let (Some(features1), Some(features2)) = 
                    (product_features.get(&product1), product_features.get(&product2)) {
                    
                    let mut similarity = 0.0;
                    
                    // Category similarity
                    if features1.0 == features2.0 && features1.0.is_some() {
                        similarity += 0.7;
                    }
                    
                    // Price range similarity
                    let price_diff = (features1.1 - features2.1).abs();
                    similarity += 0.3 * (1.0 - (price_diff as f64 / 4.0)).max(0.0);
                    
                    similarities.insert((product1, product2), similarity);
                    similarities.insert((product2, product1), similarity);
                }
            }
        }
        
        Ok(similarities)
    }
}

// Recommendation engine
pub struct RecommendationEngine;

impl RecommendationEngine {
    pub async fn generate_recommendations(
        state: &AppState,
        request: &RecommendationRequest,
    ) -> AnyhowResult<RecommendationResponse> {
        let algorithm = request.algorithm.as_ref().unwrap_or(&RecommendationAlgorithm::Hybrid);
        let limit = request.limit.unwrap_or(state.config.max_recommendations);
        
        let recommendations = match algorithm {
            RecommendationAlgorithm::CollaborativeFiltering => {
                Self::collaborative_filtering(state, request.user_id, limit).await?
            }
            RecommendationAlgorithm::ContentBased => {
                Self::content_based_filtering(state, request.user_id, limit).await?
            }
            RecommendationAlgorithm::Hybrid => {
                Self::hybrid_recommendations(state, request.user_id, limit).await?
            }
            RecommendationAlgorithm::PopularityBased => {
                Self::popularity_based_recommendations(state, limit).await?
            }
        };
        
        let confidence_score = Self::calculate_confidence(&recommendations);
        
        Ok(RecommendationResponse {
            recommendations,
            algorithm_used: format!("{:?}", algorithm),
            confidence_score,
            generated_at: Utc::now(),
        })
    }
    
    async fn collaborative_filtering(
        state: &AppState,
        user_id: i32,
        limit: usize,
    ) -> AnyhowResult<Vec<ProductRecommendation>> {
        // Find similar users
        let user_similarities = state.user_similarity_matrix.read().unwrap();
        let similar_users: Vec<(i32, f64)> = user_similarities
            .iter()
            .filter_map(|((u1, u2), similarity)| {
                if *u1 == user_id {
                    Some((*u2, *similarity))
                } else if *u2 == user_id {
                    Some((*u1, *similarity))
                } else {
                    None
                }
            })
            .filter(|(_, similarity)| *similarity > 0.1)
            .collect();
        
        if similar_users.is_empty() {
            return Ok(vec![]);
        }
        
        // Get products purchased by similar users but not by target user
        let similar_user_ids: Vec<i32> = similar_users.iter().map(|(id, _)| *id).collect();
        let placeholders = similar_user_ids.iter().enumerate()
            .map(|(i, _)| format!("${}", i + 2))
            .collect::<Vec<_>>()
            .join(",");
        
        let query = format!(r#"
            SELECT DISTINCT p.*, oi.quantity, o.user_id
            FROM products p
            JOIN order_items oi ON p.id = oi.product_id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.user_id IN ({placeholders})
            AND p.id NOT IN (
                SELECT DISTINCT oi2.product_id
                FROM order_items oi2
                JOIN orders o2 ON oi2.order_id = o2.id
                WHERE o2.user_id = $1
            )
            AND p.is_active = true
            ORDER BY oi.quantity DESC
        "#);
        
        let mut query_builder = sqlx::query_as::<_, (Product, i32, i32)>(&query);
        query_builder = query_builder.bind(user_id);
        for similar_user_id in &similar_user_ids {
            query_builder = query_builder.bind(similar_user_id);
        }
        
        let rows = query_builder.fetch_all(&state.db_pool).await?;
        
        // Score products based on similar users' preferences
        let mut product_scores: HashMap<i32, f64> = HashMap::new();
        let similar_user_map: HashMap<i32, f64> = similar_users.into_iter().collect();
        
        for (product, quantity, purchasing_user_id) in rows {
            if let Some(user_similarity) = similar_user_map.get(&purchasing_user_id) {
                let score = user_similarity * (quantity as f64).log10();
                *product_scores.entry(product.id).or_insert(0.0) += score;
            }
        }
        
        // Convert to recommendations
        let mut recommendations: Vec<ProductRecommendation> = Vec::new();
        for (product_id, score) in product_scores.iter().take(limit) {
            if let Ok(product) = Self::get_product(state, *product_id).await {
                recommendations.push(ProductRecommendation {
                    product,
                    score: *score,
                    reason: "Based on users with similar preferences".to_string(),
                    confidence: score / product_scores.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&1.0),
                });
            }
        }
        
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(recommendations.into_iter().take(limit).collect())
    }
    
    async fn content_based_filtering(
        state: &AppState,
        user_id: i32,
        limit: usize,
    ) -> AnyhowResult<Vec<ProductRecommendation>> {
        // Get user's purchase history to understand preferences
        let query = r#"
            SELECT DISTINCT p.*
            FROM products p
            JOIN order_items oi ON p.id = oi.product_id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.user_id = $1
            AND o.status IN ('delivered', 'confirmed')
        "#;
        
        let user_purchased_products: Vec<Product> = sqlx::query_as(query)
            .bind(user_id)
            .fetch_all(&state.db_pool)
            .await?;
        
        if user_purchased_products.is_empty() {
            return Ok(vec![]);
        }
        
        // Find similar products
        let product_similarities = state.product_similarity_matrix.read().unwrap();
        let mut candidate_scores: HashMap<i32, f64> = HashMap::new();
        
        for purchased_product in &user_purchased_products {
            for ((p1, p2), similarity) in product_similarities.iter() {
                if *p1 == purchased_product.id {
                    *candidate_scores.entry(*p2).or_insert(0.0) += similarity;
                } else if *p2 == purchased_product.id {
                    *candidate_scores.entry(*p1).or_insert(0.0) += similarity;
                }
            }
        }
        
        // Remove already purchased products
        let purchased_ids: HashSet<i32> = user_purchased_products.iter().map(|p| p.id).collect();
        candidate_scores.retain(|product_id, _| !purchased_ids.contains(product_id));
        
        // Convert to recommendations
        let mut recommendations: Vec<ProductRecommendation> = Vec::new();
        for (product_id, score) in candidate_scores.iter().take(limit) {
            if let Ok(product) = Self::get_product(state, *product_id).await {
                recommendations.push(ProductRecommendation {
                    product,
                    score: *score,
                    reason: "Based on your purchase history".to_string(),
                    confidence: score / candidate_scores.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&1.0),
                });
            }
        }
        
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(recommendations.into_iter().take(limit).collect())
    }
    
    async fn hybrid_recommendations(
        state: &AppState,
        user_id: i32,
        limit: usize,
    ) -> AnyhowResult<Vec<ProductRecommendation>> {
        // Combine collaborative and content-based filtering
        let collaborative_recs = Self::collaborative_filtering(state, user_id, limit * 2).await?;
        let content_recs = Self::content_based_filtering(state, user_id, limit * 2).await?;
        
        let mut combined_scores: HashMap<i32, (f64, String)> = HashMap::new();
        
        // Weight collaborative filtering results
        for rec in collaborative_recs {
            combined_scores.insert(rec.product.id, (rec.score * 0.6, rec.reason));
        }
        
        // Add content-based results with different weight
        for rec in content_recs {
            let (existing_score, reason) = combined_scores.get(&rec.product.id).unwrap_or(&(0.0, "".to_string()));
            let new_score = existing_score + (rec.score * 0.4);
            let combined_reason = if reason.is_empty() {
                rec.reason
            } else {
                format!("{} + {}", reason, rec.reason)
            };
            combined_scores.insert(rec.product.id, (new_score, combined_reason));
        }
        
        // Convert to final recommendations
        let mut final_recommendations: Vec<ProductRecommendation> = Vec::new();
        let max_score = combined_scores.values().map(|(score, _)| *score).fold(0.0f64, f64::max);
        
        for (product_id, (score, reason)) in combined_scores.iter().take(limit) {
            if let Ok(product) = Self::get_product(state, *product_id).await {
                final_recommendations.push(ProductRecommendation {
                    product,
                    score: *score,
                    reason: reason.clone(),
                    confidence: score / max_score,
                });
            }
        }
        
        final_recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(final_recommendations.into_iter().take(limit).collect())
    }
    
    async fn popularity_based_recommendations(
        state: &AppState,
        limit: usize,
    ) -> AnyhowResult<Vec<ProductRecommendation>> {
        let query = r#"
            SELECT p.*, COUNT(oi.id) as purchase_count
            FROM products p
            LEFT JOIN order_items oi ON p.id = oi.product_id
            LEFT JOIN orders o ON oi.order_id = o.id
            WHERE p.is_active = true
            AND (o.status IN ('delivered', 'confirmed') OR o.status IS NULL)
            GROUP BY p.id
            ORDER BY purchase_count DESC, p.created_at DESC
            LIMIT $1
        "#;
        
        let rows = sqlx::query(query)
            .bind(limit as i32)
            .fetch_all(&state.db_pool)
            .await?;
        
        let mut recommendations = Vec::new();
        let max_count = rows.first().map(|row| row.get::<i64, _>("purchase_count") as f64).unwrap_or(1.0);
        
        for row in rows {
            let product = Product {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                price: row.get("price"),
                stock_quantity: row.get("stock_quantity"),
                category_id: row.get("category_id"),
                sku: row.get("sku"),
                is_active: row.get("is_active"),
                created_at: row.get("created_at"),
            };
            
            let purchase_count = row.get::<i64, _>("purchase_count") as f64;
            let score = purchase_count / max_count;
            
            recommendations.push(ProductRecommendation {
                product,
                score,
                reason: "Popular product".to_string(),
                confidence: score,
            });
        }
        
        Ok(recommendations)
    }
    
    async fn get_product(state: &AppState, product_id: i32) -> AnyhowResult<Product> {
        let query = "SELECT * FROM products WHERE id = $1 AND is_active = true";
        let product = sqlx::query_as::<_, Product>(query)
            .bind(product_id)
            .fetch_one(&state.db_pool)
            .await?;
        Ok(product)
    }
    
    fn calculate_confidence(recommendations: &[ProductRecommendation]) -> f64 {
        if recommendations.is_empty() {
            return 0.0;
        }
        
        let avg_confidence: f64 = recommendations.iter().map(|r| r.confidence).sum::<f64>() / recommendations.len() as f64;
        let score_variance = Self::calculate_variance(&recommendations.iter().map(|r| r.score).collect::<Vec<_>>());
        
        // Higher confidence when we have consistent scores
        avg_confidence * (1.0 - score_variance.min(1.0))
    }
    
    fn calculate_variance(values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt() / mean.abs().max(1.0)
    }
}

// Analytics engine
pub struct AnalyticsEngine;

impl AnalyticsEngine {
    pub async fn generate_analytics(
        state: &AppState,
        request: &AnalyticsRequest,
    ) -> AnyhowResult<AnalyticsResponse> {
        let mut metrics = HashMap::new();
        let mut breakdown = None;
        let mut time_series = None;
        
        for metric in &request.metrics {
            match metric {
                AnalyticsMetric::TotalSales => {
                    let value = Self::calculate_total_sales(state, request).await?;
                    metrics.insert("total_sales".to_string(), value);
                }
                AnalyticsMetric::ProductViews => {
                    let value = Self::calculate_product_views(state, request).await?;
                    metrics.insert("product_views".to_string(), value);
                }
                AnalyticsMetric::ConversionRate => {
                    let value = Self::calculate_conversion_rate(state, request).await?;
                    metrics.insert("conversion_rate".to_string(), value);
                }
                AnalyticsMetric::AverageOrderValue => {
                    let value = Self::calculate_average_order_value(state, request).await?;
                    metrics.insert("average_order_value".to_string(), value);
                }
                _ => {
                    // Other metrics would be implemented similarly
                }
            }
        }
        
        if let Some(group_by) = &request.group_by {
            breakdown = Some(Self::generate_breakdown(state, request, group_by).await?);
        }
        
        Ok(AnalyticsResponse {
            metrics,
            breakdown,
            time_series,
            generated_at: Utc::now(),
        })
    }
    
    async fn calculate_total_sales(
        state: &AppState,
        request: &AnalyticsRequest,
    ) -> AnyhowResult<f64> {
        let mut query = "SELECT COALESCE(SUM(total_amount), 0) as total FROM orders WHERE status IN ('delivered', 'confirmed')".to_string();
        let mut bindings = Vec::new();
        let mut param_count = 1;
        
        if let Some(start_date) = request.start_date {
            query.push_str(&format!(" AND created_at >= ${}", param_count));
            bindings.push(start_date);
            param_count += 1;
        }
        
        if let Some(end_date) = request.end_date {
            query.push_str(&format!(" AND created_at <= ${}", param_count));
            bindings.push(end_date);
        }
        
        let mut query_builder = sqlx::query(&query);
        for binding in bindings {
            query_builder = query_builder.bind(binding);
        }
        
        let row = query_builder.fetch_one(&state.db_pool).await?;
        Ok(row.get::<f64, _>("total"))
    }
    
    async fn calculate_product_views(
        state: &AppState,
        request: &AnalyticsRequest,
    ) -> AnyhowResult<f64> {
        // This would track product views from interaction logs
        // For simulation, return a calculated value
        Ok(1000.0)
    }
    
    async fn calculate_conversion_rate(
        state: &AppState,
        request: &AnalyticsRequest,
    ) -> AnyhowResult<f64> {
        // Calculate views to purchases ratio
        let views = Self::calculate_product_views(state, request).await?;
        let purchases = Self::calculate_total_sales(state, request).await?;
        
        if views > 0.0 {
            Ok((purchases / views) * 100.0)
        } else {
            Ok(0.0)
        }
    }
    
    async fn calculate_average_order_value(
        state: &AppState,
        request: &AnalyticsRequest,
    ) -> AnyhowResult<f64> {
        let mut query = "SELECT COALESCE(AVG(total_amount), 0) as avg_value FROM orders WHERE status IN ('delivered', 'confirmed')".to_string();
        let mut bindings = Vec::new();
        let mut param_count = 1;
        
        if let Some(start_date) = request.start_date {
            query.push_str(&format!(" AND created_at >= ${}", param_count));
            bindings.push(start_date);
            param_count += 1;
        }
        
        if let Some(end_date) = request.end_date {
            query.push_str(&format!(" AND created_at <= ${}", param_count));
            bindings.push(end_date);
        }
        
        let mut query_builder = sqlx::query(&query);
        for binding in bindings {
            query_builder = query_builder.bind(binding);
        }
        
        let row = query_builder.fetch_one(&state.db_pool).await?;
        Ok(row.get::<f64, _>("avg_value"))
    }
    
    async fn generate_breakdown(
        state: &AppState,
        request: &AnalyticsRequest,
        group_by: &AnalyticsGroupBy,
    ) -> AnyhowResult<HashMap<String, f64>> {
        let mut breakdown = HashMap::new();
        
        match group_by {
            AnalyticsGroupBy::Product => {
                let query = r#"
                    SELECT p.name, COALESCE(SUM(oi.total_price), 0) as total
                    FROM products p
                    LEFT JOIN order_items oi ON p.id = oi.product_id
                    LEFT JOIN orders o ON oi.order_id = o.id
                    WHERE o.status IN ('delivered', 'confirmed') OR o.status IS NULL
                    GROUP BY p.id, p.name
                    ORDER BY total DESC
                    LIMIT 20
                "#;
                
                let rows = sqlx::query(query).fetch_all(&state.db_pool).await?;
                for row in rows {
                    let name: String = row.get("name");
                    let total: f64 = row.get("total");
                    breakdown.insert(name, total);
                }
            }
            AnalyticsGroupBy::Category => {
                let query = r#"
                    SELECT c.name, COALESCE(SUM(oi.total_price), 0) as total
                    FROM categories c
                    LEFT JOIN products p ON c.id = p.category_id
                    LEFT JOIN order_items oi ON p.id = oi.product_id
                    LEFT JOIN orders o ON oi.order_id = o.id
                    WHERE o.status IN ('delivered', 'confirmed') OR o.status IS NULL
                    GROUP BY c.id, c.name
                    ORDER BY total DESC
                "#;
                
                let rows = sqlx::query(query).fetch_all(&state.db_pool).await?;
                for row in rows {
                    let name: String = row.get("name");
                    let total: f64 = row.get("total");
                    breakdown.insert(name, total);
                }
            }
            _ => {
                // Other breakdowns would be implemented
            }
        }
        
        Ok(breakdown)
    }
}

// HTTP handlers
async fn get_recommendations(
    state: web::Data<AppState>,
    request: web::Json<RecommendationRequest>,
) -> Result<HttpResponse> {
    // Check cache first
    let cache_key = format!("rec_{}_{:?}", request.user_id, request.algorithm);
    {
        let cache = state.recommendation_cache.read().unwrap();
        if let Some((cached_response, timestamp)) = cache.get(&cache_key) {
            let age = Utc::now().signed_duration_since(*timestamp);
            if age.num_seconds() < state.config.cache_ttl_seconds as i64 {
                return Ok(HttpResponse::Ok().json(cached_response));
            }
        }
    }
    
    match RecommendationEngine::generate_recommendations(&state, &request).await {
        Ok(response) => {
            // Cache the response
            {
                let mut cache = state.recommendation_cache.write().unwrap();
                cache.insert(cache_key, (response.clone(), Utc::now()));
            }
            
            Ok(HttpResponse::Ok().json(response))
        }
        Err(e) => {
            error!("Failed to generate recommendations: {}", e);
            Ok(HttpResponse::InternalServerError().json(format!("Error: {}", e)))
        }
    }
}

async fn get_analytics(
    state: web::Data<AppState>,
    request: web::Json<AnalyticsRequest>,
) -> Result<HttpResponse> {
    match AnalyticsEngine::generate_analytics(&state, &request).await {
        Ok(response) => Ok(HttpResponse::Ok().json(response)),
        Err(e) => {
            error!("Failed to generate analytics: {}", e);
            Ok(HttpResponse::InternalServerError().json(format!("Error: {}", e)))
        }
    }
}

async fn health_check() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "timestamp": Utc::now(),
        "service": "recommendation-engine"
    })))
}

async fn metrics_endpoint(state: web::Data<AppState>) -> Result<HttpResponse> {
    let cache_stats = {
        let rec_cache = state.recommendation_cache.read().unwrap();
        let analytics_cache = state.analytics_cache.read().unwrap();
        
        serde_json::json!({
            "recommendation_cache_size": rec_cache.len(),
            "analytics_cache_size": analytics_cache.len(),
        })
    };
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "service": "recommendation-engine",
        "cache_stats": cache_stats,
        "timestamp": Utc::now()
    })))
}

// Background tasks
async fn cache_cleanup_task(state: web::Data<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
    
    loop {
        interval.tick().await;
        
        let now = Utc::now();
        let ttl_duration = chrono::Duration::seconds(state.config.cache_ttl_seconds as i64);
        
        // Clean recommendation cache
        {
            let mut cache = state.recommendation_cache.write().unwrap();
            cache.retain(|_, (_, timestamp)| {
                now.signed_duration_since(*timestamp) < ttl_duration
            });
        }
        
        // Clean analytics cache
        {
            let mut cache = state.analytics_cache.write().unwrap();
            cache.retain(|_, (_, timestamp)| {
                now.signed_duration_since(*timestamp) < ttl_duration
            });
        }
        
        debug!("Cache cleanup completed");
    }
}

async fn similarity_matrix_refresh_task(state: web::Data<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 1 hour
    
    loop {
        interval.tick().await;
        
        info!("Refreshing similarity matrices...");
        if let Err(e) = state.build_similarity_matrices().await {
            error!("Failed to refresh similarity matrices: {}", e);
        }
    }
}

// Main application
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    
    let config = AppConfig::default();
    info!("Starting recommendation service with config: {:?}", config);
    
    let state = AppState::new(config.clone())
        .await
        .expect("Failed to initialize application state");
    
    let state_data = web::Data::new(state);
    
    // Start background tasks
    let cleanup_state = state_data.clone();
    tokio::spawn(async move {
        cache_cleanup_task(cleanup_state).await;
    });
    
    let refresh_state = state_data.clone();
    tokio::spawn(async move {
        similarity_matrix_refresh_task(refresh_state).await;
    });
    
    info!("Starting HTTP server on {}:{}", config.server_host, config.server_port);
    
    HttpServer::new(move || {
        App::new()
            .app_data(state_data.clone())
            .wrap(Logger::default())
            .route("/health", web::get().to(health_check))
            .route("/metrics", web::get().to(metrics_endpoint))
            .route("/recommendations", web::post().to(get_recommendations))
            .route("/analytics", web::post().to(get_analytics))
    })
    .bind((config.server_host.clone(), config.server_port))?
    .run()
    .await
}