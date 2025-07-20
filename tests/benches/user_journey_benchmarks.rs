/*!
Phase 5.3: User Journey Benchmarks
End-to-end benchmarks that simulate complete user workflows and application scenarios
*/

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// User session simulation
#[derive(Clone)]
struct UserSession {
    user_id: String,
    session_id: String,
    start_time: Instant,
    operations: Vec<UserOperation>,
}

#[derive(Clone)]
struct UserOperation {
    operation_type: OperationType,
    timestamp: Duration,
    metadata: HashMap<String, String>,
}

#[derive(Clone)]
enum OperationType {
    Search(String),
    Browse(String),
    Recommend,
    Add(String),
    Update(String),
    Delete(String),
}

impl UserSession {
    fn new(user_id: String) -> Self {
        Self {
            user_id,
            session_id: format!("session_{}", uuid::Uuid::new_v4()),
            start_time: Instant::now(),
            operations: Vec::new(),
        }
    }
    
    fn add_operation(&mut self, operation_type: OperationType, metadata: HashMap<String, String>) {
        let timestamp = self.start_time.elapsed();
        self.operations.push(UserOperation {
            operation_type,
            timestamp,
            metadata,
        });
    }
}

// Simulate a complete e-commerce customer journey
fn benchmark_ecommerce_customer_journey(c: &mut Criterion) {
    let mut group = c.benchmark_group("ecommerce_customer_journey");
    
    for catalog_size in [10_000, 50_000, 100_000] {
        group.throughput(Throughput::Elements(catalog_size as u64));
        
        group.bench_with_input(BenchmarkId::new("complete_shopping_experience", catalog_size), &catalog_size, |b, &catalog_size| {
            let temp_dir = TempDir::new().unwrap();
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(384, 6, 64).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            let mut rng = StdRng::seed_from_u64(42);
            
            // Setup product catalog
            let categories = ["electronics", "books", "clothing", "home", "sports", "beauty"];
            let brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"];
            let price_ranges = [(10.0, 50.0), (50.0, 200.0), (200.0, 1000.0)];
            
            for i in 0..*catalog_size {
                let category = categories[i % categories.len()];
                let brand = brands[i % brands.len()];
                let (min_price, max_price) = price_ranges[i % price_ranges.len()];
                let price = rng.gen_range(min_price..max_price);
                
                let product_name = format!("{} {} Product {}", brand, category, i);
                let description = format!("High-quality {} from {} with excellent features and competitive pricing at ${:.2}", 
                    category, brand, price);
                
                let mut metadata = HashMap::new();
                metadata.insert("category".to_string(), category.to_string());
                metadata.insert("brand".to_string(), brand.to_string());
                metadata.insert("price".to_string(), price.to_string());
                metadata.insert("rating".to_string(), rng.gen_range(3.0..5.0).to_string());
                metadata.insert("in_stock".to_string(), "true".to_string());
                
                // Product embeddings incorporating multiple factors
                let embedding: Vec<f32> = (0..384).map(|j| {
                    let category_factor = (category.len() * 13) as f32 * 0.01;
                    let brand_factor = (brand.len() * 7) as f32 * 0.005;
                    let price_factor = (price / 100.0).ln().abs() * 0.1;
                    ((j as f32 * 0.01) + category_factor + brand_factor + price_factor + i as f32 * 0.0001).sin()
                }).collect();
                
                let key = EntityKey::from_hash(&format!("product_{}", i));
                let content_id = interner.insert(&format!("{}\n{}", product_name, description));
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding,
                    metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark complete customer journey
            b.iter(|| {
                let mut session = UserSession::new("customer_123".to_string());
                let journey_start = Instant::now();
                
                // Phase 1: Initial search for electronics
                let search_query = "high quality electronics bluetooth wireless";
                let search_embedding: Vec<f32> = (0..384).map(|i| {
                    let electronics_factor = 0.8; // Strong electronics signal
                    let quality_factor = 0.6;
                    ((i as f32 * 0.01) + electronics_factor + quality_factor).sin()
                }).collect();
                
                let search_results = black_box(graph.find_similar_entities(&search_embedding, 20));
                session.add_operation(
                    OperationType::Search(search_query.to_string()),
                    HashMap::from([("results_count".to_string(), search_results.len().to_string())])
                );
                
                // Phase 2: Browse specific products (simulate clicks)
                let mut viewed_products = Vec::new();
                for (i, result) in search_results.iter().take(5).enumerate() {
                    viewed_products.push(result.key);
                    session.add_operation(
                        OperationType::Browse(format!("product_view_{}", i)),
                        HashMap::from([("product_id".to_string(), format!("{:?}", result.key))])
                    );
                }
                
                // Phase 3: Get personalized recommendations based on browsing
                let mut browsing_profile: Vec<f32> = vec![0.0; 384];
                let mut profile_count = 0;
                
                for product_key in &viewed_products {
                    for entity in graph.get_all_entities() {
                        if entity.key == *product_key {
                            for (j, &value) in entity.embedding.iter().enumerate() {
                                browsing_profile[j] += value;
                            }
                            profile_count += 1;
                            break;
                        }
                    }
                }
                
                if profile_count > 0 {
                    for value in &mut browsing_profile {
                        *value /= profile_count as f32;
                    }
                }
                
                let recommendations = black_box(graph.find_similar_entities(&browsing_profile, 15));
                session.add_operation(
                    OperationType::Recommend,
                    HashMap::from([("recommendations_count".to_string(), recommendations.len().to_string())])
                );
                
                // Phase 4: Filter by price range and category
                let mut filtered_products = Vec::new();
                for entity in recommendations.iter().take(10) {
                    if let Some(price_str) = entity.metadata.get("price") {
                        if let Ok(price) = price_str.parse::<f32>() {
                            if price >= 50.0 && price <= 300.0 {
                                if let Some(category) = entity.metadata.get("category") {
                                    if category == "electronics" {
                                        filtered_products.push(entity);
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Phase 5: Compare products (detailed views)
                let mut comparison_features = HashMap::new();
                for product in filtered_products.iter().take(3) {
                    if let Some(rating) = product.metadata.get("rating") {
                        comparison_features.insert(format!("{:?}_rating", product.key), rating.clone());
                    }
                    if let Some(price) = product.metadata.get("price") {
                        comparison_features.insert(format!("{:?}_price", product.key), price.clone());
                    }
                    
                    session.add_operation(
                        OperationType::Browse("detailed_comparison".to_string()),
                        HashMap::from([("product_id".to_string(), format!("{:?}", product.key))])
                    );
                }
                
                // Phase 6: Final selection and purchase simulation
                if let Some(selected_product) = filtered_products.first() {
                    session.add_operation(
                        OperationType::Add("cart".to_string()),
                        HashMap::from([
                            ("product_id".to_string(), format!("{:?}", selected_product.key)),
                            ("action".to_string(), "add_to_cart".to_string())
                        ])
                    );
                }
                
                let journey_duration = journey_start.elapsed();
                black_box((session, journey_duration, filtered_products.len()))
            });
        });
    }
    
    group.finish();
}

// Simulate content discovery and consumption workflow
fn benchmark_content_discovery_journey(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_discovery_journey");
    
    for content_count in [25_000, 100_000, 250_000] {
        group.throughput(Throughput::Elements(content_count as u64));
        
        group.bench_with_input(BenchmarkId::new("content_consumption_flow", content_count), &content_count, |b, &content_count| {
            let temp_dir = TempDir::new().unwrap();
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(512, 8, 64).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            let mut rng = StdRng::seed_from_u64(789);
            
            // Setup content library
            let content_types = ["article", "video", "podcast", "tutorial", "news"];
            let topics = ["technology", "science", "health", "finance", "entertainment", "sports"];
            let difficulty_levels = ["beginner", "intermediate", "advanced"];
            let languages = ["english", "spanish", "french", "german"];
            
            for i in 0..*content_count {
                let content_type = content_types[i % content_types.len()];
                let topic = topics[i % topics.len()];
                let difficulty = difficulty_levels[i % difficulty_levels.len()];
                let language = languages[i % languages.len()];
                
                let title = format!("{} {} Content: Advanced {} Techniques", 
                    content_type, topic, difficulty);
                let description = format!("Comprehensive {} covering {} topics at {} level in {}", 
                    content_type, topic, difficulty, language);
                
                let mut metadata = HashMap::new();
                metadata.insert("content_type".to_string(), content_type.to_string());
                metadata.insert("topic".to_string(), topic.to_string());
                metadata.insert("difficulty".to_string(), difficulty.to_string());
                metadata.insert("language".to_string(), language.to_string());
                metadata.insert("duration".to_string(), rng.gen_range(300..7200).to_string());
                metadata.insert("rating".to_string(), rng.gen_range(2.0..5.0).to_string());
                metadata.insert("view_count".to_string(), rng.gen_range(100..100000).to_string());
                
                // Content embeddings with rich feature representation
                let embedding: Vec<f32> = (0..512).map(|j| {
                    let type_factor = (content_type.len() * 11) as f32 * 0.02;
                    let topic_factor = (topic.len() * 13) as f32 * 0.01;
                    let difficulty_factor = match difficulty {
                        "beginner" => 0.1,
                        "intermediate" => 0.5,
                        "advanced" => 0.9,
                        _ => 0.5,
                    };
                    let language_factor = (language.len() * 7) as f32 * 0.005;
                    
                    ((j as f32 * 0.02) + type_factor + topic_factor + difficulty_factor + language_factor).sin()
                }).collect();
                
                let key = EntityKey::from_hash(&format!("content_{}", i));
                let content_id = interner.insert(&format!("{}\n{}", title, description));
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding,
                    metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark content discovery journey
            b.iter(|| {
                let mut session = UserSession::new("content_user_456".to_string());
                let journey_start = Instant::now();
                
                // Phase 1: Initial interest-based discovery
                let user_interests = vec!["technology", "science"];
                let tech_science_embedding: Vec<f32> = (0..512).map(|i| {
                    let tech_factor = 0.7;
                    let science_factor = 0.6;
                    ((i as f32 * 0.02) + tech_factor + science_factor).sin()
                }).collect();
                
                let initial_results = black_box(graph.find_similar_entities(&tech_science_embedding, 30));
                session.add_operation(
                    OperationType::Search("technology science".to_string()),
                    HashMap::from([("initial_results".to_string(), initial_results.len().to_string())])
                );
                
                // Phase 2: Filter by user preferences (intermediate difficulty, video content)
                let mut filtered_content = Vec::new();
                for entity in initial_results.iter() {
                    let is_intermediate = entity.metadata.get("difficulty") == Some(&"intermediate".to_string());
                    let is_video = entity.metadata.get("content_type") == Some(&"video".to_string());
                    let is_english = entity.metadata.get("language") == Some(&"english".to_string());
                    
                    if is_intermediate && (is_video || entity.metadata.get("content_type") == Some(&"tutorial".to_string())) && is_english {
                        if let Some(rating_str) = entity.metadata.get("rating") {
                            if let Ok(rating) = rating_str.parse::<f32>() {
                                if rating >= 3.5 {
                                    filtered_content.push(entity);
                                }
                            }
                        }
                    }
                }
                
                // Phase 3: Consume content (simulate viewing/reading)
                let mut consumed_content = Vec::new();
                let mut total_engagement_time = 0;
                
                for (i, content) in filtered_content.iter().take(5).enumerate() {
                    let engagement_duration = if let Some(duration_str) = content.metadata.get("duration") {
                        if let Ok(duration) = duration_str.parse::<i32>() {
                            // Simulate partial consumption (60-90% of content)
                            let consumption_rate = rng.gen_range(0.6..0.9);
                            (duration as f32 * consumption_rate) as i32
                        } else {
                            300 // Default 5 minutes
                        }
                    } else {
                        300
                    };
                    
                    total_engagement_time += engagement_duration;
                    consumed_content.push(content.key);
                    
                    session.add_operation(
                        OperationType::Browse(format!("content_consumption_{}", i)),
                        HashMap::from([
                            ("content_id".to_string(), format!("{:?}", content.key)),
                            ("engagement_time".to_string(), engagement_duration.to_string())
                        ])
                    );
                }
                
                // Phase 4: Generate personalized recommendations based on consumption
                let mut consumption_profile: Vec<f32> = vec![0.0; 512];
                let mut profile_weight = 0;
                
                for content_key in &consumed_content {
                    for entity in graph.get_all_entities() {
                        if entity.key == *content_key {
                            for (j, &value) in entity.embedding.iter().enumerate() {
                                consumption_profile[j] += value;
                            }
                            profile_weight += 1;
                            break;
                        }
                    }
                }
                
                if profile_weight > 0 {
                    for value in &mut consumption_profile {
                        *value /= profile_weight as f32;
                    }
                }
                
                let recommendations = black_box(graph.find_similar_entities(&consumption_profile, 25));
                session.add_operation(
                    OperationType::Recommend,
                    HashMap::from([
                        ("recommendations_count".to_string(), recommendations.len().to_string()),
                        ("based_on_consumption".to_string(), consumed_content.len().to_string())
                    ])
                );
                
                // Phase 5: Explore related topics and cross-recommendations
                let mut related_topics = Vec::new();
                for entity in recommendations.iter().take(10) {
                    if let Some(topic) = entity.metadata.get("topic") {
                        if !user_interests.contains(topic) {
                            related_topics.push(topic.clone());
                        }
                    }
                }
                
                // Phase 6: Create user learning path
                let mut learning_path = Vec::new();
                let difficulty_progression = ["beginner", "intermediate", "advanced"];
                
                for difficulty in &difficulty_progression {
                    let difficulty_embedding: Vec<f32> = consumption_profile.iter().enumerate().map(|(i, &val)| {
                        let difficulty_bonus = match *difficulty {
                            "beginner" => 0.1,
                            "intermediate" => 0.5,
                            "advanced" => 0.9,
                            _ => 0.5,
                        };
                        val + difficulty_bonus * (i as f32 * 0.001).sin()
                    }).collect();
                    
                    let difficulty_content = graph.find_similar_entities(&difficulty_embedding, 5);
                    learning_path.extend(difficulty_content);
                }
                
                session.add_operation(
                    OperationType::Browse("learning_path_generation".to_string()),
                    HashMap::from([
                        ("path_length".to_string(), learning_path.len().to_string()),
                        ("total_engagement".to_string(), total_engagement_time.to_string())
                    ])
                );
                
                let journey_duration = journey_start.elapsed();
                black_box((session, journey_duration, learning_path.len(), total_engagement_time))
            });
        });
    }
    
    group.finish();
}

// Simulate research and knowledge discovery workflow
fn benchmark_research_workflow_journey(c: &mut Criterion) {
    let mut group = c.benchmark_group("research_workflow_journey");
    
    for knowledge_base_size in [50_000, 200_000, 500_000] {
        group.throughput(Throughput::Elements(knowledge_base_size as u64));
        
        group.bench_with_input(BenchmarkId::new("academic_research_flow", knowledge_base_size), &knowledge_base_size, |b, &knowledge_base_size| {
            let temp_dir = TempDir::new().unwrap();
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(768, 12, 64).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            let mut rng = StdRng::seed_from_u64(101112);
            
            // Setup academic knowledge base
            let research_fields = ["computer_science", "biology", "physics", "chemistry", "mathematics", "psychology"];
            let publication_types = ["journal_article", "conference_paper", "book_chapter", "thesis", "review"];
            let years = 2015..=2024;
            
            for i in 0..*knowledge_base_size {
                let field = research_fields[i % research_fields.len()];
                let pub_type = publication_types[i % publication_types.len()];
                let year = years.clone().nth(i % years.len()).unwrap_or(2024);
                
                let title = format!("Advanced Research in {}: Novel Approaches and Methodologies {}", 
                    field.replace("_", " "), i);
                let abstract_text = format!("This {} presents groundbreaking research in {} with significant implications for the field. \
                    Our methodology demonstrates novel approaches to understanding complex phenomena in {}.",
                    pub_type.replace("_", " "), field.replace("_", " "), field.replace("_", " "));
                
                let mut metadata = HashMap::new();
                metadata.insert("field".to_string(), field.to_string());
                metadata.insert("publication_type".to_string(), pub_type.to_string());
                metadata.insert("year".to_string(), year.to_string());
                metadata.insert("citation_count".to_string(), rng.gen_range(0..500).to_string());
                metadata.insert("h_index".to_string(), rng.gen_range(1..50).to_string());
                metadata.insert("impact_factor".to_string(), rng.gen_range(0.5..10.0).to_string());
                
                // Research embeddings with field-specific characteristics
                let embedding: Vec<f32> = (0..768).map(|j| {
                    let field_factor = match field {
                        "computer_science" => 0.1,
                        "biology" => 0.2,
                        "physics" => 0.3,
                        "chemistry" => 0.4,
                        "mathematics" => 0.5,
                        "psychology" => 0.6,
                        _ => 0.0,
                    };
                    
                    let type_factor = match pub_type {
                        "journal_article" => 0.8,
                        "conference_paper" => 0.6,
                        "book_chapter" => 0.4,
                        "thesis" => 0.9,
                        "review" => 0.7,
                        _ => 0.5,
                    };
                    
                    let year_factor = (year - 2015) as f32 * 0.01;
                    
                    ((j as f32 * 0.005) + field_factor + type_factor + year_factor + i as f32 * 0.0001).sin()
                }).collect();
                
                let key = EntityKey::from_hash(&format!("research_{}", i));
                let content_id = interner.insert(&format!("{}\n\nAbstract:\n{}", title, abstract_text));
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding,
                    metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark research workflow
            b.iter(|| {
                let mut session = UserSession::new("researcher_789".to_string());
                let journey_start = Instant::now();
                
                // Phase 1: Literature review - broad topic search
                let research_topic = "machine learning applications in biology";
                let topic_embedding: Vec<f32> = (0..768).map(|i| {
                    let cs_factor = 0.1; // Computer science
                    let bio_factor = 0.2; // Biology
                    let ml_factor = 0.5; // Machine learning emphasis
                    ((i as f32 * 0.005) + cs_factor + bio_factor + ml_factor).sin()
                }).collect();
                
                let literature_results = black_box(graph.find_similar_entities(&topic_embedding, 100));
                session.add_operation(
                    OperationType::Search(research_topic.to_string()),
                    HashMap::from([("literature_count".to_string(), literature_results.len().to_string())])
                );
                
                // Phase 2: Filter by quality metrics (high impact, recent)
                let mut high_quality_papers = Vec::new();
                for entity in literature_results.iter() {
                    let year_ok = if let Some(year_str) = entity.metadata.get("year") {
                        if let Ok(year) = year_str.parse::<i32>() {
                            year >= 2020 // Recent papers
                        } else { false }
                    } else { false };
                    
                    let citation_ok = if let Some(citation_str) = entity.metadata.get("citation_count") {
                        if let Ok(citations) = citation_str.parse::<i32>() {
                            citations >= 10 // Well-cited papers
                        } else { false }
                    } else { false };
                    
                    let impact_ok = if let Some(impact_str) = entity.metadata.get("impact_factor") {
                        if let Ok(impact) = impact_str.parse::<f32>() {
                            impact >= 2.0 // High-impact journals
                        } else { false }
                    } else { false };
                    
                    if year_ok && citation_ok && impact_ok {
                        high_quality_papers.push(entity);
                    }
                }
                
                // Phase 3: Deep reading and note-taking simulation
                let mut research_notes = Vec::new();
                let mut key_concepts = Vec::new();
                
                for (i, paper) in high_quality_papers.iter().take(20).enumerate() {
                    // Simulate reading time based on publication type
                    let reading_time = match paper.metadata.get("publication_type").map(|s| s.as_str()) {
                        Some("journal_article") => rng.gen_range(30..60),
                        Some("conference_paper") => rng.gen_range(15..30),
                        Some("book_chapter") => rng.gen_range(45..90),
                        Some("thesis") => rng.gen_range(120..240),
                        Some("review") => rng.gen_range(60..120),
                        _ => 30,
                    };
                    
                    research_notes.push(format!("Notes from paper {}: {}", i, reading_time));
                    if let Some(field) = paper.metadata.get("field") {
                        key_concepts.push(field.clone());
                    }
                    
                    session.add_operation(
                        OperationType::Browse(format!("deep_read_{}", i)),
                        HashMap::from([
                            ("paper_id".to_string(), format!("{:?}", paper.key)),
                            ("reading_time_minutes".to_string(), reading_time.to_string())
                        ])
                    );
                }
                
                // Phase 4: Citation network analysis
                let mut citation_network = Vec::new();
                for paper in high_quality_papers.iter().take(10) {
                    // Find papers that cite similar concepts
                    let similar_papers = graph.find_similar_entities(&paper.embedding, 15);
                    citation_network.extend(similar_papers);
                }
                
                // Phase 5: Identify research gaps
                let mut research_gaps = Vec::new();
                let gap_keywords = ["novel", "unexplored", "limited", "gap", "future work"];
                
                for entity in citation_network.iter() {
                    let content_str = interner.get(entity.content).unwrap_or("");
                    for keyword in &gap_keywords {
                        if content_str.to_lowercase().contains(keyword) {
                            research_gaps.push(entity.key);
                            break;
                        }
                    }
                }
                
                // Phase 6: Synthesis and new research direction
                let mut synthesis_profile: Vec<f32> = vec![0.0; 768];
                let mut synthesis_count = 0;
                
                for paper in high_quality_papers.iter().take(15) {
                    for (j, &value) in paper.embedding.iter().enumerate() {
                        synthesis_profile[j] += value;
                    }
                    synthesis_count += 1;
                }
                
                if synthesis_count > 0 {
                    for value in &mut synthesis_profile {
                        *value /= synthesis_count as f32;
                    }
                }
                
                // Add novelty factor for new research direction
                for (i, value) in synthesis_profile.iter_mut().enumerate() {
                    *value += (i as f32 * 0.001).cos() * 0.1; // Innovation vector
                }
                
                let novel_directions = black_box(graph.find_similar_entities(&synthesis_profile, 30));
                session.add_operation(
                    OperationType::Add("research_proposal".to_string()),
                    HashMap::from([
                        ("synthesis_papers".to_string(), synthesis_count.to_string()),
                        ("novel_directions".to_string(), novel_directions.len().to_string()),
                        ("research_gaps".to_string(), research_gaps.len().to_string())
                    ])
                );
                
                let journey_duration = journey_start.elapsed();
                black_box((session, journey_duration, research_notes.len(), novel_directions.len()))
            });
        });
    }
    
    group.finish();
}

// Simulate collaborative knowledge building workflow
fn benchmark_collaborative_knowledge_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("collaborative_knowledge_building");
    
    for knowledge_entries in [25_000, 100_000, 250_000] {
        group.throughput(Throughput::Elements(knowledge_entries as u64));
        
        group.bench_with_input(BenchmarkId::new("wiki_like_collaboration", knowledge_entries), &knowledge_entries, |b, &knowledge_entries| {
            let temp_dir = TempDir::new().unwrap();
            let graph = Arc::new(Mutex::new(EntityGraph::new()));
            let quantizer = Arc::new(ProductQuantizer::new(512, 8, 64).unwrap());
            let interner = Arc::new(Mutex::new(StringInterner::new()));
            let storage = Arc::new(PersistentMMapStorage::new(Some(temp_dir.path())).unwrap());
            
            let mut rng = StdRng::seed_from_u64(131415);
            
            // Pre-populate with initial knowledge base
            {
                let mut graph_guard = graph.lock().unwrap();
                let mut interner_guard = interner.lock().unwrap();
                
                let domains = ["science", "technology", "history", "arts", "culture", "politics"];
                let entry_types = ["article", "definition", "example", "tutorial", "reference"];
                
                for i in 0..*knowledge_entries {
                    let domain = domains[i % domains.len()];
                    let entry_type = entry_types[i % entry_types.len()];
                    
                    let title = format!("{} {}: Entry {}", domain, entry_type, i);
                    let content = format!("Comprehensive {} about {} covering fundamental concepts and advanced topics with detailed explanations and examples.", 
                        entry_type, domain);
                    
                    let mut metadata = HashMap::new();
                    metadata.insert("domain".to_string(), domain.to_string());
                    metadata.insert("entry_type".to_string(), entry_type.to_string());
                    metadata.insert("contributor_count".to_string(), rng.gen_range(1..10).to_string());
                    metadata.insert("edit_count".to_string(), rng.gen_range(1..50).to_string());
                    metadata.insert("quality_score".to_string(), rng.gen_range(0.5..1.0).to_string());
                    
                    let embedding: Vec<f32> = (0..512).map(|j| {
                        let domain_factor = (domain.len() * 17) as f32 * 0.01;
                        let type_factor = (entry_type.len() * 11) as f32 * 0.005;
                        ((j as f32 * 0.008) + domain_factor + type_factor + i as f32 * 0.0001).sin()
                    }).collect();
                    
                    let key = EntityKey::from_hash(&format!("knowledge_{}", i));
                    let content_id = interner_guard.insert(&format!("{}\n\n{}", title, content));
                    
                    let entity = Entity {
                        key,
                        content: content_id,
                        embedding,
                        metadata,
                    };
                    
                    graph_guard.add_entity(entity);
                }
            }
            
            // Benchmark collaborative workflow
            b.iter(|| {
                let workflow_start = Instant::now();
                let mut collaborative_operations = Vec::new();
                
                // Phase 1: Multiple users contributing simultaneously
                let contributor_count = 5;
                let operations_per_contributor = 20;
                
                let handles: Vec<_> = (0..contributor_count).map(|contributor_id| {
                    let graph = graph.clone();
                    let interner = interner.clone();
                    let mut contributor_rng = StdRng::seed_from_u64(contributor_id as u64 + 1000);
                    
                    thread::spawn(move || {
                        let mut operations = Vec::new();
                        
                        for op_id in 0..operations_per_contributor {
                            let operation_type = contributor_rng.gen_range(0..100);
                            
                            if operation_type < 40 { // 40% search and read
                                let search_embedding: Vec<f32> = (0..512).map(|_| contributor_rng.gen_range(-1.0..1.0)).collect();
                                
                                {
                                    let graph_guard = graph.lock().unwrap();
                                    let results = graph_guard.find_similar_entities(&search_embedding, 10);
                                    operations.push(("search".to_string(), results.len()));
                                }
                                
                            } else if operation_type < 60 { // 20% edit existing content
                                let edit_id = format!("edit_{}_{}", contributor_id, op_id);
                                operations.push(("edit".to_string(), 1));
                                
                            } else if operation_type < 80 { // 20% add new content
                                let new_title = format!("Contributor {} Entry {}", contributor_id, op_id);
                                let new_content = format!("New knowledge entry created by contributor {} with detailed information and cross-references.", contributor_id);
                                
                                let mut metadata = HashMap::new();
                                metadata.insert("contributor".to_string(), contributor_id.to_string());
                                metadata.insert("created_timestamp".to_string(), "2024-01-01T00:00:00Z".to_string());
                                
                                let embedding: Vec<f32> = (0..512).map(|j| {
                                    let contributor_signature = contributor_id as f32 * 0.1;
                                    ((j as f32 * 0.008) + contributor_signature + op_id as f32 * 0.001).sin()
                                }).collect();
                                
                                let key = EntityKey::from_hash(&format!("contrib_{}_{}", contributor_id, op_id));
                                let content_id = {
                                    let mut interner_guard = interner.lock().unwrap();
                                    interner_guard.insert(&format!("{}\n\n{}", new_title, new_content))
                                };
                                
                                let entity = Entity {
                                    key,
                                    content: content_id,
                                    embedding,
                                    metadata,
                                };
                                
                                {
                                    let mut graph_guard = graph.lock().unwrap();
                                    graph_guard.add_entity(entity);
                                }
                                
                                operations.push(("create".to_string(), 1));
                                
                            } else { // 20% link and cross-reference
                                let link_embedding: Vec<f32> = (0..512).map(|_| contributor_rng.gen_range(-0.5..0.5)).collect();
                                
                                {
                                    let graph_guard = graph.lock().unwrap();
                                    let related_entries = graph_guard.find_similar_entities(&link_embedding, 5);
                                    operations.push(("link".to_string(), related_entries.len()));
                                }
                            }
                        }
                        
                        operations
                    })
                }).collect();
                
                // Collect results from all contributors
                for handle in handles {
                    let contributor_operations = handle.join().unwrap();
                    collaborative_operations.extend(contributor_operations);
                }
                
                // Phase 2: Quality assessment and moderation
                let quality_checks = {
                    let graph_guard = graph.lock().unwrap();
                    let mut quality_scores = Vec::new();
                    
                    for entity in graph_guard.get_all_entities().iter().take(100) {
                        if let Some(quality_str) = entity.metadata.get("quality_score") {
                            if let Ok(quality) = quality_str.parse::<f32>() {
                                quality_scores.push(quality);
                            }
                        }
                    }
                    
                    quality_scores
                };
                
                // Phase 3: Knowledge graph analysis and relationship discovery
                let relationship_analysis = {
                    let graph_guard = graph.lock().unwrap();
                    let mut cluster_analysis = Vec::new();
                    
                    // Sample entities for clustering analysis
                    let sample_entities: Vec<_> = graph_guard.get_all_entities().iter().take(50).collect();
                    
                    for entity in sample_entities {
                        let similar_entities = graph_guard.find_similar_entities(&entity.embedding, 8);
                        cluster_analysis.push(similar_entities.len());
                    }
                    
                    cluster_analysis
                };
                
                let workflow_duration = workflow_start.elapsed();
                black_box((collaborative_operations.len(), quality_checks.len(), relationship_analysis.len(), workflow_duration))
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    user_journey_benches,
    benchmark_ecommerce_customer_journey,
    benchmark_content_discovery_journey,
    benchmark_research_workflow_journey,
    benchmark_collaborative_knowledge_building
);

criterion_main!(user_journey_benches);