/*!
Phase 5.3: Industry-Specific Scenario Benchmarks
Benchmarks that simulate specific industry use cases and workflows
*/

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// Industry-specific entity types and patterns

// Healthcare: Medical records, research papers, drug information
#[derive(Clone)]
struct MedicalRecord {
    patient_id: String,
    record_type: String, // "diagnosis", "treatment", "lab_result", "imaging"
    content: String,
    icd_codes: Vec<String>,
    embedding: Vec<f32>,
    timestamp: String,
    metadata: HashMap<String, String>,
}

impl MedicalRecord {
    fn new_diagnosis(patient_id: &str, diagnosis: &str, icd_codes: Vec<String>) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("specialty".to_string(), "internal_medicine".to_string());
        metadata.insert("severity".to_string(), "moderate".to_string());
        
        // Medical embeddings based on diagnosis and codes
        let embedding: Vec<f32> = (0..768).map(|i| {
            let code_factor = icd_codes.len() as f32 * 0.1;
            ((i as f32 * 0.001) + code_factor).sin()
        }).collect();
        
        Self {
            patient_id: patient_id.to_string(),
            record_type: "diagnosis".to_string(),
            content: diagnosis.to_string(),
            icd_codes,
            embedding,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            metadata,
        }
    }
    
    fn new_research_paper(title: &str, abstract_text: &str, keywords: Vec<String>) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("publication_type".to_string(), "research".to_string());
        metadata.insert("peer_reviewed".to_string(), "true".to_string());
        metadata.insert("keywords".to_string(), keywords.join(","));
        
        // Research paper embeddings
        let embedding: Vec<f32> = (0..768).map(|i| {
            let keyword_factor = keywords.len() as f32 * 0.05;
            ((i as f32 * 0.002) + keyword_factor).cos()
        }).collect();
        
        Self {
            patient_id: "research".to_string(),
            record_type: "research_paper".to_string(),
            content: format!("{}\n\n{}", title, abstract_text),
            icd_codes: vec![],
            embedding,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            metadata,
        }
    }
}

// E-commerce: Products, users, reviews, recommendations
#[derive(Clone)]
struct EcommerceItem {
    item_id: String,
    item_type: String, // "product", "user", "review", "transaction"
    content: String,
    categories: Vec<String>,
    embedding: Vec<f32>,
    metadata: HashMap<String, String>,
}

impl EcommerceItem {
    fn new_product(id: &str, name: &str, description: &str, categories: Vec<String>, price: f32) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("price".to_string(), price.to_string());
        metadata.insert("availability".to_string(), "in_stock".to_string());
        metadata.insert("brand".to_string(), "generic".to_string());
        
        // Product embeddings incorporating price and categories
        let embedding: Vec<f32> = (0..512).map(|i| {
            let category_factor = categories.len() as f32 * 0.1;
            let price_factor = (price / 100.0).ln().abs();
            ((i as f32 * 0.003) + category_factor + price_factor).sin()
        }).collect();
        
        Self {
            item_id: id.to_string(),
            item_type: "product".to_string(),
            content: format!("{}\n{}", name, description),
            categories,
            embedding,
            metadata,
        }
    }
    
    fn new_user_profile(id: &str, preferences: Vec<String>, purchase_history: Vec<String>) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("account_type".to_string(), "premium".to_string());
        metadata.insert("join_date".to_string(), "2023-01-01".to_string());
        metadata.insert("purchase_count".to_string(), purchase_history.len().to_string());
        
        // User embeddings based on preferences and history
        let embedding: Vec<f32> = (0..512).map(|i| {
            let pref_factor = preferences.len() as f32 * 0.05;
            let history_factor = purchase_history.len() as f32 * 0.02;
            ((i as f32 * 0.004) + pref_factor + history_factor).cos()
        }).collect();
        
        Self {
            item_id: id.to_string(),
            item_type: "user".to_string(),
            content: format!("User preferences: {}", preferences.join(", ")),
            categories: preferences,
            embedding,
            metadata,
        }
    }
}

// Financial: Trading data, risk assessments, compliance documents
#[derive(Clone)]
struct FinancialDocument {
    doc_id: String,
    doc_type: String, // "trade", "risk_report", "compliance", "market_data"
    content: String,
    risk_level: f32,
    embedding: Vec<f32>,
    metadata: HashMap<String, String>,
}

impl FinancialDocument {
    fn new_trade_record(id: &str, instrument: &str, amount: f32, risk_score: f32) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("instrument_type".to_string(), instrument.to_string());
        metadata.insert("amount".to_string(), amount.to_string());
        metadata.insert("currency".to_string(), "USD".to_string());
        metadata.insert("exchange".to_string(), "NYSE".to_string());
        
        // Financial embeddings incorporating risk and amount
        let embedding: Vec<f32> = (0..256).map(|i| {
            let risk_factor = risk_score * 0.1;
            let amount_factor = (amount.abs() / 1000.0).ln().abs();
            ((i as f32 * 0.005) + risk_factor + amount_factor).sin()
        }).collect();
        
        Self {
            doc_id: id.to_string(),
            doc_type: "trade".to_string(),
            content: format!("Trade record for {} instrument, amount: {}", instrument, amount),
            risk_level: risk_score,
            embedding,
            metadata,
        }
    }
    
    fn new_risk_report(id: &str, risk_category: &str, assessment: &str, risk_level: f32) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("risk_category".to_string(), risk_category.to_string());
        metadata.insert("assessment_date".to_string(), "2024-01-01".to_string());
        metadata.insert("reviewer".to_string(), "risk_team".to_string());
        
        // Risk report embeddings
        let embedding: Vec<f32> = (0..256).map(|i| {
            let category_hash = risk_category.len() as f32 * 0.1;
            ((i as f32 * 0.006) + risk_level + category_hash).cos()
        }).collect();
        
        Self {
            doc_id: id.to_string(),
            doc_type: "risk_report".to_string(),
            content: format!("Risk assessment for {}: {}", risk_category, assessment),
            risk_level,
            embedding,
            metadata,
        }
    }
}

// Benchmark healthcare workflow
fn benchmark_healthcare_knowledge_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("healthcare_knowledge_retrieval");
    
    for record_count in [5000, 10000, 25000] {
        group.throughput(Throughput::Elements(record_count as u64));
        
        group.bench_with_input(BenchmarkId::new("medical_diagnosis_lookup", record_count), &record_count, |b, &record_count| {
            let temp_dir = TempDir::new().unwrap();
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(768, 12, 64).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            // Generate medical records
            let diagnoses = [
                "Type 2 Diabetes Mellitus", "Hypertension", "Coronary Artery Disease",
                "Chronic Obstructive Pulmonary Disease", "Depression", "Anxiety Disorder",
                "Osteoarthritis", "Chronic Kidney Disease", "Atrial Fibrillation", "Asthma"
            ];
            
            let icd_codes = [
                vec!["E11.9"], vec!["I10"], vec!["I25.9"],
                vec!["J44.9"], vec!["F32.9"], vec!["F41.9"],
                vec!["M19.9"], vec!["N18.9"], vec!["I48.91"], vec!["J45.9"]
            ];
            
            for i in 0..record_count {
                let diagnosis_idx = i % diagnoses.len();
                let record = MedicalRecord::new_diagnosis(
                    &format!("patient_{}", i),
                    diagnoses[diagnosis_idx],
                    icd_codes[diagnosis_idx].clone()
                );
                
                let key = EntityKey::from_hash(&record.patient_id);
                let content_id = interner.insert(&record.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: record.embedding,
                    metadata: record.metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Add research papers
            let research_papers = [
                ("Diabetes Management in Primary Care", "Recent advances in diabetes management...", vec!["diabetes", "primary_care"]),
                ("Hypertension Treatment Guidelines", "Updated guidelines for hypertension treatment...", vec!["hypertension", "guidelines"]),
                ("COPD Therapeutic Approaches", "Novel therapeutic approaches for COPD patients...", vec!["copd", "therapy"]),
            ];
            
            for (title, abstract_text, keywords) in research_papers.iter() {
                let paper = MedicalRecord::new_research_paper(title, abstract_text, keywords.clone());
                let key = EntityKey::from_hash(&format!("paper_{}", title));
                let content_id = interner.insert(&paper.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: paper.embedding,
                    metadata: paper.metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark: Find similar cases for diagnosis
            b.iter(|| {
                // Simulate doctor looking up similar cases for a new diabetes patient
                let query_record = MedicalRecord::new_diagnosis(
                    "new_patient",
                    "Type 2 Diabetes Mellitus with complications",
                    vec!["E11.9".to_string()]
                );
                
                let similar_cases = black_box(graph.find_similar_entities(&query_record.embedding, 10));
                black_box(similar_cases)
            });
        });
    }
    
    group.finish();
}

fn benchmark_ecommerce_recommendation_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("ecommerce_recommendation_engine");
    
    for catalog_size in [10000, 50000, 100000] {
        group.throughput(Throughput::Elements(catalog_size as u64));
        
        group.bench_with_input(BenchmarkId::new("personalized_recommendations", catalog_size), &catalog_size, |b, &catalog_size| {
            let temp_dir = TempDir::new().unwrap();
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(512, 8, 64).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            let mut rng = StdRng::seed_from_u64(42);
            
            // Generate product catalog
            let categories = ["electronics", "books", "clothing", "home", "sports", "beauty", "toys", "automotive"];
            let product_names = ["Premium", "Deluxe", "Standard", "Basic", "Pro", "Max", "Ultra", "Compact"];
            
            for i in 0..catalog_size {
                let category = categories[i % categories.len()];
                let name = product_names[i % product_names.len()];
                let price = rng.gen_range(10.0..1000.0);
                
                let product = EcommerceItem::new_product(
                    &format!("product_{}", i),
                    &format!("{} {}", name, category),
                    &format!("High-quality {} product with excellent features", category),
                    vec![category.to_string()],
                    price
                );
                
                let key = EntityKey::from_hash(&product.item_id);
                let content_id = interner.insert(&product.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: product.embedding,
                    metadata: product.metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Generate user profiles
            let user_count = catalog_size / 100;
            for i in 0..user_count {
                let category_count = rng.gen_range(1..4);
                let user_categories: Vec<String> = categories.choose_multiple(&mut rng, category_count)
                    .map(|s| s.to_string())
                    .collect();
                
                let purchase_history: Vec<String> = (0..rng.gen_range(5..20))
                    .map(|j| format!("product_{}", j))
                    .collect();
                
                let user = EcommerceItem::new_user_profile(
                    &format!("user_{}", i),
                    user_categories,
                    purchase_history
                );
                
                let key = EntityKey::from_hash(&user.item_id);
                let content_id = interner.insert(&user.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: user.embedding,
                    metadata: user.metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark: Generate personalized recommendations
            b.iter(|| {
                // Simulate recommendation generation for a user interested in electronics
                let sample_user = EcommerceItem::new_user_profile(
                    "test_user",
                    vec!["electronics".to_string(), "books".to_string()],
                    vec!["product_1".to_string(), "product_100".to_string()]
                );
                
                let recommendations = black_box(graph.find_similar_entities(&sample_user.embedding, 20));
                black_box(recommendations)
            });
        });
    }
    
    group.finish();
}

fn benchmark_financial_risk_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("financial_risk_analysis");
    
    for document_count in [5000, 25000, 50000] {
        group.throughput(Throughput::Elements(document_count as u64));
        
        group.bench_with_input(BenchmarkId::new("risk_pattern_detection", document_count), &document_count, |b, &document_count| {
            let temp_dir = TempDir::new().unwrap();
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(256, 4, 64).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            let mut rng = StdRng::seed_from_u64(123);
            
            // Generate trading records
            let instruments = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "BRK.A"];
            
            for i in 0..(document_count / 2) {
                let instrument = instruments[i % instruments.len()];
                let amount = rng.gen_range(-1000000.0..1000000.0);
                let risk_score = rng.gen_range(0.0..10.0);
                
                let trade = FinancialDocument::new_trade_record(
                    &format!("trade_{}", i),
                    instrument,
                    amount,
                    risk_score
                );
                
                let key = EntityKey::from_hash(&trade.doc_id);
                let content_id = interner.insert(&trade.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: trade.embedding,
                    metadata: trade.metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Generate risk reports
            let risk_categories = ["market_risk", "credit_risk", "operational_risk", "liquidity_risk"];
            
            for i in 0..(document_count / 2) {
                let category = risk_categories[i % risk_categories.len()];
                let assessment = format!("Detailed risk assessment for {} scenario {}", category, i);
                let risk_level = rng.gen_range(1.0..10.0);
                
                let report = FinancialDocument::new_risk_report(
                    &format!("risk_{}", i),
                    category,
                    &assessment,
                    risk_level
                );
                
                let key = EntityKey::from_hash(&report.doc_id);
                let content_id = interner.insert(&report.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: report.embedding,
                    metadata: report.metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark: Risk pattern analysis
            b.iter(|| {
                // Simulate risk analysis for high-risk scenarios
                let query_risk = FinancialDocument::new_risk_report(
                    "query_risk",
                    "market_risk",
                    "High volatility market conditions with elevated systemic risk",
                    8.5
                );
                
                let similar_risks = black_box(graph.find_similar_entities(&query_risk.embedding, 15));
                
                // Analyze risk patterns by aggregating similar documents
                let mut high_risk_count = 0;
                let mut total_risk_sum = 0.0;
                
                for entity in graph.get_all_entities() {
                    if let Some(risk_str) = entity.metadata.get("risk_level") {
                        if let Ok(risk_level) = risk_str.parse::<f32>() {
                            total_risk_sum += risk_level;
                            if risk_level > 7.0 {
                                high_risk_count += 1;
                            }
                        }
                    }
                }
                
                black_box((similar_risks, high_risk_count, total_risk_sum))
            });
        });
    }
    
    group.finish();
}

fn benchmark_content_discovery_platform(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_discovery_platform");
    
    for content_size in [20000, 100000, 200000] {
        group.throughput(Throughput::Elements(content_size as u64));
        
        group.bench_with_input(BenchmarkId::new("content_recommendation", content_size), &content_size, |b, &content_size| {
            let temp_dir = TempDir::new().unwrap();
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(384, 6, 64).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            let mut rng = StdRng::seed_from_u64(456);
            
            // Content types and topics
            let content_types = ["article", "video", "podcast", "infographic", "tutorial"];
            let topics = ["technology", "science", "health", "finance", "entertainment", "sports", "travel", "food"];
            let content_templates = [
                "Comprehensive guide to {}",
                "Latest trends in {}",
                "How to improve your {}",
                "Understanding {} fundamentals",
                "Advanced {} techniques",
            ];
            
            // Generate diverse content
            for i in 0..content_size {
                let content_type = content_types[i % content_types.len()];
                let topic = topics[i % topics.len()];
                let template = content_templates[i % content_templates.len()];
                
                let title = template.replace("{}", topic);
                let content = format!("{}. This {} covers various aspects of {} with detailed explanations and examples.", 
                    title, content_type, topic);
                
                let mut metadata = HashMap::new();
                metadata.insert("content_type".to_string(), content_type.to_string());
                metadata.insert("topic".to_string(), topic.to_string());
                metadata.insert("duration".to_string(), rng.gen_range(300..3600).to_string());
                metadata.insert("rating".to_string(), rng.gen_range(1.0..5.0).to_string());
                
                // Content embeddings based on type and topic
                let embedding: Vec<f32> = (0..384).map(|j| {
                    let type_factor = content_type.len() as f32 * 0.01;
                    let topic_factor = topic.len() as f32 * 0.02;
                    ((j as f32 * 0.007) + type_factor + topic_factor + i as f32 * 0.001).sin()
                }).collect();
                
                let key = EntityKey::from_hash(&format!("content_{}", i));
                let content_id = interner.insert(&content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding,
                    metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark: Content discovery and recommendation
            b.iter(|| {
                // Simulate user looking for technology content
                let query_embedding: Vec<f32> = (0..384).map(|i| {
                    let tech_factor = 0.5; // Strong preference for technology
                    ((i as f32 * 0.007) + tech_factor).sin()
                }).collect();
                
                let recommended_content = black_box(graph.find_similar_entities(&query_embedding, 25));
                
                // Simulate content filtering by type and rating
                let mut filtered_results = Vec::new();
                for entity in recommended_content.iter().take(25) {
                    if let Some(rating_str) = entity.metadata.get("rating") {
                        if let Ok(rating) = rating_str.parse::<f32>() {
                            if rating >= 3.5 {
                                filtered_results.push(entity);
                            }
                        }
                    }
                }
                
                black_box(filtered_results)
            });
        });
    }
    
    group.finish();
}

fn benchmark_scientific_literature_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("scientific_literature_search");
    
    for paper_count in [10000, 50000, 100000] {
        group.throughput(Throughput::Elements(paper_count as u64));
        
        group.bench_with_input(BenchmarkId::new("research_paper_discovery", paper_count), &paper_count, |b, &paper_count| {
            let temp_dir = TempDir::new().unwrap();
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(768, 12, 64).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            let mut rng = StdRng::seed_from_u64(789);
            
            // Scientific domains and keywords
            let domains = ["computer_science", "biology", "physics", "chemistry", "medicine", "mathematics"];
            let cs_keywords = ["machine_learning", "algorithms", "distributed_systems", "security", "databases"];
            let bio_keywords = ["genetics", "protein_folding", "cell_biology", "evolution", "ecology"];
            let physics_keywords = ["quantum_mechanics", "relativity", "thermodynamics", "optics", "particle_physics"];
            
            let all_keywords = [cs_keywords, bio_keywords, physics_keywords, cs_keywords, bio_keywords, cs_keywords];
            
            // Generate research papers
            for i in 0..paper_count {
                let domain = domains[i % domains.len()];
                let keywords = &all_keywords[i % all_keywords.len()];
                let paper_keywords: Vec<String> = keywords.choose_multiple(&mut rng, rng.gen_range(2..5))
                    .map(|s| s.to_string())
                    .collect();
                
                let title = format!("Advanced Research in {}: A Study of {}", 
                    domain.replace("_", " "), paper_keywords.join(" and "));
                let abstract_text = format!("This paper presents novel findings in {} research, focusing on {}. \
                    Our methodology demonstrates significant improvements in understanding {}.",
                    domain, paper_keywords[0], paper_keywords.join(", "));
                
                let paper = MedicalRecord::new_research_paper(&title, &abstract_text, paper_keywords);
                
                let key = EntityKey::from_hash(&format!("paper_{}", i));
                let content_id = interner.insert(&paper.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: paper.embedding,
                    metadata: paper.metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark: Scientific literature search
            b.iter(|| {
                // Simulate researcher searching for machine learning papers
                let query_paper = MedicalRecord::new_research_paper(
                    "Deep Learning Applications in Computer Vision",
                    "Exploring the latest deep learning techniques for computer vision tasks including object detection and image classification.",
                    vec!["machine_learning".to_string(), "computer_vision".to_string(), "deep_learning".to_string()]
                );
                
                let related_papers = black_box(graph.find_similar_entities(&query_paper.embedding, 30));
                
                // Simulate citation analysis - find papers that might cite similar work
                let mut citation_candidates = Vec::new();
                for entity in related_papers.iter().take(10) {
                    if entity.metadata.get("publication_type") == Some(&"research".to_string()) {
                        citation_candidates.push(entity);
                    }
                }
                
                black_box(citation_candidates)
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    industry_scenario_benches,
    benchmark_healthcare_knowledge_retrieval,
    benchmark_ecommerce_recommendation_engine,
    benchmark_financial_risk_analysis,
    benchmark_content_discovery_platform,
    benchmark_scientific_literature_search
);

criterion_main!(industry_scenario_benches);