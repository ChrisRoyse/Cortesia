//! Comprehensive Mock Data Integration for Enhanced Knowledge Storage System
//! 
//! This module provides complete mock data sets, realistic document collections,
//! entity knowledge bases, relationship networks, performance data, and error scenarios
//! for thorough testing of the enhanced knowledge storage system.

use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use llmkg::enhanced_knowledge_storage::types::ComplexityLevel;

/// Result of processing a task (mock version)
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub task_id: String,
    pub processing_time: Duration,
    pub quality_score: f32,
    pub model_used: String,
    pub success: bool,
    pub output: String,
    pub metadata: ProcessingMetadata,
}

/// Additional metadata about processing (mock version)
#[derive(Debug, Clone, Default)]
pub struct ProcessingMetadata {
    pub memory_used: u64,
    pub cache_hit: bool,
    pub model_load_time: Option<Duration>,
    pub inference_time: Duration,
}

/// Document complexity levels for categorizing test documents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DocumentComplexity {
    Simple,
    Medium, 
    Complex,
    Technical,
}

/// Test document with expected processing outcomes
#[derive(Debug, Clone)]
pub struct TestDocument {
    pub title: String,
    pub content: String,
    pub expected_entities: Vec<String>,
    pub expected_relationships: Vec<(String, String, String)>, // (source, predicate, target)
    pub expected_chunks: usize,
    pub complexity_level: ComplexityLevel,
    pub expected_processing_time: Duration,
    pub expected_quality_score: f32,
    pub document_type: DocumentType,
}

/// Types of documents for categorized testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentType {
    Scientific,
    Technical,
    Narrative,
    Mixed,
    Temporal,
    Multilingual,
}

/// Comprehensive mock document collection
pub struct MockDocumentCollection {
    pub scientific_papers: Vec<TestDocument>,
    pub technical_documentation: Vec<TestDocument>,
    pub narrative_content: Vec<TestDocument>,
    pub mixed_content: Vec<TestDocument>,
    pub temporal_content: Vec<TestDocument>,
    pub multilingual_content: Vec<TestDocument>,
}

impl MockDocumentCollection {
    pub fn create_comprehensive_set() -> Self {
        Self {
            scientific_papers: Self::create_scientific_papers(),
            technical_documentation: Self::create_technical_docs(),
            narrative_content: Self::create_narrative_content(),
            mixed_content: Self::create_mixed_content(),
            temporal_content: Self::create_temporal_content(),
            multilingual_content: Self::create_multilingual_content(),
        }
    }

    fn create_scientific_papers() -> Vec<TestDocument> {
        vec![
            TestDocument {
                title: "Quantum Computing Advances in Machine Learning".to_string(),
                content: r#"Recent developments in quantum computing have shown remarkable potential for accelerating machine learning algorithms. Quantum neural networks leverage superposition and entanglement to process information in ways impossible for classical systems. The Variational Quantum Eigensolver (VQE) represents a breakthrough in hybrid classical-quantum algorithms, particularly for optimization problems in molecular simulation and drug discovery."#.to_string(),
                expected_entities: vec![
                    "quantum computing".to_string(),
                    "machine learning".to_string(),
                    "quantum neural networks".to_string(),
                    "superposition".to_string(),
                    "entanglement".to_string(),
                    "Variational Quantum Eigensolver".to_string(),
                    "VQE".to_string(),
                    "drug discovery".to_string(),
                ],
                expected_relationships: vec![
                    ("quantum computing".to_string(), "accelerates".to_string(), "machine learning".to_string()),
                    ("quantum neural networks".to_string(), "leverage".to_string(), "superposition".to_string()),
                    ("VQE".to_string(), "used_for".to_string(), "drug discovery".to_string()),
                ],
                expected_chunks: 3,
                complexity_level: ComplexityLevel::High,
                expected_processing_time: Duration::from_millis(2000),
                expected_quality_score: 0.92,
                document_type: DocumentType::Scientific,
            },
            TestDocument {
                title: "Neural Architecture Search and AutoML".to_string(),
                content: r#"Neural Architecture Search (NAS) automates the design of neural network architectures, reducing the need for human expertise in network design. Differentiable NAS approaches like DARTS (Differentiable Architecture Search) enable efficient search through continuous relaxation of the architecture search space. AutoML platforms integrate NAS with hyperparameter optimization and model selection to provide end-to-end machine learning pipelines."#.to_string(),
                expected_entities: vec![
                    "Neural Architecture Search".to_string(),
                    "NAS".to_string(),
                    "AutoML".to_string(),
                    "DARTS".to_string(),
                    "hyperparameter optimization".to_string(),
                ],
                expected_relationships: vec![
                    ("NAS".to_string(), "automates".to_string(), "network design".to_string()),
                    ("DARTS".to_string(), "is_type_of".to_string(), "NAS".to_string()),
                    ("AutoML".to_string(), "integrates".to_string(), "NAS".to_string()),
                ],
                expected_chunks: 2,
                complexity_level: ComplexityLevel::High,
                expected_processing_time: Duration::from_millis(1800),
                expected_quality_score: 0.89,
                document_type: DocumentType::Scientific,
            },
            TestDocument {
                title: "CRISPR Gene Editing and Therapeutic Applications".to_string(),
                content: r#"CRISPR-Cas9 technology has revolutionized gene editing with its precision and efficiency. The system uses guide RNAs to direct the Cas9 nuclease to specific DNA sequences for targeted modifications. Clinical trials have demonstrated success in treating sickle cell disease and beta-thalassemia. Prime editing and base editing represent newer CRISPR variants with reduced off-target effects and enhanced precision for therapeutic applications."#.to_string(),
                expected_entities: vec![
                    "CRISPR-Cas9".to_string(),
                    "gene editing".to_string(),
                    "guide RNAs".to_string(),
                    "Cas9 nuclease".to_string(),
                    "sickle cell disease".to_string(),
                    "beta-thalassemia".to_string(),
                    "prime editing".to_string(),
                    "base editing".to_string(),
                ],
                expected_relationships: vec![
                    ("CRISPR-Cas9".to_string(), "revolutionized".to_string(), "gene editing".to_string()),
                    ("guide RNAs".to_string(), "direct".to_string(), "Cas9 nuclease".to_string()),
                    ("CRISPR".to_string(), "treats".to_string(), "sickle cell disease".to_string()),
                ],
                expected_chunks: 3,
                complexity_level: ComplexityLevel::High,
                expected_processing_time: Duration::from_millis(1900),
                expected_quality_score: 0.91,
                document_type: DocumentType::Scientific,
            },
        ]
    }

    fn create_technical_docs() -> Vec<TestDocument> {
        vec![
            TestDocument {
                title: "REST API Design Best Practices".to_string(),
                content: r#"RESTful API design follows principles of statelessness, uniform interface, and resource-based URLs. HTTP methods (GET, POST, PUT, DELETE) map to CRUD operations. Status codes provide semantic meaning: 200 for success, 404 for not found, 500 for server errors. Authentication can be implemented using JWT tokens, OAuth 2.0, or API keys. Rate limiting prevents abuse and ensures fair usage across clients."#.to_string(),
                expected_entities: vec![
                    "REST API".to_string(),
                    "HTTP methods".to_string(),
                    "CRUD operations".to_string(),
                    "status codes".to_string(),
                    "JWT tokens".to_string(),
                    "OAuth 2.0".to_string(),
                    "API keys".to_string(),
                    "rate limiting".to_string(),
                ],
                expected_relationships: vec![
                    ("HTTP methods".to_string(), "map_to".to_string(), "CRUD operations".to_string()),
                    ("JWT tokens".to_string(), "used_for".to_string(), "authentication".to_string()),
                    ("rate limiting".to_string(), "prevents".to_string(), "abuse".to_string()),
                ],
                expected_chunks: 4,
                complexity_level: ComplexityLevel::Medium,
                expected_processing_time: Duration::from_millis(800),
                expected_quality_score: 0.87,
                document_type: DocumentType::Technical,
            },
            TestDocument {
                title: "Docker Container Orchestration with Kubernetes".to_string(),
                content: r#"Kubernetes orchestrates containerized applications across clusters of machines. Pods are the smallest deployable units, containing one or more containers. Services provide stable network endpoints for pods. Deployments manage pod lifecycle and scaling. ConfigMaps and Secrets handle configuration and sensitive data. Ingress controllers route external traffic to internal services. Helm charts package Kubernetes applications for easy deployment and management."#.to_string(),
                expected_entities: vec![
                    "Kubernetes".to_string(),
                    "Docker".to_string(),
                    "containers".to_string(),
                    "Pods".to_string(),
                    "Services".to_string(),
                    "Deployments".to_string(),
                    "ConfigMaps".to_string(),
                    "Secrets".to_string(),
                    "Helm charts".to_string(),
                ],
                expected_relationships: vec![
                    ("Kubernetes".to_string(), "orchestrates".to_string(), "containers".to_string()),
                    ("Pods".to_string(), "contain".to_string(), "containers".to_string()),
                    ("Services".to_string(), "provide_endpoints_for".to_string(), "Pods".to_string()),
                ],
                expected_chunks: 5,
                complexity_level: ComplexityLevel::Medium,
                expected_processing_time: Duration::from_millis(900),
                expected_quality_score: 0.85,
                document_type: DocumentType::Technical,
            },
        ]
    }

    fn create_narrative_content() -> Vec<TestDocument> {
        vec![
            TestDocument {
                title: "The Innovation Journey of Steve Jobs".to_string(),
                content: r#"Steve Jobs co-founded Apple Computer in 1976 with Steve Wozniak in a garage in Los Altos, California. After being ousted from Apple in 1985, Jobs founded NeXT Computer and acquired Pixar Animation Studios. He returned to Apple in 1997 as interim CEO, leading the company's renaissance with the iMac, iPod, iPhone, and iPad. Jobs' design philosophy emphasized simplicity, elegance, and user experience, transforming multiple industries including personal computing, music, telecommunications, and tablet computing."#.to_string(),
                expected_entities: vec![
                    "Steve Jobs".to_string(),
                    "Steve Wozniak".to_string(),
                    "Apple Computer".to_string(),
                    "NeXT Computer".to_string(),
                    "Pixar Animation Studios".to_string(),
                    "iMac".to_string(),
                    "iPod".to_string(),
                    "iPhone".to_string(),
                    "iPad".to_string(),
                ],
                expected_relationships: vec![
                    ("Steve Jobs".to_string(), "co-founded".to_string(), "Apple Computer".to_string()),
                    ("Steve Jobs".to_string(), "founded".to_string(), "NeXT Computer".to_string()),
                    ("Steve Jobs".to_string(), "acquired".to_string(), "Pixar Animation Studios".to_string()),
                ],
                expected_chunks: 3,
                complexity_level: ComplexityLevel::Medium,
                expected_processing_time: Duration::from_millis(750),
                expected_quality_score: 0.88,
                document_type: DocumentType::Narrative,
            },
        ]
    }

    fn create_mixed_content() -> Vec<TestDocument> {
        vec![
            TestDocument {
                title: "Climate Change: Science, Technology, and Policy".to_string(),
                content: r#"Climate change represents one of the most pressing challenges of the 21st century. Scientific consensus, based on decades of research and data from institutions like NOAA and NASA, confirms that human activities are the primary driver of recent warming. Carbon dioxide levels have increased from 280 ppm in pre-industrial times to over 410 ppm today. Renewable energy technologies, including solar photovoltaics and wind turbines, have achieved cost parity with fossil fuels in many markets. The Paris Agreement, signed by 196 countries, aims to limit global warming to 1.5°C above pre-industrial levels."#.to_string(),
                expected_entities: vec![
                    "climate change".to_string(),
                    "NOAA".to_string(),
                    "NASA".to_string(),
                    "carbon dioxide".to_string(),
                    "solar photovoltaics".to_string(),
                    "wind turbines".to_string(),
                    "Paris Agreement".to_string(),
                ],
                expected_relationships: vec![
                    ("human activities".to_string(), "cause".to_string(), "climate change".to_string()),
                    ("renewable energy".to_string(), "includes".to_string(), "solar photovoltaics".to_string()),
                    ("Paris Agreement".to_string(), "aims_to_limit".to_string(), "global warming".to_string()),
                ],
                expected_chunks: 4,
                complexity_level: ComplexityLevel::High,
                expected_processing_time: Duration::from_millis(1500),
                expected_quality_score: 0.90,
                document_type: DocumentType::Mixed,
            },
        ]
    }

    fn create_temporal_content() -> Vec<TestDocument> {
        vec![
            TestDocument {
                title: "The Timeline of Artificial Intelligence Development".to_string(),
                content: r#"1943: Warren McCulloch and Walter Pitts published the first mathematical model of neural networks. 1950: Alan Turing proposed the Turing Test in his paper "Computing Machinery and Intelligence." 1956: The Dartmouth Conference, organized by John McCarthy, officially coined the term "artificial intelligence." 1969: The first AI winter began due to overinflated expectations. 1997: IBM's Deep Blue defeated chess champion Garry Kasparov. 2012: AlexNet won the ImageNet competition, sparking the deep learning revolution. 2016: Google's AlphaGo defeated world Go champion Lee Sedol. 2022: OpenAI released ChatGPT, bringing AI to mainstream consumer adoption."#.to_string(),
                expected_entities: vec![
                    "Warren McCulloch".to_string(),
                    "Walter Pitts".to_string(),
                    "Alan Turing".to_string(),
                    "Turing Test".to_string(),
                    "John McCarthy".to_string(),
                    "Dartmouth Conference".to_string(),
                    "Deep Blue".to_string(),
                    "Garry Kasparov".to_string(),
                    "AlexNet".to_string(),
                    "AlphaGo".to_string(),
                    "Lee Sedol".to_string(),
                    "ChatGPT".to_string(),
                ],
                expected_relationships: vec![
                    ("McCulloch and Pitts".to_string(), "published".to_string(), "neural network model".to_string()),
                    ("Alan Turing".to_string(), "proposed".to_string(), "Turing Test".to_string()),
                    ("Deep Blue".to_string(), "defeated".to_string(), "Garry Kasparov".to_string()),
                ],
                expected_chunks: 6,
                complexity_level: ComplexityLevel::High,
                expected_processing_time: Duration::from_millis(1700),
                expected_quality_score: 0.93,
                document_type: DocumentType::Temporal,
            },
        ]
    }

    fn create_multilingual_content() -> Vec<TestDocument> {
        vec![
            TestDocument {
                title: "Global Perspectives on Innovation".to_string(),
                content: r#"Innovation transcends borders and languages. In Silicon Valley, English dominates tech discourse. In Shenzhen (深圳), Chinese companies like Tencent (腾讯) and Huawei (华为) drive technological advancement. German engineering excellence is exemplified by companies like Siemens and BMW in München. Japanese precision manufacturing, known as "monozukuri" (ものづくり), continues to influence global production standards. French luxury brands maintain their prestige through heritage and craftsmanship."#.to_string(),
                expected_entities: vec![
                    "Silicon Valley".to_string(),
                    "Shenzhen".to_string(),
                    "深圳".to_string(),
                    "Tencent".to_string(),
                    "腾讯".to_string(),
                    "Huawei".to_string(),
                    "华为".to_string(),
                    "Siemens".to_string(),
                    "BMW".to_string(),
                    "München".to_string(),
                    "monozukuri".to_string(),
                    "ものづくり".to_string(),
                ],
                expected_relationships: vec![
                    ("Tencent".to_string(), "located_in".to_string(), "Shenzhen".to_string()),
                    ("BMW".to_string(), "located_in".to_string(), "München".to_string()),
                    ("monozukuri".to_string(), "influences".to_string(), "production standards".to_string()),
                ],
                expected_chunks: 4,
                complexity_level: ComplexityLevel::Medium,
                expected_processing_time: Duration::from_millis(1000),
                expected_quality_score: 0.82,
                document_type: DocumentType::Multilingual,
            },
        ]
    }

    pub fn get_all_documents(&self) -> Vec<&TestDocument> {
        let mut all_docs = Vec::new();
        all_docs.extend(&self.scientific_papers);
        all_docs.extend(&self.technical_documentation);
        all_docs.extend(&self.narrative_content);
        all_docs.extend(&self.mixed_content);
        all_docs.extend(&self.temporal_content);
        all_docs.extend(&self.multilingual_content);
        all_docs
    }

    pub fn get_documents_by_complexity(&self, complexity: ComplexityLevel) -> Vec<&TestDocument> {
        self.get_all_documents()
            .into_iter()
            .filter(|doc| doc.complexity_level == complexity)
            .collect()
    }

    pub fn get_documents_by_type(&self, doc_type: DocumentType) -> Vec<&TestDocument> {
        self.get_all_documents()
            .into_iter()
            .filter(|doc| doc.document_type == doc_type)
            .collect()
    }
}

/// Entity types for knowledge base organization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Concept,
    Technology,
    Event,
    Product,
}

/// Comprehensive entity with rich metadata
#[derive(Debug, Clone)]
pub struct MockEntity {
    pub name: String,
    pub entity_type: EntityType,
    pub aliases: Vec<String>,
    pub description: String,
    pub properties: HashMap<String, String>,
    pub confidence: f32,
    pub created_at: Instant,
}

/// Relationship types for comprehensive relationship modeling
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationshipType {
    // Structural relationships
    IsA,
    PartOf,
    Contains,
    LocatedIn,
    
    // Temporal relationships
    Precedes,
    Follows,
    CausedBy,
    ResultsIn,
    
    // Functional relationships
    CreatedBy,
    FoundedBy,
    WorksFor,
    CompetesWith,
    CollaboratesWith,
    
    // Technical relationships
    ImplementsUsing,
    BasedOn,
    EnhancedBy,
    ReplacedBy,
    
    // Custom relationships
    Custom(String),
}

/// Complex relationship with metadata
#[derive(Debug, Clone)]
pub struct ComplexRelationship {
    pub source: String,
    pub predicate: RelationshipType,
    pub target: String,
    pub confidence: f32,
    pub supporting_evidence: Vec<String>,
    pub temporal_context: Option<String>,
    pub created_at: Instant,
}

/// Temporal relationship with time bounds
#[derive(Debug, Clone)]
pub struct TemporalRelationship {
    pub relationship: ComplexRelationship,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub duration: Option<Duration>,
}

/// Causal relationship with strength
#[derive(Debug, Clone)]
pub struct CausalRelationship {
    pub cause: String,
    pub effect: String,
    pub causal_strength: f32,
    pub mechanism: Option<String>,
    pub supporting_evidence: Vec<String>,
}

/// Comprehensive entity knowledge base
pub struct MockEntityKnowledgeBase {
    pub persons: HashMap<String, MockEntity>,
    pub organizations: HashMap<String, MockEntity>,
    pub locations: HashMap<String, MockEntity>,
    pub concepts: HashMap<String, MockEntity>,
    pub technologies: HashMap<String, MockEntity>,
    pub events: HashMap<String, MockEntity>,
    pub products: HashMap<String, MockEntity>,
}

impl MockEntityKnowledgeBase {
    pub fn create_comprehensive_kb() -> Self {
        let mut kb = Self {
            persons: HashMap::new(),
            organizations: HashMap::new(),
            locations: HashMap::new(),
            concepts: HashMap::new(),
            technologies: HashMap::new(),
            events: HashMap::new(),
            products: HashMap::new(),
        };

        // Add famous scientists and technologists
        kb.add_person("Albert Einstein", vec!["Einstein"], "Theoretical physicist known for the theory of relativity", vec![
            ("birth_year", "1879"),
            ("death_year", "1955"),
            ("nationality", "German-American"),
            ("field", "Physics"),
            ("nobel_prize", "1921"),
        ]);

        kb.add_person("Steve Jobs", vec!["Jobs"], "Co-founder of Apple Inc., visionary entrepreneur", vec![
            ("birth_year", "1955"),
            ("death_year", "2011"),
            ("nationality", "American"),
            ("company", "Apple Inc."),
            ("role", "CEO"),
        ]);

        kb.add_person("Alan Turing", vec!["Turing"], "Computer scientist and mathematician, father of computer science", vec![
            ("birth_year", "1912"),
            ("death_year", "1954"),
            ("nationality", "British"),
            ("field", "Computer Science"),
            ("famous_for", "Turing Test"),
        ]);

        // Add major organizations
        kb.add_organization("Apple Inc.", vec!["Apple"], "Technology company specializing in consumer electronics", vec![
            ("founded", "1976"),
            ("founders", "Steve Jobs, Steve Wozniak, Ronald Wayne"),
            ("headquarters", "Cupertino, California"),
            ("industry", "Technology"),
        ]);

        kb.add_organization("Google", vec!["Alphabet Inc."], "Multinational technology company specializing in search and AI", vec![
            ("founded", "1998"),
            ("founders", "Larry Page, Sergey Brin"),
            ("headquarters", "Mountain View, California"),
            ("industry", "Technology"),
        ]);

        // Add key concepts
        kb.add_concept("Artificial Intelligence", vec!["AI"], "Intelligence demonstrated by machines", vec![
            ("field", "Computer Science"),
            ("emerged", "1950s"),
            ("applications", "Machine Learning, Natural Language Processing"),
        ]);

        kb.add_concept("Quantum Computing", vec!["Quantum Information Processing"], "Computing using quantum-mechanical phenomena", vec![
            ("field", "Quantum Physics"),
            ("key_concepts", "Superposition, Entanglement"),
            ("applications", "Cryptography, Optimization"),
        ]);

        // Add technologies
        kb.add_technology("CRISPR", vec!["CRISPR-Cas9"], "Gene editing technology", vec![
            ("invented", "2012"),
            ("inventors", "Jennifer Doudna, Emmanuelle Charpentier"),
            ("applications", "Gene therapy, Agriculture"),
        ]);

        // Add locations
        kb.add_location("Silicon Valley", vec!["Silicon Valley, CA"], "Technology hub in California", vec![
            ("state", "California"),
            ("country", "United States"),
            ("known_for", "Technology companies"),
        ]);

        kb
    }

    fn add_person(&mut self, name: &str, aliases: Vec<&str>, description: &str, properties: Vec<(&str, &str)>) {
        let entity = MockEntity {
            name: name.to_string(),
            entity_type: EntityType::Person,
            aliases: aliases.into_iter().map(|s| s.to_string()).collect(),
            description: description.to_string(),
            properties: properties.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            confidence: 0.95,
            created_at: Instant::now(),
        };
        self.persons.insert(name.to_string(), entity);
    }

    fn add_organization(&mut self, name: &str, aliases: Vec<&str>, description: &str, properties: Vec<(&str, &str)>) {
        let entity = MockEntity {
            name: name.to_string(),
            entity_type: EntityType::Organization,
            aliases: aliases.into_iter().map(|s| s.to_string()).collect(),
            description: description.to_string(),
            properties: properties.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            confidence: 0.92,
            created_at: Instant::now(),
        };
        self.organizations.insert(name.to_string(), entity);
    }

    fn add_concept(&mut self, name: &str, aliases: Vec<&str>, description: &str, properties: Vec<(&str, &str)>) {
        let entity = MockEntity {
            name: name.to_string(),
            entity_type: EntityType::Concept,
            aliases: aliases.into_iter().map(|s| s.to_string()).collect(),
            description: description.to_string(),
            properties: properties.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            confidence: 0.88,
            created_at: Instant::now(),
        };
        self.concepts.insert(name.to_string(), entity);
    }

    fn add_technology(&mut self, name: &str, aliases: Vec<&str>, description: &str, properties: Vec<(&str, &str)>) {
        let entity = MockEntity {
            name: name.to_string(),
            entity_type: EntityType::Technology,
            aliases: aliases.into_iter().map(|s| s.to_string()).collect(),
            description: description.to_string(),
            properties: properties.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            confidence: 0.90,
            created_at: Instant::now(),
        };
        self.technologies.insert(name.to_string(), entity);
    }

    fn add_location(&mut self, name: &str, aliases: Vec<&str>, description: &str, properties: Vec<(&str, &str)>) {
        let entity = MockEntity {
            name: name.to_string(),
            entity_type: EntityType::Location,
            aliases: aliases.into_iter().map(|s| s.to_string()).collect(),
            description: description.to_string(),
            properties: properties.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            confidence: 0.95,
            created_at: Instant::now(),
        };
        self.locations.insert(name.to_string(), entity);
    }

    pub fn get_all_entities(&self) -> Vec<&MockEntity> {
        let mut entities = Vec::new();
        entities.extend(self.persons.values());
        entities.extend(self.organizations.values());
        entities.extend(self.locations.values());
        entities.extend(self.concepts.values());
        entities.extend(self.technologies.values());
        entities.extend(self.events.values());
        entities.extend(self.products.values());
        entities
    }

    pub fn get_entities_by_type(&self, entity_type: EntityType) -> Vec<&MockEntity> {
        match entity_type {
            EntityType::Person => self.persons.values().collect(),
            EntityType::Organization => self.organizations.values().collect(),
            EntityType::Location => self.locations.values().collect(),
            EntityType::Concept => self.concepts.values().collect(),
            EntityType::Technology => self.technologies.values().collect(),
            EntityType::Event => self.events.values().collect(),
            EntityType::Product => self.products.values().collect(),
        }
    }
}

/// Mock relationship network with complex multi-hop connections
pub struct MockRelationshipNetwork {
    pub scientific_relationships: Vec<ComplexRelationship>,
    pub temporal_relationships: Vec<TemporalRelationship>,
    pub causal_relationships: Vec<CausalRelationship>,
    pub organizational_relationships: Vec<ComplexRelationship>,
    pub technical_relationships: Vec<ComplexRelationship>,
}

impl MockRelationshipNetwork {
    pub fn create_multi_hop_network() -> Self {
        let mut network = Self {
            scientific_relationships: Vec::new(),
            temporal_relationships: Vec::new(),
            causal_relationships: Vec::new(),
            organizational_relationships: Vec::new(),
            technical_relationships: Vec::new(),
        };

        // Create multi-hop scientific relationships
        // Einstein → Relativity → GPS → Satellites
        network.scientific_relationships.push(ComplexRelationship {
            source: "Albert Einstein".to_string(),
            predicate: RelationshipType::CreatedBy,
            target: "Theory of Relativity".to_string(),
            confidence: 0.98,
            supporting_evidence: vec!["Published in 1905 and 1915".to_string(), "Revolutionized physics".to_string()],
            temporal_context: Some("1905-1915".to_string()),
            created_at: Instant::now(),
        });

        network.technical_relationships.push(ComplexRelationship {
            source: "Theory of Relativity".to_string(),
            predicate: RelationshipType::EnhancedBy,
            target: "GPS Technology".to_string(),
            confidence: 0.87,
            supporting_evidence: vec!["Time dilation corrections required for accuracy".to_string()],
            temporal_context: Some("1970s-present".to_string()),
            created_at: Instant::now(),
        });

        // Jobs → Apple → iPhone → Mobile Revolution
        network.organizational_relationships.push(ComplexRelationship {
            source: "Steve Jobs".to_string(),
            predicate: RelationshipType::FoundedBy,
            target: "Apple Inc.".to_string(),
            confidence: 0.95,
            supporting_evidence: vec!["Co-founded in 1976".to_string()],
            temporal_context: Some("1976".to_string()),
            created_at: Instant::now(),
        });

        network.technical_relationships.push(ComplexRelationship {
            source: "Apple Inc.".to_string(),
            predicate: RelationshipType::CreatedBy,
            target: "iPhone".to_string(),
            confidence: 0.98,
            supporting_evidence: vec!["Launched in 2007".to_string(), "Revolutionized smartphones".to_string()],
            temporal_context: Some("2007".to_string()),
            created_at: Instant::now(),
        });

        network.causal_relationships.push(CausalRelationship {
            cause: "iPhone".to_string(),
            effect: "Mobile Revolution".to_string(),
            causal_strength: 0.92,
            mechanism: Some("Touchscreen interface and app ecosystem".to_string()),
            supporting_evidence: vec!["Changed entire mobile industry".to_string(), "Created app economy".to_string()],
        });

        // Turing → Computer Science → AI → Machine Learning
        network.scientific_relationships.push(ComplexRelationship {
            source: "Alan Turing".to_string(),
            predicate: RelationshipType::CreatedBy,
            target: "Computer Science".to_string(),
            confidence: 0.93,
            supporting_evidence: vec!["Turing machine concept".to_string(), "Computational theory".to_string()],
            temporal_context: Some("1936-1950".to_string()),
            created_at: Instant::now(),
        });

        network.technical_relationships.push(ComplexRelationship {
            source: "Computer Science".to_string(),
            predicate: RelationshipType::Contains,
            target: "Artificial Intelligence".to_string(),
            confidence: 0.90,
            supporting_evidence: vec!["AI is a subfield of CS".to_string()],
            temporal_context: Some("1950s-present".to_string()),
            created_at: Instant::now(),
        });

        network.technical_relationships.push(ComplexRelationship {
            source: "Artificial Intelligence".to_string(),
            predicate: RelationshipType::Contains,
            target: "Machine Learning".to_string(),
            confidence: 0.95,
            supporting_evidence: vec!["ML is a subset of AI".to_string()],
            temporal_context: Some("1959-present".to_string()),
            created_at: Instant::now(),
        });

        // Add temporal relationships
        network.temporal_relationships.push(TemporalRelationship {
            relationship: ComplexRelationship {
                source: "Dartmouth Conference".to_string(),
                predicate: RelationshipType::Precedes,
                target: "AI Winter".to_string(),
                confidence: 0.85,
                supporting_evidence: vec!["AI hype followed by disappointment".to_string()],
                temporal_context: Some("1956-1974".to_string()),
                created_at: Instant::now(),
            },
            start_time: Some("1956".to_string()),
            end_time: Some("1974".to_string()),
            duration: Some(Duration::from_secs(18 * 365 * 24 * 3600)), // 18 years
        });

        network
    }

    pub fn get_all_relationships(&self) -> Vec<&ComplexRelationship> {
        let mut relationships = Vec::new();
        relationships.extend(&self.scientific_relationships);
        relationships.extend(&self.organizational_relationships);
        relationships.extend(&self.technical_relationships);
        relationships.extend(self.temporal_relationships.iter().map(|tr| &tr.relationship));
        relationships
    }

    pub fn find_multi_hop_path(&self, start: &str, end: &str, max_hops: usize) -> Vec<Vec<&ComplexRelationship>> {
        // Simple BFS implementation for finding paths
        let mut paths = Vec::new();
        let all_rels = self.get_all_relationships();
        
        // This is a simplified implementation - in reality would use proper graph traversal
        for rel in &all_rels {
            if rel.source == start {
                if rel.target == end {
                    paths.push(vec![*rel]);
                } else if max_hops > 1 {
                    // Find next hop relationships
                    for next_rel in &all_rels {
                        if next_rel.source == rel.target && next_rel.target == end {
                            paths.push(vec![*rel, *next_rel]);
                        }
                    }
                }
            }
        }
        
        paths
    }
}

/// Mock performance data for realistic benchmarking
pub struct MockPerformanceData {
    pub processing_times: HashMap<ComplexityLevel, Duration>,
    pub memory_usage: HashMap<String, u64>,
    pub accuracy_metrics: HashMap<String, f32>,
    pub throughput_metrics: HashMap<String, f32>,
    pub model_loading_times: HashMap<String, Duration>,
}

impl MockPerformanceData {
    pub fn create_realistic_benchmarks() -> Self {
        Self {
            processing_times: [
                (ComplexityLevel::Low, Duration::from_millis(250)),
                (ComplexityLevel::Medium, Duration::from_millis(750)),
                (ComplexityLevel::High, Duration::from_secs(2)),
            ].into_iter().collect(),
            
            memory_usage: [
                ("smollm2_135m".to_string(), 270_000_000), // 270MB
                ("smollm2_360m".to_string(), 720_000_000), // 720MB
                ("smollm2_1_7b".to_string(), 3_400_000_000), // 3.4GB
                ("system_overhead".to_string(), 50_000_000), // 50MB
            ].into_iter().collect(),
            
            accuracy_metrics: [
                ("entity_extraction".to_string(), 0.87),
                ("relationship_mapping".to_string(), 0.82),
                ("semantic_chunking".to_string(), 0.91),
                ("multi_hop_reasoning".to_string(), 0.79),
                ("temporal_reasoning".to_string(), 0.74),
            ].into_iter().collect(),
            
            throughput_metrics: [
                ("documents_per_minute".to_string(), 12.5),
                ("entities_per_second".to_string(), 45.8),
                ("relationships_per_second".to_string(), 23.2),
            ].into_iter().collect(),
            
            model_loading_times: [
                ("smollm2_135m".to_string(), Duration::from_secs(8)),
                ("smollm2_360m".to_string(), Duration::from_secs(15)),
                ("smollm2_1_7b".to_string(), Duration::from_secs(45)),
            ].into_iter().collect(),
        }
    }

    pub fn get_expected_processing_time(&self, complexity: ComplexityLevel) -> Duration {
        self.processing_times.get(&complexity).copied().unwrap_or(Duration::from_secs(1))
    }

    pub fn get_model_memory_usage(&self, model_id: &str) -> u64 {
        self.memory_usage.get(model_id).copied().unwrap_or(100_000_000)
    }

    pub fn get_accuracy_metric(&self, metric_name: &str) -> f32 {
        self.accuracy_metrics.get(metric_name).copied().unwrap_or(0.8)
    }
}

/// Error scenarios for comprehensive testing
#[derive(Debug, Clone)]
pub struct ErrorScenario {
    pub name: String,
    pub description: String,
    pub setup: fn() -> Result<(), String>,
    pub expected_error: ProcessingError,
    pub recovery_strategy: RecoveryStrategy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingError {
    InsufficientMemory,
    ModelLoadingTimeout,
    ProcessingTimeout,
    MalformedInput,
    NetworkError,
    CacheCorruption,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    EvictOldestModel,
    RetryWithSmallerModel,
    SkipAndContinue,
    ResetAndRestart,
    FallbackToCache,
}

/// Mock error scenarios for testing system resilience
pub struct MockErrorScenarios {
    pub memory_pressure_scenarios: Vec<ErrorScenario>,
    pub processing_timeout_scenarios: Vec<ErrorScenario>,
    pub malformed_input_scenarios: Vec<ErrorScenario>,
    pub network_error_scenarios: Vec<ErrorScenario>,
}

impl MockErrorScenarios {
    pub fn create_comprehensive_error_set() -> Self {
        Self {
            memory_pressure_scenarios: vec![
                ErrorScenario {
                    name: "Multiple Large Models".to_string(),
                    description: "Loading multiple large models simultaneously".to_string(),
                    setup: || {
                        // Mock setup function
                        Ok(())
                    },
                    expected_error: ProcessingError::InsufficientMemory,
                    recovery_strategy: RecoveryStrategy::EvictOldestModel,
                },
                ErrorScenario {
                    name: "Memory Fragmentation".to_string(),
                    description: "System memory becomes fragmented".to_string(),
                    setup: || {
                        Ok(())
                    },
                    expected_error: ProcessingError::InsufficientMemory,
                    recovery_strategy: RecoveryStrategy::ResetAndRestart,
                },
            ],
            processing_timeout_scenarios: vec![
                ErrorScenario {
                    name: "Complex Document Timeout".to_string(),
                    description: "Processing very complex document exceeds timeout".to_string(),
                    setup: || {
                        Ok(())
                    },
                    expected_error: ProcessingError::ProcessingTimeout,
                    recovery_strategy: RecoveryStrategy::RetryWithSmallerModel,
                },
            ],
            malformed_input_scenarios: vec![
                ErrorScenario {
                    name: "Invalid UTF-8".to_string(),
                    description: "Document contains invalid UTF-8 sequences".to_string(),
                    setup: || {
                        Ok(())
                    },
                    expected_error: ProcessingError::MalformedInput,
                    recovery_strategy: RecoveryStrategy::SkipAndContinue,
                },
            ],
            network_error_scenarios: vec![
                ErrorScenario {
                    name: "Model Loading Failure".to_string(),
                    description: "Failed to load model from local model_weights directory".to_string(),
                    setup: || {
                        Ok(())
                    },
                    expected_error: ProcessingError::NetworkError,
                    recovery_strategy: RecoveryStrategy::FallbackToCache,
                },
            ],
        }
    }

    pub fn get_all_scenarios(&self) -> Vec<&ErrorScenario> {
        let mut scenarios = Vec::new();
        scenarios.extend(&self.memory_pressure_scenarios);
        scenarios.extend(&self.processing_timeout_scenarios);
        scenarios.extend(&self.malformed_input_scenarios);
        scenarios.extend(&self.network_error_scenarios);
        scenarios
    }
}

/// Comprehensive mock system that integrates all components
pub struct MockEnhancedKnowledgeSystem {
    pub documents: MockDocumentCollection,
    pub entities: MockEntityKnowledgeBase,
    pub relationships: MockRelationshipNetwork,
    pub performance_data: MockPerformanceData,
    pub error_scenarios: MockErrorScenarios,
}

impl MockEnhancedKnowledgeSystem {
    pub fn new() -> Self {
        Self {
            documents: MockDocumentCollection::create_comprehensive_set(),
            entities: MockEntityKnowledgeBase::create_comprehensive_kb(),
            relationships: MockRelationshipNetwork::create_multi_hop_network(),
            performance_data: MockPerformanceData::create_realistic_benchmarks(),
            error_scenarios: MockErrorScenarios::create_comprehensive_error_set(),
        }
    }

    pub fn with_documents(mut self, documents: MockDocumentCollection) -> Self {
        self.documents = documents;
        self
    }

    pub fn with_entities(mut self, entities: MockEntityKnowledgeBase) -> Self {
        self.entities = entities;
        self
    }

    pub fn with_relationships(mut self, relationships: MockRelationshipNetwork) -> Self {
        self.relationships = relationships;
        self
    }

    pub fn with_performance_data(mut self, performance_data: MockPerformanceData) -> Self {
        self.performance_data = performance_data;
        self
    }

    /// Mock processing of a document type
    pub async fn process_document_type(&self, doc_type: DocumentType) -> Result<ProcessingResult, ProcessingError> {
        let documents = self.documents.get_documents_by_type(doc_type);
        if documents.is_empty() {
            return Err(ProcessingError::MalformedInput);
        }

        let doc = documents[0];
        let processing_time = self.performance_data.get_expected_processing_time(doc.complexity_level);
        
        // Simulate processing delay
        tokio::time::sleep(Duration::from_millis(10)).await; // Shortened for tests

        Ok(ProcessingResult {
            task_id: "mock_task_123".to_string(),
            processing_time,
            quality_score: doc.expected_quality_score,
            model_used: format!("smollm2_{}", match doc.complexity_level {
                ComplexityLevel::Low => "135m",
                ComplexityLevel::Medium => "360m", 
                ComplexityLevel::High => "1_7b",
            }),
            success: true,
            output: format!("Processed {} document: {}", doc_type_to_string(doc_type), doc.title),
            metadata: ProcessingMetadata {
                memory_used: self.performance_data.get_model_memory_usage("smollm2_135m"),
                cache_hit: false,
                model_load_time: Some(Duration::from_secs(8)),
                inference_time: processing_time,
            },
        })
    }

    /// Mock multi-hop reasoning query
    pub async fn perform_multi_hop_query(&self, query: &str) -> Result<MultiHopResult, ProcessingError> {
        // Simple mock implementation
        let paths = self.relationships.find_multi_hop_path("Albert Einstein", "GPS Technology", 3);
        
        Ok(MultiHopResult {
            query: query.to_string(),
            hops: paths.first().map(|p| p.len()).unwrap_or(0),
            confidence: 0.78,
            reasoning_path: paths.first().map(|p| {
                p.iter().map(|rel| format!("{} -> {} -> {}", rel.source, rel.predicate_string(), rel.target)).collect::<Vec<_>>()
            }).unwrap_or_default(),
            processing_time: Duration::from_millis(500),
        })
    }

    /// Mock concurrent document processing
    pub async fn process_concurrent_documents(&self, count: usize) -> Result<LoadTestResult, ProcessingError> {
        let all_docs = self.documents.get_all_documents();
        let mut success_count = 0;
        let mut total_time = Duration::new(0, 0);

        for i in 0..count {
            let doc = all_docs[i % all_docs.len()];
            let processing_time = self.performance_data.get_expected_processing_time(doc.complexity_level);
            
            // Simulate some failures under load
            if i % 20 == 19 { // 5% failure rate
                continue;
            }
            
            success_count += 1;
            total_time += processing_time;
            
            // Simulate brief processing delay
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        Ok(LoadTestResult {
            total_documents: count,
            successful_documents: success_count,
            failed_documents: count - success_count,
            success_rate: success_count as f32 / count as f32,
            total_processing_time: total_time,
            average_processing_time: total_time / success_count as u32,
        })
    }
}

impl Default for MockEnhancedKnowledgeSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of multi-hop reasoning query
#[derive(Debug, Clone)]
pub struct MultiHopResult {
    pub query: String,
    pub hops: usize,
    pub confidence: f32,
    pub reasoning_path: Vec<String>,
    pub processing_time: Duration,
}

/// Result of load testing
#[derive(Debug, Clone)]
pub struct LoadTestResult {
    pub total_documents: usize,
    pub successful_documents: usize,
    pub failed_documents: usize,
    pub success_rate: f32,
    pub total_processing_time: Duration,
    pub average_processing_time: Duration,
}

// Utility functions
fn doc_type_to_string(doc_type: DocumentType) -> &'static str {
    match doc_type {
        DocumentType::Scientific => "Scientific",
        DocumentType::Technical => "Technical",
        DocumentType::Narrative => "Narrative",
        DocumentType::Mixed => "Mixed",
        DocumentType::Temporal => "Temporal",
        DocumentType::Multilingual => "Multilingual",
    }
}

impl RelationshipType {
    fn predicate_string(&self) -> String {
        match self {
            RelationshipType::IsA => "is_a".to_string(),
            RelationshipType::PartOf => "part_of".to_string(),
            RelationshipType::Contains => "contains".to_string(),
            RelationshipType::LocatedIn => "located_in".to_string(),
            RelationshipType::Precedes => "precedes".to_string(),
            RelationshipType::Follows => "follows".to_string(),
            RelationshipType::CausedBy => "caused_by".to_string(),
            RelationshipType::ResultsIn => "results_in".to_string(),
            RelationshipType::CreatedBy => "created_by".to_string(),
            RelationshipType::FoundedBy => "founded_by".to_string(),
            RelationshipType::WorksFor => "works_for".to_string(),
            RelationshipType::CompetesWith => "competes_with".to_string(),
            RelationshipType::CollaboratesWith => "collaborates_with".to_string(),
            RelationshipType::ImplementsUsing => "implements_using".to_string(),
            RelationshipType::BasedOn => "based_on".to_string(),
            RelationshipType::EnhancedBy => "enhanced_by".to_string(),
            RelationshipType::ReplacedBy => "replaced_by".to_string(),
            RelationshipType::Custom(s) => s.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_document_collection_creation() {
        let collection = MockDocumentCollection::create_comprehensive_set();
        
        assert!(!collection.scientific_papers.is_empty());
        assert!(!collection.technical_documentation.is_empty());
        assert!(!collection.narrative_content.is_empty());
        
        // Test filtering by complexity
        let high_complexity_docs = collection.get_documents_by_complexity(ComplexityLevel::High);
        assert!(!high_complexity_docs.is_empty());
        
        // Test filtering by type
        let scientific_docs = collection.get_documents_by_type(DocumentType::Scientific);
        assert!(!scientific_docs.is_empty());
    }

    #[test]
    fn test_entity_knowledge_base_creation() {
        let kb = MockEntityKnowledgeBase::create_comprehensive_kb();
        
        assert!(!kb.persons.is_empty());
        assert!(!kb.organizations.is_empty());  
        assert!(!kb.concepts.is_empty());
        
        // Test entity retrieval
        let einstein = kb.persons.get("Albert Einstein");
        assert!(einstein.is_some());
        assert_eq!(einstein.unwrap().entity_type, EntityType::Person);
        
        // Test entity filtering
        let people = kb.get_entities_by_type(EntityType::Person);
        assert!(!people.is_empty());
    }

    #[test]
    fn test_relationship_network_creation() {
        let network = MockRelationshipNetwork::create_multi_hop_network();
        
        assert!(!network.scientific_relationships.is_empty());
        assert!(!network.technical_relationships.is_empty());
        
        // Test multi-hop path finding
        let paths = network.find_multi_hop_path("Albert Einstein", "GPS Technology", 3);
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_performance_data_creation() {
        let perf_data = MockPerformanceData::create_realistic_benchmarks();
        
        assert!(perf_data.get_expected_processing_time(ComplexityLevel::Low) < 
               perf_data.get_expected_processing_time(ComplexityLevel::High));
               
        assert!(perf_data.get_model_memory_usage("smollm2_135m") < 
               perf_data.get_model_memory_usage("smollm2_1_7b"));
               
        assert!(perf_data.get_accuracy_metric("entity_extraction") > 0.0);
    }

    #[tokio::test]
    async fn test_mock_system_integration() {
        let system = MockEnhancedKnowledgeSystem::new();
        
        // Test document processing
        let result = system.process_document_type(DocumentType::Scientific).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.quality_score > 0.8);
        
        // Test multi-hop reasoning
        let reasoning_result = system.perform_multi_hop_query("How did Einstein influence GPS?").await;
        assert!(reasoning_result.is_ok());
        let reasoning_result = reasoning_result.unwrap();
        assert!(reasoning_result.confidence > 0.7);
        
        // Test load testing
        let load_result = system.process_concurrent_documents(10).await;
        assert!(load_result.is_ok());
        let load_result = load_result.unwrap();
        assert!(load_result.success_rate > 0.8);
    }
}