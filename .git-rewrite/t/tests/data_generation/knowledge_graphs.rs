//! Knowledge Graph Generation
//! 
//! Provides generation of domain-specific knowledge graphs with realistic structures.

use crate::infrastructure::deterministic_rng::DeterministicRng;
use crate::data_generation::graph_topologies::{TestGraph, TestEntity, TestEdge, GraphProperties, ConnectivityType};
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Ontology definition for knowledge graph generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ontology {
    pub entity_types: Vec<EntityType>,
    pub relationship_types: Vec<RelationshipType>,
    pub hierarchies: Vec<Hierarchy>,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityType {
    pub id: u32,
    pub name: String,
    pub attributes: Vec<AttributeSpec>,
    pub expected_count_ratio: f64, // Ratio of total entities
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipType {
    pub id: u32,
    pub name: String,
    pub source_type: u32,
    pub target_type: u32,
    pub probability: f64, // Probability of connection
    pub weight_range: (f32, f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeSpec {
    pub name: String,
    pub data_type: AttributeDataType,
    pub distribution: AttributeDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeDataType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeDistribution {
    Uniform { min: f64, max: f64 },
    Normal { mean: f64, std_dev: f64 },
    PowerLaw { alpha: f64 },
    Categorical { values: Vec<String>, weights: Vec<f64> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hierarchy {
    pub parent_type: u32,
    pub child_type: u32,
    pub depth_distribution: Vec<f64>, // Probability of each depth level
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub constraint_type: ConstraintType,
    pub entity_types: Vec<u32>,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    MinDegree,
    MaxDegree,
    ClusteringCoefficient,
    PathLengthDistribution,
}

/// Author productivity statistics for academic paper generation
#[derive(Debug, Clone)]
pub struct AuthorProductivity {
    pub h_index: u32,
    pub paper_count: u32,
    pub collaboration_factor: f32,
    pub field_specialization: Vec<String>,
}

/// Knowledge graph generator with domain-specific patterns
pub struct KnowledgeGraphGenerator {
    rng: DeterministicRng,
    ontology: Ontology,
}

impl KnowledgeGraphGenerator {
    /// Create a new knowledge graph generator
    pub fn new(seed: u64, ontology: Ontology) -> Self {
        let mut rng = DeterministicRng::new(seed);
        rng.set_label("knowledge_graph_generator".to_string());
        
        Self { rng, ontology }
    }

    /// Generate academic paper citation network
    /// Expected properties:
    /// - Citation power law distribution
    /// - Temporal ordering (papers cite older papers)
    /// - Co-authorship clustering
    pub fn generate_academic_papers(&mut self, paper_count: u64) -> Result<TestGraph> {
        if paper_count == 0 {
            return Err(anyhow!("Paper count must be positive"));
        }

        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: 0,
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
                density: 0.0,
                connectivity: ConnectivityType::ScaleFree,
                expected_path_length: 0.0,
            },
        };

        // Generate papers with temporal ordering
        let mut papers = Vec::new();
        for i in 0..paper_count {
            let paper_id = i as u32;
            let year = 2000 + (i % 24) as u32; // 24-year span
            let citation_count = self.calculate_expected_citations(i, paper_count);
            
            let paper = TestEntity {
                id: paper_id,
                name: format!("Paper_{}", i),
                entity_type: "Paper".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("year".to_string(), year.to_string());
                    attrs.insert("citation_count".to_string(), citation_count.to_string());
                    attrs.insert("venue".to_string(), self.generate_venue());
                    attrs.insert("field".to_string(), self.generate_research_field());
                    attrs.insert("quality_score".to_string(), self.generate_quality_score().to_string());
                    attrs
                },
            };
            
            papers.push(paper.clone());
            graph.entities.push(paper);
        }

        // Generate authors with realistic distribution
        let author_count = (paper_count as f64 * 0.3) as u64; // 30% unique authors
        let mut authors = Vec::new();
        
        for i in 0..author_count {
            let author_id = (paper_count + i) as u32;
            let productivity = self.generate_author_productivity(i, author_count);
            
            let author = TestEntity {
                id: author_id,
                name: format!("Author_{}", i),
                entity_type: "Author".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("h_index".to_string(), productivity.h_index.to_string());
                    attrs.insert("paper_count".to_string(), productivity.paper_count.to_string());
                    attrs.insert("collaboration_factor".to_string(), productivity.collaboration_factor.to_string());
                    attrs.insert("specialization".to_string(), productivity.field_specialization.join(","));
                    attrs
                },
            };
            
            authors.push(author.clone());
            graph.entities.push(author);
        }

        // Generate venues
        let venue_count = (paper_count as f64 * 0.05) as u64; // 5% unique venues
        for i in 0..venue_count {
            let venue_id = (paper_count + author_count + i) as u32;
            let venue = TestEntity {
                id: venue_id,
                name: format!("Venue_{}", i),
                entity_type: "Venue".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("impact_factor".to_string(), self.generate_impact_factor().to_string());
                    attrs.insert("acceptance_rate".to_string(), self.generate_acceptance_rate().to_string());
                    attrs.insert("venue_type".to_string(), self.generate_venue_type());
                    attrs
                },
            };
            graph.entities.push(venue);
        }

        // Generate citation network with power law distribution
        self.generate_citation_network(&mut graph, &papers)?;
        
        // Generate authorship relationships
        self.generate_authorship_relationships(&mut graph, &papers, &authors)?;
        
        // Generate publication relationships
        self.generate_publication_relationships(&mut graph, &papers, venue_count)?;

        // Update graph properties
        graph.properties.entity_count = graph.entities.len() as u64;
        graph.properties.edge_count = graph.edges.len() as u64;
        graph.properties.average_degree = if graph.entities.len() > 0 {
            (2.0 * graph.edges.len() as f64) / (graph.entities.len() as f64)
        } else {
            0.0
        };

        Ok(graph)
    }

    /// Generate social network with known community structure
    /// Expected properties:
    /// - High clustering coefficient (0.3-0.7)
    /// - Small world property (avg path length ~ log(n))
    /// - Community structure with modularity > 0.3
    /// - Degree distribution follows power law
    pub fn generate_social_network(&mut self, user_count: u64) -> Result<TestGraph> {
        if user_count == 0 {
            return Err(anyhow!("User count must be positive"));
        }

        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: user_count,
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
                density: 0.0,
                connectivity: ConnectivityType::SmallWorld,
                expected_path_length: (user_count as f64).ln(),
            },
        };

        // Generate users with social attributes
        for i in 0..user_count {
            let user = TestEntity {
                id: i as u32,
                name: format!("User_{}", i),
                entity_type: "User".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("age".to_string(), self.generate_age().to_string());
                    attrs.insert("location".to_string(), self.generate_location());
                    attrs.insert("interests".to_string(), self.generate_interests().join(","));
                    attrs.insert("activity_level".to_string(), self.generate_activity_level().to_string());
                    attrs.insert("community".to_string(), self.assign_community(i, user_count).to_string());
                    attrs
                },
            };
            graph.entities.push(user);
        }

        // Generate friendships with community structure
        self.generate_social_relationships(&mut graph, user_count)?;

        // Update graph properties
        graph.properties.edge_count = graph.edges.len() as u64;
        graph.properties.average_degree = (2.0 * graph.edges.len() as f64) / (user_count as f64);
        graph.properties.clustering_coefficient = self.estimate_clustering_coefficient(&graph);

        Ok(graph)
    }

    /// Generate biological pathway network
    /// Entities: Proteins, Genes, Pathways, Compounds
    /// Relationships: Interacts, Regulates, Catalyzes, PartOf
    /// Expected properties:
    /// - Scale-free degree distribution
    /// - High clustering (biological modules)
    /// - Hierarchical organization
    pub fn generate_biological_pathway(&mut self, protein_count: u64) -> Result<TestGraph> {
        if protein_count == 0 {
            return Err(anyhow!("Protein count must be positive"));
        }

        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: 0,
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
                density: 0.0,
                connectivity: ConnectivityType::ScaleFree,
                expected_path_length: 0.0,
            },
        };

        // Generate proteins
        for i in 0..protein_count {
            let protein = TestEntity {
                id: i as u32,
                name: format!("Protein_{}", i),
                entity_type: "Protein".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("molecular_weight".to_string(), self.generate_molecular_weight().to_string());
                    attrs.insert("function".to_string(), self.generate_protein_function());
                    attrs.insert("localization".to_string(), self.generate_cellular_localization());
                    attrs.insert("expression_level".to_string(), self.generate_expression_level().to_string());
                    attrs
                },
            };
            graph.entities.push(protein);
        }

        // Generate genes (1:1 with proteins for simplicity)
        for i in 0..protein_count {
            let gene_id = (protein_count + i) as u32;
            let gene = TestEntity {
                id: gene_id,
                name: format!("Gene_{}", i),
                entity_type: "Gene".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("chromosome".to_string(), self.generate_chromosome().to_string());
                    attrs.insert("start_position".to_string(), self.generate_genomic_position().to_string());
                    attrs.insert("gene_length".to_string(), self.generate_gene_length().to_string());
                    attrs.insert("coding_protein".to_string(), i.to_string());
                    attrs
                },
            };
            graph.entities.push(gene);
        }

        // Generate pathways
        let pathway_count = (protein_count as f64 * 0.1) as u64; // 10% of proteins
        for i in 0..pathway_count {
            let pathway_id = (2 * protein_count + i) as u32;
            let pathway = TestEntity {
                id: pathway_id,
                name: format!("Pathway_{}", i),
                entity_type: "Pathway".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("pathway_type".to_string(), self.generate_pathway_type());
                    attrs.insert("organism".to_string(), "Homo sapiens".to_string());
                    attrs.insert("protein_count".to_string(), self.generate_pathway_size().to_string());
                    attrs
                },
            };
            graph.entities.push(pathway);
        }

        // Generate compounds
        let compound_count = (protein_count as f64 * 0.05) as u64; // 5% of proteins
        for i in 0..compound_count {
            let compound_id = (2 * protein_count + pathway_count + i) as u32;
            let compound = TestEntity {
                id: compound_id,
                name: format!("Compound_{}", i),
                entity_type: "Compound".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("molecular_formula".to_string(), self.generate_molecular_formula());
                    attrs.insert("compound_type".to_string(), self.generate_compound_type());
                    attrs.insert("solubility".to_string(), self.generate_solubility().to_string());
                    attrs
                },
            };
            graph.entities.push(compound);
        }

        // Generate biological relationships
        self.generate_protein_interactions(&mut graph, protein_count)?;
        self.generate_gene_protein_relationships(&mut graph, protein_count)?;
        self.generate_pathway_memberships(&mut graph, protein_count, pathway_count)?;
        self.generate_compound_interactions(&mut graph, protein_count, compound_count)?;

        // Update graph properties
        graph.properties.entity_count = graph.entities.len() as u64;
        graph.properties.edge_count = graph.edges.len() as u64;
        graph.properties.average_degree = if graph.entities.len() > 0 {
            (2.0 * graph.edges.len() as f64) / (graph.entities.len() as f64)
        } else {
            0.0
        };

        Ok(graph)
    }

    // Helper methods for academic paper generation

    fn calculate_expected_citations(&mut self, paper_index: u64, total_papers: u64) -> u32 {
        // Power law distribution for citations: older papers can have more citations
        let age_factor = (total_papers - paper_index) as f64 / total_papers as f64;
        let base_citations = self.rng.poisson(age_factor * 10.0);
        
        // Add some high-impact papers (top 1%)
        if self.rng.next_f64() < 0.01 {
            base_citations + self.rng.poisson(100.0)
        } else {
            base_citations
        }
    }

    fn generate_venue(&mut self) -> String {
        let venues = vec![
            "Nature", "Science", "Cell", "ICML", "NeurIPS", "ICLR", 
            "AAAI", "IJCAI", "ACL", "EMNLP", "CVPR", "ICCV"
        ];
        venues[self.rng.next_usize(venues.len())].to_string()
    }

    fn generate_research_field(&mut self) -> String {
        let fields = vec![
            "Machine Learning", "Computer Vision", "Natural Language Processing",
            "Robotics", "Bioinformatics", "Neuroscience", "Physics", "Chemistry"
        ];
        fields[self.rng.next_usize(fields.len())].to_string()
    }

    fn generate_quality_score(&mut self) -> f32 {
        // Beta distribution for quality scores (concentrated around 0.6-0.8)
        let alpha = 2.0;
        let beta = 2.0;
        let u1 = self.rng.next_f64();
        let u2 = self.rng.next_f64();
        
        // Simple beta approximation using uniform random variables
        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / beta);
        (x / (x + y)) as f32
    }

    fn generate_author_productivity(&mut self, author_index: u64, total_authors: u64) -> AuthorProductivity {
        // Power law for productivity (Pareto principle)
        let productivity_rank = author_index as f64 / total_authors as f64;
        let base_productivity = 1.0 / productivity_rank.max(0.01);
        
        let paper_count = (base_productivity * 5.0) as u32 + 1;
        let h_index = (paper_count as f64 * 0.6) as u32;
        let collaboration_factor = self.rng.range_f64(0.3, 0.9) as f32;
        
        let specializations = vec![
            "Machine Learning", "Computer Vision", "NLP", "Systems", 
            "Theory", "HCI", "Security", "Databases"
        ];
        let spec_count = self.rng.range_i32(1, 4) as usize;
        let field_specialization = self.rng.sample(&specializations, spec_count);

        AuthorProductivity {
            h_index,
            paper_count,
            collaboration_factor,
            field_specialization,
        }
    }

    fn generate_citation_network(&mut self, graph: &mut TestGraph, papers: &[TestEntity]) -> Result<()> {
        // Papers can only cite papers published before them
        for (i, paper) in papers.iter().enumerate() {
            let paper_year: u32 = paper.attributes["year"].parse().unwrap_or(2000);
            let citation_count: u32 = paper.attributes["citation_count"].parse().unwrap_or(0);
            
            // Generate citations based on temporal ordering and power law
            let mut citations_made = 0;
            let target_citations = citation_count.min(i as u32); // Can't cite more than available papers
            
            while citations_made < target_citations && citations_made < i as u32 {
                // Preferentially cite recent, high-quality papers
                let mut cited_paper_idx = None;
                let mut attempts = 0;
                
                while cited_paper_idx.is_none() && attempts < 100 {
                    let candidate_idx = self.rng.next_usize(i); // Only earlier papers
                    let candidate = &papers[candidate_idx];
                    let candidate_year: u32 = candidate.attributes["year"].parse().unwrap_or(2000);
                    let candidate_quality: f32 = candidate.attributes["quality_score"].parse().unwrap_or(0.5);
                    
                    // Probability based on recency and quality
                    let time_diff = (paper_year - candidate_year) as f64 + 1.0;
                    let cite_probability = (candidate_quality as f64 * 0.8) / time_diff.ln();
                    
                    if self.rng.next_f64() < cite_probability {
                        cited_paper_idx = Some(candidate_idx);
                    }
                    attempts += 1;
                }
                
                if let Some(cited_idx) = cited_paper_idx {
                    let edge = TestEdge {
                        source: paper.id,
                        target: papers[cited_idx].id,
                        weight: 1.0,
                        edge_type: "cites".to_string(),
                    };
                    graph.edges.push(edge);
                }
                
                citations_made += 1;
            }
        }
        
        Ok(())
    }

    fn generate_authorship_relationships(&mut self, graph: &mut TestGraph, papers: &[TestEntity], authors: &[TestEntity]) -> Result<()> {
        for paper in papers {
            // Each paper has 1-5 authors (realistic distribution)
            let author_count = match self.rng.next_u32() % 100 {
                0..=40 => 1,  // 40% single author
                41..=70 => 2, // 30% two authors
                71..=85 => 3, // 15% three authors
                86..=95 => 4, // 10% four authors
                _ => 5,       // 5% five authors
            };
            
            let selected_authors = self.rng.sample(authors, author_count.min(authors.len()));
            
            for (order, author) in selected_authors.iter().enumerate() {
                let edge = TestEdge {
                    source: author.id,
                    target: paper.id,
                    weight: 1.0 / (order + 1) as f32, // First author has higher weight
                    edge_type: "authored_by".to_string(),
                };
                graph.edges.push(edge);
            }
        }
        
        Ok(())
    }

    fn generate_publication_relationships(&mut self, graph: &mut TestGraph, papers: &[TestEntity], venue_count: u64) -> Result<()> {
        let paper_count = papers.len() as u64;
        
        for paper in papers {
            // Each paper is published in exactly one venue
            let venue_id = (paper_count + (paper_count as f64 * 0.3) as u64 + 
                           self.rng.next_u64() % venue_count) as u32;
            
            let edge = TestEdge {
                source: paper.id,
                target: venue_id,
                weight: 1.0,
                edge_type: "published_in".to_string(),
            };
            graph.edges.push(edge);
        }
        
        Ok(())
    }

    // Helper methods for social network generation

    fn generate_age(&mut self) -> u32 {
        // Normal distribution around 35, std dev 15
        let age = self.rng.normal(35.0, 15.0);
        (age.max(13.0).min(90.0)) as u32
    }

    fn generate_location(&mut self) -> String {
        let cities = vec![
            "New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", 
            "Toronto", "Seattle", "San Francisco", "Boston"
        ];
        cities[self.rng.next_usize(cities.len())].to_string()
    }

    fn generate_interests(&mut self) -> Vec<String> {
        let interests = vec![
            "Technology", "Sports", "Music", "Travel", "Food", "Art", 
            "Science", "Politics", "Gaming", "Photography", "Reading", "Movies"
        ];
        let count = self.rng.range_i32(2, 6) as usize;
        self.rng.sample(&interests, count)
    }

    fn generate_activity_level(&mut self) -> f32 {
        // Log-normal distribution for activity
        let log_activity = self.rng.normal(0.0, 1.0);
        (log_activity.exp() as f32).min(10.0)
    }

    fn assign_community(&mut self, user_index: u64, total_users: u64) -> u32 {
        // Power law community sizes
        let community_count = (total_users as f64).sqrt() as u32;
        let community_sizes = self.generate_power_law_sizes(community_count, total_users);
        
        let mut cumulative_size = 0;
        for (community_id, &size) in community_sizes.iter().enumerate() {
            cumulative_size += size;
            if user_index < cumulative_size {
                return community_id as u32;
            }
        }
        
        (community_count - 1) // Fallback to last community
    }

    fn generate_power_law_sizes(&mut self, count: u32, total: u64) -> Vec<u64> {
        let mut sizes = Vec::new();
        let mut remaining = total;
        
        for i in 0..count {
            let size = if i == count - 1 {
                remaining // Last community gets remaining users
            } else {
                let proportion = 1.0 / ((i + 1) as f64).powf(1.5); // Power law
                (total as f64 * proportion * 0.3) as u64 + 1
            };
            sizes.push(size);
            remaining = remaining.saturating_sub(size);
        }
        
        sizes
    }

    fn generate_social_relationships(&mut self, graph: &mut TestGraph, user_count: u64) -> Result<()> {
        // Generate friendships with community structure and small-world properties
        for i in 0..user_count {
            let user_community = self.assign_community(i, user_count);
            let target_degree = self.generate_social_degree();
            let mut current_degree = 0;
            
            while current_degree < target_degree && current_degree < user_count - 1 {
                let mut friend_id = None;
                let mut attempts = 0;
                
                while friend_id.is_none() && attempts < 100 {
                    let candidate_id = self.rng.next_u64() % user_count;
                    if candidate_id != i {
                        let candidate_community = self.assign_community(candidate_id, user_count);
                        
                        // Higher probability of friendship within same community
                        let friendship_prob = if user_community == candidate_community {
                            0.1 // 10% chance within community
                        } else {
                            0.001 // 0.1% chance across communities
                        };
                        
                        if self.rng.next_f64() < friendship_prob {
                            friend_id = Some(candidate_id);
                        }
                    }
                    attempts += 1;
                }
                
                if let Some(friend) = friend_id {
                    let edge = TestEdge {
                        source: i as u32,
                        target: friend as u32,
                        weight: 1.0,
                        edge_type: "friend".to_string(),
                    };
                    graph.edges.push(edge);
                }
                
                current_degree += 1;
            }
        }
        
        Ok(())
    }

    fn generate_social_degree(&mut self) -> u32 {
        // Power law degree distribution (Dunbar's number consideration)
        let u = self.rng.next_f64();
        let degree = (1.0 / u.powf(1.0 / 2.1)) as u32; // Power law exponent ~2.1
        degree.min(150) // Dunbar's number cap
    }

    // Helper methods for biological pathway generation

    fn generate_molecular_weight(&mut self) -> f32 {
        // Log-normal distribution for protein molecular weights
        let log_mw = self.rng.normal(10.5, 0.8); // ~50 kDa average
        log_mw.exp() as f32
    }

    fn generate_protein_function(&mut self) -> String {
        let functions = vec![
            "Enzyme", "Transcription Factor", "Receptor", "Structural", 
            "Transport", "Signal Transduction", "Immune Response", "Metabolism"
        ];
        functions[self.rng.next_usize(functions.len())].to_string()
    }

    fn generate_cellular_localization(&mut self) -> String {
        let locations = vec![
            "Nucleus", "Cytoplasm", "Membrane", "Mitochondria", 
            "Endoplasmic Reticulum", "Golgi", "Extracellular"
        ];
        locations[self.rng.next_usize(locations.len())].to_string()
    }

    fn generate_expression_level(&mut self) -> f32 {
        // Log-normal distribution for expression levels
        let log_expr = self.rng.normal(5.0, 2.0);
        log_expr.exp() as f32
    }

    fn generate_chromosome(&mut self) -> u32 {
        self.rng.range_i32(1, 23) as u32 // Human chromosomes 1-22
    }

    fn generate_genomic_position(&mut self) -> u64 {
        self.rng.next_u64() % 250_000_000 // ~250 Mb chromosome size
    }

    fn generate_gene_length(&mut self) -> u32 {
        // Log-normal distribution for gene lengths
        let log_length = self.rng.normal(8.0, 1.5); // ~3kb average
        (log_length.exp() as u32).max(100)
    }

    fn generate_pathway_type(&mut self) -> String {
        let types = vec![
            "Metabolic", "Signaling", "Gene Expression", "DNA Repair", 
            "Cell Cycle", "Apoptosis", "Immune Response"
        ];
        types[self.rng.next_usize(types.len())].to_string()
    }

    fn generate_pathway_size(&mut self) -> u32 {
        // Power law distribution for pathway sizes
        let u = self.rng.next_f64();
        ((1.0 / u.powf(0.5)) as u32).max(5).min(500)
    }

    fn generate_molecular_formula(&mut self) -> String {
        let c_count = self.rng.range_i32(1, 30);
        let h_count = self.rng.range_i32(1, c_count * 3);
        let o_count = self.rng.range_i32(0, 10);
        let n_count = self.rng.range_i32(0, 5);
        
        format!("C{}H{}O{}N{}", c_count, h_count, o_count, n_count)
    }

    fn generate_compound_type(&mut self) -> String {
        let types = vec![
            "Metabolite", "Drug", "Hormone", "Neurotransmitter", 
            "Lipid", "Carbohydrate", "Amino Acid", "Nucleotide"
        ];
        types[self.rng.next_usize(types.len())].to_string()
    }

    fn generate_solubility(&mut self) -> f32 {
        // Log-normal distribution for solubility
        let log_sol = self.rng.normal(-2.0, 2.0);
        log_sol.exp() as f32
    }

    fn generate_protein_interactions(&mut self, graph: &mut TestGraph, protein_count: u64) -> Result<()> {
        // Scale-free network for protein-protein interactions
        for i in 0..protein_count {
            let degree = self.generate_ppi_degree();
            let mut interactions = 0;
            
            while interactions < degree && interactions < protein_count - 1 {
                let target = self.rng.next_u64() % protein_count;
                if target != i {
                    let edge = TestEdge {
                        source: i as u32,
                        target: target as u32,
                        weight: self.rng.range_f64(0.1, 1.0) as f32,
                        edge_type: "interacts_with".to_string(),
                    };
                    graph.edges.push(edge);
                }
                interactions += 1;
            }
        }
        
        Ok(())
    }

    fn generate_ppi_degree(&mut self) -> u32 {
        // Power law degree distribution for PPI networks
        let u = self.rng.next_f64();
        (1.0 / u.powf(1.0 / 2.5)) as u32 // Exponent ~2.5 for PPI networks
    }

    fn generate_gene_protein_relationships(&mut self, graph: &mut TestGraph, protein_count: u64) -> Result<()> {
        // 1:1 relationship between genes and proteins
        for i in 0..protein_count {
            let edge = TestEdge {
                source: (protein_count + i) as u32, // Gene ID
                target: i as u32,                    // Protein ID
                weight: 1.0,
                edge_type: "codes_for".to_string(),
            };
            graph.edges.push(edge);
        }
        
        Ok(())
    }

    fn generate_pathway_memberships(&mut self, graph: &mut TestGraph, protein_count: u64, pathway_count: u64) -> Result<()> {
        let pathway_start_id = 2 * protein_count;
        
        for pathway_idx in 0..pathway_count {
            let pathway_id = (pathway_start_id + pathway_idx) as u32;
            let pathway_size = self.generate_pathway_size().min(protein_count as u32);
            
            // Select random proteins for this pathway
            let selected_proteins = self.rng.sample(
                &(0..protein_count as u32).collect::<Vec<_>>(), 
                pathway_size as usize
            );
            
            for &protein_id in &selected_proteins {
                let edge = TestEdge {
                    source: protein_id,
                    target: pathway_id,
                    weight: 1.0,
                    edge_type: "part_of".to_string(),
                };
                graph.edges.push(edge);
            }
        }
        
        Ok(())
    }

    fn generate_compound_interactions(&mut self, graph: &mut TestGraph, protein_count: u64, compound_count: u64) -> Result<()> {
        let compound_start_id = (2 * protein_count + (protein_count as f64 * 0.1) as u64) as u32;
        
        for compound_idx in 0..compound_count {
            let compound_id = compound_start_id + compound_idx as u32;
            let interaction_count = self.rng.range_i32(1, 10) as u32; // 1-10 protein interactions per compound
            
            for _ in 0..interaction_count {
                let protein_id = self.rng.next_u32() % protein_count as u32;
                let edge = TestEdge {
                    source: compound_id,
                    target: protein_id,
                    weight: self.rng.range_f64(0.1, 1.0) as f32,
                    edge_type: "binds_to".to_string(),
                };
                graph.edges.push(edge);
            }
        }
        
        Ok(())
    }

    fn estimate_clustering_coefficient(&self, graph: &TestGraph) -> f64 {
        // Simple estimation of clustering coefficient
        // In a real implementation, this would compute the exact value
        match graph.properties.connectivity {
            ConnectivityType::SmallWorld => 0.3 + self.rng.next_f64() * 0.4, // 0.3-0.7
            ConnectivityType::ScaleFree => 0.1 + self.rng.next_f64() * 0.2,  // 0.1-0.3
            ConnectivityType::Random => 0.05 + self.rng.next_f64() * 0.1,   // 0.05-0.15
            _ => 0.1,
        }
    }

    fn generate_impact_factor(&mut self) -> f32 {
        // Log-normal distribution for impact factors
        let log_if = self.rng.normal(0.5, 1.0);
        log_if.exp() as f32
    }

    fn generate_acceptance_rate(&mut self) -> f32 {
        // Beta distribution for acceptance rates (0-1)
        let alpha = 2.0;
        let beta = 5.0;
        let u1 = self.rng.next_f64();
        let u2 = self.rng.next_f64();
        
        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / beta);
        (x / (x + y)) as f32
    }

    fn generate_venue_type(&mut self) -> String {
        let types = vec!["Conference", "Journal", "Workshop", "Symposium"];
        types[self.rng.next_usize(types.len())].to_string()
    }
}

/// Create default academic ontology
pub fn create_academic_ontology() -> Ontology {
    Ontology {
        entity_types: vec![
            EntityType {
                id: 0,
                name: "Paper".to_string(),
                attributes: vec![
                    AttributeSpec {
                        name: "year".to_string(),
                        data_type: AttributeDataType::Integer,
                        distribution: AttributeDistribution::Uniform { min: 2000.0, max: 2024.0 },
                    },
                    AttributeSpec {
                        name: "citation_count".to_string(),
                        data_type: AttributeDataType::Integer,
                        distribution: AttributeDistribution::PowerLaw { alpha: 2.5 },
                    },
                ],
                expected_count_ratio: 0.6,
            },
            EntityType {
                id: 1,
                name: "Author".to_string(),
                attributes: vec![
                    AttributeSpec {
                        name: "h_index".to_string(),
                        data_type: AttributeDataType::Integer,
                        distribution: AttributeDistribution::PowerLaw { alpha: 2.0 },
                    },
                ],
                expected_count_ratio: 0.3,
            },
            EntityType {
                id: 2,
                name: "Venue".to_string(),
                attributes: vec![
                    AttributeSpec {
                        name: "impact_factor".to_string(),
                        data_type: AttributeDataType::Float,
                        distribution: AttributeDistribution::Normal { mean: 2.0, std_dev: 1.5 },
                    },
                ],
                expected_count_ratio: 0.05,
            },
        ],
        relationship_types: vec![
            RelationshipType {
                id: 0,
                name: "cites".to_string(),
                source_type: 0,
                target_type: 0,
                probability: 0.1,
                weight_range: (1.0, 1.0),
            },
            RelationshipType {
                id: 1,
                name: "authored_by".to_string(),
                source_type: 1,
                target_type: 0,
                probability: 0.8,
                weight_range: (0.2, 1.0),
            },
        ],
        hierarchies: vec![],
        constraints: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_academic_paper_generation() {
        let ontology = create_academic_ontology();
        let mut generator = KnowledgeGraphGenerator::new(42, ontology);
        
        let graph = generator.generate_academic_papers(100).unwrap();
        
        // Should have papers, authors, and venues
        assert!(graph.entities.len() > 100);
        assert!(graph.edges.len() > 0);
        
        // Check for different entity types
        let paper_count = graph.entities.iter().filter(|e| e.entity_type == "Paper").count();
        let author_count = graph.entities.iter().filter(|e| e.entity_type == "Author").count();
        let venue_count = graph.entities.iter().filter(|e| e.entity_type == "Venue").count();
        
        assert_eq!(paper_count, 100);
        assert!(author_count > 0);
        assert!(venue_count > 0);
    }

    #[test]
    fn test_social_network_generation() {
        let ontology = Ontology {
            entity_types: vec![],
            relationship_types: vec![],
            hierarchies: vec![],
            constraints: vec![],
        };
        let mut generator = KnowledgeGraphGenerator::new(42, ontology);
        
        let graph = generator.generate_social_network(50).unwrap();
        
        assert_eq!(graph.entities.len(), 50);
        assert!(graph.edges.len() > 0);
        
        // All entities should be users
        for entity in &graph.entities {
            assert_eq!(entity.entity_type, "User");
            assert!(entity.attributes.contains_key("age"));
            assert!(entity.attributes.contains_key("community"));
        }
    }

    #[test]
    fn test_biological_pathway_generation() {
        let ontology = Ontology {
            entity_types: vec![],
            relationship_types: vec![],
            hierarchies: vec![],
            constraints: vec![],
        };
        let mut generator = KnowledgeGraphGenerator::new(42, ontology);
        
        let graph = generator.generate_biological_pathway(20).unwrap();
        
        // Should have proteins, genes, pathways, and compounds
        let protein_count = graph.entities.iter().filter(|e| e.entity_type == "Protein").count();
        let gene_count = graph.entities.iter().filter(|e| e.entity_type == "Gene").count();
        let pathway_count = graph.entities.iter().filter(|e| e.entity_type == "Pathway").count();
        let compound_count = graph.entities.iter().filter(|e| e.entity_type == "Compound").count();
        
        assert_eq!(protein_count, 20);
        assert_eq!(gene_count, 20); // 1:1 with proteins
        assert!(pathway_count > 0);
        assert!(compound_count > 0);
        assert!(graph.edges.len() > 0);
    }

    #[test]
    fn test_deterministic_generation() {
        let ontology = create_academic_ontology();
        let mut gen1 = KnowledgeGraphGenerator::new(12345, ontology.clone());
        let mut gen2 = KnowledgeGraphGenerator::new(12345, ontology);
        
        let graph1 = gen1.generate_academic_papers(20).unwrap();
        let graph2 = gen2.generate_academic_papers(20).unwrap();
        
        // Same seed should produce identical graphs
        assert_eq!(graph1.entities.len(), graph2.entities.len());
        assert_eq!(graph1.edges.len(), graph2.edges.len());
        
        for (e1, e2) in graph1.entities.iter().zip(graph2.entities.iter()) {
            assert_eq!(e1.id, e2.id);
            assert_eq!(e1.name, e2.name);
            assert_eq!(e1.entity_type, e2.entity_type);
        }
    }

    #[test]
    fn test_invalid_parameters() {
        let ontology = create_academic_ontology();
        let mut generator = KnowledgeGraphGenerator::new(42, ontology);
        
        // Test zero counts
        assert!(generator.generate_academic_papers(0).is_err());
        assert!(generator.generate_social_network(0).is_err());
        assert!(generator.generate_biological_pathway(0).is_err());
    }
}