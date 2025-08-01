//! Expected Extraction Results for Testing
//! 
//! Contains expected results for knowledge extraction and processing tests.
//! Used to validate that processing components produce correct outputs.

/// Expected entities for simple document
pub const SIMPLE_DOCUMENT_EXPECTED_ENTITIES: &[&str] = &[
    "fox",
    "dog", 
    "alphabet",
    "Canidae family",
    "mammals"
];

/// Expected relationships for simple document
pub const SIMPLE_DOCUMENT_EXPECTED_RELATIONSHIPS: &[(&str, &str, &str)] = &[
    ("fox", "jumps_over", "dog"),
    ("foxes", "belong_to", "Canidae family"),
    ("foxes", "are", "carnivorous mammals"),
    ("sentence", "contains", "alphabet letters")
];

/// Expected entities for complex scientific document
pub const COMPLEX_DOCUMENT_EXPECTED_ENTITIES: &[&str] = &[
    "quantum computing",
    "machine learning",
    "neural networks",
    "qubits",
    "superposition",
    "entanglement",
    "Variational Quantum Eigensolver",
    "VQE",
    "Hamiltonian"
];

/// Expected relationships for complex scientific document
pub const COMPLEX_DOCUMENT_EXPECTED_RELATIONSHIPS: &[(&str, &str, &str)] = &[
    ("quantum computing", "offers", "exponential speedups"),
    ("qubits", "exist_in", "superposition states"),
    ("VQE", "is_a", "hybrid classical-quantum algorithm"),
    ("VQE", "finds", "ground state energies"),
    ("VQE", "used_for", "drug discovery")
];

/// Expected semantic themes for documents
pub struct ExpectedThemes;

impl ExpectedThemes {
    pub fn simple_document_themes() -> Vec<&'static str> {
        vec!["animals", "language", "basic facts"]
    }
    
    pub fn complex_document_themes() -> Vec<&'static str> {
        vec!["quantum computing", "machine learning", "scientific research", "algorithms"]
    }
    
    pub fn multilingual_document_themes() -> Vec<&'static str> {
        vec!["language processing", "internationalization", "cultural context"]
    }
}

/// Expected processing quality scores
pub struct ExpectedQualityScores;

impl ExpectedQualityScores {
    pub fn simple_document_min_score() -> f64 { 0.8 }
    pub fn complex_document_min_score() -> f64 { 0.9 }
    pub fn technical_document_min_score() -> f64 { 0.85 }
    pub fn multilingual_document_min_score() -> f64 { 0.75 }
}

/// Expected extraction confidence thresholds
pub struct ExtractionConfidence;

impl ExtractionConfidence {
    pub fn entity_min_confidence() -> f64 { 0.7 }
    pub fn relationship_min_confidence() -> f64 { 0.6 }
    pub fn theme_min_confidence() -> f64 { 0.8 }
    pub fn semantic_similarity_threshold() -> f64 { 0.75 }
}