//! Constants and static data for divergent thinking

use std::collections::HashMap;

/// Default exploration breadth
pub const DEFAULT_EXPLORATION_BREADTH: usize = 20;

/// Default creativity threshold
pub const DEFAULT_CREATIVITY_THRESHOLD: f32 = 0.3;

/// Default maximum path length
pub const DEFAULT_MAX_PATH_LENGTH: usize = 5;

/// Default novelty weight
pub const DEFAULT_NOVELTY_WEIGHT: f32 = 0.4;

/// Default relevance weight
pub const DEFAULT_RELEVANCE_WEIGHT: f32 = 0.6;

/// Default activation decay
pub const DEFAULT_ACTIVATION_DECAY: f32 = 0.9;

/// Default minimum activation threshold
pub const DEFAULT_MIN_ACTIVATION: f32 = 0.1;

/// Default maximum number of results
pub const DEFAULT_MAX_RESULTS: usize = 50;

/// Domain hierarchy for hierarchical relevance calculation
pub fn get_domain_hierarchy() -> HashMap<String, Vec<String>> {
    let mut hierarchy = HashMap::new();
    
    // Animal hierarchy
    hierarchy.insert("animal".to_string(), vec![
        "mammal".to_string(),
        "bird".to_string(),
        "fish".to_string(),
        "reptile".to_string(),
        "amphibian".to_string(),
        "insect".to_string(),
    ]);
    
    hierarchy.insert("mammal".to_string(), vec![
        "carnivore".to_string(),
        "herbivore".to_string(),
        "omnivore".to_string(),
        "primate".to_string(),
        "rodent".to_string(),
        "ungulate".to_string(),
    ]);
    
    hierarchy.insert("bird".to_string(), vec![
        "raptor".to_string(),
        "songbird".to_string(),
        "waterfowl".to_string(),
        "seabird".to_string(),
        "gamebird".to_string(),
    ]);
    
    hierarchy.insert("fish".to_string(), vec![
        "freshwater".to_string(),
        "saltwater".to_string(),
        "tropical".to_string(),
        "coldwater".to_string(),
        "deepwater".to_string(),
    ]);
    
    // Technology hierarchy
    hierarchy.insert("technology".to_string(), vec![
        "computer".to_string(),
        "software".to_string(),
        "hardware".to_string(),
        "network".to_string(),
        "mobile".to_string(),
        "ai".to_string(),
    ]);
    
    hierarchy.insert("computer".to_string(), vec![
        "laptop".to_string(),
        "desktop".to_string(),
        "server".to_string(),
        "processor".to_string(),
        "memory".to_string(),
        "storage".to_string(),
    ]);
    
    hierarchy.insert("software".to_string(), vec![
        "application".to_string(),
        "system".to_string(),
        "database".to_string(),
        "framework".to_string(),
        "library".to_string(),
        "tool".to_string(),
    ]);
    
    hierarchy.insert("ai".to_string(), vec![
        "machine_learning".to_string(),
        "deep_learning".to_string(),
        "pattern_recognition".to_string(),
        "natural_language".to_string(),
        "computer_vision".to_string(),
        "robotics".to_string(),
    ]);
    
    // Science hierarchy
    hierarchy.insert("science".to_string(), vec![
        "physics".to_string(),
        "chemistry".to_string(),
        "biology".to_string(),
        "mathematics".to_string(),
        "astronomy".to_string(),
        "geology".to_string(),
    ]);
    
    hierarchy.insert("physics".to_string(), vec![
        "mechanics".to_string(),
        "thermodynamics".to_string(),
        "electromagnetism".to_string(),
        "quantum".to_string(),
        "relativity".to_string(),
        "optics".to_string(),
    ]);
    
    hierarchy.insert("chemistry".to_string(), vec![
        "organic".to_string(),
        "inorganic".to_string(),
        "physical".to_string(),
        "analytical".to_string(),
        "biochemistry".to_string(),
        "materials".to_string(),
    ]);
    
    hierarchy.insert("biology".to_string(), vec![
        "molecular".to_string(),
        "cellular".to_string(),
        "genetics".to_string(),
        "ecology".to_string(),
        "evolution".to_string(),
        "anatomy".to_string(),
    ]);
    
    hierarchy.insert("mathematics".to_string(), vec![
        "algebra".to_string(),
        "geometry".to_string(),
        "calculus".to_string(),
        "statistics".to_string(),
        "topology".to_string(),
        "logic".to_string(),
    ]);
    
    hierarchy
}

/// Semantic fields for semantic relevance calculation
pub fn get_semantic_fields() -> HashMap<String, Vec<String>> {
    let mut fields = HashMap::new();
    
    // Color field
    fields.insert("color".to_string(), vec![
        "red".to_string(),
        "blue".to_string(),
        "green".to_string(),
        "yellow".to_string(),
        "orange".to_string(),
        "purple".to_string(),
        "pink".to_string(),
        "brown".to_string(),
        "black".to_string(),
        "white".to_string(),
        "gray".to_string(),
    ]);
    
    // Size field
    fields.insert("size".to_string(), vec![
        "tiny".to_string(),
        "small".to_string(),
        "medium".to_string(),
        "large".to_string(),
        "huge".to_string(),
        "massive".to_string(),
        "microscopic".to_string(),
        "gigantic".to_string(),
    ]);
    
    // Shape field
    fields.insert("shape".to_string(), vec![
        "round".to_string(),
        "square".to_string(),
        "rectangular".to_string(),
        "triangular".to_string(),
        "oval".to_string(),
        "circular".to_string(),
        "spherical".to_string(),
        "cylindrical".to_string(),
        "cubic".to_string(),
    ]);
    
    // Texture field
    fields.insert("texture".to_string(), vec![
        "smooth".to_string(),
        "rough".to_string(),
        "soft".to_string(),
        "hard".to_string(),
        "bumpy".to_string(),
        "silky".to_string(),
        "fuzzy".to_string(),
        "sticky".to_string(),
        "slippery".to_string(),
    ]);
    
    // Temperature field
    fields.insert("temperature".to_string(), vec![
        "hot".to_string(),
        "cold".to_string(),
        "warm".to_string(),
        "cool".to_string(),
        "freezing".to_string(),
        "boiling".to_string(),
        "lukewarm".to_string(),
        "scalding".to_string(),
    ]);
    
    // Emotion field
    fields.insert("emotion".to_string(), vec![
        "happy".to_string(),
        "sad".to_string(),
        "angry".to_string(),
        "excited".to_string(),
        "calm".to_string(),
        "nervous".to_string(),
        "confident".to_string(),
        "worried".to_string(),
        "surprised".to_string(),
        "confused".to_string(),
    ]);
    
    // Movement field
    fields.insert("movement".to_string(), vec![
        "fast".to_string(),
        "slow".to_string(),
        "running".to_string(),
        "walking".to_string(),
        "jumping".to_string(),
        "flying".to_string(),
        "swimming".to_string(),
        "crawling".to_string(),
        "spinning".to_string(),
        "dancing".to_string(),
    ]);
    
    // Time field
    fields.insert("time".to_string(), vec![
        "morning".to_string(),
        "afternoon".to_string(),
        "evening".to_string(),
        "night".to_string(),
        "dawn".to_string(),
        "dusk".to_string(),
        "midnight".to_string(),
        "noon".to_string(),
        "yesterday".to_string(),
        "today".to_string(),
        "tomorrow".to_string(),
    ]);
    
    fields
}

/// Domain patterns for domain-specific recognition
pub fn get_domain_patterns() -> HashMap<String, Vec<String>> {
    let mut patterns = HashMap::new();
    
    // Food patterns
    patterns.insert("food".to_string(), vec![
        "eat".to_string(),
        "drink".to_string(),
        "cook".to_string(),
        "recipe".to_string(),
        "ingredient".to_string(),
        "taste".to_string(),
        "flavor".to_string(),
        "spice".to_string(),
        "meal".to_string(),
        "restaurant".to_string(),
        "kitchen".to_string(),
    ]);
    
    // Transportation patterns
    patterns.insert("transportation".to_string(), vec![
        "car".to_string(),
        "train".to_string(),
        "plane".to_string(),
        "bus".to_string(),
        "bicycle".to_string(),
        "motorcycle".to_string(),
        "ship".to_string(),
        "boat".to_string(),
        "travel".to_string(),
        "journey".to_string(),
        "drive".to_string(),
        "fly".to_string(),
    ]);
    
    // Music patterns
    patterns.insert("music".to_string(), vec![
        "song".to_string(),
        "melody".to_string(),
        "rhythm".to_string(),
        "instrument".to_string(),
        "piano".to_string(),
        "guitar".to_string(),
        "drum".to_string(),
        "violin".to_string(),
        "concert".to_string(),
        "band".to_string(),
        "singer".to_string(),
        "sound".to_string(),
    ]);
    
    // Sports patterns
    patterns.insert("sports".to_string(), vec![
        "game".to_string(),
        "play".to_string(),
        "team".to_string(),
        "player".to_string(),
        "ball".to_string(),
        "score".to_string(),
        "win".to_string(),
        "lose".to_string(),
        "competition".to_string(),
        "athlete".to_string(),
        "exercise".to_string(),
        "training".to_string(),
    ]);
    
    // Weather patterns
    patterns.insert("weather".to_string(), vec![
        "rain".to_string(),
        "snow".to_string(),
        "sun".to_string(),
        "cloud".to_string(),
        "wind".to_string(),
        "storm".to_string(),
        "thunder".to_string(),
        "lightning".to_string(),
        "fog".to_string(),
        "humidity".to_string(),
        "temperature".to_string(),
        "forecast".to_string(),
    ]);
    
    // Education patterns
    patterns.insert("education".to_string(), vec![
        "school".to_string(),
        "student".to_string(),
        "teacher".to_string(),
        "learn".to_string(),
        "study".to_string(),
        "book".to_string(),
        "class".to_string(),
        "homework".to_string(),
        "exam".to_string(),
        "knowledge".to_string(),
        "university".to_string(),
        "degree".to_string(),
    ]);
    
    // Nature patterns
    patterns.insert("nature".to_string(), vec![
        "tree".to_string(),
        "flower".to_string(),
        "forest".to_string(),
        "mountain".to_string(),
        "river".to_string(),
        "ocean".to_string(),
        "grass".to_string(),
        "rock".to_string(),
        "earth".to_string(),
        "sky".to_string(),
        "wildlife".to_string(),
        "environment".to_string(),
    ]);
    
    // Business patterns
    patterns.insert("business".to_string(), vec![
        "company".to_string(),
        "office".to_string(),
        "work".to_string(),
        "employee".to_string(),
        "manager".to_string(),
        "meeting".to_string(),
        "project".to_string(),
        "customer".to_string(),
        "product".to_string(),
        "service".to_string(),
        "market".to_string(),
        "profit".to_string(),
    ]);
    
    patterns
}

/// Common stop words for query processing
pub fn get_stop_words() -> Vec<String> {
    vec![
        "the".to_string(),
        "a".to_string(),
        "an".to_string(),
        "and".to_string(),
        "or".to_string(),
        "but".to_string(),
        "in".to_string(),
        "on".to_string(),
        "at".to_string(),
        "to".to_string(),
        "for".to_string(),
        "of".to_string(),
        "with".to_string(),
        "by".to_string(),
        "is".to_string(),
        "are".to_string(),
        "was".to_string(),
        "were".to_string(),
        "be".to_string(),
        "been".to_string(),
        "have".to_string(),
        "has".to_string(),
        "had".to_string(),
        "do".to_string(),
        "does".to_string(),
        "did".to_string(),
        "will".to_string(),
        "would".to_string(),
        "could".to_string(),
        "should".to_string(),
        "may".to_string(),
        "might".to_string(),
        "must".to_string(),
        "can".to_string(),
        "this".to_string(),
        "that".to_string(),
        "these".to_string(),
        "those".to_string(),
        "i".to_string(),
        "you".to_string(),
        "he".to_string(),
        "she".to_string(),
        "it".to_string(),
        "we".to_string(),
        "they".to_string(),
        "me".to_string(),
        "him".to_string(),
        "her".to_string(),
        "us".to_string(),
        "them".to_string(),
        "my".to_string(),
        "your".to_string(),
        "his".to_string(),
        "her".to_string(),
        "its".to_string(),
        "our".to_string(),
        "their".to_string(),
        "what".to_string(),
        "which".to_string(),
        "who".to_string(),
        "whom".to_string(),
        "whose".to_string(),
        "when".to_string(),
        "where".to_string(),
        "why".to_string(),
        "how".to_string(),
        "all".to_string(),
        "any".to_string(),
        "both".to_string(),
        "each".to_string(),
        "few".to_string(),
        "more".to_string(),
        "most".to_string(),
        "other".to_string(),
        "some".to_string(),
        "such".to_string(),
        "only".to_string(),
        "own".to_string(),
        "same".to_string(),
        "so".to_string(),
        "than".to_string(),
        "too".to_string(),
        "very".to_string(),
        "just".to_string(),
        "now".to_string(),
        "here".to_string(),
        "there".to_string(),
        "up".to_string(),
        "down".to_string(),
        "out".to_string(),
        "off".to_string(),
        "over".to_string(),
        "under".to_string(),
        "again".to_string(),
        "further".to_string(),
        "then".to_string(),
        "once".to_string(),
    ]
}

/// Exploration type keywords for type inference
pub fn get_exploration_type_keywords() -> HashMap<String, Vec<String>> {
    let mut keywords = HashMap::new();
    
    keywords.insert("creative".to_string(), vec![
        "creative".to_string(),
        "innovative".to_string(),
        "original".to_string(),
        "unique".to_string(),
        "novel".to_string(),
        "artistic".to_string(),
        "imaginative".to_string(),
        "inventive".to_string(),
    ]);
    
    keywords.insert("analytical".to_string(), vec![
        "analyze".to_string(),
        "logical".to_string(),
        "systematic".to_string(),
        "structured".to_string(),
        "methodical".to_string(),
        "rational".to_string(),
        "scientific".to_string(),
        "precise".to_string(),
    ]);
    
    keywords.insert("associative".to_string(), vec![
        "connect".to_string(),
        "relate".to_string(),
        "associate".to_string(),
        "link".to_string(),
        "similar".to_string(),
        "comparable".to_string(),
        "parallel".to_string(),
        "analogous".to_string(),
    ]);
    
    keywords.insert("exploratory".to_string(), vec![
        "explore".to_string(),
        "discover".to_string(),
        "investigate".to_string(),
        "examine".to_string(),
        "research".to_string(),
        "study".to_string(),
        "probe".to_string(),
        "delve".to_string(),
    ]);
    
    keywords
}

/// Creativity boost factors for different connection types
pub fn get_creativity_boost_factors() -> HashMap<String, f32> {
    let mut factors = HashMap::new();
    
    factors.insert("cross_domain".to_string(), 1.5);
    factors.insert("distant_connection".to_string(), 1.3);
    factors.insert("novel_path".to_string(), 1.4);
    factors.insert("unexpected_link".to_string(), 1.6);
    factors.insert("metaphorical".to_string(), 1.2);
    factors.insert("analogical".to_string(), 1.1);
    factors.insert("semantic_leap".to_string(), 1.3);
    factors.insert("conceptual_bridge".to_string(), 1.4);
    
    factors
}

/// Relevance weight factors for different aspects
pub fn get_relevance_weights() -> HashMap<String, f32> {
    let mut weights = HashMap::new();
    
    weights.insert("semantic".to_string(), 0.4);
    weights.insert("hierarchical".to_string(), 0.3);
    weights.insert("lexical".to_string(), 0.2);
    weights.insert("domain".to_string(), 0.1);
    
    weights
}

/// Novelty calculation parameters
pub fn get_novelty_parameters() -> HashMap<String, f32> {
    let mut params = HashMap::new();
    
    params.insert("path_length_weight".to_string(), 0.3);
    params.insert("connection_strength_weight".to_string(), 0.4);
    params.insert("concept_diversity_weight".to_string(), 0.3);
    params.insert("min_novelty_threshold".to_string(), 0.1);
    params.insert("max_novelty_boost".to_string(), 2.0);
    
    params
}