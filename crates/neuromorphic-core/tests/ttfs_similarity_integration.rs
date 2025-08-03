//! Integration tests for TTFS concept similarity

use neuromorphic_core::ttfs_concept::{
    ConceptBuilder, ConceptSimilarity, SimilarityConfig, FastSimilarity, TTFSConcept
};
use std::collections::HashMap;

/// Test real-world animal concept similarity
#[test]
fn test_animal_taxonomy_similarity() {
    // Create animal concepts with biologically-inspired features
    let mammals = vec![
        ("dog", vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7]), // Domestic, social
        ("wolf", vec![0.8, 0.9, 0.7, 0.6, 0.4, 0.7, 0.9, 0.8]), // Wild, pack hunter
        ("cat", vec![0.9, 0.7, 0.6, 0.5, 0.8, 0.9, 0.6, 0.5]),  // Domestic, solitary
        ("lion", vec![0.7, 0.9, 0.8, 0.7, 0.3, 0.6, 0.8, 0.9]), // Wild, social predator
    ];
    
    let birds = vec![
        ("eagle", vec![0.3, 0.4, 0.9, 0.8, 0.7, 0.3, 0.9, 0.8]),  // Flying predator
        ("sparrow", vec![0.4, 0.3, 0.8, 0.7, 0.9, 0.4, 0.7, 0.6]), // Small flying
    ];
    
    let reptiles = vec![
        ("snake", vec![0.1, 0.2, 0.3, 0.9, 0.8, 0.1, 0.2, 0.9]),   // Limbless
        ("lizard", vec![0.2, 0.3, 0.4, 0.8, 0.7, 0.2, 0.3, 0.8]),  // Small reptile
    ];
    
    // Build concepts
    let mut concepts = HashMap::new();
    
    for (name, features) in mammals.iter().chain(birds.iter()).chain(reptiles.iter()) {
        let concept = ConceptBuilder::new()
            .name(*name)
            .features(features.clone())
            .tag("animal")
            .build()
            .unwrap();
        concepts.insert(name.to_string(), concept);
    }
    
    // Test similarity within and across taxonomic groups
    let calc = ConceptSimilarity::new(SimilarityConfig::default());
    
    // Dogs should be most similar to wolves
    let dog = &concepts["dog"];
    let wolf = &concepts["wolf"];
    let cat = &concepts["cat"];
    let eagle = &concepts["eagle"];
    let snake = &concepts["snake"];
    
    let dog_wolf_sim = calc.similarity(dog, wolf);
    let dog_cat_sim = calc.similarity(dog, cat);
    let dog_eagle_sim = calc.similarity(dog, eagle);
    let dog_snake_sim = calc.similarity(dog, snake);
    
    // Verify taxonomic similarity hierarchy
    assert!(dog_wolf_sim > dog_cat_sim, "Dogs should be more similar to wolves than cats");
    assert!(dog_cat_sim > dog_eagle_sim, "Mammals should be more similar to each other than to birds");
    assert!(dog_eagle_sim > dog_snake_sim, "Warm-blooded animals should be more similar than to reptiles");
    
    // Test symmetry
    assert_eq!(calc.similarity(dog, wolf), calc.similarity(wolf, dog));
    
    // Test find_most_similar
    let all_animals: Vec<TTFSConcept> = concepts.values().cloned().collect();
    let most_similar_to_dog = calc.find_most_similar(dog, &all_animals, 3);
    
    // Wolf should be the most similar to dog
    assert_eq!(most_similar_to_dog[0].0, wolf.id);
    assert!(most_similar_to_dog[0].1 > 0.7); // High similarity
}

/// Test language concept similarity
#[test]
fn test_language_concept_similarity() {
    // Create word concepts with semantic features
    // Features: [emotional_valence, size, animacy, concreteness, frequency, complexity, formality, temporal]
    let words = vec![
        ("happy", vec![0.9, 0.5, 0.7, 0.6, 0.8, 0.3, 0.5, 0.5]),
        ("joyful", vec![0.95, 0.5, 0.7, 0.6, 0.6, 0.4, 0.6, 0.5]),
        ("sad", vec![0.1, 0.5, 0.7, 0.6, 0.8, 0.3, 0.5, 0.5]),
        ("table", vec![0.5, 0.7, 0.0, 0.9, 0.9, 0.2, 0.5, 0.0]),
        ("chair", vec![0.5, 0.6, 0.0, 0.9, 0.9, 0.2, 0.5, 0.0]),
        ("run", vec![0.6, 0.5, 0.8, 0.7, 0.9, 0.3, 0.4, 0.7]),
        ("sprint", vec![0.7, 0.5, 0.8, 0.7, 0.5, 0.4, 0.5, 0.8]),
    ];
    
    let mut concepts = HashMap::new();
    for (word, features) in words.iter() {
        let concept = ConceptBuilder::new()
            .name(*word)
            .features(features.clone())
            .tag("language")
            .build()
            .unwrap();
        concepts.insert(word.to_string(), concept);
    }
    
    let calc = ConceptSimilarity::new(SimilarityConfig::default());
    
    // Test synonym similarity
    let happy = &concepts["happy"];
    let joyful = &concepts["joyful"];
    let sad = &concepts["sad"];
    
    let synonym_sim = calc.similarity(happy, joyful);
    let antonym_sim = calc.similarity(happy, sad);
    
    println!("Synonym similarity (happy-joyful): {}", synonym_sim);
    println!("Antonym similarity (happy-sad): {}", antonym_sim);
    
    assert!(synonym_sim > 0.6, "Synonyms should have high similarity (got {})", synonym_sim);
    assert!(antonym_sim < 0.7, "Antonyms should have lower similarity than synonyms (got {})", antonym_sim);
    
    // Test furniture similarity
    let table = &concepts["table"];
    let chair = &concepts["chair"];
    let furniture_sim = calc.similarity(table, chair);
    
    assert!(furniture_sim > 0.7, "Related objects should have high similarity");
    
    // Test action verb similarity
    let run = &concepts["run"];
    let sprint = &concepts["sprint"];
    let action_sim = calc.similarity(run, sprint);
    
    println!("Action similarity (run-sprint): {}", action_sim);
    assert!(action_sim > 0.65, "Related actions should have high similarity (got {})", action_sim);
}

/// Test LSH performance and accuracy trade-off
#[test]
fn test_lsh_approximation_quality() {
    let feature_dim = 128;
    let num_concepts = 100;
    
    // Create a set of random concepts
    let mut concepts = Vec::new();
    for i in 0..num_concepts {
        let mut features = vec![0.5; feature_dim];
        // Add variation based on concept index
        for j in 0..feature_dim {
            features[j] += ((i * j) as f32 / (num_concepts * feature_dim) as f32) - 0.25;
            features[j] = features[j].clamp(0.0, 1.0);
        }
        
        let concept = ConceptBuilder::new()
            .name(&format!("concept_{}", i))
            .features(features)
            .build()
            .unwrap();
        concepts.push(concept);
    }
    
    // Compare exact vs approximate similarity
    let calc = ConceptSimilarity::new(SimilarityConfig::default());
    let fast_sim = FastSimilarity::new_with_seed(feature_dim, 32, 42);
    
    let mut total_error = 0.0;
    let mut comparisons = 0;
    
    // Sample pairs for comparison
    for i in 0..10 {
        for j in i+1..20 {
            let exact_sim = calc.semantic_similarity(
                &concepts[i].semantic_features,
                &concepts[j].semantic_features
            );
            
            let hash_i = fast_sim.compute_hash(&concepts[i].semantic_features);
            let hash_j = fast_sim.compute_hash(&concepts[j].semantic_features);
            let approx_sim = fast_sim.approximate_similarity(hash_i, hash_j);
            
            let error = (exact_sim - approx_sim).abs();
            total_error += error;
            comparisons += 1;
        }
    }
    
    let avg_error = total_error / comparisons as f32;
    assert!(avg_error < 0.3, "Average LSH approximation error should be reasonable");
}

/// Test batch similarity computation
#[test]
fn test_batch_similarity_performance() {
    use std::time::Instant;
    
    // Create test dataset
    let mut concepts = Vec::new();
    for i in 0..50 {
        let features = vec![0.5 + (i as f32 * 0.01); 64];
        let concept = ConceptBuilder::new()
            .name(&format!("batch_test_{}", i))
            .features(features)
            .build()
            .unwrap();
        concepts.push(concept);
    }
    
    let calc = ConceptSimilarity::new(SimilarityConfig {
        use_cache: true,
        ..Default::default()
    });
    
    // Warm up cache
    for i in 0..5 {
        for j in i+1..10 {
            calc.similarity(&concepts[i], &concepts[j]);
        }
    }
    
    // Measure batch performance
    let start = Instant::now();
    let mut similarities = Vec::new();
    
    for i in 0..concepts.len() {
        for j in i+1..concepts.len().min(i + 11) {  // i+11 to include up to 10 comparisons
            let sim = calc.similarity(&concepts[i], &concepts[j]);
            similarities.push((i, j, sim));
        }
    }
    
    let duration = start.elapsed();
    let avg_time_us = duration.as_micros() / similarities.len() as u128;
    
    println!("Batch similarity: {} comparisons in {:?}", similarities.len(), duration);
    println!("Average time per comparison: {}μs", avg_time_us);
    
    assert!(avg_time_us < 10000, "Batch similarity should be reasonably fast (got {}μs)", avg_time_us);
    
    // Calculate expected number of comparisons
    // For each concept i, we compare with min(10, n-i-1) following concepts
    let expected_comparisons: usize = (0..concepts.len())
        .map(|i| (concepts.len() - i - 1).min(10))
        .sum();
    
    assert_eq!(similarities.len(), expected_comparisons, 
        "Should have correct number of comparisons");
}

/// Test edge cases and boundary conditions
#[test]
fn test_similarity_edge_cases() {
    let calc = ConceptSimilarity::new(SimilarityConfig::default());
    
    // Empty features
    let empty_concept = ConceptBuilder::new()
        .name("empty")
        .features(vec![])
        .build()
        .unwrap();
    
    // Zero features
    let zero_concept = ConceptBuilder::new()
        .name("zero")
        .features(vec![0.0; 32])
        .build()
        .unwrap();
    
    // Max features
    let max_concept = ConceptBuilder::new()
        .name("max")
        .features(vec![1.0; 32])
        .build()
        .unwrap();
    
    // Test empty concept similarity - empty features should give 0 similarity
    if empty_concept.semantic_features.is_empty() {
        let empty_sim = calc.semantic_similarity(&empty_concept.semantic_features, &empty_concept.semantic_features);
        assert_eq!(empty_sim, 0.0, "Empty semantic features should have 0 similarity");
    }
    
    // Test zero vector similarity
    let zero_sim = calc.semantic_similarity(&zero_concept.semantic_features, &zero_concept.semantic_features);
    assert_eq!(zero_sim, 0.0, "Zero vectors should have 0 similarity");
    
    // Test max vector similarity
    let max_sim = calc.similarity(&max_concept, &max_concept);
    assert!((max_sim - 1.0).abs() < 0.01, "Identical max vectors should have ~1.0 similarity");
    
    // Test different length features (should return 0)
    let features1 = vec![0.5; 32];
    let features2 = vec![0.5; 64];
    let diff_len_sim = calc.semantic_similarity(&features1, &features2);
    assert_eq!(diff_len_sim, 0.0, "Different length features should have 0 similarity");
}