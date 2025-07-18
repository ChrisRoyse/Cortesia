// Simple Zero-Copy Benchmark for Phase 5.1 validation
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llmkg::core::types::EntityData;
use llmkg::storage::zero_copy::{ZeroCopySerializer, ZeroCopyDeserializer};

/// Generate test entities
fn generate_entities(count: usize) -> Vec<EntityData> {
    (0..count)
        .map(|i| EntityData {
            type_id: (i % 10) as u16,
            properties: format!(r#"{{"id": {}, "name": "Entity {}", "data": "test"}}"#, i, i),
            embedding: (0..96).map(|j| (i + j) as f32 / 1000.0).collect(),
        })
        .collect()
}

fn benchmark_zero_copy(c: &mut Criterion) {
    let entities = generate_entities(1000);
    
    // Test serialization
    c.bench_function("zero_copy_serialize_1k", |b| {
        b.iter(|| {
            let mut serializer = ZeroCopySerializer::new();
            for entity in black_box(&entities) {
                serializer.add_entity(entity, 96).unwrap();
            }
            black_box(serializer.finalize().unwrap())
        })
    });
    
    // Test deserialization  
    let mut serializer = ZeroCopySerializer::new();
    for entity in &entities {
        serializer.add_entity(entity, 96).unwrap();
    }
    let data = serializer.finalize().unwrap();
    
    c.bench_function("zero_copy_deserialize_1k", |b| {
        b.iter(|| {
            black_box(unsafe { ZeroCopyDeserializer::new(black_box(&data)).unwrap() })
        })
    });
    
    // Test access
    let deserializer = unsafe { ZeroCopyDeserializer::new(&data).unwrap() };
    c.bench_function("zero_copy_access_1k", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(deserializer.get_entity(i as u32));
            }
        })
    });
}

criterion_group!(zero_copy_benches, benchmark_zero_copy);
criterion_main!(zero_copy_benches);