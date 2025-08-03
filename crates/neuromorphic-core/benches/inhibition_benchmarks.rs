//! Performance benchmarks for lateral inhibition network

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neuromorphic_core::spiking_column::{
    InhibitionConfig, LateralInhibitionNetwork, SpikingCorticalColumn,
};
use std::time::Duration;

fn benchmark_winner_take_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("winner_take_all");
    
    for size in [10, 50, 100, 200, 400].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
                
                // Register columns in a grid
                let grid_size = (size as f32).sqrt() as usize;
                for i in 0..size {
                    let x = (i % grid_size) as f32;
                    let y = (i / grid_size) as f32;
                    network.register_column(i as u32, (x, y, 0.0));
                }
                
                b.iter(|| {
                    let candidates: Vec<_> = (0..size)
                        .map(|i| (i as u32, 0.5 + (i as f32 * 0.001)))
                        .collect();
                    black_box(network.compete(candidates))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_spatial_inhibition(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_inhibition");
    
    for radius in [1.0, 3.0, 5.0, 10.0, 20.0].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(radius),
            radius,
            |b, &radius| {
                let config = InhibitionConfig {
                    radius: radius,
                    spatial_decay: 0.3,
                    ..Default::default()
                };
                
                let network = LateralInhibitionNetwork::new(config);
                
                // Create 100 columns in a 10x10 grid
                for i in 0..100 {
                    let x = (i % 10) as f32;
                    let y = (i / 10) as f32;
                    network.register_column(i, (x, y, 0.0));
                }
                
                b.iter(|| {
                    let candidates: Vec<_> = (0..100)
                        .map(|i| (i as u32, 0.3 + (i as f32 * 0.005)))
                        .collect();
                    black_box(network.compete(candidates))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_inhibition_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("inhibition_application");
    
    group.bench_function("apply_single", |b| {
        let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
        
        b.iter(|| {
            network.apply_inhibition(black_box(1), black_box(0.7));
        });
    });
    
    group.bench_function("apply_batch_100", |b| {
        let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
        
        b.iter(|| {
            for i in 0..100 {
                network.apply_inhibition(black_box(i), black_box(0.5));
            }
        });
    });
    
    group.bench_function("clear_all_100", |b| {
        let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
        
        // Pre-populate
        for i in 0..100 {
            network.apply_inhibition(i, 0.5);
        }
        
        b.iter(|| {
            network.clear_all_inhibition();
            // Re-populate for next iteration
            for i in 0..100 {
                network.apply_inhibition(i, 0.5);
            }
        });
    });
    
    group.finish();
}

fn benchmark_column_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_integration");
    
    group.bench_function("column_network_check", |b| {
        let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
        let column = SpikingCorticalColumn::new(1);
        
        network.apply_inhibition(1, 0.7);
        
        b.iter(|| {
            black_box(column.is_inhibited_by(&network))
        });
    });
    
    group.bench_function("column_apply_inhibition", |b| {
        let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
        let column = SpikingCorticalColumn::new(1);
        
        column.activate_with_strength(0.9).unwrap();
        network.apply_inhibition(1, 0.5);
        
        b.iter(|| {
            column.apply_network_inhibition(&network);
        });
    });
    
    group.finish();
}

fn benchmark_distance_calculation(c: &mut Criterion) {
    c.bench_function("euclidean_distance_3d", |b| {
        let pos1: (f32, f32, f32) = (1.0, 2.0, 3.0);
        let pos2: (f32, f32, f32) = (4.0, 5.0, 6.0);
        
        b.iter(|| {
            let dx = pos1.0 - pos2.0;
            let dy = pos1.1 - pos2.1;
            let dz = pos1.2 - pos2.2;
            black_box((dx * dx + dy * dy + dz * dz).sqrt())
        });
    });
}

fn benchmark_competition_at_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_competition");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("1000_columns", |b| {
        let config = InhibitionConfig {
            base_strength: 0.6,
            spatial_decay: 0.2,
            radius: 15.0,
            competition_threshold: 0.01,
        };
        
        let network = LateralInhibitionNetwork::new(config);
        
        // Create 1000 columns in a 32x32 grid (1024 positions)
        for i in 0..1000 {
            let x = (i % 32) as f32;
            let y = (i / 32) as f32;
            network.register_column(i, (x, y, 0.0));
        }
        
        b.iter(|| {
            let candidates: Vec<_> = (0..1000)
                .map(|i| {
                    let activation = 0.3 + (i as f32 * 0.0005) + ((i % 13) as f32 * 0.02);
                    (i as u32, activation)
                })
                .collect();
            black_box(network.compete(candidates))
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_winner_take_all,
    benchmark_spatial_inhibition,
    benchmark_inhibition_application,
    benchmark_column_integration,
    benchmark_distance_calculation,
    benchmark_competition_at_scale
);

criterion_main!(benches);