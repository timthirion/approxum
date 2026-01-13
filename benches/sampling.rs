//! Benchmarks for point sampling algorithms.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use approxum::sampling::{poisson_disk, poisson_disk_with_seed, PoissonDiskSampler};

fn bench_poisson_disk(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_disk");

    // Vary domain size with fixed density
    for size in [10, 50, 100, 200] {
        let min_distance = 1.0;
        let size_f = size as f64;

        group.bench_with_input(
            BenchmarkId::new("domain_size", size),
            &(size_f, min_distance),
            |b, &(size, min_dist)| {
                b.iter(|| {
                    poisson_disk::<f64>(
                        black_box(size),
                        black_box(size),
                        black_box(min_dist),
                        black_box(30),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_poisson_disk_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_disk_density");

    // Fixed domain, vary density
    let size = 100.0;

    for min_dist in [0.5, 1.0, 2.0, 5.0] {
        group.bench_with_input(
            BenchmarkId::new("min_distance", format!("{:.1}", min_dist)),
            &min_dist,
            |b, &min_dist| {
                b.iter(|| {
                    poisson_disk::<f64>(
                        black_box(size),
                        black_box(size),
                        black_box(min_dist),
                        black_box(30),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_poisson_disk_attempts(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_disk_attempts");

    // Vary max_attempts to see impact on quality vs speed
    let size = 50.0;
    let min_dist = 1.0;

    for attempts in [10, 20, 30, 50] {
        group.bench_with_input(
            BenchmarkId::new("max_attempts", attempts),
            &attempts,
            |b, &attempts| {
                b.iter(|| {
                    poisson_disk::<f64>(
                        black_box(size),
                        black_box(size),
                        black_box(min_dist),
                        black_box(attempts),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_poisson_disk_seeded(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_disk_seeded");

    let size = 100.0;
    let min_dist = 1.0;

    // Compare seeded vs non-seeded (should be similar)
    group.bench_function("with_seed", |b| {
        b.iter(|| {
            poisson_disk_with_seed::<f64>(
                black_box(size),
                black_box(size),
                black_box(min_dist),
                black_box(30),
                black_box(12345),
            )
        })
    });

    group.bench_function("auto_seed", |b| {
        b.iter(|| {
            poisson_disk::<f64>(
                black_box(size),
                black_box(size),
                black_box(min_dist),
                black_box(30),
            )
        })
    });

    group.finish();
}

fn bench_poisson_sampler_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_sampler_reuse");

    let size = 50.0;
    let min_dist = 1.0;

    // Fresh sampler each time
    group.bench_function("fresh_sampler", |b| {
        b.iter(|| {
            let mut sampler: PoissonDiskSampler<f64> =
                PoissonDiskSampler::new(size, size, min_dist, 12345);
            sampler.generate(30)
        })
    });

    // Reused sampler (note: RNG state advances, so results differ)
    let mut sampler: PoissonDiskSampler<f64> = PoissonDiskSampler::new(size, size, min_dist, 12345);
    group.bench_function("reused_sampler", |b| {
        b.iter(|| sampler.generate(black_box(30)))
    });

    group.finish();
}

fn bench_poisson_f32_vs_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_f32_vs_f64");

    let size = 100.0;
    let min_dist = 1.0;

    group.bench_function("f64", |b| {
        b.iter(|| {
            poisson_disk::<f64>(
                black_box(size),
                black_box(size),
                black_box(min_dist),
                black_box(30),
            )
        })
    });

    group.bench_function("f32", |b| {
        b.iter(|| {
            poisson_disk::<f32>(
                black_box(100.0f32),
                black_box(100.0f32),
                black_box(1.0f32),
                black_box(30),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_poisson_disk,
    bench_poisson_disk_density,
    bench_poisson_disk_attempts,
    bench_poisson_disk_seeded,
    bench_poisson_sampler_reuse,
    bench_poisson_f32_vs_f64
);
criterion_main!(benches);
