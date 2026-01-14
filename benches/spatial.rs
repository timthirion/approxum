//! Benchmarks for spatial data structures and distance computations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use approxum::bounds::Aabb2;
use approxum::distance::{distance_transform, sdf_polygon, Circle, SdfGrid};
use approxum::spatial::Bvh;
use approxum::Point2;

/// Generates random AABBs for BVH testing.
fn generate_random_aabbs(count: usize, seed: u64) -> Vec<Aabb2<f64>> {
    let mut aabbs = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        // xorshift for deterministic random
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = (state as f64 / u64::MAX as f64) * 100.0;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let y = (state as f64 / u64::MAX as f64) * 100.0;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let w = (state as f64 / u64::MAX as f64) * 5.0 + 0.5;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let h = (state as f64 / u64::MAX as f64) * 5.0 + 0.5;

        aabbs.push(Aabb2::new(Point2::new(x, y), Point2::new(x + w, y + h)));
    }

    aabbs
}

/// Generates random query points.
fn generate_random_points(count: usize, seed: u64) -> Vec<Point2<f64>> {
    let mut points = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = (state as f64 / u64::MAX as f64) * 100.0;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let y = (state as f64 / u64::MAX as f64) * 100.0;

        points.push(Point2::new(x, y));
    }

    points
}

fn bench_bvh_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_construction");

    for count in [100, 1000, 10000, 50000] {
        let aabbs = generate_random_aabbs(count, 12345);
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("aabbs", count), &aabbs, |b, boxes| {
            b.iter(|| Bvh::build(black_box(boxes), 4))
        });
    }

    group.finish();
}

fn bench_bvh_query_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_query_point");

    for count in [1000, 10000, 50000] {
        let aabbs = generate_random_aabbs(count, 12345);
        let bvh = Bvh::build(&aabbs, 4);
        let query_points = generate_random_points(1000, 54321);

        group.throughput(Throughput::Elements(1000));

        group.bench_with_input(
            BenchmarkId::new("queries_1000", count),
            &(&bvh, &aabbs, &query_points),
            |b, (bvh, aabbs, points)| {
                b.iter(|| {
                    for p in points.iter() {
                        let _ = bvh.query_point(black_box(aabbs), black_box(*p));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_bvh_query_aabb(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_query_aabb");

    let count = 10000;
    let aabbs = generate_random_aabbs(count, 12345);
    let bvh = Bvh::build(&aabbs, 4);

    // Small query box
    let small_query = Aabb2::new(Point2::new(45.0, 45.0), Point2::new(55.0, 55.0));
    group.bench_function("small_query", |b| {
        b.iter(|| bvh.query_aabb(black_box(&aabbs), black_box(small_query)))
    });

    // Large query box
    let large_query = Aabb2::new(Point2::new(20.0, 20.0), Point2::new(80.0, 80.0));
    group.bench_function("large_query", |b| {
        b.iter(|| bvh.query_aabb(black_box(&aabbs), black_box(large_query)))
    });

    group.finish();
}

fn bench_distance_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_transform");

    for size in [64, 128, 256, 512] {
        // Create a grid with some feature points (circle pattern)
        let mut grid = vec![false; size * size];
        let center = size / 2;
        let radius = size / 4;

        for y in 0..size {
            for x in 0..size {
                let dx = x as i64 - center as i64;
                let dy = y as i64 - center as i64;
                if dx * dx + dy * dy <= (radius * radius) as i64 {
                    grid[y * size + x] = true;
                }
            }
        }

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("circle", size),
            &(&grid, size),
            |b, (grid, size)| {
                b.iter(|| {
                    distance_transform::<f64>(black_box(grid), black_box(*size), black_box(*size))
                })
            },
        );
    }

    group.finish();
}

fn bench_sdf_polygon(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdf_polygon");

    // Simple square
    let square = vec![
        Point2::new(0.0, 0.0),
        Point2::new(10.0, 0.0),
        Point2::new(10.0, 10.0),
        Point2::new(0.0, 10.0),
    ];

    // Complex polygon (many vertices)
    let complex: Vec<Point2<f64>> = (0..100)
        .map(|i| {
            let angle = i as f64 / 100.0 * 2.0 * std::f64::consts::PI;
            let r = 5.0 + (angle * 5.0).sin();
            Point2::new(r * angle.cos(), r * angle.sin())
        })
        .collect();

    let query_points = generate_random_points(1000, 99999);

    group.bench_function("square_1000_queries", |b| {
        b.iter(|| {
            for p in &query_points {
                let _ = sdf_polygon(black_box(*p), black_box(&square));
            }
        })
    });

    group.bench_function("complex_100v_1000_queries", |b| {
        b.iter(|| {
            for p in &query_points {
                let _ = sdf_polygon(black_box(*p), black_box(&complex));
            }
        })
    });

    group.finish();
}

fn bench_sdf_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdf_grid");

    let circle = Circle::new(Point2::new(50.0, 50.0), 30.0);

    // Grid generation
    for size in [64, 128, 256] {
        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("generate", size), &size, |b, &size| {
            b.iter(|| {
                SdfGrid::from_shape(
                    black_box(&circle),
                    black_box(size),
                    black_box(size),
                    black_box(Point2::new(0.0, 0.0)),
                    black_box(100.0 / size as f64),
                )
            })
        });
    }

    // Grid sampling
    let grid = SdfGrid::from_shape(&circle, 256, 256, Point2::new(0.0, 0.0), 100.0 / 256.0);
    let query_points = generate_random_points(10000, 11111);

    group.bench_function("sample_10000_queries", |b| {
        b.iter(|| {
            for p in &query_points {
                let _ = grid.sample(black_box(*p));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bvh_construction,
    bench_bvh_query_point,
    bench_bvh_query_aabb,
    bench_distance_transform,
    bench_sdf_polygon,
    bench_sdf_grid
);
criterion_main!(benches);
