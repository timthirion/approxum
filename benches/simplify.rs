//! Benchmarks for polyline simplification algorithms.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use approxum::simplify::{rdp, topology_preserving, visvalingam};
use approxum::Point2;

/// Generates a noisy sine wave polyline.
fn generate_sine_wave(num_points: usize) -> Vec<Point2<f64>> {
    (0..num_points)
        .map(|i| {
            let t = i as f64 / num_points as f64 * 4.0 * std::f64::consts::PI;
            let noise = ((i * 17) % 100) as f64 / 1000.0; // Deterministic "noise"
            Point2::new(t, t.sin() + noise)
        })
        .collect()
}

/// Generates a random walk polyline.
fn generate_random_walk(num_points: usize, seed: u64) -> Vec<Point2<f64>> {
    let mut points = Vec::with_capacity(num_points);
    let mut x = 0.0;
    let mut y = 0.0;
    let mut state = seed;

    for _ in 0..num_points {
        points.push(Point2::new(x, y));

        // Simple xorshift for deterministic "random" steps
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        let angle = (state as f64 / u64::MAX as f64) * 2.0 * std::f64::consts::PI;
        x += angle.cos() * 0.1;
        y += angle.sin() * 0.1;
    }

    points
}

fn bench_rdp(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdp");

    for size in [100, 1000, 10000, 50000] {
        let points = generate_sine_wave(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("sine_wave", size), &points, |b, pts| {
            b.iter(|| rdp(black_box(pts), black_box(0.01)))
        });
    }

    // Also test with random walk data
    for size in [1000, 10000] {
        let points = generate_random_walk(size, 12345);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("random_walk", size), &points, |b, pts| {
            b.iter(|| rdp(black_box(pts), black_box(0.01)))
        });
    }

    group.finish();
}

fn bench_visvalingam(c: &mut Criterion) {
    let mut group = c.benchmark_group("visvalingam");

    for size in [100, 1000, 10000, 50000] {
        let points = generate_sine_wave(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("sine_wave", size), &points, |b, pts| {
            b.iter(|| visvalingam(black_box(pts), black_box(0.001)))
        });
    }

    for size in [1000, 10000] {
        let points = generate_random_walk(size, 12345);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("random_walk", size), &points, |b, pts| {
            b.iter(|| visvalingam(black_box(pts), black_box(0.001)))
        });
    }

    group.finish();
}

fn bench_topology_preserving(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_preserving");

    // Topology-preserving is O(n²) so use smaller sizes
    for size in [100, 500, 1000, 2000] {
        let points = generate_sine_wave(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("sine_wave", size), &points, |b, pts| {
            b.iter(|| topology_preserving(black_box(pts), black_box(0.001), black_box(1e-10)))
        });
    }

    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplify_comparison");

    let size = 5000;
    let points = generate_sine_wave(size);
    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("rdp", |b| {
        b.iter(|| rdp(black_box(&points), black_box(0.01)))
    });

    group.bench_function("visvalingam", |b| {
        b.iter(|| visvalingam(black_box(&points), black_box(0.001)))
    });

    // topology_preserving is too slow for 5000 points in a comparison benchmark

    group.finish();
}

criterion_group!(
    benches,
    bench_rdp,
    bench_visvalingam,
    bench_topology_preserving,
    bench_comparison
);
criterion_main!(benches);
