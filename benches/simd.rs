//! Benchmarks comparing SIMD vs scalar implementations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use approxum::curves::{CubicBezier2, QuadraticBezier2};
use approxum::primitives::{Point2, Segment2};
use approxum::simd::{
    distances_to_point, distances_to_segment, eval_cubic_batch, eval_quadratic_batch,
    nearest_point_index, points_within_radius, CubicBezier2x4,
};

/// Generates random points for benchmarking.
fn generate_points(count: usize, seed: u64) -> Vec<Point2<f32>> {
    let mut points = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = (state as f32 / u64::MAX as f32) * 100.0;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let y = (state as f32 / u64::MAX as f32) * 100.0;

        points.push(Point2::new(x, y));
    }

    points
}

fn bench_distances_to_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_to_point");

    let target = Point2::new(50.0f32, 50.0f32);

    for count in [100, 1000, 10000, 100000] {
        let points = generate_points(count, 12345);
        group.throughput(Throughput::Elements(count as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", count), &points, |b, pts| {
            b.iter(|| {
                pts.iter()
                    .map(|p| p.distance(black_box(target)))
                    .collect::<Vec<_>>()
            })
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", count), &points, |b, pts| {
            b.iter(|| distances_to_point(black_box(pts), black_box(target)))
        });
    }

    group.finish();
}

fn bench_distances_to_segment(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_to_segment");

    let segment = Segment2::new(Point2::new(0.0f32, 0.0f32), Point2::new(100.0f32, 0.0f32));

    for count in [100, 1000, 10000, 100000] {
        let points = generate_points(count, 12345);
        group.throughput(Throughput::Elements(count as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", count), &points, |b, pts| {
            b.iter(|| {
                pts.iter()
                    .map(|p| black_box(segment).distance_to_point(*p))
                    .collect::<Vec<_>>()
            })
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", count), &points, |b, pts| {
            b.iter(|| distances_to_segment(black_box(pts), black_box(segment)))
        });
    }

    group.finish();
}

fn bench_nearest_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_point");

    let target = Point2::new(50.0f32, 50.0f32);

    for count in [100, 1000, 10000, 100000] {
        let points = generate_points(count, 12345);
        group.throughput(Throughput::Elements(count as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", count), &points, |b, pts| {
            b.iter(|| {
                pts.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da = a.distance_squared(black_box(target));
                        let db = b.distance_squared(black_box(target));
                        da.partial_cmp(&db).unwrap()
                    })
                    .map(|(i, _)| i)
            })
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", count), &points, |b, pts| {
            b.iter(|| nearest_point_index(black_box(pts), black_box(target)))
        });
    }

    group.finish();
}

fn bench_points_in_radius(c: &mut Criterion) {
    let mut group = c.benchmark_group("points_in_radius");

    let target = Point2::new(50.0f32, 50.0f32);
    let radius = 10.0f32;
    let radius_sq = radius * radius;

    for count in [100, 1000, 10000, 100000] {
        let points = generate_points(count, 12345);
        group.throughput(Throughput::Elements(count as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", count), &points, |b, pts| {
            b.iter(|| {
                pts.iter()
                    .enumerate()
                    .filter(|(_, p)| p.distance_squared(black_box(target)) <= radius_sq)
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>()
            })
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", count), &points, |b, pts| {
            b.iter(|| points_within_radius(black_box(pts), black_box(target), black_box(radius)))
        });
    }

    group.finish();
}

fn bench_cubic_bezier_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("cubic_bezier_batch_eval");

    let curve = CubicBezier2::new(
        Point2::new(0.0f32, 0.0f32),
        Point2::new(3.0f32, 10.0f32),
        Point2::new(7.0f32, 10.0f32),
        Point2::new(10.0f32, 0.0f32),
    );

    for count in [16, 64, 256, 1024, 4096] {
        let params: Vec<f32> = (0..count).map(|i| i as f32 / (count - 1) as f32).collect();
        group.throughput(Throughput::Elements(count as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", count), &params, |b, ts| {
            b.iter(|| {
                ts.iter()
                    .map(|&t| black_box(&curve).eval(t))
                    .collect::<Vec<_>>()
            })
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", count), &params, |b, ts| {
            b.iter(|| eval_cubic_batch(black_box(&curve), black_box(ts)))
        });
    }

    group.finish();
}

fn bench_quadratic_bezier_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("quadratic_bezier_batch_eval");

    let curve = QuadraticBezier2::new(
        Point2::new(0.0f32, 0.0f32),
        Point2::new(5.0f32, 10.0f32),
        Point2::new(10.0f32, 0.0f32),
    );

    for count in [16, 64, 256, 1024, 4096] {
        let params: Vec<f32> = (0..count).map(|i| i as f32 / (count - 1) as f32).collect();
        group.throughput(Throughput::Elements(count as u64));

        // Scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", count), &params, |b, ts| {
            b.iter(|| {
                ts.iter()
                    .map(|&t| black_box(&curve).eval(t))
                    .collect::<Vec<_>>()
            })
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", count), &params, |b, ts| {
            b.iter(|| eval_quadratic_batch(black_box(&curve), black_box(ts)))
        });
    }

    group.finish();
}

fn bench_simd_4_vs_8(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_batch_size");

    let curve = CubicBezier2::new(
        Point2::new(0.0f32, 0.0f32),
        Point2::new(3.0f32, 10.0f32),
        Point2::new(7.0f32, 10.0f32),
        Point2::new(10.0f32, 0.0f32),
    );
    let simd_curve = CubicBezier2x4::from_curve(&curve);

    // Single batch of 4
    group.bench_function("cubic_4_evals", |b| {
        let t = wide::f32x4::new([0.0, 0.25, 0.5, 0.75]);
        b.iter(|| simd_curve.eval(black_box(t)))
    });

    // 4 individual scalar evals
    group.bench_function("cubic_4_scalar", |b| {
        b.iter(|| {
            [
                curve.eval(black_box(0.0f32)),
                curve.eval(black_box(0.25f32)),
                curve.eval(black_box(0.5f32)),
                curve.eval(black_box(0.75f32)),
            ]
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distances_to_point,
    bench_distances_to_segment,
    bench_nearest_point,
    bench_points_in_radius,
    bench_cubic_bezier_eval,
    bench_quadratic_bezier_eval,
    bench_simd_4_vs_8
);
criterion_main!(benches);
