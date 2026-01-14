//! Benchmarks for curve operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use approxum::curves::{fit_cubic, Arc2, CubicBezier2, QuadraticBezier2};
use approxum::Point2;

fn bench_quadratic_bezier_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("quadratic_bezier_eval");

    let curve = QuadraticBezier2::new(
        Point2::new(0.0, 0.0),
        Point2::new(5.0, 10.0),
        Point2::new(10.0, 0.0),
    );

    // Single evaluation
    group.bench_function("single", |b| b.iter(|| curve.eval(black_box(0.5))));

    // Multiple evaluations
    for count in [10, 100, 1000] {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("batch", count), &count, |b, &count| {
            b.iter(|| {
                for i in 0..count {
                    let t = i as f64 / count as f64;
                    let _ = curve.eval(black_box(t));
                }
            })
        });
    }

    group.finish();
}

fn bench_cubic_bezier_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("cubic_bezier_eval");

    let curve = CubicBezier2::new(
        Point2::new(0.0, 0.0),
        Point2::new(3.0, 10.0),
        Point2::new(7.0, 10.0),
        Point2::new(10.0, 0.0),
    );

    group.bench_function("single", |b| b.iter(|| curve.eval(black_box(0.5))));

    for count in [10, 100, 1000] {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("batch", count), &count, |b, &count| {
            b.iter(|| {
                for i in 0..count {
                    let t = i as f64 / count as f64;
                    let _ = curve.eval(black_box(t));
                }
            })
        });
    }

    group.finish();
}

fn bench_cubic_bezier_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("cubic_bezier_split");

    let curve = CubicBezier2::new(
        Point2::new(0.0, 0.0),
        Point2::new(3.0, 10.0),
        Point2::new(7.0, 10.0),
        Point2::new(10.0, 0.0),
    );

    group.bench_function("split_at_0.5", |b| b.iter(|| curve.split(black_box(0.5))));

    group.bench_function("split_at_0.25", |b| b.iter(|| curve.split(black_box(0.25))));

    group.finish();
}

fn bench_cubic_bezier_to_polyline(c: &mut Criterion) {
    let mut group = c.benchmark_group("cubic_bezier_to_polyline");

    let curve = CubicBezier2::new(
        Point2::new(0.0, 0.0),
        Point2::new(3.0, 10.0),
        Point2::new(7.0, 10.0),
        Point2::new(10.0, 0.0),
    );

    // Vary tolerance (tighter = more points)
    for tolerance in [1.0, 0.1, 0.01, 0.001] {
        group.bench_with_input(
            BenchmarkId::new("tolerance", format!("{}", tolerance)),
            &tolerance,
            |b, &tol| b.iter(|| curve.to_polyline(black_box(tol))),
        );
    }

    group.finish();
}

fn bench_arc_to_polyline(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_to_polyline");

    // Create arc from three points
    let arc = Arc2::from_three_points(
        Point2::new(0.0, 0.0),
        Point2::new(5.0, 5.0),
        Point2::new(10.0, 0.0),
    )
    .unwrap();

    for tolerance in [1.0, 0.1, 0.01, 0.001] {
        group.bench_with_input(
            BenchmarkId::new("tolerance", format!("{}", tolerance)),
            &tolerance,
            |b, &tol| b.iter(|| arc.to_polyline(black_box(tol))),
        );
    }

    group.finish();
}

fn bench_arc_from_three_points(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_from_three_points");

    let p1 = Point2::new(0.0, 0.0);
    let p2 = Point2::new(5.0, 5.0);
    let p3 = Point2::new(10.0, 0.0);

    group.bench_function("construct", |b| {
        b.iter(|| Arc2::from_three_points(black_box(p1), black_box(p2), black_box(p3)))
    });

    group.finish();
}

/// Generate points roughly along a cubic bezier for fitting tests.
fn generate_curve_points(num_points: usize) -> Vec<Point2<f64>> {
    let curve = CubicBezier2::new(
        Point2::new(0.0, 0.0),
        Point2::new(3.0, 10.0),
        Point2::new(7.0, 10.0),
        Point2::new(10.0, 0.0),
    );

    (0..num_points)
        .map(|i| {
            let t = i as f64 / (num_points - 1) as f64;
            let p = curve.eval(t);
            // Add small deterministic noise
            let noise = ((i * 17) % 100) as f64 / 10000.0;
            Point2::new(p.x + noise, p.y + noise)
        })
        .collect()
}

fn bench_fit_cubic(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_cubic");

    for num_points in [10, 50, 100, 200] {
        let points = generate_curve_points(num_points);
        group.throughput(Throughput::Elements(num_points as u64));

        group.bench_with_input(BenchmarkId::new("points", num_points), &points, |b, pts| {
            b.iter(|| fit_cubic(black_box(pts)))
        });
    }

    group.finish();
}

fn bench_comparison_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("curve_eval_comparison");

    let quad = QuadraticBezier2::new(
        Point2::new(0.0, 0.0),
        Point2::new(5.0, 10.0),
        Point2::new(10.0, 0.0),
    );

    let cubic = CubicBezier2::new(
        Point2::new(0.0, 0.0),
        Point2::new(3.0, 10.0),
        Point2::new(7.0, 10.0),
        Point2::new(10.0, 0.0),
    );

    group.bench_function("quadratic_1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let t = i as f64 / 1000.0;
                let _ = quad.eval(black_box(t));
            }
        })
    });

    group.bench_function("cubic_1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let t = i as f64 / 1000.0;
                let _ = cubic.eval(black_box(t));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_quadratic_bezier_eval,
    bench_cubic_bezier_eval,
    bench_cubic_bezier_split,
    bench_cubic_bezier_to_polyline,
    bench_arc_to_polyline,
    bench_arc_from_three_points,
    bench_fit_cubic,
    bench_comparison_eval
);
criterion_main!(benches);
