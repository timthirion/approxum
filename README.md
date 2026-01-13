# approxum

"The purpose of computing is insight, not numbers." - Richard Hamming

Geometric approximation algorithms in Rust. When good enough is good enough.

## Installation

```bash
cargo add approxum
```

## Quick Example

```rust
use approxum::{Point2, simplify, sampling, curves::CubicBezier2};

// Simplify a noisy GPS track
let track: Vec<Point2<f64>> = load_track();
let simplified = simplify::rdp(&track, 0.001);
println!("Reduced {} points to {}", track.len(), simplified.len());

// Generate blue noise points
let points = sampling::poisson_disk(100.0, 100.0, 2.5, 30);

// Discretize a Bézier curve
let curve = CubicBezier2::new(
    Point2::new(0.0, 0.0),
    Point2::new(1.0, 2.0),
    Point2::new(3.0, 2.0),
    Point2::new(4.0, 0.0),
);
let polyline = curve.to_polyline(0.01);
```

## Features

**Polyline Simplification**
- `rdp` - Ramer-Douglas-Peucker (O(n²) worst, O(n log n) typical)
- `visvalingam` - Area-based simplification (O(n log n))
- `topology_preserving` - Prevents self-intersection

**Curve Operations**
- `QuadraticBezier2`, `CubicBezier2` - Bézier curves with adaptive discretization
- `Arc2` - Circular arcs from three points or bulge factor
- `fit_cubic` - Least-squares Bézier fitting to point data

**Bounding Volumes**
- `Aabb2` - Axis-aligned bounding boxes
- `BoundingCircle` - Minimum enclosing circle (Welzl's algorithm)

**Spatial Data Structures**
- `Bvh` - Bounding volume hierarchy with SAH construction

**Sampling**
- `poisson_disk` - Blue noise sampling (Bridson's algorithm)

**Distance Fields**
- `sdf_circle`, `sdf_polygon` - Signed distance functions
- `distance_transform` - Euclidean distance transform (O(n))
- `SdfGrid` - Grid-based SDF with bilinear sampling

**Tolerance-Aware Predicates**
- `orient2d`, `point_on_segment`, `segments_intersect` - Explicit epsilon parameters

**SIMD Acceleration** (with `simd` feature)
- Batch point distance calculations
- Vectorized curve evaluation

## Gallery

### Polyline Simplification

![RDP and Visvalingam simplification comparison](screenshots/simplification.svg)

### Bézier Curves & Arc Discretization

![Cubic Bézier and circular arc with control points](screenshots/curves.svg)

### Poisson Disk Sampling

![Blue noise point distribution](screenshots/poisson.svg)

### Bounding Volume Hierarchy

![BVH tree structure visualization](screenshots/bvh.svg)

### Signed Distance Field

![SDF visualization with contour lines](screenshots/sdf.svg)

## Companion to exactum

**approxum** is designed as a companion to [exactum](https://github.com/...), which provides exact integer geometry. Use exactum when correctness is critical; use approxum for:

- Preprocessing noisy input data
- LOD generation and mesh simplification
- Visualization and rendering
- "Good enough" spatial queries

## Minimum Supported Rust Version

Rust 1.70 or later.

## License

Apache-2.0
