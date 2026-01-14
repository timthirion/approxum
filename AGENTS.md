# approxum

A Rust library for geometric approximation algorithms.

## Philosophy

Not everything needs to be exact. **approxum** provides algorithms for when you need practical results — trading precision for speed, simplicity, or tractability.

---

## Project Status

- **CI**: GitHub Actions (test, clippy, fmt)
- **Features**: `simd` (SIMD acceleration), `serde` (serialization)
- **MSRV**: Rust 1.70+

---

## Modules

### primitives/

Core geometric types generic over `f32`/`f64`.

- `Point2<F>`, `Point3<F>` — Points
- `Vec2<F>`, `Vec3<F>` — Vectors
- `Segment2<F>`, `Segment3<F>` — Line segments
- `Ellipse2<F>` — Ellipse with local coordinate transforms

### simplify/

Polyline simplification algorithms.

| Algorithm | Description | Complexity |
|-----------|-------------|------------|
| `rdp` | Ramer-Douglas-Peucker | O(n²) worst, O(n log n) typical |
| `visvalingam` | Area-based simplification | O(n log n) |
| `radial` | Radial distance filtering | O(n) |
| `topology` | Topology-preserving simplification | O(n²) |

### curves/

Curve discretization, fitting, and operations.

- `CubicBezier2`, `QuadraticBezier2` — Bézier curves with adaptive discretization
- `Arc2` — Circular arcs
- `fit_cubic` — Least-squares Bézier fitting
- `offset_cubic_to_polyline` — Parallel curve generation
- `intersect_cubic_cubic` — Curve intersection detection

### polygon/

Polygon operations including boolean operations.

**Boolean Operations** (fixed boundary-tracing algorithm):
- `polygon_union` — Union of two polygons
- `polygon_intersection` — Intersection of two polygons
- `polygon_difference` — Difference (A - B)
- `polygon_xor` — Symmetric difference

**Other Operations**:
- `offset_polygon` — Polygon inflation/deflation with miter/bevel/round joins
- `stroke_polyline` — Convert paths to outline polygons
- `straight_skeleton` — Straight skeleton computation
- `minkowski_sum`, `minkowski_difference` — Shape dilation/erosion
- `visibility_polygon` — Compute visible area from a point
- `triangulate_polygon` — Ear clipping triangulation
- `convex_decomposition` — Split concave polygons into convex pieces

### bounds/

Bounding volume computation.

| Type | Description |
|------|-------------|
| `Aabb2` | Axis-aligned bounding box |
| `Obb2` | Oriented bounding box (tighter fit) |
| `BoundingCircle` | Minimum enclosing circle (Welzl's algorithm) |
| `BoundingCapsule` | Capsule fitting |

### spatial/

Spatial data structures.

- `Bvh` — Bounding volume hierarchy with SAH construction
- `KdTree` — K-d tree for nearest neighbor queries

### triangulation/

Triangulation and diagram algorithms.

- `delaunay_triangulation` — Delaunay triangulation
- `voronoi_diagram` — Voronoi diagram construction

### sampling/

Point generation.

- `poisson_disk` — Blue noise sampling (Bridson's algorithm)
- `sobol_sequence` — Low-discrepancy Sobol sequences

### distance/

Distance fields and signed distance functions.

- `sdf_circle`, `sdf_polygon` — Signed distance functions
- `distance_transform` — Euclidean distance transform (O(n))
- `SdfGrid` — Grid-based SDF with bilinear sampling

### tolerance/

Epsilon-aware geometric operations.

**Predicates**:
- `orient2d`, `point_on_segment`, `segments_intersect` — With explicit epsilon

**Cleanup**:
- `weld_vertices` — Merge nearby points
- `snap_to_grid` — Quantize to grid
- `remove_degenerate_edges` — Remove zero-length edges

**Metrics**:
- `hausdorff_distance` — Maximum deviation between shapes
- `frechet_distance` — Curve similarity (discrete approximation)

### hull/

Convex hull computation.

- `convex_hull` — Graham scan algorithm

### io/

Input/output utilities.

- `parse_svg_path` — Parse SVG path commands
- `svg_path_to_polylines` — Convert SVG paths to polylines
- `polyline_to_svg_path` — Export polylines to SVG

### simd/ (feature = "simd")

SIMD-accelerated operations using the `wide` crate.

- Batch point distance calculations
- Vectorized curve evaluation

---

## Design Principles

1. **Explicit tolerances** — No hidden epsilons; caller specifies tolerance
2. **Fallible operations** — Return `Option`/`Result` when operations can fail
3. **Generic over float type** — Support both `f32` and `f64`
4. **No allocations in hot paths** — Preallocate where possible
5. **SIMD-friendly** — Data layouts amenable to vectorization

---

## Dependencies

```toml
[dependencies]
num-traits = "0.2"
thiserror = "1.0"

[features]
default = []
simd = ["wide"]          # SIMD acceleration
serde = ["dep:serde"]    # Serialization
```

---

## Recent Changes

### Polygon Boolean Operations Fix

The `polygon_union` function was fixed to use proper boundary-tracing instead of convex hull or angle-sorting approaches. The algorithm now:

1. Finds all intersection points between polygon edges
2. Sorts intersections along each edge by parameter t
3. Traces the outer boundary by following edges and switching polygons at intersections
4. Correctly handles the loop termination when returning to the start

This produces correct results for overlapping convex polygons (e.g., the "peanut" shape from two overlapping circles).

### Gallery Images

All gallery images in `screenshots/` now use consistent dark backgrounds (`#1a1a2e`) for better visibility on both light and dark themes.

---

## Example Usage

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

---

## References

- Ramer, U. (1972). "An iterative procedure for the polygonal approximation of plane curves"
- Douglas, D. & Peucker, T. (1973). "Algorithms for the reduction of the number of points required to represent a digitized line"
- Visvalingam, M. & Whyatt, J.D. (1993). "Line generalisation by repeated elimination of points"
- Welzl, E. (1991). "Smallest enclosing disks (balls and ellipsoids)"
- Bridson, R. (2007). "Fast Poisson disk sampling in arbitrary dimensions"
