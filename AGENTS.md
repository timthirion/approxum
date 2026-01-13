# approxum

A Rust library for geometric approximation algorithms.

## Philosophy

Not everything needs to be exact. **approxum** provides algorithms for when you need practical results — trading precision for speed, simplicity, or tractability.

Companion to [exactum](https://github.com/...) (exact integer geometry). Use exactum when correctness is critical; use approxum for preprocessing, optimization, and "good enough" computation.

---

## Modules

### simplify/

Polyline and polygon simplification.

| Algorithm | Description | Complexity |
|-----------|-------------|------------|
| `rdp` | Ramer-Douglas-Peucker | O(n²) worst, O(n log n) typical |
| `visvalingam` | Area-based simplification | O(n log n) |
| `radial` | Radial distance filtering | O(n) |
| `topology` | Topology-preserving simplification | O(n²) |

**Use cases:** LOD generation, GIS data reduction, GPS track compression

### curves/

Curve discretization and fitting.

**Discretization (curve → polyline):**
- `bezier_to_polyline` — Adaptive subdivision
- `arc_to_polyline` — Circular arc approximation
- `ellipse_to_polygon` — Ellipse discretization

**Fitting (points → curve):**
- `fit_bezier` — Least-squares Bézier fitting
- `fit_arc` — Circular arc fitting
- `fit_line` — Linear regression

**Use cases:** CNC toolpaths, font rendering, CAD import/export

### bounds/

Bounding volume computation.

| Type | Description |
|------|-------------|
| `Aabb` | Axis-aligned bounding box |
| `Obb` | Oriented bounding box (tighter fit) |
| `BoundingCircle` | Minimum enclosing circle |
| `BoundingCapsule` | Capsule (cylinder + hemispheres) |
| `ConvexHull` | Convex hull as bounding volume |

**Hierarchies:**
- `Bvh` — Bounding volume hierarchy (SAH construction)
- `LooseQuadtree` — Spatial index for dynamic objects

**Use cases:** Collision broad-phase, frustum culling, spatial queries

### tolerance/

Epsilon-aware geometric operations.

**Predicates with tolerance:**
- `point_on_line(p, line, eps)` — Is point within epsilon of line?
- `segments_intersect(s1, s2, eps)` — Intersection with tolerance
- `polygons_equal(p1, p2, eps)` — Approximate equality

**Cleanup operations:**
- `weld_vertices(points, eps)` — Merge nearby points
- `snap_to_grid(points, grid_size)` — Quantize to grid
- `remove_degenerate_edges(poly, eps)` — Remove zero-length edges

**Distance metrics:**
- `hausdorff_distance(a, b)` — Maximum deviation between shapes
- `frechet_distance_approx(a, b)` — Approximate Fréchet distance

**Use cases:** CAD import cleanup, GIS data fusion, 3D scan processing

### distance/

Distance computations and fields.

- `sdf_2d` — 2D signed distance field generation
- `sdf_3d` — 3D signed distance field
- `distance_transform` — Grid-based distance transform
- `approximate_nn` — Approximate nearest neighbor queries

**Use cases:** Collision detection, implicit surfaces, pathfinding

### sampling/

Point generation and interpolation.

**Sampling:**
- `poisson_disk` — Blue noise sampling
- `stratified` — Stratified random sampling
- `halton` — Halton low-discrepancy sequence
- `sobol` — Sobol sequence

**Interpolation:**
- `barycentric` — Barycentric interpolation in triangles
- `idw` — Inverse distance weighting
- `natural_neighbor` — Natural neighbor interpolation

**Use cases:** Texture synthesis, remeshing, point cloud resampling

### float_geo/

Floating-point geometric primitives and operations.

**Types:**
- `Point2<F>`, `Point3<F>` — Generic over `f32`/`f64`
- `Vec2<F>`, `Vec3<F>` — Vectors
- `Segment2<F>`, `Segment3<F>` — Line segments
- `Polygon2<F>` — 2D polygon

**Operations:**
- `orient2d(a, b, c)` — Orientation with configurable epsilon
- `line_intersection(l1, l2)` — Returns `Option` for near-parallel
- `polygon_area(poly)` — Signed area
- `centroid(poly)` — Geometric center

---

## Design Principles

1. **Explicit tolerances** — No hidden epsilons; caller specifies tolerance
2. **Fallible operations** — Return `Option`/`Result` when operations can fail
3. **Generic over float type** — Support both `f32` and `f64`
4. **No allocations in hot paths** — Preallocate where possible
5. **SIMD-friendly** — Data layouts amenable to vectorization

---

## Error Handling

```rust
pub enum ApproxError {
    /// Points are too close together for reliable computation
    DegenerateInput,
    /// Lines are nearly parallel
    NearParallel,
    /// Tolerance is too small for the input scale
    ToleranceTooSmall,
    /// Algorithm did not converge
    ConvergenceFailed { iterations: usize },
}
```

---

## Bridge to exactum

```rust
use exactum::Point2 as ExactPoint;
use approxum::Point2 as FloatPoint;

// Exact → Approximate
impl<T: IntCoord> From<ExactPoint<T>> for FloatPoint<f64> {
    fn from(p: ExactPoint<T>) -> Self {
        FloatPoint::new(p.x.to_f64(), p.y.to_f64())
    }
}

// Approximate → Exact (lossy, requires snapping)
let snapped: Vec<ExactPoint<i64>> = approxum::snap_to_grid(&float_points, grid_size);
```

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
exactum = ["dep:exactum"] # Bridge to exactum
```

---

## Development Phases

**Phase 1: Core**
- [ ] `Point2`, `Vec2`, `Segment2` for `f32`/`f64`
- [ ] Basic predicates with epsilon
- [ ] AABB, minimum enclosing circle

**Phase 2: Simplification**
- [ ] Ramer-Douglas-Peucker
- [ ] Visvalingam-Whyatt
- [ ] Topology-preserving variant

**Phase 3: Curves**
- [ ] Bézier discretization (adaptive)
- [ ] Arc discretization
- [ ] Bézier fitting

**Phase 4: Spatial**
- [ ] BVH construction
- [ ] Signed distance fields
- [ ] Poisson disk sampling

**Phase 5: Polish**
- [ ] SIMD optimization
- [ ] Benchmarks
- [ ] exactum bridge

---

## Example Usage

```rust
use approxum::{Point2, simplify, bounds};

fn main() {
    // Simplify a noisy GPS track
    let track: Vec<Point2<f64>> = load_gpx("track.gpx");
    let simplified = simplify::rdp(&track, 0.0001); // ~10m tolerance in degrees
    println!("Reduced {} points to {}", track.len(), simplified.len());

    // Compute bounding circle for collision detection
    let circle = bounds::minimum_enclosing_circle(&simplified);
    println!("Bounding circle: center={:?}, radius={}", circle.center, circle.radius);

    // Snap to integer grid for exactum
    let grid_size = 1000; // 1000 units per degree
    let snapped: Vec<exactum::Point2<i64>> = approxum::snap_to_grid(&simplified, grid_size);
}
```

---

## References

- Ramer, U. (1972). "An iterative procedure for the polygonal approximation of plane curves"
- Douglas, D. & Peucker, T. (1973). "Algorithms for the reduction of the number of points required to represent a digitized line"
- Visvalingam, M. & Whyatt, J.D. (1993). "Line generalisation by repeated elimination of points"
- Welzl, E. (1991). "Smallest enclosing disks (balls and ellipsoids)"
- Bridson, R. (2007). "Fast Poisson disk sampling in arbitrary dimensions"
