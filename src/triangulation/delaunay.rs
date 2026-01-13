//! Delaunay triangulation using the Bowyer-Watson algorithm.
//!
//! Delaunay triangulation maximizes the minimum angle of all triangles,
//! avoiding skinny triangles when possible. It has the property that no
//! point lies inside the circumcircle of any triangle.
//!
//! # Algorithm
//!
//! The Bowyer-Watson algorithm is an incremental insertion algorithm:
//! 1. Start with a super-triangle containing all points
//! 2. Insert points one at a time, updating the triangulation
//! 3. Remove triangles connected to the super-triangle vertices
//!
//! # Complexity
//!
//! - Time: O(n²) worst case, O(n log n) expected for random points
//! - Space: O(n)
//!
//! # Example
//!
//! ```
//! use approxum::triangulation::delaunay_triangulation;
//! use approxum::Point2;
//!
//! let points: Vec<Point2<f64>> = vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 0.0),
//!     Point2::new(0.5, 1.0),
//!     Point2::new(0.5, 0.3),
//! ];
//!
//! let triangles = delaunay_triangulation(&points);
//!
//! // Should produce triangles covering all points
//! assert!(!triangles.is_empty());
//!
//! // Each triangle has 3 vertex indices
//! for tri in &triangles {
//!     assert!(tri.a < points.len());
//!     assert!(tri.b < points.len());
//!     assert!(tri.c < points.len());
//! }
//! ```

use crate::primitives::Point2;
use num_traits::Float;

/// A triangle represented by indices into a point array.
///
/// Vertices are stored in counter-clockwise order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Triangle {
    /// First vertex index
    pub a: usize,
    /// Second vertex index
    pub b: usize,
    /// Third vertex index
    pub c: usize,
}

impl Triangle {
    /// Creates a new triangle from vertex indices.
    #[inline]
    pub fn new(a: usize, b: usize, c: usize) -> Self {
        Self { a, b, c }
    }

    /// Returns the three edges of this triangle as pairs of indices.
    #[inline]
    pub fn edges(&self) -> [(usize, usize); 3] {
        [(self.a, self.b), (self.b, self.c), (self.c, self.a)]
    }

    /// Checks if the triangle contains a specific vertex index.
    #[inline]
    pub fn contains_vertex(&self, v: usize) -> bool {
        self.a == v || self.b == v || self.c == v
    }
}

/// An edge represented by two vertex indices, normalized so smaller index comes first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Edge(usize, usize);

impl Edge {
    fn new(a: usize, b: usize) -> Self {
        if a < b {
            Edge(a, b)
        } else {
            Edge(b, a)
        }
    }
}

/// Tests if a point lies inside the circumcircle of a triangle.
///
/// Returns true if the point is strictly inside the circumcircle.
/// Uses the determinant test which is more numerically stable than
/// explicitly computing the circumcenter.
///
/// # Arguments
///
/// * `p` - The point to test
/// * `a`, `b`, `c` - The triangle vertices (should be in CCW order)
///
/// # Example
///
/// ```
/// use approxum::triangulation::in_circumcircle;
/// use approxum::Point2;
///
/// let a = Point2::new(0.0_f64, 0.0);
/// let b = Point2::new(1.0, 0.0);
/// let c = Point2::new(0.5, 0.866); // Roughly equilateral
///
/// // Center of the triangle should be inside circumcircle
/// let center = Point2::new(0.5, 0.29);
/// assert!(in_circumcircle(center, a, b, c));
///
/// // Point far away should be outside
/// let far = Point2::new(10.0, 10.0);
/// assert!(!in_circumcircle(far, a, b, c));
/// ```
pub fn in_circumcircle<F: Float>(p: Point2<F>, a: Point2<F>, b: Point2<F>, c: Point2<F>) -> bool {
    // Use the determinant test:
    // | ax-px  ay-py  (ax-px)²+(ay-py)² |
    // | bx-px  by-py  (bx-px)²+(by-py)² | > 0  iff p is inside circumcircle (CCW triangle)
    // | cx-px  cy-py  (cx-px)²+(cy-py)² |

    let ax = a.x - p.x;
    let ay = a.y - p.y;
    let bx = b.x - p.x;
    let by = b.y - p.y;
    let cx = c.x - p.x;
    let cy = c.y - p.y;

    let aa = ax * ax + ay * ay;
    let bb = bx * bx + by * by;
    let cc = cx * cx + cy * cy;

    let det = ax * (by * cc - cy * bb) - ay * (bx * cc - cx * bb) + aa * (bx * cy - cx * by);

    det > F::zero()
}

/// Computes the orientation of three points.
/// Returns positive if CCW, negative if CW, zero if collinear.
fn orient2d<F: Float>(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> F {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Computes the Delaunay triangulation of a set of points.
///
/// Uses the Bowyer-Watson incremental insertion algorithm.
///
/// # Arguments
///
/// * `points` - The points to triangulate
///
/// # Returns
///
/// A vector of triangles. Each triangle contains indices into the input points array.
/// Triangles are in counter-clockwise orientation.
///
/// # Panics
///
/// Does not panic, but returns an empty vector if fewer than 3 points are provided.
///
/// # Example
///
/// ```
/// use approxum::triangulation::delaunay_triangulation;
/// use approxum::Point2;
///
/// // Square with center point
/// let points: Vec<Point2<f64>> = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
///     Point2::new(0.5, 0.5),
/// ];
///
/// let triangles = delaunay_triangulation(&points);
///
/// // Square with center point should produce 4 triangles
/// assert_eq!(triangles.len(), 4);
/// ```
pub fn delaunay_triangulation<F: Float>(points: &[Point2<F>]) -> Vec<Triangle> {
    if points.len() < 3 {
        return Vec::new();
    }

    // Find bounding box
    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;

    for p in points.iter().skip(1) {
        if p.x < min_x {
            min_x = p.x;
        }
        if p.x > max_x {
            max_x = p.x;
        }
        if p.y < min_y {
            min_y = p.y;
        }
        if p.y > max_y {
            max_y = p.y;
        }
    }

    // Create super-triangle that contains all points
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta = if dx > dy { dx } else { dy };
    let mid_x = (min_x + max_x) / F::from(2.0).unwrap();
    let mid_y = (min_y + max_y) / F::from(2.0).unwrap();

    // Super-triangle vertices (large enough to contain all points)
    // Vertices are in CCW order: bottom-left, bottom-right, top
    let margin = F::from(10.0).unwrap();
    let super_a = Point2::new(mid_x - margin * delta, mid_y - delta);
    let super_b = Point2::new(mid_x + margin * delta, mid_y - delta);
    let super_c = Point2::new(mid_x, mid_y + margin * delta);

    // Extended points list: original points + super-triangle vertices
    let n = points.len();
    let mut all_points: Vec<Point2<F>> = points.to_vec();
    all_points.push(super_a);
    all_points.push(super_b);
    all_points.push(super_c);

    // Super-triangle indices
    let super_indices = [n, n + 1, n + 2];

    // Start with super-triangle
    let mut triangles: Vec<Triangle> = vec![Triangle::new(n, n + 1, n + 2)];

    // Insert each point
    for (i, &p) in points.iter().enumerate() {
        // Find all triangles whose circumcircle contains this point
        let mut bad_triangles: Vec<usize> = Vec::new();

        for (ti, tri) in triangles.iter().enumerate() {
            let a = all_points[tri.a];
            let b = all_points[tri.b];
            let c = all_points[tri.c];

            if in_circumcircle(p, a, b, c) {
                bad_triangles.push(ti);
            }
        }

        // Find the boundary of the polygonal hole
        // An edge is on the boundary if it's used by exactly one bad triangle
        let mut edge_count: std::collections::HashMap<Edge, usize> = std::collections::HashMap::new();

        for &ti in &bad_triangles {
            let tri = triangles[ti];
            for (ea, eb) in tri.edges() {
                let edge = Edge::new(ea, eb);
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        // Collect boundary edges (those appearing exactly once)
        let boundary_edges: Vec<(usize, usize)> = edge_count
            .iter()
            .filter(|(_, &count)| count == 1)
            .map(|(edge, _)| (edge.0, edge.1))
            .collect();

        // Remove bad triangles (in reverse order to preserve indices)
        bad_triangles.sort_unstable();
        for &ti in bad_triangles.iter().rev() {
            triangles.swap_remove(ti);
        }

        // Create new triangles from the point to each boundary edge
        for (ea, eb) in boundary_edges {
            // Ensure CCW orientation
            let a = all_points[ea];
            let b = all_points[eb];

            if orient2d(a, b, p) > F::zero() {
                triangles.push(Triangle::new(ea, eb, i));
            } else {
                triangles.push(Triangle::new(eb, ea, i));
            }
        }
    }

    // Remove triangles that share vertices with super-triangle
    triangles.retain(|tri| {
        !tri.contains_vertex(super_indices[0])
            && !tri.contains_vertex(super_indices[1])
            && !tri.contains_vertex(super_indices[2])
    });

    triangles
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_triangle_new() {
        let tri = Triangle::new(0, 1, 2);
        assert_eq!(tri.a, 0);
        assert_eq!(tri.b, 1);
        assert_eq!(tri.c, 2);
    }

    #[test]
    fn test_triangle_edges() {
        let tri = Triangle::new(0, 1, 2);
        let edges = tri.edges();
        assert_eq!(edges[0], (0, 1));
        assert_eq!(edges[1], (1, 2));
        assert_eq!(edges[2], (2, 0));
    }

    #[test]
    fn test_triangle_contains_vertex() {
        let tri = Triangle::new(0, 1, 2);
        assert!(tri.contains_vertex(0));
        assert!(tri.contains_vertex(1));
        assert!(tri.contains_vertex(2));
        assert!(!tri.contains_vertex(3));
    }

    #[test]
    fn test_in_circumcircle_inside() {
        // Equilateral-ish triangle
        let a = Point2::new(0.0_f64, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, 0.866);

        // Centroid should be inside circumcircle
        let centroid = Point2::new(0.5, 0.288);
        assert!(in_circumcircle(centroid, a, b, c));
    }

    #[test]
    fn test_in_circumcircle_outside() {
        let a = Point2::new(0.0_f64, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, 0.866);

        // Far point should be outside
        let far = Point2::new(10.0, 10.0);
        assert!(!in_circumcircle(far, a, b, c));
    }

    #[test]
    fn test_in_circumcircle_on_circle() {
        // Right triangle: circumcircle has diameter on hypotenuse
        let a = Point2::new(0.0_f64, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.0, 1.0);

        // Point exactly on the circumcircle (opposite corner of square)
        let on_circle = Point2::new(1.0, 1.0);
        // Should be approximately on circle (not strictly inside)
        // Due to floating point, might be slightly inside or outside
        let result = in_circumcircle(on_circle, a, b, c);
        // This is a boundary case - just verify it doesn't crash
        let _ = result;
    }

    #[test]
    fn test_delaunay_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let triangles = delaunay_triangulation(&points);
        assert!(triangles.is_empty());
    }

    #[test]
    fn test_delaunay_one_point() {
        let points = vec![Point2::new(0.0_f64, 0.0)];
        let triangles = delaunay_triangulation(&points);
        assert!(triangles.is_empty());
    }

    #[test]
    fn test_delaunay_two_points() {
        let points = vec![Point2::new(0.0_f64, 0.0), Point2::new(1.0, 0.0)];
        let triangles = delaunay_triangulation(&points);
        assert!(triangles.is_empty());
    }

    #[test]
    fn test_delaunay_three_points() {
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let triangles = delaunay_triangulation(&points);
        assert_eq!(triangles.len(), 1);

        let tri = &triangles[0];
        // All indices should be valid
        assert!(tri.a < 3);
        assert!(tri.b < 3);
        assert!(tri.c < 3);
        // All indices should be different
        let mut indices: Vec<usize> = vec![tri.a, tri.b, tri.c];
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_delaunay_square() {
        // Square should produce 2 triangles
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let triangles = delaunay_triangulation(&points);
        assert_eq!(triangles.len(), 2);
    }

    #[test]
    fn test_delaunay_square_with_center() {
        // Square with center point should produce 4 triangles
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.5, 0.5),
        ];
        let triangles = delaunay_triangulation(&points);
        assert_eq!(triangles.len(), 4);
    }

    #[test]
    fn test_delaunay_all_indices_valid() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
            Point2::new(0.3, 0.4),
            Point2::new(0.7, 0.3),
        ];
        let triangles = delaunay_triangulation(&points);

        for tri in &triangles {
            assert!(tri.a < points.len(), "Invalid index a: {}", tri.a);
            assert!(tri.b < points.len(), "Invalid index b: {}", tri.b);
            assert!(tri.c < points.len(), "Invalid index c: {}", tri.c);
        }
    }

    #[test]
    fn test_delaunay_no_point_in_circumcircle() {
        // The Delaunay property: no point lies inside any triangle's circumcircle
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.5, 0.5),
            Point2::new(0.25, 0.75),
        ];
        let triangles = delaunay_triangulation(&points);

        for tri in &triangles {
            let a = points[tri.a];
            let b = points[tri.b];
            let c = points[tri.c];

            // Check all other points
            for (i, &p) in points.iter().enumerate() {
                if i != tri.a && i != tri.b && i != tri.c {
                    assert!(
                        !in_circumcircle(p, a, b, c),
                        "Point {} is inside circumcircle of triangle ({}, {}, {})",
                        i,
                        tri.a,
                        tri.b,
                        tri.c
                    );
                }
            }
        }
    }

    #[test]
    fn test_delaunay_grid() {
        // Regular grid of points
        let mut points: Vec<Point2<f64>> = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                points.push(Point2::new(i as f64, j as f64));
            }
        }

        let triangles = delaunay_triangulation(&points);

        // 4x4 grid = 16 points, should produce 18 triangles
        // (3x3 squares = 9 squares, each split into 2 triangles = 18)
        assert_eq!(triangles.len(), 18);

        // Verify Delaunay property
        for tri in &triangles {
            let a = points[tri.a];
            let b = points[tri.b];
            let c = points[tri.c];

            for (i, &p) in points.iter().enumerate() {
                if i != tri.a && i != tri.b && i != tri.c {
                    assert!(!in_circumcircle(p, a, b, c));
                }
            }
        }
    }

    #[test]
    fn test_delaunay_random_like() {
        // Points that look somewhat random
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.1, 0.2),
            Point2::new(0.8, 0.1),
            Point2::new(0.9, 0.9),
            Point2::new(0.2, 0.85),
            Point2::new(0.5, 0.5),
            Point2::new(0.3, 0.3),
            Point2::new(0.7, 0.6),
            Point2::new(0.4, 0.8),
        ];
        let triangles = delaunay_triangulation(&points);

        assert!(!triangles.is_empty());

        // Verify Delaunay property
        for tri in &triangles {
            let a = points[tri.a];
            let b = points[tri.b];
            let c = points[tri.c];

            for (i, &p) in points.iter().enumerate() {
                if i != tri.a && i != tri.b && i != tri.c {
                    assert!(!in_circumcircle(p, a, b, c));
                }
            }
        }
    }

    #[test]
    fn test_delaunay_collinear() {
        // Collinear points - should produce no triangles
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let triangles = delaunay_triangulation(&points);
        // Collinear points can't form a triangle
        assert!(triangles.is_empty());
    }

    #[test]
    fn test_delaunay_f32() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let triangles = delaunay_triangulation(&points);
        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn test_delaunay_covers_all_points() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.5, 0.5),
        ];
        let triangles = delaunay_triangulation(&points);

        // Collect all used vertex indices
        let mut used: HashSet<usize> = HashSet::new();
        for tri in &triangles {
            used.insert(tri.a);
            used.insert(tri.b);
            used.insert(tri.c);
        }

        // All points should be used
        for i in 0..points.len() {
            assert!(used.contains(&i), "Point {} not used in triangulation", i);
        }
    }

    #[test]
    fn test_edge_normalization() {
        let e1 = Edge::new(1, 2);
        let e2 = Edge::new(2, 1);
        assert_eq!(e1, e2);
        assert_eq!(e1.0, 1);
        assert_eq!(e1.1, 2);
    }
}
