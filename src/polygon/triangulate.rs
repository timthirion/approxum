//! Polygon triangulation using ear clipping.
//!
//! Converts a simple polygon into a set of triangles that exactly cover it.
//!
//! # Algorithm
//!
//! The ear clipping algorithm works by repeatedly finding and removing "ears":
//! - An ear is a triangle formed by three consecutive vertices
//! - The middle vertex must be convex (reflex vertices cannot form ears)
//! - No other polygon vertices may be inside the ear triangle
//!
//! # Complexity
//!
//! - Time: O(n²) for a polygon with n vertices
//! - Space: O(n)
//!
//! # Example
//!
//! ```
//! use approxum::polygon::{Polygon, triangulate_polygon};
//! use approxum::Point2;
//!
//! let square = Polygon::new(vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 0.0),
//!     Point2::new(1.0, 1.0),
//!     Point2::new(0.0, 1.0),
//! ]);
//!
//! let triangles = triangulate_polygon(&square);
//!
//! // A square is divided into 2 triangles
//! assert_eq!(triangles.len(), 2);
//! ```

use crate::polygon::Polygon;
use crate::primitives::Point2;
use num_traits::Float;

/// A triangle from polygon triangulation, represented by three points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolygonTriangle<F> {
    /// First vertex of the triangle.
    pub a: Point2<F>,
    /// Second vertex of the triangle.
    pub b: Point2<F>,
    /// Third vertex of the triangle.
    pub c: Point2<F>,
}

impl<F: Float> PolygonTriangle<F> {
    /// Creates a new triangle from three points.
    #[inline]
    pub fn new(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> Self {
        Self { a, b, c }
    }

    /// Computes the area of the triangle.
    pub fn area(&self) -> F {
        let two = F::from(2.0).unwrap();
        ((self.b.x - self.a.x) * (self.c.y - self.a.y)
            - (self.c.x - self.a.x) * (self.b.y - self.a.y))
            .abs()
            / two
    }

    /// Returns the centroid of the triangle.
    pub fn centroid(&self) -> Point2<F> {
        let three = F::from(3.0).unwrap();
        Point2::new(
            (self.a.x + self.b.x + self.c.x) / three,
            (self.a.y + self.b.y + self.c.y) / three,
        )
    }
}

/// Result of polygon triangulation with vertex indices.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangulationResult {
    /// Triangle vertex indices. Each triple (i, j, k) represents a triangle
    /// using vertices from the original polygon.
    pub indices: Vec<(usize, usize, usize)>,
}

impl TriangulationResult {
    /// Returns the number of triangles.
    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if there are no triangles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Triangulates a polygon using the ear clipping algorithm.
///
/// Returns a vector of triangles that exactly cover the polygon.
///
/// # Arguments
///
/// * `polygon` - The polygon to triangulate (must be a simple polygon)
///
/// # Returns
///
/// A vector of triangles. For a polygon with n vertices, returns n-2 triangles.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, triangulate_polygon};
/// use approxum::Point2;
///
/// // L-shaped polygon (concave)
/// let l_shape = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(2.0, 1.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(1.0, 2.0),
///     Point2::new(0.0, 2.0),
/// ]);
///
/// let triangles = triangulate_polygon(&l_shape);
/// assert_eq!(triangles.len(), 4); // 6 vertices -> 4 triangles
/// ```
pub fn triangulate_polygon<F: Float>(polygon: &Polygon<F>) -> Vec<PolygonTriangle<F>> {
    let result = triangulate_polygon_indexed(polygon);

    result
        .indices
        .iter()
        .map(|&(i, j, k)| {
            PolygonTriangle::new(
                polygon.vertices[i],
                polygon.vertices[j],
                polygon.vertices[k],
            )
        })
        .collect()
}

/// Triangulates a polygon and returns vertex indices.
///
/// This is more memory-efficient when you need to work with indices
/// rather than duplicating vertex data.
///
/// # Arguments
///
/// * `polygon` - The polygon to triangulate
///
/// # Returns
///
/// A `TriangulationResult` containing triangle indices.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, triangulate_polygon_indexed};
/// use approxum::Point2;
///
/// let pentagon = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(2.5, 1.5),
///     Point2::new(1.0, 2.5),
///     Point2::new(-0.5, 1.5),
/// ]);
///
/// let result = triangulate_polygon_indexed(&pentagon);
/// assert_eq!(result.len(), 3); // 5 vertices -> 3 triangles
///
/// // Each triangle uses indices into the original polygon
/// for (i, j, k) in &result.indices {
///     assert!(*i < 5 && *j < 5 && *k < 5);
/// }
/// ```
pub fn triangulate_polygon_indexed<F: Float>(polygon: &Polygon<F>) -> TriangulationResult {
    let n = polygon.len();

    if n < 3 {
        return TriangulationResult {
            indices: Vec::new(),
        };
    }

    if n == 3 {
        return TriangulationResult {
            indices: vec![(0, 1, 2)],
        };
    }

    // Ensure CCW winding
    let vertices: Vec<Point2<F>> = if polygon.signed_area() < F::zero() {
        polygon.vertices.iter().rev().copied().collect()
    } else {
        polygon.vertices.clone()
    };

    // Map from working indices to original indices
    let mut index_map: Vec<usize> = if polygon.signed_area() < F::zero() {
        (0..n).rev().collect()
    } else {
        (0..n).collect()
    };

    // Working copy of vertices
    let mut remaining: Vec<Point2<F>> = vertices;
    let mut triangles: Vec<(usize, usize, usize)> = Vec::with_capacity(n - 2);

    // Ear clipping loop
    let mut max_iterations = n * n; // Safety limit

    while remaining.len() > 3 && max_iterations > 0 {
        max_iterations -= 1;
        let m = remaining.len();
        let mut ear_found = false;

        for i in 0..m {
            let prev = (i + m - 1) % m;
            let next = (i + 1) % m;

            if is_ear(&remaining, prev, i, next) {
                // Record the triangle using original indices
                triangles.push((index_map[prev], index_map[i], index_map[next]));

                // Remove the ear vertex
                remaining.remove(i);
                index_map.remove(i);
                ear_found = true;
                break;
            }
        }

        if !ear_found {
            // No ear found - polygon may be degenerate
            // Try to salvage by just connecting remaining vertices
            break;
        }
    }

    // Handle the last triangle
    if remaining.len() == 3 {
        triangles.push((index_map[0], index_map[1], index_map[2]));
    }

    TriangulationResult { indices: triangles }
}

/// Checks if vertex at index `i` forms an ear with its neighbors.
fn is_ear<F: Float>(vertices: &[Point2<F>], prev: usize, curr: usize, next: usize) -> bool {
    let a = vertices[prev];
    let b = vertices[curr];
    let c = vertices[next];

    // Check if the vertex is convex (left turn for CCW polygon)
    if !is_convex(a, b, c) {
        return false;
    }

    // Check that no other vertices are inside this triangle
    for (i, &vertex) in vertices.iter().enumerate() {
        if i == prev || i == curr || i == next {
            continue;
        }
        if point_in_triangle(vertex, a, b, c) {
            return false;
        }
    }

    true
}

/// Checks if vertex b is convex (forms a left turn from a to c).
#[inline]
fn is_convex<F: Float>(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> bool {
    cross(a, b, c) > F::zero()
}

/// Cross product of vectors (b-a) and (c-a).
#[inline]
fn cross<F: Float>(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> F {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Checks if point p is inside triangle abc.
fn point_in_triangle<F: Float>(p: Point2<F>, a: Point2<F>, b: Point2<F>, c: Point2<F>) -> bool {
    let d1 = sign(p, a, b);
    let d2 = sign(p, b, c);
    let d3 = sign(p, c, a);

    let has_neg = d1 < F::zero() || d2 < F::zero() || d3 < F::zero();
    let has_pos = d1 > F::zero() || d2 > F::zero() || d3 > F::zero();

    !(has_neg && has_pos)
}

/// Sign of the cross product for point-in-triangle test.
#[inline]
fn sign<F: Float>(p1: Point2<F>, p2: Point2<F>, p3: Point2<F>) -> F {
    (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
}

/// Computes the total area of a triangulation.
///
/// Useful for verifying that the triangulation covers the original polygon.
pub fn triangulation_area<F: Float>(triangles: &[PolygonTriangle<F>]) -> F {
    triangles
        .iter()
        .map(|t| t.area())
        .fold(F::zero(), |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_triangle_area() {
        let tri = PolygonTriangle::new(
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 2.0),
        );
        assert!(approx_eq(tri.area(), 2.0, 1e-10));
    }

    #[test]
    fn test_triangle_centroid() {
        let tri = PolygonTriangle::new(
            Point2::new(0.0_f64, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(0.0, 3.0),
        );
        let c = tri.centroid();
        assert!(approx_eq(c.x, 1.0, 1e-10));
        assert!(approx_eq(c.y, 1.0, 1e-10));
    }

    #[test]
    fn test_triangulate_triangle() {
        let triangle = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ]);

        let result = triangulate_polygon(&triangle);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_triangulate_square() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);

        let result = triangulate_polygon(&square);
        assert_eq!(result.len(), 2);

        // Total area should equal square area
        let total_area = triangulation_area(&result);
        assert!(approx_eq(total_area, 1.0, 1e-10));
    }

    #[test]
    fn test_triangulate_pentagon() {
        let pentagon = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.5, 1.5),
            Point2::new(1.0, 2.5),
            Point2::new(-0.5, 1.5),
        ]);

        let result = triangulate_polygon(&pentagon);
        assert_eq!(result.len(), 3); // 5 vertices -> 3 triangles
    }

    #[test]
    fn test_triangulate_l_shape() {
        // L-shaped polygon (concave)
        let l_shape = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let result = triangulate_polygon(&l_shape);
        assert_eq!(result.len(), 4); // 6 vertices -> 4 triangles

        // Total area should equal L-shape area (3.0)
        let total_area = triangulation_area(&result);
        assert!(approx_eq(total_area, 3.0, 1e-10));
    }

    #[test]
    fn test_triangulate_star() {
        // Simple star shape (concave)
        let star = Polygon::new(vec![
            Point2::new(0.0_f64, 3.0), // Top
            Point2::new(1.0, 1.0),     // Inner
            Point2::new(3.0, 1.0),     // Right
            Point2::new(1.5, 0.0),     // Inner
            Point2::new(2.0, -2.0),    // Bottom right
            Point2::new(0.0, -0.5),    // Inner
            Point2::new(-2.0, -2.0),   // Bottom left
            Point2::new(-1.5, 0.0),    // Inner
            Point2::new(-3.0, 1.0),    // Left
            Point2::new(-1.0, 1.0),    // Inner
        ]);

        let result = triangulate_polygon(&star);
        assert_eq!(result.len(), 8); // 10 vertices -> 8 triangles
    }

    #[test]
    fn test_triangulate_indexed() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);

        let result = triangulate_polygon_indexed(&square);
        assert_eq!(result.len(), 2);

        // All indices should be valid
        for (i, j, k) in &result.indices {
            assert!(*i < 4);
            assert!(*j < 4);
            assert!(*k < 4);
        }
    }

    #[test]
    fn test_triangulate_empty() {
        let empty: Polygon<f64> = Polygon::empty();
        let result = triangulate_polygon(&empty);
        assert!(result.is_empty());
    }

    #[test]
    fn test_triangulate_two_vertices() {
        let line = Polygon::new(vec![Point2::new(0.0_f64, 0.0), Point2::new(1.0, 0.0)]);
        let result = triangulate_polygon(&line);
        assert!(result.is_empty());
    }

    #[test]
    fn test_triangulate_cw_polygon() {
        // CW polygon (should be auto-corrected)
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 0.0),
        ]);

        let result = triangulate_polygon(&square);
        assert_eq!(result.len(), 2);

        let total_area = triangulation_area(&result);
        assert!(approx_eq(total_area, 1.0, 1e-10));
    }

    #[test]
    fn test_triangulate_hexagon() {
        // Regular hexagon
        let hex = Polygon::new(vec![
            Point2::new(2.0_f64, 0.0),
            Point2::new(1.0, 1.732),
            Point2::new(-1.0, 1.732),
            Point2::new(-2.0, 0.0),
            Point2::new(-1.0, -1.732),
            Point2::new(1.0, -1.732),
        ]);

        let result = triangulate_polygon(&hex);
        assert_eq!(result.len(), 4); // 6 vertices -> 4 triangles

        // Hexagon area ≈ 10.392
        let total_area = triangulation_area(&result);
        assert!(approx_eq(total_area, hex.area(), 1e-6));
    }

    #[test]
    fn test_triangulate_arrow() {
        // Arrow/chevron shape (concave)
        let arrow = Polygon::new(vec![
            Point2::new(0.0_f64, 2.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 0.0),
            Point2::new(0.5, -1.0),
            Point2::new(-0.5, -1.0),
            Point2::new(-0.5, 0.0),
            Point2::new(-1.0, 0.0),
        ]);

        let result = triangulate_polygon(&arrow);
        assert_eq!(result.len(), 5); // 7 vertices -> 5 triangles

        let total_area = triangulation_area(&result);
        assert!(approx_eq(total_area, arrow.area(), 1e-10));
    }

    #[test]
    fn test_triangulation_result_methods() {
        let result = TriangulationResult {
            indices: vec![(0, 1, 2), (0, 2, 3)],
        };
        assert_eq!(result.len(), 2);
        assert!(!result.is_empty());

        let empty = TriangulationResult { indices: vec![] };
        assert!(empty.is_empty());
    }

    #[test]
    fn test_point_in_triangle() {
        let a = Point2::new(0.0_f64, 0.0);
        let b = Point2::new(2.0, 0.0);
        let c = Point2::new(1.0, 2.0);

        // Inside
        assert!(point_in_triangle(Point2::new(1.0, 0.5), a, b, c));
        // On edge
        assert!(point_in_triangle(Point2::new(1.0, 0.0), a, b, c));
        // Outside
        assert!(!point_in_triangle(Point2::new(2.0, 2.0), a, b, c));
    }

    #[test]
    fn test_f32() {
        let square: Polygon<f32> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);

        let result = triangulate_polygon(&square);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_area_preservation() {
        // Test that triangulation preserves area for various shapes
        let shapes: Vec<Polygon<f64>> = vec![
            // Square
            Polygon::new(vec![
                Point2::new(0.0, 0.0),
                Point2::new(4.0, 0.0),
                Point2::new(4.0, 4.0),
                Point2::new(0.0, 4.0),
            ]),
            // Pentagon
            Polygon::new(vec![
                Point2::new(0.0, 0.0),
                Point2::new(3.0, 0.0),
                Point2::new(4.0, 2.0),
                Point2::new(1.5, 4.0),
                Point2::new(-1.0, 2.0),
            ]),
            // L-shape
            Polygon::new(vec![
                Point2::new(0.0, 0.0),
                Point2::new(3.0, 0.0),
                Point2::new(3.0, 1.0),
                Point2::new(1.0, 1.0),
                Point2::new(1.0, 3.0),
                Point2::new(0.0, 3.0),
            ]),
        ];

        for shape in shapes {
            let triangles = triangulate_polygon(&shape);
            let tri_area = triangulation_area(&triangles);
            assert!(
                approx_eq(tri_area, shape.area(), 1e-10),
                "Area mismatch: triangulation {} vs polygon {}",
                tri_area,
                shape.area()
            );
        }
    }
}
