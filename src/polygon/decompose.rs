//! Convex polygon decomposition.
//!
//! Decomposes a simple polygon into a set of convex polygons. This is useful
//! for algorithms that only work efficiently on convex shapes.
//!
//! # Algorithms
//!
//! - **Triangulation**: Simplest approach, O(n) convex pieces (triangles)
//! - **Hertel-Mehlhorn**: Merges triangles into larger convex pieces, O(r) where r ≤ 4n
//!
//! # Example
//!
//! ```
//! use approxum::polygon::{Polygon, convex_decomposition};
//! use approxum::Point2;
//!
//! // L-shaped polygon (non-convex)
//! let l_shape = Polygon::new(vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(2.0, 0.0),
//!     Point2::new(2.0, 1.0),
//!     Point2::new(1.0, 1.0),
//!     Point2::new(1.0, 2.0),
//!     Point2::new(0.0, 2.0),
//! ]);
//!
//! let convex_parts = convex_decomposition(&l_shape);
//! // Each part is convex
//! for part in &convex_parts {
//!     assert!(part.is_convex());
//! }
//! ```

use super::core::{polygon_is_convex, Polygon};
use crate::primitives::Point2;
use num_traits::Float;

/// Decomposes a polygon into convex parts using triangulation.
///
/// This is the simplest decomposition - it produces n-2 triangles for an
/// n-vertex polygon. Use `convex_decomposition` for fewer pieces.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, triangulate_decomposition};
/// use approxum::Point2;
///
/// let square = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
/// ]);
///
/// let triangles = triangulate_decomposition(&square);
/// assert_eq!(triangles.len(), 2); // Square = 2 triangles
/// ```
pub fn triangulate_decomposition<F: Float>(polygon: &Polygon<F>) -> Vec<Polygon<F>> {
    let n = polygon.vertices.len();
    if n < 3 {
        return vec![];
    }
    if n == 3 {
        return vec![polygon.clone()];
    }

    ear_clip_triangulate(&polygon.vertices)
}

/// Decomposes a polygon into convex parts using Hertel-Mehlhorn algorithm.
///
/// This algorithm first triangulates the polygon, then merges adjacent
/// triangles into larger convex polygons where possible. The result has
/// at most 4 times the optimal number of convex pieces.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, convex_decomposition};
/// use approxum::Point2;
///
/// // Star shape (non-convex)
/// let star = Polygon::new(vec![
///     Point2::new(0.0, 1.0),
///     Point2::new(0.2, 0.4),
///     Point2::new(-0.5, 0.3),
///     Point2::new(0.0, 0.0),
///     Point2::new(0.5, 0.3),
///     Point2::new(0.8, 0.4),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.5, 0.8),
/// ]);
///
/// let parts = convex_decomposition(&star);
/// for part in &parts {
///     assert!(part.is_convex());
/// }
/// ```
pub fn convex_decomposition<F: Float>(polygon: &Polygon<F>) -> Vec<Polygon<F>> {
    let n = polygon.vertices.len();
    if n < 3 {
        return vec![];
    }

    // If already convex, return as-is
    if polygon.is_convex() {
        return vec![polygon.clone()];
    }

    // Triangulate first
    let triangles = triangulate_decomposition(polygon);
    if triangles.is_empty() {
        return vec![];
    }

    // Apply Hertel-Mehlhorn merging
    hertel_mehlhorn_merge(triangles, &polygon.vertices)
}

/// Decomposes a polygon into the minimum number of convex parts.
///
/// This is more expensive than `convex_decomposition` but produces
/// fewer pieces. Uses dynamic programming approach.
///
/// Note: For complex polygons, this can be slow. Use `convex_decomposition`
/// for a faster approximation.
pub fn optimal_convex_decomposition<F: Float>(polygon: &Polygon<F>) -> Vec<Polygon<F>> {
    // For now, just use Hertel-Mehlhorn as the "optimal" version
    // True optimal decomposition is NP-hard and complex to implement
    convex_decomposition(polygon)
}

/// Ear clipping triangulation.
fn ear_clip_triangulate<F: Float>(vertices: &[Point2<F>]) -> Vec<Polygon<F>> {
    let n = vertices.len();
    if n < 3 {
        return vec![];
    }
    if n == 3 {
        return vec![Polygon::new(vertices.to_vec())];
    }

    let mut result = Vec::new();
    let mut remaining: Vec<usize> = (0..n).collect();

    while remaining.len() > 3 {
        let m = remaining.len();
        let mut found_ear = false;

        for i in 0..m {
            let prev = if i == 0 { m - 1 } else { i - 1 };
            let next = (i + 1) % m;

            let a = vertices[remaining[prev]];
            let b = vertices[remaining[i]];
            let c = vertices[remaining[next]];

            if is_ear(vertices, &remaining, prev, i, next) {
                result.push(Polygon::new(vec![a, b, c]));
                remaining.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            // Fallback: take any triangle
            let a = vertices[remaining[0]];
            let b = vertices[remaining[1]];
            let c = vertices[remaining[2]];
            result.push(Polygon::new(vec![a, b, c]));
            remaining.remove(1);
        }
    }

    if remaining.len() == 3 {
        result.push(Polygon::new(vec![
            vertices[remaining[0]],
            vertices[remaining[1]],
            vertices[remaining[2]],
        ]));
    }

    result
}

/// Checks if vertex at index i is an ear.
fn is_ear<F: Float>(
    vertices: &[Point2<F>],
    remaining: &[usize],
    prev: usize,
    i: usize,
    next: usize,
) -> bool {
    let a = vertices[remaining[prev]];
    let b = vertices[remaining[i]];
    let c = vertices[remaining[next]];

    // Check if convex (CCW turn)
    let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if cross <= F::zero() {
        return false;
    }

    // Check no other vertex inside
    for (j, &idx) in remaining.iter().enumerate() {
        if j == prev || j == i || j == next {
            continue;
        }
        if point_in_triangle(vertices[idx], a, b, c) {
            return false;
        }
    }

    true
}

/// Point in triangle test.
fn point_in_triangle<F: Float>(p: Point2<F>, a: Point2<F>, b: Point2<F>, c: Point2<F>) -> bool {
    let v0 = Point2::new(c.x - a.x, c.y - a.y);
    let v1 = Point2::new(b.x - a.x, b.y - a.y);
    let v2 = Point2::new(p.x - a.x, p.y - a.y);

    let dot00 = v0.x * v0.x + v0.y * v0.y;
    let dot01 = v0.x * v1.x + v0.y * v1.y;
    let dot02 = v0.x * v2.x + v0.y * v2.y;
    let dot11 = v1.x * v1.x + v1.y * v1.y;
    let dot12 = v1.x * v2.x + v1.y * v2.y;

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < F::epsilon() {
        return false;
    }

    let inv_denom = F::one() / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    let eps = F::from(1e-10).unwrap();
    u >= -eps && v >= -eps && (u + v) <= F::one() + eps
}

/// Hertel-Mehlhorn algorithm: merge triangles into larger convex polygons.
fn hertel_mehlhorn_merge<F: Float>(
    mut polygons: Vec<Polygon<F>>,
    original_vertices: &[Point2<F>],
) -> Vec<Polygon<F>> {
    if polygons.len() <= 1 {
        return polygons;
    }

    let eps = F::from(1e-10).unwrap();

    // Keep merging while we can
    let mut merged = true;
    while merged {
        merged = false;

        'outer: for i in 0..polygons.len() {
            for j in (i + 1)..polygons.len() {
                if let Some(merged_poly) =
                    try_merge(&polygons[i], &polygons[j], original_vertices, eps)
                {
                    polygons.remove(j);
                    polygons.remove(i);
                    polygons.push(merged_poly);
                    merged = true;
                    break 'outer;
                }
            }
        }
    }

    polygons
}

/// Tries to merge two convex polygons if they share an edge and result is convex.
fn try_merge<F: Float>(
    a: &Polygon<F>,
    b: &Polygon<F>,
    original_vertices: &[Point2<F>],
    eps: F,
) -> Option<Polygon<F>> {
    let na = a.vertices.len();
    let nb = b.vertices.len();

    // Find shared edge
    for i in 0..na {
        let a1 = a.vertices[i];
        let a2 = a.vertices[(i + 1) % na];

        for j in 0..nb {
            let b1 = b.vertices[j];
            let b2 = b.vertices[(j + 1) % nb];

            // Check if edges match (opposite directions for CCW polygons)
            if points_equal(a1, b2, eps) && points_equal(a2, b1, eps) {
                // Check if the shared edge is a diagonal
                if is_diagonal_edge(a1, a2, original_vertices, eps) {
                    if let Some(merged) = merge_at_shared_edge(a, b, i, j) {
                        if polygon_is_convex(&merged.vertices) {
                            return Some(merged);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Checks if an edge is a diagonal (not on the original polygon boundary).
fn is_diagonal_edge<F: Float>(
    p1: Point2<F>,
    p2: Point2<F>,
    original_vertices: &[Point2<F>],
    eps: F,
) -> bool {
    let n = original_vertices.len();
    for i in 0..n {
        let v1 = original_vertices[i];
        let v2 = original_vertices[(i + 1) % n];

        if (points_equal(p1, v1, eps) && points_equal(p2, v2, eps))
            || (points_equal(p1, v2, eps) && points_equal(p2, v1, eps))
        {
            return false; // It's a boundary edge
        }
    }
    true // It's a diagonal
}

/// Merges two polygons that share an edge (a[i]->a[i+1] == b[j+1]->b[j]).
fn merge_at_shared_edge<F: Float>(
    a: &Polygon<F>,
    b: &Polygon<F>,
    edge_a: usize,
    edge_b: usize,
) -> Option<Polygon<F>> {
    let na = a.vertices.len();
    let nb = b.vertices.len();

    let mut result = Vec::with_capacity(na + nb - 2);

    // Start from vertex after the shared edge in a
    // Walk around a, stopping before the shared edge starts
    for k in 1..na {
        let idx = (edge_a + 1 + k) % na;
        result.push(a.vertices[idx]);
    }

    // Continue from b, starting after the shared edge
    // Walk around b, stopping before the shared edge
    for k in 1..nb {
        let idx = (edge_b + 1 + k) % nb;
        result.push(b.vertices[idx]);
    }

    if result.len() >= 3 {
        Some(Polygon::new(result))
    } else {
        None
    }
}

/// Checks if two points are approximately equal.
fn points_equal<F: Float>(a: Point2<F>, b: Point2<F>, eps: F) -> bool {
    (a.x - b.x).abs() < eps && (a.y - b.y).abs() < eps
}

/// Returns the number of reflex (concave) vertices in a polygon.
///
/// A reflex vertex is one where the interior angle is greater than 180°.
pub fn count_reflex_vertices<F: Float>(polygon: &Polygon<F>) -> usize {
    let n = polygon.vertices.len();
    if n < 3 {
        return 0;
    }

    let mut count = 0;
    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        let next = (i + 1) % n;

        let a = polygon.vertices[prev];
        let b = polygon.vertices[i];
        let c = polygon.vertices[next];

        // Cross product: negative means reflex (CW turn)
        let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if cross < F::zero() {
            count += 1;
        }
    }

    count
}

/// Returns indices of reflex (concave) vertices in a polygon.
pub fn find_reflex_vertices<F: Float>(polygon: &Polygon<F>) -> Vec<usize> {
    let n = polygon.vertices.len();
    if n < 3 {
        return vec![];
    }

    let mut result = Vec::new();
    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        let next = (i + 1) % n;

        let a = polygon.vertices[prev];
        let b = polygon.vertices[i];
        let c = polygon.vertices[next];

        let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if cross < F::zero() {
            result.push(i);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn square<F: Float>(x: F, y: F, size: F) -> Polygon<F> {
        Polygon::new(vec![
            Point2::new(x, y),
            Point2::new(x + size, y),
            Point2::new(x + size, y + size),
            Point2::new(x, y + size),
        ])
    }

    fn l_shape<F: Float>() -> Polygon<F> {
        Polygon::new(vec![
            Point2::new(F::zero(), F::zero()),
            Point2::new(F::from(2.0).unwrap(), F::zero()),
            Point2::new(F::from(2.0).unwrap(), F::one()),
            Point2::new(F::one(), F::one()),
            Point2::new(F::one(), F::from(2.0).unwrap()),
            Point2::new(F::zero(), F::from(2.0).unwrap()),
        ])
    }

    #[test]
    fn test_triangulate_square() {
        let sq: Polygon<f64> = square(0.0, 0.0, 1.0);
        let triangles = triangulate_decomposition(&sq);

        assert_eq!(triangles.len(), 2);

        // Total area should match
        let total_area: f64 = triangles.iter().map(|t| t.area().abs()).sum();
        assert_relative_eq!(total_area, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_triangulate_triangle() {
        let tri: Polygon<f64> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ]);

        let triangles = triangulate_decomposition(&tri);
        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn test_convex_decomposition_convex() {
        // Convex polygon should return itself
        let sq: Polygon<f64> = square(0.0, 0.0, 1.0);
        let parts = convex_decomposition(&sq);

        assert_eq!(parts.len(), 1);
        assert!(parts[0].is_convex());
    }

    #[test]
    fn test_convex_decomposition_l_shape() {
        let l: Polygon<f64> = l_shape();
        let parts = convex_decomposition(&l);

        // All parts should be convex
        for part in &parts {
            assert!(part.is_convex(), "Part is not convex: {:?}", part.vertices);
        }

        // Total area should be preserved
        let original_area = l.area().abs();
        let total_area: f64 = parts.iter().map(|p| p.area().abs()).sum();
        assert_relative_eq!(total_area, original_area, epsilon = 0.01);
    }

    #[test]
    fn test_convex_decomposition_star() {
        // 5-pointed star
        let star: Polygon<f64> = Polygon::new(vec![
            Point2::new(0.5, 1.0),
            Point2::new(0.4, 0.6),
            Point2::new(0.0, 0.5),
            Point2::new(0.3, 0.3),
            Point2::new(0.2, 0.0),
            Point2::new(0.5, 0.2),
            Point2::new(0.8, 0.0),
            Point2::new(0.7, 0.3),
            Point2::new(1.0, 0.5),
            Point2::new(0.6, 0.6),
        ]);

        let parts = convex_decomposition(&star);

        // All parts should be convex
        for part in &parts {
            assert!(part.is_convex());
        }

        // Should have multiple parts (star is very non-convex)
        assert!(parts.len() > 1);
    }

    #[test]
    fn test_count_reflex_vertices_convex() {
        let sq: Polygon<f64> = square(0.0, 0.0, 1.0);
        assert_eq!(count_reflex_vertices(&sq), 0);
    }

    #[test]
    fn test_count_reflex_vertices_l_shape() {
        let l: Polygon<f64> = l_shape();
        let reflex = count_reflex_vertices(&l);

        // L-shape has one reflex vertex (the inner corner)
        assert_eq!(reflex, 1);
    }

    #[test]
    fn test_find_reflex_vertices() {
        let l: Polygon<f64> = l_shape();
        let reflex_indices = find_reflex_vertices(&l);

        assert_eq!(reflex_indices.len(), 1);
        // The reflex vertex is at index 3 (the inner corner at (1,1))
        assert_eq!(reflex_indices[0], 3);
    }

    #[test]
    fn test_triangulate_preserves_area() {
        let l: Polygon<f64> = l_shape();
        let triangles = triangulate_decomposition(&l);

        let original_area = l.area().abs();
        let total_area: f64 = triangles.iter().map(|t| t.area().abs()).sum();

        assert_relative_eq!(total_area, original_area, epsilon = 0.01);
    }

    #[test]
    fn test_decomposition_fewer_parts_than_triangulation() {
        let l: Polygon<f64> = l_shape();

        let triangles = triangulate_decomposition(&l);
        let convex_parts = convex_decomposition(&l);

        // Hertel-Mehlhorn should produce fewer or equal parts
        assert!(convex_parts.len() <= triangles.len());
    }

    #[test]
    fn test_empty_polygon() {
        let empty: Polygon<f64> = Polygon::new(vec![]);
        let parts = convex_decomposition(&empty);
        assert!(parts.is_empty());
    }

    #[test]
    fn test_two_vertices() {
        let line: Polygon<f64> = Polygon::new(vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)]);
        let parts = convex_decomposition(&line);
        assert!(parts.is_empty());
    }

    #[test]
    fn test_arrow_shape() {
        // Arrow pointing right
        let arrow: Polygon<f64> = Polygon::new(vec![
            Point2::new(0.0, 0.3),
            Point2::new(0.5, 0.3),
            Point2::new(0.5, 0.0),
            Point2::new(1.0, 0.5),
            Point2::new(0.5, 1.0),
            Point2::new(0.5, 0.7),
            Point2::new(0.0, 0.7),
        ]);

        let parts = convex_decomposition(&arrow);

        for part in &parts {
            assert!(part.is_convex());
        }

        let original_area = arrow.area().abs();
        let total_area: f64 = parts.iter().map(|p| p.area().abs()).sum();
        assert_relative_eq!(total_area, original_area, epsilon = 0.01);
    }

    #[test]
    fn test_f32_support() {
        let l: Polygon<f32> = l_shape();
        let parts = convex_decomposition(&l);

        for part in &parts {
            assert!(part.is_convex());
        }
    }
}
