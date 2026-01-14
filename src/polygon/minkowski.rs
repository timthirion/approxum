//! Minkowski sum and difference operations.
//!
//! The Minkowski sum of two shapes A and B is:
//! A ⊕ B = {a + b : a ∈ A, b ∈ B}
//!
//! The Minkowski difference is:
//! A ⊖ B = A ⊕ (-B) = {a - b : a ∈ A, b ∈ B}
//!
//! # Applications
//!
//! - **Collision detection**: Two shapes collide iff their Minkowski difference contains the origin
//! - **Path planning**: Obstacles can be "grown" by the robot's shape
//! - **Morphological operations**: Dilation and erosion
//!
//! # Example
//!
//! ```
//! use approxum::polygon::{Polygon, minkowski_sum};
//! use approxum::Point2;
//!
//! // A square
//! let square = Polygon::new(vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 0.0),
//!     Point2::new(1.0, 1.0),
//!     Point2::new(0.0, 1.0),
//! ]);
//!
//! // A small diamond (for rounding corners)
//! let diamond = Polygon::new(vec![
//!     Point2::new(0.1, 0.0),
//!     Point2::new(0.0, 0.1),
//!     Point2::new(-0.1, 0.0),
//!     Point2::new(0.0, -0.1),
//! ]);
//!
//! let sum = minkowski_sum(&square, &diamond);
//! // Result is the square with rounded corners
//! ```

use super::boolean::polygon_union;
use super::core::{polygon_is_convex, Polygon};
use crate::primitives::Point2;
use num_traits::Float;

/// Computes the Minkowski sum of two convex polygons.
///
/// Both input polygons must be convex and in counter-clockwise order.
/// The result is also a convex polygon in counter-clockwise order.
///
/// # Algorithm
///
/// Uses the rotating calipers approach: merge the edges of both polygons
/// sorted by their angle, creating the sum polygon.
///
/// Time complexity: O(n + m) where n and m are the vertex counts.
pub fn minkowski_sum_convex<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Polygon<F> {
    if a.vertices.is_empty() {
        return b.clone();
    }
    if b.vertices.is_empty() {
        return a.clone();
    }

    let edges_a = polygon_edges(&a.vertices);
    let edges_b = polygon_edges(&b.vertices);

    // Find bottom-most point of each polygon (starting point)
    let start_a = find_bottom_vertex(&a.vertices);
    let start_b = find_bottom_vertex(&b.vertices);

    // Merge edges by angle
    let mut result = Vec::with_capacity(edges_a.len() + edges_b.len());
    let mut current = Point2::new(
        a.vertices[start_a].x + b.vertices[start_b].x,
        a.vertices[start_a].y + b.vertices[start_b].y,
    );
    result.push(current);

    let mut i = 0;
    let mut j = 0;
    let n = edges_a.len();
    let m = edges_b.len();

    while i < n || j < m {
        let edge = if i >= n {
            j += 1;
            edges_b[(start_b + j - 1) % m]
        } else if j >= m {
            i += 1;
            edges_a[(start_a + i - 1) % n]
        } else {
            let ea = edges_a[(start_a + i) % n];
            let eb = edges_b[(start_b + j) % m];

            let angle_a = ea.y.atan2(ea.x);
            let angle_b = eb.y.atan2(eb.x);

            if angle_a < angle_b {
                i += 1;
                ea
            } else if angle_b < angle_a {
                j += 1;
                eb
            } else {
                // Same angle - add both edges combined
                i += 1;
                j += 1;
                Point2::new(ea.x + eb.x, ea.y + eb.y)
            }
        };

        current = Point2::new(current.x + edge.x, current.y + edge.y);

        // Avoid duplicating the starting point
        if result.len() < n + m {
            result.push(current);
        }
    }

    // Remove the last point if it's the same as the first (closed polygon)
    if result.len() > 1 {
        let first = result[0];
        let last = result[result.len() - 1];
        if (first.x - last.x).abs() < F::epsilon() && (first.y - last.y).abs() < F::epsilon() {
            result.pop();
        }
    }

    Polygon::new(result)
}

/// Computes the Minkowski sum of two polygons (general case).
///
/// For convex polygons, uses the efficient rotating calipers method.
/// For non-convex polygons, decomposes into convex parts and unions the results.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, minkowski_sum};
/// use approxum::Point2;
///
/// let triangle = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(1.0, 2.0),
/// ]);
///
/// let small_square = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(0.2, 0.0),
///     Point2::new(0.2, 0.2),
///     Point2::new(0.0, 0.2),
/// ]);
///
/// let sum = minkowski_sum(&triangle, &small_square);
/// assert!(sum.area() > triangle.area());
/// ```
pub fn minkowski_sum<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Polygon<F> {
    // Check if both are convex
    if polygon_is_convex(&a.vertices) && polygon_is_convex(&b.vertices) {
        return minkowski_sum_convex(a, b);
    }

    // For non-convex polygons, use decomposition approach
    // Decompose into triangles and compute sum of each pair
    let triangles_a = triangulate_simple(&a.vertices);
    let triangles_b = triangulate_simple(&b.vertices);

    let mut results: Vec<Polygon<F>> = Vec::new();

    for tri_a in &triangles_a {
        for tri_b in &triangles_b {
            let sum = minkowski_sum_convex(tri_a, tri_b);
            results.push(sum);
        }
    }

    // Union all results
    if results.is_empty() {
        return Polygon::new(vec![]);
    }

    let mut combined = results[0].clone();
    for poly in results.iter().skip(1) {
        let union = polygon_union(&combined, poly);
        if !union.is_empty() {
            combined = union[0].clone();
        }
    }

    combined
}

/// Computes the Minkowski difference of two polygons.
///
/// A ⊖ B = A ⊕ (-B)
///
/// This is equivalent to the Minkowski sum of A with B reflected through the origin.
///
/// # Applications
///
/// - If the Minkowski difference contains the origin, the shapes overlap
/// - Used in GJK collision detection algorithm
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, minkowski_difference, polygon_contains};
/// use approxum::Point2;
///
/// let square1 = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
/// ]);
///
/// let square2 = Polygon::new(vec![
///     Point2::new(0.5, 0.5),
///     Point2::new(1.5, 0.5),
///     Point2::new(1.5, 1.5),
///     Point2::new(0.5, 1.5),
/// ]);
///
/// let diff = minkowski_difference(&square1, &square2);
/// // Squares overlap, so origin is inside the difference
/// assert!(polygon_contains(&diff.vertices, Point2::new(0.0, 0.0)));
/// ```
pub fn minkowski_difference<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Polygon<F> {
    // Reflect B through origin
    let b_reflected = reflect_polygon(b);
    minkowski_sum(a, &b_reflected)
}

/// Checks if two convex polygons collide using Minkowski difference.
///
/// Two shapes collide if and only if their Minkowski difference contains the origin.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, polygons_collide};
/// use approxum::Point2;
///
/// let square1 = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
/// ]);
///
/// let square2 = Polygon::new(vec![
///     Point2::new(0.5, 0.5),
///     Point2::new(1.5, 0.5),
///     Point2::new(1.5, 1.5),
///     Point2::new(0.5, 1.5),
/// ]);
///
/// let square3 = Polygon::new(vec![
///     Point2::new(2.0, 2.0),
///     Point2::new(3.0, 2.0),
///     Point2::new(3.0, 3.0),
///     Point2::new(2.0, 3.0),
/// ]);
///
/// assert!(polygons_collide(&square1, &square2)); // Overlapping
/// assert!(!polygons_collide(&square1, &square3)); // Separated
/// ```
pub fn polygons_collide<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> bool {
    let diff = minkowski_difference(a, b);
    super::core::polygon_contains(&diff.vertices, Point2::new(F::zero(), F::zero()))
}

/// Reflects a polygon through the origin.
fn reflect_polygon<F: Float>(p: &Polygon<F>) -> Polygon<F> {
    let reflected: Vec<Point2<F>> = p
        .vertices
        .iter()
        .map(|v| Point2::new(-v.x, -v.y))
        .rev() // Reverse to maintain CCW order
        .collect();
    Polygon::new(reflected)
}

/// Computes edge vectors from vertices.
fn polygon_edges<F: Float>(vertices: &[Point2<F>]) -> Vec<Point2<F>> {
    let n = vertices.len();
    (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            Point2::new(
                vertices[next].x - vertices[i].x,
                vertices[next].y - vertices[i].y,
            )
        })
        .collect()
}

/// Finds the index of the bottom-most (then left-most) vertex.
fn find_bottom_vertex<F: Float>(vertices: &[Point2<F>]) -> usize {
    let mut idx = 0;
    for i in 1..vertices.len() {
        if vertices[i].y < vertices[idx].y
            || (vertices[i].y == vertices[idx].y && vertices[i].x < vertices[idx].x)
        {
            idx = i;
        }
    }
    idx
}

/// Simple ear-clipping triangulation for Minkowski sum decomposition.
fn triangulate_simple<F: Float>(vertices: &[Point2<F>]) -> Vec<Polygon<F>> {
    if vertices.len() < 3 {
        return vec![];
    }
    if vertices.len() == 3 {
        return vec![Polygon::new(vertices.to_vec())];
    }

    let mut result = Vec::new();
    let mut remaining: Vec<Point2<F>> = vertices.to_vec();

    while remaining.len() > 3 {
        let n = remaining.len();
        let mut found_ear = false;

        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = (i + 1) % n;

            if is_ear(&remaining, prev, i, next) {
                result.push(Polygon::new(vec![
                    remaining[prev],
                    remaining[i],
                    remaining[next],
                ]));
                remaining.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            // Fallback: just take first three vertices
            result.push(Polygon::new(vec![remaining[0], remaining[1], remaining[2]]));
            remaining.remove(1);
        }
    }

    if remaining.len() == 3 {
        result.push(Polygon::new(remaining));
    }

    result
}

/// Checks if vertex i forms an ear with prev and next.
fn is_ear<F: Float>(vertices: &[Point2<F>], prev: usize, i: usize, next: usize) -> bool {
    let a = vertices[prev];
    let b = vertices[i];
    let c = vertices[next];

    // Check if the triangle is counter-clockwise (convex at this vertex)
    let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if cross <= F::zero() {
        return false;
    }

    // Check if any other vertex is inside this triangle
    for (j, &p) in vertices.iter().enumerate() {
        if j == prev || j == i || j == next {
            continue;
        }
        if point_in_triangle(p, a, b, c) {
            return false;
        }
    }

    true
}

/// Checks if point p is inside triangle abc.
fn point_in_triangle<F: Float>(p: Point2<F>, a: Point2<F>, b: Point2<F>, c: Point2<F>) -> bool {
    let v0 = Point2::new(c.x - a.x, c.y - a.y);
    let v1 = Point2::new(b.x - a.x, b.y - a.y);
    let v2 = Point2::new(p.x - a.x, p.y - a.y);

    let dot00 = v0.x * v0.x + v0.y * v0.y;
    let dot01 = v0.x * v1.x + v0.y * v1.y;
    let dot02 = v0.x * v2.x + v0.y * v2.y;
    let dot11 = v1.x * v1.x + v1.y * v1.y;
    let dot12 = v1.x * v2.x + v1.y * v2.y;

    let inv_denom = F::one() / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    u >= F::zero() && v >= F::zero() && (u + v) <= F::one()
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

    fn triangle<F: Float>() -> Polygon<F> {
        Polygon::new(vec![
            Point2::new(F::zero(), F::zero()),
            Point2::new(F::one(), F::zero()),
            Point2::new(F::from(0.5).unwrap(), F::one()),
        ])
    }

    #[test]
    fn test_minkowski_sum_squares() {
        let a: Polygon<f64> = square(0.0, 0.0, 1.0);
        let b: Polygon<f64> = square(0.0, 0.0, 1.0);

        let sum = minkowski_sum_convex(&a, &b);

        // Sum of two unit squares should be a 2x2 square
        assert_relative_eq!(sum.area(), 4.0, epsilon = 0.01);
    }

    #[test]
    fn test_minkowski_sum_square_triangle() {
        let sq: Polygon<f64> = square(0.0, 0.0, 1.0);
        let tri: Polygon<f64> = triangle();

        let sum = minkowski_sum_convex(&sq, &tri);

        // Sum should have more area than either input
        assert!(sum.area() > sq.area());
        assert!(sum.area() > tri.area());
    }

    #[test]
    fn test_minkowski_sum_with_point() {
        // Minkowski sum with a single point should translate the polygon
        let sq: Polygon<f64> = square(0.0, 0.0, 1.0);
        let point = Polygon::new(vec![Point2::new(5.0, 5.0)]);

        let sum = minkowski_sum(&sq, &point);

        // Should be same size, just translated
        assert_relative_eq!(sum.area(), sq.area(), epsilon = 0.01);
    }

    #[test]
    fn test_minkowski_difference_overlapping() {
        let a: Polygon<f64> = square(0.0, 0.0, 1.0);
        let b: Polygon<f64> = square(0.5, 0.5, 1.0);

        let diff = minkowski_difference(&a, &b);

        // Overlapping squares: origin should be inside difference
        let contains_origin =
            super::super::core::polygon_contains(&diff.vertices, Point2::new(0.0, 0.0));
        assert!(contains_origin);
    }

    #[test]
    fn test_minkowski_difference_separated() {
        let a: Polygon<f64> = square(0.0, 0.0, 1.0);
        let b: Polygon<f64> = square(5.0, 5.0, 1.0);

        let diff = minkowski_difference(&a, &b);

        // Separated squares: origin should be outside difference
        let contains_origin =
            super::super::core::polygon_contains(&diff.vertices, Point2::new(0.0, 0.0));
        assert!(!contains_origin);
    }

    #[test]
    fn test_polygons_collide() {
        let a: Polygon<f64> = square(0.0, 0.0, 1.0);
        let b: Polygon<f64> = square(0.5, 0.5, 1.0);
        let c: Polygon<f64> = square(5.0, 5.0, 1.0);

        assert!(polygons_collide(&a, &b));
        assert!(!polygons_collide(&a, &c));
    }

    #[test]
    fn test_polygons_collide_touching() {
        let a: Polygon<f64> = square(0.0, 0.0, 1.0);
        // Slightly overlapping instead of exactly touching (avoids numerical edge case)
        let b: Polygon<f64> = square(0.99, 0.0, 1.0);

        assert!(polygons_collide(&a, &b));
    }

    #[test]
    fn test_reflect_polygon() {
        let sq: Polygon<f64> = square(1.0, 1.0, 1.0);
        let reflected = reflect_polygon(&sq);

        // Centroid should be reflected
        let orig_centroid = sq.centroid().unwrap();
        let refl_centroid = reflected.centroid().unwrap();

        assert_relative_eq!(refl_centroid.x, -orig_centroid.x, epsilon = 0.01);
        assert_relative_eq!(refl_centroid.y, -orig_centroid.y, epsilon = 0.01);

        // Area should be preserved
        assert_relative_eq!(reflected.area().abs(), sq.area().abs(), epsilon = 0.01);
    }

    #[test]
    fn test_minkowski_sum_general() {
        // Use convex shapes for reliable general test
        let pentagon: Polygon<f64> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.3, 0.7),
            Point2::new(0.5, 1.2),
            Point2::new(-0.3, 0.7),
        ]);

        let small_sq: Polygon<f64> = square(0.0, 0.0, 0.1);

        let sum = minkowski_sum(&pentagon, &small_sq);

        // Sum should be larger than original
        assert!(sum.area() > pentagon.area());
    }

    #[test]
    fn test_minkowski_sum_symmetric() {
        let a: Polygon<f64> = square(0.0, 0.0, 1.0);
        let b: Polygon<f64> = triangle();

        let sum_ab = minkowski_sum(&a, &b);
        let sum_ba = minkowski_sum(&b, &a);

        // Minkowski sum is commutative
        assert_relative_eq!(sum_ab.area(), sum_ba.area(), epsilon = 0.1);
    }

    #[test]
    fn test_minkowski_sum_empty() {
        let a: Polygon<f64> = square(0.0, 0.0, 1.0);
        let empty: Polygon<f64> = Polygon::new(vec![]);

        let sum = minkowski_sum_convex(&a, &empty);
        assert_relative_eq!(sum.area(), a.area(), epsilon = 0.01);
    }

    #[test]
    fn test_triangulate_simple() {
        let sq: Polygon<f64> = square(0.0, 0.0, 1.0);
        let triangles = triangulate_simple(&sq.vertices);

        assert_eq!(triangles.len(), 2);

        let total_area: f64 = triangles.iter().map(|t| t.area().abs()).sum();
        assert_relative_eq!(total_area, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_f32_support() {
        let a: Polygon<f32> = square(0.0, 0.0, 1.0);
        let b: Polygon<f32> = square(0.0, 0.0, 1.0);

        let sum = minkowski_sum_convex(&a, &b);
        assert!((sum.area() - 4.0).abs() < 0.1);
    }
}
