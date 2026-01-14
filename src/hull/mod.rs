//! Convex hull algorithms.
//!
//! This module provides algorithms for computing convex hulls of point sets.
//!
//! # Example
//!
//! ```
//! use approxum::hull::convex_hull;
//! use approxum::Point2;
//!
//! let points: Vec<Point2<f64>> = vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 0.0),
//!     Point2::new(0.5, 0.5), // Interior point
//!     Point2::new(1.0, 1.0),
//!     Point2::new(0.0, 1.0),
//! ];
//!
//! let hull = convex_hull(&points);
//!
//! // Hull should be the 4 corners (interior point excluded)
//! assert_eq!(hull.len(), 4);
//! ```

use crate::primitives::Point2;
use num_traits::Float;

/// Computes the convex hull of a set of points using Andrew's monotone chain algorithm.
///
/// Returns the hull vertices in counter-clockwise order, starting from the
/// bottom-left point. The first and last points are NOT the same (the hull
/// is implicitly closed).
///
/// # Algorithm
///
/// Andrew's monotone chain algorithm:
/// 1. Sort points lexicographically (by x, then by y)
/// 2. Build the lower hull from left to right
/// 3. Build the upper hull from right to left
/// 4. Concatenate, removing duplicate endpoints
///
/// # Complexity
///
/// - Time: O(n log n) due to sorting
/// - Space: O(n)
///
/// # Arguments
///
/// * `points` - The input points
///
/// # Returns
///
/// The convex hull vertices in CCW order. Returns an empty vector if fewer
/// than 1 point is provided, a single point if 1 point is provided, or
/// 2 points if all points are collinear.
///
/// # Example
///
/// ```
/// use approxum::hull::convex_hull;
/// use approxum::Point2;
///
/// let points: Vec<Point2<f64>> = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(0.0, 2.0),
/// ];
///
/// let hull = convex_hull(&points);
/// assert_eq!(hull.len(), 4); // Square corners
/// ```
pub fn convex_hull<F: Float>(points: &[Point2<F>]) -> Vec<Point2<F>> {
    if points.is_empty() {
        return Vec::new();
    }
    if points.len() == 1 {
        return vec![points[0]];
    }
    if points.len() == 2 {
        return vec![points[0], points[1]];
    }

    // Sort points lexicographically
    let mut sorted: Vec<Point2<F>> = points.to_vec();
    sorted.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Build lower hull
    let mut lower: Vec<Point2<F>> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2
            && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], &p) <= F::zero()
        {
            lower.pop();
        }
        lower.push(p);
    }

    // Build upper hull
    let mut upper: Vec<Point2<F>> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2
            && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], &p) <= F::zero()
        {
            upper.pop();
        }
        upper.push(p);
    }

    // Remove last point of each half because it's repeated
    lower.pop();
    upper.pop();

    // Concatenate
    lower.extend(upper);
    lower
}

/// Computes the convex hull and returns the indices of hull vertices.
///
/// This is useful when you need to track which original points are on the hull.
///
/// # Arguments
///
/// * `points` - The input points
///
/// # Returns
///
/// Indices into the original points array, in CCW order.
///
/// # Example
///
/// ```
/// use approxum::hull::convex_hull_indices;
/// use approxum::Point2;
///
/// let points: Vec<Point2<f64>> = vec![
///     Point2::new(0.0, 0.0),  // 0
///     Point2::new(1.0, 0.0),  // 1
///     Point2::new(0.5, 0.5),  // 2 - interior
///     Point2::new(1.0, 1.0),  // 3
///     Point2::new(0.0, 1.0),  // 4
/// ];
///
/// let indices = convex_hull_indices(&points);
///
/// // Should not include index 2 (interior point)
/// assert!(!indices.contains(&2));
/// assert_eq!(indices.len(), 4);
/// ```
pub fn convex_hull_indices<F: Float>(points: &[Point2<F>]) -> Vec<usize> {
    if points.is_empty() {
        return Vec::new();
    }
    if points.len() == 1 {
        return vec![0];
    }
    if points.len() == 2 {
        return vec![0, 1];
    }

    // Create indexed points and sort
    let mut indexed: Vec<(usize, Point2<F>)> = points.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Build lower hull
    let mut lower: Vec<(usize, Point2<F>)> = Vec::new();
    for &(idx, p) in &indexed {
        while lower.len() >= 2
            && cross(&lower[lower.len() - 2].1, &lower[lower.len() - 1].1, &p) <= F::zero()
        {
            lower.pop();
        }
        lower.push((idx, p));
    }

    // Build upper hull
    let mut upper: Vec<(usize, Point2<F>)> = Vec::new();
    for &(idx, p) in indexed.iter().rev() {
        while upper.len() >= 2
            && cross(&upper[upper.len() - 2].1, &upper[upper.len() - 1].1, &p) <= F::zero()
        {
            upper.pop();
        }
        upper.push((idx, p));
    }

    // Remove duplicate endpoints
    lower.pop();
    upper.pop();

    // Extract indices
    lower.extend(upper);
    lower.into_iter().map(|(idx, _)| idx).collect()
}

/// Computes the area of a convex hull.
///
/// Uses the shoelace formula on the hull vertices.
///
/// # Arguments
///
/// * `hull` - The convex hull vertices in order (CCW or CW)
///
/// # Returns
///
/// The area of the hull. Returns 0 for fewer than 3 vertices.
///
/// # Example
///
/// ```
/// use approxum::hull::{convex_hull, convex_hull_area};
/// use approxum::Point2;
///
/// let points: Vec<Point2<f64>> = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(0.0, 2.0),
/// ];
///
/// let hull = convex_hull(&points);
/// let area = convex_hull_area(&hull);
///
/// assert!((area - 4.0).abs() < 1e-10); // 2x2 square
/// ```
pub fn convex_hull_area<F: Float>(hull: &[Point2<F>]) -> F {
    if hull.len() < 3 {
        return F::zero();
    }

    let mut area = F::zero();
    let n = hull.len();

    for i in 0..n {
        let j = (i + 1) % n;
        area = area + hull[i].x * hull[j].y;
        area = area - hull[j].x * hull[i].y;
    }

    area.abs() / F::from(2.0).unwrap()
}

/// Computes the perimeter of a convex hull.
///
/// # Arguments
///
/// * `hull` - The convex hull vertices in order
///
/// # Returns
///
/// The perimeter (sum of edge lengths). Returns 0 for fewer than 2 vertices.
///
/// # Example
///
/// ```
/// use approxum::hull::{convex_hull, convex_hull_perimeter};
/// use approxum::Point2;
///
/// let points: Vec<Point2<f64>> = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
/// ];
///
/// let hull = convex_hull(&points);
/// let perimeter = convex_hull_perimeter(&hull);
///
/// assert!((perimeter - 4.0).abs() < 1e-10); // Unit square
/// ```
pub fn convex_hull_perimeter<F: Float>(hull: &[Point2<F>]) -> F {
    if hull.len() < 2 {
        return F::zero();
    }

    let mut perimeter = F::zero();
    let n = hull.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let dx = hull[j].x - hull[i].x;
        let dy = hull[j].y - hull[i].y;
        perimeter = perimeter + (dx * dx + dy * dy).sqrt();
    }

    perimeter
}

/// Tests if a point is inside a convex hull.
///
/// A point on the boundary is considered inside.
///
/// # Arguments
///
/// * `hull` - The convex hull vertices in CCW order
/// * `point` - The point to test
///
/// # Returns
///
/// `true` if the point is inside or on the boundary of the hull.
///
/// # Example
///
/// ```
/// use approxum::hull::{convex_hull, point_in_convex_hull};
/// use approxum::Point2;
///
/// let points: Vec<Point2<f64>> = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(0.0, 2.0),
/// ];
///
/// let hull = convex_hull(&points);
///
/// assert!(point_in_convex_hull(&hull, Point2::new(1.0, 1.0))); // Center
/// assert!(point_in_convex_hull(&hull, Point2::new(0.0, 0.0))); // Corner
/// assert!(!point_in_convex_hull(&hull, Point2::new(3.0, 3.0))); // Outside
/// ```
pub fn point_in_convex_hull<F: Float>(hull: &[Point2<F>], point: Point2<F>) -> bool {
    if hull.len() < 3 {
        return false;
    }

    // For a convex hull in CCW order, the point is inside if it's
    // on the left side (or on) every edge
    let n = hull.len();
    for i in 0..n {
        let j = (i + 1) % n;
        if cross(&hull[i], &hull[j], &point) < F::zero() {
            return false;
        }
    }

    true
}

/// Computes the convex hull of a set of points and returns both hull and area.
///
/// This is more efficient than calling `convex_hull` and `convex_hull_area`
/// separately if you need both.
///
/// # Example
///
/// ```
/// use approxum::hull::convex_hull_with_area;
/// use approxum::Point2;
///
/// let points: Vec<Point2<f64>> = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
///     Point2::new(0.5, 0.5), // Interior
/// ];
///
/// let (hull, area) = convex_hull_with_area(&points);
///
/// assert_eq!(hull.len(), 4);
/// assert!((area - 1.0).abs() < 1e-10);
/// ```
pub fn convex_hull_with_area<F: Float>(points: &[Point2<F>]) -> (Vec<Point2<F>>, F) {
    let hull = convex_hull(points);
    let area = convex_hull_area(&hull);
    (hull, area)
}

/// Cross product of vectors OA and OB where O is the origin point.
/// Positive if counter-clockwise, negative if clockwise, zero if collinear.
#[inline]
fn cross<F: Float>(o: &Point2<F>, a: &Point2<F>, b: &Point2<F>) -> F {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_convex_hull_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let hull = convex_hull(&points);
        assert!(hull.is_empty());
    }

    #[test]
    fn test_convex_hull_single() {
        let points = vec![Point2::new(1.0_f64, 2.0)];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 1);
        assert_eq!(hull[0], points[0]);
    }

    #[test]
    fn test_convex_hull_two_points() {
        let points = vec![Point2::new(0.0_f64, 0.0), Point2::new(1.0, 1.0)];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 2);
    }

    #[test]
    fn test_convex_hull_triangle() {
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_convex_hull_square() {
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_convex_hull_with_interior() {
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
            Point2::new(1.0, 1.0), // Interior point
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4); // Interior point excluded
    }

    #[test]
    fn test_convex_hull_collinear() {
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        ];
        let hull = convex_hull(&points);
        // Collinear points: hull is just the two endpoints
        assert_eq!(hull.len(), 2);
    }

    #[test]
    fn test_convex_hull_ccw_order() {
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let hull = convex_hull(&points);

        // Verify CCW order by checking cross products
        for i in 0..hull.len() {
            let j = (i + 1) % hull.len();
            let k = (i + 2) % hull.len();
            let cross_val = cross(&hull[i], &hull[j], &hull[k]);
            assert!(cross_val >= 0.0, "Hull not in CCW order at vertex {}", i);
        }
    }

    #[test]
    fn test_convex_hull_indices_basic() {
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 0.5), // Interior
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let indices = convex_hull_indices(&points);

        // Should not include index 2 (interior point)
        assert!(!indices.contains(&2));
        assert_eq!(indices.len(), 4);

        // All indices should be valid
        for &i in &indices {
            assert!(i < points.len());
        }
    }

    #[test]
    fn test_convex_hull_area_triangle() {
        let hull = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 2.0),
        ];
        let area = convex_hull_area(&hull);
        // Triangle area = 0.5 * base * height = 0.5 * 2 * 2 = 2
        assert!(approx_eq(area, 2.0, 1e-10));
    }

    #[test]
    fn test_convex_hull_area_square() {
        let hull = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        let area = convex_hull_area(&hull);
        assert!(approx_eq(area, 4.0, 1e-10));
    }

    #[test]
    fn test_convex_hull_area_empty() {
        let hull: Vec<Point2<f64>> = vec![];
        assert_eq!(convex_hull_area(&hull), 0.0);

        let hull = vec![Point2::new(0.0_f64, 0.0)];
        assert_eq!(convex_hull_area(&hull), 0.0);

        let hull = vec![Point2::new(0.0_f64, 0.0), Point2::new(1.0, 1.0)];
        assert_eq!(convex_hull_area(&hull), 0.0);
    }

    #[test]
    fn test_convex_hull_perimeter_square() {
        let hull = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let perimeter = convex_hull_perimeter(&hull);
        assert!(approx_eq(perimeter, 4.0, 1e-10));
    }

    #[test]
    fn test_convex_hull_perimeter_triangle() {
        // Equilateral triangle with side length 2
        let hull = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 3.0_f64.sqrt()),
        ];
        let perimeter = convex_hull_perimeter(&hull);
        // Perimeter = 3 * 2 = 6
        assert!(approx_eq(perimeter, 6.0, 1e-10));
    }

    #[test]
    fn test_point_in_convex_hull_inside() {
        let hull = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];

        // Center point
        assert!(point_in_convex_hull(&hull, Point2::new(1.0, 1.0)));

        // Off-center point
        assert!(point_in_convex_hull(&hull, Point2::new(0.5, 0.5)));
    }

    #[test]
    fn test_point_in_convex_hull_on_boundary() {
        let hull = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];

        // Corner
        assert!(point_in_convex_hull(&hull, Point2::new(0.0, 0.0)));

        // Edge midpoint
        assert!(point_in_convex_hull(&hull, Point2::new(1.0, 0.0)));
    }

    #[test]
    fn test_point_in_convex_hull_outside() {
        let hull = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];

        assert!(!point_in_convex_hull(&hull, Point2::new(3.0, 3.0)));
        assert!(!point_in_convex_hull(&hull, Point2::new(-1.0, 1.0)));
        assert!(!point_in_convex_hull(&hull, Point2::new(1.0, -1.0)));
    }

    #[test]
    fn test_convex_hull_with_area_combined() {
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.5, 0.5),
        ];

        let (hull, area) = convex_hull_with_area(&points);

        assert_eq!(hull.len(), 4);
        assert!(approx_eq(area, 1.0, 1e-10));
    }

    #[test]
    fn test_convex_hull_f32() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_convex_hull_many_interior_points() {
        let mut points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];

        // Add many interior points
        for i in 1..10 {
            for j in 1..10 {
                points.push(Point2::new(i as f64, j as f64));
            }
        }

        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4); // Only the 4 corners
    }

    #[test]
    fn test_convex_hull_duplicate_points() {
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(0.0, 0.0), // Duplicate
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let hull = convex_hull(&points);
        // Should still produce correct hull
        assert!(hull.len() >= 3);
    }

    #[test]
    fn test_convex_hull_pentagon() {
        // Regular pentagon-ish shape
        let points = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 1.5),
            Point2::new(1.0, 3.0),
            Point2::new(-1.0, 1.5),
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 5);
    }

    #[test]
    fn test_cross_product() {
        let o = Point2::new(0.0_f64, 0.0);
        let a = Point2::new(1.0, 0.0);
        let b = Point2::new(0.0, 1.0);

        // O -> A -> B is CCW, so cross product is positive
        assert!(cross(&o, &a, &b) > 0.0);

        // O -> B -> A is CW, so cross product is negative
        assert!(cross(&o, &b, &a) < 0.0);
    }
}
