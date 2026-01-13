//! Hausdorff distance computation.
//!
//! The Hausdorff distance measures how far two point sets are from each other.
//! It's the maximum distance from any point in one set to its nearest point
//! in the other set.
//!
//! # Definition
//!
//! For point sets A and B:
//! - **Directed Hausdorff**: h(A,B) = max_{a∈A} min_{b∈B} d(a,b)
//! - **Hausdorff distance**: H(A,B) = max(h(A,B), h(B,A))
//!
//! # Use Cases
//!
//! - Measuring similarity between shapes
//! - Evaluating simplification quality
//! - Shape matching and registration
//! - Comparing polylines or polygons
//!
//! # Example
//!
//! ```
//! use approxum::tolerance::{hausdorff_distance, directed_hausdorff};
//! use approxum::Point2;
//!
//! let a = vec![
//!     Point2::new(0.0_f64, 0.0),
//!     Point2::new(1.0, 0.0),
//!     Point2::new(2.0, 0.0),
//! ];
//! let b = vec![
//!     Point2::new(0.0_f64, 0.5),
//!     Point2::new(1.0, 0.5),
//!     Point2::new(2.0, 0.5),
//! ];
//!
//! // All points in A are 0.5 away from their nearest point in B
//! let dist = hausdorff_distance(&a, &b);
//! assert!((dist - 0.5).abs() < 1e-10);
//! ```

use crate::primitives::{Point2, Segment2};
use num_traits::Float;

/// Computes the directed Hausdorff distance from point set A to point set B.
///
/// This is the maximum distance from any point in A to its nearest point in B:
/// h(A,B) = max_{a∈A} min_{b∈B} d(a,b)
///
/// Note: This is asymmetric. h(A,B) ≠ h(B,A) in general.
///
/// # Arguments
///
/// * `a` - Source point set
/// * `b` - Target point set
///
/// # Returns
///
/// The directed Hausdorff distance, or 0 if either set is empty.
///
/// # Complexity
///
/// O(n*m) where n = |A| and m = |B|.
pub fn directed_hausdorff<F: Float>(a: &[Point2<F>], b: &[Point2<F>]) -> F {
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }

    let mut max_dist = F::zero();

    for pa in a {
        // Find minimum distance from pa to any point in b
        let mut min_dist_sq = F::infinity();
        for pb in b {
            let dist_sq = pa.distance_squared(*pb);
            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
            }
        }

        if min_dist_sq > max_dist * max_dist {
            max_dist = min_dist_sq.sqrt();
        }
    }

    max_dist
}

/// Computes the (symmetric) Hausdorff distance between two point sets.
///
/// H(A,B) = max(h(A,B), h(B,A))
///
/// This is the true Hausdorff distance, which is symmetric.
///
/// # Arguments
///
/// * `a` - First point set
/// * `b` - Second point set
///
/// # Returns
///
/// The Hausdorff distance, or 0 if either set is empty.
///
/// # Complexity
///
/// O(n*m) where n = |A| and m = |B|.
///
/// # Example
///
/// ```
/// use approxum::tolerance::hausdorff_distance;
/// use approxum::Point2;
///
/// let a = vec![
///     Point2::new(0.0_f64, 0.0),
///     Point2::new(1.0, 0.0),
/// ];
/// let b = vec![
///     Point2::new(0.0_f64, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(2.0, 0.0),  // Extra point, 1.0 away from nearest in a
/// ];
///
/// // The extra point (2,0) is 1.0 away from nearest point (1,0) in a
/// let dist = hausdorff_distance(&a, &b);
/// assert!((dist - 1.0).abs() < 0.01);
/// ```
pub fn hausdorff_distance<F: Float>(a: &[Point2<F>], b: &[Point2<F>]) -> F {
    let h_ab = directed_hausdorff(a, b);
    let h_ba = directed_hausdorff(b, a);
    h_ab.max(h_ba)
}

/// Computes the directed Hausdorff distance from polyline A to polyline B.
///
/// Unlike the point-based version, this considers the line segments, not just
/// the vertices. It samples points along the segments of A and computes the
/// minimum distance to the segments of B.
///
/// # Arguments
///
/// * `a` - Source polyline vertices
/// * `b` - Target polyline vertices
/// * `sample_density` - Number of samples per unit length along polyline A
///
/// # Returns
///
/// The directed Hausdorff distance between polylines.
///
/// # Complexity
///
/// O(s * m) where s is the number of samples and m is the number of segments in B.
pub fn directed_hausdorff_polyline<F: Float>(
    a: &[Point2<F>],
    b: &[Point2<F>],
    sample_density: F,
) -> F {
    if a.len() < 2 || b.len() < 2 {
        return directed_hausdorff(a, b);
    }

    let mut max_dist = F::zero();

    // Sample points along polyline A
    for i in 0..a.len() - 1 {
        let seg_a = Segment2::new(a[i], a[i + 1]);
        let seg_len = seg_a.length();

        // Determine number of samples for this segment
        let num_samples = (seg_len * sample_density).ceil().max(F::one());
        let num_samples_int = num_samples.to_usize().unwrap_or(1).max(1);

        for j in 0..=num_samples_int {
            let t = F::from(j).unwrap() / num_samples;
            let sample_point = seg_a.point_at(t);

            // Find minimum distance from sample_point to any segment in B
            let min_dist = min_distance_to_polyline(sample_point, b);

            if min_dist > max_dist {
                max_dist = min_dist;
            }
        }
    }

    max_dist
}

/// Computes the (symmetric) Hausdorff distance between two polylines.
///
/// This considers the full polyline geometry, not just vertices.
///
/// # Arguments
///
/// * `a` - First polyline vertices
/// * `b` - Second polyline vertices
/// * `sample_density` - Number of samples per unit length
///
/// # Returns
///
/// The Hausdorff distance between polylines.
pub fn hausdorff_distance_polyline<F: Float>(
    a: &[Point2<F>],
    b: &[Point2<F>],
    sample_density: F,
) -> F {
    let h_ab = directed_hausdorff_polyline(a, b, sample_density);
    let h_ba = directed_hausdorff_polyline(b, a, sample_density);
    h_ab.max(h_ba)
}

/// Computes the exact Hausdorff distance between two polylines.
///
/// For polygonal chains, the maximum Hausdorff distance always occurs at a
/// vertex of one polyline. This algorithm checks the distance from each vertex
/// to the opposite polyline.
///
/// More accurate than the sampling-based approach and runs in O(n*m) time.
///
/// # Arguments
///
/// * `a` - First polyline vertices
/// * `b` - Second polyline vertices
///
/// # Returns
///
/// The exact Hausdorff distance between polylines.
pub fn hausdorff_distance_polyline_exact<F: Float>(a: &[Point2<F>], b: &[Point2<F>]) -> F {
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }

    if a.len() == 1 && b.len() == 1 {
        return a[0].distance(b[0]);
    }

    let mut max_dist = F::zero();

    // Check distances from all vertices of A to polyline B
    for pa in a {
        let dist = min_distance_to_polyline(*pa, b);
        if dist > max_dist {
            max_dist = dist;
        }
    }

    // Check distances from all vertices of B to polyline A
    for pb in b {
        let dist = min_distance_to_polyline(*pb, a);
        if dist > max_dist {
            max_dist = dist;
        }
    }

    max_dist
}

/// Computes the minimum distance from a point to a polyline.
fn min_distance_to_polyline<F: Float>(p: Point2<F>, polyline: &[Point2<F>]) -> F {
    if polyline.is_empty() {
        return F::infinity();
    }

    if polyline.len() == 1 {
        return p.distance(polyline[0]);
    }

    let mut min_dist_sq = F::infinity();

    for i in 0..polyline.len() - 1 {
        let seg = Segment2::new(polyline[i], polyline[i + 1]);
        let dist_sq = seg.distance_squared_to_point(p);
        if dist_sq < min_dist_sq {
            min_dist_sq = dist_sq;
        }
    }

    min_dist_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_directed_hausdorff_identical() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];

        let dist = directed_hausdorff(&points, &points);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_directed_hausdorff_shifted() {
        let a = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let b = vec![
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
        ];

        // All points in A are exactly 1.0 away from nearest point in B
        let dist = directed_hausdorff(&a, &b);
        assert!(approx_eq(dist, 1.0, 1e-10));
    }

    #[test]
    fn test_directed_hausdorff_asymmetric() {
        let a = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        let b = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0), // Extra point
        ];

        // A → B: max distance is 0 (both points of A are in B)
        let h_ab = directed_hausdorff(&a, &b);
        assert!(approx_eq(h_ab, 0.0, 1e-10));

        // B → A: max distance is 1.0 (point (2,0) is 1.0 from nearest point in A)
        let h_ba = directed_hausdorff(&b, &a);
        assert!(approx_eq(h_ba, 1.0, 1e-10));
    }

    #[test]
    fn test_hausdorff_symmetric() {
        let a = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        let b = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];

        let dist = hausdorff_distance(&a, &b);
        assert!(approx_eq(dist, 1.0, 1e-10));

        // Should be symmetric
        let dist_rev = hausdorff_distance(&b, &a);
        assert!(approx_eq(dist, dist_rev, 1e-10));
    }

    #[test]
    fn test_hausdorff_empty() {
        let a: Vec<Point2<f64>> = vec![];
        let b = vec![Point2::new(0.0, 0.0)];

        assert_eq!(hausdorff_distance(&a, &b), 0.0);
        assert_eq!(hausdorff_distance(&b, &a), 0.0);
    }

    #[test]
    fn test_hausdorff_single_point() {
        let a = vec![Point2::new(0.0, 0.0)];
        let b = vec![Point2::new(3.0, 4.0)];

        let dist = hausdorff_distance(&a, &b);
        assert!(approx_eq(dist, 5.0, 1e-10)); // 3-4-5 triangle
    }

    #[test]
    fn test_hausdorff_polyline_exact_line() {
        // Original line: (0,0) to (2,0)
        // Simplified: same points
        let a = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];
        let b = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];

        let dist = hausdorff_distance_polyline_exact(&a, &b);
        assert!(approx_eq(dist, 0.0, 1e-10));
    }

    #[test]
    fn test_hausdorff_polyline_exact_triangle() {
        // Triangle with a point removed
        let original = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];
        let simplified = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];

        // The middle point (1,1) is distance 1.0 from the line (0,0)-(2,0)
        // (straight down to (1,0) on the x-axis)
        let dist = hausdorff_distance_polyline_exact(&original, &simplified);
        assert!(approx_eq(dist, 1.0, 0.01));
    }

    #[test]
    fn test_hausdorff_polyline_sampled() {
        let a = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];
        let b = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];

        // With high sample density, should be close to exact (1.0)
        let dist = hausdorff_distance_polyline(&a, &b, 10.0);
        assert!(approx_eq(dist, 1.0, 0.1));
    }

    #[test]
    fn test_hausdorff_polyline_parallel_lines() {
        // Two parallel horizontal lines
        let a = vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)];
        let b = vec![Point2::new(0.0, 2.0), Point2::new(10.0, 2.0)];

        let dist = hausdorff_distance_polyline_exact(&a, &b);
        assert!(approx_eq(dist, 2.0, 1e-10));
    }

    #[test]
    fn test_hausdorff_simplification_quality() {
        // A simple bump: goes up and back down
        let original = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 1.0), // Peak
            Point2::new(3.0, 0.0),
            Point2::new(4.0, 0.0),
        ];

        // Simplified to just a straight line
        let simplified = vec![Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)];

        // Hausdorff distance should be 1.0 (the peak height)
        let dist = hausdorff_distance_polyline_exact(&original, &simplified);
        assert!(approx_eq(dist, 1.0, 0.01));
    }

    #[test]
    fn test_min_distance_to_polyline() {
        let polyline = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
        ];

        // Point directly above first segment
        let p1 = Point2::new(5.0, 3.0);
        assert!(approx_eq(min_distance_to_polyline(p1, &polyline), 3.0, 1e-10));

        // Point at corner
        let p2 = Point2::new(10.0, 0.0);
        assert!(approx_eq(min_distance_to_polyline(p2, &polyline), 0.0, 1e-10));
    }

    #[test]
    fn test_f32() {
        let a: Vec<Point2<f32>> = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        let b: Vec<Point2<f32>> = vec![Point2::new(0.0, 1.0), Point2::new(1.0, 1.0)];

        let dist = hausdorff_distance(&a, &b);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_directed_hausdorff_polyline() {
        // Line going up
        let a = vec![Point2::new(0.0, 0.0), Point2::new(0.0, 10.0)];
        // Line shifted right
        let b = vec![Point2::new(2.0, 0.0), Point2::new(2.0, 10.0)];

        // Every point on A is 2.0 away from B
        let dist = directed_hausdorff_polyline(&a, &b, 1.0);
        assert!(approx_eq(dist, 2.0, 0.1));
    }
}
