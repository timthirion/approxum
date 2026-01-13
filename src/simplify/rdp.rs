//! Ramer-Douglas-Peucker polyline simplification.
//!
//! The RDP algorithm recursively simplifies a polyline by removing points
//! that are within a specified tolerance of the line segment connecting
//! the endpoints.
//!
//! Time complexity: O(n²) worst case, O(n log n) typical.

use crate::primitives::{Point2, Segment2};
use num_traits::Float;

/// Simplifies a polyline using the Ramer-Douglas-Peucker algorithm.
///
/// Returns a new vector containing only the points that remain after
/// simplification. The first and last points are always preserved.
///
/// # Arguments
///
/// * `points` - The input polyline as a slice of points
/// * `epsilon` - Distance tolerance. Points within this distance of the
///   simplified line are removed.
///
/// # Returns
///
/// A new vector with the simplified polyline. Returns an empty vector if
/// input has fewer than 2 points.
///
/// # Example
///
/// ```
/// use approxum::{Point2, simplify::rdp};
///
/// let points = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.1),  // Close to the line, will be removed
///     Point2::new(2.0, 0.0),
///     Point2::new(3.0, 2.0),  // Far from line, will be kept
///     Point2::new(4.0, 0.0),
/// ];
///
/// let simplified = rdp(&points, 0.5);
/// assert!(simplified.len() < points.len());
/// ```
pub fn rdp<F: Float>(points: &[Point2<F>], epsilon: F) -> Vec<Point2<F>> {
    let indices = rdp_indices(points, epsilon);
    indices.into_iter().map(|i| points[i]).collect()
}

/// Simplifies a polyline and returns the indices of retained points.
///
/// This is useful when you need to preserve the relationship between
/// the simplified points and associated data (e.g., timestamps, elevations).
///
/// # Arguments
///
/// * `points` - The input polyline as a slice of points
/// * `epsilon` - Distance tolerance
///
/// # Returns
///
/// A vector of indices into the original point array, in order.
/// Always includes index 0 and the last index if the input has >= 2 points.
pub fn rdp_indices<F: Float>(points: &[Point2<F>], epsilon: F) -> Vec<usize> {
    let n = points.len();
    if n < 2 {
        return (0..n).collect();
    }

    // Track which points to keep
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    // Recursively process
    rdp_recursive(points, 0, n - 1, epsilon, &mut keep);

    // Collect indices of kept points
    keep.iter()
        .enumerate()
        .filter_map(|(i, &k)| if k { Some(i) } else { None })
        .collect()
}

/// Recursive RDP implementation.
///
/// Processes the segment from `start` to `end` (inclusive).
fn rdp_recursive<F: Float>(
    points: &[Point2<F>],
    start: usize,
    end: usize,
    epsilon: F,
    keep: &mut [bool],
) {
    if end <= start + 1 {
        return; // No points between start and end
    }

    // Find the point with maximum distance from the line segment
    let segment = Segment2::new(points[start], points[end]);
    let mut max_dist = F::zero();
    let mut max_idx = start;

    for i in (start + 1)..end {
        let dist = segment.distance_to_point(points[i]);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    // If max distance exceeds epsilon, keep that point and recurse
    if max_dist > epsilon {
        keep[max_idx] = true;
        rdp_recursive(points, start, max_idx, epsilon, keep);
        rdp_recursive(points, max_idx, end, epsilon, keep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rdp_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let result = rdp(&points, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rdp_single_point() {
        let points = vec![Point2::new(1.0, 2.0)];
        let result = rdp(&points, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].x, 1.0);
    }

    #[test]
    fn test_rdp_two_points() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(10.0, 10.0)];
        let result = rdp(&points, 1.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_rdp_straight_line() {
        // Points on a straight line should simplify to just endpoints
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 3.0),
            Point2::new(4.0, 4.0),
        ];
        let result = rdp(&points, 0.1);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].x, 0.0);
        assert_eq!(result[1].x, 4.0);
    }

    #[test]
    fn test_rdp_l_shape() {
        // L-shaped path: corner point should be preserved
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(5.0, 0.0), // Corner
            Point2::new(5.0, 5.0),
        ];
        let result = rdp(&points, 0.1);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_rdp_with_noise() {
        // Line with small deviations that should be smoothed out
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.05), // Small deviation
            Point2::new(2.0, -0.03), // Small deviation
            Point2::new(3.0, 0.02), // Small deviation
            Point2::new(4.0, 0.0),
        ];
        let result = rdp(&points, 0.1);
        assert_eq!(result.len(), 2); // Only endpoints
    }

    #[test]
    fn test_rdp_preserves_significant_points() {
        // Path with a significant deviation in the middle
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(5.0, 5.0), // Significant deviation
            Point2::new(8.0, 0.0),
            Point2::new(10.0, 0.0),
        ];
        let result = rdp(&points, 0.5);

        // Should keep: start, peak, end (at minimum)
        assert!(result.len() >= 3);

        // The peak point should be preserved
        assert!(result.iter().any(|p| (p.x - 5.0).abs() < 0.01 && (p.y - 5.0).abs() < 0.01));
    }

    #[test]
    fn test_rdp_indices() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),  // 0 - keep
            Point2::new(1.0, 0.0),  // 1 - remove
            Point2::new(2.0, 0.0),  // 2 - remove
            Point2::new(3.0, 5.0),  // 3 - keep (significant deviation)
            Point2::new(4.0, 0.0),  // 4 - remove
            Point2::new(5.0, 0.0),  // 5 - keep (endpoint)
        ];
        let indices = rdp_indices(&points, 0.5);

        assert!(indices.contains(&0)); // First point
        assert!(indices.contains(&3)); // Peak
        assert!(indices.contains(&5)); // Last point
    }

    #[test]
    fn test_rdp_zero_epsilon() {
        // With epsilon = 0, should keep all non-collinear points
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];
        let result = rdp(&points, 0.0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_rdp_large_epsilon() {
        // With very large epsilon, should simplify to just endpoints
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 100.0),
            Point2::new(2.0, -100.0),
            Point2::new(3.0, 50.0),
            Point2::new(4.0, 0.0),
        ];
        let result = rdp(&points, 1000.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_rdp_complex_path() {
        // More complex path simulating GPS track
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.1),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.1),
            Point2::new(4.0, 0.0),
            Point2::new(5.0, 3.0),  // Significant turn
            Point2::new(6.0, 3.1),
            Point2::new(7.0, 3.0),
            Point2::new(8.0, 0.0),  // Return
            Point2::new(9.0, 0.1),
            Point2::new(10.0, 0.0),
        ];

        let result = rdp(&points, 0.5);

        // Should significantly reduce point count
        assert!(result.len() < points.len());

        // First and last should be preserved
        assert_relative_eq!(result.first().unwrap().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.last().unwrap().x, 10.0, epsilon = 1e-10);

        // The significant turn should be preserved
        assert!(result.iter().any(|p| (p.y - 3.0).abs() < 0.5));
    }

    #[test]
    fn test_rdp_f32() {
        // Verify it works with f32
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let result = rdp(&points, 0.1);
        assert_eq!(result.len(), 2);
    }
}
