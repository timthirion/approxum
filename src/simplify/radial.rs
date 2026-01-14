//! Radial distance simplification.
//!
//! A fast O(n) simplification algorithm that removes points within a specified
//! distance of the last kept point. Useful for filtering GPS jitter and
//! reducing point density in streaming data.

use crate::primitives::Point2;
use num_traits::Float;

/// Simplifies a polyline by removing points within a minimum distance.
///
/// This is a simple O(n) algorithm that walks through the points and keeps
/// only those that are at least `min_distance` away from the last kept point.
/// The first and last points are always preserved.
///
/// # Arguments
///
/// * `points` - The input polyline
/// * `min_distance` - Minimum distance between consecutive kept points
///
/// # Returns
///
/// A new vector containing the simplified polyline.
///
/// # Complexity
///
/// O(n) time, O(k) space where k is the number of output points.
///
/// # Example
///
/// ```
/// use approxum::simplify::radial;
/// use approxum::Point2;
///
/// // Points with some close together
/// let points = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(0.1, 0.0),  // Too close, will be removed
///     Point2::new(0.2, 0.0),  // Too close, will be removed
///     Point2::new(1.0, 0.0),  // Far enough, kept
///     Point2::new(1.05, 0.0), // Too close, will be removed
///     Point2::new(2.0, 0.0),  // Far enough, kept
/// ];
///
/// let simplified = radial(&points, 0.5);
/// assert_eq!(simplified.len(), 3); // First, 1.0, and last
/// ```
pub fn radial<F: Float>(points: &[Point2<F>], min_distance: F) -> Vec<Point2<F>> {
    let indices = radial_indices(points, min_distance);
    indices.iter().map(|&i| points[i]).collect()
}

/// Returns indices of points to keep after radial simplification.
///
/// This is useful when you need to preserve the relationship to the original
/// data or when working with associated per-point attributes.
pub fn radial_indices<F: Float>(points: &[Point2<F>], min_distance: F) -> Vec<usize> {
    if points.len() <= 2 {
        return (0..points.len()).collect();
    }

    let min_dist_sq = min_distance * min_distance;
    let mut indices = Vec::with_capacity(points.len() / 2); // Estimate

    // Always keep first point
    indices.push(0);
    let mut last_kept = points[0];

    // Check each point against the last kept point
    for (i, &point) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        if point.distance_squared(last_kept) >= min_dist_sq {
            indices.push(i);
            last_kept = point;
        }
    }

    // Always keep last point
    let last_idx = points.len() - 1;
    if indices.last() != Some(&last_idx) {
        indices.push(last_idx);
    }

    indices
}

/// Simplifies a polyline to approximately a target number of points.
///
/// Uses binary search to find the appropriate distance threshold that results
/// in approximately `target_count` points.
///
/// # Arguments
///
/// * `points` - The input polyline
/// * `target_count` - Desired number of output points (minimum 2)
///
/// # Returns
///
/// A new vector containing the simplified polyline. The actual count may
/// differ slightly from the target.
///
/// # Complexity
///
/// O(n log n) time due to binary search over distance thresholds.
pub fn radial_by_count<F: Float>(points: &[Point2<F>], target_count: usize) -> Vec<Point2<F>> {
    let indices = radial_indices_by_count(points, target_count);
    indices.iter().map(|&i| points[i]).collect()
}

/// Returns indices of points after radial simplification to a target count.
pub fn radial_indices_by_count<F: Float>(points: &[Point2<F>], target_count: usize) -> Vec<usize> {
    let n = points.len();

    if n <= 2 || target_count >= n {
        return (0..n).collect();
    }

    let target_count = target_count.max(2);

    // Find the bounding box to estimate distance range
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

    let diagonal = ((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y)).sqrt();

    // Binary search for the right distance threshold
    let mut lo = F::zero();
    let mut hi = diagonal;

    for _ in 0..50 {
        // Max iterations
        let mid = (lo + hi) / (F::one() + F::one());
        let indices = radial_indices(points, mid);
        let count = indices.len();

        if count == target_count {
            return indices;
        }

        if count > target_count {
            lo = mid;
        } else {
            hi = mid;
        }

        // Check for convergence
        if (hi - lo) < diagonal * F::from(1e-10).unwrap() {
            break;
        }
    }

    // Return the closest result
    let lo_indices = radial_indices(points, lo);
    let hi_indices = radial_indices(points, hi);

    let lo_diff = (lo_indices.len() as i64 - target_count as i64).abs();
    let hi_diff = (hi_indices.len() as i64 - target_count as i64).abs();

    if lo_diff <= hi_diff {
        lo_indices
    } else {
        hi_indices
    }
}

/// Streaming radial simplification filter.
///
/// This struct maintains state for incremental simplification, useful for
/// processing streaming point data (e.g., real-time GPS).
///
/// # Example
///
/// ```
/// use approxum::simplify::RadialFilter;
/// use approxum::Point2;
///
/// let mut filter = RadialFilter::new(1.0);
///
/// // Process points one at a time
/// assert!(filter.push(Point2::new(0.0, 0.0)).is_some());  // First point, kept
/// assert!(filter.push(Point2::new(0.5, 0.0)).is_none());  // Too close
/// assert!(filter.push(Point2::new(1.5, 0.0)).is_some());  // Far enough
/// ```
pub struct RadialFilter<F> {
    min_distance_sq: F,
    last_point: Option<Point2<F>>,
}

impl<F: Float> RadialFilter<F> {
    /// Creates a new radial filter with the specified minimum distance.
    pub fn new(min_distance: F) -> Self {
        Self {
            min_distance_sq: min_distance * min_distance,
            last_point: None,
        }
    }

    /// Processes a point, returning it if it should be kept.
    ///
    /// Returns `Some(point)` if the point is far enough from the last kept point,
    /// or `None` if it should be filtered out.
    pub fn push(&mut self, point: Point2<F>) -> Option<Point2<F>> {
        match self.last_point {
            None => {
                self.last_point = Some(point);
                Some(point)
            }
            Some(last) => {
                if point.distance_squared(last) >= self.min_distance_sq {
                    self.last_point = Some(point);
                    Some(point)
                } else {
                    None
                }
            }
        }
    }

    /// Resets the filter state.
    pub fn reset(&mut self) {
        self.last_point = None;
    }

    /// Returns the last kept point, if any.
    pub fn last_point(&self) -> Option<Point2<F>> {
        self.last_point
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radial_basic() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.1, 0.0),
            Point2::new(0.2, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.05, 0.0),
            Point2::new(2.0, 0.0),
        ];

        let simplified = radial(&points, 0.5);

        // Should keep: first (0,0), (1,0), and last (2,0)
        assert_eq!(simplified.len(), 3);
        assert_eq!(simplified[0], points[0]);
        assert_eq!(simplified[1], points[3]); // (1, 0)
        assert_eq!(simplified[2], points[5]); // (2, 0)
    }

    #[test]
    fn test_radial_preserves_endpoints() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.01, 0.0),
            Point2::new(0.02, 0.0),
        ];

        let simplified = radial(&points, 1.0);

        // Even though all points are close, first and last are kept
        assert_eq!(simplified.len(), 2);
        assert_eq!(simplified[0], points[0]);
        assert_eq!(simplified[1], points[2]);
    }

    #[test]
    fn test_radial_empty_and_small() {
        let empty: Vec<Point2<f64>> = vec![];
        assert_eq!(radial(&empty, 1.0).len(), 0);

        let single = vec![Point2::new(0.0, 0.0)];
        assert_eq!(radial(&single, 1.0).len(), 1);

        let two = vec![Point2::new(0.0, 0.0), Point2::new(0.1, 0.0)];
        assert_eq!(radial(&two, 1.0).len(), 2);
    }

    #[test]
    fn test_radial_indices() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.1, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];

        let indices = radial_indices(&points, 0.5);

        assert_eq!(indices, vec![0, 2, 3]);
    }

    #[test]
    fn test_radial_by_count() {
        let points: Vec<Point2<f64>> = (0..100).map(|i| Point2::new(i as f64, 0.0)).collect();

        let simplified = radial_by_count(&points, 10);

        // Should be close to 10 points
        assert!(simplified.len() >= 8 && simplified.len() <= 12);

        // First and last should be preserved
        assert_eq!(simplified[0], points[0]);
        assert_eq!(simplified[simplified.len() - 1], points[99]);
    }

    #[test]
    fn test_radial_filter_streaming() {
        let mut filter: RadialFilter<f64> = RadialFilter::new(1.0);

        // First point always kept
        assert!(filter.push(Point2::new(0.0, 0.0)).is_some());

        // Points too close are filtered
        assert!(filter.push(Point2::new(0.5, 0.0)).is_none());
        assert!(filter.push(Point2::new(0.9, 0.0)).is_none());

        // Point far enough is kept
        assert!(filter.push(Point2::new(1.5, 0.0)).is_some());

        // Next point relative to new last point
        assert!(filter.push(Point2::new(2.0, 0.0)).is_none());
        assert!(filter.push(Point2::new(3.0, 0.0)).is_some());
    }

    #[test]
    fn test_radial_filter_reset() {
        let mut filter: RadialFilter<f64> = RadialFilter::new(1.0);

        filter.push(Point2::new(0.0, 0.0));
        filter.push(Point2::new(5.0, 0.0));

        assert_eq!(filter.last_point(), Some(Point2::new(5.0, 0.0)));

        filter.reset();
        assert_eq!(filter.last_point(), None);

        // After reset, any point is accepted
        assert!(filter.push(Point2::new(0.0, 0.0)).is_some());
    }

    #[test]
    fn test_radial_diagonal_movement() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.5, 0.5), // Distance ~0.707
            Point2::new(1.0, 1.0), // Distance ~0.707 from previous
            Point2::new(2.0, 2.0), // Distance ~1.414 from previous
        ];

        let simplified = radial(&points, 1.0);

        // (0,0), (1,1) should be filtered as ~0.707 < 1.0
        // But (2,2) is ~1.414 from (0,0) so kept
        assert!(simplified.len() >= 2);
        assert_eq!(simplified[0], points[0]);
        assert_eq!(simplified[simplified.len() - 1], points[3]);
    }

    #[test]
    fn test_radial_f32() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.1, 0.0),
            Point2::new(1.0, 0.0),
        ];

        let simplified = radial(&points, 0.5);
        assert_eq!(simplified.len(), 2);
    }
}
