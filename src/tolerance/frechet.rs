//! Fréchet distance computation.
//!
//! The Fréchet distance measures similarity between curves while respecting
//! the ordering of points. It's often described as the "dog walking" distance:
//! imagine a person walking along one curve and a dog along another, connected
//! by a leash. The Fréchet distance is the minimum leash length needed for both
//! to walk from start to end (varying speed, but never backtracking).
//!
//! # Discrete vs Continuous
//!
//! - **Discrete Fréchet**: Only considers vertices, O(nm) time
//! - **Continuous Fréchet**: Considers all points along segments, more complex
//!
//! This module provides both the exact discrete version and an approximate
//! continuous version using sampling.
//!
//! # Comparison with Hausdorff
//!
//! - Hausdorff ignores point ordering (good for shape comparison)
//! - Fréchet respects ordering (better for path/trajectory comparison)
//!
//! # Example
//!
//! ```
//! use approxum::tolerance::discrete_frechet_distance;
//! use approxum::Point2;
//!
//! let path1 = vec![
//!     Point2::new(0.0_f64, 0.0),
//!     Point2::new(1.0, 0.0),
//!     Point2::new(2.0, 0.0),
//! ];
//! let path2 = vec![
//!     Point2::new(0.0_f64, 1.0),
//!     Point2::new(1.0, 1.0),
//!     Point2::new(2.0, 1.0),
//! ];
//!
//! // Parallel paths 1 unit apart
//! let dist = discrete_frechet_distance(&path1, &path2);
//! assert!((dist - 1.0).abs() < 1e-10);
//! ```

use crate::primitives::{Point2, Segment2};
use num_traits::Float;

/// Computes the discrete Fréchet distance between two polylines.
///
/// This considers only the vertices of the polylines. The algorithm uses
/// dynamic programming with O(nm) time and O(nm) space complexity.
///
/// # Arguments
///
/// * `p` - First polyline (sequence of points)
/// * `q` - Second polyline (sequence of points)
///
/// # Returns
///
/// The discrete Fréchet distance, or 0 if either polyline is empty.
///
/// # Complexity
///
/// O(nm) time and space, where n = |p| and m = |q|.
///
/// # Example
///
/// ```
/// use approxum::tolerance::discrete_frechet_distance;
/// use approxum::Point2;
///
/// // Two paths that diverge and converge
/// let p = vec![
///     Point2::new(0.0_f64, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(2.0, 0.0),
/// ];
/// let q = vec![
///     Point2::new(0.0_f64, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(2.0, 0.0),
/// ];
///
/// let dist = discrete_frechet_distance(&p, &q);
/// // Maximum deviation is 1.0 at the middle point
/// assert!((dist - 1.0).abs() < 0.01);
/// ```
pub fn discrete_frechet_distance<F: Float>(p: &[Point2<F>], q: &[Point2<F>]) -> F {
    if p.is_empty() || q.is_empty() {
        return F::zero();
    }

    let n = p.len();
    let m = q.len();

    // dp[i][j] = Fréchet distance for p[0..=i] and q[0..=j]
    let mut dp = vec![vec![F::neg_infinity(); m]; n];

    // Base case: starting points
    dp[0][0] = p[0].distance(q[0]);

    // Fill first row: can only come from the left
    for j in 1..m {
        dp[0][j] = dp[0][j - 1].max(p[0].distance(q[j]));
    }

    // Fill first column: can only come from above
    for i in 1..n {
        dp[i][0] = dp[i - 1][0].max(p[i].distance(q[0]));
    }

    // Fill the rest of the table
    for i in 1..n {
        for j in 1..m {
            let dist = p[i].distance(q[j]);
            let prev_min = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = dist.max(prev_min);
        }
    }

    dp[n - 1][m - 1]
}

/// Computes the discrete Fréchet distance with O(n) space optimization.
///
/// Uses only two rows of the DP table at a time, reducing space from O(nm) to O(m).
/// Useful for very long polylines where memory is a concern.
///
/// # Arguments
///
/// * `p` - First polyline
/// * `q` - Second polyline
///
/// # Returns
///
/// The discrete Fréchet distance.
pub fn discrete_frechet_distance_linear_space<F: Float>(p: &[Point2<F>], q: &[Point2<F>]) -> F {
    if p.is_empty() || q.is_empty() {
        return F::zero();
    }

    let n = p.len();
    let m = q.len();

    // Only keep two rows
    let mut prev = vec![F::neg_infinity(); m];
    let mut curr = vec![F::neg_infinity(); m];

    // Initialize first row
    prev[0] = p[0].distance(q[0]);
    for j in 1..m {
        prev[j] = prev[j - 1].max(p[0].distance(q[j]));
    }

    // Process remaining rows
    for p_point in p.iter().take(n).skip(1) {
        curr[0] = prev[0].max(p_point.distance(q[0]));

        for j in 1..m {
            let dist = p_point.distance(q[j]);
            let prev_min = prev[j].min(curr[j - 1]).min(prev[j - 1]);
            curr[j] = dist.max(prev_min);
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m - 1]
}

/// Computes an approximate continuous Fréchet distance using sampling.
///
/// This samples points along the polyline segments to approximate the
/// continuous Fréchet distance. Higher sample density gives more accuracy
/// but increases computation time.
///
/// # Arguments
///
/// * `p` - First polyline
/// * `q` - Second polyline
/// * `samples_per_segment` - Number of sample points per segment (minimum 1)
///
/// # Returns
///
/// An approximation of the continuous Fréchet distance.
///
/// # Complexity
///
/// O((n*s) * (m*s)) where s is samples_per_segment.
pub fn frechet_distance_approx<F: Float>(
    p: &[Point2<F>],
    q: &[Point2<F>],
    samples_per_segment: usize,
) -> F {
    if p.len() < 2 || q.len() < 2 {
        return discrete_frechet_distance(p, q);
    }

    let samples = samples_per_segment.max(1);

    // Sample points along both polylines
    let p_sampled = sample_polyline(p, samples);
    let q_sampled = sample_polyline(q, samples);

    discrete_frechet_distance(&p_sampled, &q_sampled)
}

/// Samples points along a polyline.
fn sample_polyline<F: Float>(polyline: &[Point2<F>], samples_per_segment: usize) -> Vec<Point2<F>> {
    if polyline.len() < 2 {
        return polyline.to_vec();
    }

    let mut result = Vec::with_capacity((polyline.len() - 1) * samples_per_segment + 1);

    for i in 0..polyline.len() - 1 {
        let seg = Segment2::new(polyline[i], polyline[i + 1]);

        for j in 0..samples_per_segment {
            let t = F::from(j).unwrap() / F::from(samples_per_segment).unwrap();
            result.push(seg.point_at(t));
        }
    }

    // Add the last point
    result.push(*polyline.last().unwrap());

    result
}

/// Decides if the Fréchet distance between two curves is at most `threshold`.
///
/// This is a decision version that can be faster than computing the exact
/// distance when you only need to know if curves are "similar enough".
///
/// Uses the free-space diagram approach with O(nm) complexity.
///
/// # Arguments
///
/// * `p` - First polyline
/// * `q` - Second polyline
/// * `threshold` - Maximum allowed Fréchet distance
///
/// # Returns
///
/// `true` if the Fréchet distance is at most `threshold`.
pub fn frechet_distance_at_most<F: Float>(p: &[Point2<F>], q: &[Point2<F>], threshold: F) -> bool {
    if p.is_empty() || q.is_empty() {
        return true;
    }

    // Check if starting points are within threshold
    if p[0].distance(q[0]) > threshold {
        return false;
    }

    // Check if ending points are within threshold
    if p.last().unwrap().distance(*q.last().unwrap()) > threshold {
        return false;
    }

    let n = p.len();
    let m = q.len();

    // Reachability table: can we reach (i,j) with leash length <= threshold?
    let mut reachable = vec![vec![false; m]; n];

    // Base case
    reachable[0][0] = true;

    // Fill the table
    for i in 0..n {
        for j in 0..m {
            if i == 0 && j == 0 {
                continue;
            }

            // Can only reach (i,j) if the distance is within threshold
            if p[i].distance(q[j]) > threshold {
                continue;
            }

            // Check if we can reach from any valid predecessor
            let from_left = j > 0 && reachable[i][j - 1];
            let from_above = i > 0 && reachable[i - 1][j];
            let from_diag = i > 0 && j > 0 && reachable[i - 1][j - 1];

            reachable[i][j] = from_left || from_above || from_diag;
        }
    }

    reachable[n - 1][m - 1]
}

/// Computes the Fréchet distance using binary search with the decision algorithm.
///
/// This can be faster in practice for large inputs because the decision
/// algorithm has better constants than the full DP.
///
/// # Arguments
///
/// * `p` - First polyline
/// * `q` - Second polyline
/// * `tolerance` - Precision of the result
///
/// # Returns
///
/// The Fréchet distance within the specified tolerance.
pub fn frechet_distance_binary_search<F: Float>(
    p: &[Point2<F>],
    q: &[Point2<F>],
    _tolerance: F,
) -> F {
    if p.is_empty() || q.is_empty() {
        return F::zero();
    }

    // Collect all pairwise distances as candidate thresholds
    let mut candidates: Vec<F> = Vec::with_capacity(p.len() * q.len());
    for pi in p {
        for qj in q {
            candidates.push(pi.distance(*qj));
        }
    }

    // Sort candidates
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup();

    // Binary search for the minimum threshold
    let mut lo = 0;
    let mut hi = candidates.len();

    while lo < hi {
        let mid = (lo + hi) / 2;
        if frechet_distance_at_most(p, q, candidates[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    if lo < candidates.len() {
        candidates[lo]
    } else {
        // Fallback: compute directly
        discrete_frechet_distance(p, q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_discrete_frechet_identical() {
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];

        let dist = discrete_frechet_distance(&p, &p);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_discrete_frechet_parallel_lines() {
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let q = vec![
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
        ];

        let dist = discrete_frechet_distance(&p, &q);
        assert!(approx_eq(dist, 1.0, 1e-10));
    }

    #[test]
    fn test_discrete_frechet_diverging_paths() {
        // Paths that start together, diverge, then converge
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0), // Goes up
            Point2::new(2.0, 0.0),
        ];
        let q = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0), // Stays low
            Point2::new(2.0, 0.0),
        ];

        let dist = discrete_frechet_distance(&p, &q);
        // Maximum distance is 2.0 at the middle points
        assert!(approx_eq(dist, 2.0, 1e-10));
    }

    #[test]
    fn test_discrete_frechet_different_lengths() {
        let p = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];
        let q = vec![
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
        ];

        let dist = discrete_frechet_distance(&p, &q);
        // The optimal coupling requires p[0] to match q[0] and q[1],
        // then p[1] to match q[2]. Distance from p[0] to q[1] is sqrt(2).
        let sqrt2 = std::f64::consts::SQRT_2;
        assert!(approx_eq(dist, sqrt2, 1e-10));
    }

    #[test]
    fn test_discrete_frechet_empty() {
        let p: Vec<Point2<f64>> = vec![];
        let q = vec![Point2::new(0.0, 0.0)];

        assert_eq!(discrete_frechet_distance(&p, &q), 0.0);
        assert_eq!(discrete_frechet_distance(&q, &p), 0.0);
    }

    #[test]
    fn test_discrete_frechet_single_points() {
        let p = vec![Point2::new(0.0, 0.0)];
        let q = vec![Point2::new(3.0, 4.0)];

        let dist = discrete_frechet_distance(&p, &q);
        assert!(approx_eq(dist, 5.0, 1e-10)); // 3-4-5 triangle
    }

    #[test]
    fn test_frechet_vs_hausdorff() {
        // Fréchet respects ordering, Hausdorff doesn't
        // These two paths have the same point sets but different orders
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        // Same points but middle is shifted
        let q = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0), // Different middle
            Point2::new(2.0, 0.0),
        ];

        let frechet = discrete_frechet_distance(&p, &q);
        // Fréchet distance is 1.0 (must match middle points)
        assert!(approx_eq(frechet, 1.0, 1e-10));
    }

    #[test]
    fn test_linear_space_matches_standard() {
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 1.0),
        ];
        let q = vec![
            Point2::new(0.0, 0.5),
            Point2::new(1.0, 0.5),
            Point2::new(2.0, 0.5),
        ];

        let dist1 = discrete_frechet_distance(&p, &q);
        let dist2 = discrete_frechet_distance_linear_space(&p, &q);

        assert!(approx_eq(dist1, dist2, 1e-10));
    }

    #[test]
    fn test_frechet_approx() {
        let p = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];
        let q = vec![Point2::new(0.0, 1.0), Point2::new(2.0, 1.0)];

        // Discrete only checks endpoints
        let discrete = discrete_frechet_distance(&p, &q);

        // Approximate with sampling should give similar result for straight lines
        let approx = frechet_distance_approx(&p, &q, 10);

        assert!(approx_eq(discrete, approx, 0.1));
    }

    #[test]
    fn test_frechet_at_most() {
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let q = vec![
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
        ];

        // Distance is exactly 1.0
        assert!(frechet_distance_at_most(&p, &q, 1.0));
        assert!(frechet_distance_at_most(&p, &q, 1.5));
        assert!(!frechet_distance_at_most(&p, &q, 0.9));
    }

    #[test]
    fn test_frechet_at_most_requires_ordering() {
        // These paths cross - Fréchet must respect ordering
        let p = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 2.0)];
        let q = vec![Point2::new(0.0, 2.0), Point2::new(2.0, 0.0)];

        // Starting points are 2.0 apart, ending points are 2.0 apart
        // Fréchet distance is 2.0
        let dist = discrete_frechet_distance(&p, &q);
        assert!(approx_eq(dist, 2.0, 1e-10));

        assert!(frechet_distance_at_most(&p, &q, 2.0));
        assert!(!frechet_distance_at_most(&p, &q, 1.9));
    }

    #[test]
    fn test_frechet_binary_search() {
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let q = vec![
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
        ];

        let dist = frechet_distance_binary_search(&p, &q, 0.01);
        assert!(approx_eq(dist, 1.0, 0.01));
    }

    #[test]
    fn test_f32() {
        let p: Vec<Point2<f32>> = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        let q: Vec<Point2<f32>> = vec![Point2::new(0.0, 1.0), Point2::new(1.0, 1.0)];

        let dist = discrete_frechet_distance(&p, &q);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sample_polyline() {
        let polyline = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];

        let sampled = sample_polyline(&polyline, 4);

        // Should have 4 samples on segment + 1 endpoint = 5 points
        assert_eq!(sampled.len(), 5);
        assert!(approx_eq(sampled[0].x, 0.0, 1e-10));
        assert!(approx_eq(sampled[1].x, 0.5, 1e-10));
        assert!(approx_eq(sampled[2].x, 1.0, 1e-10));
        assert!(approx_eq(sampled[3].x, 1.5, 1e-10));
        assert!(approx_eq(sampled[4].x, 2.0, 1e-10));
    }

    #[test]
    fn test_frechet_symmetric() {
        let p = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];
        let q = vec![Point2::new(0.0, 0.5), Point2::new(2.0, 0.5)];

        let dist_pq = discrete_frechet_distance(&p, &q);
        let dist_qp = discrete_frechet_distance(&q, &p);

        // Fréchet distance should be symmetric
        assert!(approx_eq(dist_pq, dist_qp, 1e-10));
    }
}
