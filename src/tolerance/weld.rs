//! Vertex welding and point merging operations.
//!
//! These functions merge nearby points within a specified tolerance, useful for:
//! - Cleaning up imported CAD/GIS data
//! - Removing duplicate vertices
//! - Simplifying point clouds
//! - Preparing geometry for exact algorithms
//!
//! # Example
//!
//! ```
//! use approxum::tolerance::weld_vertices;
//! use approxum::Point2;
//!
//! let points = vec![
//!     Point2::new(0.0_f64, 0.0),
//!     Point2::new(0.001, 0.001),  // Very close to first point
//!     Point2::new(1.0, 0.0),
//!     Point2::new(1.002, 0.0),    // Very close to third point
//! ];
//!
//! let welded = weld_vertices(&points, 0.01);
//! assert_eq!(welded.len(), 2);  // Two clusters merged
//! ```

use crate::primitives::Point2;
use num_traits::Float;

/// Merges points that are within `epsilon` distance of each other.
///
/// Uses a simple O(n²) algorithm that iterates through points and merges
/// clusters. Each cluster is represented by its centroid.
///
/// # Arguments
///
/// * `points` - Input points to weld
/// * `epsilon` - Maximum distance for points to be considered the same
///
/// # Returns
///
/// A new vector with merged points. Each output point is the centroid of
/// a cluster of nearby input points.
///
/// # Complexity
///
/// O(n²) time, O(n) space.
///
/// # Example
///
/// ```
/// use approxum::tolerance::weld_vertices;
/// use approxum::Point2;
///
/// let points = vec![
///     Point2::new(0.0_f64, 0.0),
///     Point2::new(0.05, 0.05),
///     Point2::new(1.0, 1.0),
/// ];
///
/// let welded = weld_vertices(&points, 0.1);
/// assert_eq!(welded.len(), 2);  // First two merged, third separate
/// ```
pub fn weld_vertices<F: Float>(points: &[Point2<F>], epsilon: F) -> Vec<Point2<F>> {
    if points.is_empty() {
        return Vec::new();
    }

    let eps_sq = epsilon * epsilon;
    let mut used = vec![false; points.len()];
    let mut result = Vec::new();

    for i in 0..points.len() {
        if used[i] {
            continue;
        }

        // Start a new cluster with this point
        let mut cluster_indices = vec![i];
        used[i] = true;

        // Find all points within epsilon of ANY point in the cluster
        // We use iterative merging to handle transitive closeness
        let mut changed = true;
        while changed {
            changed = false;
            for j in 0..points.len() {
                if used[j] {
                    continue;
                }

                // Check distance to any point in the cluster
                let mut close_to_cluster = false;
                for &k in &cluster_indices {
                    let dx = points[j].x - points[k].x;
                    let dy = points[j].y - points[k].y;
                    if dx * dx + dy * dy <= eps_sq {
                        close_to_cluster = true;
                        break;
                    }
                }

                if close_to_cluster {
                    cluster_indices.push(j);
                    used[j] = true;
                    changed = true;
                }
            }
        }

        // Compute centroid of cluster
        let mut sum_x = F::zero();
        let mut sum_y = F::zero();
        for &k in &cluster_indices {
            sum_x = sum_x + points[k].x;
            sum_y = sum_y + points[k].y;
        }
        let count = F::from(cluster_indices.len()).unwrap();
        result.push(Point2::new(sum_x / count, sum_y / count));
    }

    result
}

/// Merges points and returns both the welded points and an index mapping.
///
/// The index mapping shows which output point each input point was merged into.
///
/// # Arguments
///
/// * `points` - Input points to weld
/// * `epsilon` - Maximum distance for points to be considered the same
///
/// # Returns
///
/// A tuple of:
/// - `Vec<Point2<F>>`: The welded points (centroids of clusters)
/// - `Vec<usize>`: Index mapping where `mapping[i]` is the index in the output
///   that input point `i` was merged into
///
/// # Example
///
/// ```
/// use approxum::tolerance::weld_vertices_indexed;
/// use approxum::Point2;
///
/// let points = vec![
///     Point2::new(0.0_f64, 0.0),   // -> output 0
///     Point2::new(0.05, 0.05),     // -> output 0 (merged with first)
///     Point2::new(1.0, 1.0),       // -> output 1
/// ];
///
/// let (welded, mapping) = weld_vertices_indexed(&points, 0.1);
/// assert_eq!(welded.len(), 2);
/// assert_eq!(mapping, vec![0, 0, 1]);
/// ```
pub fn weld_vertices_indexed<F: Float>(
    points: &[Point2<F>],
    epsilon: F,
) -> (Vec<Point2<F>>, Vec<usize>) {
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let eps_sq = epsilon * epsilon;
    let mut mapping = vec![usize::MAX; points.len()];
    let mut result = Vec::new();

    for i in 0..points.len() {
        if mapping[i] != usize::MAX {
            continue;
        }

        let output_idx = result.len();

        // Start a new cluster
        let mut cluster_indices = vec![i];
        mapping[i] = output_idx;

        // Find all points within epsilon of ANY point in the cluster
        let mut changed = true;
        while changed {
            changed = false;
            for j in 0..points.len() {
                if mapping[j] != usize::MAX {
                    continue;
                }

                // Check distance to any point in the cluster
                let mut close_to_cluster = false;
                for &k in &cluster_indices {
                    let dx = points[j].x - points[k].x;
                    let dy = points[j].y - points[k].y;
                    if dx * dx + dy * dy <= eps_sq {
                        close_to_cluster = true;
                        break;
                    }
                }

                if close_to_cluster {
                    cluster_indices.push(j);
                    mapping[j] = output_idx;
                    changed = true;
                }
            }
        }

        // Compute centroid
        let mut sum_x = F::zero();
        let mut sum_y = F::zero();
        for &k in &cluster_indices {
            sum_x = sum_x + points[k].x;
            sum_y = sum_y + points[k].y;
        }
        let count = F::from(cluster_indices.len()).unwrap();
        result.push(Point2::new(sum_x / count, sum_y / count));
    }

    (result, mapping)
}

/// Merges points keeping the first point in each cluster (no averaging).
///
/// Unlike `weld_vertices` which computes centroids, this function keeps
/// the original coordinates of the first point encountered in each cluster.
/// Useful when you need to preserve exact coordinates.
///
/// # Arguments
///
/// * `points` - Input points to weld
/// * `epsilon` - Maximum distance for points to be considered the same
///
/// # Returns
///
/// A new vector with one point per cluster (the first point found).
pub fn weld_vertices_keep_first<F: Float>(points: &[Point2<F>], epsilon: F) -> Vec<Point2<F>> {
    if points.is_empty() {
        return Vec::new();
    }

    let eps_sq = epsilon * epsilon;
    let mut used = vec![false; points.len()];
    let mut result = Vec::new();

    for i in 0..points.len() {
        if used[i] {
            continue;
        }

        // Keep this point as the cluster representative
        result.push(points[i]);
        used[i] = true;

        // Mark all nearby points as used
        for j in (i + 1)..points.len() {
            if used[j] {
                continue;
            }

            let dx = points[j].x - points[i].x;
            let dy = points[j].y - points[i].y;

            if dx * dx + dy * dy <= eps_sq {
                used[j] = true;
            }
        }
    }

    result
}

/// Removes duplicate points from a polyline while preserving order.
///
/// Consecutive points within epsilon distance are merged. This is useful
/// for cleaning up polylines with redundant vertices.
///
/// # Arguments
///
/// * `points` - Input polyline vertices
/// * `epsilon` - Maximum distance for points to be considered duplicates
///
/// # Returns
///
/// A new polyline with consecutive duplicates removed.
///
/// # Example
///
/// ```
/// use approxum::tolerance::remove_duplicate_vertices;
/// use approxum::Point2;
///
/// let points = vec![
///     Point2::new(0.0_f64, 0.0),
///     Point2::new(0.001, 0.0),  // Duplicate of previous
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 0.0),    // Exact duplicate
///     Point2::new(2.0, 0.0),
/// ];
///
/// let cleaned = remove_duplicate_vertices(&points, 0.01);
/// assert_eq!(cleaned.len(), 3);
/// ```
pub fn remove_duplicate_vertices<F: Float>(points: &[Point2<F>], epsilon: F) -> Vec<Point2<F>> {
    if points.is_empty() {
        return Vec::new();
    }

    let eps_sq = epsilon * epsilon;
    let mut result = Vec::with_capacity(points.len());
    result.push(points[0]);

    for p in points.iter().skip(1) {
        let last = result.last().unwrap();
        let dx = p.x - last.x;
        let dy = p.y - last.y;

        if dx * dx + dy * dy > eps_sq {
            result.push(*p);
        }
    }

    result
}

/// Snaps points to a regular grid.
///
/// Each coordinate is rounded to the nearest multiple of `grid_size`.
/// Points that snap to the same grid location are not automatically merged;
/// use `weld_vertices` afterward if needed.
///
/// # Arguments
///
/// * `points` - Input points
/// * `grid_size` - Size of grid cells
///
/// # Returns
///
/// Points snapped to the grid.
///
/// # Example
///
/// ```
/// use approxum::tolerance::snap_to_grid;
/// use approxum::Point2;
///
/// let points = vec![
///     Point2::new(0.3_f64, 0.7),
///     Point2::new(1.2, 1.8),
/// ];
///
/// let snapped = snap_to_grid(&points, 0.5);
/// assert_eq!(snapped[0], Point2::new(0.5, 0.5));
/// assert_eq!(snapped[1], Point2::new(1.0, 2.0));
/// ```
pub fn snap_to_grid<F: Float>(points: &[Point2<F>], grid_size: F) -> Vec<Point2<F>> {
    points
        .iter()
        .map(|p| {
            let x = (p.x / grid_size).round() * grid_size;
            let y = (p.y / grid_size).round() * grid_size;
            Point2::new(x, y)
        })
        .collect()
}

/// Snaps points to grid and then welds duplicates.
///
/// Combines `snap_to_grid` and `weld_vertices` for convenience.
/// Uses a small epsilon relative to grid size to handle floating-point errors.
///
/// # Arguments
///
/// * `points` - Input points
/// * `grid_size` - Size of grid cells
///
/// # Returns
///
/// Unique grid-snapped points.
pub fn snap_and_weld<F: Float>(points: &[Point2<F>], grid_size: F) -> Vec<Point2<F>> {
    let snapped = snap_to_grid(points, grid_size);
    let epsilon = grid_size * F::from(0.001).unwrap();
    weld_vertices(&snapped, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_weld_vertices_basic() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.05, 0.05),
            Point2::new(1.0, 1.0),
        ];

        let welded = weld_vertices(&points, 0.1);

        assert_eq!(welded.len(), 2);
        // First cluster centroid
        assert!(approx_eq(welded[0].x, 0.025, 0.01));
        assert!(approx_eq(welded[0].y, 0.025, 0.01));
        // Second point unchanged
        assert!(approx_eq(welded[1].x, 1.0, 0.01));
        assert!(approx_eq(welded[1].y, 1.0, 0.01));
    }

    #[test]
    fn test_weld_vertices_no_merging() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];

        let welded = weld_vertices(&points, 0.1);
        assert_eq!(welded.len(), 3);
    }

    #[test]
    fn test_weld_vertices_all_same() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.01, 0.01),
            Point2::new(0.02, 0.0),
        ];

        let welded = weld_vertices(&points, 0.1);
        assert_eq!(welded.len(), 1);
    }

    #[test]
    fn test_weld_vertices_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let welded = weld_vertices(&points, 0.1);
        assert!(welded.is_empty());
    }

    #[test]
    fn test_weld_vertices_single() {
        let points = vec![Point2::new(1.0, 2.0)];
        let welded = weld_vertices(&points, 0.1);
        assert_eq!(welded.len(), 1);
        assert_eq!(welded[0], points[0]);
    }

    #[test]
    fn test_weld_vertices_indexed() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.05, 0.05),
            Point2::new(1.0, 1.0),
            Point2::new(1.02, 1.02),
        ];

        let (welded, mapping) = weld_vertices_indexed(&points, 0.1);

        assert_eq!(welded.len(), 2);
        assert_eq!(mapping.len(), 4);
        assert_eq!(mapping[0], 0); // First point -> cluster 0
        assert_eq!(mapping[1], 0); // Second point -> cluster 0
        assert_eq!(mapping[2], 1); // Third point -> cluster 1
        assert_eq!(mapping[3], 1); // Fourth point -> cluster 1
    }

    #[test]
    fn test_weld_vertices_keep_first() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.05, 0.05),
            Point2::new(1.0, 1.0),
        ];

        let welded = weld_vertices_keep_first(&points, 0.1);

        assert_eq!(welded.len(), 2);
        // First point is kept exactly
        assert_eq!(welded[0], points[0]);
        // Third point is kept exactly
        assert_eq!(welded[1], points[2]);
    }

    #[test]
    fn test_remove_duplicate_vertices() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.001, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];

        let cleaned = remove_duplicate_vertices(&points, 0.01);

        assert_eq!(cleaned.len(), 3);
        assert_eq!(cleaned[0], points[0]);
        assert_eq!(cleaned[1], points[2]);
        assert_eq!(cleaned[2], points[4]);
    }

    #[test]
    fn test_remove_duplicate_vertices_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let cleaned = remove_duplicate_vertices(&points, 0.01);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn test_remove_duplicate_vertices_preserves_non_consecutive() {
        // Points that are close but not consecutive should be kept
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 0.0), // Same as first but not consecutive
        ];

        let cleaned = remove_duplicate_vertices(&points, 0.01);
        assert_eq!(cleaned.len(), 3);
    }

    #[test]
    fn test_snap_to_grid() {
        let points = vec![
            Point2::new(0.3, 0.7),
            Point2::new(1.2, 1.8),
            Point2::new(-0.3, -0.7),
        ];

        let snapped = snap_to_grid(&points, 0.5);

        assert!(approx_eq(snapped[0].x, 0.5, 1e-10));
        assert!(approx_eq(snapped[0].y, 0.5, 1e-10));
        assert!(approx_eq(snapped[1].x, 1.0, 1e-10));
        assert!(approx_eq(snapped[1].y, 2.0, 1e-10));
        assert!(approx_eq(snapped[2].x, -0.5, 1e-10));
        assert!(approx_eq(snapped[2].y, -0.5, 1e-10));
    }

    #[test]
    fn test_snap_and_weld() {
        let points = vec![
            Point2::new(0.1, 0.1),
            Point2::new(0.2, 0.2),  // Both snap to (0, 0)
            Point2::new(1.0, 1.0),
        ];

        let result = snap_and_weld(&points, 0.5);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_weld_transitive() {
        // Points form a chain where each is close to the next
        // but endpoints are far apart
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.05, 0.0),
            Point2::new(0.10, 0.0),
            Point2::new(0.15, 0.0),
        ];

        // With epsilon 0.06, each point is within range of neighbors
        // Should merge into one cluster via transitive closure
        let welded = weld_vertices(&points, 0.06);

        // All points form one cluster due to iterative merging
        assert_eq!(welded.len(), 1);
    }

    #[test]
    fn test_f32() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.05, 0.05),
            Point2::new(1.0, 1.0),
        ];

        let welded = weld_vertices(&points, 0.1);
        assert_eq!(welded.len(), 2);
    }

    #[test]
    fn test_weld_large_cluster() {
        // Create a grid of nearby points
        let mut points = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                points.push(Point2::new(i as f64 * 0.01, j as f64 * 0.01));
            }
        }

        // Should all merge into one cluster
        let welded = weld_vertices(&points, 0.2);
        assert_eq!(welded.len(), 1);
    }
}
