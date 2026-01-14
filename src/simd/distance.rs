//! SIMD-accelerated distance computations.

#![allow(clippy::needless_range_loop)]

use wide::{f32x4, CmpLe};

use crate::primitives::{Point2, Segment2};

use super::point::Point2x4;

/// Computes squared distances from multiple points to a single target point.
///
/// Processes points in batches of 4 for SIMD efficiency.
///
/// # Arguments
///
/// * `points` - Slice of points to compute distances from
/// * `target` - The target point to compute distances to
///
/// # Returns
///
/// Vector of squared distances, one per input point.
pub fn distances_squared_to_point(points: &[Point2<f32>], target: Point2<f32>) -> Vec<f32> {
    let n = points.len();
    let mut result = Vec::with_capacity(n);

    // Process in batches of 4
    let chunks = n / 4;
    for i in 0..chunks {
        let batch = Point2x4::from_slice(&points[i * 4..]);
        let distances = batch.distance_squared_to(target);
        result.extend_from_slice(&distances.to_array());
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        let dx = points[i].x - target.x;
        let dy = points[i].y - target.y;
        result.push(dx * dx + dy * dy);
    }

    result
}

/// Computes distances from multiple points to a single target point.
///
/// Processes points in batches of 4 for SIMD efficiency.
pub fn distances_to_point(points: &[Point2<f32>], target: Point2<f32>) -> Vec<f32> {
    let n = points.len();
    let mut result = Vec::with_capacity(n);

    // Process in batches of 4
    let chunks = n / 4;
    for i in 0..chunks {
        let batch = Point2x4::from_slice(&points[i * 4..]);
        let distances = batch.distance_to(target);
        result.extend_from_slice(&distances.to_array());
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        let dx = points[i].x - target.x;
        let dy = points[i].y - target.y;
        result.push((dx * dx + dy * dy).sqrt());
    }

    result
}

/// Computes distances from multiple points to a single line segment.
///
/// Processes points in batches of 4 for SIMD efficiency.
pub fn distances_to_segment(points: &[Point2<f32>], segment: Segment2<f32>) -> Vec<f32> {
    let n = points.len();
    let mut result = Vec::with_capacity(n);

    // Precompute segment properties
    let v = segment.end - segment.start;
    let len_sq = v.x * v.x + v.y * v.y;

    if len_sq < 1e-10 {
        // Degenerate segment - just compute distance to start point
        return distances_to_point(points, segment.start);
    }

    let inv_len_sq = 1.0 / len_sq;

    // SIMD constants
    let start_x = f32x4::splat(segment.start.x);
    let start_y = f32x4::splat(segment.start.y);
    let vx = f32x4::splat(v.x);
    let vy = f32x4::splat(v.y);
    let inv_len_sq_simd = f32x4::splat(inv_len_sq);
    let zero = f32x4::splat(0.0);
    let one = f32x4::splat(1.0);

    // Process in batches of 4
    let chunks = n / 4;
    for i in 0..chunks {
        let batch = Point2x4::from_slice(&points[i * 4..]);

        // Vector from segment start to each point
        let wx = batch.x - start_x;
        let wy = batch.y - start_y;

        // Project onto segment (clamped to [0, 1])
        let t = (wx * vx + wy * vy) * inv_len_sq_simd;
        let t_clamped = t.max(zero).min(one);

        // Closest point on segment
        let closest_x = start_x + vx * t_clamped;
        let closest_y = start_y + vy * t_clamped;

        // Distance to closest point
        let dx = batch.x - closest_x;
        let dy = batch.y - closest_y;
        let dist = (dx * dx + dy * dy).sqrt();

        result.extend_from_slice(&dist.to_array());
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        result.push(segment.distance_to_point(points[i]));
    }

    result
}

/// Finds the index of the point closest to the target.
///
/// Returns `None` if the slice is empty.
pub fn nearest_point_index(points: &[Point2<f32>], target: Point2<f32>) -> Option<usize> {
    if points.is_empty() {
        return None;
    }

    let n = points.len();
    let mut best_idx = 0;
    let mut best_dist_sq = f32::INFINITY;

    // Process in batches of 4
    let chunks = n / 4;
    for i in 0..chunks {
        let batch = Point2x4::from_slice(&points[i * 4..]);
        let distances = batch.distance_squared_to(target).to_array();

        for (j, &dist_sq) in distances.iter().enumerate() {
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_idx = i * 4 + j;
            }
        }
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        let dx = points[i].x - target.x;
        let dy = points[i].y - target.y;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_idx = i;
        }
    }

    Some(best_idx)
}

/// Finds all points within a given distance of the target.
///
/// Returns indices of points within the radius.
pub fn points_within_radius(
    points: &[Point2<f32>],
    target: Point2<f32>,
    radius: f32,
) -> Vec<usize> {
    let n = points.len();
    let mut result = Vec::new();
    let radius_sq = radius * radius;
    let radius_sq_simd = f32x4::splat(radius_sq);

    // Process in batches of 4
    let chunks = n / 4;
    for i in 0..chunks {
        let batch = Point2x4::from_slice(&points[i * 4..]);
        let distances_sq = batch.distance_squared_to(target);

        // Compare with radius
        let mask = distances_sq.cmp_le(radius_sq_simd);
        let mask_bits = mask.move_mask();

        // Extract matching indices
        if mask_bits != 0 {
            for j in 0..4 {
                if (mask_bits & (1 << j)) != 0 {
                    result.push(i * 4 + j);
                }
            }
        }
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        let dx = points[i].x - target.x;
        let dy = points[i].y - target.y;
        if dx * dx + dy * dy <= radius_sq {
            result.push(i);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distances_to_point() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(0.0, 4.0),
            Point2::new(3.0, 4.0),
            Point2::new(6.0, 8.0),
        ];
        let target = Point2::new(0.0, 0.0);

        let distances = distances_to_point(&points, target);

        assert_eq!(distances.len(), 5);
        assert!((distances[0] - 0.0).abs() < 1e-6);
        assert!((distances[1] - 3.0).abs() < 1e-6);
        assert!((distances[2] - 4.0).abs() < 1e-6);
        assert!((distances[3] - 5.0).abs() < 1e-6);
        assert!((distances[4] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_distances_to_segment() {
        let segment = Segment2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0));

        let points = vec![
            Point2::new(5.0, 0.0),  // On segment
            Point2::new(5.0, 3.0),  // Above middle
            Point2::new(-1.0, 0.0), // Beyond start
            Point2::new(11.0, 0.0), // Beyond end
            Point2::new(0.0, 4.0),  // Above start
        ];

        let distances = distances_to_segment(&points, segment);

        assert_eq!(distances.len(), 5);
        assert!(distances[0] < 1e-6); // On segment
        assert!((distances[1] - 3.0).abs() < 1e-6); // 3 units above
        assert!((distances[2] - 1.0).abs() < 1e-6); // 1 unit beyond start
        assert!((distances[3] - 1.0).abs() < 1e-6); // 1 unit beyond end
        assert!((distances[4] - 4.0).abs() < 1e-6); // 4 units above start
    }

    #[test]
    fn test_nearest_point_index() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(5.0, 5.0),
            Point2::new(2.0, 2.0),
            Point2::new(7.0, 7.0),
        ];

        let target = Point2::new(2.5, 2.5);
        let idx = nearest_point_index(&points, target);

        assert_eq!(idx, Some(3)); // Point at (2, 2) is closest
    }

    #[test]
    fn test_points_within_radius() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(10.0, 0.0),
        ];

        let target = Point2::new(0.0, 0.0);
        let indices = points_within_radius(&points, target, 2.5);

        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_empty_points() {
        let points: Vec<Point2<f32>> = vec![];
        let target = Point2::new(0.0, 0.0);

        assert!(nearest_point_index(&points, target).is_none());
        assert!(distances_to_point(&points, target).is_empty());
    }
}
