//! Visvalingam-Whyatt polyline simplification.
//!
//! This algorithm iteratively removes the point that contributes the least
//! "effective area" (the area of the triangle formed with its neighbors).
//! Unlike RDP, it produces more aesthetically pleasing results and has
//! consistent O(n log n) time complexity.

use crate::primitives::Point2;
use num_traits::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Simplifies a polyline using the Visvalingam-Whyatt algorithm.
///
/// Removes points until no point has an effective area less than `min_area`.
///
/// # Arguments
///
/// * `points` - The input polyline as a slice of points
/// * `min_area` - Minimum effective area threshold. Points forming triangles
///   with area less than this will be removed.
///
/// # Returns
///
/// A new vector with the simplified polyline.
///
/// # Example
///
/// ```
/// use approxum::{Point2, simplify::visvalingam};
///
/// let points = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.1),
///     Point2::new(2.0, 0.0),
///     Point2::new(3.0, 2.0),
///     Point2::new(4.0, 0.0),
/// ];
///
/// let simplified = visvalingam(&points, 0.5);
/// assert!(simplified.len() <= points.len());
/// ```
pub fn visvalingam<F: Float>(points: &[Point2<F>], min_area: F) -> Vec<Point2<F>> {
    let indices = visvalingam_indices(points, min_area);
    indices.into_iter().map(|i| points[i]).collect()
}

/// Simplifies a polyline to a target number of points using Visvalingam-Whyatt.
///
/// # Arguments
///
/// * `points` - The input polyline
/// * `target_count` - Desired number of points in the output (minimum 2)
///
/// # Returns
///
/// A new vector with at most `target_count` points.
pub fn visvalingam_by_count<F: Float>(points: &[Point2<F>], target_count: usize) -> Vec<Point2<F>> {
    let indices = visvalingam_indices_by_count(points, target_count);
    indices.into_iter().map(|i| points[i]).collect()
}

/// Simplifies a polyline and returns indices of retained points.
///
/// # Arguments
///
/// * `points` - The input polyline
/// * `min_area` - Minimum effective area threshold
///
/// # Returns
///
/// Indices of retained points, in order.
pub fn visvalingam_indices<F: Float>(points: &[Point2<F>], min_area: F) -> Vec<usize> {
    visvalingam_impl(points, Some(min_area), None)
}

/// Simplifies to a target count and returns indices of retained points.
pub fn visvalingam_indices_by_count<F: Float>(
    points: &[Point2<F>],
    target_count: usize,
) -> Vec<usize> {
    visvalingam_impl(points, None, Some(target_count))
}

/// Entry for the priority queue.
struct AreaEntry<F> {
    index: usize,
    area: F,
    /// Generation counter to handle stale entries
    generation: usize,
}

impl<F: Float> PartialEq for AreaEntry<F> {
    fn eq(&self, other: &Self) -> bool {
        self.area == other.area
    }
}

impl<F: Float> Eq for AreaEntry<F> {}

impl<F: Float> PartialOrd for AreaEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Float> Ord for AreaEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .area
            .partial_cmp(&self.area)
            .unwrap_or(Ordering::Equal)
    }
}

/// Computes the area of a triangle formed by three points.
///
/// Returns the absolute value of the signed area.
fn triangle_area<F: Float>(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> F {
    let two = F::one() + F::one();
    ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs() / two
}

/// Core Visvalingam implementation.
fn visvalingam_impl<F: Float>(
    points: &[Point2<F>],
    min_area: Option<F>,
    target_count: Option<usize>,
) -> Vec<usize> {
    let n = points.len();

    if n < 3 {
        return (0..n).collect();
    }

    // Linked list structure: prev[i] and next[i] give neighbors
    // Using usize::MAX as null sentinel
    let mut prev: Vec<usize> = (0..n)
        .map(|i| if i == 0 { usize::MAX } else { i - 1 })
        .collect();
    let mut next: Vec<usize> = (0..n)
        .map(|i| if i == n - 1 { usize::MAX } else { i + 1 })
        .collect();

    // Track which points are still active
    let mut active = vec![true; n];

    // Generation counters to invalidate stale heap entries
    let mut generation = vec![0usize; n];

    // Current areas for each point
    let mut areas: Vec<F> = vec![F::infinity(); n];

    // Priority queue (min-heap via reversed Ord)
    let mut heap = BinaryHeap::new();

    // Initialize areas for interior points
    for i in 1..n - 1 {
        let area = triangle_area(points[prev[i]], points[i], points[next[i]]);
        areas[i] = area;
        heap.push(AreaEntry {
            index: i,
            area,
            generation: 0,
        });
    }

    // Count of remaining points
    let mut remaining = n;
    let target = target_count.unwrap_or(2).max(2);

    // Process until we hit our stopping condition
    while let Some(entry) = heap.pop() {
        // Skip stale entries
        if !active[entry.index] || generation[entry.index] != entry.generation {
            continue;
        }

        // Check stopping conditions
        if let Some(min) = min_area {
            if entry.area >= min {
                break; // All remaining points exceed threshold
            }
        }
        if remaining <= target {
            break;
        }

        // Remove this point
        let i = entry.index;
        active[i] = false;
        remaining -= 1;

        let p = prev[i];
        let nx = next[i];

        // Update linked list
        if p != usize::MAX {
            next[p] = nx;
        }
        if nx != usize::MAX {
            prev[nx] = p;
        }

        // Update areas for affected neighbors
        // Previous point (if it's an interior point)
        if p != usize::MAX && prev[p] != usize::MAX {
            let new_area = triangle_area(points[prev[p]], points[p], points[next[p]]);
            // Enforce monotonicity: area can't decrease below the just-removed point's area
            areas[p] = new_area.max(entry.area);
            generation[p] += 1;
            heap.push(AreaEntry {
                index: p,
                area: areas[p],
                generation: generation[p],
            });
        }

        // Next point (if it's an interior point)
        if nx != usize::MAX && next[nx] != usize::MAX {
            let new_area = triangle_area(points[prev[nx]], points[nx], points[next[nx]]);
            areas[nx] = new_area.max(entry.area);
            generation[nx] += 1;
            heap.push(AreaEntry {
                index: nx,
                area: areas[nx],
                generation: generation[nx],
            });
        }
    }

    // Collect remaining indices
    active
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| if a { Some(i) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_triangle_area() {
        let a: Point2<f64> = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.0, 1.0);
        assert_relative_eq!(triangle_area(a, b, c), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_visvalingam_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let result = visvalingam(&points, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_visvalingam_single() {
        let points = vec![Point2::new(1.0, 2.0)];
        let result = visvalingam(&points, 1.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_visvalingam_two_points() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(10.0, 10.0)];
        let result = visvalingam(&points, 1.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_visvalingam_straight_line() {
        // Points on a straight line have zero area triangles
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(4.0, 0.0),
        ];
        let result = visvalingam(&points, 0.1);
        assert_eq!(result.len(), 2); // Only endpoints
    }

    #[test]
    fn test_visvalingam_l_shape() {
        // L-shaped path: corner has significant area
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(5.0, 0.0), // Corner
            Point2::new(5.0, 5.0),
        ];
        // Area of this triangle is 12.5
        let result = visvalingam(&points, 1.0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_visvalingam_removes_small_area() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.1), // Small area triangle
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 5.0), // Large area triangle
            Point2::new(4.0, 0.0),
        ];
        let result = visvalingam(&points, 1.0);

        // Should remove the small deviation but keep the large one
        assert!(result.len() >= 3);
        assert!(result.iter().any(|p| (p.y - 5.0).abs() < 0.01));
    }

    #[test]
    fn test_visvalingam_by_count() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
            Point2::new(5.0, 1.0),
            Point2::new(6.0, 0.0),
        ];

        let result = visvalingam_by_count(&points, 4);
        assert_eq!(result.len(), 4);

        // First and last should be preserved
        assert_relative_eq!(result.first().unwrap().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.last().unwrap().x, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_visvalingam_by_count_minimum() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        // Request fewer than 2 points
        let result = visvalingam_by_count(&points, 1);
        assert_eq!(result.len(), 2); // Minimum is 2
    }

    #[test]
    fn test_visvalingam_indices() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0), // 0
            Point2::new(1.0, 0.0), // 1 - small area, remove
            Point2::new(2.0, 0.0), // 2 - small area, remove
            Point2::new(3.0, 5.0), // 3 - large area, keep
            Point2::new(4.0, 0.0), // 4 - small area, remove
            Point2::new(5.0, 0.0), // 5
        ];

        let indices = visvalingam_indices(&points, 1.0);
        assert!(indices.contains(&0)); // First
        assert!(indices.contains(&3)); // Large deviation
        assert!(indices.contains(&5)); // Last
    }

    #[test]
    fn test_visvalingam_preserves_order() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 3.0),
            Point2::new(4.0, 0.0),
        ];

        let result = visvalingam(&points, 0.1);

        // Check that points are in increasing x order
        for i in 1..result.len() {
            assert!(result[i].x > result[i - 1].x);
        }
    }

    #[test]
    fn test_visvalingam_complex_path() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.1),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.1),
            Point2::new(4.0, 0.0),
            Point2::new(5.0, 3.0),
            Point2::new(6.0, 3.1),
            Point2::new(7.0, 3.0),
            Point2::new(8.0, 0.0),
            Point2::new(9.0, 0.1),
            Point2::new(10.0, 0.0),
        ];

        let result = visvalingam(&points, 0.5);

        assert!(result.len() < points.len());
        assert_relative_eq!(result.first().unwrap().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.last().unwrap().x, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_visvalingam_f32() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let result = visvalingam(&points, 0.1);
        assert_eq!(result.len(), 2);
    }
}
