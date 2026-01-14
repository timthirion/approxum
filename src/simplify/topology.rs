//! Topology-preserving polyline simplification.
//!
//! This module provides simplification algorithms that guarantee the
//! simplified polyline does not self-intersect. This is important for
//! GIS applications where topological correctness must be maintained.

use crate::primitives::{Point2, Segment2};
use crate::tolerance::segments_intersect;
use crate::tolerance::SegmentIntersection;
use num_traits::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Simplifies a polyline while preserving topology (no self-intersections).
///
/// Uses a modified Visvalingam-Whyatt algorithm that skips point removal
/// if it would cause the polyline to self-intersect.
///
/// # Arguments
///
/// * `points` - The input polyline as a slice of points
/// * `min_area` - Minimum effective area threshold
/// * `eps` - Tolerance for intersection tests
///
/// # Returns
///
/// A new vector with the simplified polyline that does not self-intersect.
///
/// # Example
///
/// ```
/// use approxum::{Point2, simplify::topology_preserving};
///
/// let points = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(4.0, 0.0),
///     Point2::new(2.0, 1.0),  // Removing this could cause self-intersection
///     Point2::new(0.0, 0.0),
/// ];
///
/// let simplified = topology_preserving(&points, 0.1, 1e-10);
/// // Result is guaranteed not to self-intersect
/// ```
pub fn topology_preserving<F: Float>(points: &[Point2<F>], min_area: F, eps: F) -> Vec<Point2<F>> {
    let indices = topology_preserving_indices(points, min_area, eps);
    indices.into_iter().map(|i| points[i]).collect()
}

/// Simplifies to a target count while preserving topology.
///
/// Note: May return more points than requested if removing additional
/// points would cause self-intersection.
pub fn topology_preserving_by_count<F: Float>(
    points: &[Point2<F>],
    target_count: usize,
    eps: F,
) -> Vec<Point2<F>> {
    let indices = topology_preserving_indices_by_count(points, target_count, eps);
    indices.into_iter().map(|i| points[i]).collect()
}

/// Returns indices of retained points after topology-preserving simplification.
pub fn topology_preserving_indices<F: Float>(
    points: &[Point2<F>],
    min_area: F,
    eps: F,
) -> Vec<usize> {
    topology_impl(points, Some(min_area), None, eps)
}

/// Returns indices of retained points, simplifying to target count.
pub fn topology_preserving_indices_by_count<F: Float>(
    points: &[Point2<F>],
    target_count: usize,
    eps: F,
) -> Vec<usize> {
    topology_impl(points, None, Some(target_count), eps)
}

/// Entry for the priority queue.
struct AreaEntry<F> {
    index: usize,
    area: F,
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
        // Reverse ordering for min-heap
        other
            .area
            .partial_cmp(&self.area)
            .unwrap_or(Ordering::Equal)
    }
}

/// Computes triangle area.
fn triangle_area<F: Float>(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> F {
    let two = F::one() + F::one();
    ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs() / two
}

/// Checks if removing a point would cause self-intersection.
///
/// When we remove point at index `i`, we create a new segment from
/// prev[i] to next[i]. We need to check if this segment intersects
/// any other segment in the polyline (except adjacent ones).
fn would_self_intersect<F: Float>(
    points: &[Point2<F>],
    remove_idx: usize,
    prev: &[usize],
    next: &[usize],
    active: &[bool],
    eps: F,
) -> bool {
    let p = prev[remove_idx];
    let n = next[remove_idx];

    if p == usize::MAX || n == usize::MAX {
        return false; // Endpoint, can't cause intersection
    }

    let new_segment = Segment2::new(points[p], points[n]);

    // Walk through all active segments and check for intersection
    // A segment goes from idx to next[idx]
    let mut idx = 0;
    // Find first active point
    while idx < points.len() && !active[idx] {
        idx += 1;
    }

    let first_active = idx;
    if idx >= points.len() {
        return false;
    }

    loop {
        let seg_start = idx;
        let seg_end = next[idx];

        if seg_end == usize::MAX {
            break;
        }

        // Skip segments adjacent to the new segment
        // Adjacent segments share an endpoint with the new segment
        let is_adjacent = seg_start == p
            || seg_start == n
            || seg_end == p
            || seg_end == n
            || seg_start == remove_idx
            || seg_end == remove_idx;

        if !is_adjacent {
            let existing = Segment2::new(points[seg_start], points[seg_end]);

            match segments_intersect(new_segment, existing, eps) {
                SegmentIntersection::Point { t1, t2, .. } => {
                    // True intersection (not just touching at endpoints)
                    // Check if intersection is in the interior of both segments
                    if t1 > eps && t1 < F::one() - eps && t2 > eps && t2 < F::one() - eps {
                        return true;
                    }
                }
                SegmentIntersection::Overlapping { .. } => {
                    return true;
                }
                SegmentIntersection::None => {}
            }
        }

        idx = seg_end;
        if idx == first_active || next[idx] == usize::MAX {
            break;
        }
    }

    false
}

/// Core implementation.
fn topology_impl<F: Float>(
    points: &[Point2<F>],
    min_area: Option<F>,
    target_count: Option<usize>,
    eps: F,
) -> Vec<usize> {
    let n = points.len();

    if n < 3 {
        return (0..n).collect();
    }

    // Linked list
    let mut prev: Vec<usize> = (0..n)
        .map(|i| if i == 0 { usize::MAX } else { i - 1 })
        .collect();
    let mut next: Vec<usize> = (0..n)
        .map(|i| if i == n - 1 { usize::MAX } else { i + 1 })
        .collect();

    let mut active = vec![true; n];
    let mut generation = vec![0usize; n];
    let mut areas: Vec<F> = vec![F::infinity(); n];
    let mut heap = BinaryHeap::new();

    // Initialize
    for i in 1..n - 1 {
        let area = triangle_area(points[prev[i]], points[i], points[next[i]]);
        areas[i] = area;
        heap.push(AreaEntry {
            index: i,
            area,
            generation: 0,
        });
    }

    let mut remaining = n;
    let target = target_count.unwrap_or(2).max(2);

    // Track the last area we removed at (for monotonicity)
    #[allow(unused_assignments)]
    let mut last_removed_area = F::zero();

    while let Some(entry) = heap.pop() {
        if !active[entry.index] || generation[entry.index] != entry.generation {
            continue;
        }

        // Check stopping conditions
        if let Some(min) = min_area {
            if entry.area >= min {
                break;
            }
        }
        if remaining <= target {
            break;
        }

        let i = entry.index;

        // TOPOLOGY CHECK: Would removing this point cause self-intersection?
        if would_self_intersect(points, i, &prev, &next, &active, eps) {
            // Can't remove this point, mark it with infinite area so we don't try again
            areas[i] = F::infinity();
            generation[i] += 1;
            // Don't re-add to heap with infinity - it would never be selected anyway
            continue;
        }

        // Safe to remove
        active[i] = false;
        remaining -= 1;
        last_removed_area = entry.area;

        let p = prev[i];
        let nx = next[i];

        // Update linked list
        if p != usize::MAX {
            next[p] = nx;
        }
        if nx != usize::MAX {
            prev[nx] = p;
        }

        // Update neighbor areas
        if p != usize::MAX && prev[p] != usize::MAX {
            let new_area = triangle_area(points[prev[p]], points[p], points[next[p]]);
            areas[p] = new_area.max(last_removed_area);
            generation[p] += 1;
            heap.push(AreaEntry {
                index: p,
                area: areas[p],
                generation: generation[p],
            });
        }

        if nx != usize::MAX && next[nx] != usize::MAX {
            let new_area = triangle_area(points[prev[nx]], points[nx], points[next[nx]]);
            areas[nx] = new_area.max(last_removed_area);
            generation[nx] += 1;
            heap.push(AreaEntry {
                index: nx,
                area: areas[nx],
                generation: generation[nx],
            });
        }
    }

    active
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| if a { Some(i) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to check if a polyline self-intersects.
    fn has_self_intersection<F: Float>(points: &[Point2<F>], eps: F) -> bool {
        let n = points.len();
        if n < 4 {
            return false;
        }

        for i in 0..n - 1 {
            let s1 = Segment2::new(points[i], points[i + 1]);
            // Check against non-adjacent segments
            for j in i + 2..n - 1 {
                // Skip if segments share an endpoint
                if j == i + 1 || (i == 0 && j == n - 2) {
                    continue;
                }
                let s2 = Segment2::new(points[j], points[j + 1]);

                match segments_intersect(s1, s2, eps) {
                    SegmentIntersection::Point { t1, t2, .. } => {
                        if t1 > eps && t1 < F::one() - eps && t2 > eps && t2 < F::one() - eps {
                            return true;
                        }
                    }
                    SegmentIntersection::Overlapping { .. } => return true,
                    SegmentIntersection::None => {}
                }
            }
        }
        false
    }

    #[test]
    fn test_topology_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let result = topology_preserving(&points, 1.0, 1e-10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_topology_simple() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let result = topology_preserving(&points, 0.1, 1e-10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_topology_preserves_no_self_intersection() {
        // A figure-8 like shape that could self-intersect if simplified naively
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 4.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 4.0),
            Point2::new(0.0, 0.0),
        ];

        let result = topology_preserving(&points, 0.01, 1e-10);

        // Result should not self-intersect
        assert!(!has_self_intersection(&result, 1e-10));
    }

    #[test]
    fn test_topology_spiral() {
        // A spiral that could self-intersect
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(1.0, 10.0),
            Point2::new(1.0, 1.0),
            Point2::new(9.0, 1.0),
            Point2::new(9.0, 9.0),
            Point2::new(2.0, 9.0),
        ];

        let result = topology_preserving(&points, 1.0, 1e-10);
        assert!(!has_self_intersection(&result, 1e-10));
    }

    #[test]
    fn test_topology_by_count() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        ];

        let result = topology_preserving_by_count(&points, 3, 1e-10);

        // Should have at most target count (may have more if topology prevents removal)
        assert!(result.len() >= 2);
        assert!(!has_self_intersection(&result, 1e-10));
    }

    #[test]
    fn test_topology_indices() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.1),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.1),
            Point2::new(4.0, 0.0),
        ];

        let indices = topology_preserving_indices(&points, 0.1, 1e-10);

        assert!(indices.contains(&0));
        assert!(indices.contains(&4));
    }

    #[test]
    fn test_topology_prevents_crossing() {
        // Shape where naive simplification would create a crossing
        // Start -> up-right -> down -> up-left -> back
        // If we remove the middle point, the line would cross
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),  // 0
            Point2::new(5.0, 5.0),  // 1
            Point2::new(10.0, 0.0), // 2
            Point2::new(5.0, -1.0), // 3 - small deviation, might be removed
            Point2::new(0.0, 0.0),  // 4 - back to start (closed)
        ];

        // With regular Visvalingam, point 3 might be removed, causing
        // segment 2->4 to cross segment 0->1
        let result = topology_preserving(&points, 5.0, 1e-10);

        // Should not self-intersect
        assert!(!has_self_intersection(&result, 1e-10));
    }

    #[test]
    fn test_topology_hourglass() {
        // Hourglass shape - must preserve the crossing point
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 4.0),
            Point2::new(2.0, 2.0), // Center point
            Point2::new(4.0, 4.0),
            Point2::new(4.0, 0.0),
            Point2::new(2.0, 2.0), // Back to center
            Point2::new(0.0, 0.0),
        ];

        let result = topology_preserving(&points, 0.5, 1e-10);

        // The result should not self-intersect
        // (the original already touches at center, which is allowed)
        assert!(!has_self_intersection(&result, 1e-10));
    }

    #[test]
    fn test_topology_preserves_endpoints() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(5.0, 5.0),
            Point2::new(10.0, 0.0),
        ];

        let result = topology_preserving(&points, 100.0, 1e-10);

        // Even with large tolerance, endpoints should be preserved
        assert!(result.len() >= 2);
        assert_eq!(result.first().unwrap().x, 0.0);
        assert_eq!(result.last().unwrap().x, 10.0);
    }
}
